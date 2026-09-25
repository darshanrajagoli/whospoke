"""Hindi ASR backends with a common ``transcribe(wavs, sr) -> list[str]`` interface.

Two pretrained AI4Bharat models are wrapped:

* :class:`IndicConformerASR` -- ``ai4bharat/indic-conformer-600m-multilingual``
  (hybrid CTC/RNNT Conformer, exported as TorchScript preprocessor + ONNX graphs).
* :class:`IndicWav2VecASR` -- ``ai4bharat/indicwav2vec-hindi`` (wav2vec2 CTC,
  greedy decoding, no LM).

Both take mono float32 audio (resampled to 16 kHz if needed), skip clips shorter
than 0.1 s, split clips longer than ``max_chunk_s`` at low-energy points, and
return Devanagari text. Heavy dependencies (torch, onnxruntime, transformers)
are imported lazily inside the constructors.
"""
from __future__ import annotations

import contextlib
import gc
import json
import os
import re

import numpy as np

SAMPLE_RATE = 16000
MIN_DURATION_S = 0.1

__all__ = ["IndicConformerASR", "IndicWav2VecASR", "SAMPLE_RATE", "MIN_DURATION_S"]


# --------------------------------------------------------------------------- utils
@contextlib.contextmanager
def _no_jit_fusion(torch):
    """Switch off TorchScript's CPU/GPU fusers for the block (see IndicConformerASR._encode).

    Same effect as ``torch.jit.fuser("none")``, without its per-call deprecation warnings about nvfuser.
    """
    old = (torch._C._jit_can_fuse_on_cpu(), torch._C._jit_can_fuse_on_gpu(), torch._C._jit_texpr_fuser_enabled())
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    try:
        yield
    finally:
        torch._C._jit_override_can_fuse_on_cpu(old[0])
        torch._C._jit_override_can_fuse_on_gpu(old[1])
        torch._C._jit_set_texpr_fuser_enabled(old[2])


def _resolve_device(device: str | None) -> str:
    import torch

    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def _prepare(wav: np.ndarray, sr: int) -> np.ndarray:
    """Return a contiguous mono float32 array at 16 kHz."""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:  # (T, C) or (C, T): average the channel axis
        wav = wav.mean(axis=0 if wav.shape[0] < wav.shape[1] else 1)
    wav = wav.reshape(-1)
    if sr != SAMPLE_RATE and wav.size:
        import torch
        import torchaudio.functional as AF

        wav = AF.resample(torch.from_numpy(wav), sr, SAMPLE_RATE).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)


def _split_long(wav: np.ndarray, max_s: float, search_s: float = 5.0) -> list[np.ndarray]:
    """Split ``wav`` into pieces of at most ``max_s`` seconds.

    Each cut is placed at the quietest 25 ms frame within the last ``search_s``
    seconds of the allowed window, so words are rarely cut in half.
    """
    max_n = int(max_s * SAMPLE_RATE)
    if len(wav) <= max_n:
        return [wav]
    hop = int(0.025 * SAMPLE_RATE)
    pieces, start = [], 0
    while len(wav) - start > max_n:
        lo = start + max_n - int(search_s * SAMPLE_RATE)
        region = wav[lo:start + max_n]
        n_frames = len(region) // hop
        energy = (region[: n_frames * hop].reshape(n_frames, hop) ** 2).mean(axis=1)
        cut = lo + int(np.argmin(energy)) * hop + hop // 2
        pieces.append(wav[start:cut])
        start = cut
    pieces.append(wav[start:])
    return pieces


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class _ASRBackend:
    """Shared preprocessing, chunking and length-sorted batching."""

    name = "base"
    max_chunk_s = 30.0
    batch_size = 4

    def transcribe(self, wavs: list[np.ndarray], sr: int = SAMPLE_RATE) -> list[str]:
        """Transcribe a list of mono waveforms; returns one string per input."""
        chunks: list[tuple[int, np.ndarray]] = []
        for i, w in enumerate(wavs):
            w = _prepare(w, sr)
            if len(w) < MIN_DURATION_S * SAMPLE_RATE:
                continue
            chunks += [(i, c) for c in _split_long(w, self.max_chunk_s)
                       if len(c) >= MIN_DURATION_S * SAMPLE_RATE]

        # Longest first: batches have similar lengths (little padding), and the first,
        # largest allocation is reused by later ones instead of the caches growing.
        order = sorted(range(len(chunks)), key=lambda k: -len(chunks[k][1]))
        texts: list[str] = [""] * len(chunks)
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            for k, t in zip(idx, self._transcribe_batch([chunks[k][1] for k in idx])):
                texts[k] = t

        out: list[list[str]] = [[] for _ in wavs]
        for (i, _), t in zip(chunks, texts):  # chunks are in original temporal order
            if t:
                out[i].append(t)
        return [_clean(" ".join(p)) for p in out]

    def _transcribe_batch(self, batch: list[np.ndarray]) -> list[str]:
        raise NotImplementedError

    def __call__(self, wavs: list[np.ndarray], sr: int = SAMPLE_RATE) -> list[str]:
        return self.transcribe(wavs, sr)


# ------------------------------------------------------------------ IndicConformer
class IndicConformerASR(_ASRBackend):
    """AI4Bharat IndicConformer-600M (multilingual), used for Hindi.

    Loads the repo's exported components directly (TorchScript mel preprocessor,
    ONNX encoder / CTC head / RNNT prediction + joint networks) instead of the
    ``trust_remote_code`` wrapper, which has no device control, no batching and
    uses default CUDA EP options that are very slow on this graph (see notes).

    Args:
        device: ``"cuda"``, ``"cpu"`` or ``None`` (auto).
        decoding: ``"ctc"`` (batched greedy CTC) or ``"rnnt"`` (greedy RNNT,
            one utterance at a time; slightly more accurate, slower).
        language: language code of the multilingual model (default ``"hi"``).
        batch_size: utterances per encoder call.
        max_chunk_s: longer inputs are split at pauses into pieces of this size.
    """

    name = "indicconformer"
    repo_id = "ai4bharat/indic-conformer-600m-multilingual"
    BLANK_ID = 256  # blank in the per-language (257-way) output space
    SOS_ID = 5632   # global blank, used as start token of the prediction network
    PRED_RNN_LAYERS, PRED_RNN_HIDDEN = 2, 640
    RNNT_MAX_SYMBOLS = 10

    def __init__(self, device: str | None = None, decoding: str = "ctc", language: str = "hi",
                 batch_size: int = 4, max_chunk_s: float = 30.0):
        if decoding not in ("ctc", "rnnt"):
            raise ValueError(f"decoding must be 'ctc' or 'rnnt', got {decoding!r}")
        import torch  # must precede onnxruntime so ORT reuses torch's CUDA/cuDNN DLLs
        import onnxruntime as ort
        from huggingface_hub import snapshot_download

        self.device = _resolve_device(device)
        self.decoding, self.language = decoding, language
        self.batch_size, self.max_chunk_s = batch_size, max_chunk_s
        self._ort = ort
        self._assets = os.path.join(snapshot_download(self.repo_id), "assets")

        with open(os.path.join(self._assets, "vocab.json"), encoding="utf-8") as f:
            self.vocab: list[str] = json.load(f)[language]
        with open(os.path.join(self._assets, "language_masks.json")) as f:
            self._lang_idx = np.flatnonzero(np.asarray(json.load(f)[language], dtype=bool))

        self._torch = torch
        self.preprocessor = torch.jit.load(os.path.join(self._assets, "preprocessor.ts"),
                                           map_location=self.device).eval()
        self.encoder = self._session("encoder")
        self._ctc = self._rnnt = None
        if decoding == "ctc":
            self._ctc = self._session("ctc_decoder")
        else:
            self._load_rnnt()

    # -- loading ---------------------------------------------------------------
    def _session(self, name: str, cpu: bool = False):
        ort = self._ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if self.device.startswith("cuda") and not cpu:
            opts = {
                "device_id": int(self.device.split(":")[1]) if ":" in self.device else 0,
                # Default (EXHAUSTIVE) search re-benchmarks every new input length (~50 s each);
                # without NC1D padding cuDNN picks a very slow depthwise-conv kernel.
                "cudnn_conv_algo_search": "DEFAULT",
                "cudnn_conv1d_pad_to_nc1d": "1",
                "arena_extend_strategy": "kSameAsRequested",
            }
            providers = [("CUDAExecutionProvider", opts), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(os.path.join(self._assets, f"{name}.onnx"), so, providers=providers)

    def _load_rnnt(self) -> None:
        # The joint encoder projection runs once per utterance; the per-step networks are
        # tiny, so they run on CPU to avoid a host<->device copy on every decoding step.
        self._rnnt = {
            "joint_enc": self._session("joint_enc"),
            "decoder": self._session("rnnt_decoder", cpu=True),
            "joint_pred": self._session("joint_pred", cpu=True),
            "joint_pre": self._session("joint_pre_net", cpu=True),
            "joint_post": self._session(f"joint_post_net_{self.language}", cpu=True),
        }

    # -- inference -------------------------------------------------------------
    def _encode(self, batch: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # Features are computed per utterance: the exported preprocessor's normalisation
        # statistics include zero padding, whereas the encoder masks padded frames correctly.
        torch = self._torch
        feats = []
        # The preprocessor is TorchScript. `import speechbrain` (pulled in by pyannote / SepFormer) globally turns off
        # TorchScript's profiling executor; the legacy executor then fuses the complex-valued STFT ops into a
        # runtime-compiled CUDA kernel that NVRTC fails to compile ("c10::complex" not found) and the ASR crashes.
        # Disabling the fuser here only affects speed; the features are identical.
        with torch.inference_mode(), _no_jit_fusion(torch):
            for w in batch:
                f, _ = self.preprocessor(input_signal=torch.from_numpy(w)[None].to(self.device),
                                         length=torch.tensor([len(w)], device=self.device))
                feats.append(f[0].float().cpu().numpy())
        n_frames = np.array([f.shape[1] for f in feats], dtype=np.int64)
        x = np.zeros((len(feats), feats[0].shape[0], int(n_frames.max())), dtype=np.float32)
        for i, f in enumerate(feats):
            x[i, :, : f.shape[1]] = f
        enc, enc_lens = self.encoder.run(["outputs", "encoded_lengths"], {"audio_signal": x, "length": n_frames})
        return enc, enc_lens  # (B, 1024, T'), (B,)

    def _detok(self, ids) -> str:
        return "".join(self.vocab[int(i)] for i in ids).replace("▁", " ")

    def _transcribe_batch(self, batch: list[np.ndarray]) -> list[str]:
        if self.decoding == "rnnt":  # greedy RNNT is sequential; encode one at a time
            return [self._rnnt_decode(*self._encode([w])) for w in batch]
        enc, enc_lens = self._encode(batch)
        if self._ctc is None:
            self._ctc = self._session("ctc_decoder")
        logits = self._ctc.run(["logprobs"], {"encoder_output": enc})[0]  # (B, T', 5633)
        best = logits[:, :, self._lang_idx].argmax(-1)
        out = []
        for b, n in enumerate(enc_lens):
            path = best[b, : int(n)]
            keep = np.ones_like(path, dtype=bool)
            keep[1:] = path[1:] != path[:-1]
            out.append(self._detok(t for t in path[keep] if t != self.BLANK_ID))
        return out

    def _rnnt_decode(self, enc: np.ndarray, enc_lens: np.ndarray) -> str:
        """Greedy RNNT decoding, following the reference ``model_onnx.py``."""
        if self._rnnt is None:
            self._load_rnnt()
        s = self._rnnt
        f_all = s["joint_enc"].run(["output"], {"input": enc.transpose(0, 2, 1)})[0]  # (1, T', H)
        hyp = [self.SOS_ID]
        state = (np.zeros((self.PRED_RNN_LAYERS, 1, self.PRED_RNN_HIDDEN), np.float32),
                 np.zeros((self.PRED_RNN_LAYERS, 1, self.PRED_RNN_HIDDEN), np.float32))
        g_cache = None  # prediction-network output only changes when a token is emitted
        for t in range(int(enc_lens[0])):
            f = f_all[:, t:t + 1, :]
            for _ in range(self.RNNT_MAX_SYMBOLS):
                if g_cache is None:
                    g, _, h, c = s["decoder"].run(
                        ["outputs", "prednet_lengths", "states", "162"],
                        {"targets": np.array([[hyp[-1]]], np.int32), "target_length": np.array([1], np.int32),
                         "states.1": state[0], "onnx::Slice_3": state[1]})
                    g_cache = (s["joint_pred"].run(["output"], {"input": g.transpose(0, 2, 1)})[0], (h, c))
                z = s["joint_pre"].run(["output"], {"input": f + g_cache[0]})[0]
                tok = int(s["joint_post"].run(["output"], {"input": z})[0].reshape(-1).argmax())
                if tok == self.BLANK_ID:
                    break
                hyp.append(tok)
                state, g_cache = g_cache[1], None
        return self._detok(hyp[1:])

    def close(self) -> None:
        """Release model sessions and cached GPU memory."""
        self.encoder = self._ctc = self._rnnt = self.preprocessor = None
        gc.collect()
        if self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()


# ------------------------------------------------------------------ IndicWav2Vec
class IndicWav2VecASR(_ASRBackend):
    """AI4Bharat IndicWav2Vec-Hindi (wav2vec2-large CTC) with greedy decoding.

    Args:
        device: ``"cuda"``, ``"cpu"`` or ``None`` (auto).
        fp16: run in half precision on GPU (default True; ~2x faster, same output
            in practice). Ignored on CPU.
        batch_size: utterances per forward pass (padded, with attention mask).
        max_chunk_s: longer inputs are split at pauses into pieces of this size.
    """

    name = "indicwav2vec"
    repo_id = "ai4bharat/indicwav2vec-hindi"

    def __init__(self, device: str | None = None, fp16: bool = True, batch_size: int = 4,
                 max_chunk_s: float = 20.0):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor

        self._torch = torch
        self.device = _resolve_device(device)
        self.batch_size, self.max_chunk_s = batch_size, max_chunk_s
        self.dtype = torch.float16 if (fp16 and self.device.startswith("cuda")) else torch.float32
        # preprocessor_config.json names Wav2Vec2ProcessorWithLM, but the repo ships no LM,
        # so the plain processor is loaded explicitly.
        self.processor = Wav2Vec2Processor.from_pretrained(self.repo_id)
        # The repo only has pytorch_model.bin, which transformers>=4.50 refuses to torch.load
        # on torch<2.6 (CVE-2025-32434). Load it ourselves with weights_only=True (safe) and
        # let from_pretrained handle key remapping (weight-norm g/v -> parametrizations).
        state = torch.load(hf_hub_download(self.repo_id, "pytorch_model.bin"),
                           map_location="cpu", weights_only=True)
        config = Wav2Vec2Config.from_pretrained(self.repo_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(None, config=config, state_dict=state)
        self.model = self.model.to(self.device, self.dtype).eval()
        del state

    def _transcribe_batch(self, batch: list[np.ndarray]) -> list[str]:
        torch = self._torch
        inputs = self.processor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                                padding=True, return_attention_mask=True)
        with torch.inference_mode():
            logits = self.model(inputs.input_values.to(self.device, self.dtype),
                                attention_mask=inputs.attention_mask.to(self.device)).logits
        ids = logits.argmax(-1).cpu()
        return self.processor.batch_decode(ids, skip_special_tokens=True)

    def close(self) -> None:
        """Release the model and cached GPU memory."""
        self.model = None
        gc.collect()
        if self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()
