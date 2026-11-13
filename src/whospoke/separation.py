"""Stage 1 — blind source separation: split overlapping voices into one track per speaker.

Two pretrained 2-speaker separators are wrapped behind one interface:

* ``convtasnet`` — Conv-TasNet (Luo & Mesgarani, 2019), the model named in the proposal.
  Checkpoint: Asteroid ``JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k`` (trained on noisy 2-speaker mixtures).
* ``sepformer``  — SepFormer (Subakan et al., 2021), a transformer separator, as a stronger
  reference. Checkpoint: SpeechBrain ``speechbrain/sepformer-whamr16k`` (noisy + reverberant mixtures).

Recordings longer than a few seconds are processed in overlapping windows. Each window's two
outputs come back in arbitrary order, so consecutive windows are aligned by correlating their
shared region before crossfading ("stitching").
"""
from __future__ import annotations

import numpy as np
import torch

from .audio import SR

SEPARATORS = {
    "convtasnet": "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k",
    "convtasnet-clean": "JorisCos/ConvTasNet_Libri2Mix_sepclean_16k",
    "sepformer": "speechbrain/sepformer-whamr16k",
}
# Checkpoints made in this project (not downloadable): path relative to the project folder.
# ``convtasnet-hindi`` = ``convtasnet`` fine-tuned on Hindi mixtures by scripts/finetune_convtasnet.py (A3).
LOCAL_SEPARATORS = {
    "convtasnet-hindi": "models/convtasnet_hindi.pth",
}
# Music separators (the voice + mridangam duo, docs/MUSIC_EXTENSION_PLAN.md). Unlike the speech models, their two
# outputs have a fixed meaning: output 0 = melody (voice, drone), output 1 = percussion (mridangam).
MUSIC_SEPARATORS = {
    "hpss": "harmonic-percussive separation (librosa median filtering; no training)",
    "demucs": "htdemucs (Meta, music stems): vocals+other -> output 0, drums+bass -> output 1",
}


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Sum over sources of the cosine similarity between matching rows of ``a`` and ``b``."""
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
    return float(np.sum(num / den))


class Separator:
    """Pretrained 2-output separator. ``separate(wav) -> (2, T)`` for mono 16 kHz input."""

    n_src = 2

    def __init__(self, name: str = "convtasnet", device: str | None = None,
                 window_s: float = 8.0, hop_s: float = 6.0):
        known = {**SEPARATORS, **LOCAL_SEPARATORS, **MUSIC_SEPARATORS}
        if name not in known:
            raise ValueError(f"unknown separator {name!r}; choose from {list(known)}")
        self.name = name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.window, self.hop = int(window_s * SR), int(hop_s * SR)
        if name == "hpss":
            self._run = self._run_hpss
        elif name == "demucs":
            from demucs.pretrained import get_model

            self._model = get_model("htdemucs").to(self.device).eval()
            self._run = self._run_demucs
        elif name.startswith("convtasnet"):
            from pathlib import Path

            from asteroid.models import ConvTasNet

            source = SEPARATORS.get(name)
            if source is None:
                source = Path(__file__).resolve().parents[2] / LOCAL_SEPARATORS[name]
                if not source.exists():
                    raise FileNotFoundError(f"{source} not found: make it with scripts/finetune_convtasnet.py")
                source = str(source)
            self._model = ConvTasNet.from_pretrained(source).to(self.device).eval()
            self._run = self._run_asteroid
        else:
            from pathlib import Path

            from speechbrain.inference.separation import SepformerSeparation

            save = Path(__file__).resolve().parents[2] / "models" / name
            self._model = SepformerSeparation.from_hparams(
                SEPARATORS[name], savedir=str(save), run_opts={"device": str(self.device)})
            self._run = self._run_speechbrain

    @torch.inference_mode()
    def _run_asteroid(self, x: np.ndarray) -> np.ndarray:
        y = self._model(torch.from_numpy(x)[None].to(self.device))[0]
        return y.float().cpu().numpy()

    @staticmethod
    def _run_hpss(x: np.ndarray) -> np.ndarray:
        import librosa

        spec = librosa.stft(x, n_fft=1024, hop_length=256)
        harm, perc = librosa.decompose.hpss(spec)
        return np.stack([librosa.istft(m, hop_length=256, length=len(x)) for m in (harm, perc)])

    @torch.inference_mode()
    def _run_demucs(self, x: np.ndarray) -> np.ndarray:
        import torchaudio.functional as AF
        from demucs.apply import apply_model

        sr = self._model.samplerate                                  # 44.1 kHz stereo model
        wav = AF.resample(torch.from_numpy(x), SR, sr)
        stems = apply_model(self._model, wav.expand(2, -1)[None].to(self.device), device=self.device,
                            progress=False)[0].mean(1)               # (4 stems, T), mono
        idx = {s: i for i, s in enumerate(self._model.sources)}
        out = torch.stack([stems[idx["vocals"]] + stems[idx["other"]], stems[idx["drums"]] + stems[idx["bass"]]])
        return AF.resample(out.cpu(), sr, SR).numpy()

    @torch.inference_mode()
    def _run_speechbrain(self, x: np.ndarray) -> np.ndarray:
        y = self._model.separate_batch(torch.from_numpy(x)[None].to(self.device))[0]
        return y.T.float().cpu().numpy()

    def _separate_window(self, x: np.ndarray) -> np.ndarray:
        """Separate one window; outputs rescaled so they add back up to the input's level."""
        y = self._run(x.astype(np.float32))[:, : len(x)]
        if y.shape[1] < len(x):
            y = np.pad(y, ((0, 0), (0, len(x) - y.shape[1])))
        # Separators trained with a scale-invariant loss return each source with an arbitrary gain
        # (and sign). Fit per-source gains so that g0*y0 + g1*y1 ≈ x ("mixture consistency").
        # A small ridge term keeps a near-silent output from being blown up.
        yy = y.astype(np.float64)
        gram = yy @ yy.T
        gram += np.eye(len(yy)) * 1e-3 * np.trace(gram) / len(yy) + 1e-12
        g = np.linalg.solve(gram, yy @ x.astype(np.float64))
        return (y * g[:, None]).astype(np.float32)

    def separate(self, wav: np.ndarray) -> np.ndarray:
        wav = np.asarray(wav, np.float32)
        n = len(wav)
        if n <= self.window:
            return self._separate_window(wav)
        out = np.zeros((2, n), np.float32)
        weight = np.zeros(n, np.float32)
        ov = self.window - self.hop
        fade_in = np.linspace(0, 1, ov, dtype=np.float32)
        for k, a in enumerate(range(0, n - ov, self.hop)):
            b = min(a + self.window, n)
            y = self._separate_window(wav[a:b])
            w = np.ones(b - a, np.float32)
            if k > 0:
                # [a, a+ov) was already estimated by the previous window: keep the output order
                # that best continues it, then crossfade.
                prev = out[:, a: a + ov] / np.maximum(weight[a: a + ov], 1e-6)
                if _similarity(prev, y[::-1, :ov]) > _similarity(prev, y[:, :ov]):
                    y = y[::-1]
                w[:ov] = fade_in
            if b < n:
                w[-ov:] = np.minimum(w[-ov:], fade_in[::-1])
            out[:, a:b] += y * w
            weight[a:b] += w
        return out / np.maximum(weight, 1e-6)
