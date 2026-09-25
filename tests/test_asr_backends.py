"""Tests for whospoke.asr_backends.

Fast tests exercise input handling without loading any model; ``slow`` tests load
each backend and transcribe real IndicVoices Hindi utterances.
"""
import io
import re
import unicodedata
from pathlib import Path

import numpy as np
import pytest

from whospoke import asr_backends as A

from whospoke.paths import RAW  # noqa: E402

VALID = RAW / "indicvoices/hindi/valid-00000-of-00001.parquet"
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
SR = A.SAMPLE_RATE


def normalize(text: str) -> str:
    text = re.sub(r"<[^>]*>", " ", text)
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------------------------------------------------------- fast
class _Echo(A._ASRBackend):
    """Backend stub that reports chunk durations instead of running a model."""

    name, max_chunk_s = "echo", 30.0

    def __init__(self):
        self.calls = []

    def _transcribe_batch(self, batch):
        self.calls.append([len(w) for w in batch])
        return [f"{len(w) / SR:.2f}" for w in batch]


def test_empty_and_short_inputs_return_empty_without_model_call():
    asr = _Echo()
    out = asr.transcribe([np.zeros(0, np.float32), np.zeros(int(0.05 * SR), np.float32)])
    assert out == ["", ""]
    assert asr.calls == []


def test_output_aligned_with_inputs():
    asr = _Echo()
    wavs = [np.zeros(0), np.random.randn(2 * SR), np.zeros(10), np.random.randn(SR // 2)]
    out = asr.transcribe(wavs)
    assert out == ["", "2.00", "", "0.50"]


@pytest.mark.parametrize("cls", [A.IndicConformerASR, A.IndicWav2VecASR])
def test_real_backends_skip_short_inputs(cls, monkeypatch):
    asr = cls.__new__(cls)  # bypass model loading
    monkeypatch.setattr(asr, "_transcribe_batch", lambda b: pytest.fail("model called"), raising=False)
    assert asr.transcribe([np.zeros(0, np.float32), np.ones(int(0.099 * SR), np.float32)]) == ["", ""]


def test_resampling_and_stereo():
    stereo_8k = np.random.randn(8000, 2).astype(np.float32)  # 1 s, (T, C)
    mono = A._prepare(stereo_8k, 8000)
    assert mono.ndim == 1 and mono.dtype == np.float32 and len(mono) == SR
    assert _Echo().transcribe([np.random.randn(44100)], sr=44100) == ["1.00"]


def test_long_input_is_chunked_at_quiet_points():
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(65 * SR).astype(np.float32)
    wav[27 * SR:int(27.3 * SR)] = 0.0  # a pause inside the search window of the first cut
    pieces = A._split_long(wav, max_s=30.0)
    assert np.array_equal(np.concatenate(pieces), wav)
    assert all(len(p) <= 30 * SR for p in pieces)
    assert 27 * SR <= len(pieces[0]) <= 27.3 * SR
    asr = _Echo()
    assert len(asr.transcribe([wav])[0].split()) == len(pieces)


# ----------------------------------------------------------------------------- slow
@pytest.fixture(scope="module")
def utterances():
    pd = pytest.importorskip("pandas")
    sf = pytest.importorskip("soundfile")
    if not VALID.exists():
        pytest.skip(f"missing {VALID}")
    df = pd.read_parquet(VALID)
    df = df[(df.duration >= 2) & (df.duration <= 10) & (df.text.map(normalize) != "")]
    df = df.sample(3, random_state=0)
    wavs = [sf.read(io.BytesIO(a["bytes"]), dtype="float32")[0] for a in df.audio_filepath]
    return wavs, list(df.text)


@pytest.fixture(scope="module", params=["indicconformer-ctc", "indicconformer-rnnt", "indicwav2vec"])
def backend(request):
    if request.param == "indicwav2vec":
        asr = A.IndicWav2VecASR()
    else:
        asr = A.IndicConformerASR(decoding=request.param.split("-")[1])
    yield asr
    asr.close()


@pytest.mark.slow
def test_transcribes_real_hindi(backend, utterances):
    jiwer = pytest.importorskip("jiwer")
    wavs, refs = utterances
    hyps = backend.transcribe(wavs)
    assert len(hyps) == len(wavs)
    for h in hyps:
        assert h and DEVANAGARI.search(h), f"{backend.name}: non-Devanagari output {h!r}"
    cer = jiwer.cer([normalize(r) for r in refs], [normalize(h) for h in hyps])
    assert cer < 0.6, f"{backend.name}: CER {cer:.3f}"


@pytest.mark.slow
def test_mixed_batch_and_long_input(backend, utterances):
    wavs, _ = utterances
    long = np.concatenate(wavs * 10)[: 60 * SR]  # ~60 s, exercises chunking
    out = backend.transcribe([np.zeros(0, np.float32), wavs[0][: int(0.05 * SR)], long])
    assert out[0] == "" and out[1] == ""
    assert len(out[2].split()) > 20
