"""Order-B splice with the second separated voice, using a fake separator and fake voice fingerprints.

Speaker A is a 200 Hz tone and speaker B a 500 Hz tone. The fake separator splits a mixture at 350 Hz and returns the
two outputs in reversed order, as a real model may; the fake fingerprint is the energy share of each tone.
"""
import numpy as np
import pytest

import whospoke.pipeline as pl
from whospoke.audio import SR
from whospoke.diarization import Diarization, Turn


def tone(f: float, start: float, end: float, n: int) -> np.ndarray:
    x = np.zeros(n, np.float32)
    a, b = int(start * SR), int(end * SR)
    x[a:b] = np.sin(2 * np.pi * f * np.arange(b - a) / SR)
    return x


class FakeSeparator:
    name = "fake"

    def separate(self, x: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), 1 / SR)
        low = np.fft.irfft(np.where(freqs < 350, spec, 0), len(x))
        high = np.fft.irfft(np.where(freqs >= 350, spec, 0), len(x))
        return np.stack([high, low]).astype(np.float32)      # B first, A second


def fake_embed(wav, windows, device=None):
    out = []
    for s, e in windows:
        seg = wav[int(s * SR): int(e * SR)]
        spec = np.abs(np.fft.rfft(seg)) ** 2
        freqs = np.fft.rfftfreq(len(seg), 1 / SR)
        out.append([spec[freqs < 350].sum(), spec[freqs >= 350].sum()])
    return np.asarray(out, np.float32) + 1e-6


@pytest.fixture
def pipe(monkeypatch):
    monkeypatch.setattr(pl, "embed", fake_embed)
    p = pl.Pipeline("B", separator=None, diarizer=object(), asr_model=object())
    p.separator = FakeSeparator()
    return p


N = 6 * SR
MIX = tone(200, 0, 4, N) + tone(500, 3, 6, N)                   # A 0-4 s, B 3-6 s: overlap 3-4 s
CENTROIDS = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0])}


def diar(turns):
    return Diarization(turns, dict(CENTROIDS), {"overlap": [(3.0, 4.0)]})


def test_interrupter_gets_the_second_output(pipe):
    # Like the tuned Order B (overlap_aware=False): the region belongs to A only, B's turn starts after it.
    d = diar([Turn(0.0, 4.0, "A"), Turn(4.0, 6.0, "B")])
    pipe.second_voice = 0.5
    units = pipe._targeted_separation(MIX, d)
    extra = [(t, a) for t, a, src in units if src == "second"]
    assert len(extra) == 1
    t, audio = extra[0]
    assert (t.speaker, t.start, t.end) == ("B", 3.0, 4.0)
    # the extra turn's audio is B's voice inside the region and silence outside it
    inside = slice(int(3.2 * SR), int(3.8 * SR))
    assert np.allclose(audio[inside], MIX[inside] - tone(200, 0, 4, N)[inside], atol=1e-3)
    assert not audio[: int(3 * SR)].any() and not audio[int(4 * SR):].any()
    # A's own turn still gets A's voice in the region (the first output is unchanged by the new step)
    a_audio = next(a for t, a, src in units if t.speaker == "A")
    assert np.allclose(a_audio[inside], tone(200, 0, 4, N)[inside], atol=1e-3)


def test_no_extra_turn_when_both_speakers_already_have_the_region(pipe):
    pipe.second_voice = -2.0                                  # accept anything: the rule itself must say no
    units = pipe._targeted_separation(MIX, diar([Turn(0.0, 4.0, "A"), Turn(3.0, 6.0, "B")]))
    assert not any(src == "second" for _, _, src in units)


def test_threshold_and_default_off(pipe):
    d = diar([Turn(0.0, 4.0, "A"), Turn(4.0, 6.0, "B")])
    _, candidates = pipe._separate_units(MIX, d)
    assert len(candidates) == 1 and candidates[0][0] > 0.9    # like B (≈1) minus like A (≈0)
    assert pipe.second_voice is None                          # off unless asked for
    assert not any(src == "second" for _, _, src in pipe._targeted_separation(MIX, d))
    pipe.second_voice = candidates[0][0] + 0.01
    assert not any(src == "second" for _, _, src in pipe._targeted_separation(MIX, d))
