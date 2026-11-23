"""Music extension: activity, timbre fingerprints, tonic matching and the separators' fixed output meaning."""
import numpy as np
import pytest

from whospoke.audio import SR
from whospoke.diarization import energy_regions, timbre_embed
from whospoke.synth_music import nearest_stroke_tonic, pitch_class

RNG = np.random.default_rng(0)
T = np.arange(6 * SR) / SR
TONE = (0.3 * np.sin(2 * np.pi * 220 * T) * (1 + 0.1 * np.sin(2 * np.pi * 5 * T))).astype(np.float32)


def strokes(n: int, every_s: float) -> np.ndarray:
    x = np.zeros(n, np.float32)
    decay = np.exp(-np.arange(1600) / 200) * RNG.standard_normal(1600)
    for k in range(0, n - 1600, int(every_s * SR)):
        x[k: k + 1600] += decay.astype(np.float32) * 0.5
    return x


def test_energy_regions_bridge_gaps_between_strokes_but_not_long_silences():
    x = np.zeros(10 * SR, np.float32)
    x[: 4 * SR] = strokes(4 * SR, 0.35)                 # strokes every 0.35 s: one region
    x[7 * SR:] = strokes(3 * SR, 0.35)                  # 3 s of silence, then another region
    regions = energy_regions(x)
    assert len(regions) == 2
    assert regions[0][0] < 0.1 and 3.5 < regions[0][1] < 4.1 and 6.9 < regions[1][0] < 7.1


def test_timbre_fingerprint_tells_a_sustained_tone_from_a_drum():
    drum = strokes(6 * SR, 0.3)
    wins = [(0.0, 1.5), (2.0, 3.5), (4.0, 5.5)]
    a, b = timbre_embed(TONE, wins), timbre_embed(drum, wins)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    assert (a @ a.T).min() > (a @ b.T).max() + 0.2      # tone windows agree with each other far more than with drum
    assert (b @ b.T).min() > (a @ b.T).max() + 0.2


def test_hpss_outputs_have_a_fixed_meaning():
    """Output 0 is the melody and output 1 the percussion, which the "direct" who-plays-when relies on (D41)."""
    from whospoke.metrics import si_sdr
    from whospoke.separation import Separator

    drum = strokes(6 * SR, 0.3)
    melody, perc = Separator("hpss", device="cpu").separate(TONE + drum)
    assert si_sdr(melody, TONE) > si_sdr(melody, drum) + 10
    assert si_sdr(perc, drum) > si_sdr(perc, TONE) + 10


def test_long_audio_never_swaps_music_outputs(monkeypatch):
    """Window stitching may reorder a speech separator's outputs, but never a music separator's (output 0 stays melody).

    The keep-or-swap check is forced to prefer "swap" in every window: a speech separator then swaps, while HPSS and
    Demucs must keep their fixed order (0 = melody, 1 = percussion) for the whole recording.
    """
    import whospoke.separation as sep_mod
    from whospoke.metrics import si_sdr

    n = 30 * SR
    t = np.arange(n) / SR
    melody = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    drum = strokes(n, 0.3)
    calls = {"k": 0}

    def prefer_swap(a, b):                          # called (swapped, kept) per window: make "swapped" win
        calls["k"] += 1
        return 1.0 if calls["k"] % 2 else 0.0

    monkeypatch.setattr(sep_mod, "_similarity", prefer_swap)

    class FixedOrder(sep_mod.Separator):             # returns the true sources of each window, in order
        def __init__(self, name):
            self.name, self.window, self.hop = name, 8 * SR, 6 * SR

        def _run(self, x):
            a = next(k for k in range(0, n - len(x) + 1, SR) if np.allclose((melody + drum)[k:k + len(x)], x))
            return np.stack([melody[a:a + len(x)], drum[a:a + len(x)]])

    for name in ("hpss", "demucs"):
        out = FixedOrder(name).separate(melody + drum)
        assert si_sdr(out[0], melody) > 30 and si_sdr(out[1], drum) > 30, f"{name}: outputs were swapped"
    out = FixedOrder("convtasnet").separate(melody + drum)   # speech separator: the forced swap does happen
    assert si_sdr(out[0], melody) < 0


@pytest.mark.slow
def test_demucs_is_repeatable():
    """Demucs's random "shift trick" is seeded, so the same input always gives the same separation, and the global
    generator is left as it was."""
    import random

    pytest.importorskip("demucs")                     # optional install (requirements.txt), music extension only
    from whospoke.separation import Separator

    x = (RNG.standard_normal(5 * SR) * 0.05).astype(np.float32)
    sep = Separator("demucs", device="cpu")
    random.seed(123)
    first = sep.separate(x)
    after = random.random()
    random.seed(123)
    assert random.random() == after                   # separating did not touch the global generator
    assert np.array_equal(first, sep.separate(x))


def test_tonic_matching():
    assert pitch_class(138.59) == 1 and nearest_stroke_tonic(138.59) == "C#"   # C#3
    assert nearest_stroke_tonic(149.4) == "D"
    assert nearest_stroke_tonic(210.07) == "B"           # G#: B (3 semitones) is nearer than E (4)
