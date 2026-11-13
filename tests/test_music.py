"""Music extension: activity, timbre fingerprints, tonic matching and the separators' fixed output meaning."""
import numpy as np

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


def test_tonic_matching():
    assert pitch_class(138.59) == 1 and nearest_stroke_tonic(138.59) == "C#"   # C#3
    assert nearest_stroke_tonic(149.4) == "D"
    assert nearest_stroke_tonic(210.07) == "B"           # G#: B (3 semitones) is nearer than E (4)
