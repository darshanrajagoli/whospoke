import numpy as np
import pytest

from whospoke.audio import SR
from whospoke.corpora import Utterance
from whospoke.synth import SynthConfig, overlap_stats, read_rttm, simulate, trim_silence


def tone(seconds: float, freq: float, silence: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    x = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    pad = np.zeros(int(silence * SR), np.float32)
    return np.concatenate([pad, x, pad])


def speakers(n: int = 2, k: int = 6):
    return {f"S{i + 1}": [Utterance(f"spk{i}", f"u{i}_{j}", tone(1.0 + 0.3 * j, 200 + 150 * i, 0.3)) for j in range(k)]
            for i in range(n)}


@pytest.mark.parametrize("n,ovl", [(2, 0.0), (2, 0.8), (3, 0.5)])
def test_mixture_equals_sum_of_sources_and_noise(n, ovl):
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(200 * SR).astype(np.float32) * 0.01
    conv = simulate("t", speakers(n), SynthConfig(n_speakers=n, target_s=15, overlap_prob=ovl, snr_db=10, seed=1), noise)
    recon = sum(conv.sources.values()) + conv.noise
    assert np.allclose(conv.mixture, recon, atol=1e-5)
    assert np.max(np.abs(conv.mixture)) <= 0.9 + 1e-6


def test_no_overlap_config_produces_no_overlap_and_never_three_speakers():
    conv = simulate("t", speakers(2), SynthConfig(target_s=30, overlap_prob=0.0, snr_db=None, seed=2))
    assert conv.overlap_ratio() == 0
    conv3 = simulate("t", speakers(3), SynthConfig(n_speakers=3, target_s=30, overlap_prob=1.0, snr_db=None, seed=3))
    assert overlap_stats(conv3.segments, conv3.duration)["max_concurrent"] <= 2
    assert conv3.overlap_ratio() > 0


def test_segments_match_audio_and_keep_utterance_order():
    conv = simulate("t", speakers(2), SynthConfig(target_s=100, overlap_prob=0.5, snr_db=None, seed=4))
    for spk in conv.speakers:
        segs = [s for s in conv.segments if s.speaker == spk]
        assert [s.text for s in segs] == sorted([s.text for s in segs], key=lambda t: int(t.split("_")[1]))
        for s in segs:   # every labelled segment actually contains that speaker's audio
            chunk = conv.sources[spk][int(s.start * SR): int(s.end * SR)]
            assert np.sqrt(np.mean(chunk ** 2)) > 1e-3


def test_trim_silence_removes_padding():
    x = tone(1.0, 300, silence=0.5)
    y = trim_silence(x)
    assert abs(len(y) / SR - 1.1) < 0.05


def test_save_roundtrip(tmp_path):
    conv = simulate("conv1", speakers(2), SynthConfig(target_s=10, overlap_prob=0.3, snr_db=None, seed=5))
    out = conv.save(tmp_path)
    segs = read_rttm(out / "reference.rttm")
    assert len(segs) == len(conv.segments)
    assert (out / "mixture.wav").exists() and (out / "sources" / "S1.wav").exists()
