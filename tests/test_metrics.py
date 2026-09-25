import numpy as np
import pytest

from whospoke import metrics


def test_si_sdr_is_scale_invariant_and_high_for_identical():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(16000)
    assert metrics.si_sdr(ref, ref) > 60
    est = ref + 0.3 * rng.standard_normal(16000)
    assert metrics.si_sdr(3.7 * est, ref) == pytest.approx(metrics.si_sdr(est, ref), abs=1e-6)
    assert metrics.si_sdr(-est, ref) == pytest.approx(metrics.si_sdr(est, ref), abs=1e-6)


def test_si_sdr_known_value():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(100_000)
    noise = rng.standard_normal(100_000)
    noise -= ref * np.dot(noise, ref) / np.dot(ref, ref)       # orthogonal noise
    noise *= np.linalg.norm(ref) / np.linalg.norm(noise) / 10  # 20 dB
    assert metrics.si_sdr(ref + noise, ref) == pytest.approx(20.0, abs=0.05)


def test_pit_finds_the_swap():
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal((2, 8000))
    score, perm = metrics.pit_si_sdr([b, a], [a, b])
    assert perm == (1, 0) and score > 60


def test_der_perfect_and_confused():
    ref = [(0, 5, "A"), (5, 10, "B")]
    assert metrics.der(ref, [(0, 5, "x"), (5, 10, "y")], collar=0)["der"] == pytest.approx(0)
    swapped_half = metrics.der(ref, [(0, 10, "x")], collar=0)
    assert swapped_half["confusion"] == pytest.approx(0.5)
    missed = metrics.der(ref, [(0, 5, "x")], collar=0)
    assert missed["missed"] == pytest.approx(0.5)


def test_normalise_strips_danda_and_punctuation():
    assert metrics.normalise("नमस्ते, आप कैसे हैं।  ठीक?") == "नमस्ते आप कैसे हैं ठीक"


def test_wer_cer_basic():
    assert metrics.wer("एक दो तीन", "एक दो तीन") == 0
    assert metrics.wer("एक दो तीन", "एक तीन") == pytest.approx(1 / 3)
    assert metrics.cer("abc", "abd") == pytest.approx(1 / 3)
    assert metrics.wer("", "") == 0


def test_corpus_rate_pools_tokens():
    pairs = [("a b", "a b"), ("c d e f", "c x e f")]
    assert metrics.corpus_rate(pairs) == pytest.approx(1 / 6)


def test_cpwer_speaker_permutation_and_missing_speaker():
    ref = {"S1": "एक दो तीन", "S2": "चार पांच"}
    hyp = {"Speaker_A": "चार पांच", "Speaker_B": "एक दो तीन"}
    r = metrics.cp_error(ref, hyp)
    assert r["rate"] == 0 and r["mapping"] == {"S1": "Speaker_B", "S2": "Speaker_A"}
    # everything attributed to one speaker: S2's words count as deletions (+ insertions into S1)
    merged = metrics.cp_error(ref, {"Speaker_A": "एक दो तीन चार पांच"})
    assert merged["rate"] == pytest.approx(4 / 5)
