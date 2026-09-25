import numpy as np
import pytest

from whospoke import clustering as clu


def blobs(k: int, n: int = 40, dim: int = 64, spread: float = 0.15, seed: int = 0):
    """``k`` well-separated speakers, ``n`` windows each, on the unit sphere."""
    rng = np.random.default_rng(seed)
    centres = clu.l2norm(rng.standard_normal((k, dim)))
    x = np.concatenate([c + spread * rng.standard_normal((n, dim)) / np.sqrt(dim) * 4 for c in centres])
    y = np.repeat(np.arange(k), n)
    return x.astype(np.float32), y


def purity(pred, truth):
    total = 0
    for c in set(pred):
        total += np.bincount(truth[pred == c]).max()
    return total / len(truth)


@pytest.mark.parametrize("method", ["spectral", "gmm"])
@pytest.mark.parametrize("k", [2, 3, 4])
def test_estimates_speaker_count_and_groups_correctly(method, k):
    x, y = blobs(k, seed=k)
    pred = clu.make(method)(x)
    assert len(set(pred)) == k
    assert purity(pred, y) > 0.95


@pytest.mark.parametrize("method", ["spectral", "gmm"])
def test_single_speaker(method):
    x, _ = blobs(1)
    assert len(set(clu.make(method)(x))) == 1


def test_oracle_speaker_count_is_respected():
    x, _ = blobs(3)
    assert len(set(clu.SpectralClustering()(x, n_speakers=2))) == 2


def test_consolidate_merges_duplicate_and_absorbs_tiny_clusters():
    x, y = blobs(2, n=50)
    split = y.copy()
    split[:25] = 2                     # speaker 0 wrongly split in two halves
    split[-1] = 3                      # one stray window as its own "speaker"
    out = clu.consolidate(x, split, merge_sim=0.6, min_frac=0.03)
    assert len(set(out)) == 2
    assert purity(out, y) == 1.0


def test_relabel_by_first_appearance():
    assert clu.relabel_by_first_appearance(np.array([2, 2, 0, 1, 0])).tolist() == [0, 0, 1, 2, 1]
