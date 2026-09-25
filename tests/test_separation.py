"""Window stitching and gain fitting of the separator, using a fake 'oracle' separator."""
import numpy as np

from whospoke.audio import SR
from whospoke.metrics import pit_si_sdr
from whospoke.separation import Separator


class OracleSeparator(Separator):
    """Returns the true sources for each window, in random order, sign and scale — like a real model."""

    def __init__(self, sources: np.ndarray, seed: int = 0):
        self.name, self.window, self.hop = "oracle", int(8 * SR), int(6 * SR)
        self.sources = sources
        self.rng = np.random.default_rng(seed)
        self.cursor = 0

    def _run(self, x):
        a = self._locate(x)
        y = self.sources[:, a: a + len(x)].copy()
        if self.rng.random() < 0.5:
            y = y[::-1]
        return y * self.rng.choice([-1, 1], size=(2, 1)) * self.rng.uniform(0.2, 3.0, size=(2, 1))

    def _locate(self, x):
        mix = self.sources.sum(0)
        for a in range(0, mix.shape[0] - len(x) + 1, int(SR)):
            if np.allclose(mix[a: a + len(x)], x, atol=1e-6):
                return a
        raise AssertionError("window not found")


def test_stitching_recovers_long_sources_despite_random_permutations():
    rng = np.random.default_rng(0)
    t = np.arange(40 * SR) / SR
    s = np.stack([np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(t)), rng.standard_normal(len(t)) * 0.3]).astype(np.float32)
    sep = OracleSeparator(s)
    out = sep.separate(s.sum(0))
    score, _ = pit_si_sdr(list(out), list(s))
    assert score > 30


def test_gain_fit_restores_mixture_consistency():
    rng = np.random.default_rng(1)
    s = rng.standard_normal((2, 3 * SR)).astype(np.float32)
    sep = OracleSeparator(s, seed=3)
    y = sep._separate_window(s.sum(0))
    assert np.allclose(y.sum(0), s.sum(0), atol=1e-2)
