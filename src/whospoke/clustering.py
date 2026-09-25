"""Unsupervised speaker clustering — the two methods named in the proposal.

Input: one embedding ("voice fingerprint") per short window of speech. Output: a speaker label per
window. Neither method is told how many people are in the recording; both estimate it.

* **Spectral clustering** (Ng, Jordan & Weiss 2001; with the pruning + eigengap recipe of
  Park et al. 2019, "Auto-tuning spectral clustering for speaker diarization"):
  build a cosine-similarity graph between windows, keep only each window's strongest links,
  and read the number of speakers from the largest gap in the graph Laplacian's eigenvalues.
* **Gaussian Mixture Model**: reduce embeddings with PCA, fit GMMs with 1..K components and keep
  the one with the lowest Bayesian Information Criterion (BIC) — a fit-vs-complexity trade-off.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


@dataclass
class SpectralClustering:
    """Spectral clustering with row-wise p-pruning and eigengap speaker counting."""

    p_percentile: float = 0.2     # fraction of strongest neighbours kept per row
    max_speakers: int = 8
    min_speakers: int = 1
    single_speaker_sim: float = 0.55   # if nearly all windows are this similar, it is one speaker
    seed: int = 0

    name = "spectral"

    def affinity(self, emb: np.ndarray) -> np.ndarray:
        x = l2norm(emb)
        a = x @ x.T
        n = len(a)
        k = max(2, int(np.ceil(self.p_percentile * n)))
        pruned = np.zeros_like(a)
        idx = np.argsort(-a, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        pruned[rows, idx] = a[rows, idx]
        pruned = np.clip(pruned, 0, None)
        return 0.5 * (pruned + pruned.T)

    def __call__(self, emb: np.ndarray, n_speakers: int | None = None) -> np.ndarray:
        n = len(emb)
        if n <= 2:
            return np.zeros(n, dtype=int)
        if n_speakers is None and self._looks_like_one_speaker(emb):
            return np.zeros(n, dtype=int)
        a = self.affinity(emb)
        np.fill_diagonal(a, 0.0)
        deg = a.sum(1)
        lap = np.diag(deg) - a                        # unnormalised Laplacian (Park et al.)
        vals, vecs = eigh(lap)
        kmax = min(self.max_speakers, n - 1)
        if n_speakers is None:
            gaps = np.diff(vals[: kmax + 1])
            k = int(np.argmax(gaps[self.min_speakers - 1:]) + self.min_speakers)
        else:
            k = n_speakers
        k = max(1, min(k, kmax))
        if k == 1:
            return np.zeros(n, dtype=int)
        spec = vecs[:, :k]
        return KMeans(k, n_init=10, random_state=self.seed).fit_predict(l2norm(spec))

    def _looks_like_one_speaker(self, emb: np.ndarray) -> bool:
        x = l2norm(emb)
        sim = x @ x.T
        return float(np.percentile(sim[np.triu_indices(len(x), 1)], 10)) > self.single_speaker_sim


@dataclass
class GMMClustering:
    """PCA-reduced embeddings → diagonal-covariance GMMs; number of speakers chosen by BIC."""

    pca_dims: int = 8
    max_speakers: int = 8
    covariance_type: str = "diag"
    seed: int = 0

    name = "gmm"

    def __call__(self, emb: np.ndarray, n_speakers: int | None = None) -> np.ndarray:
        n = len(emb)
        if n <= 2:
            return np.zeros(n, dtype=int)
        x = l2norm(emb)
        d = min(self.pca_dims, n - 1, x.shape[1])
        z = PCA(d, random_state=self.seed).fit_transform(x)
        ks = [n_speakers] if n_speakers else range(1, min(self.max_speakers, n // 3) + 1)
        best, best_bic = None, np.inf
        for k in ks:
            gmm = GaussianMixture(k, covariance_type=self.covariance_type, n_init=3,
                                  reg_covar=1e-4, random_state=self.seed).fit(z)
            bic = gmm.bic(z)
            if bic < best_bic:
                best, best_bic = gmm, bic
        return best.predict(z)


def consolidate(emb: np.ndarray, labels: np.ndarray, merge_sim: float = 0.6, min_frac: float = 0.03) -> np.ndarray:
    """Clean up a clustering: absorb tiny clusters, then merge clusters whose voices are near-identical.

    * a cluster holding fewer than ``min_frac`` of the windows is too small to be a real speaker
      in a conversation (usually a noisy or overlapped patch) → each of its windows moves to the
      nearest remaining centroid;
    * two clusters whose mean fingerprints have cosine similarity above ``merge_sim`` are the same
      person split in two → merged (repeatedly, most similar pair first).
    """
    x = l2norm(emb)
    labels = labels.copy()
    counts = np.bincount(labels)
    big = [c for c in range(len(counts)) if counts[c] >= max(1, min_frac * len(labels))]
    if big and len(big) < len(set(labels.tolist())):
        cents = np.stack([l2norm(x[labels == c].mean(0, keepdims=True))[0] for c in big])
        small = ~np.isin(labels, big)
        labels[small] = np.array(big)[np.argmax(x[small] @ cents.T, axis=1)]
    while True:
        ids = sorted(set(labels.tolist()))
        if len(ids) < 2:
            break
        cents = np.stack([l2norm(x[labels == c].mean(0, keepdims=True))[0] for c in ids])
        sim = cents @ cents.T
        np.fill_diagonal(sim, -1)
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        if sim[i, j] < merge_sim:
            break
        labels[labels == ids[j]] = ids[i]
    return relabel_by_first_appearance(labels)


def make(name: str, **kw):
    return {"spectral": SpectralClustering, "gmm": GMMClustering}[name](**kw)


def relabel_by_first_appearance(labels: np.ndarray) -> np.ndarray:
    """Rename cluster ids 0,1,2,... in order of first appearance (so Speaker_A speaks first)."""
    mapping: dict[int, int] = {}
    for lab in labels:
        mapping.setdefault(int(lab), len(mapping))
    return np.array([mapping[int(lab)] for lab in labels], dtype=int)
