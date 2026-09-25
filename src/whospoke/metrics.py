"""Evaluation metrics for every stage.

Stage 1 — separation : SI-SDR / SI-SDR improvement (dB, higher is better)
Stage 2 — diarization: DER and its parts (missed speech, false alarm, speaker confusion), JER
Stage 3 — transcripts: WER / CER, and cpWER — the "who said what" score: each hypothesised
          speaker's words are matched to the best reference speaker before counting errors,
          so a transcript is only right if the words *and* the speaker attribution are right.
"""
from __future__ import annotations

import itertools
import re
import unicodedata
from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

# ------------------------------------------------------------------ separation


def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant signal-to-distortion ratio in dB (Le Roux et al., 2019)."""
    est = np.asarray(est, np.float64) - np.mean(est)
    ref = np.asarray(ref, np.float64) - np.mean(ref)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)))


def pit_si_sdr(ests: Sequence[np.ndarray], refs: Sequence[np.ndarray]) -> tuple[float, tuple[int, ...]]:
    """Mean SI-SDR under the best assignment of estimates to references (permutation-invariant)."""
    best, best_perm = -np.inf, ()
    for perm in itertools.permutations(range(len(ests)), len(refs)):
        score = float(np.mean([si_sdr(ests[p], r) for p, r in zip(perm, refs)]))
        if score > best:
            best, best_perm = score, perm
    return best, best_perm


# ------------------------------------------------------------------ diarization

Turn = tuple[float, float, str]  # (start, end, speaker)


def _annotation(turns: Sequence[Turn], uri: str = "x"):
    from pyannote.core import Annotation, Segment

    ann = Annotation(uri=uri)
    for i, (s, e, spk) in enumerate(turns):
        if e > s:
            ann[Segment(s, e), i] = spk
    return ann


def der(reference: Sequence[Turn], hypothesis: Sequence[Turn], collar: float = 0.25) -> dict:
    """Diarization error rate with components. Overlapped speech is scored (not skipped)."""
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    d = metric(_annotation(reference), _annotation(hypothesis), detailed=True)
    total = d["total"] or 1.0
    return {
        "der": d["diarization error rate"],
        "missed": d["missed detection"] / total,
        "false_alarm": d["false alarm"] / total,
        "confusion": d["confusion"] / total,
        "total_s": d["total"],
    }


def jer(reference: Sequence[Turn], hypothesis: Sequence[Turn], collar: float = 0.25) -> float:
    from pyannote.metrics.diarization import JaccardErrorRate

    return float(JaccardErrorRate(collar=collar, skip_overlap=False)(_annotation(reference), _annotation(hypothesis)))


# ------------------------------------------------------------------ transcripts

_PUNCT = re.compile(r"[।॥.,!?;:\"'()\[\]{}<>\-–—…/\\|*#@%&+=~`^_]")


def normalise(text: str) -> str:
    """Canonical form for scoring: Unicode NFC, no punctuation or danda, single spaces, lower-case Latin."""
    text = unicodedata.normalize("NFC", text)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.lower().split())


def _edits(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance between token lists."""
    if not ref:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def error_counts(ref: str, hyp: str, unit: str = "word") -> tuple[int, int]:
    """(edit distance, reference length) at word or character level, after normalisation."""
    ref, hyp = normalise(ref), normalise(hyp)
    if unit == "word":
        r, h = ref.split(), hyp.split()
    else:
        r, h = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    return _edits(r, h), len(r)


def wer(ref: str, hyp: str) -> float:
    e, n = error_counts(ref, hyp, "word")
    return e / max(n, 1)


def cer(ref: str, hyp: str) -> float:
    e, n = error_counts(ref, hyp, "char")
    return e / max(n, 1)


def corpus_rate(pairs: Sequence[tuple[str, str]], unit: str = "word") -> float:
    """Pooled error rate over many (ref, hyp) pairs — total edits / total reference tokens."""
    e = n = 0
    for r, h in pairs:
        de, dn = error_counts(r, h, unit)
        e, n = e + de, n + dn
    return e / max(n, 1)


def cp_error(ref_by_spk: dict[str, str], hyp_by_spk: dict[str, str], unit: str = "word") -> dict:
    """Concatenated minimum-permutation error rate (cpWER / cpCER, CHiME-6 style).

    Each speaker's text is concatenated in time order; hypothesis speakers are assigned to reference
    speakers (Hungarian algorithm on edit counts); unmatched speakers are compared with empty text.
    """
    refs, hyps = list(ref_by_spk), list(hyp_by_spk)
    n = max(len(refs), len(hyps))
    cost = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            r = ref_by_spk[refs[i]] if i < len(refs) else ""
            h = hyp_by_spk[hyps[j]] if j < len(hyps) else ""
            cost[i, j] = error_counts(r, h, unit)[0]
    rows, cols = linear_sum_assignment(cost)
    errors = int(cost[rows, cols].sum())
    n_ref = sum(error_counts(t, "", unit)[1] for t in ref_by_spk.values())
    mapping = {refs[i]: hyps[j] for i, j in zip(rows, cols) if i < len(refs) and j < len(hyps)}
    return {"rate": errors / max(n_ref, 1), "errors": errors, "n_ref": n_ref, "mapping": mapping}


def matched_tokens(ref: list[str], hyp: list[str]) -> list[bool]:
    """For each reference token, was it recognised exactly (aligned to an identical hypothesis token)?"""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ok = [False] * n
    i, j = n, m
    while i > 0 and j > 0:
        if d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ok[i - 1] = ref[i - 1] == hyp[j - 1]
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return ok
