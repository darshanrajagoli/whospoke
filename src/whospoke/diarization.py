"""Stage 2 — speaker diarization: *who* spoke *when*, without knowing the speakers in advance.

Our diarizer follows the proposal step by step:

1. **Voice activity detection** (pyannote ``segmentation-3.0``): where is anyone speaking?
2. **Speaker embeddings** (pyannote's WeSpeaker ResNet-34): a 256-number "voice fingerprint" for
   every 1.5 s window of speech (windows every 0.75 s).
3. **Unsupervised clustering** — Spectral Clustering or a GMM (see ``clustering.py``) groups
   fingerprints that sound like the same person.
4. **Timeline**: each 50 ms frame of speech takes the label of the windows covering it; very short
   flickers are smoothed away; consecutive frames with one label become a turn.
5. **Overlap handling** (optional): where the overlap detector says two people talk at once, the
   second speaker is the nearest *other* speaker in time (Bullock et al., 2020).

``PyannoteDiarizer`` wraps the complete off-the-shelf pyannote 3.1 pipeline as a reference point.
"""
from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from . import clustering as clu
from . import vad
from .audio import SR, fmt_ts

FRAME = 0.05  # seconds per timeline frame


@dataclass
class Turn:
    start: float
    end: float
    speaker: str
    stream: int | None = None   # which separated track the turn was found in (Order A)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_tuple(self) -> tuple[float, float, str]:
        return (self.start, self.end, self.speaker)


@dataclass
class Diarization:
    turns: list[Turn]
    centroids: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    @property
    def speakers(self) -> list[str]:
        return sorted({t.speaker for t in self.turns})

    def tuples(self) -> list[tuple[float, float, str]]:
        return [t.as_tuple() for t in self.turns]

    def timeline_json(self) -> list[dict]:
        """The proposal's deliverable format: ``[00:12 - 00:45]: Speaker_A``."""
        return [{"start": fmt_ts(t.start), "end": fmt_ts(t.end), "start_s": round(t.start, 2),
                 "end_s": round(t.end, 2), "speaker": t.speaker} for t in sorted(self.turns, key=lambda t: t.start)]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.timeline_json(), indent=1), encoding="utf-8")


def speaker_name(i: int) -> str:
    return f"Speaker_{string.ascii_uppercase[i]}" if i < 26 else f"Speaker_{i}"


# ------------------------------------------------------------------ embeddings

@lru_cache(maxsize=2)
def _embedding_model(device: str):
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

    return PretrainedSpeakerEmbedding("pyannote/wespeaker-voxceleb-resnet34-LM", device=torch.device(device))


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def embed(wav: np.ndarray, windows: list[tuple[float, float]], device: str | None = None,
          batch: int = 32) -> np.ndarray:
    """One embedding per ``(start, end)`` window. Windows are zero-padded to a common length."""
    model = _embedding_model(device or default_device())
    if not windows:
        return np.zeros((0, model.dimension), np.float32)
    segs = [wav[int(s * SR): int(e * SR)] for s, e in windows]
    longest = max(max(len(g) for g in segs), model.min_num_samples)
    out = []
    for i in range(0, len(segs), batch):
        chunk = segs[i: i + batch]
        x = np.zeros((len(chunk), 1, longest), np.float32)
        masks = np.zeros((len(chunk), longest), np.float32)
        for j, seg in enumerate(chunk):
            x[j, 0, : len(seg)] = seg
            masks[j, : len(seg)] = 1.0
        out.append(model(torch.from_numpy(x), masks=torch.from_numpy(masks)))
    emb = np.concatenate(out).astype(np.float32)
    return np.nan_to_num(emb)


def sliding_windows(regions: list[tuple[float, float]], win: float = 1.5, hop: float = 0.75,
                    min_len: float = 0.4) -> list[tuple[float, float]]:
    """Cover each speech region with windows; regions shorter than ``win`` get one window."""
    out = []
    for s, e in regions:
        if e - s < min_len:
            continue
        if e - s <= win:
            out.append((s, e))
            continue
        t = s
        while t + win < e:
            out.append((round(t, 3), round(t + win, 3)))
            t += hop
        out.append((round(max(s, e - win), 3), round(e, 3)))
    return out


# ------------------------------------------------------------------ timeline construction

def _frame_labels(windows, labels, regions, duration, k):
    """Per-frame speaker label (-1 = silence) from window labels, weighted towards window centres."""
    n = int(np.ceil(duration / FRAME)) + 1
    votes = np.zeros((n, max(k, 1)), np.float32)
    for (s, e), lab in zip(windows, labels):
        a, b = int(s / FRAME), max(int(s / FRAME) + 1, int(e / FRAME))
        w = 1.0 - np.abs(np.linspace(-1, 1, b - a)) * 0.9     # triangular weight
        votes[a:b, lab] += w
    speech = np.zeros(n, bool)
    for s, e in regions:
        speech[int(s / FRAME): int(np.ceil(e / FRAME))] = True
    lab = np.where(speech & (votes.sum(1) > 0), votes.argmax(1), -1)
    # speech frames not covered by any window (very short regions): nearest labelled frame
    idx = np.flatnonzero(speech & (lab < 0))
    have = np.flatnonzero(lab >= 0)
    if len(idx) and len(have):
        nearest = have[np.clip(np.searchsorted(have, idx), 0, len(have) - 1)]
        lab[idx] = lab[nearest]
    return lab


def _smooth(lab: np.ndarray, min_frames: int) -> np.ndarray:
    """Absorb speaker runs shorter than ``min_frames`` into the longer neighbouring run."""
    lab = lab.copy()
    for _ in range(3):
        runs = _runs(lab)
        changed = False
        for i, (a, b, v) in enumerate(runs):
            if v < 0 or b - a >= min_frames:
                continue
            left = runs[i - 1] if i > 0 and runs[i - 1][1] == a and runs[i - 1][2] >= 0 else None
            right = runs[i + 1] if i + 1 < len(runs) and runs[i + 1][0] == b and runs[i + 1][2] >= 0 else None
            cand = [r for r in (left, right) if r is not None]
            if cand:
                lab[a:b] = max(cand, key=lambda r: r[1] - r[0])[2]
                changed = True
        if not changed:
            break
    return lab


def _runs(lab: np.ndarray) -> list[tuple[int, int, int]]:
    if len(lab) == 0:
        return []
    change = np.flatnonzero(np.diff(lab)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(lab)]
    return [(int(a), int(b), int(lab[a])) for a, b in zip(starts, ends)]


def _turns_from_frames(lab: np.ndarray, names: dict[int, str], stream: int | None = None) -> list[Turn]:
    return [Turn(round(a * FRAME, 3), round(b * FRAME, 3), names[v], stream) for a, b, v in _runs(lab) if v >= 0]


def add_overlap_speakers(turns: list[Turn], overlaps: list[tuple[float, float]]) -> list[Turn]:
    """For each overlap region, add the nearest-in-time *other* speaker as a second speaker."""
    extra = []
    for s, e in overlaps:
        mid = 0.5 * (s + e)
        active = {t.speaker for t in turns if t.start < e and t.end > s}
        others = [t for t in turns if t.speaker not in active] if active else []
        if not others or not active:
            continue
        near = min(others, key=lambda t: 0.0 if t.start <= mid <= t.end else min(abs(t.start - mid), abs(t.end - mid)))
        extra.append(Turn(s, e, near.speaker))
    return merge_turns(turns + extra)


def merge_turns(turns: list[Turn], gap: float = 0.0) -> list[Turn]:
    """Merge same-speaker turns that touch/overlap (or are within ``gap`` seconds)."""
    out: list[Turn] = []
    for t in sorted(turns, key=lambda t: (t.speaker, t.start)):
        if out and out[-1].speaker == t.speaker and t.start <= out[-1].end + gap:
            out[-1].end = max(out[-1].end, t.end)
        else:
            out.append(Turn(t.start, t.end, t.speaker, t.stream))
    return sorted(out, key=lambda t: t.start)


# ------------------------------------------------------------------ diarizers

@dataclass
class Features:
    """Everything the neural models compute for one track (cacheable; clustering is cheap afterwards)."""

    duration: float
    speech: list[tuple[float, float]]
    windows: list[tuple[float, float]]
    embeddings: np.ndarray
    overlaps: list[tuple[float, float]]
    energy: np.ndarray            # per-frame level in dB (used for leakage checks between tracks)


@dataclass
class Diarizer:
    """VAD → embeddings → clustering → timeline. ``clustering`` is ``"spectral"`` or ``"gmm"``."""

    clustering: str = "spectral"
    window: float = 1.5
    hop: float = 0.75
    min_turn: float = 0.3
    overlap_aware: bool = True
    merge_sim: float | None = 0.6      # clean-up after clustering (None = off); tuned on the dev set
    min_cluster_frac: float = 0.03
    leak_db: float = 6.0               # Order A: a track this much quieter than another is leakage
    cluster_kwargs: dict = field(default_factory=dict)
    device: str | None = None

    @property
    def name(self) -> str:
        return f"ours-{self.clustering}"

    # --- step 1: neural analysis (slow part) -------------------------------------------------
    def analyse(self, wav: np.ndarray, overlaps: bool = True) -> Features:
        dur = len(wav) / SR
        speech = vad.speech_regions(wav, self.device)
        wins = sliding_windows(speech, self.window, self.hop)
        emb = embed(wav, wins, self.device)
        ovl = vad.overlap_regions(wav, self.device) if overlaps else []
        return Features(dur, speech, wins, emb, ovl, _frame_energy(wav, int(np.ceil(dur / FRAME)) + 1))

    # --- step 2: clustering + timeline (fast part) -------------------------------------------
    def _cluster(self, emb: np.ndarray, n_speakers: int | None) -> np.ndarray:
        labels = clu.make(self.clustering, **self.cluster_kwargs)(emb, n_speakers)
        if n_speakers is None and self.merge_sim is not None:
            labels = clu.consolidate(emb, labels, self.merge_sim, self.min_cluster_frac)
        return clu.relabel_by_first_appearance(labels)

    def assign(self, f: Features, n_speakers: int | None = None) -> Diarization:
        if not f.windows:
            return Diarization([], info={"speech": f.speech})
        labels = self._cluster(f.embeddings, n_speakers)
        k = int(labels.max()) + 1
        names = {i: speaker_name(i) for i in range(k)}
        lab = _smooth(_frame_labels(f.windows, labels, f.speech, f.duration, k), int(round(self.min_turn / FRAME)))
        turns = _turns_from_frames(lab, names)
        if self.overlap_aware and k > 1:
            turns = add_overlap_speakers(turns, f.overlaps)
        centroids = {names[i]: clu.l2norm(f.embeddings[labels == i]).mean(0) for i in range(k)}
        return Diarization(turns, centroids, {"speech": f.speech, "overlap": f.overlaps, "n_windows": len(f.windows)})

    def __call__(self, wav: np.ndarray, n_speakers: int | None = None) -> Diarization:
        # Overlap detection always runs: even when it is not used to add second speakers to the timeline,
        # Order B needs the detected overlap regions to know where to separate (info["overlap"]).
        return self.assign(self.analyse(wav, overlaps=True), n_speakers)

    # --- Order A: separated tracks -----------------------------------------------------------
    def analyse_streams(self, streams: np.ndarray) -> list[Features]:
        return [self.analyse(s, overlaps=False) for s in streams]

    def assign_streams(self, feats: list[Features], n_speakers: int | None = None) -> Diarization:
        """Diarize already-separated tracks jointly.

        Windows from every track are clustered together, so a person keeps one label even when
        the separator moves them from one track to the other. When the *same* speaker is found in
        two tracks at the same time, or a track is much quieter than another active one, that is
        separator leakage and the weaker copy is dropped.
        """
        if not any(f.windows for f in feats):
            return Diarization([], info={"speech": [f.speech for f in feats]})
        emb = np.concatenate([f.embeddings for f in feats])
        owner = np.concatenate([np.full(len(f.windows), si) for si, f in enumerate(feats)]).astype(int)
        labels = self._cluster(emb, n_speakers)
        k = int(labels.max()) + 1
        names = {i: speaker_name(i) for i in range(k)}
        dur = feats[0].duration
        n = len(feats[0].energy)
        min_frames = int(round(self.min_turn / FRAME))
        frame_lab = np.stack([_smooth(_frame_labels(f.windows, labels[owner == si], f.speech, dur, k)[:n], min_frames)
                              for si, f in enumerate(feats)])
        energy = np.stack([f.energy for f in feats])
        for a in range(len(feats)):
            for b in range(a + 1, len(feats)):
                clash = (frame_lab[a] >= 0) & (frame_lab[a] == frame_lab[b])
                a_quieter = energy[a] < energy[b]
                frame_lab[a][clash & a_quieter] = -1
                frame_lab[b][clash & ~a_quieter] = -1
        for si in range(len(feats)):
            other = np.max(np.delete(energy, si, axis=0), axis=0)
            weak = (energy[si] < other - self.leak_db) & (frame_lab[si] >= 0) & _other_active(frame_lab, si)
            frame_lab[si][weak] = -1
        turns = []
        for si in range(len(feats)):
            turns += _turns_from_frames(_smooth(frame_lab[si], min_frames), names, si)
        turns = [t for t in turns if t.duration >= self.min_turn]
        centroids = {names[i]: clu.l2norm(emb[labels == i]).mean(0) for i in range(k)}
        return Diarization(sorted(turns, key=lambda t: t.start), centroids,
                           {"speech": [f.speech for f in feats], "n_windows": int(len(emb))})

    def diarize_streams(self, streams: np.ndarray, n_speakers: int | None = None) -> Diarization:
        return self.assign_streams(self.analyse_streams(streams), n_speakers)


def _frame_energy(wav: np.ndarray, n: int) -> np.ndarray:
    hop = int(FRAME * SR)
    e = np.full(n, -120.0)
    m = min(n, len(wav) // hop)
    fr = wav[: m * hop].reshape(m, hop)
    e[:m] = 10 * np.log10(np.mean(fr.astype(np.float64) ** 2, axis=1) + 1e-12)
    return e


def _other_active(frame_lab: np.ndarray, si: int) -> np.ndarray:
    return np.any(np.delete(frame_lab, si, axis=0) >= 0, axis=0)


class PyannoteDiarizer:
    """Off-the-shelf pyannote/speaker-diarization-3.1 (reference system, not our method)."""

    name = "pyannote-3.1"

    def __init__(self, device: str | None = None):
        from pyannote.audio import Pipeline

        self.pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        self.pipe.to(torch.device(device or default_device()))

    def __call__(self, wav: np.ndarray, n_speakers: int | None = None) -> Diarization:
        kw = {"num_speakers": n_speakers} if n_speakers else {}
        ann = self.pipe({"waveform": torch.from_numpy(np.asarray(wav, np.float32))[None], "sample_rate": SR}, **kw)
        order = {}
        for seg, _, spk in ann.itertracks(yield_label=True):
            order.setdefault(spk, len(order))
        turns = [Turn(round(seg.start, 3), round(seg.end, 3), speaker_name(order[spk]))
                 for seg, _, spk in ann.itertracks(yield_label=True)]
        return Diarization(merge_turns(turns), {}, {})
