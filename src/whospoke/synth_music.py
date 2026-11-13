"""Music simulator: a Carnatic voice + mridangam duo with exact ground truth (the music extension of ``synth.py``).

The speech set rebuilds conversations from single-speaker recordings. This module does the same for music:

* **Voice (S1):** solo Carnatic singing with its drone, from the *Carnatic Varnam* dataset (28 varnams, 6 singers,
  all in adi tala, audio CC BY 3.0 via Freesound). Phrases are whole tala cycles cut at the dataset's annotated
  cycle starts, so every phrase comes with its real beat times (8 beats per cycle).
* **Mridangam (S2):** single strokes from the *Mridangam Stroke* dataset (7,162 strokes, 10 stroke types, 6 tonics,
  CC BY-NC 4.0), placed on that beat grid: resonant strokes (tham, thom, dheem) on the strong beats of adi tala,
  lighter strokes on the subdivisions, 2 or 4 strokes per beat, occasional rests and fills. The drum is tuned to the
  tonic closest to the singer's.
* **Overlap level** = how often the mridangam accompanies a sung phrase; between phrases it plays short interludes,
  so there is drum-only, voice-only and both-at-once time, as in a concert.

Because we place every sound ourselves, the clean tracks and "who plays when" are known exactly, so separation
(SI-SDR) and who-plays-when (DER) can be scored with the speech set's code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .audio import SR, active_rms, db_to_gain, load, scale_to_snr
from .paths import RAW
from .synth import Conversation, Segment, SynthConfig, speech_activity

VARNAM = RAW / "carnatic_varnam" / "carnatic_varnam_1.1"
STROKES = RAW / "mridangam_stroke" / "mridangam_stroke_1.0"
STROKE_TONICS = {"B": 11, "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4}     # pitch class of each drum's tonic
# Singers are split, never shared: dev singers' recordings are only in dev, the others only in test.
DEV_SINGERS = ("vignesh", "dharini")
STRONG = {1, 5, 7}             # adi tala = laghu (beats 1-4) + two drutams (5-6, 7-8): their first beats are stressed


def pitch_class(hz: float) -> int:
    return int(round(12 * np.log2(hz / 440.0) + 9)) % 12


def nearest_stroke_tonic(hz: float) -> str:
    pc = pitch_class(hz)
    return min(STROKE_TONICS, key=lambda t: min((pc - STROKE_TONICS[t]) % 12, (STROKE_TONICS[t] - pc) % 12))


# ------------------------------------------------------------------ data access
@dataclass(frozen=True)
class Recording:
    path: Path
    singer: str
    raga: str
    tonic_hz: float
    beats: tuple[tuple[int, int, float], ...]     # (cycle, beat 1-8, time in s)

    @property
    def id(self) -> str:
        return f"{self.raga}-{self.singer}"


def recordings() -> list[Recording]:
    """Every annotated recording (27 of the 28: one has no tala annotation)."""
    import yaml

    tonics = yaml.safe_load((VARNAM / "Notations_Annotations" / "annotations" / "tonics.yaml").read_text())
    out = []
    for mp3 in sorted((VARNAM / "Audio").glob("*.mp3")):
        m = re.search(r"varnam-by-([a-z]+)-in-([a-z]+)-raaga", mp3.name)
        singer, raga = m.group(1), m.group(2)
        svl = VARNAM / "Notations_Annotations" / "annotations" / "taalas" / raga / f"{singer}.svl"
        if svl.exists():
            out.append(Recording(mp3, singer, raga, float(tonics[singer]), read_svl(svl)))
    return out


def read_svl(path: Path) -> tuple[tuple[int, int, float], ...]:
    text = path.read_text(encoding="utf-8")
    sr = float(re.search(r'sampleRate="(\d+)"', text).group(1))
    pts = re.findall(r'<point frame="(\d+)" label="(\d+)\.(\d+)"', text)
    return tuple(sorted((int(c), int(b), int(f) / sr) for f, c, b in pts))


def cycles(rec: Recording) -> list[np.ndarray]:
    """Complete annotated cycles: the 8 beat times plus the next cycle's start (9 times each)."""
    by_cycle: dict[int, dict[int, float]] = {}
    for c, b, t in rec.beats:
        by_cycle.setdefault(c, {})[b] = t
    out = []
    for c in sorted(by_cycle):
        cur, nxt = by_cycle[c], by_cycle.get(c + 1, {})
        if all(b in cur for b in range(1, 9)) and 1 in nxt and 3.0 < nxt[1] - cur[1] < 12.0:
            out.append(np.array([cur[b] for b in range(1, 9)] + [nxt[1]]))
    return out


@lru_cache(maxsize=None)
def voice_audio(path: Path) -> np.ndarray:
    return load(path)


@lru_cache(maxsize=None)
def stroke_bank(tonic: str, split: str) -> dict[str, list[np.ndarray]]:
    """All strokes of one drum (tonic), by stroke type. Dev uses odd-numbered takes, test even-numbered ones."""
    bank: dict[str, list[np.ndarray]] = {}
    for f in sorted((STROKES / tonic).glob("*.wav")):
        m = re.search(r"__([a-z]+)-[a-z#]+-(\d+)\.wav$", f.name.lower())
        if not m or (int(m.group(2)) % 2 == 1) != (split == "dev"):
            continue
        w = load(f)
        bank.setdefault(m.group(1), []).append(w / (float(np.max(np.abs(w))) or 1.0))
    return bank


# ------------------------------------------------------------------ the duo
@dataclass
class MusicConfig:
    accompany_prob: float = 0.3      # chance a sung phrase is accompanied (the overlap level)
    partial_prob: float = 0.5        # an accompanied phrase is accompanied only for its last 2-4 beats (low overlap)
    interlude_prob: float = 0.8      # chance of a drum-only interlude between phrases
    target_s: float = 45.0
    level_db: tuple[float, float] = (-4.0, 4.0)   # drum level relative to the voice
    snr_db: float | None = None
    seed: int = 0


class MusicPiece(Conversation):
    """A ``Conversation`` whose reference activity bridges the gaps between drum strokes."""

    def activity(self) -> list[Segment]:
        act = speech_activity(self.sources["S1"], self.segments, "S1")
        act += speech_activity(self.sources["S2"], self.segments, "S2", fill_gap_s=0.6)
        return sorted(act, key=lambda s: s.start)


def _stroke_times(beats: np.ndarray, rng: np.random.Generator, fill: bool) -> list[tuple[float, str, float]]:
    """(time, stroke type, gain dB) for the beats ``beats[i] -> beats[i+1]``; ``beats`` carries beat numbers 1-8."""
    out = []
    per_beat = 4 if (fill or rng.random() < 0.35) else 2
    for i in range(len(beats) - 1):
        num, t0, t1 = int(beats[i][0]), beats[i][1], beats[i + 1][1]
        for k in range(per_beat):
            t = t0 + (t1 - t0) * k / per_beat
            if k == 0:
                kind = str(rng.choice(["tham", "thom", "dheem"])) if num in STRONG else str(rng.choice(["dhin", "num", "tha"]))
                gain = 3.0 if num in STRONG else 0.0
            else:
                if rng.random() < (0.05 if fill else 0.18):
                    continue                                       # rests
                kind = str(rng.choice(["ta", "thi", "tha", "cha", "num", "dhin"], p=[.25, .3, .2, .1, .08, .07]))
                gain = -3.0
            out.append((t, kind, gain + rng.uniform(-3, 3)))
    return out


def simulate_duo(piece_id: str, rec: Recording, cfg: MusicConfig, split: str, noise: np.ndarray | None = None):
    """Lay sung cycles and mridangam patterns on one timeline. Returns a ``MusicPiece`` (S1 voice, S2 mridangam)."""
    rng = np.random.default_rng(cfg.seed)
    voice = voice_audio(rec.path)
    cyc = cycles(rec)
    tonic = nearest_stroke_tonic(rec.tonic_hz)
    bank = stroke_bank(tonic, split)
    start_cycle = int(rng.integers(0, max(1, len(cyc) - 12)))

    placed_voice: list[tuple[float, np.ndarray, int]] = []      # (start s, audio, cycle index)
    patterns: list[tuple[float, float, list]] = []              # (start s, end s, strokes)
    t, ci = float(rng.uniform(0.3, 0.8)), start_cycle
    while t < cfg.target_s and ci < len(cyc):
        # a sung phrase: 1-2 consecutive annotated cycles, cut at the annotated cycle starts
        n_cyc = int(rng.integers(1, 3))
        group = cyc[ci: ci + n_cyc]
        ci += n_cyc + int(rng.integers(0, 3))
        if not group:
            break
        a, b = group[0][0], group[-1][-1]
        clip = voice[int(a * SR): int(b * SR)]
        offset = t - a                                               # maps recording time -> piece time
        placed_voice.append((t, clip, ci))
        grid = [(beat + 1, tt + offset) for g in group for beat, tt in enumerate(g[:8])] + [(1, b + offset)]
        if rng.random() < cfg.accompany_prob:                        # accompaniment
            if rng.random() < cfg.partial_prob:
                grid = grid[-int(rng.integers(3, 6)):]               # only the last 2-4 beats
            patterns.append((grid[0][1], grid[-1][1], _stroke_times(np.array(grid, dtype=object), rng, False)))
        t += len(clip) / SR
        if rng.random() < cfg.interlude_prob and t < cfg.target_s:   # drum-only interlude, same tempo
            beat = float(np.mean(np.diff(group[-1])))
            n = int(rng.integers(2, 5))
            grid = [((k % 8) + 1, t + 0.05 + k * beat) for k in range(n + 1)]
            patterns.append((grid[0][1], grid[-1][1], _stroke_times(np.array(grid, dtype=object), rng, True)))
            t = grid[-1][1]
        t += float(rng.uniform(0.2, 0.8))

    total = int(np.ceil((t + 1.5) * SR))
    s1, s2 = np.zeros(total, np.float32), np.zeros(total, np.float32)
    segments = []
    for start, clip, idx in placed_voice:
        k = int(round(start * SR))
        s1[k: k + len(clip)] += clip[: total - k]
        segments.append(Segment("S1", round(start, 3), round(start + len(clip) / SR, 3), f"{rec.raga} cycle {idx}"))
    for p0, p1, strokes in patterns:
        last = p0
        for when, kind, gain_db in strokes:
            choices = bank.get(kind) or bank["tha"]
            w = choices[int(rng.integers(len(choices)))] * db_to_gain(gain_db)
            k = int(round(when * SR))
            n = min(len(w), total - k)
            if n > 0:
                s2[k: k + n] += w[:n]
                last = max(last, when + n / SR)
        segments.append(Segment("S2", round(p0, 3), round(last, 3), "mridangam"))

    s1 *= 0.05 / max(active_rms(s1), 1e-4)
    s2 *= 0.05 / max(active_rms(s2), 1e-4) * db_to_gain(rng.uniform(*cfg.level_db))
    music = s1 + s2
    noise_track = None
    if noise is not None and cfg.snr_db is not None:
        noise_track = scale_to_snr(music, noise[:total].astype(np.float32), cfg.snr_db)
    mixture = music + (noise_track if noise_track is not None else 0.0)
    g = min(1.0, 0.9 / (float(np.max(np.abs(mixture))) or 1.0))
    sources = {"S1": (s1 * g).astype(np.float32), "S2": (s2 * g).astype(np.float32)}
    noise_track = None if noise_track is None else (noise_track * g).astype(np.float32)
    segments.sort(key=lambda s: s.start)
    syn = SynthConfig(n_speakers=2, target_s=cfg.target_s, overlap_prob=cfg.accompany_prob, snr_db=cfg.snr_db,
                      seed=cfg.seed)
    piece = MusicPiece(piece_id, (mixture * g).astype(np.float32), sources, noise_track, segments, syn)
    piece.extra = {"recording": rec.id, "singer": rec.singer, "raga": rec.raga, "singer_tonic_hz": rec.tonic_hz,
                   "mridangam_tonic": tonic, "sources": {"S1": "voice (with drone)", "S2": "mridangam"},
                   "music_config": {k: v for k, v in vars(cfg).items()}}
    return piece
