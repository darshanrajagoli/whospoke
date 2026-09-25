"""Background noise for the simulated conversations: "village" and "market" soundscapes.

The proposal describes recordings made "surrounded by ambient village or market noise". We build
exactly those two soundscapes from real recordings:

* a continuous **bed** of real ambience from DEMAND (CC-BY-4.0; 16 kHz field recordings), and
* sparse **events** on top from ESC-50 (cows, roosters, dogs, crows, horns, engines, bells, ...).

Why not Vaani? We tried harvesting speech-free audio from Project Vaani (see DECISIONS.md, D11):
Vaani clips are trimmed so tightly around the speech that almost nothing speech-free is left
(~1 s per 500 s of audio), and noise inside speech cannot be separated from the voice.

Dev/test split: the first half of every DEMAND recording and ESC-50 folds 4–5 are reserved for
the development set (used for tuning); the second half and folds 1–3 are used for the test set.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import SR, db_to_gain, load, rms
from .paths import RAW


@dataclass(frozen=True)
class Soundscape:
    beds: tuple[str, ...]      # DEMAND environments
    events: tuple[str, ...]    # ESC-50 categories


SOUNDSCAPES: dict[str, Soundscape] = {
    "village": Soundscape(
        beds=("NFIELD", "NPARK", "DLIVING", "DKITCHEN"),
        events=("cow", "rooster", "hen", "dog", "crow", "sheep", "insects", "crickets", "chirping_birds", "frog"),
    ),
    "market": Soundscape(
        beds=("STRAFFIC", "SPSQUARE", "PSTATION", "TBUS"),
        events=("car_horn", "engine", "siren", "train", "church_bells", "dog", "crow", "footsteps", "clock_alarm"),
    ),
}
SPLITS = ("dev", "test")


@lru_cache(maxsize=None)
def _bed(env: str, split: str) -> np.ndarray:
    wav = load(RAW / "demand" / env / "ch01.wav")
    half = len(wav) // 2
    return wav[:half] if split == "dev" else wav[half:]


@lru_cache(maxsize=None)
def _events(categories: tuple[str, ...], split: str) -> list[np.ndarray]:
    root = RAW / "esc50" / "ESC-50-master"
    meta = pd.read_csv(root / "meta" / "esc50.csv")
    folds = (4, 5) if split == "dev" else (1, 2, 3)
    meta = meta[meta.category.isin(categories) & meta.fold.isin(folds)]
    clips = []
    for f in meta.filename:
        w = _trim(load(root / "audio" / f))
        if len(w) > 0.2 * SR:
            clips.append(w)
    return clips


def _trim(wav: np.ndarray, frame: int = 320, top_db: float = 30.0) -> np.ndarray:
    """Cut leading/trailing near-silence from an event clip."""
    n = len(wav) // frame
    if n == 0:
        return wav
    e = 10 * np.log10(np.mean(wav[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    on = np.flatnonzero(e > e.max() - top_db)
    return wav[on[0] * frame: (on[-1] + 1) * frame] if len(on) else wav[:0]


class NoiseBank:
    """Generates ``n`` samples of a soundscape: looped ambience bed + randomly placed events."""

    def __init__(self, soundscape: str, split: str = "test", events_per_min: float = 8.0,
                 event_db: tuple[float, float] = (0.0, 8.0)):
        if soundscape not in SOUNDSCAPES:
            raise ValueError(f"unknown soundscape {soundscape!r}; choose from {list(SOUNDSCAPES)}")
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.name, self.split = soundscape, split
        self.spec = SOUNDSCAPES[soundscape]
        self.events_per_min = events_per_min
        self.event_db = event_db

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        bed_src = _bed(str(rng.choice(self.spec.beds)), self.split)
        off = int(rng.integers(0, len(bed_src)))
        reps = int(np.ceil((off + n) / len(bed_src)))
        bed = np.tile(bed_src, reps)[off: off + n].astype(np.float32)
        bed_level = rms(bed)
        bed = bed / bed_level * 0.05 if bed_level > 1e-6 else bed

        events = _events(self.spec.events, self.split)
        n_ev = rng.poisson(self.events_per_min * n / SR / 60)
        for _ in range(n_ev if events else 0):
            ev = events[int(rng.integers(len(events)))]
            ev = ev / max(rms(ev), 1e-6) * 0.05 * db_to_gain(rng.uniform(*self.event_db))
            at = int(rng.integers(0, max(1, n - len(ev))))
            seg = ev[: n - at]
            bed[at: at + len(seg)] += seg
        return bed
