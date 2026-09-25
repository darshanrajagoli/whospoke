"""Voice-activity and overlapped-speech detection with pyannote's ``segmentation-3.0`` model.

One forward pass of the model gives, for every ~17 ms frame, the probability of each
"powerset" class (silence, 1 speaker, 2 speakers, ...). From that we derive:

* speech regions   — frames where at least one person talks,
* overlap regions  — frames where two or more people talk at once.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from .audio import SR

Region = tuple[float, float]


def _device(device: str | None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


@lru_cache(maxsize=2)
def _pipelines(device: str | None = None):
    from pyannote.audio import Model
    from pyannote.audio.pipelines import OverlappedSpeechDetection, VoiceActivityDetection

    model = Model.from_pretrained("pyannote/segmentation-3.0")
    params = {"min_duration_on": 0.0, "min_duration_off": 0.0}
    vad = VoiceActivityDetection(segmentation=model).instantiate(params)
    osd = OverlappedSpeechDetection(segmentation=model).instantiate(params)
    vad.to(_device(device))
    osd.to(_device(device))
    return vad, osd


def _regions(annotation) -> list[Region]:
    return [(round(s.start, 3), round(s.end, 3)) for s in annotation.get_timeline().support()]


def _as_input(wav: np.ndarray) -> dict:
    return {"waveform": torch.from_numpy(np.asarray(wav, dtype=np.float32))[None], "sample_rate": SR}


def speech_regions(wav: np.ndarray, device: str | None = None) -> list[Region]:
    vad, _ = _pipelines(device)
    return _regions(vad(_as_input(wav)))


def overlap_regions(wav: np.ndarray, device: str | None = None) -> list[Region]:
    _, osd = _pipelines(device)
    return _regions(osd(_as_input(wav)))


def complement(regions: list[Region], duration: float, pad: float = 0.0) -> list[Region]:
    """Gaps between (padded) ``regions`` inside ``[0, duration]``."""
    out, t = [], 0.0
    for s, e in sorted(regions):
        s, e = max(0.0, s - pad), min(duration, e + pad)
        if s > t:
            out.append((t, s))
        t = max(t, e)
    if t < duration:
        out.append((t, duration))
    return out


def intersect(a: list[Region], b: list[Region]) -> list[Region]:
    out = []
    for s1, e1 in a:
        for s2, e2 in b:
            s, e = max(s1, s2), min(e1, e2)
            if e > s:
                out.append((s, e))
    return sorted(out)


def total(regions: list[Region]) -> float:
    return float(sum(e - s for s, e in regions))
