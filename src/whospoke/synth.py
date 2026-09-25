"""Conversation simulator: builds multi-speaker test recordings with exact ground truth.

Real broadcast audio has no labels, so it cannot be scored. Instead we rebuild "conversations"
from real IndicVoices phone-conversation recordings: each speaker contributes their own side of a
real call (spontaneous, naturally code-switched Hindi), turns alternate between speakers with
natural gaps, some turn changes overlap (people talking over each other), and real "village" or
"market" background noise (``noise.py``) is added at a chosen SNR.

Because we assembled the mixture ourselves we know, for every sample, who is speaking and what
they said — so separation, diarization and transcription can all be measured.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .audio import SR, active_rms, db_to_gain, save, scale_to_snr
from .corpora import Utterance


@dataclass
class SynthConfig:
    n_speakers: int = 2
    target_s: float = 75.0              # stop adding turns once the conversation is this long
    overlap_prob: float = 0.3           # chance that a turn change overlaps the previous turn
    overlap_s: tuple[float, float] = (0.5, 2.5)
    gap_s: tuple[float, float] = (0.15, 0.8)       # silence between non-overlapping turns
    chunks_per_turn: tuple[int, int] = (1, 3)      # consecutive utterances a speaker says per turn
    intra_gap_s: tuple[float, float] = (0.1, 0.35)  # pause between utterances within one turn
    level_jitter_db: float = 3.0        # speakers are not equally loud
    snr_db: float | None = 10.0         # speech-to-noise ratio; None = no noise
    seed: int = 0


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Conversation:
    id: str
    mixture: np.ndarray
    sources: dict[str, np.ndarray]      # clean per-speaker tracks, same length and gain as in the mixture
    noise: np.ndarray | None
    segments: list[Segment]
    config: SynthConfig
    extra: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.mixture) / SR

    @property
    def speakers(self) -> list[str]:
        return list(self.sources)

    def activity(self) -> list[Segment]:
        """Exact per-speaker speech activity (pauses inside utterances removed) — the DER reference."""
        return reference_activity(self.sources, self.segments)

    def overlap_ratio(self) -> float:
        """Fraction of speech time during which two or more people talk at once."""
        return overlap_stats(self.activity(), self.duration)["overlap_ratio"]

    def save(self, out_dir: str | Path, write_sources: bool = True) -> Path:
        out = Path(out_dir) / self.id
        save(out / "mixture.wav", self.mixture)
        if write_sources:
            for spk, wav in self.sources.items():
                save(out / "sources" / f"{spk}.wav", wav)
            if self.noise is not None:
                save(out / "sources" / "noise.wav", self.noise)
        activity = self.activity()
        write_rttm(out / "reference.rttm", self.id, activity)
        meta = {
            "id": self.id,
            "duration_s": round(self.duration, 3),
            "speakers": self.speakers,
            **overlap_stats(activity, self.duration),
            "config": asdict(self.config),
            "segments": [asdict(s) for s in self.segments],             # utterances with their text
            "activity": [[s.start, s.end, s.speaker] for s in activity],  # exact speech activity (DER reference)
            **self.extra,
        }
        (out / "reference.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return out


def overlap_stats(segments: list[Segment], duration: float, res: float = 0.01) -> dict:
    n = int(np.ceil(duration / res)) + 1
    count = np.zeros(n, dtype=np.int16)
    for s in segments:
        count[int(s.start / res): int(s.end / res)] += 1
    speech = float((count > 0).sum() * res)
    overlap = float((count > 1).sum() * res)
    return {
        "speech_s": round(speech, 2),
        "overlap_s": round(overlap, 2),
        "overlap_ratio": round(overlap / speech, 4) if speech else 0.0,
        "max_concurrent": int(count.max()) if n else 0,
    }


def speech_activity(source: np.ndarray, segments: list[Segment], speaker: str, top_db: float = 40.0,
                    fill_gap_s: float = 0.25, min_island_s: float = 0.1, frame_s: float = 0.02) -> list[Segment]:
    """Exact reference speech activity for one speaker, read off their CLEAN track.

    Utterance boundaries include pauses *inside* the utterance (IndicVoices clips run up to 30 s).
    Within each of the speaker's utterances, frames louder than ``top_db`` below that utterance's
    loudest frame count as speech; pauses shorter than ``fill_gap_s`` are bridged (normal gaps between
    words), and islands shorter than ``min_island_s`` are dropped. Model-free, so no system is favoured.
    """
    hop = int(frame_s * SR)
    out: list[Segment] = []
    for seg in segments:
        if seg.speaker != speaker:
            continue
        a, b = int(seg.start * SR), int(seg.end * SR)
        x = source[a:b]
        n = len(x) // hop
        if n == 0:
            continue
        e = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1) + 1e-12)
        on = e > e.max() - top_db
        runs, i = [], 0
        while i < n:
            if on[i]:
                j = i
                while j < n and on[j]:
                    j += 1
                runs.append([i, j])
                i = j
            else:
                i += 1
        merged: list[list[int]] = []
        for r in runs:
            if merged and (r[0] - merged[-1][1]) * frame_s < fill_gap_s:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        for s, t in merged:
            if (t - s) * frame_s >= min_island_s:
                out.append(Segment(speaker, round(seg.start + s * frame_s, 3), round(seg.start + t * frame_s, 3), ""))
    return out


def reference_activity(conv_sources: dict[str, np.ndarray], segments: list[Segment]) -> list[Segment]:
    act = [s for spk, src in conv_sources.items() for s in speech_activity(src, segments, spk)]
    return sorted(act, key=lambda s: s.start)


def write_rttm(path: str | Path, uri: str, segments: list[Segment]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"SPEAKER {uri} 1 {s.start:.3f} {s.duration:.3f} <NA> <NA> {s.speaker} <NA> <NA>"
        for s in sorted(segments, key=lambda s: s.start)
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_rttm(path: str | Path) -> list[Segment]:
    segs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            start, dur = float(p[3]), float(p[4])
            segs.append(Segment(p[7], start, start + dur, ""))
    return segs


def trim_silence(wav: np.ndarray, top_db: float = 35.0, frame: int = 320, pad_s: float = 0.05) -> np.ndarray:
    """Remove leading/trailing frames more than ``top_db`` below the loudest frame (keep ``pad_s``)."""
    n = len(wav) // frame
    if n < 3:
        return wav
    e = 10 * np.log10(np.mean(wav[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1) + 1e-12)
    on = np.flatnonzero(e > e.max() - top_db)
    pad = int(pad_s * SR)
    a = max(0, on[0] * frame - pad)
    b = min(len(wav), (on[-1] + 1) * frame + pad)
    return wav[a:b]


def simulate(
    conv_id: str,
    speakers: dict[str, list[Utterance]],
    cfg: SynthConfig,
    noise: np.ndarray | None = None,
) -> Conversation:
    """Interleave each speaker's utterances (kept in their original order) into one conversation.

    ``speakers`` maps a speaker label to that speaker's utterances. ``noise`` (if given and
    ``cfg.snr_db`` is set) must be at least as long as the result; it is trimmed and scaled.
    """
    rng = np.random.default_rng(cfg.seed)
    labels = list(speakers)
    # Trim silence at utterance edges so reference speech segments are tight.
    queues = {s: [Utterance(u.speaker, u.text, trim_silence(u.wav)) for u in utts] for s, utts in speakers.items()}
    queues = {s: [u for u in q if u.duration > 0.2] for s, q in queues.items()}
    gains = {s: db_to_gain(rng.uniform(-cfg.level_jitter_db, cfg.level_jitter_db)) for s in labels}

    # 1) Plan turns: who speaks, which utterances, and where each utterance starts.
    placed: list[tuple[str, Utterance, float]] = []   # (speaker, utterance, start time)
    cur = str(rng.choice(labels))
    prev_start = prev_end = prev_prev_end = 0.0
    while True:
        avail = [s for s in labels if queues[s]]
        if not avail or (placed and prev_end >= cfg.target_s):
            break
        if cur not in avail:
            cur = str(rng.choice(avail))
        k = int(rng.integers(cfg.chunks_per_turn[0], cfg.chunks_per_turn[1] + 1))
        utts = [queues[cur].pop(0) for _ in range(min(k, len(queues[cur])))]
        gaps = [0.0] + [float(rng.uniform(*cfg.intra_gap_s)) for _ in utts[1:]]
        turn_len = sum(u.duration for u in utts) + sum(gaps)

        if not placed:
            start = float(rng.uniform(0.2, 0.8))
        elif rng.random() < cfg.overlap_prob:
            ov = min(rng.uniform(*cfg.overlap_s), 0.5 * (prev_end - prev_start), 0.5 * turn_len)
            start = max(prev_end - ov, prev_prev_end + 0.05)   # never three people at once
        else:
            start = prev_end + float(rng.uniform(*cfg.gap_s))

        pos = start
        for u, gap in zip(utts, gaps):
            pos += gap
            placed.append((cur, u, pos))
            pos += u.duration
        prev_prev_end, prev_start, prev_end = prev_end, start, max(prev_end, pos)
        others = [s for s in labels if s != cur and queues[s]]
        cur = str(rng.choice(others)) if others else cur

    # 2) Render each speaker's clean track (every utterance levelled to the same speech RMS).
    total = int(np.ceil((prev_end + rng.uniform(0.3, 0.8)) * SR))
    sources = {s: np.zeros(total, dtype=np.float32) for s in labels}
    segments: list[Segment] = []
    for spk, u, start in placed:
        a = int(round(start * SR))
        sources[spk][a: a + len(u.wav)] += u.wav * (gains[spk] * 0.05 / max(active_rms(u.wav), 1e-4))
        segments.append(Segment(spk, round(a / SR, 3), round((a + len(u.wav)) / SR, 3), u.text))

    speech = sum(sources.values())
    noise_track = None
    if noise is not None and cfg.snr_db is not None:
        if len(noise) < total:
            raise ValueError(f"noise too short: {len(noise)} < {total}")
        noise_track = scale_to_snr(speech, noise[:total].astype(np.float32), cfg.snr_db)
    mixture = speech + (noise_track if noise_track is not None else 0.0)

    # One common gain keeps mixture == sum(sources) + noise while avoiding clipping.
    peak = float(np.max(np.abs(mixture))) or 1.0
    g = min(1.0, 0.9 / peak)
    sources = {s: (w * g).astype(np.float32) for s, w in sources.items()}
    noise_track = None if noise_track is None else (noise_track * g).astype(np.float32)
    mixture = (mixture * g).astype(np.float32)
    segments.sort(key=lambda s: s.start)
    return Conversation(conv_id, mixture, sources, noise_track, segments, cfg)
