"""Stage 1 benchmark — how well does each separator pull overlapping voices apart?

Two views, matching the two pipeline orders:

* **overlap regions** (what Order B separates): every stretch where exactly two reference speakers
  overlap, plus 1 s of context either side, is separated and scored against the two clean voices.
* **whole recording** (what Order A separates): 2-speaker conversations separated end-to-end
  (window stitching included) and scored against the two full clean tracks.

Metric: SI-SDR improvement over the unprocessed mixture (dB; higher is better; 0 = no help).

    python scripts/eval_separation.py --split dev
    python scripts/eval_separation.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whospoke.paths import SYNTH, conversation  # noqa: E402
from whospoke.audio import SR, load  # noqa: E402
from whospoke.metrics import pit_si_sdr, si_sdr  # noqa: E402
from whospoke.separation import SEPARATORS, Separator  # noqa: E402
from whospoke.synth import Segment, overlap_stats  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]


def load_conv(path: Path):
    ref = json.loads((path / "reference.json").read_text(encoding="utf-8"))
    mix = load(path / "mixture.wav")
    srcs = {s: load(path / "sources" / f"{s}.wav") for s in ref["speakers"]}
    segs = [Segment(**s) for s in ref["segments"]]
    return ref, mix, srcs, segs


def two_speaker_overlaps(segs: list[Segment], min_len: float = 0.3) -> list[tuple[float, float, str, str]]:
    out = []
    for i, a in enumerate(segs):
        for b in segs[i + 1:]:
            if a.speaker == b.speaker:
                continue
            s, e = max(a.start, b.start), min(a.end, b.end)
            if e - s >= min_len:
                out.append((s, e, a.speaker, b.speaker))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--models", nargs="+", default=list(SEPARATORS))
    ap.add_argument("--context", type=float, default=1.0)
    args = ap.parse_args()

    index = pd.read_csv(SYNTH / args.split / "index.csv")
    models = {m: Separator(m) for m in args.models}
    rows = []
    for r in index.itertuples():
        ref, mix, srcs, segs = load_conv(conversation(r.path))
        # --- overlap regions
        for (s, e, a, b) in two_speaker_overlaps(segs):
            lo, hi = int(max(0, s - args.context) * SR), int(min(len(mix) / SR, e + args.context) * SR)
            x, refs = mix[lo:hi], [srcs[a][lo:hi], srcs[b][lo:hi]]
            base = float(np.mean([si_sdr(x, y) for y in refs]))
            for name, sep in models.items():
                t = time.perf_counter()
                est = sep.separate(x)
                dt = time.perf_counter() - t
                score, _ = pit_si_sdr(list(est), refs)
                rows.append({"id": r.id, "view": "overlap-region", "model": name, "noise": r.noise,
                             "n_speakers": r.n_speakers, "overlap_level": r.overlap_level, "dur_s": (hi - lo) / SR,
                             "si_sdr_mix": base, "si_sdr": score, "si_sdri": score - base, "rtf": dt / ((hi - lo) / SR)})
        # --- whole recording (2 speakers only: a 2-output separator cannot hold 3 voices)
        if r.n_speakers == 2 and r.overlap_level != "ovl-none":
            refs = [srcs[s] for s in ref["speakers"]]
            base = float(np.mean([si_sdr(mix, y) for y in refs]))
            for name, sep in models.items():
                t = time.perf_counter()
                est = sep.separate(mix)
                dt = time.perf_counter() - t
                score, _ = pit_si_sdr(list(est), refs)
                rows.append({"id": r.id, "view": "whole-recording", "model": name, "noise": r.noise,
                             "n_speakers": 2, "overlap_level": r.overlap_level, "dur_s": len(mix) / SR,
                             "si_sdr_mix": base, "si_sdr": score, "si_sdri": score - base, "rtf": dt / (len(mix) / SR)})
        print(f"{r.id}: done", flush=True)

    df = pd.DataFrame(rows)
    out = PROJECT / "results" / f"separation_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = df.groupby(["view", "model", "noise"]).si_sdri.mean().unstack("noise").round(2)
    summary["all"] = df.groupby(["view", "model"]).si_sdri.mean().round(2)
    summary["n"] = df.groupby(["view", "model"]).size()
    summary["rtf"] = df.groupby(["view", "model"]).rtf.mean().round(4)
    print(summary.to_string())
    summary.to_csv(PROJECT / "results" / f"separation_{args.split}_summary.csv")


if __name__ == "__main__":
    main()
