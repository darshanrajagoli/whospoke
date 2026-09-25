"""Command line: ``python -m whospoke run recording.wav --out results/demo``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audio import load
from .diarization import Diarizer

TUNED = Path(__file__).resolve().parents[2] / "results" / "tuned_params.json"


def tuned_diarizer(order: str, clustering: str) -> Diarizer:
    """Diarizer with the settings tuned on the dev set (falls back to defaults if not tuned yet)."""
    if TUNED.exists():
        p = json.loads(TUNED.read_text(encoding="utf-8")).get(f"{order}-{clustering}")
        if p:
            return Diarizer(p["clustering"], merge_sim=p["merge_sim"], min_cluster_frac=p["min_cluster_frac"],
                            overlap_aware=p["overlap_aware"], cluster_kwargs=p["cluster_kwargs"])
    return Diarizer(clustering)


def main() -> None:
    ap = argparse.ArgumentParser(prog="whospoke", description="Who spoke what and when — Hindi/Hinglish audio.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="process one audio file")
    run.add_argument("audio")
    run.add_argument("--out", default=None, help="output folder (default: results/runs/<file name>)")
    run.add_argument("--order", choices=["A", "B"], default="B")
    run.add_argument("--clustering", choices=["spectral", "gmm"], default="spectral")
    run.add_argument("--asr", choices=["indicconformer", "indicwav2vec"], default="indicconformer")
    run.add_argument("--speakers", type=int, default=None, help="number of speakers, if known")
    args = ap.parse_args()

    from .pipeline import Pipeline

    pipe = Pipeline(args.order, diarizer=tuned_diarizer(args.order, args.clustering), asr=args.asr)
    result = pipe.run(load(args.audio), n_speakers=args.speakers)
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[2] / "results" / "runs" / Path(args.audio).stem
    result.save(out)
    print(result.transcript(hinglish=True))
    print(f"\nsaved to {out}  |  timings (s): {result.timings}")


if __name__ == "__main__":
    main()
