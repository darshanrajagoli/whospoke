"""Command line for the complete WhoSpoke pipeline.

Examples::

    python -m whospoke run recording.wav
    python -m whospoke run recording.wav --postprocess
    python -m whospoke postprocess results/demo/transcript.json
    python -m whospoke separate recording.wav

``--postprocess`` and ``postprocess`` need the local Stage-4 LLM server: ``python scripts/serve_llm.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .audio import load, save
from .llm_postprocess import (DEFAULT_BASE_URL, DEFAULT_CONTEXT, DEFAULT_MODEL, DEFAULT_TIMEOUT, LLMConnectionError,
                             PostProcessor)

TUNED = Path(__file__).resolve().parents[2] / "results" / "tuned_params.json"


def tuned_diarizer(order: str, clustering: str):
    """Diarizer with the settings tuned on the dev set (falls back to defaults if not tuned yet)."""
    from .diarization import Diarizer

    if TUNED.exists():
        p = json.loads(TUNED.read_text(encoding="utf-8")).get(f"{order}-{clustering}")
        if p:
            return Diarizer(p["clustering"], merge_sim=p["merge_sim"], min_cluster_frac=p["min_cluster_frac"],
                            overlap_aware=p["overlap_aware"], cluster_kwargs=p["cluster_kwargs"])
    return Diarizer(clustering)


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Stage 4 (LLM post-processing; see docs/MILESTONE4.md)")
    g.add_argument("--llm-url", default=DEFAULT_BASE_URL,
                   help=f"OpenAI-compatible server started by scripts/serve_llm.py (default: {DEFAULT_BASE_URL})")
    g.add_argument("--llm-model", default=DEFAULT_MODEL, help=f"model name sent to the server (default: {DEFAULT_MODEL})")
    g.add_argument("--llm-context", type=int, default=DEFAULT_CONTEXT,
                   help=f"model context window in tokens; chunks are sized to fit it (default: {DEFAULT_CONTEXT})")
    g.add_argument("--llm-timeout", type=int, default=DEFAULT_TIMEOUT, help="seconds to wait for one LLM reply")
    g.add_argument("--llm-no-schema", action="store_true",
                   help="do not ask the server to enforce the JSON schema (the Python checks still run)")


def build_postprocessor(args: argparse.Namespace) -> PostProcessor:
    return PostProcessor(model=args.llm_model, base_url=args.llm_url, context_tokens=args.llm_context,
                         timeout_s=args.llm_timeout, json_schema=not args.llm_no_schema)


def default_out(audio: str) -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "runs" / Path(audio).stem


def separate_audio(args: argparse.Namespace) -> None:
    """Stage 1 on its own (the proposal's Milestone-1 deliverable): one .wav per separated voice.

    The whole recording is separated, as in Order A. The tracks are not labelled with speakers: that is Stage 2's job.
    """
    from .separation import Separator

    tracks = Separator(args.separator).separate(load(args.audio))
    peak = float(np.abs(tracks).max()) if tracks.size else 0.0
    if peak > 0.99:                   # the gain fit can exceed full scale; one factor keeps the tracks' balance
        tracks = tracks * (0.99 / peak)
    out = Path(args.out) if args.out else default_out(args.audio)
    for k, track in enumerate(tracks, 1):
        save(out / f"separated_track_{k}.wav", track)
        print(f"saved {out / f'separated_track_{k}.wav'}")


def run_audio(args: argparse.Namespace) -> None:
    from .pipeline import Pipeline

    post = build_postprocessor(args) if args.postprocess else None
    if post is not None:
        post.backend.ping()           # fail now, not after minutes of audio processing
    pipe = Pipeline(args.order, diarizer=tuned_diarizer(args.order, args.clustering), asr=args.asr,
                    postprocessor=post)
    result = pipe.run(load(args.audio), n_speakers=args.speakers)
    out = Path(args.out) if args.out else default_out(args.audio)
    result.save(out)
    print(result.transcript(hinglish=True))
    if result.report is not None:
        print(f"\n{result.report['title']}\n{result.report['executive_summary']}\nreport: {out / 'report.md'}")
    elif result.report_error:
        print(f"\nStage 4 failed ({result.report_error}); the Stage 1-3 outputs were saved. Retry it with:\n"
              f"  python -m whospoke postprocess {out / 'transcript.json'}")
    print(f"\nsaved to {out}  |  timings (s): {result.timings}")


def postprocess_transcript(args: argparse.Namespace) -> None:
    transcript = Path(args.transcript)
    out = Path(args.out) if args.out else transcript.parent
    report = build_postprocessor(args).process_file(transcript)
    report.save(out)
    d = report.diagnostics
    print(f"{report.title}\n\n{report.executive_summary}\n")
    print(f"keywords: {', '.join(report.keywords) or '-'}")
    print(f"{len(report.cleaned_dialogue)} lines, {d.lines_changed} repaired, {len(report.uncertain_lines)} flagged; "
          f"{d.llm_calls} LLM calls in {d.llm_seconds:.0f} s")
    print(f"saved {out / 'report.md'} and {out / 'report.json'}")


def main() -> None:
    # Devanagari output: a Windows console redirected to a file or pipe would otherwise use cp1252 and crash
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="whospoke", description="Who spoke what and when — Hindi/Hinglish audio.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="process one audio file")
    run.add_argument("audio")
    run.add_argument("--out", default=None, help="output folder (default: results/runs/<file name>)")
    run.add_argument("--order", choices=["A", "B"], default="B")
    run.add_argument("--clustering", choices=["spectral", "gmm"], default="spectral")
    run.add_argument("--asr", choices=["indicconformer", "indicwav2vec"], default="indicconformer")
    run.add_argument("--speakers", type=int, default=None, help="number of speakers, if known")
    run.add_argument("--postprocess", action="store_true",
                     help="also run Stage 4 (LLM report: summary, keywords, actions, cleaned dialogue)")
    add_llm_args(run)

    post = sub.add_parser("postprocess", help="run Stage 4 on an existing transcript.json")
    post.add_argument("transcript")
    post.add_argument("--out", default=None, help="output folder (default: the transcript's folder)")
    add_llm_args(post)

    sep = sub.add_parser("separate", help="Stage 1 only: write one .wav per separated voice")
    sep.add_argument("audio")
    sep.add_argument("--out", default=None, help="output folder (default: results/runs/<file name>)")
    sep.add_argument("--separator", choices=["convtasnet", "convtasnet-clean", "sepformer"], default="convtasnet")

    args = ap.parse_args()
    try:
        if args.cmd == "run":
            run_audio(args)
        elif args.cmd == "postprocess":
            postprocess_transcript(args)
        elif args.cmd == "separate":
            separate_audio(args)
    except LLMConnectionError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
