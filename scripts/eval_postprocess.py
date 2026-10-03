"""Stage 4 on the TEST set: does the LLM clean-up help, and how far do upstream errors reach the report?

Input: ``results/eval_test_indicconformer/*.json`` (the true transcript and every system's Stage-3 transcript, written
by ``scripts/evaluate.py``). No audio model is needed, only the Stage-4 LLM server (``scripts/serve_llm.py``).

Stage 4 is run on the transcripts of four systems, one more upstream error source each:

  reference     the true words and speakers               what Stage 4 does to a perfect transcript
  oracle-clean  ASR on each clean voice, true timeline    + recognition errors
  oracle-mix    ASR on the noisy mixture, true timeline   + noise and overlap
  B-spectral    the full pipeline (Order B)               + separation and diarization errors

Per conversation and system (``results/eval_postprocess.csv``):

  cpwer_raw, cpwer_clean   who-said-what error of the Stage-3 text and of the Stage-4 repaired text
  keywords_true            share of the report's keywords that were really said (true transcript or its translation)
  summary_true             share of the content words of the summary and key points that were really said
  keyword_f1, summary_f1   word overlap of the keywords / summary with the report made from the true transcript
  guardrail counts         lines repaired, reverted, restored, flagged; keywords and actions dropped
  llm_s, rtf               Stage-4 time, and that time divided by the audio length

Reports are cached in ``results/eval_postprocess/``; an interrupted run resumes where it stopped.

    python scripts/serve_llm.py &                       # once, in another terminal
    python scripts/eval_postprocess.py --groups 0 1     # 18 conversations: 2 speaker groups × all 9 conditions
    python scripts/eval_postprocess.py                  # all 63 (group 7 was used to develop the prompts, D32)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_report import boot_ci, paired_diff_ci  # noqa: E402
from whospoke.hinglish import romanise  # noqa: E402
from whospoke.llm_postprocess import (  # noqa: E402
    DEFAULT_BASE_URL, DEFAULT_CONTEXT, DEFAULT_MODEL, OpenAICompatibleLLM, PostProcessor, _content_tokens, _fingerprint,
)
from whospoke.metrics import cp_error  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
RES = PROJECT / "results"
SYSTEMS = ["reference", "oracle-clean", "oracle-mix", "B-spectral"]
PROMPT_DEV_GROUP = 7     # the Stage-4 prompts were developed on this test group, so it is never scored (D32)


def by_speaker(lines: list[dict]) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    for x in sorted(lines, key=lambda x: x["start"]):
        if x["text"].strip():
            out.setdefault(x["speaker"], []).append(x["text"])
    return {k: " ".join(v) for k, v in out.items()}


def stage3_transcript(conv: dict, system: str, duration_s: float) -> dict:
    if system == "reference":
        lines = [{"speaker": s["speaker"], "start": s["start"], "end": s["end"], "text": s["text"],
                  "hinglish": romanise(s["text"])} for s in sorted(conv["reference"], key=lambda s: s["start"])]
    else:
        lines = conv["systems"][system]
    return {"order": "B" if system == "B-spectral" else "oracle", "duration_s": duration_s, "lines": lines}


def words(texts: list[str]) -> set[str]:
    """Content words (the same filter the Stage-4 guardrails use)."""
    return {t for s in texts for t in _content_tokens(s or "") if t != "conversation"}


def f1(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    common = len(a & b)
    return 0.0 if common == 0 else 2 * common / (len(a) + len(b))


class RSSSampler:
    """Peak resident memory of the LLM server process (``--server-pid``), sampled every 0.5 s."""

    def __init__(self, pid: int | None):
        self.peak_mb, self._stop = 0.0, threading.Event()
        if pid:
            import psutil

            self._proc = psutil.Process(pid)
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.peak_mb = max(self.peak_mb, self._proc.memory_info().rss / 2**20)
            except Exception:
                return
            self._stop.wait(0.5)

    def stop(self) -> float:
        self._stop.set()
        return round(self.peak_mb, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--asr", default="indicconformer")
    ap.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    ap.add_argument("--groups", nargs="+", type=int, default=None,
                    help=f"speaker groups to use (default: all except {PROMPT_DEV_GROUP}, D32)")
    ap.add_argument("--llm-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--llm-model", default=DEFAULT_MODEL)
    ap.add_argument("--llm-context", type=int, default=DEFAULT_CONTEXT)
    ap.add_argument("--server-pid", type=int, default=None, help="LLM server process, to record its peak memory")
    ap.add_argument("--force", action="store_true", help="re-run Stage 4 even if a cached report exists")
    ap.add_argument("--workers", type=int, default=1,
                    help="reports made at once (start the server with the same --parallel); the measured Stage-4 "
                         "times then include the sharing of the CPU")
    args = ap.parse_args()

    meta = pd.read_csv(RES / f"eval_test_{args.asr}.csv")
    stage3_cpwer = meta.set_index(["id", "system"]).cpwer
    meta = meta.drop_duplicates("id").set_index("id")
    if args.groups is not None and PROMPT_DEV_GROUP in args.groups:
        raise SystemExit(f"group {PROMPT_DEV_GROUP} was used to develop the Stage-4 prompts and is not scored (D32)")
    meta = meta[meta.group != PROMPT_DEV_GROUP]
    if args.groups is not None:
        meta = meta[meta.group.isin(args.groups)]
    cache = RES / "eval_postprocess"
    cache.mkdir(parents=True, exist_ok=True)
    OpenAICompatibleLLM(args.llm_url, args.llm_model).ping()
    rss = RSSSampler(args.server_pid)
    # Conversations of one speaker group share their speech across conditions, so some Stage-3 transcripts are
    # identical (always for "reference"). Stage 4 is deterministic (temperature 0): reuse the report.
    by_hash = {}
    for f in cache.glob("*.json"):
        rep_ = json.loads(f.read_text(encoding="utf-8"))
        by_hash.setdefault((rep_["provenance"]["source_sha256"], f.stem.split("__")[1]), rep_)
    t_start = time.perf_counter()

    convs = {cid: json.loads((RES / f"eval_test_{args.asr}" / f"{cid}.json").read_text(encoding="utf-8"))
             for cid in meta.index}
    jobs = {}                                  # one Stage-4 run per distinct transcript that is not cached yet
    for cid, m in meta.iterrows():
        for system in args.systems:
            src = stage3_transcript(convs[cid], system, m.duration_s)
            key = (_fingerprint(src), system)
            if args.force or (not (cache / f"{cid}__{system}.json").exists() and key not in by_hash):
                jobs.setdefault(key, (cid, system, src))
    local = threading.local()

    def make_report(key):
        cid, system, src = jobs[key]
        if not hasattr(local, "post"):
            local.post = PostProcessor(OpenAICompatibleLLM(args.llm_url, args.llm_model),
                                       context_tokens=args.llm_context)
        rep_ = local.post.process_transcript(src).to_dict()
        (cache / f"{cid}__{system}.json").write_text(json.dumps(rep_, ensure_ascii=False, indent=1), encoding="utf-8")
        return key, rep_

    print(f"{len(jobs)} reports to make, {args.workers} at a time", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (key, rep_) in enumerate(pool.map(make_report, list(jobs)), 1):
            by_hash[key] = rep_
            print(f"  [{i}/{len(jobs)}] {jobs[key][0]} {jobs[key][1]}: {rep_['diagnostics']['llm_seconds']:.0f} s",
                  flush=True)

    rows = []
    for cid, m in meta.iterrows():
        conv = convs[cid]
        ref_text = by_speaker(conv["reference"])
        reports = {}
        for system in args.systems:
            f = cache / f"{cid}__{system}.json"
            key = (_fingerprint(stage3_transcript(conv, system, m.duration_s)), system)
            if f.exists() and not (args.force and key in jobs):
                reports[system] = json.loads(f.read_text(encoding="utf-8"))
            else:                              # an identical transcript's report
                reports[system] = by_hash[key]
                f.write_text(json.dumps(reports[system], ensure_ascii=False, indent=1), encoding="utf-8")
        ref_report = reports.get("reference")
        truth = [s["text"] for s in conv["reference"]] + [romanise(s["text"]) for s in conv["reference"]]
        if ref_report:
            truth += [x["translation"] for x in ref_report["cleaned_dialogue"]]
        truth_words = words(truth)
        for system, rep in reports.items():
            src = stage3_transcript(conv, system, m.duration_s)["lines"]
            raw = cp_error(ref_text, by_speaker(src))
            clean = cp_error(ref_text, by_speaker(rep["cleaned_dialogue"]))
            if system != "reference" and abs(raw["rate"] - stage3_cpwer[(cid, system)]) > 1e-9:
                print(f"warning: {cid} {system}: Stage-3 cpWER {raw['rate']:.4f} != evaluate.py's "
                      f"{stage3_cpwer[(cid, system)]:.4f}")
            kw = [k for k in rep["keywords"] if words([k])]
            d = rep["diagnostics"]
            rows.append({
                "id": cid, "system": system, "group": m.group, "n_speakers": m.n_speakers,
                "overlap_level": m.overlap_level, "noise": m.noise, "duration_s": m.duration_s,
                "n_lines": len(rep["cleaned_dialogue"]), "ref_words": raw["n_ref"],
                "cp_errors_raw": raw["errors"], "cp_errors_clean": clean["errors"],
                "cpwer_raw": raw["rate"], "cpwer_clean": clean["rate"],
                "keywords": len(kw),
                "keywords_true": np.mean([words([k]) <= truth_words for k in kw]) if kw else np.nan,
                "summary_true": (len(sw & truth_words) / len(sw)) if (sw := words([rep["executive_summary"],
                                                                                    *rep["key_points"]])) else np.nan,
                "keyword_f1": f1(words(rep["keywords"]), words(ref_report["keywords"])) if ref_report else np.nan,
                "summary_f1": f1(words([rep["executive_summary"]]), words([ref_report["executive_summary"]]))
                if ref_report else np.nan,
                "actions": len(rep["action_items"]),
                **{k: d.get(k, 0) for k in ("lines_changed", "lines_reverted", "lines_restored", "lines_flagged_by_llm",
                                     "translations_too_long",
                                     "missing_translations", "failed_chunks", "chunk_splits", "invalid_replies",
                                     "keywords_dropped", "actions_dropped", "synthesis_failed", "llm_calls",
                                     "prompt_tokens", "completion_tokens", "schema_enforced")},
                "llm_s": d["llm_seconds"], "rtf": d["llm_seconds"] / m.duration_s,
            })
        last = rows[-len(reports):]
        print(f"{cid}: " + "  ".join(f"{r['system']}:{r['cpwer_raw']:.2f}->{r['cpwer_clean']:.2f}" for r in last),
              flush=True)

    df = pd.DataFrame(rows)
    out = RES / "eval_postprocess.csv"
    df.to_csv(out, index=False)
    summary = {"model": args.llm_model, "context_tokens": args.llm_context, "conversations": int(df.id.nunique()),
               "server_peak_rss_mb": rss.stop() or None, "workers": args.workers,
               "wall_s_this_run": round(time.perf_counter() - t_start, 1)}
    # this run only (git-ignored, overwritten by the next run): to keep it, save it as
    # eval_postprocess_summary_gXX_..._<machine>.json with "groups", "machine" and "server" added (D33)
    (RES / "eval_postprocess_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print(f"\n{df.id.nunique()} conversations  ({out})")
    piv_raw = df.assign(r=df.cp_errors_raw / df.ref_words).pivot(index="id", columns="system", values="r")
    piv_clean = df.assign(r=df.cp_errors_clean / df.ref_words).pivot(index="id", columns="system", values="r")
    for s in args.systems:
        g = df[df.system == s]
        mean, lo, hi = paired_diff_ci(piv_clean[s], piv_raw[s])
        k_lo, k_hi = boot_ci(g.keywords_true.dropna().to_numpy()) if g.keywords_true.notna().any() else (np.nan,) * 2
        print(f"{s:13s} cpWER {g.cp_errors_raw.sum() / g.ref_words.sum():6.1%} -> "
              f"{g.cp_errors_clean.sum() / g.ref_words.sum():6.1%}  (Δ {100 * mean:+.1f} pts [{100 * lo:+.1f}, "
              f"{100 * hi:+.1f}])  keywords really said {g.keywords_true.mean():.0%} [{k_lo:.0%}, {k_hi:.0%}]  "
              f"summary words really said {g.summary_true.mean():.0%}  "
              f"keyword F1 {g.keyword_f1.mean():.2f}  summary F1 {g.summary_f1.mean():.2f}  RTF {g.rtf.mean():.2f}")


if __name__ == "__main__":
    main()
