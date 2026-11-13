"""Order B, Stage 1 policy — chosen on the DEV set only.

Question: once we know where people overlap, what audio should the ASR hear for each turn?

  mixture             no separation at all
  turn                replace the whole overlapping turn by its separated voice (first design)
  splice-<c>          separate each overlap region (with c seconds of context) and replace only the overlapped
                      stretch by the separated voice; the rest of the turn stays the original mixture
  splice-0.5+second   splice-0.5, and the region's other separated output becomes an extra turn for the speaker it
                      sounds like when that speaker has no turn there (``Pipeline.second_voice``). Its threshold is
                      tuned here: each threshold in SECOND_THRESHOLDS is a row ``splice-0.5+second@<thr>``. The
                      extra turns are transcribed once and the thresholds only decide which of them are kept.

Each policy is scored by cpWER (who-said-what word error) with (a) the true timeline and (b) our tuned
Order-B spectral diarization. Output: results/separation_policy_dev.csv; the winner is used on the test set.
Every second-voice candidate (score, speaker, text) goes to results/second_voice_candidates_<split>.csv.

    python scripts/tune_separation_policy.py
    python scripts/tune_separation_policy.py --policies mixture splice-0.5 splice-0.5+second --out separation_policy_dev_second.csv

``--split test`` is for the single test run after the dev choice: give it ``--thresholds <chosen>``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import diarizer_from, reference_diarization  # noqa: E402
from tune_diarization import cached_features  # noqa: E402
from whospoke.audio import load  # noqa: E402
from whospoke.diarization import Diarizer  # noqa: E402
from whospoke.metrics import cp_error  # noqa: E402
from whospoke.paths import SYNTH, conversation  # noqa: E402
from whospoke.pipeline import Pipeline  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
POLICIES = {"mixture": None, "turn": ("turn", 0.0), "splice-0.5": ("splice", 0.5), "splice-1.0": ("splice", 1.0),
            "splice-0.5+second": ("splice", 0.5)}
SECOND_THRESHOLDS = [-1.0, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]


def by_speaker(lines) -> dict[str, str]:
    hyp: dict[str, list[str]] = {}
    for ln in sorted(lines, key=lambda l: l.start):
        hyp.setdefault(ln.speaker, []).append(ln.text)
    return {k: " ".join(v) for k, v in hyp.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    ap.add_argument("--timelines", nargs="+", default=["oracle", "B-spectral"], choices=["oracle", "B-spectral"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=SECOND_THRESHOLDS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="CSV name in results/ (default separation_policy_<split>.csv)")
    ap.add_argument("--separator", default="convtasnet",
                    help="separator for the overlap regions, e.g. convtasnet-hindi (scripts/finetune_convtasnet.py)")
    args = ap.parse_args()

    tuned = json.loads((RESULTS / "tuned_params.json").read_text(encoding="utf-8"))
    diar = diarizer_from(tuned["B-spectral"])
    pipe = Pipeline("B", separator=args.separator, diarizer=diar, asr="indicconformer")
    # The diarization features (speech, fingerprints, detected overlaps) do not depend on the separator, so they are
    # always cached under the released model's name; the regions themselves are separated by ``args.separator``.
    base = Diarizer()
    sep_cache = {"convtasnet": pipe.separator} if args.separator == "convtasnet" else {}
    index = pd.read_csv(SYNTH / args.split / "index.csv")
    if args.limit:
        index = index.head(args.limit)
    rows, cand_rows = [], []
    # Each finished conversation is kept, so an interrupted run resumes where it stopped (results/cache is git-ignored).
    setup = (sorted(args.policies), sorted(args.timelines), sorted(args.thresholds))
    if args.separator != "convtasnet":
        setup += (args.separator,)
    key = hashlib.sha1(repr(setup).encode()).hexdigest()[:10]
    done_dir = RESULTS / "cache" / f"policy_{args.split}_{key}"
    done_dir.mkdir(parents=True, exist_ok=True)
    for r in index.itertuples():
        saved = done_dir / f"{r.id}.pkl"
        if saved.exists():
            conv_rows, conv_cands = pickle.loads(saved.read_bytes())
            rows += conv_rows
            cand_rows += conv_cands
            print(f"{r.id} (kept from an earlier run)", flush=True)
            continue
        n_rows, n_cands = len(rows), len(cand_rows)
        path = conversation(r.path)
        meta = json.loads((path / "reference.json").read_text(encoding="utf-8"))
        ref_text: dict[str, list[str]] = {}
        for s in sorted(meta["segments"], key=lambda s: s["start"]):
            ref_text.setdefault(s["speaker"], []).append(s["text"])
        ref_text = {k: " ".join(v) for k, v in ref_text.items()}
        mix = load(path / "mixture.wav")
        feats = cached_features(r.id, path, "convtasnet", base, sep_cache)
        timelines = {"oracle": reference_diarization(meta["segments"]), "B-spectral": diar.assign(feats["mix"])}
        timelines["oracle"].info["overlap"] = feats["mix"].overlaps      # same detector output for both
        for tl_name in args.timelines:
            d = timelines[tl_name]
            done: dict[tuple, tuple] = {}                 # (mode, context) -> (units, lines, candidates, their lines)

            def row(pol: str, units, lines, n_second: int = 0) -> dict:
                cp = cp_error(ref_text, by_speaker(lines), "word")
                return {"id": r.id, "timeline": tl_name, "policy": pol, "overlap_level": r.overlap_level,
                        "noise": r.noise, "errors": cp["errors"], "ref_words": cp["n_ref"],
                        "separated_frac": sum(u[2] == "separated" for u in units) / max(1, len(units)),
                        "n_second": n_second}

            for pol in args.policies:
                cfg = POLICIES[pol]
                if cfg is None:
                    units = [(t, mix, "mixture") for t in d.turns]
                    rows.append(row(pol, units, pipe._transcribe(units)))
                    continue
                if cfg not in done:
                    pipe.sep_mode, pipe.sep_context_s = cfg
                    units, cands = pipe._separate_units(mix, d)
                    extra = [(t, a, "second") for _, t, a in cands]
                    lines = pipe._transcribe(units + extra)
                    done[cfg] = (units, lines[: len(units)], cands, lines[len(units):])
                units, lines, cands, extra_lines = done[cfg]
                if not pol.endswith("+second"):
                    rows.append(row(pol, units, lines))
                    continue
                for (score, t, _), ln in zip(cands, extra_lines):
                    cand_rows.append({"id": r.id, "timeline": tl_name, "start": t.start, "end": t.end,
                                      "speaker": t.speaker, "score": round(score, 4), "text": ln.text})
                for thr in args.thresholds:
                    keep = [ln for (score, _, _), ln in zip(cands, extra_lines) if score >= thr]
                    rows.append(row(f"{pol}@{thr:g}", units, lines + keep, len(keep)))
        saved.write_bytes(pickle.dumps((rows[n_rows:], cand_rows[n_cands:])))
        print(r.id, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / (args.out or f"separation_policy_{args.split}.csv"), index=False)
    if cand_rows:
        pd.DataFrame(cand_rows).to_csv(RESULTS / f"second_voice_candidates_{args.split}.csv", index=False,
                                       encoding="utf-8")
    summary = df.groupby(["timeline", "policy"]).apply(
        lambda g: pd.Series({"cpWER": g.errors.sum() / g.ref_words.sum(), "separated_frac": g.separated_frac.mean(),
                             "n_second": g.n_second.sum()}), include_groups=False)
    by_ovl = df.groupby(["timeline", "policy", "overlap_level"]).apply(
        lambda g: g.errors.sum() / g.ref_words.sum(), include_groups=False).unstack()
    print(summary.round(4).to_string())
    print(by_ovl.round(4).to_string())


if __name__ == "__main__":
    main()
