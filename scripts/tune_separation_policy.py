"""Order B, Stage 1 policy — chosen on the DEV set only.

Question: once we know where people overlap, what audio should the ASR hear for each turn?

  mixture          no separation at all
  turn             replace the whole overlapping turn by its separated voice (first design)
  splice-<c>       separate each overlap region (with c seconds of context) and replace only the overlapped
                   stretch by the separated voice; the rest of the turn stays the original mixture

Each policy is scored by cpWER (who-said-what word error) with (a) the true timeline and (b) our tuned
Order-B spectral diarization. Output: results/separation_policy_dev.csv; the winner is used on the test set.

    python scripts/tune_separation_policy.py
"""
from __future__ import annotations

import json
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
POLICIES = {"mixture": None, "turn": ("turn", 0.0), "splice-0.5": ("splice", 0.5), "splice-1.0": ("splice", 1.0)}


def main() -> None:
    tuned = json.loads((RESULTS / "tuned_params.json").read_text(encoding="utf-8"))
    diar = diarizer_from(tuned["B-spectral"])
    pipe = Pipeline("B", separator="convtasnet", diarizer=diar, asr="indicconformer")
    base, sep_cache = Diarizer(), {"convtasnet": pipe.separator}
    index = pd.read_csv(SYNTH / "dev" / "index.csv")
    rows = []
    for r in index.itertuples():
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
        for tl_name, d in timelines.items():
            for pol, cfg in POLICIES.items():
                if cfg is None:
                    units = [(t, mix, "mixture") for t in d.turns]
                else:
                    pipe.sep_mode, pipe.sep_context_s = cfg
                    units = pipe._targeted_separation(mix, d)
                hyp: dict[str, list[str]] = {}
                for ln in sorted(pipe._transcribe(units), key=lambda l: l.start):
                    hyp.setdefault(ln.speaker, []).append(ln.text)
                cp = cp_error(ref_text, {k: " ".join(v) for k, v in hyp.items()}, "word")
                rows.append({"id": r.id, "timeline": tl_name, "policy": pol, "overlap_level": r.overlap_level,
                             "noise": r.noise, "errors": cp["errors"], "ref_words": cp["n_ref"],
                             "separated_frac": sum(u[2] == "separated" for u in units) / max(1, len(units))})
        print(r.id, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "separation_policy_dev.csv", index=False)
    summary = df.groupby(["timeline", "policy"]).apply(
        lambda g: pd.Series({"cpWER": g.errors.sum() / g.ref_words.sum(), "separated_frac": g.separated_frac.mean()}),
        include_groups=False)
    by_ovl = df.groupby(["timeline", "policy", "overlap_level"]).apply(
        lambda g: g.errors.sum() / g.ref_words.sum(), include_groups=False).unstack()
    print(summary.round(4).to_string())
    print(by_ovl.round(4).to_string())


if __name__ == "__main__":
    main()
