"""Tune the diarizer's clustering settings on the DEV set (never on test).

For each dev conversation the slow neural features (VAD, voice fingerprints, overlap detection)
are computed once — on the raw mixture (Order B) and on the Conv-TasNet-separated tracks
(Order A) — and cached. Then every combination of clustering settings is scored by DER.

    python scripts/tune_diarization.py
Writes results/tuning_diarization_dev.csv and results/tuned_params.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whospoke.paths import SYNTH, conversation  # noqa: E402
from whospoke.audio import load  # noqa: E402
from whospoke.diarization import Diarizer  # noqa: E402
from whospoke.metrics import der  # noqa: E402
from whospoke.separation import Separator  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
CACHE = PROJECT / "results" / "cache" / "features"

GRID = {
    "spectral": {"p_percentile": [0.1, 0.2, 0.3, 0.4]},
    "gmm": {"pca_dims": [4, 8, 16]},
}
MERGE = [None, 0.4, 0.5, 0.6, 0.7]
MIN_FRAC = [0.0, 0.03, 0.06]


def cached_features(conv_id: str, path: Path, separator: str, base: Diarizer, sep_cache: dict):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{conv_id}__{separator}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    mix = load(path / "mixture.wav")
    if separator not in sep_cache:
        sep_cache[separator] = Separator(separator)
    streams = sep_cache[separator].separate(mix)
    feats = {"mix": base.analyse(mix, overlaps=True), "streams": base.analyse_streams(streams)}
    f.write_bytes(pickle.dumps(feats))
    return feats


def reference(path: Path):
    ref = json.loads((path / "reference.json").read_text(encoding="utf-8"))
    return [tuple(a) for a in ref["activity"]], len(ref["speakers"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--separator", default="convtasnet")
    args = ap.parse_args()

    index = pd.read_csv(SYNTH / args.split / "index.csv")
    base = Diarizer()
    sep_cache: dict = {}
    data = []
    for r in index.itertuples():
        feats = cached_features(r.id, conversation(r.path), args.separator, base, sep_cache)
        ref, n = reference(conversation(r.path))
        data.append((r, feats, ref, n))
        print(f"features: {r.id}", flush=True)

    rows = []
    for method, grid in GRID.items():
        keys = list(grid)
        for values in itertools.product(*grid.values()):
            ck = dict(zip(keys, values))
            for merge, min_frac, ovl in itertools.product(MERGE, MIN_FRAC, [True, False]):
                d = Diarizer(method, merge_sim=merge, min_cluster_frac=min_frac, overlap_aware=ovl, cluster_kwargs=ck)
                for order in ("B", "A"):
                    if order == "A" and not ovl:
                        continue   # overlap detection is not used on separated tracks
                    ders, spk_err = [], []
                    for r, feats, ref, n in data:
                        hyp = d.assign(feats["mix"]) if order == "B" else d.assign_streams(feats["streams"])
                        ders.append(der(ref, hyp.tuples())["der"])
                        spk_err.append(abs(len(hyp.speakers) - n))
                    rows.append({"order": order, "clustering": method, **{f"ck_{k}": v for k, v in ck.items()},
                                 "merge_sim": merge, "min_cluster_frac": min_frac, "overlap_aware": ovl,
                                 "der": float(np.mean(ders)), "spk_count_err": float(np.mean(spk_err))})
    df = pd.DataFrame(rows)
    out = PROJECT / "results" / f"tuning_diarization_{args.split}.csv"
    df.to_csv(out, index=False)

    best = {}
    for (order, method), g in df.groupby(["order", "clustering"]):
        b = g.sort_values(["der", "spk_count_err"]).iloc[0]
        ck = {k[3:]: (None if pd.isna(v) else (int(v) if k == "ck_pca_dims" else float(v)))
              for k, v in b.items() if k.startswith("ck_") and not pd.isna(v)}
        default = g[(g.merge_sim.isna()) & (g.min_cluster_frac == 0.0) & (g.overlap_aware)]
        best[f"{order}-{method}"] = {
            "clustering": method, "cluster_kwargs": ck,
            "merge_sim": None if pd.isna(b.merge_sim) else float(b.merge_sim),
            "min_cluster_frac": float(b.min_cluster_frac), "overlap_aware": bool(b.overlap_aware),
            "dev_der": round(float(b.der), 4), "dev_spk_count_err": round(float(b.spk_count_err), 3),
            "dev_der_without_cleanup": round(float(default.der.min()), 4) if len(default) else None,
        }
        print(f"Order {order} {method:9s}: DER {b.der:.3f} (no clean-up: {best[f'{order}-{method}']['dev_der_without_cleanup']}) "
              f"| {best[f'{order}-{method}']}")
    (PROJECT / "results" / "tuned_params.json").write_text(json.dumps(best, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
