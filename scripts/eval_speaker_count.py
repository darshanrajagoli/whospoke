"""Stage 2 when the number of people is known: does telling the diarizer the speaker count help?

    python scripts/eval_speaker_count.py

Runs our tuned Order-B diarizers (spectral, GMM) and pyannote 3.1 on the 72 test conversations three ways:

* ``auto``   the speaker count is found by the clustering (the published system: DER 20.3 %),
* ``given``  the true count is passed in (``python -m whospoke run --speakers N``),
* ``hybrid`` the automatic result is kept when it found N speakers, else the clustering is redone with N
             (what demo/live.py does when it is told how many people are talking).

Same features (results/cache/features, built by scripts/tune_diarization.py or evaluate.py) and the same DER scoring
(0.25 s collar, overlap scored) as scripts/evaluate.py. pyannote is run on the audio (GPU, ~8 min; --no-pyannote
skips it). Writes results/speaker_count_test.{csv,json}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tune_diarization import cached_features, reference  # noqa: E402
from whospoke.__main__ import tuned_diarizer  # noqa: E402
from whospoke.audio import load  # noqa: E402
from whospoke.diarization import Diarizer, PyannoteDiarizer  # noqa: E402
from whospoke.metrics import der  # noqa: E402
from whospoke.paths import RESULTS, SYNTH, conversation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pyannote", action="store_true", help="only our clusterings (no GPU needed)")
    args = ap.parse_args()
    index = pd.read_csv(SYNTH / "test" / "index.csv")
    base, sep_cache, rows = Diarizer(), {}, []
    systems = {name: tuned_diarizer("B", name) for name in ("spectral", "gmm")}
    pya = None if args.no_pyannote else PyannoteDiarizer()
    for r in index.itertuples():
        path = conversation(r.path)
        feats = cached_features(r.id, path, "convtasnet", base, sep_cache)["mix"]
        ref_turns, n_true = reference(path)
        runs = {name: (d.assign(feats), d.assign(feats, n_speakers=n_true)) for name, d in systems.items()}
        if pya is not None:
            wav = load(path / "mixture.wav")
            runs["pyannote"] = (pya(wav), pya(wav, n_true))
        for name, (auto, given) in runs.items():
            for mode, hyp in (("auto", auto), ("given", given),
                              ("hybrid", auto if len(auto.speakers) == n_true else given)):
                score = der(ref_turns, hyp.tuples())
                rows.append({"id": r.id, "clustering": name, "mode": mode, "n_speakers": n_true,
                             "noise": r.noise, "overlap_level": r.overlap_level, "n_found": len(hyp.speakers),
                             **{k: score[k] for k in ("der", "missed", "false_alarm", "confusion")}})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "speaker_count_test.csv", index=False)

    rng = np.random.default_rng(0)
    summary = {}
    for (name, mode), x in df.groupby(["clustering", "mode"], sort=False):
        two = x[x.n_speakers == 2]
        summary[f"{name}-{mode}"] = {
            "der": round(x.der.mean(), 4), "confusion": round(x.confusion.mean(), 4),
            "der_2spk": round(two.der.mean(), 4), "exact_count": round((x.n_found == x.n_speakers).mean(), 4),
            "found_one_of_2spk": int((two.n_found == 1).sum()), "n": len(x)}
    for name in runs:
        a = df[(df.clustering == name) & (df["mode"] == "auto")].set_index("id").der
        for mode in ("given", "hybrid"):
            diff = (df[(df.clustering == name) & (df["mode"] == mode)].set_index("id").der - a).to_numpy()
            boot = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
            summary[f"{name}-{mode}"]["delta_vs_auto"] = round(diff.mean(), 4)
            summary[f"{name}-{mode}"]["delta_ci95"] = [round(np.percentile(boot, 2.5), 4), round(np.percentile(boot, 97.5), 4)]
    (RESULTS / "speaker_count_test.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for k, v in summary.items():
        delta = f"  {100 * v['delta_vs_auto']:+.1f} vs auto (95 % CI {100 * v['delta_ci95'][0]:+.1f} to " \
                f"{100 * v['delta_ci95'][1]:+.1f})" if "delta_vs_auto" in v else ""
        print(f"{k:16s} DER {100 * v['der']:5.1f} %  2-speaker DER {100 * v['der_2spk']:5.1f} %  "
              f"count exact {100 * v['exact_count']:3.0f} %  2-speaker said one: {v['found_one_of_2spk']:2d}/36{delta}")


if __name__ == "__main__":
    main()
