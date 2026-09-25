"""End-to-end evaluation on the simulated TEST set: every system, every conversation.

Systems (all estimate the number of speakers themselves unless marked oracle):

  oracle-clean   true turns, each speaker's CLEAN voice → ASR           (ASR's own error floor)
  oracle-mix     true turns, noisy overlapped MIXTURE → ASR             (+ noise & overlap)
  oracle-sep     true turns, overlapped turns separated (Order B) → ASR (can separation undo overlap?)
  B-spectral     Order B with our spectral-clustering diarizer          (proposed system)
  B-gmm          Order B with our GMM diarizer
  B-nosep        B-spectral without the separation step                  (ablation)
  B-pyannote     Order B with off-the-shelf pyannote 3.1 diarization    (reference)
  A-spectral     Order A (proposal's order): separate everything first, spectral clustering
  A-gmm          Order A with GMM clustering

Metrics per conversation: DER (+ missed / false alarm / confusion), JER, speaker-count error,
cpWER and cpCER (who-said-what), and the fraction of turns that went through separation.

    python scripts/evaluate.py --split test --asr indicconformer
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tune_diarization import cached_features, reference  # noqa: E402
from whospoke.paths import SYNTH, conversation  # noqa: E402
from whospoke.audio import load  # noqa: E402
from whospoke.diarization import Diarization, Diarizer, PyannoteDiarizer, Turn, merge_turns  # noqa: E402
from whospoke.metrics import cp_error, der, jer  # noqa: E402
from whospoke.pipeline import Pipeline  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"


def diarizer_from(params: dict) -> Diarizer:
    return Diarizer(params["clustering"], merge_sim=params["merge_sim"], min_cluster_frac=params["min_cluster_frac"],
                    overlap_aware=params["overlap_aware"], cluster_kwargs=params["cluster_kwargs"])


def reference_diarization(segments: list[dict]) -> Diarization:
    """Ground-truth turns (consecutive utterances of one speaker merged when < 0.5 s apart)."""
    return Diarization(merge_turns([Turn(s["start"], s["end"], s["speaker"]) for s in segments], gap=0.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--asr", default="indicconformer")
    ap.add_argument("--separator", default="convtasnet")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tuned = json.loads((RESULTS / "tuned_params.json").read_text(encoding="utf-8"))
    diar = {k: diarizer_from(v) for k, v in tuned.items()}          # keys like "B-spectral", "A-gmm"
    pipe = Pipeline("B", separator=args.separator, diarizer=diar["B-spectral"], asr=args.asr)
    pyannote = PyannoteDiarizer()
    base = Diarizer()
    sep_cache = {args.separator: pipe.separator}

    index = pd.read_csv(SYNTH / args.split / "index.csv")
    if args.limit:
        index = index.head(args.limit)
    out_dir = RESULTS / f"eval_{args.split}_{args.asr}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in index.itertuples():
        path = conversation(r.path)
        ref_meta = json.loads((path / "reference.json").read_text(encoding="utf-8"))
        ref_turns, n_true = reference(path)
        ref_text = {}
        for s in sorted(ref_meta["segments"], key=lambda s: s["start"]):
            ref_text.setdefault(s["speaker"], []).append(s["text"])
        ref_text = {k: " ".join(v) for k, v in ref_text.items()}
        mix = load(path / "mixture.wav")
        clean = {s: load(path / "sources" / f"{s}.wav") for s in ref_meta["speakers"]}
        feats = cached_features(r.id, path, args.separator, base, sep_cache)
        streams = pipe.separator.separate(mix)
        oracle = reference_diarization(ref_meta["segments"])

        t = time.perf_counter()
        pya = pyannote(mix)
        pya_time = time.perf_counter() - t
        diarizations = {
            "B-spectral": diar["B-spectral"].assign(feats["mix"]),
            "B-gmm": diar["B-gmm"].assign(feats["mix"]),
            "B-pyannote": pya,
            "A-spectral": diar["A-spectral"].assign_streams(feats["streams"]),
            "A-gmm": diar["A-gmm"].assign_streams(feats["streams"]),
        }
        diarizations["B-nosep"] = diarizations["B-spectral"]
        # Every Order-B system gets the same detected overlap regions (pyannote's detector, run once).
        oracle.info["overlap"] = feats["mix"].overlaps
        pya.info["overlap"] = feats["mix"].overlaps

        units = {
            "oracle-clean": [(tt, clean[tt.speaker], "clean") for tt in oracle.turns],
            "oracle-mix": [(tt, mix, "mixture") for tt in oracle.turns],
            "oracle-sep": pipe._targeted_separation(mix, oracle),
            "B-nosep": [(tt, mix, "mixture") for tt in diarizations["B-nosep"].turns],
        }
        for name in ("B-spectral", "B-gmm", "B-pyannote"):
            units[name] = pipe._targeted_separation(mix, diarizations[name])
        for name in ("A-spectral", "A-gmm"):
            units[name] = [(tt, streams[tt.stream], f"track-{tt.stream}") for tt in diarizations[name].turns]

        conv_out = {}
        for system, u in units.items():
            t = time.perf_counter()
            lines = pipe._transcribe(u)
            asr_time = time.perf_counter() - t
            hyp_text: dict[str, list[str]] = {}
            for ln in sorted(lines, key=lambda l: l.start):
                hyp_text.setdefault(ln.speaker, []).append(ln.text)
            hyp_text = {k: " ".join(v) for k, v in hyp_text.items()}
            cpw = cp_error(ref_text, hyp_text, "word")
            cpc = cp_error(ref_text, hyp_text, "char")
            d = oracle if system.startswith("oracle") else diarizations[system]
            dd = der(ref_turns, d.tuples())
            rows.append({
                "id": r.id, "system": system, "group": r.group, "n_speakers": r.n_speakers,
                "overlap_level": r.overlap_level, "noise": r.noise, "overlap_ratio": r.overlap_ratio, "duration_s": r.duration_s,
                "der": dd["der"], "missed": dd["missed"], "false_alarm": dd["false_alarm"], "confusion": dd["confusion"],
                "der_nocollar": der(ref_turns, d.tuples(), collar=0.0)["der"], "jer": jer(ref_turns, d.tuples()),
                "n_spk_hyp": len(d.speakers), "spk_count_err": len(d.speakers) - n_true,
                "cpwer": cpw["rate"], "cpcer": cpc["rate"], "cp_errors": cpw["errors"], "ref_words": cpw["n_ref"],
                "separated_frac": float(np.mean([x[2] == "separated" for x in u])) if u else 0.0,
                "asr_s": asr_time, "pyannote_s": pya_time if system == "B-pyannote" else np.nan,
            })
            conv_out[system] = [{"speaker": l.speaker, "start": l.start, "end": l.end, "text": l.text,
                                 "hinglish": l.hinglish, "audio": l.audio} for l in sorted(lines, key=lambda l: l.start)]
        (out_dir / f"{r.id}.json").write_text(json.dumps({"reference": ref_meta["segments"], "systems": conv_out},
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
        last = {x["system"]: x for x in rows[-len(units):]}
        print(f"{r.id}: " + "  ".join(f"{s}:{last[s]['cpwer']:.2f}/{last[s]['der']:.2f}" for s in last), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"eval_{args.split}_{args.asr}.csv", index=False)
    pooled = df.groupby("system").apply(lambda g: pd.Series({
        "cpWER": g.cp_errors.sum() / g.ref_words.sum(), "cpCER": g.cpcer.mean(), "DER": g.der.mean(),
        "missed": g.missed.mean(), "false_alarm": g.false_alarm.mean(), "confusion": g.confusion.mean(),
        "spk_count_err_abs": g.spk_count_err.abs().mean()}), include_groups=False)
    print(pooled.round(3).to_string())


if __name__ == "__main__":
    main()
