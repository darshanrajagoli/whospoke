"""Build the simulated conversation sets (dev + test) with full ground truth.

Paired design: each *group* of 2–3 real speakers is rendered under every condition
(3 overlap levels × 3 noise conditions), so differences between conditions are caused by
the condition — not by who happened to be talking.

    python scripts/build_dataset.py                 # default sizes
    python scripts/build_dataset.py --test-groups 4 --dev-groups 2   # quicker

Output: <data>/synth/<split>/<conversation_id>/{mixture.wav, sources/*.wav, reference.rttm, reference.json}
        <data>/synth/<split>/index.csv      (<data> = whospoke.paths.DATA)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whospoke.corpora import conversation_sides, fetch, read_indicvoices, recording_utterances  # noqa: E402
from whospoke.noise import NoiseBank  # noqa: E402
from whospoke.paths import DATA, SYNTH, conversation  # noqa: E402
from whospoke.synth import SynthConfig, simulate  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
OUT = SYNTH

OVERLAP = {"ovl-none": (0.0, (0.5, 2.5)), "ovl-low": (0.3, (0.5, 2.0)), "ovl-high": (0.7, (1.0, 3.5))}
NOISE = {"clean": (None, None), "village10": ("village", 10.0), "market5": ("market", 5.0)}


def pick_groups(sides: pd.DataFrame, n_groups: int, rng: np.random.Generator) -> list[list[str]]:
    """Alternate 2- and 3-speaker groups; every recording (and speaker) used at most once."""
    order = sides.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    groups, used_spk, i = [], set(), 0
    for g in range(n_groups):
        k = 2 if g % 2 == 0 else 3
        grp = []
        while len(grp) < k and i < len(order):
            r = order.iloc[i]
            i += 1
            if r.speaker_id not in used_spk:
                grp.append(r.recording)
                used_spk.add(r.speaker_id)
        if len(grp) < k:
            raise RuntimeError("not enough distinct speakers for the requested number of groups")
        groups.append(grp)
    return groups


def build(split: str, df: pd.DataFrame, sides: pd.DataFrame, n_groups: int, target_s: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = pick_groups(sides, n_groups, rng)
    rows = []
    for gi, recs in enumerate(groups):
        spk_utts = {}
        for j, rec in enumerate(recs):
            spk_utts[f"S{j + 1}"] = recording_utterances(df, rec)
        for ovl_name, (ovl_p, ovl_s) in OVERLAP.items():
            for noise_name, (scape, snr) in NOISE.items():
                cid = f"{split}_g{gi:02d}_{len(recs)}spk_{ovl_name}_{noise_name}"
                cseed = int(rng.integers(1 << 31))
                cfg = SynthConfig(n_speakers=len(recs), target_s=target_s, overlap_prob=ovl_p, overlap_s=ovl_s,
                                  snr_db=snr, seed=cseed)
                noise = None
                if scape:
                    noise = NoiseBank(scape, split).sample(int((target_s + 60) * 16000), np.random.default_rng(cseed))
                conv = simulate(cid, spk_utts, cfg, noise)
                conv.extra = {"group": gi, "overlap_level": ovl_name, "noise": noise_name,
                              "recordings": dict(zip(spk_utts, recs)),
                              "speaker_ids": {s: spk_utts[s][0].speaker for s in spk_utts}}
                out = conv.save(OUT / split)
                rows.append({"id": cid, "path": out.relative_to(DATA).as_posix(), "group": gi, "n_speakers": len(recs),
                             "overlap_level": ovl_name, "noise": noise_name, "duration_s": round(conv.duration, 2),
                             "overlap_ratio": conv.overlap_ratio()})
                print(f"  {cid}: {conv.duration:5.1f}s  overlap={conv.overlap_ratio():.1%}", flush=True)
    index = pd.DataFrame(rows)
    index.to_csv(OUT / split / "index.csv", index=False)
    return index


def refresh_references() -> None:
    """Recompute the exact speech-activity reference from the saved clean tracks (no audio is regenerated)."""
    import json

    from whospoke.audio import load
    from whospoke.synth import Segment, overlap_stats, reference_activity, write_rttm

    for split in ("test", "dev"):
        idx_path = OUT / split / "index.csv"
        if not idx_path.exists():
            continue
        index = pd.read_csv(idx_path)
        ratios = []
        for r in index.itertuples():
            d = conversation(r.path)
            meta = json.loads((d / "reference.json").read_text(encoding="utf-8"))
            segs = [Segment(**s) for s in meta["segments"]]
            sources = {s: load(d / "sources" / f"{s}.wav") for s in meta["speakers"]}
            act = reference_activity(sources, segs)
            write_rttm(d / "reference.rttm", meta["id"], act)
            meta.update(overlap_stats(act, meta["duration_s"]))
            meta["activity"] = [[s.start, s.end, s.speaker] for s in act]
            (d / "reference.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            ratios.append(meta["overlap_ratio"])
        index["overlap_ratio"] = ratios
        index.to_csv(idx_path, index=False)
        print(f"{split}: refreshed {len(index)} | mean overlap by level: "
              f"{index.groupby('overlap_level').overlap_ratio.mean().round(3).to_dict()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-groups", type=int, default=8)
    ap.add_argument("--dev-groups", type=int, default=4)
    ap.add_argument("--target-s", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=407)
    ap.add_argument("--refresh-references", action="store_true",
                    help="only recompute reference activity/RTTM/overlap for already-built conversations")
    args = ap.parse_args()
    if args.refresh_references:
        refresh_references()
        return

    valid = fetch("ai4bharat/IndicVoices", "hindi/valid-00000-of-00001.parquet", "indicvoices")
    train0 = fetch("ai4bharat/IndicVoices", "hindi/train-00000-of-00082.parquet", "indicvoices")

    test_df = read_indicvoices(valid, with_audio=True)
    test_sides = conversation_sides(test_df)
    dev_df = read_indicvoices(train0, with_audio=True)
    # Dev speakers must never appear in the test set (IndicVoices' valid split shares speakers with train).
    dev_sides = conversation_sides(dev_df)
    dev_sides = dev_sides[~dev_sides.speaker_id.isin(set(test_df.speaker_id))]
    print(f"test pool: {len(test_sides)} sides | dev pool (speaker-disjoint): {len(dev_sides)} sides")

    print("building test set ...")
    t = build("test", test_df, test_sides, args.test_groups, args.target_s, args.seed)
    print("building dev set ...")
    d = build("dev", dev_df, dev_sides, args.dev_groups, args.target_s, args.seed + 1)
    for name, idx in [("test", t), ("dev", d)]:
        print(f"{name}: {len(idx)} conversations, {idx.duration_s.sum() / 60:.1f} min, "
              f"mean overlap by level: {idx.groupby('overlap_level').overlap_ratio.mean().round(3).to_dict()}")


if __name__ == "__main__":
    main()
