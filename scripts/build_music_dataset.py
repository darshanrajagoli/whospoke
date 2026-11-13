"""Build the music sets (voice + mridangam duo): dev and test, with full ground truth.

Same paired design as the speech set (build_dataset.py): each *group* (one sung recording + its mridangam) is rendered
in every condition, 3 overlap levels x 3 noise conditions, so differences between conditions are caused by the
condition. Dev and test never share a singer, a recording, a stroke take (odd vs even takes) or a noise recording.

    python scripts/build_music_dataset.py                       # 4 dev groups (36 pieces), 8 test groups (72)

Needs Data/raw/carnatic_varnam (Zenodo 7726167) and Data/raw/mridangam_stroke (Zenodo 1265188).
Output: <data>/synth_music/<split>/<piece_id>/{mixture.wav, sources/S1.wav (voice), sources/S2.wav (mridangam),
        reference.rttm, reference.json} and <data>/synth_music/<split>/index.csv (same columns as the speech set).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whospoke.noise import NoiseBank  # noqa: E402
from whospoke.paths import DATA  # noqa: E402
from whospoke.synth_music import DEV_SINGERS, MusicConfig, recordings, simulate_duo  # noqa: E402

OUT = DATA / "synth_music"
# accompany_prob, partial_prob: how often (and how fully) the drum plays along with a sung phrase
OVERLAP = {"ovl-none": (0.0, 0.0), "ovl-low": (0.35, 0.6), "ovl-high": (0.85, 0.15)}
NOISE = {"clean": (None, None), "village10": ("village", 10.0), "market5": ("market", 5.0)}


def pick(recs, n, rng):
    """``n`` recordings, spreading over ragas and singers."""
    order = [recs[i] for i in rng.permutation(len(recs))]
    chosen, ragas, singers = [], set(), {}
    for rec in order:                         # first pass: a new raga each time, singers balanced
        if len(chosen) < n and rec.raga not in ragas and singers.get(rec.singer, 0) <= len(chosen) // 2:
            chosen.append(rec)
            ragas.add(rec.raga)
            singers[rec.singer] = singers.get(rec.singer, 0) + 1
    chosen += [r for r in order if r not in chosen][: n - len(chosen)]
    return chosen


def build(split: str, recs, n_groups: int, target_s: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for gi, rec in enumerate(pick(recs, n_groups, rng)):
        for ovl_name, (acc, part) in OVERLAP.items():
            for noise_name, (scape, snr) in NOISE.items():
                pid = f"{split}_m{gi:02d}_{ovl_name}_{noise_name}"
                cseed = int(rng.integers(1 << 31))
                cfg = MusicConfig(accompany_prob=acc, partial_prob=part, target_s=target_s, snr_db=snr, seed=cseed)
                noise = None
                if scape:
                    noise = NoiseBank(scape, split).sample(int((target_s + 60) * 16000), np.random.default_rng(cseed))
                piece = simulate_duo(pid, rec, cfg, split, noise)
                piece.extra.update({"group": gi, "overlap_level": ovl_name, "noise": noise_name})
                out = piece.save(OUT / split)
                rows.append({"id": pid, "path": out.relative_to(DATA).as_posix(), "group": gi, "n_speakers": 2,
                             "overlap_level": ovl_name, "noise": noise_name, "duration_s": round(piece.duration, 2),
                             "overlap_ratio": piece.overlap_ratio(), "recording": rec.id,
                             "mridangam_tonic": piece.extra["mridangam_tonic"]})
                print(f"  {pid}: {piece.duration:5.1f}s  overlap={piece.overlap_ratio():.1%}  ({rec.id})", flush=True)
    index = pd.DataFrame(rows)
    (OUT / split).mkdir(parents=True, exist_ok=True)
    index.to_csv(OUT / split / "index.csv", index=False)
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev-groups", type=int, default=4)
    ap.add_argument("--test-groups", type=int, default=8)
    ap.add_argument("--target-s", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=407)
    args = ap.parse_args()
    recs = recordings()
    dev = [r for r in recs if r.singer in DEV_SINGERS]
    test = [r for r in recs if r.singer not in DEV_SINGERS]
    print(f"{len(recs)} annotated recordings: {len(dev)} dev ({', '.join(DEV_SINGERS)}), {len(test)} test")
    for split, pool, n, s in (("dev", dev, args.dev_groups, args.seed + 1), ("test", test, args.test_groups, args.seed)):
        print(f"building {split} ...")
        idx = build(split, pool, n, args.target_s, s)
        print(f"{split}: {len(idx)} pieces, {idx.duration_s.sum() / 60:.1f} min, mean overlap by level: "
              f"{idx.groupby('overlap_level').overlap_ratio.mean().round(3).to_dict()}")


if __name__ == "__main__":
    main()
