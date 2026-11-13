"""Who *plays* when, on the voice + mridangam duo (build_music_dataset.py). Scored with DER, like the speech set.

Systems (Order B = on the mixture; Order A = separate first):

  B-voice            the speech diarizer exactly as tuned for speech (pyannote speech detector + WeSpeaker voices)
  B-timbre           same clustering, but "anything sounding" activity and a sound-colour (timbre) fingerprint
  B-timbre-n2        B-timbre told that there are two players
  A-<sep>-timbre     separate with <sep> (hpss | demucs), then cluster the two tracks' timbre fingerprints jointly
  A-<sep>-direct     separate with <sep>; a music separator's outputs have a fixed meaning (0 = melody, 1 = drum), so
                     each output's activity *is* that player's activity: no clustering at all. One setting (how far
                     below the loudest frame still counts as playing) is chosen on DEV and then frozen for test.

    python scripts/eval_music_diarization.py --split dev            # also prints the dev choice of --direct-db
    python scripts/eval_music_diarization.py --split test --direct-db hpss=<dev choice> demucs=<dev choice>

Output: results/music_diarization_<split>.csv (one row per piece x system) and a summary.
Separated tracks are cached in results/cache/music_streams/ (git-ignored).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import diarizer_from  # noqa: E402
from whospoke.audio import load  # noqa: E402
from whospoke.diarization import Turn, energy_regions, frame_db  # noqa: E402
from whospoke.metrics import der  # noqa: E402
from whospoke.paths import DATA  # noqa: E402
from whospoke.separation import Separator  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
CACHE = RESULTS / "cache" / "music_streams"
DIRECT_DB = [20.0, 25.0, 30.0, 35.0, 40.0]


def streams_for(pid: str, mix: np.ndarray, name: str, seps: dict) -> np.ndarray:
    f = CACHE / f"{pid}__{name}.npy"
    if f.exists():
        return np.load(f)
    if name not in seps:
        seps[name] = Separator(name)
    y = seps[name].separate(mix).astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, y)
    return y


def direct_turns(streams: np.ndarray, top_db: float) -> list[tuple[float, float, str]]:
    """Output k of a music separator = player k; active where within ``top_db`` of the loudest frame of either."""
    ref = max(float(frame_db(s).max()) for s in streams)
    names = ("melody", "drum")
    return [(s, e, names[k]) for k, st in enumerate(streams) for s, e in energy_regions(st, top_db, ref_db=ref)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--separators", nargs="+", default=["hpss", "demucs"])
    ap.add_argument("--direct-db", nargs="+", default=None,
                    help="frozen dev choices for test, e.g. hpss=30 demucs=25; without it (dev) all are tried")
    args = ap.parse_args()

    tuned = json.loads((RESULTS / "tuned_params.json").read_text(encoding="utf-8"))
    b_voice = diarizer_from(tuned["B-spectral"])
    b_timbre = dataclasses.replace(b_voice, activity="energy", fingerprint="timbre")
    a_timbre = dataclasses.replace(diarizer_from(tuned["A-spectral"]), activity="energy", fingerprint="timbre")
    chosen = {k: float(v) for k, v in (x.split("=") for x in args.direct_db)} if args.direct_db else {}

    index = pd.read_csv(DATA / "synth_music" / args.split / "index.csv")
    seps: dict = {}
    rows = []
    for r in index.itertuples():
        path = DATA / r.path
        meta = json.loads((path / "reference.json").read_text(encoding="utf-8"))
        ref = [tuple(a) for a in meta["activity"]]
        mix = load(path / "mixture.wav")
        hyps: dict[str, list] = {}
        feats = b_voice.analyse(mix, overlaps=False)
        hyps["B-voice"] = b_voice.assign(feats).tuples()
        tf = b_timbre.analyse(mix, overlaps=False)
        hyps["B-timbre"] = b_timbre.assign(tf).tuples()
        hyps["B-timbre-n2"] = b_timbre.assign(tf, n_speakers=2).tuples()
        for name in args.separators:
            st = streams_for(r.id, mix, name, seps)
            hyps[f"A-{name}-timbre"] = a_timbre.diarize_streams(st).tuples()
            for db in ([chosen[name]] if chosen else DIRECT_DB):
                hyps[f"A-{name}-direct@{db:g}"] = direct_turns(st, db)
        for system, hyp in hyps.items():
            d = der(ref, hyp)
            rows.append({"id": r.id, "system": system, "overlap_level": r.overlap_level, "noise": r.noise,
                         "der": d["der"], "missed": d["missed"], "false_alarm": d["false_alarm"],
                         "confusion": d["confusion"], "n_players_found": len({h[2] for h in hyp})})
        print(f"{r.id}: " + "  ".join(f"{s}={x['der']:.2f}" for s, x in
                                       zip(hyps, rows[-len(hyps):]) if "@" not in s), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"music_diarization_{args.split}.csv", index=False)
    summ = df.groupby("system")[["der", "missed", "false_alarm", "confusion"]].mean()
    summ["players_right"] = df.groupby("system").n_players_found.apply(lambda x: (x == 2).mean())
    print(summ.round(3).sort_values("der").to_string())
    print(df.pivot_table(index="system", columns="overlap_level", values="der").round(3).to_string())
    if args.direct_db is None:
        for name in args.separators:
            sub = summ[summ.index.str.startswith(f"A-{name}-direct@")]
            best = sub.der.idxmin()
            print(f"dev choice for A-{name}-direct: --direct-db {best.split('@')[1]} (DER {sub.der.min():.3f})")


if __name__ == "__main__":
    main()
