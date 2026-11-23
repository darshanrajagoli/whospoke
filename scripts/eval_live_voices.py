"""The live demo's "Speaker X alone" buttons: how close is each voice to the true one?

    python scripts/eval_live_voices.py

Runs demo/live.py's own pipeline (Order B, spectral clustering told there are 2 people, IndicConformer) on the 36
two-speaker test conversations and builds each speaker's voice two ways:

* ``turns``  each speaker's own turns, with Conv-TasNet run only on the overlapping stretches (the demo's default,
             Order B's method; every overlap goes to both nearby speakers),
* ``whole``  the whole recording through Conv-TasNet, tracks matched to speakers by voice fingerprint
             (``demo/live.py --voices whole``, the flash-talk version).

Each voice is scored by SI-SDR against the true source it matches best (both pairings tried), over the whole
conversation and over the overlapped stretches only; ``mix`` is the recording itself against each source.
GPU, about 10 min. Writes results/live_voices_test.csv and prints the means.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "demo")]

import live  # noqa: E402
from whospoke.__main__ import tuned_diarizer  # noqa: E402
from whospoke.asr_backends import IndicConformerASR  # noqa: E402
from whospoke.audio import SR, load  # noqa: E402
from whospoke.diarization import embed  # noqa: E402
from whospoke.metrics import si_sdr  # noqa: E402
from whospoke.paths import RESULTS, SYNTH  # noqa: E402
from whospoke.pipeline import Pipeline  # noqa: E402


def score(voices: dict, refs: list[np.ndarray], ov: np.ndarray) -> tuple[float, float]:
    spk = sorted(voices)[:2]
    if len(spk) < 2:
        return np.nan, np.nan
    best = max(((0, 1), (1, 0)), key=lambda p: sum(
        si_sdr(voices[spk[i]][:len(refs[p[i]])], refs[p[i]][:len(voices[spk[i]])]) for i in (0, 1)))
    n = min(len(ov), *(len(r) for r in refs))
    whole = np.mean([si_sdr(voices[spk[i]][:n], refs[best[i]][:n]) for i in (0, 1)])
    m = ov[:n]
    over = np.mean([si_sdr(voices[spk[i]][:n][m], refs[best[i]][:n][m]) for i in (0, 1)]) if m.any() else np.nan
    return whole, over


def main() -> None:
    pipe = Pipeline("B", diarizer=live.known_count(tuned_diarizer("B", "spectral")),
                    asr_model=IndicConformerASR(decoding="rnnt", language="hi"))
    rows = []
    for conv in sorted(p for p in (SYNTH / "test").glob("*2spk*") if p.is_dir()):
        mix = load(conv / "mixture.wav")
        refs = [load(conv / "sources" / f"S{i}.wav") for i in (1, 2)]
        res = pipe.run(mix, n_speakers=2)
        ov = np.zeros(len(mix), bool)
        for a, b in pipe._overlap_regions(res.diarization):
            ov[int(a * SR):int(b * SR)] = True
        row = {"conv": conv.name}
        for name, build in (("whole", live._voices), ("turns", live._turn_voices)):
            row[name], row[name + "_ov"] = score(build(pipe, mix, res, embed), refs, ov)
        n = min(len(mix), *(len(r) for r in refs))
        row["mix"] = np.mean([si_sdr(mix[:n], r[:n]) for r in refs])
        rows.append(row)
        print(conv.name, {k: round(v, 1) for k, v in row.items() if k != "conv"}, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "live_voices_test.csv", index=False)
    print("mean SI-SDR (dB):", d[["turns", "whole", "mix"]].mean().round(1).to_dict(),
          f"| turns better in {int((d.turns > d.whole).sum())} of {len(d)}")


if __name__ == "__main__":
    main()
