"""Export playable separated audio from the music test set, for the demo.

For each showcase condition it takes the test piece with the MEDIAN Demucs whole-piece score (a typical example, not
the best one) and writes, at 16 kHz:

    00_mixture.wav            what the microphone hears
    01_true_voice.wav         the singer alone (reference)      02_true_mridangam.wav   the drum alone (reference)
    demucs_voice.wav / demucs_mridangam.wav     hpss_voice.wav / hpss_mridangam.wav
    convtasnet_out0.wav / convtasnet_out1.wav   (a speech separator: its outputs have no fixed meaning)

plus README.md with each output's SI-SDR against the reference it is meant to match, and the attribution the datasets'
licences require. Output: results/demo_music/<piece>/ (WAVs are git-ignored; README.md files can be committed).

    python scripts/export_music_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_music_diarization import streams_for  # noqa: E402
from whospoke.audio import load, save  # noqa: E402
from whospoke.metrics import pit_si_sdr, si_sdr  # noqa: E402
from whospoke.paths import DATA  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results" / "demo_music"
SHOWCASE = [("ovl-high", "clean"), ("ovl-high", "market5"), ("ovl-low", "village10")]
ATTRIBUTION = """Audio sources (licences require attribution):
- Singing: Carnatic Varnam dataset (Koduri, Ishwar, Serrà & Serra, 2014, "Intonation analysis of ragas in Carnatic
  music", JNMR), recordings by gopalkoduri on Freesound (pack 14136), CC BY 3.0. Excerpts were cut and mixed.
- Mridangam: Mridangam Stroke dataset (Anantapadmanabhan, Bellur & Murthy, ICASSP 2013), CompMusic, CC BY-NC 4.0.
  Single strokes were arranged into patterns and mixed. Non-commercial use only.
- Noise (village/market pieces): DEMAND (CC BY 4.0) and ESC-50 (CC BY-NC 3.0).
These mixtures were made by the WhoSpoke project for evaluation; they are not real performances."""


def main() -> None:
    res = pd.read_csv(PROJECT / "results" / "separation_test_music.csv")
    whole = res[(res.view == "whole-recording") & (res.model == "demucs")]
    index = pd.read_csv(DATA / "synth_music" / "test" / "index.csv").set_index("id")
    seps: dict = {}
    summary = []
    for ovl, noise in SHOWCASE:
        cand = whole[(whole.overlap_level == ovl) & (whole.noise == noise)].sort_values("si_sdri")
        pid = cand.iloc[len(cand) // 2].id                         # the median piece of this condition
        path = DATA / index.loc[pid, "path"]
        meta = json.loads((path / "reference.json").read_text(encoding="utf-8"))
        mix = load(path / "mixture.wav")
        voice, drum = load(path / "sources" / "S1.wav"), load(path / "sources" / "S2.wav")
        out = OUT / pid
        out.mkdir(parents=True, exist_ok=True)
        save(out / "00_mixture.wav", mix)
        save(out / "01_true_voice.wav", voice)
        save(out / "02_true_mridangam.wav", drum)
        rows = [("mixture (no separation)", si_sdr(mix, voice), si_sdr(mix, drum))]
        for name in ("demucs", "hpss", "convtasnet"):
            st = streams_for(pid, mix, name, seps)
            peak = float(np.max(np.abs(st))) or 1.0
            st = st * min(1.0, 0.9 / peak)                          # one common gain: no clipping, balance kept
            if name == "convtasnet":                                # outputs have no fixed meaning: match them
                _, perm = pit_si_sdr(list(st), [voice, drum])
                save(out / "convtasnet_out0.wav", st[0])
                save(out / "convtasnet_out1.wav", st[1])
                rows.append((f"convtasnet (out{perm[0]} = voice, out{perm[1]} = drum)",
                             si_sdr(st[perm[0]], voice), si_sdr(st[perm[1]], drum)))
            else:
                save(out / f"{name}_voice.wav", st[0])
                save(out / f"{name}_mridangam.wav", st[1])
                rows.append((name, si_sdr(st[0], voice), si_sdr(st[1], drum)))
        table = "\n".join(f"| {n} | {v:+.1f} dB | {d:+.1f} dB |" for n, v, d in rows)
        (out / "README.md").write_text(
            f"# {pid}\n\n{meta['recording']} (raga {meta['raga']}, singer {meta['singer']}), mridangam tuned to "
            f"{meta['mridangam_tonic']}; overlap level {ovl} ({meta['overlap_ratio']:.0%} of playing time both play), "
            f"noise {noise}, {meta['duration_s']:.0f} s. Picked as the **median** Demucs piece of this condition on the "
            f"test set (a typical example, not the best).\n\n"
            f"SI-SDR of each output against the source it should match (higher is better):\n\n"
            f"| audio | vs true voice | vs true mridangam |\n|---|---|---|\n{table}\n\n{ATTRIBUTION}\n",
            encoding="utf-8")
        summary.append(f"{pid}: Demucs voice {rows[1][1]:+.1f} dB, drum {rows[1][2]:+.1f} dB")
        print(summary[-1], flush=True)
    (OUT / "README.md").write_text(
        "# Music demo: separated audio\n\nOne folder per condition; each has the mixture, the true voice and drum, "
        "and every separator's outputs, with scores in its README.\n\n" + "\n".join(f"- {s}" for s in summary)
        + f"\n\n{ATTRIBUTION}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
