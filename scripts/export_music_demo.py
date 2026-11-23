"""Export playable separated audio from the music test set, for the demo.

For each showcase condition it takes the test piece with the MEDIAN Demucs whole-piece score (a typical example, not
the best one) and writes, at 16 kHz:

    00_mixture.wav            what the microphone hears
    01_true_voice.wav         the singer alone (reference)      02_true_mridangam.wav   the drum alone (reference)
    demucs_voice.wav / demucs_mridangam.wav     hpss_voice.wav / hpss_mridangam.wav
    convtasnet_out0.wav / convtasnet_out1.wav   (a speech separator: its outputs have no fixed meaning)

plus README.md with each output's SI-SDR against the reference it is meant to match, and the attribution the datasets'
licences require. Output: results/demo_music/<piece>/ (WAVs are git-ignored; README.md files can be committed).

With ``--page`` it also writes results/demo_music/demo.html: one self-contained page (audio embedded, ~5 MB, not
committed) with players for a 12 s excerpt of the high-overlap clean piece, where both play the most, and spectrograms.

    python scripts/export_music_demo.py [--page]
"""
from __future__ import annotations

import argparse
import base64
import io
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


def write_page(pid: str, out_dir: Path) -> Path:
    """A self-contained HTML page: players and spectrograms for the 12 s where both play the most."""
    import librosa
    import matplotlib
    import soundfile as sf

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    sr = 16000
    meta = json.loads((DATA / "synth_music" / "test" / pid / "reference.json").read_text(encoding="utf-8"))
    names = ["00_mixture", "01_true_voice", "02_true_mridangam", "demucs_voice", "demucs_mridangam", "hpss_voice",
             "hpss_mridangam"]
    wav = {k: sf.read(out_dir / f"{k}.wav", dtype="float32")[0] for k in names}
    act = np.zeros(int(meta["duration_s"] * 100) + 1, int)
    for s, e, _ in meta["activity"]:
        act[int(s * 100): int(e * 100)] += 1
    win = 1200
    start = int(np.argmax(np.convolve((act >= 2).astype(float), np.ones(win), "valid")))
    clip = {k: v[start * sr // 100: (start + win) * sr // 100] for k, v in wav.items()}
    rows = [("Mixture (what the microphone hears)", "00_mixture", None),
            ("Voice — original (true)", "01_true_voice", None),
            ("Voice — separated by Demucs", "demucs_voice", "01_true_voice"),
            ("Voice — separated by HPSS", "hpss_voice", "01_true_voice"),
            ("Mridangam — original (true)", "02_true_mridangam", None),
            ("Mridangam — separated by Demucs", "demucs_mridangam", "02_true_mridangam"),
            ("Mridangam — separated by HPSS", "hpss_mridangam", "02_true_mridangam")]
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    specs = {k: librosa.amplitude_to_db(np.abs(librosa.stft(clip[k], n_fft=1024, hop_length=256)), ref=1.0)
             for _, k, _ in rows}
    vmax = max(s.max() for s in specs.values())
    plt.rcParams.update({"font.size": 9, "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.edgecolor": "#c9c8c3"})
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 11), sharex=True, facecolor="#fcfcfb")
    for ax, (title, k, _) in zip(axes, rows):
        im = ax.imshow(specs[k], origin="lower", aspect="auto", cmap=cmap, vmin=vmax - 70, vmax=vmax,
                       extent=[0, win / 100, 0, sr / 2000])
        ax.set_ylim(0, 4)
        ax.set_title(title, loc="left", fontsize=9, color="#0b0b0b", pad=3)
        ax.set_ylabel("kHz")
    axes[-1].set_xlabel("seconds into the excerpt")
    fig.colorbar(im, ax=axes, shrink=0.35, pad=0.01, label="loudness (dB)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    png = base64.b64encode(buf.getvalue()).decode()

    def audio_b64(x):
        x = x / (float(np.max(np.abs(x))) or 1.0) * 0.9              # each clip to the same peak, for listening
        b = io.BytesIO()
        sf.write(b, x, sr, format="WAV", subtype="PCM_16")
        return base64.b64encode(b.getvalue()).decode()

    cards = []
    for title, k, ref in rows:
        if ref:
            score = f'<span class="score">SI-SDR vs original: {si_sdr(clip[k], clip[ref]):+.1f} dB</span>'
        elif k == "00_mixture":
            score = (f'<span class="score">vs voice {si_sdr(clip[k], clip["01_true_voice"]):+.1f} dB · '
                     f'vs mridangam {si_sdr(clip[k], clip["02_true_mridangam"]):+.1f} dB</span>')
        else:
            score = ""
        cards.append(f'<div class="card"><div class="lbl">{title}</div>{score}'
                     f'<audio controls preload="none" src="data:audio/wav;base64,{audio_b64(clip[k])}"></audio></div>')
    css = (":root { --bg:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3de; }\n"
           "@media (prefers-color-scheme: dark) { :root:not([data-theme=\"light\"]) { --bg:#1a1a19; --card:#232321; "
           "--ink:#f0efec; --ink2:#c3c2b7; --line:#383835; } }\n"
           ":root[data-theme=\"dark\"] { --bg:#1a1a19; --card:#232321; --ink:#f0efec; --ink2:#c3c2b7; --line:#383835; }\n"
           "body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, -apple-system, Segoe UI, "
           "sans-serif; }\nmain { max-width:960px; margin:0 auto; padding:24px 16px 48px; }\n"
           "h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:28px 0 10px; }\n"
           "p { color:var(--ink2); margin:6px 0; }\n"
           ".grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:10px; }\n"
           ".card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; }\n"
           ".lbl { font-weight:600; font-size:14px; } .score { display:block; color:var(--ink2); font-size:13px; "
           "margin:2px 0 8px; }\naudio { width:100%; }\n"
           "figure { margin:0; background:#fcfcfb; border:1px solid var(--line); border-radius:10px; padding:8px; "
           "overflow-x:auto; }\nfigure img { width:100%; min-width:560px; display:block; } figcaption { color:var(--ink2); "
           "font-size:13px; margin-top:6px; }\nsmall { color:var(--ink2); }")
    html = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\n"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Voice and Mridangam Demo</title>\n"
            f"<style>{css}</style></head><body><main>\n<h1>Voice and mridangam: original vs separated</h1>\n"
            f"<p>Test piece <b>{pid}</b>: raga {meta['raga']} sung by {meta['singer'].capitalize()}, mridangam tuned to "
            f"{meta['mridangam_tonic']}, no noise, both playing {meta['overlap_ratio']:.0%} of the time. This 12-second "
            f"excerpt is where they overlap most. It is the <b>median</b> test piece of its condition for Demucs "
            f"(typical, not the best).</p>\n<p>Each clip is set to the same peak loudness so the quieter ones are "
            f"audible. Higher SI-SDR = closer to the original.</p>\n<h2>Listen</h2><div class=\"grid\">{''.join(cards)}"
            f"</div>\n<h2>See it</h2>\n<figure><img alt=\"Spectrograms of the mixture, the original voice and mridangam, "
            f"and each separator's outputs\" src=\"data:image/png;base64,{png}\"><figcaption>Spectrograms (time left to "
            f"right, pitch bottom to top, darker = louder). The voice and drone are horizontal lines; mridangam strokes "
            f"are vertical streaks.</figcaption></figure>\n<p><small>{ATTRIBUTION.replace(chr(10), ' ')}</small></p>\n"
            f"</main></body></html>")
    page = out_dir.parent / "demo.html"
    page.write_text(html, encoding="utf-8")
    return page


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", action="store_true", help="also write results/demo_music/demo.html")
    args = ap.parse_args()
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
    if args.page:
        first = summary[0].split(":")[0]
        print("page:", write_page(first, OUT / first))


if __name__ == "__main__":
    main()
