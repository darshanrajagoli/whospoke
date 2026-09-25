"""Generate (and optionally execute) the project notebooks from code, so they stay reproducible.

    python scripts/build_notebooks.py            # write + execute both notebooks
    python scripts/build_notebooks.py --no-run   # write only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

PROJECT = Path(__file__).resolve().parents[1]
NB = PROJECT / "notebooks"

SETUP = r'''import sys, os, json, time
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
pd.set_option("display.precision", 3)
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")'''

PLOT_STYLE = r'''# Chart style: thin marks, hairline grid, fixed categorical colour order (blue, orange, aqua).
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "grey": "#8a8984", "ink": "#0b0b0b", "ink2": "#52514e"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c9c8c3", "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.labelcolor": C["ink2"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"], "legend.frameon": False})'''


def benchmark() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [
        md("# System benchmark — latency and memory of every pipeline stage\n\n"
           "Deliverable from the proposal: *\"a validation notebook measuring the computational latency and "
           "memory consumption of each module stage.\"*\n\n"
           "We run both pipeline orders on nine test conversations (one per overlap × noise condition) and record, "
           "for each stage: wall-clock time, real-time factor (processing time ÷ audio length — below 1 means faster "
           "than real time), and peak GPU memory. Models are loaded and warmed up first, so the numbers measure "
           "processing, not loading."),
        code(SETUP),
        code(PLOT_STYLE),
        md("## 1 · Load both pipelines (models are loaded once and shared)"),
        code(r'''from whospoke.pipeline import Pipeline, make_asr
from whospoke.__main__ import tuned_diarizer
from whospoke.audio import load, SR

asr = make_asr("indicconformer")
pipes = {
    "Order A (separate → diarize → ASR)": Pipeline("A", diarizer=tuned_diarizer("A", "spectral"), asr_model=asr),
    "Order B (diarize → separate overlaps → ASR)": Pipeline("B", diarizer=tuned_diarizer("B", "spectral"), asr_model=asr),
}
for p in pipes.values():
    p.warmup()
print("ready")'''),
        md("## 2 · Model sizes"),
        code(r'''from whospoke import vad
from whospoke.diarization import _embedding_model, default_device
def n_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6
sep = pipes["Order B (diarize → separate overlaps → ASR)"].separator._model
seg = vad._pipelines(None)[0]._segmentation.model
emb = _embedding_model(default_device()).model_
rows = [("Conv-TasNet (separation)", n_params(sep)), ("pyannote segmentation-3.0 (VAD + overlap)", n_params(seg)),
        ("WeSpeaker ResNet-34 (speaker embeddings)", n_params(emb))]
try:
    rows.append((f"{asr.name} (ASR)", asr.num_parameters() / 1e6))
except Exception:
    pass
pd.DataFrame(rows, columns=["model", "parameters (millions)"])'''),
        md("## 3 · Run on one conversation per condition"),
        code(r'''from whospoke.paths import SYNTH, conversation
idx = pd.read_csv(SYNTH / "test/index.csv")
subset = idx[idx.group.isin([0, 1])].groupby(["overlap_level", "noise"]).head(1)
records = []
for name, p in pipes.items():
    for r in subset.itertuples():
        wav = load(conversation(r.path) / "mixture.wav")
        res = p.run(wav)
        for stage in ("separation", "diarization", "asr"):
            records.append({"pipeline": name, "conversation": r.id, "stage": stage, "audio_s": res.duration_s,
                            "seconds": res.timings[stage], "gpu_peak_mb": res.gpu_peak_mb.get(stage, np.nan)})
        records.append({"pipeline": name, "conversation": r.id, "stage": "total", "audio_s": res.duration_s,
                        "seconds": res.timings["total"], "gpu_peak_mb": max(res.gpu_peak_mb.values())})
bench = pd.DataFrame(records)
bench["rtf"] = bench.seconds / bench.audio_s
bench.to_csv(ROOT / "results/benchmark_stages.csv", index=False)
summary = bench.groupby(["pipeline", "stage"]).agg(seconds=("seconds", "mean"), rtf=("rtf", "mean"),
                                                   gpu_peak_mb=("gpu_peak_mb", "max")).round(3)
summary'''),
        md("## 4 · Where does the time go?"),
        code(r'''stages = ["separation", "diarization", "asr"]
colors = [C["blue"], C["orange"], C["aqua"]]
fig, ax = plt.subplots(figsize=(8, 2.6))
names = list(pipes)
for i, name in enumerate(names):
    left = 0
    for s, col in zip(stages, colors):
        v = bench[(bench.pipeline == name) & (bench.stage == s)].rtf.mean()
        ax.barh(i, v, left=left, color=col, height=0.5, edgecolor="white", linewidth=2, label=s if i == 0 else None)
        left += v
    ax.text(left + 0.002, i, f"{left:.3f}× real time", va="center", color=C["ink2"])
ax.set_yticks(range(len(names)), [n.split(" (")[0] for n in names])
ax.set_xlabel("processing time ÷ audio duration (lower is faster)")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.35))
ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(ROOT / "results/figures/latency_by_stage.png", dpi=200, bbox_inches="tight"); plt.show()'''),
        md("## 5 · Peak GPU memory per stage"),
        code(r'''mem = bench[bench.stage != "total"].groupby(["stage", "pipeline"]).gpu_peak_mb.max().unstack()
mem.round(0)'''),
        md("## 6 · Does the cost grow linearly with audio length?\n\n"
           "We chain test conversations into 1-, 2-, 4- and 8-minute recordings and time Order B."),
        code(r'''long = np.concatenate([load(conversation(p) / "mixture.wav") for p in idx.path[:8]])
p = pipes["Order B (diarize → separate overlaps → ASR)"]
scaling = []
for minutes in (1, 2, 4, 8):
    x = long[: int(minutes * 60 * SR)]
    res = p.run(x)
    scaling.append({"minutes": minutes, "seconds": res.timings["total"], "rtf": res.timings["total"] / (len(x) / SR),
                    "gpu_peak_mb": max(res.gpu_peak_mb.values())})
scaling = pd.DataFrame(scaling); scaling.to_csv(ROOT / "results/benchmark_scaling.csv", index=False)
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(scaling.minutes, scaling.seconds, color=C["blue"], lw=2, marker="o", ms=6)
ax.set_xlabel("audio length (minutes)"); ax.set_ylabel("processing time (s)")
ax.set_title("Order B: processing time vs audio length")
plt.tight_layout(); plt.savefig(ROOT / "results/figures/latency_scaling.png", dpi=200, bbox_inches="tight"); plt.show()
scaling'''),
        md("## Summary\n\nThe tables above are saved to `results/benchmark_stages.csv` and `results/benchmark_scaling.csv`; "
           "the figures to `results/figures/`. The headline numbers are quoted in `docs/RESULTS.md`."),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def walkthrough() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [
        md("# Who spoke what and when — walkthrough (runs locally or on Google Colab)\n\n"
           "This notebook takes one noisy, overlapping Hindi/Hinglish conversation through all three stages and "
           "shows what each stage produces.\n\n"
           "**On Colab:** Runtime → Change runtime type → T4 GPU. Add your Hugging Face token under *Secrets* (🔑) "
           "as `HF_TOKEN` (the account must have accepted the model terms listed in the README). Then run all cells."),
        code(r'''import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    REPO = "https://github.com/darshanrajagoli/whospoke.git"
    if not os.path.exists("whospoke"):
        subprocess.run(["git", "clone", "--depth", "1", REPO], check=True)
    os.chdir("whospoke")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "asteroid==0.7.0"], check=True)'''),
        code(SETUP),
        md("## 1 · A test conversation\n\nIf the simulated data set has not been built on this machine, we build one "
           "small conversation on the fly from IndicVoices (needs the Hugging Face token)."),
        code(r'''from whospoke.audio import load, SR
from IPython.display import Audio, display
from whospoke.paths import SYNTH, conversation
idx_path = SYNTH / "test/index.csv"
if idx_path.exists():
    idx = pd.read_csv(idx_path)
    row = idx[(idx.n_speakers == 3) & (idx.overlap_level == "ovl-high") & (idx.noise == "market5")].iloc[0]
    conv_dir = conversation(row.path)
else:
    subprocess.run([sys.executable, str(ROOT / "scripts/build_dataset.py"), "--test-groups", "2", "--dev-groups", "1"], check=True)
    idx = pd.read_csv(idx_path); row = idx[idx.overlap_level == "ovl-high"].iloc[-1]; conv_dir = conversation(row.path)
ref = json.loads((conv_dir / "reference.json").read_text(encoding="utf-8"))
mix = load(conv_dir / "mixture.wav")
print(ref["id"], f"| {ref['duration_s']:.0f} s | speakers: {ref['speakers']} | overlap: {ref['overlap_ratio']:.0%}")
display(Audio(mix, rate=SR))'''),
        md("## 2 · The full pipeline (Order B)"),
        code(r'''from whospoke.pipeline import Pipeline
from whospoke.__main__ import tuned_diarizer
pipe = Pipeline("B", diarizer=tuned_diarizer("B", "spectral"))
result = pipe.run(mix)
print(result.transcript(hinglish=True))
print("\ntimings (s):", result.timings)'''),
        md("## 3 · Stage 2 output: the speaker timeline vs the truth"),
        code(r'''fig, axes = plt.subplots(2, 1, figsize=(10, 3), sharex=True)
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
def draw(ax, turns, title):
    names = sorted({t[2] for t in turns})
    for (s, e, spk) in turns:
        ax.barh(names.index(spk), e - s, left=s, color=cols[names.index(spk) % 5], height=0.6)
    ax.set_yticks(range(len(names)), names); ax.set_title(title, loc="left"); ax.grid(axis="y", visible=False)
draw(axes[0], [(s["start"], s["end"], s["speaker"]) for s in ref["segments"]], "reference (who really spoke)")
draw(axes[1], result.diarization.tuples(), "pipeline output")
axes[1].set_xlabel("time (s)"); plt.tight_layout(); plt.show()
from whospoke.metrics import der
truth = [tuple(a) for a in ref["activity"]]   # exact speech activity (start, end, speaker), as in the evaluation
print("DER:", round(der(truth, result.diarization.tuples())["der"], 3))'''),
        md("## 4 · Stage 1 output: listen to a separated overlap"),
        code(r'''ov = pipe._overlap_regions(result.diarization)   # where the overlap detector heard two voices
print(f"{len(ov)} overlap regions, {sum(e - s for s, e in ov):.1f} s in total; "
      f"{sum(l.audio == 'separated' for l in result.lines)} of {len(result.lines)} turns used separated audio")
if ov:
    s, e = max(ov, key=lambda x: x[1] - x[0])
    a, b = int(max(0, s - 0.5) * SR), int((e + 0.5) * SR)
    est = pipe.separator.separate(mix[a:b])
    print(f"overlap {s:.1f}–{e:.1f} s: mixture, then the two separated voices")
    display(Audio(mix[a:b], rate=SR)); display(Audio(est[0], rate=SR)); display(Audio(est[1], rate=SR))'''),
        md("## 5 · Stage 3 output: who-said-what score"),
        code(r'''from whospoke.metrics import cp_error
ref_text = {}
for s in sorted(ref["segments"], key=lambda s: s["start"]):
    ref_text.setdefault(s["speaker"], []).append(s["text"])
ref_text = {k: " ".join(v) for k, v in ref_text.items()}
print("cpWER (words wrong or given to the wrong speaker):", round(cp_error(ref_text, result.text_by_speaker())["rate"], 3))
result.save(ROOT / "results/runs/walkthrough"); print("saved to results/runs/walkthrough")'''),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--kernel", default="whospoke",
                    help="Jupyter kernel to execute with (register the venv once: python -m ipykernel install --user --name whospoke)")
    args = ap.parse_args()
    NB.mkdir(exist_ok=True)
    (PROJECT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    books = {"01_benchmark.ipynb": benchmark(), "02_walkthrough.ipynb": walkthrough()}
    for name, nb in books.items():
        path = NB / name
        nbf.write(nb, path)
        if not args.no_run:
            from nbconvert.preprocessors import ExecutePreprocessor

            ExecutePreprocessor(timeout=3600, kernel_name=args.kernel).preprocess(nb, {"metadata": {"path": str(NB)}})
            nbf.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
