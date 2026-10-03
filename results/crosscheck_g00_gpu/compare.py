"""Compare the GPU re-run of test group 0 (this folder) with the CPU reports scored in results/.

Run from the repository root: python results/crosscheck_g00_gpu/compare.py
"""
import json
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]

cpu = pd.read_csv(ROOT / "results/eval_postprocess.csv")
gpu = pd.read_csv(HERE / "eval_postprocess.csv")
cpu = cpu[cpu.group == 0]
gpu = gpu[gpu.group == 0]
m = cpu.merge(gpu, on=["id", "system"], suffixes=("_cpu", "_gpu"))
print("pairs", len(m))

# identical reports? (provenance and diagnostics hold times and timestamps, so they are left out)
cdir = ROOT / "results/eval_postprocess"
gdir = HERE / "reports"
same_text = same_report = same_sum = same_lines = n_lines = 0
for _, r in m.iterrows():
    f = f"{r.id}__{r.system}.json"
    a = json.loads((cdir / f).read_text(encoding="utf-8"))
    b = json.loads((gdir / f).read_text(encoding="utf-8"))
    la = [l["text"] for l in a["cleaned_dialogue"]]
    lb = [l["text"] for l in b["cleaned_dialogue"]]
    same_text += la == lb
    same_lines += sum(x == y for x, y in zip(la, lb)); n_lines += len(la)
    same_sum += a["executive_summary"] == b["executive_summary"]
    for d in (a, b):
        d.pop("provenance"); d.pop("diagnostics")
    same_report += json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(b, sort_keys=True, ensure_ascii=False)
print("identical repaired text", same_text, "identical summary", same_sum, "identical report", same_report,
      f"lines identical {same_lines}/{n_lines}")

for s, g in m.groupby("system", sort=False):
    d = lambda sfx: 100 * (g[f"cp_errors_clean_{sfx}"].sum() - g[f"cp_errors_raw_{sfx}"].sum()) / g["ref_words_cpu"].sum()
    kt = lambda sfx: 100 * g[f"keywords_true_{sfx}"].mean()
    print(f"{s:14s} raw cpWER cpu {100*g.cp_errors_raw_cpu.sum()/g.ref_words_cpu.sum():5.1f} gpu {100*g.cp_errors_raw_gpu.sum()/g.ref_words_gpu.sum():5.1f} | "
          f"delta cpu {d('cpu'):+.2f} gpu {d('gpu'):+.2f} | kw-true cpu {kt('cpu'):.0f}% gpu {kt('gpu'):.0f}% | "
          f"sumF1 cpu {g.summary_f1_cpu.mean():.2f} gpu {g.summary_f1_gpu.mean():.2f} | "
          f"better/worse cpu {(g.cp_errors_clean_cpu<g.cp_errors_raw_cpu).sum()}/{(g.cp_errors_clean_cpu>g.cp_errors_raw_cpu).sum()} "
          f"gpu {(g.cp_errors_clean_gpu<g.cp_errors_raw_gpu).sum()}/{(g.cp_errors_clean_gpu>g.cp_errors_raw_gpu).sum()} | "
          f"actions cpu {g.actions_cpu.sum()} gpu {g.actions_gpu.sum()} | reverted cpu {g.lines_reverted_cpu.sum()} gpu {g.lines_reverted_gpu.sum()}")
tot = lambda sfx: 100 * (m[f"cp_errors_clean_{sfx}"].sum() - m[f"cp_errors_raw_{sfx}"].sum()) / m["ref_words_cpu"].sum()
print(f"all: delta cpu {tot('cpu'):+.2f} gpu {tot('gpu'):+.2f}")
per = (m.cp_errors_clean_gpu - m.cp_errors_raw_gpu) - (m.cp_errors_clean_cpu - m.cp_errors_raw_cpu)
print("per-report error-count difference gpu-cpu: mean", per.mean(), "min", per.min(), "max", per.max())
