# Cross-check: test group 0 on the GPU instead of the CPU

Groups 0–1 were scored with reports made on a 4-core cloud CPU, and groups 2–6 with reports made on the laptop GPU
(D33). Temperature is 0, so on one machine the same transcript always gives the same report. But a CPU and a GPU
build of llama.cpp add up the numbers in a different order, so they can differ in the last bits. Greedy decoding then
picks a different word where two words are almost tied, and the reply goes its own way from there. To see whether the
choice of machine changes any conclusion, the 36 group-0 reports (9 conversations × 4 inputs) were made again on the
laptop GPU (same model file, same frozen prompts, `--gpu-layers 99 --parallel 1`) in a separate copy of the repository,
so the scored results were not touched.

Files: `eval_postprocess.csv` (the same columns as `results/eval_postprocess.csv`), `reports/` (the 36 reports),
`eval_postprocess_summary.json` (this run: 1,867 s), and `compare.py`, which prints the comparison below
(`python results/crosscheck_g00_gpu/compare.py`).

| Stage-4 input | repair Δ cpWER, CPU → GPU | helped / hurt, CPU → GPU | keywords really said, CPU → GPU | action items |
|---|---|---|---|---|
| true transcript | +2.2 → +2.0 points | 0/9 → 0/9 | 96 % → 97 % | 0 and 0 |
| ASR on each clean voice | +1.0 → +1.5 | 0/7 → 0/7 | 88 % → 70 % | 0 and 0 |
| ASR on the noisy mixture | +1.0 → +1.2 | 0/7 → 1/6 | 61 % → 59 % | 0 and 0 |
| full pipeline (Order B) | +0.7 → +0.6 | 0/7 → 1/7 | 74 % → 75 % | 0 and 0 |

(Pooled over the 9 conversations of group 0; Δ is cpWER after the repair minus before.)

**What it shows.**
- **The repaired transcript barely depends on the machine.** 381 of 406 repaired lines (94 %) are identical, and 19 of
  the 36 reports have exactly the same repaired text.
- **The free text does.** The executive summary is word-for-word the same in only 2 of 36 reports. Every word the
  model writes after the first differing one depends on it, so this is expected.
- **The repair hurts on both.** Pooled over the 36 reports, it adds +1.2 cpWER points on the CPU and +1.4 on the GPU,
  never less than +0.6 for any input. The GPU run's repair helped in 2 of 36 reports instead of 0 (so the "helped in 1
  of 252" in RESULTS depends on the machine too; either way it is rare). No action items on either.
- **Keywords really said** are within 2 points on three inputs. The exception is the clean-voice input, 88 % vs 70 %
  on 9 conversations, the kind of swing a few keywords make on so few conversations.

**Conclusion.** Every Stage-4 conclusion in [docs/RESULTS.md](../../docs/RESULTS.md) holds whichever machine wrote
the reports: the repair makes the transcript slightly worse, keywords drop once ASR errors are present, and no action
items are extracted. On one group of 9 conversations, a single number can move by a few points with the machine
(one moved by 18), which is why the conclusions rest on all 63 conversations and their intervals. Mixing CPU-made
(groups 0–1) and GPU-made (groups 2–6) reports in one table is therefore acceptable, and it is stated wherever the
table appears.
