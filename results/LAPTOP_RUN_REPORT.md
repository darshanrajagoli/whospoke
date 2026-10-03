# Stage-4 laptop run — test groups 2–6 and the benchmark notebook's Stage-4 section

Run on 3 October 2026 on the team laptop, branch `claude/stage4-eval-g02-g03`. Groups 2–3 were the planned run;
groups 4–6 were added the same evening, so that Stage 4 is scored on every test group except 7 (D33).

## Machine and software

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 6 GB Laptop GPU (driver 581.95) |
| CPU / RAM | AMD Ryzen 5 7640HS (6 cores, 12 threads), 15.3 GB RAM |
| OS | Windows 11 Home |
| llama.cpp | release b11312, prebuilt `win-cuda-12.4` x64 binaries + CUDA 12.4 runtime. This is commit `0c1e570`, the same llama.cpp that `scripts/serve_llm.py` builds from `llama-cpp-python` 0.3.36 (the laptop has no CMake / C++ / CUDA toolkit, so it was not built from source) |
| Model file | `models/airavata-q4_k_m.gguf` (4,172,544,832 bytes), made by `serve_llm.py --prepare-only` from the 16-bit `Airavata.gguf` (SHA-256 checked against the Hugging Face LFS hash before quantising) |

## GPU layers

The brief asked for the largest `--gpu-layers` that fits, trying 99 first.

| server | GPU memory used | generation speed |
|---|---|---|
| `--gpu-layers 99 --parallel 1` | 5.9 of 6.0 GB | **23.0 tokens/s** |
| `--gpu-layers 28 --parallel 1` | 5.2 GB | 19.6 tokens/s |
| `--gpu-layers 99 --parallel 2` (2 requests at once) | 5.9 GB | 22.7 tokens/s for both together |

All layers fit, so the run used **`scripts/serve_llm.py --llama-bin C:\whospoke-data\llama-b11312-cuda12.4 --gpu-layers 99 --parallel 1`**.
Two parallel slots gave no extra throughput: the second slot's 2 GB cache does not fit next to the model, so the GPU is
already the limit. The evaluation therefore ran one report at a time (`--workers 1`), as the brief said. (Measured with
a 300-token generation, two rounds each; the faster round is shown.)

`pytest -m slow -k real_llm` passed against this server (83 s).

## Evaluation (`scripts/eval_postprocess.py --groups 0 1 2 3 --workers 1`)

- The 72 group 0–1 reports were reused from the cache; their CSV rows are byte-for-byte the cloud run's.
- 72 new reports (groups 2–3: 18 conversations × 4 inputs) in **3,756 s** (62.6 min), about 52 s per report, with no errors.
- Speed per report: RTF 0.59–0.96 depending on the input (Stage-4 time ÷ audio length), 22 tokens/s. That is about
  9× faster than the 4-core cloud CPU, which ran groups 0–1 at RTF 5.4–9.1 with three reports at a time.
- Server memory: peak resident memory 4.5 GB, measured by the evaluation. Most of that is the memory-mapped model
  file while it is being copied to the GPU; during generation the server holds about 1 GB of RAM (notebook §7).
- Saved in `results/eval_postprocess_summary_g02_g03_laptop.json`. The generic `eval_postprocess_summary.json`
  written by the run was restored and not committed.

cpWER before → after the Stage-4 repair, groups 2–3 only (pooled):

| Stage-4 input | before → after |
|---|---|
| true transcript | 0.0 % → 2.2 % |
| ASR on each clean voice | 22.8 % → 24.4 % |
| ASR on the noisy mixture | 36.7 % → 37.8 % |
| full pipeline (Order B) | 44.0 % → 45.2 % |

The same as on groups 0–1: the repair adds about one point.

**Groups 4–6** (`eval_postprocess.py --workers 1`, every scorable group; groups 0–3 reused from the cache): 108 new
reports (27 conversations × 4 inputs) in 5,767 s (96 min), no errors, 21–22 tokens/s, RTF 0.58–0.95. Saved in
`results/eval_postprocess_summary_g04_g05_g06_laptop.json`. Its server peak memory reads 1.0 GB, against 4.4 GB for
the groups 2–3 run with the same settings: on Windows the resident memory includes the memory-mapped model file
only while the system has memory to spare, so the number depends on what else was running.

Over all 63 scorable conversations the full-pipeline cpWER goes from 51.8 % to 52.8 % (+0.9 points, CI +0.8 to +1.1).
The tables are in `docs/RESULTS_TABLES.md`.

**Cross-check, group 0 on the GPU.** Groups 0–1 were scored with CPU-made reports. To check that the machine does not
change the conclusions, the 36 group-0 reports were made again on this GPU, in a separate copy of the repository
(1,867 s). 94 % of the repaired lines are identical to the CPU reports, the repair still adds about one cpWER point,
there are no action items, and only the free-text summaries differ. Details: `results/crosscheck_g00_gpu/README.md`.

## Benchmark notebook (`scripts/build_notebooks.py --stage4-only --kernel whospoke`)

Only §7 was executed and spliced in. Every Stage 1–3 cell and output is unchanged (checked cell by cell against the
previous commit); the only other change is the summary sentence that now lists `results/benchmark_llm.csv`.
The demo conversation (63 s, 16 lines) takes 57.7 s of Stage-4 time (RTF 0.92, 4 LLM calls, 22 tokens/s). §7 was run
again after groups 4–6, so that its speed tables list all three evaluation runs; Stage 1–3 outputs are still unchanged.
The evaluation-speed tables are printed per machine (cloud CPU groups 0–1, laptop GPU groups 2–3), never pooled.

## What went wrong

- **Wi-Fi data cap.** Campus Wi-Fi has a daily cap, and the 13.7 GB 16-bit model was larger than one day's quota.
  The first attempt (2 October) used Hugging Face's Xet transfer, which cannot resume: about 7.4 GB was lost to a
  duplicate download that had to be killed, and a 3.8 GB partial file was deleted because it could have had gaps.
  The download that worked (3 October) used plain HTTP (`HF_HUB_DISABLE_XET=1`, resumable), only on campus Wi-Fi,
  and took about 15 minutes at 13–19 MB/s.
- **One test failed on Windows** before commit 6fbc778: `test_client_reports_unreachable_server`. Windows takes about
  2 s to refuse a connection to a closed port, which was longer than the test allowed. Fixed upstream; 103 fast
  tests pass here.
- **A lost log.** The download script's log stopped updating after 16:55 because another process held the log file
  open. The download, the SHA-256 check and the quantisation still completed; the script writes its "done" marker
  only after the hash matches.
- **Low memory.** The laptop had under 1 GB of RAM free during the evaluation (the server plus a browser). The runs
  were not affected, but it makes the server's measured peak memory vary between runs (above).
- **Sleep setting.** The brief said to set standby back to 30 minutes at the end. It was already "never" before the
  run started, so it was left as it was.
