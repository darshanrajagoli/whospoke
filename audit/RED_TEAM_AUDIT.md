# Red-team audit — whospoke

**Date:** 3 October 2026 · **Branch audited:** `claude/stage4-eval-g02-g03` (Milestones 1–4, Stage 4 scored on test
groups 0–6) · **Brief:** [docs/team/RED_TEAM_AUDIT_PROMPT.md](../docs/team/RED_TEAM_AUDIT_PROMPT.md)

## Executive summary

1. **Overall verdict:** the headline results are sound. Every Stage 1–3 number we traced matches the generated tables, there is no test leakage, and the main conclusion (Order B beats Order A) still holds when whole speaker groups are resampled.
2. **Most serious, High (fixed):** Milestone 1's deliverable, separate `.wav` files per speaker, was not produced by any command; the separator only ran inside the pipeline. `python -m whospoke separate` now writes them (F1).
3. **Medium (limitation):** Stage 4 never produced a single action item in 252 test reports, although the proposal asks it to "extract actionable information" (F2).
4. **Medium (fixed):** on Windows, `python -m whospoke postprocess` crashed with `UnicodeEncodeError` when its output was redirected or piped (F5).
5. **Also fixed in this branch:** Stage 4 was scored only on the easier half of the test set (F3), the docs said the model does not fit the 6 GB GPU when it does (F4), and the Low findings F6, F8–F12. Only F2 (no action items) and F7 (spelling variants) remain, both documented as limitations.

## Findings

| ID | Severity | Area | One-line summary | File:line | Status |
|---|---|---|---|---|---|
| F1 | High | Proposal compliance | No command writes the separated per-speaker `.wav` files that Milestone 1's deliverable asks for | `src/whospoke/__main__.py:86-104`, `src/whospoke/pipeline.py:81-94` | Fixed in this branch |
| F2 | Medium | Stage 4 | Zero action items kept in all 252 evaluation reports and in the demo report | `results/eval_postprocess.csv` (`actions`), `results/demo/report.md:22-24` | Documented as limitation |
| F3 | Medium | Evaluation validity | Stage 4 was scored only on groups 0–1 (then 0–3), the easier half of the test set | `docs/RESULTS_TABLES.md:147`, `DECISIONS.md:349` | Fixed in this branch |
| F4 | Medium | Claims vs evidence | Docs said the 4-bit model "does not fully fit" the 6 GB GPU; measured, it fits | `docs/MILESTONE4.md:135-139`, `README.md:102` | Fixed in this branch |
| F5 | Medium | Code (Windows) | `whospoke postprocess` / `run` crash with `UnicodeEncodeError` when stdout is a pipe or a file | `src/whospoke/__main__.py:79-80`, `:64-66` | Fixed in this branch |
| F6 | Low | Statistics | Confidence intervals treat the 72 conversations as independent, but they are 8 speaker groups × 9 conditions | `scripts/make_report.py:47-61` | Fixed in this branch (conclusions unchanged) |
| F7 | Low | Metrics | Text normalisation does not merge chandrabindu/anusvara, nukta or ZWJ/ZWNJ spelling variants | `src/whospoke/metrics.py:84-88` | Documented as limitation |
| F8 | Low | Repository | The generic `results/eval_postprocess_summary.json` (overwritten by every run) was committed with stale content | `.gitignore:35`, `scripts/eval_postprocess.py:233-235` | Fixed in this branch |
| F9 | Low | Measurement | "Server peak memory" on Windows is the working set: the same server setup reads 4.4 GB in one run and 1.0 GB in the next | `docs/RESULTS_TABLES.md:174-175` | Fixed in this branch (explained) |
| F10 | Low | Reproducibility | With `hf_xet` installed (pinned in requirements), the 13.7 GB model download cannot resume after an interruption | `requirements.txt` (`hf_xet==1.6.0`), `results/LAPTOP_RUN_REPORT.md:64-66` | Fixed in this branch (documented) |
| F11 | Low | Claims vs evidence | Stage-4 prose quoted the 18-conversation run after the tables moved to 63 conversations | `README.md:46`, `DEVIATIONS.md:22`, `docs/RESULTS.md` Stage-4 section | Fixed in this branch |
| F12 | Low | Evaluation validity | Stage-4 reports of groups 0–1 were made on a CPU, the rest on a GPU, and "temperature 0, same report" does not hold across the two | `docs/MILESTONE4.md:97`, `DECISIONS.md` D33 | Fixed in this branch (cross-checked; conclusions unchanged) |

---

### F1 — Milestone 1 deliverable: no separated `.wav` output · High
**Status:** Fixed in this branch. `python -m whospoke separate file.wav` writes `separated_track_1.wav` and `separated_track_2.wav` (whole recording, one common gain so the tracks never clip), listed in the README, INDEX and DEVIATIONS row 1, tested in `tests/test_cli.py`. On the demo recording (5 dB market noise) the tracks are +2.5 dB SI-SDR better than the mixture.

**What is wrong.** The proposal's Milestone 1 deliverable is "a script that ingests a multi-speaker radio broadcast wave file and outputs separate, isolated .wav channels for each detected concurrent speaker layer". No command in the repository writes separated audio to disk:
- `Result.save` writes only `timeline.json`, the transcripts, `.srt` and the report (`src/whospoke/pipeline.py:81-94`);
- the CLI has only `run` and `postprocess` (`src/whospoke/__main__.py:86-104`);
- `scripts/eval_separation.py` scores the separator but saves no audio.

The walkthrough notebook plays one separated overlap in the browser (`scripts/build_notebooks.py:262`), but writes no file. `DEVIATIONS.md` does not list this, so it reads as missing rather than as a choice.

**Evidence.** `grep -rn "sf.write\|audio.save" src scripts` finds writes only in `synth.py` (the simulated test data) and `audio.py` (the helper itself).

**Why it matters.** An examiner checking the four deliverables one by one will find this one absent. The separator itself works and is measured (+9.6 dB SI-SDRi on overlaps, `docs/RESULTS_TABLES.md:20`); only the file output is missing.

**How to reproduce.** `python -m whospoke run file.wav`, then list `results/runs/file/`: there is no `.wav` file.

**Suggested fix.** Add a `separate` subcommand:
```python
# __main__.py
sep = sub.add_parser("separate", help="Stage 1 on its own: write one .wav per separated speaker")
sep.add_argument("audio"); sep.add_argument("--out", default=None)
sep.add_argument("--separator", choices=["convtasnet", "sepformer"], default="convtasnet")
# handler:
from .separation import Separator
from .audio import save
y = Separator(args.separator).separate(load(args.audio))
out = Path(args.out) if args.out else PROJECT / "results" / "runs" / Path(args.audio).stem
for k, ch in enumerate(y, 1):
    save(out / f"speaker_{k}.wav", ch)
```
Then:
- list it in the README "Run it" block and the INDEX code table;
- add one sentence to `DEVIATIONS.md` row 1: the whole-recording tracks are available with `whospoke separate`, and Order B also separates only the overlaps.

### F2 — Stage 4 never extracts an action item · Medium
**Status:** Documented as limitation. The prompts are frozen (D32); changing them would mean developing again on group 7 and re-running all 252 reports.

**What is wrong.** The report has an "Action items" section, the project's answer to the proposal's Milestone 4 objective "extract actionable information". Across all 252 evaluation reports, 63 conversations × 4 inputs, it is empty.

**Evidence.**
- The `actions` column of `results/eval_postprocess.csv` is 0 in all 252 rows. No `results/eval_postprocess/*.json` report has a non-empty `action_items`.
- The guardrail dropped 0 actions on three inputs and 4 on the full-pipeline input (`docs/RESULTS_TABLES.md:162-165`, "actions dropped"). So the model almost always returns an empty list rather than having its items removed.
- The demo report says "None stated in the conversation" (`results/demo/report.md:22-24`), although its dialogue line 9 is a clear request: "अपने कैमरे सी सी टी वी कैमरे चेक करवा लीजिए", "Get your camera checked…" (`results/demo/report.md:47`).

**Why it matters.** It is part of the proposal's stated objective, and the report shows an always-empty section. Topic, summary and keywords do carry information, so the objective is not entirely unmet.

**How to reproduce.**
```
python -c "import pandas as pd; print(pd.read_csv('results/eval_postprocess.csv').actions.sum())"   # 0
```

**Suggested fix.**
- *Now:* say it plainly in RESULTS.md, the README limitations and MILESTONE4.md: "the model proposed 4 action items in 252 reports, all unsupported; the section is effectively unused with this 7B model".
- *Later:* develop an action prompt on group 7 only, for example a separate yes/no call per line ("does this line ask someone to do something?"), and re-run the evaluation.

### F3 — Stage 4 scored on the easier half of the test set · Medium
**Status:** Fixed in this branch. All 63 conversations that may be scored (groups 0–6) are now used; group 7 stays excluded (D32, D33 at `DECISIONS.md:349`).

**What was wrong.** Stage 4 was first scored on groups 0–1 (18 conversations), then on groups 0–3 (36). Those groups are the easier half. The Stage-3 full-pipeline cpWER per group (`results/eval_test_indicconformer.csv`, B-spectral) is 30–47 % for groups 0–3 and 59–83 % for groups 4–6. The Stage-4 input cpWER was 39.1 % on groups 0–3 and 69.9 % on groups 4–6 (`results/eval_postprocess.csv`).

**Why it mattered.** The numbers did not describe the test set. The cpWER cascade into Stage 4 looked milder than it is.

**Now.** 63 conversations (`docs/RESULTS_TABLES.md:147-156`): full pipeline 51.8 % → 52.8 % after repair, Δ +0.9 points [+0.8, +1.1]. The conclusion is the same as on 18 conversations: the repair adds about one point.

**How to reproduce.** `python scripts/eval_postprocess.py` (all groups except 7; reports are cached), then `python scripts/make_report.py`.

### F4 — "Does not fit the 6 GB GPU" was wrong · Medium
**Status:** Fixed in this branch (`docs/MILESTONE4.md:135-139`, `README.md:102`, D27's pointer to D33).

**What was wrong.** MILESTONE4 and the README said the 4-bit model "does not fully fit" the 6 GB laptop GPU, and advised `--gpu-layers 20`.

**Evidence.** Measured on the RTX 3050 with `--gpu-layers 99 --parallel 1`, it used 5.9 of 6.0 GB:

| setting | generation speed |
|---|---|
| all layers on the GPU (`--gpu-layers 99`) | 23.0 tokens/s |
| 28 layers on the GPU | 19.6 tokens/s |
| `--parallel 2` (two requests at once) | 22.7 tokens/s combined, no gain |

Source: `results/LAPTOP_RUN_REPORT.md`, "GPU layers" table. Single-request demo: 57.7 s for 62.6 s of audio (`results/benchmark_llm.csv`).

**Why it mattered.** Following the old advice, someone would have run Stage 4 at about 5× the time on the CPU, or tuned a layer count for nothing. Stages 1–3 (4.5 GB) still cannot share the card with it, and the corrected text says so.

### F5 — CLI crashes on Windows when output is redirected · Medium
**Status:** Fixed in this branch: `main()` switches stdout and stderr to UTF-8. `tests/test_cli.py` runs the CLI with a cp1252 stdout; it failed before the fix and passes after. `postprocess` with `PYTHONIOENCODING=cp1252` and output redirected now exits 0.

**What is wrong.** On Windows, Python writes to a pipe or a file in the ANSI code page (cp1252). `postprocess_transcript` prints the keywords (`src/whospoke/__main__.py:80`), which are usually Devanagari, so the command dies with `UnicodeEncodeError`. The report files are already saved by then (line 77), but the command exits with a traceback and a non-zero code. That breaks any script or log redirect. `run` prints the title and summary (`:66`), which can contain non-cp1252 characters too.

**Evidence.** Same interpreter (`C:\whospoke-data\venv`), no environment overrides:
```
python -c "import sys; print(sys.stdout.encoding); print('keywords: \u0908-\u092a\u093e\u0938')" | cat
→ UnicodeEncodeError: 'charmap' codec can't encode character '\u0908' … (stdout encoding: cp1252)
```
It works in an interactive console (UTF-8 console I/O) and with `PYTHONIOENCODING=utf-8`.

**Why it matters.** Windows 11 is the project's tested platform (README "Setup"), and redirecting output to a log is the normal way to run long jobs.

**How to reproduce.** With the LLM server running:
```
python -m whospoke postprocess results/demo/transcript.json > log.txt
```

**Suggested fix.** At the start of `main()` in `src/whospoke/__main__.py`:
```python
import sys
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")
```
Then add a test that runs `main()` with a non-UTF-8 stdout. The scripts that print Devanagari (`scripts/eval_asr.py`, `scripts/evaluate.py` progress lines) can get the same two lines.

### F6 — Intervals assume independent conversations · Low
**Status:** Fixed in this branch. `scripts/make_report.py` now also prints every paired comparison with whole speaker groups resampled (RESULTS_TABLES, "The same comparisons, resampling whole speaker groups"), and RESULTS.md's "How we measured" says so. The conclusions are unchanged; the generated table's intervals (seeded) differ from the hand-computed ones below by at most 0.4 points.

**What is wrong.** `boot_ci` and `paired_diff_ci` (`scripts/make_report.py:47-61`) resample conversations. The 72 test conversations are 8 speaker groups, each rendered under 9 conditions (`scripts/build_dataset.py:59-66`). Conversations of one group share their speech, so they are not independent and the intervals are too narrow. RESULTS_TABLES says so for Stage 4 (line 149), but not for Stages 1–3.

**Evidence.** We recomputed the paired differences from `results/eval_test_indicconformer.csv`, resampling whole groups, 4,000 draws (cpWER points):

| comparison | per-conversation CI (reported) | per-group CI | groups where it holds |
|---|---|---|---|
| Order A − Order B, cpWER +6.7 | [+4.0, +9.4] | [+3.1, +10.4] | 7/8 |
| Order A − Order B, DER +5.9 | — | [+3.3, +8.5] | 8/8 |
| no separation − targeted, +0.5 | [+0.0, +1.0] | [−0.0, +1.0] | 7/8 |
| true timeline, mixture − separated, +3.0 | [+1.7, +4.3] | [+1.3, +4.5] | 7/8 |
| ours − pyannote, +0.8 | [−2.9, +4.3] | [−2.3, +3.7] | 5/8 |

**Why it matters.** An examiner who knows the design will ask. Every conclusion survives, so this is a disclosure issue, not a result issue.

**Suggested fix.**
- One sentence in RESULTS.md's intro: "the 72 conversations come from 8 speaker groups; resampling whole groups gives +3.1 to +10.4 for Order A − B, and every conclusion holds".
- Optionally, a `--cluster group` option in `boot_ci`.

### F7 — Spelling variants count as errors · Low
**Status:** Documented as limitation. The reported numbers are unchanged.

**What is wrong.** `normalise` (`src/whospoke/metrics.py:84-88`) applies NFC and removes punctuation. It does not merge:
- chandrabindu (ँ) and anusvara (ं): हाँ vs हां;
- the nukta: ज़ vs ज;
- zero-width joiners (ZWJ/ZWNJ).

These are spelling variants, not recognition errors.

**Evidence.** Rescoring `results/asr_comparison_utterances.csv` with those variants merged lowers WER by 1–2 points for both models:

| model | Vaani | IndicVoices |
|---|---|---|
| IndicConformer | 21.0 → 20.0 % | 13.9 → 12.6 % |
| IndicWav2Vec | 38.0 → 36.4 % | 36.1 → 34.3 % |

The ranking and the size of the gap are unchanged.

**Why it matters.** It is a common question about Hindi WER. Both models are scored the same way, so no comparison is biased.

**Suggested fix.** A footnote in RESULTS.md's Stage 3 section with the numbers above.

### F8 — Stale generic summary file under version control · Low
**Status:** Fixed in this branch.

**What was wrong.** `scripts/eval_postprocess.py` writes `results/eval_postprocess_summary.json` on every run, overwriting the previous run. A copy from the cloud run (18 conversations) was committed. The tables only read the per-run files `eval_postprocess_summary_g*.json`.

**Now.** The file is untracked and ignored (`.gitignore:35`). A comment at `scripts/eval_postprocess.py:233-234` says how to keep a run's summary.

### F9 — "Server peak memory" depends on Windows memory pressure · Low
**Status:** Fixed in this branch: the speed table's note in RESULTS_TABLES now explains the measure and that it is comparable only within one machine under similar load; LAPTOP_RUN_REPORT explains the 4.4 vs 1.0 GB.

**What is wrong.** The evaluation samples the server's resident set size, which on Windows is the working set. That includes pages of the memory-mapped model file, and Windows trims them when memory runs short. The same server, on the same laptop with the same command, reads **4.4 GB** for groups 2–3 and **1.0 GB** for groups 4–6 (`docs/RESULTS_TABLES.md:174-175`). On the cloud CPU it is 12.5 GB with three slots.

**Why it matters.** Side by side, the numbers suggest a real difference that isn't there.

**Suggested fix.** A note under the speed table in `scripts/make_report.py`: "server memory is the process's resident set; on Windows it includes memory-mapped model pages and shrinks when the system is short of memory, so compare it only within one machine". Alternatively, record the GPU memory from `nvidia-smi` for GPU runs.

### F10 — Model download cannot resume · Low
**Status:** Fixed in this branch: README "Stage 4 (LLM)" and MILESTONE4 "Model file" tell users on a slow or capped connection to set `HF_HUB_DISABLE_XET=1`.

**What is wrong.** `requirements.txt` pins `hf_xet==1.6.0`. With it installed, `huggingface_hub` downloads `Airavata.gguf` (13.7 GB) through Xet, which does not resume an interrupted download. On the team's capped campus Wi-Fi this lost about 7.4 GB once, plus a 3.8 GB partial file (`results/LAPTOP_RUN_REPORT.md:64-66`). With `HF_HUB_DISABLE_XET=1` the plain-HTTP download resumes.

**Suggested fix.** One sentence in README "Stage 4 (LLM)" and in MILESTONE4 "Model file": "On an unreliable or capped connection, set `HF_HUB_DISABLE_XET=1` first, so the 13.7 GB download can resume".

### F11 — Stage-4 prose behind the tables · Low
**Status:** Fixed in this branch: README, DEVIATIONS, RESULTS and MILESTONE4 quote the 63-conversation numbers, and RESULTS explains the 51.8 % vs 49.5 % difference.

**What is wrong.** The generated tables cover 63 conversations, but some prose still quotes the 18-conversation run:
- `README.md:46`: "34.6 % → 35.6 % (+1.0 points…; 18 conversations)";
- `DEVIATIONS.md:22`: "cpWER +1.0 points on the full pipeline, 18 conversations";
- the Stage-4 section of `docs/RESULTS.md`.

**Suggested fix.**
- Replace those numbers with the ones in `docs/RESULTS_TABLES.md:153-156`: full pipeline 51.8 → 52.8 %, +0.9 [+0.8, +1.1], helped 1 / hurt 50 of 63.
- Explain why 51.8 % differs from the 49.5 % headline: the headline includes group 7, whose Stage-3 cpWER is 35.6 %.

### F12 — CPU-made and GPU-made Stage-4 reports in one table · Low
**Status:** Fixed in this branch: group 0 was re-run on the GPU and compared (`results/crosscheck_g00_gpu/`), and
MILESTONE4, RESULTS and D33 now say that reports are reproducible on one machine, not across machines.

**What is wrong.** MILESTONE4 said "Temperature is 0, so the same transcript gives the same report". That holds on one
machine. A CPU and a CUDA build of llama.cpp add up floating-point numbers in a different order, and greedy decoding
can then pick a different token where two are nearly tied. Groups 0–1 were scored with CPU-made reports and groups 2–6
with GPU-made ones, so the pooled table mixes the two.

**Evidence.** The 36 group-0 reports made again on the GPU (`results/crosscheck_g00_gpu/compare.py`):
- 381 of 406 repaired lines (94 %) are identical to the CPU's, and 19 of 36 reports have identical repaired text;
- the executive summary is identical in only 2 of 36;
- the repair's cpWER change is +1.2 points on the CPU and +1.4 on the GPU (every input positive on both);
- helped in 0 vs 2 of 36 reports; 0 action items on both;
- keywords really said are within 2 points on three inputs, and 88 % vs 70 % on the fourth (9 conversations).

**Suggested fix.** State the limit of the determinism claim, keep the cross-check in the repository, and say in the
Stage-4 results that groups 0–1 were made on a CPU. Re-running groups 0–1 on the GPU is not needed: no conclusion
changes.

---

## Checked and found fine

**1. Proposal compliance.** The proposal text was extracted from the `.docx`.

| proposal item | delivered? |
|---|---|
| M1 separation (Conv-TasNet or Demucs) | Conv-TasNet; Demucs ruled out with a reason (D8). File output: `whospoke separate` (F1, fixed) |
| M2 diarization (pyannote embeddings + spectral clustering or GMM) | both clusterers written from scratch; `timeline.json` in the proposal's `[00:12 - 00:45]: Speaker_A` shape |
| M3 transcription (IndicASR or IndicWav2Vec; native or Latin script) | both models compared; Devanagari and Hinglish output |
| M4 (localized LLM; topic, error repair, summary; report with summary, keyword tags, speaker-separated dialogue) | all present; repair measured as not helping (documented); action items: F2 |
| "unified Python implementation from raw audio to summary report" | `python -m whospoke run file.wav --postprocess` |
| "benchmark notebook: latency and memory per stage" | `notebooks/01_benchmark.ipynb`, Stage 4 in §7 |
| datasets (IndicVoices, Vaani, Nirantar, AIR-RS-DB) | all four addressed in DEVIATIONS rows 2–3 |
| "training and validation pipelines" | DEVIATIONS row 4 |

**2. Leakage.**
- **Test speakers.** They come from IndicVoices *valid*. Dev speakers come from a train shard, with every test speaker removed (`scripts/build_dataset.py:132-137`). Each speaker is used at most once per split (`:36-52`).
- **Noise.** Dev uses the first half of each DEMAND recording and ESC-50 folds 4–5; test uses the second half and folds 1–3 (`src/whospoke/noise.py:49-66`). ESC-50 keeps clips from one source recording in one fold.
- **Pretrained models.** IndicConformer was trained on IndicVoices, and this is disclosed (README limitations, D19). That is why the ASR is chosen on Vaani.
- **Hinglish lexicon.** Mined from Vaani *train* only and scored on Vaani *test* (`scripts/mine_loanwords.py:1-12`, `scripts/eval_asr.py:141`).
- **Stage-4 prompts.** Developed on group 7 only, and group 7 is never scored (`scripts/eval_postprocess.py:54,131-133`).

**3. Metrics.**
- SI-SDR follows Le Roux et al. 2019 (zero-mean, optimal scaling) and has a known-value test.
- DER uses pyannote.metrics with overlap scored and a 0.25 s collar.
- cpWER uses Hungarian assignment on edit counts, compares unmatched speakers with empty text, and pools errors over all words (`src/whospoke/metrics.py:133-151`, tested for permutation and a missing speaker).
- The 8.7 % DER floor of the true timeline is explained (pauses inside reference turns count as false alarm, `docs/RESULTS.md` Stage 2).

**4. Code.**
- **Separation windowing.** Windows cover the whole signal, the crossfade weights sum to 1, and output order is aligned by correlation (`src/whospoke/separation.py:86-110`, with a test).
- **Splice mode.** Indices stay inside the separated window, and the output keeps the input length (`src/whospoke/pipeline.py:304-330`).
- **Timeline.** Frame voting, smoothing and the nearest-other-speaker overlap rule check out (`src/whospoke/diarization.py:131-196`, with tests).
- **Long turns.** They are split at the quietest 50 ms frame (`pipeline.py:159-172`).
- **Stage 4.** LLM errors are caught, and Stages 1–3 are saved when it fails (`pipeline.py:232-240`, with a test).

**5. Claims vs evidence (Stages 1–3).** All of these match `docs/RESULTS_TABLES.md`:

| claim | value |
|---|---|
| cpWER, Order B vs Order A | 49.5 % vs 55.8 % |
| DER, Order B vs Order A | 20.3 % vs 26.2 % |
| SI-SDRi, overlaps vs whole recording | +9.6 vs +4.5 dB |
| heavy-overlap cpWER, mixture → separated overlaps | 35.5 % → 27.6 % |
| DER, GMM / pyannote | 20.3 % / 18.4 % |
| WER on Vaani, IndicConformer vs IndicWav2Vec | 21.0 % vs 38.0 % |
| English words inside Hindi recognised | 78 % |
| Hinglish spelling correct | 82.0 % of 8,630 |
| Nirantar WER, IndicConformer vs IndicWav2Vec | 11.0 % vs 30.2 % |
| Order B better than Order A | 50 of 72 conversations, +6.7 [+4.0, +9.4] |

**Stage 4.** The docs say plainly that the repair does not achieve "fix syntactic errors". On 63 conversations the repair raised cpWER by:
- +2.0 points on the true transcript;
- +1.2 on ASR of the clean voices;
- +0.9 on ASR of the noisy mixture;
- +0.9 [+0.8, +1.1] on the full pipeline, where it helped in 1 of 63 conversations (`docs/RESULTS_TABLES.md:153-156`).

The guardrails held in every report:
- every reply followed the schema;
- 0 lines were lost or restored;
- 0 chunk failures;
- speakers and times are copied from Stage 3 by construction (`src/whospoke/llm_postprocess.py:704-708`).

**6. Reproducibility.**
- Versions are pinned in `requirements.txt`, with the torch, asteroid and onnxruntime install lines spelled out.
- No hard-coded user paths in `src/`, `scripts/` or `tests/`; the data folder comes from `WHOSPOKE_DATA` or `data_location.txt`.
- Gated Hugging Face pages are listed in the README.
- The Stage-4 server needs CMake and a compiler, or a llama.cpp release (documented, including the exact release the laptop used).

**7. Tests.**
- 103 fast tests pass.
- Slow tests (`-m slow`) run the real models; the real-LLM test skips when no server is running, so it reads as skipped rather than passed.
- Every Stage-4 guardrail has a scripted-reply test.
- The CLI's `separate` command and its console encoding are now tested (`tests/test_cli.py`). Not tested: Order A's leak-suppression thresholds on real audio.

**8. Hinglish romaniser.**
- Common words come out well: `डॉक्टर` → doctor, `स्कूल` → school, `रिचार्ज` → recharge, `फ़ोन` → phone, `आधार कार्ड` → aadhar card, `मैं ठीक हूँ` → main theek hoon.
- Words missing from the lexicon fall back to phonetic spelling: `कैंसल` → kainsal, `व्हाट्सएप` → vhatsaep, `इंटरव्यू` → intravyu. The README already says "readable, not a standard spelling".

**9. Stage-4 faithfulness.** Read against the transcript, the demo report gets the topic and several facts right, and mistranslates some lines ("e-pass" from एग्जाम पास, "base card" for आधार). This is documented honestly in `results/demo/README.md`.

**10. Transparency.** The two fixes made after the first test run (D25, D26) are logged, and the first run's numbers are kept (`results/eval_test_indicconformer_run1.csv`).

## The 10 hardest examiner questions

1. **"Your test data is simulated. Why should I believe it says anything about real radio?"**
   A real broadcast has no labels, so it can't be scored. The conversations use real spontaneous Hindi speech and real village/market recordings, at controlled overlap and noise. Missing effects:
   - room echo;
   - phone codecs;
   - more than two people talking at once.

   The Vaani ASR test is real field audio. The numbers measure the pipeline under controlled stress; they do not predict broadcast accuracy.

2. **"You changed the professor's order of stages. Isn't that just a different pipeline?"**
   Both orders are built and run on the same 72 conversations. Order B is better by 6.7 cpWER points: CI +4.0 to +9.4 per conversation, and +3.2 to +10.4 when resampling whole speaker groups. It wins in 50 of 72 conversations and in 7 of 8 groups. Order A is still available (`--order A`).

3. **"Where are the separated wav files Milestone 1 asks for?"**
   `python -m whospoke separate recording.wav` writes one `.wav` per separated voice of the whole recording (F1). Inside the pipeline (Order B) only the overlapping stretches are separated, because that measured better.

4. **"Your diarizer is 'within error bars' of pyannote. Isn't that just saying it's slightly worse?"**
   Yes. DER is 20.3 % vs 18.4 %, and cpWER differs by 0.8 points, with CI −2.3 to +3.6 resampling groups. Pyannote wins in 5 of 8 groups. Our version is built from scratch as the proposal asks. Its real weakness is the speaker count in 2-speaker conversations: it finds only one speaker 36 % of the time.

5. **"IndicConformer was trained on IndicVoices, and your test set is IndicVoices. Isn't the ASR result contaminated?"**
   Partly, which is why the ASR was chosen on Vaani, which neither model saw. IndicConformer wins there too, 21.0 % vs 38.0 %. The end-to-end numbers on IndicVoices flatter the ASR and say so.

6. **"Stage 4 makes the transcript worse. Why is it in the pipeline?"**
   The proposal asks for a translated, summarised report, and the LLM writes it. We measured the repair step: +0.9 to +2.0 cpWER points, helping in 1 of 63 full-pipeline conversations. So the Stage-3 transcript stays the record, and the report is a reading aid.

7. **"Can the LLM invent things?"**
   It cannot:
   - change a speaker or a time: they are not in its reply format;
   - drop or reorder a line: the schema pins every line number;
   - keep a keyword whose words were never said.

   It can still mistranslate, and the summary follows ASR mistakes. Keywords really said fall from 98.8 % on the true transcript to 60.5 % on the full pipeline.

8. **"Why does the report never list action items?"**
   On this test set the 7B model returned none: 4 proposed in 252 reports, all unsupported (F2). The prompts were frozen before scoring, so we report it rather than tune on the test set.

9. **"How much of the full-pipeline error is the diarizer's fault?"**
   About 23 of the ~50 cpWER points. ASR on the clean voices is 17.7 %, noise and overlap take it to 29.7 %, separation brings it back to 26.1 %, and our diarization takes it to 49.5 %. A word given to the wrong speaker counts twice in cpWER, which is why that metric is used.

10. **"Would any conclusion flip with a different random seed or more data?"**
    The paired differences hold when whole speaker groups are resampled (F6). Spelling-variant normalisation doesn't change the ASR ranking (F7). The weakest claims are the ones we already call ties: spectral vs GMM, and ours vs pyannote.
