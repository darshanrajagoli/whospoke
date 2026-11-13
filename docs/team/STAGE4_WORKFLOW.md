# WhoSpoke Stage 4 (Airavata) — Engineering Workflow

**Status:** Done (November 2026; outcome in DECISIONS D36–D39). Kept for the record.  
**Scope:** Stage 4 only — Airavata-backed LLM post-processing, local model/server setup, report generation, evaluation, tests, and Stage-4 documentation.  
**Canonical instructions:** This file. The legacy misspelled `worklfow.md` must contain only a short pointer to this file; do not maintain two copies.

## 1. Mission and definition of success

Make Stage 4 correct, lean, testable, reproducible within stated hardware limits, and honest about model quality. Remove only code proven to be dead. Consolidate only genuinely equivalent duplicate logic. Preserve all established behavior and the Stage-3 transcript as the source of truth.

The objective is **not** to make Airavata appear successful. The objective is to ensure that:
- the complete Stage-4 path runs from a valid Stage-3 `transcript.json` to `report.json` and `report.md`;
- report fields are grounded in the transcript or are explicitly marked as model-generated/uncertain;
- speaker identities, timestamps, line identity, and source transcript are never invented or modified by the LLM;
- failures are visible, recoverable where designed, and covered by deterministic tests;
- evaluation uses the approved data split and records the exact configuration;
- dead code and redundancy are removed only with evidence;
- Stages 1–3 and unrelated work remain untouched.

A task is not complete because code was written. It is complete only when its acceptance criteria and evidence requirements below are met.

## 2. Mandatory operating rules

### 2.1 Inspect before editing
1. Start with `git status --short --branch`, `git diff --stat`, and `git diff`.
2. Do not reset, checkout over, stash, clean, rebase, or overwrite user changes.
3. Read this workflow and the Stage-4 source documents before editing:
   - `docs/MILESTONE4.md`
   - `DECISIONS.md`, especially D27–D33
   - `audit/RED_TEAM_AUDIT.md`, especially F2–F4 and F8–F12
   - Stage-4 sections of `docs/RESULTS.md` and `docs/RESULTS_TABLES.md`
   - `DEVIATIONS.md` and the Stage-4 portion of `README.md`
4. Inspect the actual code, tests, CLI wiring, and current result artifacts. Do not assume this workflow perfectly describes the current checkout; verify every relevant claim against the repository.

### 2.2 Strict scope boundary
Primary allowed files:
- `src/whospoke/llm_postprocess.py`
- `src/whospoke/resources/airavata_chat_template.jinja`
- `scripts/serve_llm.py`
- `scripts/eval_postprocess.py`
- `tests/test_llm_postprocess.py`
- narrowly scoped Stage-4 tests, if needed
- Stage-4-specific sections in `docs/MILESTONE4.md`, `docs/RESULTS.md`, `docs/RESULTS_TABLES.md`, `README.md`, `DEVIATIONS.md`, `DECISIONS.md`, and `audit/RED_TEAM_AUDIT.md`
- `src/whospoke/pipeline.py` or `src/whospoke/__main__.py` only for a demonstrated Stage-4 integration defect, with the smallest possible patch.

Do not modify Stages 1–3, datasets, notebooks, demo content, unrelated scripts, generated results, or general project architecture. Do not reformat whole files. Do not rename public functions, classes, CLI flags, report fields, or schemas without proving all callers and compatibility requirements.

### 2.3 Evidence before implementation
Before any behavior change, write down:
- the concrete defect or unmet requirement;
- the exact source location and current behavior;
- a reproducible failing example or test;
- the intended behavior;
- the smallest proposed change;
- the regression test that will prove the fix.

If the repository does not define expected behavior, do not invent it. Record the ambiguity and ask the project owner or leave the behavior unchanged.

### 2.4 No hallucinated results
- Never fabricate test output, model output, evaluation metrics, speed, memory, or completion status.
- Label results `NOT RUN`, `BLOCKED`, `FAILED`, or `PASSED` accurately.
- A skipped real-model test is not a pass.
- Do not claim an improvement based on a few hand-picked examples.
- Preserve raw per-run artifacts and logs. Never silently overwrite the only copy of existing evidence.

### 2.5 No test leakage
- Group 7 is the only permitted speaker group for prompt development, prompt selection, or qualitative tuning.
- Groups 0–6 are the scored evaluation set. Do not inspect their report text to tune prompts, choose examples, or select a variant.
- Freeze prompt/schema/guardrail changes before the final scored evaluation.
- If a behavior-changing prompt is modified after seeing any scored result, the prior evaluation is no longer a clean final evaluation. Record the change and rerun the full eligible evaluation with a new run ID.
- Do not use evaluation reports as development fixtures. Synthetic tests and group-7 examples are permitted.

### 2.6 Preserve guardrails
Never relax evidence checks, line mapping, transcript integrity, repair limits, JSON validation, or speaker/timestamp provenance merely to increase the number of accepted outputs. A rejected model suggestion must remain rejected and be explained by diagnostics.

### 2.7 No unnecessary dependencies or abstractions
- Preserve the current local llama.cpp/OpenAI-compatible HTTP approach unless a reproducible defect proves it insufficient.
- Do not add the OpenAI SDK, hosted inference dependency, API-key requirement, second HTTP stack, framework, configuration layer, or parallel report schema without explicit approval.
- Reuse existing canonical helpers and serializers.
- Do not create wrappers whose only purpose is to rename one function call.
- Do not duplicate prompt text, schema definitions, validation rules, or report rendering.

## 3. Current repository contract to verify

The following describes the current checked-in design; verify it against the checkout before relying on it.

### 3.1 Data contract
Input: Stage-3 `transcript.json`.

Default outputs next to the input:
- `report.json` — machine-readable report, provenance, and diagnostics.
- `report.md` — human-readable report.

The Stage-3 transcript is authoritative. Stage 4 is a reading aid, not a transcript replacement. The LLM must not author or modify:
- speaker IDs/labels;
- timestamps or turn boundaries;
- source line IDs;
- original Stage-3 transcript text;
- line count or ordering.

Hinglish must come from the Stage-3 romaniser (including the existing repair-line behavior), not a separate LLM transliteration.

### 3.2 Report contract
Verify the exact schema in code and tests. The documented report includes title/topic, executive summary, key points, validated keywords, evidence-backed action items, speaker statistics, line-level dialogue, changed/uncertain lines, diagnostics, and provenance. Do not add or remove fields based only on this prose.

### 3.3 Airavata runtime
Current setup is `scripts/serve_llm.py`, llama.cpp `llama-server`, and the OpenAI-compatible `/v1/chat/completions` endpoint. The checked-in setup currently uses:
- model repository `ai4bharat/Airavata`;
- source GGUF `Airavata.gguf`;
- local Q4_K_M model path `models/airavata-q4_k_m.gguf`;
- default context 4096;
- default GPU layers 0 (CPU inference);
- default parallel slots 1;
- Airavata-specific Jinja chat template at `src/whospoke/resources/airavata_chat_template.jinja`.

The user has already been granted Hugging Face access and authenticated successfully. The `huggingface_hub==0.36.2` and `hf_xet==1.6.0` packages were installed in the repository's `venv`. llama.cpp CPU tools (`llama-server` and `llama-quantize`) were compiled successfully in the preceding setup attempt. The model download was observed at 44% at the last handoff. This status is historical and must be checked before acting.

**Do not**:
- log out of Hugging Face, replace credentials, request or print the token, or put it in source/config/logs;
- delete `models/Airavata-hf`, partial download/cache files, or the final GGUF while a download/quantization may be active;
- start a second download or second quantization process;
- rebuild llama.cpp unless the executable is missing/broken or a reproducible build defect requires it;
- run CPU/GPU benchmarks while a download or model preparation is active;
- change quantization, model source, chat template, context, or server flags without a concrete reason and test/evidence.

Use the existing `venv` if it exists; do not assume `.venv`. Check `python -c "import huggingface_hub; print(huggingface_hub.__version__)"` and `hf auth whoami` only if setup verification is needed. Never request the user's token in chat.

### 3.4 Hardware and memory claims require reconciliation
The current documentation describes a 6 GB RTX 3050 and reports a measured full-GPU run. User environment notes elsewhere identify an RTX 3050 Mobile with 4 GB VRAM. Treat this as a **hardware/documentation discrepancy**, not something to resolve by guessing.

Before changing any GPU claim:
1. inspect the actual device (`nvidia-smi`, if available);
2. distinguish physical VRAM from free VRAM and shared/system memory;
3. check whether the documented 6 GB measurement belongs to a different machine;
4. preserve old benchmark artifacts and identify their machine;
5. update only the claims that are demonstrably wrong for the current machine.

The current user-started run is CPU-only (`--gpu-layers 0 --ctx 4096 --threads 8`). Do not switch it to GPU mode during this task. Do not run Stages 1–3 concurrently with a GPU Stage-4 test.

## 4. Known Stage-4 findings and required treatment

### F2 — Action-item extraction is weak/empty
The audit reports four proposed actions across 252 reports, all unsupported and dropped; the demo also misses a clear request.

Required:
1. inspect the current action prompt, schema, parser, evidence validation, rendering, and tests;
2. reproduce behavior with scripted model replies before changing anything;
3. use group 7 only to determine whether prompt wording is a demonstrated cause;
4. preserve requirements for explicit request/commitment, real or unspecified owner, and exact existing evidence line IDs;
5. do not treat topics, questions, suggestions, or inferred obligations as actions;
6. add synthetic/scripted tests for accepted and rejected cases;
7. freeze any prompt change before groups 0–6 are evaluated;
8. if real Airavata still yields no supported actions, document that honestly. Zero actions may be the correct safe outcome.

### F3 — Evaluation coverage
The approved scored set is all 63 eligible conversations in groups 0–6. Group 7 is excluded from scored evaluation. Verify the evaluator's actual selection logic; do not rely on a comment or this workflow alone.

The evaluation currently documents four input systems: `reference`, `oracle-clean`, `oracle-mix`, and `B-spectral`. Confirm these exact choices and their data availability in code before running.

### F4 / D33 — GPU and resource claims
Do not repeat or revise hardware claims without measurement on the named machine. The repository's measured 6 GB GPU result may not describe the user's 4 GB device. Keep one request slot as the conservative default unless measurement demonstrates otherwise. Do not imply the model can share the GPU with Stages 1–3.

### F8 — Stale generic evaluation summary
Inspect `.gitignore` and evaluator output naming. Do not treat a mutable `results/eval_postprocess_summary.json` as canonical. Preserve per-run summaries with group, machine, model, and settings metadata. Do not overwrite previous results.

### F9 — Memory measurement interpretation
On Windows, RSS/working-set measurements can include memory-mapped model pages and vary with memory pressure. Do not compare raw memory values across machines/runs without recording measurement method and machine. Preserve the documented caveat unless new evidence justifies a correction.

### F10 — Interrupted downloads
Preserve resumable Hugging Face download behavior and documented `HF_HUB_DISABLE_XET=1` guidance where supported by the current downloader. Do not replace the download mechanism without reproducing a failure and testing the replacement. Do not interrupt the user's active download.

### F11 — Stale claims
Reconcile Stage-4 prose and tables against the latest raw evaluation artifacts and generated tables. Do not manually edit generated numeric tables. Do not reintroduce older 18- or 36-conversation numbers as current results.

### F12 — CPU/GPU reproducibility
Temperature 0 does not guarantee identical output across CPU/GPU builds or different hardware. Keep machine/build/configuration metadata and do not pool CPU and GPU speed rows. State reproducibility only at the level actually supported by repeated runs.

### Other known limitations
- Existing measured results say repair increases cpWER rather than improving transcript accuracy.
- Summary faithfulness is imperfect; keyword grounding does not prove summary faithfulness.
- The evaluated Airavata model did not produce supported action items.
- Prompt changes can alter output and metrics; freeze before final scoring.

These are findings to verify against current evidence, not permission to claim they remain unchanged without checking.

## 5. Execution plan — small, sequential tasks

Do not combine tasks. Each task must end with a short report: objective, files inspected/changed, commands run, exact outcomes, and next blocker. Stop if a prerequisite fails.

### Task 0 — Protect the working tree and active model setup (read-only)
1. Run `git status --short --branch`, `git diff --stat`, and `git diff`.
2. Record uncommitted changes and do not overwrite them.
3. Check whether `python scripts/serve_llm.py` is still downloading/quantizing; inspect process list and model-file sizes without killing processes.
4. Do not touch the HF cache, `models/Airavata-hf`, or `models/airavata-q4_k_m.gguf` during active preparation.
5. Confirm the repository's actual venv directory (`venv` in the supplied checkout) and Python executable.
6. Record free disk space and current model artifacts.

**Acceptance:** no files changed; user work and active model preparation remain undisturbed.

### Task 1 — Establish the Stage-4 baseline (read-only)
1. Read the source documents listed in §2.1.
2. Inspect:
   - `src/whospoke/llm_postprocess.py`
   - `src/whospoke/resources/airavata_chat_template.jinja`
   - `scripts/serve_llm.py`
   - `scripts/eval_postprocess.py`
   - `tests/test_llm_postprocess.py`
   - Stage-4 wiring in `pipeline.py` and `__main__.py`
3. Map public entry points, data flow, prompt constants, schemas, validation helpers, serializers, report fields, diagnostics, and CLI flags.
4. Run baseline fast tests:
   ```bash
   pytest tests/test_llm_postprocess.py -q
   ```
5. If imports/dependencies block the test, record the exact error. Do not label it a code failure.
6. Save a baseline diff summary in the agent response, not by adding a new repository file unless explicitly needed.

**Acceptance:** baseline behavior and test status are recorded; no edits.

### Task 2 — Build a Stage-4 symbol and dependency inventory (read-only)
1. Inventory Stage-4 imports, functions, classes, constants, prompts, schemas, and serializers.
2. Search all repository references: `src/`, `scripts/`, `tests/`, CLI entry points, notebooks, docs, and dynamic string references.
3. For each suspected dead symbol, record exact location, search scope, all references, runtime/public/test/doc status, and why it is unreachable.
4. Categorize as:
   - confirmed dead;
   - duplicate implementation;
   - possibly obsolete but referenced;
   - required fallback/diagnostic/schema/test code.
5. Do not delete code in this task. Linter warnings alone are insufficient proof.

**Acceptance:** evidence-backed inventory; ambiguous candidates remain.

### Task 3 — Lock down contract invariants with tests
Confirm or add only missing assertions for:
- exact preservation of Stage-3 line count/order/IDs;
- each report line maps to one source line;
- speaker and start/end times are copied from Stage 3;
- unknown, missing, or duplicate IDs are handled deterministically;
- rejected repair falls back to original ASR text;
- Hinglish remains Stage-3 romaniser output;
- missing/invalid English is flagged, not invented;
- keywords and actions pass grounding validation;
- diagnostics describe fallbacks and rejected outputs;
- empty transcript avoids unnecessary model calls;
- JSON and Markdown serializers agree on the same underlying report.

Use existing fixtures/helpers. Do not rewrite tests for style. Prefer deterministic fake clients and scripted replies; tests must not download the model.

Run:
```bash
pytest tests/test_llm_postprocess.py -q
```

**Acceptance:** the invariants are executable tests; no unverified behavior change.

### Task 4 — Remove confirmed dead Stage-4 code
1. Use only the approved candidates from Task 2.
2. Delete one small group at a time.
3. Remove imports made unused by that deletion.
4. Keep fallback, diagnostics, schema, compatibility, and test code unless proven unreachable.
5. Do not combine cleanup with behavior changes or broad formatting.
6. After each deletion run the focused test file and any directly affected CLI/pipeline test.

If no candidate is proven dead, make no deletion and report `none proven dead`.

**Acceptance:** each deletion has reference evidence; behavior/report schema unchanged; focused tests pass.

### Task 5 — Consolidate exact duplicate logic
1. Compare parsing, schema handling, reply validation, fallback, translation validation, keyword/action grounding, synthesis fallback, and serialization.
2. Select only duplicates with identical inputs, outputs, error handling, diagnostics, and ordering.
3. Add/confirm regression tests before consolidation.
4. Reuse the existing canonical helper; do not add generic frameworks or new abstractions.
5. Compare scripted outputs before/after for the same inputs.
6. If equivalence cannot be proved, leave both paths and explain why.

**Acceptance:** a demonstrably smaller implementation with unchanged behavior on the regression cases.

### Task 6 — Investigate action-item extraction (conditional)
1. Reproduce F2 using scripted replies and inspect group-7 development examples only.
2. Determine whether the failure is prompt wording, schema, parser, evidence validation, or absence of an explicit commitment.
3. If prompt wording is proven to be the cause, make one minimal change.
4. Add synthetic tests for:
   - explicit request with named owner;
   - explicit request with unspecified owner;
   - suggestion without commitment (reject);
   - question without commitment (reject);
   - unsupported evidence (reject);
   - nonexistent evidence line ID (reject).
5. Do not weaken evidence rules.
6. Freeze prompt/schema and record the change in `DECISIONS.md` before scored evaluation.
7. If no justified change exists, do not change the prompt.

**Acceptance:** only explicit, evidence-backed actions survive; no unsupported action is invented. If extraction remains empty, report the limitation.

### Task 7 — Audit evaluator selection, cache, and artifact naming
1. Inspect `scripts/eval_postprocess.py`, result loaders, cache identity, `.gitignore`, and current artifacts.
2. Verify groups 0–6 are included and group 7 excluded.
3. Verify the four input systems and exact eligible conversation count from data, not a hardcoded expected number alone.
4. Check whether cache identity includes transcript content and every output-affecting setting (model/revision, prompt/schema version, server settings, context, temperature, relevant code/config version).
5. If cache identity is incomplete, write a failing test and implement the smallest Stage-4-specific fix.
6. Preserve prior per-run results. Use a unique run identifier and record machine, OS, Python, model file/hash, quantization, llama.cpp version, CPU/GPU, GPU memory, context, parallel slots, workers, prompt/schema revision, and elapsed time.
7. Never overwrite an existing run summary. Do not commit a generic mutable summary as canonical.
8. Do not run full evaluation until all prompt/schema changes are frozen and model server is available.

**Acceptance:** selection, cache reuse, and output naming are verified by code/tests; no prior artifacts overwritten.

### Task 8 — Audit local Airavata setup and server behavior
1. Verify the active download has completed before starting another preparation step.
2. Verify the final GGUF exists and is non-empty; verify the quantization process exited successfully.
3. Verify `llama-server` and `llama-quantize` binaries exist and are executable; do not rebuild unnecessarily.
4. Start the server using the user's already-selected CPU configuration first:
   ```bash
   python scripts/serve_llm.py --gpu-layers 0 --ctx 4096 --threads 8
   ```
5. In another terminal, inspect:
   ```bash
   curl --fail --silent --show-error http://127.0.0.1:8080/v1/models
   ```
6. Send one small deterministic smoke request using the project's client path or a controlled OpenAI-compatible request. Verify JSON response and Airavata template behavior.
7. Confirm schema errors are visible; do not report schema-constrained generation as successful if the server rejected the schema.
8. Test failure modes using fakes, not destructive network/model operations.
9. Do not use GPU offload in this task. Any GPU experiment is a separate, explicitly measured task after CPU validation and after confirming physical VRAM.

**Acceptance:** model file, server, endpoint, model identity, template, and one smoke request are verified. Preserve server logs and exact commands.

### Task 9 — Run one end-to-end Stage-4 demo
1. Use an existing Stage-3 transcript fixture/demo, not a newly fabricated transcript.
2. Keep the Airavata server running.
3. Run the documented postprocess CLI against the selected transcript.
4. Verify both `report.json` and `report.md` are written to the expected location.
5. Validate report schema, source-line mapping, speaker/timestamp preservation, diagnostics, and provenance.
6. Compare report lines against the input transcript; do not judge quality from the summary alone.
7. Preserve the output under a unique test/run directory. Do not overwrite canonical demo results.
8. Record whether the run used the real Airavata server or a fake client.

**Acceptance:** the real local path completes and produces internally consistent artifacts, or a specific reproducible blocker is reported.

### Task 10 — Final scored evaluation (only after freeze)
1. Confirm prompt/schema/guardrail freeze and record the exact revision.
2. Confirm the evaluator includes every eligible conversation in groups 0–6 and excludes group 7.
3. Confirm all four input systems are available.
4. Run the full 63-conversation evaluation with the real Airavata server. Do not substitute a fake client.
5. Preserve raw report outputs, CSV, per-run JSON summary, server log, and configuration metadata under a unique run ID.
6. Check for missing/duplicate conversation-system pairs and verify expected coverage from actual inputs.
7. Inspect aggregate metrics and selected reports for consistency, but do not tune against scored groups.
8. Do not merge CPU and GPU results. Do not compare runtime across machines without labeling hardware.
9. Regenerate report tables only through the repository's intended report-generation script; never hand-edit generated numeric values.

**Acceptance:** full eligible evaluation is complete, coverage is proven, artifacts are preserved, and conclusions match the measurements. If resource/time limits prevent this, mark it `BLOCKED/NOT RUN`; do not imply completion.

### Task 11 — Documentation reconciliation and final verification
1. Compare every Stage-4 claim in README, milestone, results, tables, deviations, decisions, and audit against current code and preserved artifacts.
2. Fix only demonstrably stale/contradictory Stage-4 statements.
3. Preserve the known limitations unless new valid evidence changes them.
4. Never manually edit generated numeric tables.
5. Run:
   ```bash
   pytest tests/test_llm_postprocess.py -q
   pytest tests/test_pipeline_integration.py -q
   pytest tests/test_cli.py -q
   ```
6. Run `pytest -q` only if the environment supports the full suite and it is within scope/time.
7. Run the real-server/slow test only with the server already prepared and available. Report skips distinctly.
8. Review `git diff --check`, `git diff --stat`, and the complete diff. Confirm no Stage-1–3, unrelated, generated, credential, model, cache, or environment files changed.

**Acceptance:** Stage-4 tests pass; docs match measured evidence; diff is scoped and reviewable; final report is complete.

## 6. Required agent progress report format

After each task, report:

```text
Task:
Status: PASSED / FAILED / BLOCKED / NOT RUN
Files inspected:
Files changed:
Defect/evidence:
Change made (or reason for no change):
Commands run:
Exact test output/result:
Artifacts created (exact paths):
Risks/unresolved questions:
Next task allowed: YES / NO
```

Do not proceed when a task's prerequisite failed. Do not conceal a failing test by changing unrelated tests, skipping it, or weakening assertions.

## 7. Final completion report

The final report must state:
1. exact files changed;
2. confirmed dead code removed (symbol, location, reference audit), or `none proven dead`;
3. redundancy consolidated and tests proving equivalence, or `none safely consolidated`;
4. behavior changes and their regression tests;
5. exact commands and pass/fail/skip counts;
6. whether the real Airavata server was used;
7. whether the complete 63-conversation evaluation ran;
8. exact run ID and artifact paths;
9. prompt/schema revision and confirmation that group 7 alone was used for development;
10. hardware/model/server configuration and memory measurements, if measured;
11. remaining limitations, especially repair quality, summary faithfulness, and action extraction;
12. explicit confirmation that Stages 1–3 and unrelated files were not modified.

## 8. Explicitly prohibited shortcuts

- Do not rewrite `llm_postprocess.py` from scratch.
- Do not replace Airavata or switch to hosted inference.
- Do not delete code based only on names, lint output, or apparent lack of use.
- Do not weaken schema, evidence validation, or transcript-integrity guardrails.
- Do not ask the LLM to generate speaker IDs, timestamps, source IDs, or original transcript content.
- Do not use scored groups 0–6 for prompt development.
- Do not report improvements without a frozen full evaluation.
- Do not overwrite existing evaluation artifacts.
- Do not add dependencies, abstractions, or broad refactors without explicit approval.
- Do not interrupt, restart, duplicate, or clean up the user's active Airavata download/quantization.
- Do not claim a model/server/evaluation test passed if it was skipped, faked, or not run.

**The governing principle:** preserve what is correct, remove only what is proven unnecessary, change only what is proven defective, and make every claim traceable to code, tests, or retained measurements.
