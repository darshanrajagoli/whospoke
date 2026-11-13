# Agent Execution Prompt — WhoSpoke Stage 4 (Airavata)

*Used for the November Stage-4 work (DECISIONS D36–D39); kept for the record. The plan it refers to is
[STAGE4_WORKFLOW.md](STAGE4_WORKFLOW.md).*

You are the implementation agent working in the existing WhoSpoke repository. Your task is to execute the Stage-4 engineering plan in `STAGE4_WORKFLOW.md`, not to redesign the project.

## Mission

Audit, clean, test, and validate Stage 4 (Airavata LLM post-processing) end to end. Produce a correct and maintainable implementation with no dead Stage-4 code, no unnecessary duplication, no invented behavior, and no pollution of Stages 1–3 or existing user work.

Read `STAGE4_WORKFLOW.md` completely and follow its tasks in order. It is the governing specification. Do not skip prerequisites.

## Non-negotiable constraints

1. **Protect the working tree.** Start with `git status --short --branch`, `git diff --stat`, and `git diff`. Do not reset, stash, clean, checkout over, or overwrite existing changes.
2. **Stage 4 only.** Do not refactor Stages 1–3, datasets, notebooks, unrelated scripts, demo artifacts, or general architecture. Only touch files permitted by `STAGE4_WORKFLOW.md`.
3. **No speculative coding.** Every behavior change requires a concrete defect, source location, reproducible case, expected behavior, and regression test. If the requirement is unclear, stop and report the ambiguity.
4. **No dead-code guesswork.** Do not delete a symbol until you have searched direct references, imports, CLI/string references, tests, dynamic use, docs, and serialization paths. If uncertain, keep it.
5. **No redundancy theatre.** Consolidate only code with demonstrably identical behavior, return values, diagnostics, and ordering. Do not add abstractions or wrappers just to make the code look cleaner.
6. **No regression laundering.** Never weaken a test, remove a guardrail, or change expected behavior merely to make a failing test pass. Explain the failure and isolate the smallest legitimate fix.
7. **No fabricated results.** Do not invent test outputs, Airavata responses, accuracy, speed, memory, or evaluation completion. Mark anything not run as `NOT RUN` or `BLOCKED`.
8. **No evaluation leakage.** Use speaker group 7 only for prompt development. Never inspect groups 0–6 to tune prompts or select prompt variants. Freeze prompt/schema before final scored evaluation.
9. **Protect transcript truth.** Stage-3 transcript is authoritative. The LLM must not author speaker labels, timestamps, line IDs, source text, line order, or line count. Preserve existing validation and fallback behavior.
10. **No dependency creep.** Do not add an SDK, hosted API, extra HTTP client, framework, or other dependency unless explicitly approved.
11. **No credential exposure.** Never ask for, print, commit, or write the Hugging Face token. Do not alter Hugging Face authentication.
12. **No active download interference.** The user's Airavata download was at 44% when work was handed off. First determine whether it has completed. Do not kill/restart the process, delete model/cache files, start a second download/quantization, or rebuild llama.cpp unnecessarily.
13. **Do not assume hardware.** Repository docs describe a 6 GB RTX 3050; user environment notes indicate a 4 GB RTX 3050 Mobile. Treat this as unresolved until measured. Current user-started server preparation is CPU-only: `--gpu-layers 0 --ctx 4096 --threads 8`. Do not switch to GPU during this run.
14. **No broad formatting.** Keep diffs small and localized. Do not reformat entire files or alter unrelated comments/docs.
15. **Do not create a parallel workflow.** `STAGE4_WORKFLOW.md` is canonical. If `worklfow.md` exists, replace its old duplicated content with a brief pointer to `STAGE4_WORKFLOW.md`; do not maintain two copies.

## Required execution sequence

Execute the tasks in `STAGE4_WORKFLOW.md` in order:

- Task 0: protect the working tree and check active model setup (read-only).
- Task 1: establish Stage-4 baseline and run focused tests.
- Task 2: create evidence-backed dead-code/redundancy inventory (no deletion).
- Task 3: confirm report/transcript invariants with deterministic tests.
- Task 4: remove only confirmed dead Stage-4 code.
- Task 5: consolidate only exact duplicate logic.
- Task 6: investigate action-item extraction, changing it only if group-7 evidence justifies it.
- Task 7: audit evaluator group selection, cache identity, and per-run artifact naming.
- Task 8: validate the prepared Airavata model/server without disrupting the active download.
- Task 9: run one real end-to-end Stage-4 demo using an existing Stage-3 transcript.
- Task 10: run the complete scored evaluation only after prompts/schema are frozen and the server is ready.
- Task 11: reconcile documentation and run final verification.

After every task, provide the progress report format from `STAGE4_WORKFLOW.md`. Stop if a prerequisite fails. Do not silently continue past a failing test or blocked dependency.

## Important execution details

- The repository currently has `venv/` rather than `.venv/`; inspect and use the existing environment.
- Hugging Face access was granted and `hf auth login` succeeded. The required `huggingface_hub==0.36.2` and `hf_xet==1.6.0` packages were installed in that venv.
- llama.cpp CPU compilation already succeeded in the prior run, including `llama-server` and `llama-quantize`. Reuse those binaries if present.
- The model download was observed at 44% at handoff. It may have completed since then; inspect process and files before taking action.
- The intended local model path is `models/airavata-q4_k_m.gguf`. The original gated download is `Airavata.gguf`; do not delete intermediates while the process may be using them.
- The initial real server configuration should remain CPU-only:
  ```bash
  python scripts/serve_llm.py --gpu-layers 0 --ctx 4096 --threads 8
  ```
- Once it is running, verify the endpoint from another terminal:
  ```bash
  curl --fail --silent --show-error http://127.0.0.1:8080/v1/models
  ```
- Do not start full evaluation merely because the server responds. First verify evaluator coverage, cache correctness, prompt freeze, and unique output paths.

## Coding standards

- Prefer the smallest patch that fixes a demonstrated defect.
- Preserve existing naming, public interfaces, report schema, diagnostics, and deterministic ordering.
- Use existing helpers and fixtures.
- Add tests before or alongside behavior changes; use scripted/fake model responses for deterministic unit tests.
- Real-model tests must be clearly separated from fake-client tests.
- Never make unit tests download Airavata or compile llama.cpp.
- Keep generated evaluation artifacts, model files, HF cache, logs containing credentials, and virtual environments out of Git.
- Run `git diff --check` and inspect the complete diff before declaring completion.

## Stop conditions

Stop and report `BLOCKED` instead of guessing if:
- the expected Stage-4 contract conflicts with current code/docs and cannot be resolved from repository evidence;
- a change would require modifying Stages 1–3 or unrelated architecture;
- a model download/quantization is active and a proposed action could interfere;
- the evaluator cannot prove the eligible group selection;
- the cache can reuse stale reports across output-affecting configurations and a safe fix is not yet tested;
- prompt tuning would require looking at groups 0–6;
- hardware/resource claims cannot be attributed to the actual machine;
- a test fails for an unexplained reason.

## Final response required

At completion, provide:
- exact changed files;
- dead code removed with reference-audit evidence, or `none proven dead`;
- duplicate logic consolidated and equivalence tests, or `none safely consolidated`;
- every behavior change and regression test;
- exact commands and test outcomes;
- real Airavata server status and whether the demo ran;
- whether the full 63-conversation scored evaluation ran;
- run ID and exact artifact paths;
- prompt/schema revision and group-7-only development confirmation;
- measured hardware/runtime settings, if applicable;
- remaining quality limitations;
- explicit confirmation that Stages 1–3 and unrelated files were untouched.

Do not optimize for the appearance of progress. Optimize for verified correctness, minimal diffs, and trustworthy evidence.
