# Stage 4 — LLM post-processing (Milestone 4)

Stage 4 turns the raw Stage-3 transcript into the proposal's final deliverable: *"a polished, human-readable
transcript report featuring a concise executive summary, keyword tags, and speaker-separated dialogues."* It uses
AI4Bharat's **Airavata** (the proposal's example of a "localized foundational LLM"), run locally by llama.cpp.
Results are in [RESULTS.md](RESULTS.md#stage-4--llm-post-processing); the decisions are D27–D31 in
[DECISIONS.md](../DECISIONS.md).

## Quick start

```bash
python scripts/serve_llm.py                              # terminal 1: the LLM server (first run builds the model, see below)
python -m whospoke postprocess results/demo/transcript.json   # terminal 2: report for an existing Stage-3 run
python -m whospoke run my_recording.wav --postprocess         # or all four stages in one go
```

Both commands write `report.md` (for people) and `report.json` (for programs) next to `transcript.json`.
An example is [results/demo/report.md](../results/demo/report.md).

## What the report contains

| section | where it comes from |
|---|---|
| title, topic, executive summary, key points | the LLM, from the checked translations and part summaries |
| keywords | the LLM, **kept only if every content word occurs** in the transcript or its line translations |
| action items (who, what, which line) | the LLM; **kept only if** the owner is a real speaker (or "unspecified"), the cited lines exist and share a word with the action |
| speakers: talk time and number of turns | computed from the Stage-2 timeline (no LLM) |
| dialogue: time, speaker, repaired Devanagari, Hinglish, English | speaker and time from Stage 2 unchanged; Devanagari repaired and translated by the LLM; Hinglish by the Stage-3 romaniser |
| what Stage 4 changed | every line where the repaired text differs from the ASR text, side by side |
| uncertain lines | every line the model flagged, skipped or over-edited, with the reason |
| how this report was made | model, number of calls, time, and what each guardrail did; SHA-256 of the source lines |

## How it works

```
transcript.json (Stage 3)
  │  drop lines with no recognised words; number the rest by their position in transcript.json
  ▼
chunks of ≤ 12 consecutive lines, sized so that prompt + expected reply fit Airavata's 4,096-token window
  │
  ▼  call 1 per chunk — "repair, translate, flag; summary, keywords, actions for this part"
  │      input : "7 | Speaker_B | 00:06-00:20 | बाबू माशंकर इंटर कॉलेज के तो बी ए का पेपर ..."
  │      output: {"lines": [{"id": 7, "text": …, "en": …, "uncertain": false}, …], "summary": …, "keywords": […], "actions": […]}
  │      (a JSON schema lists every line number in order; llama-server turns it into a grammar)
  ▼
checks in Python (below) ─► if the reply is unusable: halve the chunk and retry; a single line that still fails keeps its ASR text, flagged
  │
  ▼  call 2 — "title, topic, summary, key points, keywords, actions" from the part summaries + English lines
checks in Python ─► report.json + report.md
```

**Why speaker labels and times never go through the model.** The model is shown them (they help it understand who
answers whom), but its reply format has no place for them: it returns text per line number only. Everything else
is copied from Stage 3, so the timeline in the report cannot be changed by the LLM (D28).

**Why the Hinglish column does not come from the LLM.** Asking a 7B model to transliterate as well would add a
third copy of every line to its reply (slower) and give spellings inconsistent with the Stage-3 Hinglish. The repaired
Devanagari goes through the same romaniser as Stage 3, so both columns agree (D30).

## Guardrails

| what can go wrong | what the code does | counted as |
|---|---|---|
| reply is not JSON, or is cut off at the token limit | halve the chunk and retry; one line that still fails keeps its ASR text, flagged | `invalid_replies`, `chunk_splits`, `failed_chunks` |
| prompt does not fit the context window, or the server times out | same as above | `chunk_splits` |
| a line number that does not exist | ignored | `unknown_line_ids` |
| a line is missing from the reply | restored from Stage 3, flagged | `lines_restored` |
| the "repair" rewrites or translates the line (> 50 % of characters changed) | reverted to the ASR text, flagged | `lines_reverted` (D29) |
| the "repair" adds words (more than 1, or 20 % of the line) | reverted to the ASR text, flagged | `lines_reverted` (D29) |
| the English is much longer than the line (> 2 × its length + 30 characters) | kept, flagged: it may add things that were not said | `translations_too_long` |
| no English translation | flagged; the English column stays empty | `missing_translations` |
| the model says a line is too garbled | kept, flagged | `lines_flagged_by_llm` |
| a keyword with a content word that was never said (common words such as में, है, haan do not count) | dropped | `keywords_dropped` |
| an action with an unknown owner, no valid evidence line, or no word in common with its evidence lines | dropped | `actions_dropped` |
| a long recording: too many part summaries for one call | consecutive part summaries are merged by the LLM, in rounds, until they fit | — |
| the final (synthesis) call fails | summary (merged part summaries), keywords and actions are taken from the per-part results | `synthesis_failed` |
| the server is still loading the model (HTTP 503) | waited for (up to 2 minutes when checking, 30 s per request) | — |
| the server is not running, or the connection drops | stops with a message telling you how to start it; `run --postprocess` checks this before the audio stages, and saves Stages 1–3 even if Stage 4 fails later | — |

All counters are in `report.json → diagnostics` and in the last section of `report.md`.
`tests/test_llm_postprocess.py` tests each row with scripted model replies and a fake HTTP server.

## The prompts

They are in `src/whospoke/llm_postprocess.py` (`SYSTEM_PROMPT`, `CHUNK_PROMPT`, `SYNTHESIS_PROMPT`). The system
prompt is three sentences: you clean up ASR transcripts of Hindi/Hinglish conversations recorded in noisy places;
the transcript is the only source of truth, never add facts, names, numbers, dates, places, events or actions; reply
with JSON only. The chunk prompt explains the numbered-line format and asks, per line, for a minimal repair ("a
repeated word, a broken word, missing punctuation… keep English words that were spoken in English… if you are not
sure, copy the line unchanged"), a faithful English translation and an `uncertain` flag. The prompts are short and
concrete because a 7B model follows short instructions better than long rule lists. Temperature is 0, so the same
transcript always gives the same report.

## Running the model

`scripts/serve_llm.py` starts llama.cpp's `llama-server` with Airavata on `http://127.0.0.1:8080/v1`:

- **Server program.** Taken from `--llama-bin`, else from PATH (a llama.cpp release; `brew install llama.cpp`;
  `winget install llama.cpp`), else built once from the llama.cpp sources that the `llama-cpp-python` package on PyPI
  bundles (needs CMake and a C++ compiler). `--build cuda` builds it with CUDA.
- **Model file.** `models/airavata-q4_k_m.gguf` (~4 GB). On the first run, the script downloads `ai4bharat/Airavata`
  from Hugging Face (~14 GB), converts it with llama.cpp's converter to an 8-bit GGUF and quantises that to 4 bits
  (Q4_K_M), deleting the intermediate files (~22 GB of free disk needed once). `--gguf` serves an existing file.
- **Chat format.** Airavata's own (`<|system|>`, `<|user|>`, `<|assistant|>`, from its model card), passed as
  `src/whospoke/resources/airavata_chat_template.jinja`.

**On the 6 GB laptop GPU.** The 4-bit weights are about 4 GB, and the cache for a full 4,096-token window adds about
2 GB (7B Llama-2 architecture: 32 layers × 4,096 dimensions × 2 × 4,096 tokens × 2 bytes). Stages 1–3 peak at 4.5 GB.
So the model cannot share the card with Stages 1–3, and does not fully fit on it alone. Two ways to run it:
- keep it on the CPU (the default, `--gpu-layers 0`): slower, nothing to tune;
- run Stage 4 as a separate step after Stage 3 has released the GPU (`run` without `--postprocess`, then
  `whospoke postprocess`), with part of the model on the GPU, e.g. `serve_llm.py --gpu-layers 20`; raise the number
  until llama-server reports that it runs out of GPU memory, then step back.

Speed and memory measured on CPU are in RESULTS.md and [notebooks/01_benchmark.ipynb](../notebooks/01_benchmark.ipynb) §7.

Any other OpenAI-compatible server also works (`--llm-url`, `--llm-model`; e.g. OpenHathi, the proposal's other
example, or a larger model). A server that does not support JSON schemas is detected automatically; the Python checks
still apply.

## Settings

| flag (`run` and `postprocess`) | default | meaning |
|---|---|---|
| `--llm-url` | `http://127.0.0.1:8080/v1` (`WHOSPOKE_LLM_URL`) | server address |
| `--llm-model` | `ai4bharat/Airavata` (`WHOSPOKE_LLM_MODEL`) | model name sent to the server and written in the report |
| `--llm-context` | 4096 (`WHOSPOKE_LLM_CONTEXT`) | the model's context window; chunks are sized to fit it |
| `--llm-timeout` | 900 s (`WHOSPOKE_LLM_TIMEOUT`) | per reply; a timed-out chunk is halved and retried |
| `--llm-no-schema` | off | do not send the JSON schema |

## How Stage 4 is evaluated

`scripts/eval_postprocess.py` runs Stage 4 on the stored Stage-3 transcripts of the test conversations
(`results/eval_test_indicconformer/*.json`, no audio models needed) for four inputs, each with one more source of
upstream error: the true transcript, ASR on each clean voice, ASR on the noisy mixture, and the full pipeline. It
measures whether the repair lowers or raises the who-said-what error (cpWER before vs after), how many of the report's
keywords and summary words were really said, how close the report is to the one made from the true transcript, what each guardrail did,
and how long it takes. `scripts/make_report.py` turns that into the Stage-4 tables in
[RESULTS_TABLES.md](RESULTS_TABLES.md) and `results/figures/stage4_cascade.png` (D31).

## Files

| file | role |
|---|---|
| `src/whospoke/llm_postprocess.py` | prompts, HTTP client, chunking, checks, report (JSON + Markdown) |
| `src/whospoke/pipeline.py` | runs Stage 4 after Stage 3 when the pipeline has a post-processor |
| `src/whospoke/__main__.py` | `run --postprocess` and `postprocess` commands |
| `src/whospoke/resources/airavata_chat_template.jinja` | Airavata's chat format for llama-server |
| `scripts/serve_llm.py` | builds the server and the 4-bit model, and starts it |
| `scripts/postprocess.py` | same as `python -m whospoke postprocess` |
| `scripts/eval_postprocess.py` | the Stage-4 evaluation on the test set |
| `tests/test_llm_postprocess.py` | guardrail, client and pipeline tests (`pytest -m slow` also tests the real server) |
