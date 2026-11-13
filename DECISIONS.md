# Decision Log

Every judgement call made while building the project, recorded **before** acting on it.
Format: what had to be decided → what we chose → why (in plain English).

---

## D1 · Order of Stage 1 (separation) and Stage 2 (diarization)
**Decided:** 2026-09-25
**Choice:** Build **both** orders and let the numbers pick the winner.
- **Order A (proposal):** separate the whole recording first → diarize the separated tracks → transcribe.
- **Order B (alternative):** diarize the original recording first → detect overlap regions → separate only those regions → transcribe.

**Why:** The proposal specifies Order A, but pretrained separators (Conv-TasNet) are trained on clean 2-speaker
lab mixtures and can add artefacts to long, noisy, real recordings. Rather than silently changing the professor's design,
we run both on identical test audio and justify the final choice with measured DER / WER / runtime.

## D2 · Clustering method for diarization
**Decided:** 2026-09-25
**Choice:** Implement **both** Spectral Clustering and GMM (the proposal says "Spectral Clustering *or* GMMs"),
compare them, and also run pyannote's own pipeline as an industry reference point.
**Why:** Costs little extra, and gives a "we evaluated the options" slide instead of an arbitrary pick.

## D3 · Evaluate on synthetic conversations built from real Hindi speech
**Decided:** 2026-09-25
**Choice:** Build test "conversations" by mixing real IndicVoices Hindi utterances from different speakers
(controlled overlap %, added real background noise — originally planned from Vaani's noise dataset, see D11 for the change).
**Why:** Real radio clips have no ground-truth labels, so they cannot be scored. With mixtures we built ourselves, we
know exactly who spoke what and when, so every stage can be measured. Real (unlabelled) audio is still used for a
qualitative demo.

## D4 · Language scope
**Decided:** 2026-09-25 (by the team)
**Choice:** Hindi + English code-switching (Hinglish) only, for the whole project (all four stages). Other Indian
languages (IndicConformer and the corpora cover 22) are a possible extension, not part of this project.

## D5 · Compute
**Decided:** 2026-09-25
**Choice:** Develop and test on the team laptop (RTX 3050, 6 GB); pretrained models only, no fine-tuning of large
models. Provide a Google Colab notebook so teammates can run it in a browser.
**Why:** Lets every run be verified locally; 6 GB is enough for inference but not for training big models.

## D6 · Git
**Decided:** 2026-09-25
**Choice:** Commit locally as work progresses; **do not push** to GitHub automatically. Large data and model files are
git-ignored.
**Why:** Safe and reversible; the team pushes when they are ready.

## D7 · Datasets from the proposal
**Decided:** 2026-09-25
| Corpus | Used? | Reason |
|---|---|---|
| IndicVoices (AI4Bharat) | ✅ Main source | Spontaneous Hindi speech with transcripts + speaker info → builds the test conversations |
| Project Vaani (IISc/ARTPARK) | ✅ | Real-world noisy Hindi (transcribed part) for ASR testing, plus its English-word tags for the Hinglish lexicon. Not used for background noise: too little speech-free audio (D11) |
| Nirantar (AI4Bharat) | ⚠️ ASR cross-check only | Distributed only as one ~198 GB `.tgz` (5 × 39.6 GB parts) with all 22 languages shuffled in one flat folder, so the Hindi files (135.6 h, ~8.5 GB) cannot be pulled out without streaming all of it. Its manifests (checked 2026-09-28) show the Hindi part comes from the same collection as IndicVoices: identical fields, and 165 of the 514 speakers in IndicVoices' Hindi *valid* split (our test speakers, D13) also appear in it. So it would add little new data and risks test-speaker leakage. A teammate streamed it on Colab and kept only Hindi (64,597 clips, 135.6 h, 490 speakers). 923 of those clips are the exact IndicVoices valid files. On an 800-clip sample with our test speakers removed (406 clips), IndicConformer scores 11.0 % WER and IndicWav2Vec 30.2 %, so the model choice (D9) is unchanged. It is reported as a cross-check in RESULTS.md, not used to build test conversations. The full 8.5 GB Hindi set stays on the teammate's Drive and isn't downloaded. We train no model (D5), so more speech from the same collection has nothing to feed into, and 406 clips already separate the two ASR models by 19 WER points. Building a second test set from the ~325 non-test speakers is logged as a possible extension (RESULTS.md, "what we will do next") |
| AIR-RS-DB | ❌ | No public source found under this name (the proposal's citation for it is blank) |

## D8 · Proposal named "Conv-TasNet or Demucs"
**Decided:** 2026-09-25
**Choice:** Conv-TasNet.
**Why:** The widely available Demucs checkpoints separate *music* (vocals vs. instruments), not one speaker from
another. Conv-TasNet is trained exactly for speaker separation.

## D9 · Proposal named "IndicASR or IndicWav2Vec"
**Decided:** 2026-09-25
**Choice:** Run **both** (IndicConformer = AI4Bharat's current "IndicASR" model; IndicWav2Vec-Hindi), keep the better one.

## D10 · Output script
**Decided:** 2026-09-25
**Choice:** Output both the native Devanagari transcript (used for scoring) and a romanised Hinglish version.
**Why:** The proposal allows "native scripts **or** standardized Latin representations"; giving both covers either reading.

## D11 · Background-noise source: DEMAND + ESC-50 instead of Vaani
**Decided:** 2026-09-25
**Tried first:** harvesting speech-free audio from Project Vaani (both the main corpus and the Noise-Event release).
**Finding:** Vaani clips are trimmed tightly around the speaker — ~1 s of speech-free audio per ~500 s — and noise
that happens *during* speech cannot be separated from the voice.
**Choice:** two soundscapes that match the proposal's "ambient village or market noise":
- *village* = DEMAND field/park/home ambience + ESC-50 cows, roosters, hens, dogs, crows, insects, birds
- *market*  = DEMAND traffic/square/station/bus ambience + ESC-50 horns, engines, sirens, trains, bells, footsteps
DEMAND is CC-BY-4.0; ESC-50 is CC-BY-NC-3.0 (fine for coursework). Noise is not one of the proposal's speech corpora,
so this does not change which speech datasets we use. Vaani is still used for real-world speech (see D7).

## D12 · Test-set design: paired conditions
**Decided:** 2026-09-25
**Choice:** each group of 2–3 real speakers is rendered under all 9 conditions:
overlap {none, low ≈3 %, high ≈12 %} × noise {clean, village @ 10 dB SNR, market @ 5 dB SNR}.
**Why:** differences between conditions are then caused by the condition, not by which speakers were drawn.
High-overlap ≈12 % matches lively real meetings (AMI-style corpora report ~10–15 %).

## D13 · Dev (tuning) speakers are disjoint from test speakers
**Decided:** 2026-09-25
**Finding:** IndicVoices' Hindi *valid* split shares 91 speakers with the first *train* shard.
**Choice:** test set = valid split; dev set = train-shard speakers that never appear in valid. All thresholds are tuned
on dev only; test numbers are reported once.

## D14 · Tight reference segments
**Decided:** 2026-09-25
**Finding:** IndicVoices utterance files include leading/trailing silence; counting it as "speech" inflated every
system's missed-speech error by ~7 % (pyannote included).
**Choice:** trim edge silence (>35 dB below the utterance peak, 50 ms kept) before placing utterances.

## D15 · Separator checkpoint & sample rate
**Decided:** 2026-09-25
**Finding:** the Asteroid Conv-TasNet "16k" checkpoints report `sample_rate=8000` in their metadata, but running them
at 16 kHz gives 12.6–14.0 dB SI-SDR improvement on Hindi 2-speaker mixtures vs 5.0 dB at 8 kHz → they are 16 kHz models.
SpeechBrain SepFormer-WHAMR16k scored only ~4.5 dB on the same mixtures (it was trained on reverberant audio).
**Choice:** Conv-TasNet (proposal's model), `sepnoisy` vs `sepclean` checkpoints compared in the Stage-1 benchmark;
SepFormer kept only as a reference row.

## D16 · Separator outputs: per-source gain fit + stitching
**Decided:** 2026-09-25
**Finding:** separators trained with a scale-invariant loss return each voice with an arbitrary gain *and sign*;
naive window stitching then swapped speakers between windows (−3 to −7 dB on 30 s audio).
**Choice:** fit per-source gains so the outputs add back up to the input (least squares, small ridge term), then align
consecutive windows by cosine similarity on their shared 2 s before crossfading.

## D17 · Cluster clean-up step
**Decided:** 2026-09-25
**Finding:** raw spectral/GMM clustering over-counted speakers on 3-speaker and overlapped audio (4–5 found for 3).
**Choice:** after clustering, absorb clusters holding < 3 % of windows into the nearest real cluster, and merge
clusters whose mean voice fingerprints are near-identical (cosine > threshold). Thresholds tuned on the dev set only.

## D18 · Separator checkpoint chosen on dev: Conv-TasNet `sepnoisy`
**Decided:** 2026-09-25 (from `results/separation_dev_summary.csv`)
| SI-SDR improvement (dB) | clean | village 10 dB | market 5 dB | all |
|---|---|---|---|---|
| overlap regions — Conv-TasNet sepnoisy | 7.05 | 5.98 | 6.26 | **6.43** |
| overlap regions — Conv-TasNet sepclean | 8.46 | 4.90 | 2.79 | 5.42 |
| overlap regions — SepFormer WHAMR | 0.26 | 0.25 | 0.47 | 0.33 |
| whole recording — Conv-TasNet sepnoisy | 3.05 | 3.60 | 0.56 | 2.40 |
**Choice:** `sepnoisy` — slightly worse than `sepclean` on clean audio but far more robust in noise, which is the
proposal's setting. **First evidence for the Order question:** the same model helps ~2.7× more when it is only asked
to separate the overlapping stretches (6.4 dB) than when it must process the whole recording (2.4 dB).

## D19 · Choosing the ASR model on Vaani, not IndicVoices
**Decided:** 2026-09-25
**Finding:** IndicConformer was trained on IndicVoices (other utterances of the same speakers), so on IndicVoices-based
audio it has a home advantage over IndicWav2Vec (trained before IndicVoices existed).
**Choice:** pick the Stage-3 model by WER on Project Vaani's transcribed Hindi *test* split — real phone recordings from
across India that neither model was trained on. IndicVoices numbers are reported alongside, with this caveat.

## D20 · Hinglish lexicon: hand-curated + mined from Vaani train annotations
**Decided:** 2026-09-25
**Finding:** Vaani annotators tag every English word spoken inside Hindi (`बिल्डिंग {building}`) — ~8,600 tagged words in
two test shards alone. The hand-curated lexicon (456 entries from IndicVoices' 2,500 most frequent words) spelled 94 %
of the words it contained correctly but covered only 43 % of Vaani's English tokens; the rules alone got 7 % of the rest.
**Choice:** mine additional pairs from Vaani **train** transcripts only (25 of 193 shards, 66k transcripts; test split
untouched for measurement). Keep a pair only if seen ≥ 2×, ≥ 70 % agreement on spelling, tagged English in ≥ 50 % of its
appearances, and not one of the frequent IndicVoices words judged Hindi (blocks e.g. `बस` "enough" → "bus").
Result: +2,147 entries.
**Also decided (tooling):** the official AI4Bharat transliterator (IndicXlit) needs `fairseq`, which does not install on
Python 3.12/Windows; the rule-based romaniser + lexicon is deterministic, testable and runs anywhere.

## D21 · Large data lives outside OneDrive
**Decided:** 2026-09-25 (requested by the team: OneDrive quota is full)
**Choice:** the 4.7 GB of corpora + simulated conversations moved to `C:\whospoke-data`. All code resolves the data folder
through `whospoke.paths.DATA` (`WHOSPOKE_DATA` env var → one-line `data_location.txt` → default `project/data`), so fresh
clones and Colab keep working with no setting. `index.csv` paths are stored relative to the data folder.

## D22 · Diarization settings frozen on dev (re-tuned against exact references)
**Decided:** 2026-09-25
**Finding:** after the references were rebuilt from exact speech activity (D14), the grid search over clustering
settings (spectral p-percentile ∈ {0.1–0.4}, GMM PCA dims ∈ {4, 8, 16}, merge threshold, small-cluster absorption,
overlap-aware assignment) on the **dev** conversations gave:

| system | dev DER | without clean-up (D17) | speaker-count error |
|---|---|---|---|
| Order B · spectral | **0.163** | 0.173 | 0.83 |
| Order B · GMM | 0.184 | 0.277 | 0.83 |
| Order A · spectral | 0.240 | 0.247 | 1.25 |
| Order A · GMM | 0.262 | 0.308 | 1.28 |

**Choice:** freeze the best setting per system in `results/tuned_params.json`; the test set is then run once with
those settings, with no further changes. Order B + spectral clustering is the proposed system; the other three are
reported as comparisons. The clean-up step helps GMM far more than spectral clustering (spectral's eigengap already
estimates the speaker count reasonably).

## D23 · Python environment lives outside OneDrive
**Decided:** 2026-09-25
**Choice:** the virtual environment moved from `project/.venv` to `C:\whospoke-data\venv` (same reason as D21), and is
registered as the Jupyter kernel `whospoke`. Nothing in the code depends on where the environment lives; README gives
the setup steps for a fresh machine.


## D24 · Stage-3 model: IndicConformer (chosen on Vaani, as planned in D19)
**Decided:** 2026-09-25
**Finding:** 400 random single-speaker utterances per set (`scripts/eval_asr.py`, `results/asr_comparison.csv`):

| model | Vaani test WER | Vaani CER | English words recognised | IndicVoices WER | speed (× real time) | GPU peak |
|---|---|---|---|---|---|---|
| **IndicConformer 600M** (RNNT) | **21.0 %** | **10.0 %** | **78 %** of 775 | 13.9 % | 0.054 (18× faster than real time) | 3.7 GB |
| IndicWav2Vec Hindi (CTC, no LM) | 38.0 % | 16.8 % | 48 % of 775 | 36.1 % | 0.007 (135×) | 5.0 GB |

**Choice:** IndicConformer is the default. It wins on the independent Vaani audio by 17 WER points and on English
words spoken inside Hindi by 30 points, so the IndicVoices "home advantage" (D19) does not change the verdict.
IndicWav2Vec stays available (`--asr indicwav2vec`) as the fast option.
**Nirantar cross-check** (added 2026-09-28, `scripts/eval_asr_nirantar.py`): 406 Nirantar Hindi clips with our test
speakers removed give 11.0 % WER for IndicConformer and 30.2 % for IndicWav2Vec, the same ranking. Nirantar's Hindi is the
IndicVoices collection (D7), so it flatters IndicConformer the same way and does not replace Vaani as the selection set.
**Romaniser:** on all 8,630 English words tagged in the two Vaani *test* shards (never used to build the lexicon),
82.0 % of the Devanagari forms come out in the correct English spelling.

## D25 · Crash fix: SpeechBrain's global TorchScript setting vs the IndicConformer preprocessor
**Decided:** 2026-09-25
**Finding:** the first end-to-end test run crashed inside IndicConformer's feature extractor (a TorchScript module)
with an NVRTC compile error (`c10::complex` not found). Cause, reproduced in isolation: `import speechbrain` (pulled in
by pyannote and SepFormer) globally disables TorchScript's profiling executor; the legacy executor then fuses the
complex-valued STFT ops into a runtime-compiled CUDA kernel that does not compile on Windows. The ASR-only benchmark
never imported SpeechBrain, which is why it ran fine.
**Choice:** run the preprocessor with TorchScript's fusers switched off for that call only (`_no_jit_fusion` in
`asr_backends.py`, equivalent to `torch.jit.fuser("none")` but without its per-call deprecation warnings). This changes
speed only; the features are identical.

## D26 · Order B separates the *detected overlap regions*, splicing in only the overlapped stretch
**Decided:** 2026-09-25
**Finding (disclosed: found on the first test run):** in that run, the tuned Order-B diarizer (`overlap_aware=False`,
chosen on dev for DER) never produced overlapping turns, and the code only separated where two *turns* overlapped.
So Order B separated nothing (0 % of turns), which is not the design in D1 ("detect overlap regions → separate only
those"). Separately, with the true timeline, replacing a *whole* overlapping turn by its separated voice made the
transcripts slightly worse (cpWER 31.2 % vs 29.7 % on the plain mixture): the separator's artefacts spread over
speech that was never overlapped.
**Choice:** implement D1 as written. Overlap detection always runs. Order B separates every region where the detector
(or two turns) says two people speak, with some context. Inside that region only, it swaps in the separated voice
that best matches the turn's speaker, with 20 ms cross-fades; the rest of the turn stays the original audio. The
policy (no separation / whole turn / splice with 0.5 s or 1 s context) is chosen on the **dev** set only
(`scripts/tune_separation_policy.py`, `results/separation_policy_dev.csv`), and the test set is then re-run once with
the chosen policy. The first test run's numbers are kept in `results/eval_test_indicconformer_run1.csv` for
transparency.
**Dev result** (36 conversations, cpWER; `results/separation_policy_dev.csv`):

| audio given to the ASR | true timeline | our diarization (B-spectral) | true timeline, heavy overlap |
|---|---|---|---|
| mixture (no separation) | 35.0 % | 46.4 % | 44.1 % |
| whole turn separated (first design) | 37.1 % | 47.2 % | 43.6 % |
| **splice, 0.5 s context** | **31.4 %** | **46.4 %** | **36.2 %** |
| splice, 1.0 s context | 32.1 % | 46.5 % | 38.1 % |

**Chosen:** splice with 0.5 s context (default `sep_context_s=0.5`). It is the best or tied policy on both timelines.
With the true timeline it removes about a fifth of the heavy-overlap errors (44.1 % → 36.2 %). With our own diarization
the gain is small, because diarization mistakes (words credited to the wrong speaker) then dominate the error.

## D27 · Stage 4 runs Airavata locally, behind an OpenAI-compatible server
**Decided:** 2026-10-01
**Choice:** Stage 4 talks to any OpenAI-compatible `/v1/chat/completions` server through the standard library (no
`openai` package, no hosted API, no key). The default is AI4Bharat **Airavata** (7B, Hindi instruction-tuned from
OpenHathi, the proposal's two examples), quantised to 4 bits (Q4_K_M) and served by llama.cpp's `llama-server`
(`scripts/serve_llm.py`), which can build both the server and the model file from PyPI and Hugging Face alone.
**Why:** the proposal asks for a *localized* foundational LLM. The 7B model in 16-bit (~14 GB) does not fit the 6 GB
laptop GPU next to Stages 1–3; at 4 bits (~4 GB) it runs on a CPU, or on the GPU on its own (measured in D33). A separate server process
keeps the 7B model's memory and dependencies out of the Stage 1–3 Python environment, and any other local model can be
swapped in with `--llm-url`.

## D28 · Speaker labels and timestamps never pass through the LLM
**Decided:** 2026-10-01
**Choice:** the model sees numbered lines ("7 | Speaker_B | 00:06-00:20 | text") and replies with text per line number
only: repaired Devanagari, English, an "uncertain" flag. Speaker and time are copied from Stage 3. The reply is
constrained by a JSON schema that lists every line number in order (llama-server compiles it into a grammar, so
only replies of that shape can be generated), and checked again in Python: unknown line numbers are dropped, missing
lines restored and flagged. An unusable reply makes the chunk be halved and retried; a single line that still fails
keeps its ASR text, flagged. The report is still written.
**Why:** Stages 1–3 are measured; Stage 4 is not allowed to undo that. A first version asked the model to return the
speaker and times as well, then overwrote them; never asking for them is simpler and shortens the reply, which is
what costs time on a CPU. A 7B model often produces broken JSON without a grammar. The Python checks still apply when
a server ignores the schema.

## D29 · A repair may change at most half of a line's characters, and may not add, drop or translate words
**Decided:** 2026-10-01
**Choice:** if the repaired line differs from the ASR line in more than 50 % of its characters (character edit
distance after the scoring normalisation, so punctuation does not count), or has more than max(1, 20 %) more words
than the ASR line, the ASR line is kept and flagged. (The word rule was added after an internal red-team check showed
that a short invented clause appended to a long line stays under the 50 % limit.) Two more rules were added after
the real model was run (D32): a repair may not drop words (only an immediately repeated word may go), and may not
remove an English loanword that the line contains (a word in the romaniser's lexicon, e.g. कंट्रोल → नियंत्रण).
**Why:** the proposal asks Stage 4 to "fix syntactic errors caused by background noise drops", i.e. small repairs. A
change of more than half the line is a rewrite: typically the model translated the line into English in the
Devanagari field, or "completed" a fragment with words nobody said. 50 % is deliberately loose for short lines (one
fixed word in a two-word line is a 20–40 % change). How often it fires, and whether repairs lower or raise cpWER, is
measured on the test set (D31).

## D30 · The Hinglish column comes from the Stage-3 romaniser, not from the LLM
**Decided:** 2026-10-01
**Choice:** the repaired Devanagari is romanised by `hinglish.py`, as in Stage 3. An unchanged line keeps its
Stage-3 Hinglish exactly.
**Why:** the reply would otherwise carry each line three times (Devanagari, Hinglish, English), which is about 50 %
more generated text and 50 % more time on a CPU, and the model's Hinglish would be spelled differently from Stage 3's.

## D31 · How Stage 4 is evaluated
**Decided:** 2026-10-01
**Choice:** `scripts/eval_postprocess.py` runs Stage 4 on the stored Stage-3 transcripts of the test conversations
(`results/eval_test_indicconformer/*.json`) for four inputs, each adding one source of upstream error: the true
transcript, ASR on each clean voice (true timeline), ASR on the noisy mixture (true timeline), and the full Order-B
pipeline. Measured per conversation: cpWER of the Stage-3 text vs the Stage-4 repaired text (paired, bootstrap CI);
the share of report keywords, and of summary and key-point content words, that were really said (found in the true
transcript or its translation); word overlap
(F1) of the keywords and summary with the report made from the true transcript; guardrail counts; time.
**Why:** a summary has no single right answer, but three things can be measured honestly: (1) whether the repair
makes the transcript better or worse against the truth, which answers the proposal's "fix syntactic errors" directly;
(2) whether upstream errors put things into the report that nobody said; (3) how far the report drifts from the one a
perfect transcript gives, which is the proposal's objective 5 ("how error propagation cascades from early acoustic
layers down to final text generations") applied to the last stage. Re-using the stored transcripts means the same
conversations as every other result, and no audio model has to run again. No human judgement or second LLM is used as
a judge: both would be harder to reproduce than the numbers above. The guardrails cannot check that a translation or a
summary sentence is faithful; "summary words really said" measures how often unsaid content gets through.
Conversations of one speaker group share their speech across the 9 conditions, so some Stage-3 transcripts are
identical; Stage 4 is deterministic, so the evaluation reuses the report for an identical transcript instead of
recomputing it, and the confidence intervals (bootstrap over conversations) are optimistic.

## D32 · Stage 4 after meeting the real model: what changed, developed on test group 7 only
**Decided:** 2026-10-02
**Context:** the prompts and guardrails of D27–D31 were written before any Airavata output had been seen. The first
real runs showed mechanical failures and quality failures. Mechanical problems were fixed wherever they appeared.
Changes made because of report *quality* were developed only on test speaker group 7 (9 conversations), and group 7 is
excluded from every Stage-4 number reported (`scripts/eval_postprocess.py` never uses it; the Stage 1–3 results are
unchanged).
**Mechanical fixes:**
- *Every JSON-schema request failed.* Current llama-server feeds the template's generation prompt (`<|assistant|>\n`)
  into the grammar, but skips its first token when that token starts with a space. Airavata's tokenizer merges the
  leading space with `<`, so the grammar lost the `<` and rejected everything ("Failed to initialize samplers"). The
  client then silently fell back to unconstrained replies. The chat template now opens the assistant turn after a
  final user message even without `add_generation_prompt`, so the generation prompt is empty; the rendered prompt is
  byte-identical to the model card format. The fallback now warns, and the real-LLM test asserts the schema was used.
- *Model file.* `ai4bharat/Airavata` publishes a 16-bit GGUF whose tokenizer equals `tokenizer.model` (same pieces and
  scores); quantising it directly to Q4_K_M needs no Python conversion step and ~18 GB of disk instead of ~22 GB.
- *Memory.* llama-server's automatic slots share one 4,096-token cache, and it keeps an extra RAM prompt cache of up to
  8 GB by default (it was killed for lack of memory while running three requests). `serve_llm.py` now sets the number
  of slots explicitly (`--parallel`, default 1), each with its own 4,096-token window, and turns that RAM cache off.
**Quality changes (group 7):** in the first reports the "English" column was mostly the Devanagari line copied, repairs
replaced English loanwords with Hindi ones (मैम "ma'am" → माँ "mother", शुगर → चीनी, हॉस्पिटल → अस्पताल) and
deleted words, translations drifted onto neighbouring lines, and the summary described the report instead of the
conversation. On the true transcript of one conversation the "repairs" added 10 cpWER points. Changes:
- the English field is named `english` (with `en`, Airavata often answered in Hindi) and the schema allows only
  printable ASCII in it, so the line cannot be copied; Devanagari in it counts as no translation;
- the chunk prompt says to copy a line unless a word is clearly broken or repeated, never to use synonyms, never to drop
  or add words, and to keep English words written in Devanagari;
- a repair that drops a word (other than an immediate repeat) or removes a loanword is reverted (D29);
- chunks hold at most 6 lines instead of 12: with 12, translations drifted by one or two lines;
- when several different lines get the same English sentence (a degenerate reply), those translations are dropped;
- the synthesis must return at least one key point (it often returned none), and its summary prompt asks for what the
  speakers talk about, not a description of the report.
On the group-7 reports the cpWER change from repair was +1.0 to +2.3 points with the first prompts and −0.4 to +10
points after the first round of changes; with the final version it was −0.5 to 0 points (3 reports), and the English
column became English. It is still a 7B model at 4 bits: many translations stay wrong, summaries are often generic
("a conversation between two people"), and the measured effect is reported as it is (RESULTS.md). The development
reports are not kept in the repository; group 7 is simply not scored.
**Why:** the freeze rule keeps the reported numbers honest: nothing reported was looked at while the prompts were
changed. Group 7 was chosen because it is a three-speaker group, the harder case.

## D33 · Stage 4 on the laptop GPU, scored on every test group except 7, speed never pooled
**Decided:** 2026-10-03
**Context:** groups 0–1 were evaluated on a 4-core cloud CPU (no GPU), three reports at a time. The rest was run on
the team laptop (RTX 3050, 6 GB).
**Choice:**
- *Server.* A prebuilt llama.cpp release (b11312, `win-cuda-12.4`) is used on the laptop through
  `serve_llm.py --llama-bin`. It is the same llama.cpp commit (`0c1e570`) that `serve_llm.py` builds from
  `llama-cpp-python` 0.3.36, so the model file and the server behave as in the cloud run; the laptop has no CMake,
  C++ compiler or CUDA toolkit to build it. The model file is quantised by `serve_llm.py --prepare-only` from the
  16-bit GGUF after checking its SHA-256 against the Hugging Face hash.
- *GPU layers.* All layers on the GPU (`--gpu-layers 99`): with one 4,096-token slot the server uses 5.9 of the
  6.0 GB, and generates 23 tokens/s, against 19.6 tokens/s with 28 layers. A second slot (`--parallel 2`) gave no extra
  throughput (22.7 tokens/s for both together): its 2 GB cache does not fit next to the model. So one request at a
  time. Stage 4 still cannot share the card with Stages 1–3 (4.5 GB), so on this laptop it runs as a separate step
  after Stage 3, or on the CPU.
- *Coverage.* Stage 4 is scored on all 63 test conversations that may be scored (groups 0–6), not only groups 0–3.
  The cached reports of groups 0–3 are reused (Stage 4 is deterministic at temperature 0 on one machine); groups 4–6
  are new.
- *CPU and GPU reports in one table.* Groups 0–1 were reported on the CPU and groups 2–6 on the GPU. The two builds
  give slightly different reports for the same transcript (different floating-point order, then greedy decoding). So
  group 0 was made again on the GPU and compared (`results/crosscheck_g00_gpu/`): 94 % of the repaired lines are
  identical, the repair still adds about one cpWER point (+1.2 on the CPU, +1.4 on the GPU), there are no action
  items on either, and the free-text summaries differ. The conclusions do not depend on the machine, so the scored
  reports are kept as they are.
- *Speed.* Each evaluation run keeps its own summary (`results/eval_postprocess_summary_gXX_..._<machine>.json`), and
  the tables and the benchmark notebook show one row per run. Times from a CPU and a GPU are never averaged together.
**Why:** groups 0–3 happen to be the easier half of the test set (full-pipeline cpWER 39.1 % vs 59.8 % for groups 4–7),
so results from them alone would not describe the test set. On the GPU a report takes about 50 s instead of
several minutes, so the remaining groups cost about 1.5 hours. Group 7 stays excluded (D32). Averaging the speed of two
very different machines would describe neither.

## D34 · The live demo is told how many people are talking
**Decided:** 2026-10-27
**Context:** `demo/live.py` (two people, one laptop microphone) re-runs the pipeline on the recording so far every
~2 s. In a rehearsal with a test conversation, the automatic count jumped between 3 and 6 speakers during the first
30 s before settling on 2. On the full test set, our spectral clustering finds the count exactly in 31 % of the
conversations and says "one speaker" for 13 of the 36 two-person ones.
**Measured** (`scripts/eval_speaker_count.py`, the 72 test conversations, same features and scoring as
`evaluate.py`): with the true count given, DER is 15.2 % instead of 20.3 % (−5.1 points, 95 % CI −8.0 to −2.4).
Keeping the automatic result when it found the given count, and clustering again with the count otherwise, gives
14.8 %. pyannote 3.1 gains as much (18.4 % → 13.1 % given). Details in RESULTS.md, "When the number of people is
known".
**Choice:** the demo has a People setting, 2 by default. With a count set, it keeps the automatic clustering when
that found the count, and otherwise clusters again with the count fixed. "Count them" (the published behaviour) is
one click away. The published results keep the automatic count. The known-count numbers are reported separately.
**Why:** two presenters on stage know they are two, giving the count is an existing option of the pipeline
(`--speakers`), and the clustering still decides who is who and when. The "kept if right" rule rather than always
fixing the count: with a fixed count the clean-up (D17) does not run, so on a short recording a small noisy cluster
can take the second label. Keeping a correct automatic result avoids that, and is also slightly better over the
full test set (14.8 % vs 15.2 %).

## D35 · An English line in the live demo, from a translation model that is not ours
**Decided:** 2026-10-27
**Context:** the live demo may be shown to people who read neither Devanagari nor Kannada. The Stage-4 report
gives English for Hindi only, about two minutes after a take, and not at all for Kannada.
**Choice:** `demo/indictrans.py` puts an English line under each transcript line during the take: AI4Bharat's
IndicTrans2 (Indic → English, distilled 200M) when its gated weights are available, else Meta's NLLB-200 (distilled
600M). It runs on the CPU in a background thread, because Stages 1–3 use 5.98 of the 6.14 GB GPU, and it translates
only lines that have stopped changing (ended at least 3 s before the newest audio), plus every line after Stop. The
side panel names the model and marks it "not ours". It is a presentation aid only: nothing in the pipeline, the
evaluation or the report uses it.
**How:** IndicTrans2's preprocessing (normalisation, placeholders, transliteration into Devanagari) is AI4Bharat's
IndicTransToolkit (MIT), which ships as a Cython extension, and this laptop has no C compiler. So its Indic-to-English
part is reproduced in plain Python, with attribution. Both models publish `pytorch_model.bin` only, which transformers
refuses to load with torch 2.5 (CVE-2025-32434). So the official file is read once with torch's safe loader and
saved as safetensors under the data folder. Community safetensors uploads are not used.
**Why:** the English line lets everyone follow a Hindi or Kannada take without waiting for the report. Keeping it
off the GPU and off the pipeline's critical path costs the live passes about half a second. In a rehearsal take,
9 of 15 lines had English before Stop and all 16 by the end of the final pass.

## D36 · Stage-4 reports are cached by transcript and prompt revision
**Decided:** 2026-11-10
**Context:** `scripts/eval_postprocess.py` reuses the report of an identical transcript (Stage 4 is deterministic at
temperature 0). The cache was keyed on the transcript alone, so after an edit to a prompt or a schema the old
reports would still have been reused and scored as if they came from the new prompts.
**Choice:** `llm_postprocess.prompt_schema_revision()` hashes everything in the module that changes the model's output
without changing the transcript: the four prompts, every JSON schema, `MAX_LINES_PER_CHUNK`, `MAX_EDIT_RATIO`,
`UNSPECIFIED` and `VALIDATOR_RULES_VERSION` (D38). The 12-character id (`PROMPT_SCHEMA_REVISION`) is written into
every report's provenance and its Markdown, and into the run summary. The evaluator keys its cache on it, so an edit
to a prompt or schema makes the cached reports stale and they are made again. A report written before the id existed
is accepted as it is, so this change alone does not regenerate anything.
**Why:** an evaluation must score reports made by the code it describes. Tests:
`test_prompt_schema_revision_is_stable_and_recorded`, `test_revision_changes_when_output_affecting_config_changes`,
`test_revision_matches_gates_cache_reuse`.

## D37 · Action items: checked on group 7, prompts left as they are
**Decided:** 2026-11-10
**Context:** no action item survives in any of the 252 evaluation reports (audit finding F2). Either the model proposes
none, or the checks drop them. This was tested with the real Airavata server on three group-7 conversations only
(D32), recording what the model proposes before the checks.
**Finding:** with the current prompts the model proposes no action items on any of the three, so the checks drop
nothing: they are not the cause. With stronger wording the model does propose items, but bad ones: an owner that is not
a speaker (`"Madam"`) and a question that the word-overlap check accepts (`"Motorcade is moving, what is it?"`).
**Choice:** no prompt change. A test now pins the prompts to "explicit requests, instructions or promises only", so a
later edit cannot quietly widen what counts as an action.
**Why:** an empty action list is honest; a full one made of questions and suggestions is not. The limit is the 7B
model's precision, not the prompt or the checks. Evidence: `results/stage4_probe/actions_g07/` (`findings.json`, the
two probe scripts and their logs).

## D38 · The cache also checks model and context; questions and suggestions are not action items
**Decided:** 2026-11-10
**Context:** D36 covers the prompts, but the model and its context length also change a report. The group-7 probe
(D37) also showed the action check accepting a question.
**Choice:**
- *Cache.* The evaluator's key is (transcript, `PROMPT_SCHEMA_REVISION`, model, context, input), and
  `cache_compatible()` gates every reuse. Each report records `context_tokens`. A report from before these fields
  existed is reused only for the default model and the default context (4,096), the only setting it can have been
  made with. The model is identified by its name (`--llm-model`); the server does not expose the file's hash, so a
  different file under the same name needs `--force`.
- *Action items.* `_actions` now also drops information questions ("…, what is it?"), suggestions and hypotheticals
  ("Maybe…", "We should…") and first-person wishes ("I want…"). It keeps instructions ("Bring…"), promises
  ("… will …"), items with an unspecified owner, and polite requests ("Will you…?"). This raised
  `VALIDATOR_RULES_VERSION` to 2; the current revision is `b03fa93a833a`. Known gap: a request phrased as a need
  ("I need you to bring the file") is dropped too. Fixing it would change the revision and remake all 252 reports,
  and the model proposes almost no action items anyway (D37), so it is left as it is.
**Why:** cached reports are reused only when nothing that changes their content has changed. An action list should
hold only things someone was asked or promised to do. Tests: `test_cache_compatible_*`,
`test_actions_reject_information_questions_suggestions_and_wishes`,
`test_actions_keep_explicit_requests_commitments_and_polite_questions`.

## D39 · All 252 Stage-4 reports made again, on one machine, with the current code
**Decided:** 2026-11-11
**Context:** the scored reports were made in October, before D36–D38, and on two machines (groups 0–1 on a cloud CPU,
groups 2–6 on the laptop GPU; D33).
**What ran:** `python scripts/eval_postprocess.py --force` (run `2026-11-10_s4eval_r1`): 63 conversations × 4 inputs =
252 reports, groups 0–6 (group 7 excluded, D32), revision `b03fa93a833a`. The model was Airavata Q4_K_M on llama.cpp,
on a Linux PC, CPU only (`--gpu-layers 0 --ctx 4096 --threads 8 --parallel 1`), at temperature 0, one report at a
time. It took 14.4 hours (51,837 s), and the server's peak memory was 5.0 GB. The run started on 10 November and ended on
11 November, so the reports carry both dates. All 252 record the revision and context. Re-scoring them with the code in
this repository reproduces `results/eval_postprocess.csv` exactly, and a new evaluation run would remake none of them. The
earlier reports, their CSV and their three run summaries are in git at commit `df497d8`.
**Result:** the conclusions are unchanged. cpWER after the repair moved by at most 0.1 points; the report measures
moved by a few points (keywords really said, full pipeline: 60 → 64 %; summary words really said, true transcript:
50 → 56 %).
- cpWER after the repair: true transcript 0.0 → 1.9 %, clean voices 18.0 → 19.0 %, noisy mixture 30.6 → 31.3 %,
  full pipeline 51.8 → 52.7 %. The repair still adds errors on every input.
- Still no action items. The model proposed 4, all on the full pipeline, and all were dropped.
- The run's timing summary is `results/eval_postprocess_summary_g00_…_g06_linux-cpu.json`, and the run's command,
  configuration and log are in `results/stage4_probe/2026-11-10_s4eval_r1/`.
**Why:** all scored reports now come from one machine, one build and the code in the repository. So the CPU-vs-GPU
caveat of D33 (and `results/crosscheck_g00_gpu/`, which compares the October reports) no longer applies to them.

## D40 · Other Indian languages run, but only Hindi is evaluated
**Decided:** 2026-11-12
**Context:** D4 keeps the project to Hindi and English. IndicConformer and IndicVoices cover 22 Indian languages, and
the live demo is also run in Kannada.
**Choice:** `src/whospoke/languages.py` lists the 22 languages: each one's IndicConformer code, IndicVoices folder,
script and Latin name, and its IndicWav2Vec model where AI4Bharat publishes one (Hindi, Bengali, Gujarati, Odia).
`--language` (the CLI, `Pipeline`, `make_asr`, `PostProcessor`, `scripts/eval_asr.py`, `scripts/build_dataset.py`)
picks one; Hindi stays the default. `romanise` detects the script and writes any Indic script in Latin letters
(Kannada `ನಾಳೆ ಮೀಟಿಂಗ್ ಇದೆ` → `naale meeting ide`); the lexicon of English words exists for Hindi only. For Hindi
nothing changes: the romanised test transcripts, the Stage-4 prompts and their revision (`b03fa93a833a`) are
identical, re-scoring the 252 Stage-4 reports gives the same CSV, and `eval_asr.py` runs for Hindi as before (the
romaniser's English-spelling accuracy is still 82.0 %).
**Not evaluated:** D4 stands. There is no WER, DER or report score for any other language, and Stage 4 uses
Airavata, a Hindi model: it runs on another language but nobody has checked the result, and the CLI warns. The
IndicWav2Vec models for Bengali, Gujarati and Odia are gated on Hugging Face (accept their terms first); Tamil and
Telugu have none, so `--asr indicwav2vec` refuses them.
**Why:** it changes nothing for Hindi and lets Stages 1–3 (and the demo's Kannada lines) work on other languages.
Saying more than "it runs" would need test sets in those languages.

## D41 · The interrupter's separated voice (the second output) is not used: it lost on dev
**Decided:** 2026-11-12
**Evidence that motivated it:** with the true timeline, splicing separated overlaps lowers dev cpWER by 3.5 points
(35.0 → 31.4 %); with our Order-B diarization by nothing (46.4 → 46.4 %). The tuned Order-B diarizer has
`overlap_aware=False` (D22), so an overlap region usually lies inside one turn only, of the speaker already talking.
`_targeted_separation` gives that speaker the separated output that matches them and **discards the other output**,
so the interrupting speaker's overlapped words never reach the transcript. On dev, 19.5 % of turns get separated
audio with our diarization vs 50.2 % with the true timeline.
**Change tried:** `Pipeline(second_voice=<threshold>)` (`_separate_units` in `pipeline.py`). Per region, the output
that no turn used is scored contrastively, as in `pick`: cosine to a speaker's centroid minus cosine to the centroid
of the speaker who already has the region. If the best-scoring speaker has **no turn in that region**, the region
(≥ 0.3 s) becomes an extra turn for them, transcribed from that output; a speaker who already has a turn there never
gets one, so no words are counted twice. Unit tests: `tests/test_second_voice.py`.
**Dev result** (36 conversations, 5,687 reference words, cpWER; `scripts/tune_separation_policy.py --policies mixture
splice-0.5 splice-0.5+second --out separation_policy_dev_second.csv` → `results/separation_policy_dev_second.csv`;
every candidate turn with its score and text in `results/second_voice_candidates_dev.csv`). Run on the CPU (Ryzen 7
5700U); the mixture and splice-0.5 rows reproduce D26's GPU numbers to 0.1 point:

| policy | our diarization (B-spectral) | extra turns added | true timeline | extra turns added |
|---|---|---|---|---|
| mixture (no separation) | 46.40 % | — | 34.97 % | — |
| **splice-0.5 (current)** | **46.42 %** | — | **31.44 %** | — |
| + second voice, threshold −1 (all candidates) | 46.98 % | 76 | 32.04 % | 37 |
| + second voice, threshold 0.0 | 46.70 % | 51 | 31.55 % | 7 |
| + second voice, threshold 0.1 | 46.62 % | 35 | 31.49 % | 3 |
| + second voice, threshold 0.2 | 46.49 % | 22 | 31.46 % | 1 |
| + second voice, threshold 0.3 | 46.46 % | 9 | 31.44 % | 0 |
| + second voice, threshold 0.4 | 46.46 % | 2 | 31.44 % | 0 |

With our diarization, every threshold is worse than splice-0.5, and the higher the threshold the closer it gets to
splice-0.5, because fewer extra turns are added. At threshold −1 it is worse in 18 conversations and better in 5
(+32 errors). At 0.3 it is worse in 3 and better in 2 (+2 errors). By overlap level, at 0.3: heavy overlap
50.07 → 50.02 %, low 46.02 → 46.07 %, none 42.10 → 42.22 %.
**Why it does not help (from the candidates file, checked against the true overlaps in `reference.json`):**
- the candidates are tiny: median 0.58 s, median **one** recognised word, 9 of the 76 empty. A detected region is a
  fragment of the interrupter's speech, often starting or ending mid-word, so the ASR gets little to work with;
- **33 of the 76** (B-spectral) contain no true overlap at all: the detector fired on single-speaker speech, so the
  "second voice" is leakage or artefact, and its words are insertions;
- only 55 % of candidate time is true overlap; for the highest-scoring candidates (score ≥ 0.2) it is 68 %, so the
  score does pick better candidates but not well enough to win;
- with the true timeline the interrupter already has a turn in almost every real overlap, so the candidates that
  remain are nearly all false detections (19 % true-overlap time): adding them can only hurt there, and it does
  (worse in 15 conversations, better in 0, at threshold −1).
**Choice:** keep splice-0.5 as it is. `second_voice` stays in the code **off by default** (`None`), so this dev
experiment can be re-run, but no pipeline, script or demo turns it on. **The test set was not run**, as planned: the
rule was to run it once only if the change won on dev.
**What would be needed to make the idea work** (not done): longer extra turns, joined to the interrupter's adjacent turn
when there is one, so the ASR hears whole words; and a better overlap detector, or the true overlap count, since 43 %
of candidates are false detections. The bigger lever remains the diarization itself (RESULTS.md, "What limits the system", item 1:
the speaker count).

## D42 · Extension to music: a voice + mridangam duo, where separating first wins
**Decided:** 2026-11-13
**Why:** the professor asked for the pipeline to be extended to music datasets. As for speech, only material that lets
us build mixtures with an exact answer key can be scored, so the extension uses two *solo* datasets from our music list
(docs/MUSIC_EXTENSION_PLAN.md): **Carnatic Varnam** (28 sung varnams, 6 singers, adi tala, tala-cycle annotations; audio
CC BY 3.0, Zenodo 7726167) and **Mridangam Stroke** (7,162 single strokes, 10 types, 6 tonics; CC BY-NC 4.0, Zenodo
1265188). RaagaDhvani v3, the first choice, lists no files and is CC BY-ND 4.0.
**Data** (`src/whospoke/synth_music.py`, `scripts/build_music_dataset.py`): phrases of 1–2 annotated tala cycles; strokes
placed on those cycles' beats (resonant strokes on beats 1, 5, 7; 2 or 4 per beat; rests, fills); drum tuned to the tonic
nearest the singer's; accompaniment probability sets the overlap level (none / low / high); the speech set's noise.
Paired design as in D12: 4 dev groups (36 pieces, singers Vignesh and Dharini) and 8 test groups (72 pieces, the other
4 singers); odd vs even stroke takes and dev vs test noise halves also kept apart.
**Systems:** Stage 1: Conv-TasNet, HPSS (librosa median-filter separation), Demucs `htdemucs` (vocals+other → melody,
drums+bass → drum), behind the existing `Separator`. Stage 2: the speech diarizer as is; with `activity="energy"` and
`fingerprint="timbre"` (new options, off by default); Order A with joint clustering; and Order A "direct", where a music
separator's fixed outputs give each player's activity. One setting (the direct systems' activity threshold) was chosen on
dev from 20–40 dB: 20 dB for both, the edge of the grid. Test was run once.
**Test result, Stage 1** (SI-SDRi; `results/separation_test_music*.csv`; 143 overlap regions, 48 whole pieces):

| separator | overlaps | whole pieces | overlaps improved |
|---|---|---|---|
| **Demucs** | **+8.86 dB** | **+9.50 dB** | 100 % |
| HPSS | +4.43 dB | +4.08 dB | 99 % |
| Conv-TasNet | −0.62 dB | −2.50 dB | 46 % |

Demucs − HPSS +4.42 dB (95 % CI +3.86 to +5.00, paired over regions). Dev gave the same ranking (+6.30 / +3.54 / +0.75).
**Test result, Stage 2** (DER; `results/music_diarization_test.csv`):

| system | DER | missed | false alarm | wrong player | 2 players found |
|---|---|---|---|---|---|
| **Order A, Demucs, direct** | **11.7 %** | 5.3 % | 5.6 % | 0.8 % | 100 % |
| Order B, timbre fingerprints | 30.5 % | 16.1 % | 4.2 % | 10.2 % | 0 % |
| Order A, Demucs + timbre clustering | 37.5 % | 28.3 % | 3.9 % | 5.2 % | 8 % |
| Order A, HPSS, direct | 38.0 % | 8.6 % | 28.7 % | 0.7 % | 100 % |
| Order B, speech diarizer as is | 46.0 % | 40.0 % | 0.3 % | 5.7 % | 14 % |

Demucs-direct − Order-B timbre: −18.8 points (95 % CI −22.8 to −14.8 over pieces; −23.1 to −13.2 resampling groups).
Dev: 15.4 % vs 28.2 %.
**Conclusions:** (1) the music separator is right for music as the speech separator is for speech (D8 from the other
side); (2) for music the best order is the opposite of speech: **separate first**. Overlap is the norm in a concert and a
one-label-per-frame timeline cannot hold two players (Order-B DER rises from 17.5 % with no overlap to 46.6 % with heavy
overlap), while a music separator's outputs already say who is playing; (3) the from-scratch timbre fingerprint beats the
voice-trained WeSpeaker on the mixture. **Limits:** simulated, rule-based drumming, voice-led in no/low overlap; 6 singers;
the threshold sits at the grid's edge; the separator wrapper's window stitching can swap fixed outputs (rare: 0.8 % wrong
player); no transcript (cpWER) for music. No speech result changes.
