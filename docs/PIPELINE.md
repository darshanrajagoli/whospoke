# How the pipeline works — technical reference

This document explains every stage: what goes in, what comes out, which model does the work,
and why it is built that way. Code references point at `src/whospoke/`.
For measured results see [RESULTS.md](RESULTS.md); for every judgement call see [../DECISIONS.md](../DECISIONS.md).

```
                   ┌──────────────────────────────────────────────────────────────────────┐
 noisy, overlapping│  ORDER B (chosen)                                                    │
 Hindi/Hinglish    │  1. diarize + detect overlaps ─► 2. separate only overlaps ─► 3. ASR │
 recording ───────►│                                                                      │──► timeline.json
 (16 kHz mono)     │  ORDER A (proposal's order, also built and measured)                 │    transcript.txt / _hinglish.txt
                   │  1. separate whole file ─► 2. diarize both tracks jointly ─► 3. ASR  │    transcript.srt / .json
                   └──────────────────────────────────────────────────────────────────────┘
                                                      │
                                                      ▼  4. LLM post-processing (Airavata, local; optional)
                                                   report.md / report.json
                                                   (summary, topic, keywords, action items,
                                                    repaired + translated speaker dialogue)
```

---

## Stage 1 — Blind source separation (`separation.py`)

**Job.** Turn one recording where two people talk at once into two recordings, one per voice.

**Model.** Conv-TasNet (Luo & Mesgarani, 2019), the model named in the proposal. Checkpoint
`JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k` (Asteroid), trained on noisy two-speaker English
mixtures at 16 kHz. It works on raw waveforms: a learned 1-D convolutional *encoder* turns audio
into a feature map, a stack of dilated convolutions (the *temporal convolutional network*)
predicts one *mask* per speaker, and a *decoder* turns each masked feature map back into audio.

**Engineering details that matter.**
- *Gain and sign.* The model is trained with a scale-invariant loss, so each output comes back at
  an arbitrary volume and possibly upside-down (negative sign). We fit one gain per output so that
  the outputs add back up to the input (least squares with a small ridge term) — "mixture consistency".
- *Long audio.* Recordings are processed in 8 s windows with 2 s overlap. The model's two outputs can
  swap order between windows, so each window is aligned to the previous one by cosine similarity on
  the shared 2 s before crossfading. Without this, long outputs scored −3 to −7 dB (speakers swapped).
- *Which checkpoint.* Chosen on the dev set: the noise-trained checkpoint keeps +6 dB in market noise,
  the clean-trained one drops to +2.8 dB (DECISIONS D18). The metadata of these checkpoints claims
  8 kHz, but they are 16 kHz models (D15).

**Metric.** SI-SDR improvement (dB): how much cleaner each separated voice is than the raw mixture.

---

## Stage 2 — Speaker diarization (`vad.py`, `diarization.py`, `clustering.py`)

**Job.** Produce a timeline *[00:12 – 00:45] Speaker_A* without being told who the speakers are or how many.

1. **Voice activity detection.** pyannote `segmentation-3.0` gives, every ~17 ms, the probability of
   silence / one speaker / two speakers ("powerset" output). Frames with ≥ 1 speaker are speech;
   frames with ≥ 2 are *overlap* (used later).
2. **Speaker embeddings.** Speech is cut into 1.5 s windows every 0.75 s. Each window is turned into a
   256-number vector by WeSpeaker ResNet-34 (`pyannote/wespeaker-voxceleb-resnet34-LM`), trained so
   that windows from the same person point in similar directions (cosine similarity).
3. **Unsupervised clustering** — the proposal's two options, both implemented from scratch:
   - *Spectral clustering* (Ng, Jordan & Weiss 2001; recipe of Park et al. 2019): build a cosine
     similarity graph, keep only each window's strongest `p` % of links (pruning removes noisy
     links), symmetrise, compute the graph Laplacian. The number of speakers is where the gap between
     consecutive Laplacian eigenvalues is largest ("eigengap"); k-means on the first `k`
     eigenvectors gives the labels.
   - *Gaussian Mixture Model*: L2-normalise, reduce to a few dimensions with PCA, fit GMMs with
     1…8 components and keep the one with the lowest BIC (Bayesian Information Criterion — rewards
     fit, penalises extra components).
   - *Clean-up* (both): clusters with < 3 % of windows are absorbed by the nearest cluster; clusters
     whose mean embeddings are near-identical are merged (D17). Thresholds were tuned on the dev set.
4. **Timeline.** Each 50 ms speech frame gets the label of the windows covering it (centre-weighted
   vote); speaker flickers shorter than 0.3 s are absorbed; runs of one label become turns.
5. **Overlap-aware assignment.** Where the overlap detector fires, the nearest *other* speaker in time
   is added as a second speaker (Bullock et al., 2020) — otherwise every overlap is counted as missed speech.

*Order A variant* (`Diarizer.assign_streams`): the same steps run on both separated tracks; windows
from both tracks are clustered **together** so a person keeps one name when the separator moves them
between tracks; if the same person is found on both tracks at once (leakage), the quieter copy is dropped.

*Reference system*: `PyannoteDiarizer` wraps the complete off-the-shelf pyannote 3.1 pipeline, to show
how our from-scratch clustering compares with an industry tool.

**Metrics.** DER — the fraction of speech time that is *missed*, *falsely detected* or given to the
*wrong speaker* (0.25 s forgiveness collar at turn edges, overlapped speech included). JER — the same
idea averaged per speaker. Speaker-count error — found vs true number of speakers.

---

## Stage 3 — Regional & code-switched transcription (`asr_backends.py`, `hinglish.py`, `pipeline.py`)

**Job.** Text for every turn, in native script and in Hinglish (Latin script).

**Models.** Both proposal options were run (D9):
- **IndicConformer** (AI4Bharat's current "IndicASR"): a Conformer — convolution + self-attention —
  trained on thousands of hours of Indian-language speech; 600 M-parameter multilingual checkpoint, Hindi decoding.
- **IndicWav2Vec-Hindi**: wav2vec 2.0 self-supervised pre-training on 40 Indian languages, fine-tuned for
  Hindi with CTC (per-frame character prediction).
The choice between them was made on Vaani's real-world test recordings, which neither model saw in
training (IndicConformer was trained on IndicVoices, which our simulated data is built from).

**Which audio each turn is transcribed from.**
- Order B: the overlap regions are those found by the overlap detector, plus any place where two turns overlap.
  Each region is separated once (Stage 1 on the region plus 0.5 s of context). For every turn that touches a region,
  only the overlapped stretch is replaced by the separated voice that sounds most like *this* speaker (highest cosine
  to their voice fingerprint, lowest to the other speaker's), with 20 ms cross-fades. The rest of the turn, and every
  turn without overlap, is transcribed straight from the mixture. Replacing the whole turn instead was worse on dev,
  because separator artefacts spread over speech that was never overlapped (D26).
- Order A: every turn is transcribed from the separated track it was found on.
Turns longer than 25 s are split at the quietest 50 ms near the cut.

**Hinglish.** The ASR writes English words in Devanagari (`होमवर्क`). `hinglish.py` converts:
English loanwords via a lexicon (hand-curated from IndicVoices + mined from Vaani's train-split
annotations, D19/D20) → `homework`; Hindi words by rule with Hindi schwa deletion (`कमरा` → *kamra*) and
chat-style vowels (`बात` → *baat*, `था` → *tha*).

**Metrics.** WER / CER for single utterances; **cpWER** for conversations — each hypothesised speaker's
words are matched to the best reference speaker (Hungarian algorithm) before counting errors, so the
score only rewards the *right words attributed to the right person*. English-word recall on Vaani —
how many of the English words spoken inside Hindi sentences come out exactly right.

---

## Stage 4 — LLM post-processing (`llm_postprocess.py`)

**Job.** Turn the Stage-3 transcript into the proposal's final deliverable: a human-readable report with an executive
summary, topic, keyword tags and speaker-separated dialogue, repaired and translated. Full reference:
[MILESTONE4.md](MILESTONE4.md).

**Model.** AI4Bharat **Airavata**: OpenHathi (Llama-2 7B with an extended Hindi vocabulary) instruction-tuned on Hindi
and English instruction data. Quantised to 4 bits (Q4_K_M, ~4 GB) and served locally by llama.cpp's `llama-server`
(`scripts/serve_llm.py`), with its own chat format and a 4,096-token context window (D27).

**Two calls.**
1. *Per chunk* of ≤ 12 lines (sized to fit the context window): the model sees numbered lines
   (`7 | Speaker_B | 00:06-00:20 | <Devanagari>`) and returns, per line number, the line with only obvious ASR mistakes
   repaired, a faithful English translation and an "uncertain" flag, plus a summary, keywords and explicit actions
   for that part.
2. *Synthesis*: from the part summaries and the English lines, a title, topic, 2–4 sentence executive summary, key
   points, keywords and action items.

**Guardrails** (D28–D30).
- Speaker labels and timestamps are never part of the reply format; they are copied from Stage 3.
- The reply must follow a JSON schema that lists every line number in order (llama-server enforces it with a
  grammar), and is checked again in Python: unknown line numbers are dropped, missing lines restored and flagged; an
  unusable reply makes the chunk be halved and retried, and one line that still fails keeps its ASR text.
- A "repair" that changes more than half of a line's characters is reverted (the model rewrote or translated it).
- Keywords must occur in the transcript or its translations; action items must name a real speaker (or
  "unspecified") and cite real lines.
- The Hinglish column is made by the Stage-3 romaniser from the repaired Devanagari.
Every guardrail's count goes into the report, and each one has a test.

**Metrics** (D31). cpWER of the Stage-3 text vs the repaired text (does the repair help?); share of report keywords
that were really said; word overlap (F1) of keywords and summary with the report made from the true transcript;
time per conversation.

## How errors cascade (proposal objective 5)

`scripts/evaluate.py` runs *oracle* versions of the pipeline that replace one stage with the truth:

| system | diarization | audio given to ASR | isolates |
|---|---|---|---|
| oracle-clean | true | each speaker's clean voice | the ASR's own error |
| oracle-mix | true | noisy overlapped mixture | + damage from noise and overlap |
| oracle-sep | true | mixture, overlaps separated | how much separation repairs |
| B-spectral | ours | mixture, overlaps separated | + diarization errors (full system) |

The differences between consecutive rows are the error contributed by each stage.

`scripts/eval_postprocess.py` carries the cascade into Stage 4: it runs the LLM on the true transcript and on the
oracle-clean, oracle-mix and B-spectral transcripts of the same conversations, and measures how much of each
upstream error survives into the report (keywords that were never said, drift of the summary from the one made
from the true transcript), and whether the LLM repair removes any of it (cpWER before vs after Stage 4).

---

## Outputs of one run (`python -m whospoke run file.wav [--postprocess]`)

| file | content |
|---|---|
| `timeline.json` | the proposal's Stage-2 deliverable: `[{"start": "00:12", "end": "00:45", "speaker": "Speaker_A"}, …]` |
| `transcript.txt` | the Stage-3 deliverable: `[00:12 - 00:45] Speaker_A: …` in Devanagari |
| `transcript_hinglish.txt` | same, romanised |
| `transcript.srt` | subtitles (play the audio with them in VLC) |
| `transcript.json` | everything above + per-stage timings and GPU memory |
| `report.md` | the Stage-4 deliverable (with `--postprocess`): summary, topic, keywords, action items, speakers, repaired and translated dialogue, what was changed and why |
| `report.json` | the same, machine-readable, with every guardrail's count and the SHA-256 of the transcript lines it was made from |
