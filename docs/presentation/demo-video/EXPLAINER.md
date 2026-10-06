# Flash talk explainer: every slide, every number

For Vismay and Darshan. This explains everything in the deck we submitted ([vismAI.pptx](vismAI.pptx)): every label,
number and clip, what it means and where it comes from. Rehearse from [SCRIPT.md](SCRIPT.md); read this once so no
line on a slide is a surprise.

## The talk in one minute

People talk over each other in Hindi and English with a market in the background. whospoke works out **who said
what, and when**. The talk does three things, one per slide:

1. **Slide 1 (Vismay): what it is, who built what, and what it sounds like.** An 18 s clip of a noisy test
   conversation plays while whospoke's timeline and transcript appear. Then the two voices from the overlapping part,
   pulled apart. The four rows are the four milestones in the order the pipeline runs them, with their owners.
2. **Slide 2 (Darshan): does it work?** The bars split the error into its sources: recognition alone 17.7 %, noise and
   overlap take it to 29.7 %, working out who is speaking takes it to 49.5 %. The proposal's order does worse
   (55.8 %). The four cards give one number per milestone.
3. **Slide 3 (Darshan): the report, what we learned, what's next.** The LLM writes a readable report but can't fix
   the transcript, and earlier mistakes show up in it. The error breakdown says what to improve first.

Tone throughout: measured numbers, honest about limits. Always "error", never "accuracy"; no "state of the art".

---

## Slide 1: whospoke (Vismay)

### Left side

| On the slide | What it means | Source |
|---|---|---|
| "Speaker-attributed transcription" | A transcript where every line has a speaker and a time, not just the words | The proposal |
| "4 / 4 milestones complete" | Separation, diarization, transcription and the LLM report are all built, measured and documented | [README](../../../README.md), "Headline results" |
| "The pipeline, in order" | The order the pipeline actually runs: M2 first, then M1 only on the overlaps, then M3, then M4. The proposal's order is M1 → M2 (see "Order A and B" below) | D1, D26 in [DECISIONS.md](../../../DECISIONS.md) |
| M2 · "Speaker embeddings + our own clustering" | pyannote finds where anyone is speaking and where two people are. Every 1.5 s window of speech becomes a 256-number voice fingerprint (WeSpeaker ResNet-34). Spectral clustering and a GMM, both written from scratch, group the fingerprints into speakers without knowing how many there are | [PIPELINE.md §2](../../PIPELINE.md), `clustering.py` |
| M1 · "Conv-TasNet, only where voices overlap" | A neural network that splits audio into one track per speaker. It is run only on the stretches where M2 found two voices | [PIPELINE.md §1](../../PIPELINE.md) |
| M3 · "IndicConformer + Hinglish romaniser" | AI4Bharat's 600 M-parameter speech recogniser writes Devanagari; our romaniser turns it into Hinglish (Latin script) with a 2,600-word English-loanword list, so "बिल्डिंग" comes out as "building" | [PIPELINE.md §3](../../PIPELINE.md), [DATASETS.md §3](../../DATASETS.md) |
| M4 · "Airavata 7B LLM, 4-bit, on a laptop GPU" | AI4Bharat's 7-billion-parameter Hindi chat model, shrunk to 4 bits per weight (~4 GB) so it fits on the 6 GB RTX 3050, run with llama.cpp | [MILESTONE4.md](../../MILESTONE4.md) |
| Names | The work split: Vismay M1, Darshan M2, Abhinav M3, Shrivaths M4. The FAQ asks for a contributions statement; this slide is it | — |

### The big clip (18 s)

| | |
|---|---|
| Audio | Test conversation `test_g02_2spk_ovl-high_market5`, 00:26.4–00:44.8: two speakers, heavy overlap, market noise at 5 dB SNR (the noise is nearly as loud as the speech). Unedited apart from a 0.3 s fade-in and 0.5 s fade-out |
| "Test conversation … real Hindi speech and real market noise" | The speech is two real people's sides of real IndicVoices phone calls; the noise is real field recordings (DEMAND + ESC-50). We mixed them, which is why we know the right answer ([DATASETS.md §1](../../DATASETS.md)) |
| Timeline bars | Speaker turns from `results/demo/timeline.json` (M2's output), drawn up to the playhead. Orange = Speaker A, blue = Speaker B |
| Transcript | Hinglish lines from `results/demo/transcript.json` (M3's output), each shown when its line starts |
| "whospoke's own output" | Nothing is hand-edited. [make_demo_video.py](make_demo_video.py) replays the stored output in time with the audio. Say "what whospoke wrote", never "live" |
| Why this conversation | Its error is close to the test-set median: a typical case, not the best one |

### The two separated voices (7.5 s each)

| | |
|---|---|
| "00:28–00:35, both talking at once" | The stretch of the clip where both people speak at the same time |
| "pulled apart by Conv-TasNet (M1)" | M1's separator run on the recording (as `python -m whospoke separate` does), then cut to that stretch. Each track is labelled with the M2 speaker whose voice fingerprint it matches, so the colours match the big clip |
| How good | Checked against the true voices: SI-SDR +9.4 and +8.2 dB for the separated tracks, against +1.6 and −3.9 dB for the raw mix. The true voices are used only for this check |
| Made by | [make_speaker_clips.py](make_speaker_clips.py) |

### Order A and Order B (Vismay's last line)

- **Order A (the proposal):** separate the whole recording first, then diarize the separated tracks.
- **Order B (ours):** diarize the original recording first, then separate only where two voices overlap.

That is our one design change from the proposal. Slide 2 shows it wins, which is why Vismay hands over with "Darshan
will show you why".

---

## Slide 2: results (Darshan)

**Title, "All four milestones work on unseen test audio".** Every number on this slide is from the test set, which
was never used to tune anything.

**The test set.** 72 conversations, 83 minutes. 2 or 3 speakers each, from real conversational Hindi (IndicVoices),
under 3 overlap levels × 3 noise conditions (clean, village at 10 dB, market at 5 dB). Settings were tuned on a
separate dev set of 36 conversations with different speakers and different noise ([DATASETS.md](../../DATASETS.md)).

**The measure: cpWER** (concatenated minimum-permutation word error rate, from the CHiME-6 challenge). Each speaker's
words are joined into one stream, our speakers are matched to the true speakers in the way that gives the fewest
errors, and then words are compared. So a word is wrong if it is misheard, *or* heard right but given to the wrong
speaker. It counts both recognition and diarization mistakes, which is why the numbers look high. Lower is better.
Scored on Devanagari text.

### The bars

| Bar | Number | What it is | Source ([RESULTS_TABLES.md](../../RESULTS_TABLES.md)) |
|---|---|---|---|
| "ASR alone: clean voices, true speaker timeline" | 17.7 % | IndicConformer on each speaker's own clean recording, cut by the true timeline. Only recognition errors | row `oracle-clean` |
| "+ noise and overlapping speech (true timeline)" | 29.7 % | The same, but on the noisy mixture | row `oracle-mix` |
| "+ our diarization = full pipeline (Order B)" | 49.5 % | Everything automatic: our clustering, Order B | row `B-spectral` |
| "Proposal's order (A) … DER 26.2 %" | 55.8 % | Everything automatic, Order A | row `A-spectral` |
| "Ours: 11 % fewer errors, better in 50 of 72" | | (55.8 − 49.5) / 55.8 = 11 % relative. In points: 6.3 overall, 6.7 per conversation on average (95 % CI 4.0 to 9.4). Order B wins in 50 of the 72 conversations | [README](../../../README.md) |

What the script says about them:

- **"Diarization adds about 20 points, the largest share":** the steps add 17.7, then 12.0, then 19.8 points.
- **"Spectral and GMM as the proposal asks":** the proposal says "spectral clustering or GMM"; we wrote both. They tie
  (DER 20.3 % each). Spectral is the default.
- **"Within the confidence interval of pyannote 3.1":** our DER is 20.3 % (95 % CI 17.3–23.6), pyannote 3.1, the
  standard open-source diarizer, gets 18.4 % (15.5–21.4). The intervals overlap, so the data can't tell them apart.
  Not "better than pyannote".
- **"On separated audio its error rises from 20 to 26 percent":** DER is 20.3 % in Order B, 26.2 % in Order A. The
  separator leaves artefacts and gaps that confuse the speaker fingerprints, and Order A misses more speech (10.3 % vs
  6.7 %).

### The cards

| Card | Number | What it means | Source |
|---|---|---|---|
| M1 Separation | +9.6 dB | SI-SDR improvement on the 245 overlapping stretches: how much cleaner each voice is after separation than in the mix. +10 dB ≈ the other voice is 10× quieter. On the whole recording (what Order A asks of it) it's only +4.5 dB | [RESULTS.md](../../RESULTS.md), Stage 1 |
| M2 Diarization | 20.3 % | DER (diarization error rate): the share of speaking time missed (6.7 %), falsely detected (1.6 %) or given to the wrong speaker (11.9 %), with a 0.25 s tolerance at boundaries. pyannote 3.1: 18.4 % | RESULTS.md, Stage 2 |
| M3 Transcription | 21 % | WER on 400 clips of real-world Hindi phone recordings (Project Vaani) that neither model was trained on. IndicWav2Vec, the proposal's alternative: 38 %. The model was chosen on Vaani, not IndicVoices, because IndicConformer was trained on IndicVoices | D19, D24; [DATASETS.md §2](../../DATASETS.md) |
| M4 LLM report | ~1 min | LLM time per minute of audio on the laptop GPU: the 63 s demo took 58 s (4 calls, 22 tokens/s) | RESULTS.md, "Speed and memory" |

---

## Slide 3: the report, and where we go next (Darshan)

**Left card, "Real report output".** Copied from [results/demo/report.md](../../../results/demo/report.md) (the
same conversation as the clip). "Speaker A lost his ID card" is one of its key points, and correct. Line 13 is the
Hinglish transcript with the model's English translation. The caption lists what the report has: a summary, keywords,
and an English line for every line, made by Airavata 7B, 4-bit, run locally with llama.cpp.

### What we learned

| Point | What it means | Evidence |
|---|---|---|
| "It never touches who or when" | The LLM must reply in a fixed JSON format that has no speaker or time fields. Speakers and times are copied from M2, so the model can't move a line or give it to someone else | D28; every reply followed the format (the server enforces it) |
| "Its repair does not help: +0.9 points of cpWER on 63 test conversations" | We asked it to fix recognition mistakes. Full-pipeline error went from 51.8 % to 52.8 % (+0.9, 95 % CI +0.8 to +1.1); it lowered the error in 1 conversation and raised it in 50. So the M3 transcript stays the record, and the report is a reading aid | RESULTS.md, Stage 4 |
| Why 63, and why 51.8 not 49.5 | Test group 7 was used to write the prompts, so it is never scored. Without it the full-pipeline error is 51.8 % | D32, D33 |
| "Errors cascade: 99 % … 60 %" | Share of the report's keywords that were really said: 99 % when given the true transcript, 60 % from the real pipeline's output | RESULTS.md, Stage 4 |

### Where we go next

| Point | What it means | Evidence |
|---|---|---|
| "Who-spoke-when … about 20 of the 49.5 points, mostly speaker confusion" | The 19.8 points diarization adds are the largest share; within DER, wrong-speaker time (11.9 %) is bigger than missed (6.7 %) and false alarm (1.6 %). The fingerprint model was trained on VoxCeleb (mostly English); fine-tuning it on Hindi conversations is the first step | RESULTS_TABLES.md, `B-spectral` |
| "Action items. Empty in all 252 reports" | 63 conversations × 4 inputs = 252 reports, and every action-items section is empty. The model proposed 4 in total, and all were dropped for not matching their lines. The code works with any OpenAI-compatible LLM server, so a larger model is one setting away | RESULTS.md, "Action items: none" |
| "Separation … trained on English mixtures (Libri2Mix)" | The Conv-TasNet checkpoint has never heard Hindi. Training it on Indian voices and noise is the next step | [DATASETS.md §5](../../DATASETS.md) |
| Footer: `python -m whospoke run recording.wav --postprocess` | The real command that runs all four stages, from a recording to the report | [README](../../../README.md) |

---

## Words you might be asked about

| Term | One line |
|---|---|
| Diarization | Working out who spoke when, without knowing the speakers in advance |
| Speaker embedding | A short vector (a voice fingerprint); the same person's speech gives vectors pointing the same way |
| Spectral clustering / GMM | Two ways to group fingerprints into speakers: by a similarity graph, or by fitting Gaussian blobs |
| SI-SDR | How close a separated track is to the true voice, in dB, ignoring volume; we report the gain over the mix |
| WER / cpWER | Share of words wrong; cpWER also counts words given to the wrong speaker |
| DER | Share of speaking time missed, falsely detected, or given to the wrong speaker |
| SNR 5 dB | Speech only about 3× more powerful than the noise: loud |
| 4-bit | Each model weight stored in 4 bits instead of 16, so the 7B model shrinks from ~14 GB to ~4 GB |
| CHiME-6 | A speech-recognition challenge on dinner-party recordings, where cpWER comes from |

## Things to avoid saying

| Don't say | Why | Say instead |
|---|---|---|
| "50 % accuracy" | It's an error rate, and a strict one | "49.5 % error, counting misheard words and wrong speakers" |
| "11 points better" | It's 6.3 points, or 11 % relative | "11 percent fewer errors" |
| "better than pyannote" | It's tied within error bars | "within the confidence interval of pyannote" |
| "the LLM fixes the transcript" | It makes it slightly worse | "the transcript stays the record" |
| "it extracts action items" | It doesn't | "next step" |
| "live" / "real time" for the clip | The clip replays stored output | "what whospoke wrote for this audio" |
| "a real recording" | It's built from real speech and real noise | "a test conversation built from real speech" |

## If someone asks anyway

The FAQ suggests there's no Q&A, but just in case:

- *"Why is the error so high?"* The measure is strict and the test is hard: overlap, loud noise, two languages. With
  clean voices and the true timeline it's 17.7 %; the breakdown measures every part after that.
- *"Why not use real recordings?"* Real recordings have no labels, so nothing could be scored. Built conversations
  give the exact answer for every stage (D3).
- *"Is that real output?"* Yes, from `results/demo/`, on a conversation near the test-set median.
- *"Why not just use pyannote?"* The proposal asked for spectral clustering or a GMM, so we wrote both, and ran
  pyannote 3.1 as a reference. They're within error bars.
- *"Does it run on a laptop?"* Yes. Stages 1–3 on the 6 GB RTX 3050 (16× faster than real time), and the 4-bit LLM on
  the same GPU as a separate step.

## On the day

[vismAI.pptx](vismAI.pptx) has all three clips embedded ([build_pptx.py](build_pptx.py)). Open it in PowerPoint (not
the Drive preview, which can't play video). A click on a clip plays it with sound and a second click pauses it; a
click anywhere else moves to the next slide. Order on slide 1: the big clip, then "Speaker A", then "Speaker B".

| Risk | Fallback |
|---|---|
| Clips won't play, or no sound | Vismay plays all three from his phone, held to the mic (links and setup at the top of [SCRIPT.md](SCRIPT.md)). The slide still shows the full timeline and transcript |
| No sound at all | The big clip still works silently; skip the two voice clips and say "each voice comes out on its own track" |
| Running long (script is about 3:55) | Skip the "Per stage" line at the end of slide 2 |
| Clicking next to a clip | That advances the slide; practise clicking on the clip itself |

## Key files

| File | What's in it |
|---|---|
| [SCRIPT.md](SCRIPT.md) | The words to say, with cues (also in the slides' speaker notes) |
| [vismAI.pptx](vismAI.pptx), [vismAI.pdf](vismAI.pdf) | The submitted deck, built by [build_pptx.py](build_pptx.py) from [slides/](slides/) |
| [deck/](deck/) | The slides' source |
| [whospoke_demo.mp4](whospoke_demo.mp4), [speaker_a.mp4](speaker_a.mp4), [speaker_b.mp4](speaker_b.mp4) | The three clips, made by [make_demo_video.py](make_demo_video.py) and [make_speaker_clips.py](make_speaker_clips.py) |
| [../../RESULTS.md](../../RESULTS.md), [../../RESULTS_TABLES.md](../../RESULTS_TABLES.md) | Every number on slide 2 and 3, with intervals |
| [../../DATASETS.md](../../DATASETS.md) | Where the test data comes from |
| [../../../results/demo/](../../../results/demo/) | The demo conversation's output and report |
