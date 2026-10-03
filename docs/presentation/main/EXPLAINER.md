# Flash talk, main version: what we say and why

For Vismay and Darshan. Read this once, then rehearse from [SCRIPT.md](SCRIPT.md).

## (a) ELI5: the talk in one minute

We have about three and a half minutes to show that all four milestones of our proposal are done and that they work.
The talk does three things, one per slide:

1. **Slide 1 (Vismay): what the project is and who did what.** People talk over each other in Hindi and English with
   a market in the background. whospoke works out who said what, and when. The four boxes are the four milestones, in
   the order our pipeline runs them, each with the name of the person who built it. The black strip is a real line our
   system wrote.
2. **Slide 2 (Darshan): does it work?** The bar chart splits our error into three parts. Speech recognition alone
   gets 17.7 % of words wrong. Noise and overlap take that to 29.7 %. Working out who is speaking takes it to 49.5 %.
   The grey bar is the order the proposal suggested: it does worse, at 55.8 %. The four cards give one headline number
   per milestone.
3. **Slide 3 (Darshan): the report, what we learned, and what's next.** The LLM writes a readable report, but it
   can't fix the transcript, and mistakes made earlier show up in it. Our breakdown shows where to improve next.

The tone throughout: plain facts, measured numbers, honest about the limits. No "amazing", no "state of the art".

---

## (b) Slide by slide: every number, what it means, where it comes from

### Slide 1: whospoke (Vismay)

| On the slide | What it means | Source |
|---|---|---|
| "4 / 4 milestones complete" | Separation, diarization, transcription and the LLM report are all built, measured and documented. The FAQ says this is what is graded. | README, "Headline results" |
| The four boxes | The proposal's milestones in the order our pipeline runs them: diarization (M2) first, then separation (M1), but only where two voices overlap, then transcription (M3), then the report (M4). | README, "How it works"; D26 in DECISIONS.md |
| Names on the boxes | The team's work split: Vismay M1, Darshan M2, Abhinav M3, Shrivaths M4. The FAQ requires a contributions slide or statement; this slide is it. | — |
| The black strip | A real line from the demo conversation (heavy overlap, 5 dB market noise), copied from our output. "aai D card" is how the recogniser spelled "ID card". | `results/demo/transcript_hinglish.txt`, line 6 |

**Why the order matters:** the proposal separates the whole recording first and then diarizes (Order A). We diarize
the original recording first and separate only the overlapping stretches (Order B). That is our one design change, and
slide 2 shows that it wins. Vismay's last line hands over to it.

### Slide 2: results (Darshan)

**The test set.** 72 conversations, 83 minutes, built from real conversational Hindi (IndicVoices) with real village
and market noise, at three levels of overlap and three levels of noise. The test speakers were never used to tune
anything.

**The measure: cpWER.** A word counts as wrong if it is misheard, *or* if it is heard right but given to the wrong
speaker. That is why the numbers look high: the measure counts mistakes from both the recogniser and the diarizer.
Always say "error", never "accuracy". Lower is better.

| Bar | Number | What it is | Source |
|---|---|---|---|
| Speech recogniser alone, one clean voice | 17.7 % | the recogniser on each speaker's own clean recording, with the true timeline | `docs/RESULTS_TABLES.md`, row `oracle-clean` |
| + noise and overlapping speech | 29.7 % | the same, but on the noisy mixture | row `oracle-mix` |
| + automatic who-spoke-when (our pipeline) | 49.5 % | everything automatic, Order B with our spectral clustering | row `B-spectral` |
| Proposal's order | 55.8 % | everything automatic, Order A | row `A-spectral` |

- **"The largest source of error":** the three steps add 17.7, then 12.0, then 19.8 points. Diarization's 19.8 is the
  largest. That is a measured fact, not modesty.
- **"Within error bars of pyannote":** our diarization error (DER) is 20.3 %, 95 % interval 17.3–23.6 %. Off-the-shelf
  pyannote 3.1 gets 18.4 %, interval 15.5–21.4 %. The intervals overlap, so the data can't tell the two apart.
- **"Both the spectral clustering and the Gaussian mixture":** the proposal says "spectral clustering or GMM". We wrote
  both from scratch, and they score the same (DER 20.3 % each).
- **"Diarization works best on the original recording":** Order B's diarization error is 20.3 %, Order A's is 26.2 %.
  Diarizing audio that has already been through the separator makes more mistakes.
- **"11 percent fewer errors":** (55.8 − 49.5) / 55.8 = 11 %. This is the relative drop. In points it is 6.3 overall,
  and 6.7 on average per conversation (95 % interval 4.0 to 9.4). Order B is better in 50 of the 72 conversations.
- **The cards:**
  - +9.6 dB is how much clearer the separator makes speech where two people overlap (SI-SDR improvement).
  - 21 % vs 38 % is word error on real-world Hindi from the Vaani dataset: IndicConformer, our choice, vs
    IndicWav2Vec, the alternative named in the proposal.
  - "~1 min per minute of audio" is the LLM report on the RTX 3050 laptop GPU (real-time factor 0.6–1.0).

### Slide 3: the report and what's next (Darshan)

- **The left card** is real output from `results/demo/report.md`: a key point it got right ("Speaker A lost his ID
  card") and line 13 with its English translation.
- **"It never touches who or when":** the LLM's reply has no field for speakers or times. They are copied from Stage 2
  (diarization), so the model cannot move a line or give it to someone else. (D28)
- **"Its repair does not help":** we asked the LLM to fix recognition mistakes. On 63 test conversations, full-pipeline
  error went from 51.8 % to 52.8 % (+0.9 points, interval +0.8 to +1.1). So the Stage-3 transcript stays the record,
  and the report is a reading aid. (Why 51.8 and not 49.5: test group 7 was used to write the LLM prompts, so it is not
  scored here.)
- **"Errors cascade":** 99 % of the report's keywords were really said when it was given the perfect transcript, and
  60 % when it was given the real recogniser's output.
- **Where we go next:**
  - Who-spoke-when is the largest share of the error (19.8 of 49.5 points), so the first step is voice fingerprints
    tuned on Hindi conversations.
  - The LLM never lists action items. The code accepts any OpenAI-compatible LLM server, so a larger model is a
    one-setting change; the extension is to test one.
  - The separator (Conv-TasNet) was trained on English speech (Libri2Mix). Training it on Indian voices and noise is
    the next step.
- **The footer** is the real command that runs everything, from a recording to the report.

---

## Things to avoid saying

| Don't say | Why | Say instead |
|---|---|---|
| "50 % accuracy" | It's an error rate, and a strict one | "49.5 % error, counting misheard words and wrong speakers" |
| "11 points better" | It's 6.3 points, or 11 % relative | "11 percent fewer errors" |
| "better than pyannote" | It isn't; the two are tied within error bars | "within error bars of pyannote" |
| "the LLM fixes the transcript" | It makes it slightly worse | "the transcript stays the record" |
| "it extracts action items" | It doesn't | (say nothing, or "next step") |
| "real time" for the whole pipeline | Only the report step was timed at about 1× | "about a minute per minute of audio, for the report" |

## If someone asks anyway

The FAQ suggests there is no Q&A, but just in case:

- *"Why is the error so high?"* — The measure is strict and the test is hard: overlap, loud noise, two languages.
  With a perfect speaker timeline and clean voices it's 17.7 %. The rest is noise and working out who is speaking,
  and our breakdown measures each part.
- *"Is that real output?"* — Yes. It's from `results/demo/`, on a test conversation whose error is close to the test
  set's median. It's a typical case, not our best.
- *"Why not just use pyannote?"* — The proposal asked for spectral clustering or a GMM, so we wrote both. We also ran
  pyannote 3.1 as a reference, and it's within error bars of ours.
- *"Does it run on a laptop?"* — Yes. Stages 1–3 run on the 6 GB RTX 3050, and so does the 4-bit LLM, as a separate step.

## Key files

| File | What's in it |
|---|---|
| [SCRIPT.md](SCRIPT.md) | The words to say, with cues |
| [deck/](deck/) | The slides' source (published as a Slides artifact; export it to PowerPoint or PDF from the artifact page) |
| [../../../README.md](../../../README.md) | Headline results |
| [../../RESULTS.md](../../RESULTS.md), [../../RESULTS_TABLES.md](../../RESULTS_TABLES.md) | Every number on slide 2, with intervals |
| [../../../results/demo/](../../../results/demo/) | The demo conversation's output and report |
