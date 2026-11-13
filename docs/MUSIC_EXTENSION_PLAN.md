# Extending WhoSpoke to music: three datasets and the experiments they allow

The professor asked for the pipeline to be extended to music datasets. This page picks **three** from the team's list of music
datasets and says, for each one, what experiment it supports, which
stage it exercises, and how it is scored. The guiding rule is the same one the speech work follows: **an experiment is
only worth running if we can build or obtain an exact reference to score it against**, and every choice is made on a
dev split before a test split is scored once.

Facts marked † come from the dataset's own pages and should be checked against the downloaded copy.

> **Status, 13 Nov 2026: experiment 2 + 3 has been implemented, with one dataset swapped.** RaagaDhvani v3 lists no
> downloadable files on Mendeley Data and is CC BY-ND 4.0 (adaptations may not be shared), so the melodic source is
> **Carnatic Varnam** from the same list: 28 solo sung varnams (6 singers, adi tala, tala-cycle annotations, audio
> CC BY 3.0 via Freesound; Zenodo 7726167). It suits the mridangam better: same tradition, and strokes can be placed on
> the singer's annotated beats. The Mridangam Stroke licence is **CC BY-NC 4.0** on Zenodo (not CC BY 3.0 as `mirdata`
> says); our copy has all 7,162 strokes. Code: `src/whospoke/synth_music.py`, `scripts/build_music_dataset.py`,
> `scripts/eval_music_diarization.py`, HPSS and Demucs in `src/whospoke/separation.py`. Results: DECISIONS.md D41
> and RESULTS.md. Experiment 1 (DEAM) has not been run. Two more changes from the plan below: dev and test are split
> by singer and by stroke take (odd vs even), not by drum tonic (the drum follows the singer's tonic);
> and the instrument fingerprint is a hand-made timbre feature (`timbre_embed`), not a pretrained audio embedding.

## The choice in one table

| # | Dataset | What it adds | Stages it tests | Reference we score against | Effort |
|---|---|---|---|---|---|
| 1 | **DEAM** | real music as the **background** of our existing Hindi conversations ("talking over music") | all four, unchanged | the existing exact speech references (cpWER, DER, SI-SDR) | low: swap the noise source in `synth.py` |
| 2 | **Mridangam Stroke** | percussion as a **source** to be separated and timed | 1 (separation) and 2 ("who plays when") | exact stems and onsets, because we build the mixtures ourselves | medium |
| 3 | **RaagaDhvani v3** | a melodic solo instrument (flute) to mix with the mridangam | 1 and 2, with 2 | same as above | medium (combined with #2) |

Datasets 2 and 3 form **one** experiment: a simulated Indian duo (flute + mridangam), built exactly as `synth.py`
builds conversations from single-speaker speech.

## 1. DEAM: conversations over background music

**Why this one.** It needs no new code beyond a noise source, it keeps every existing metric, and background music is
a realistic condition for the project's setting (shops, buses, weddings, TV in the room). Music is also a harder
"noise" than DEMAND or ESC-50 for this pipeline, because it is harmonic and sometimes contains singing, which the
speech detector and the separator can mistake for a third person.

**Facts.** 1,802 songs and excerpts with valence/arousal ratings; the 2013–2014 parts are 45-second excerpts taken at
random points of songs, the 2015 part is full songs; dynamic ratings at about 2 Hz; audio under Creative Commons
licences (Free Music Archive)†. The exact CC variant per track must be checked before any audio is redistributed.

**Experiment.**
1. Split DEAM by **song** into dev and test halves (as the DEMAND/ESC-50 noise is split now, D13), so no song is in both.
2. Add a `music` condition to `scripts/build_dataset.py`: same speaker groups and overlap levels, background from DEAM at
   **10 dB and 5 dB SNR** (matching village/market), excerpts concatenated to cover ~70 s.
3. Run the unchanged pipeline (`scripts/evaluate.py`) and compare against the existing noise conditions: SI-SDRi on
   overlaps, DER, cpWER, Order A vs Order B.
4. Extra analysis that DEAM's labels make possible: does the error grow with the music's **arousal** (energetic music)?
   Correlate per-conversation cpWER and false-alarm time with the mean dynamic arousal of the background excerpt. Also
   split songs with and without vocals: sung lyrics are the likeliest to be detected as an extra speaker.

**Expected outcome to test, not to assume.** False-alarm speech and extra speakers rise most when the music has
vocals; Order B's advantage over Order A should hold or grow, because separating the whole recording now also has to
deal with music.

## 2 + 3. Mridangam Stroke + RaagaDhvani: a simulated flute–mridangam duo

**Why these two.** Both are **single-source recordings**: one stroke or one flute at a time. That is the property our
speech simulation relies on (one clean IndicVoices speaker per track), so the same method gives mixtures with **exact
stems and exact "who plays when" references**. Real ensemble recordings such as Saraga do not give that for Hindustani
music (below).

**Facts.**
- *Mridangam Stroke:* 7,162 single strokes, 10 stroke types (bheem, cha, dheem, dhin, num, ta, tha, tham, thi, thom), 6
  tonics (B, C, C#, D, D#, E), played by a professional percussionist, 44.1 kHz WAV, CC BY-NC 4.0 (Zenodo 1265188;
  our copy: 7,162 files, checksum verified).
- *RaagaDhvani v3:* solo flute in 11 Carnatic ragas with emotion annotations; CC BY-ND 4.0; no files listed for v3.
  **Replaced by Carnatic Varnam** (see the status note at the top).

**Experiment.**
1. **Build mixtures** with a new `synth_music.py` mirroring `synth.py`: a flute phrase (resampled to 16 kHz) plus a
   mridangam track assembled from strokes at a steady tempo with random fills. Control the overlap (fraction of time
   both play) and the level ratio. Keep each stem and write `reference.rttm` (who plays when) and stroke onsets.
   Split by flute recording and by mridangam tonic between dev and test.
2. **Stage 1, separation.** Score SI-SDRi on the overlapping stretches (`scripts/eval_separation.py` works as is once
   the stems exist). Compare our speech separator (Conv-TasNet) with a **music** separator (Demucs, which we rejected
   for speech in D8 because it separates stems, not people). This is the mirror image of D8: it tests whether the right
   separator depends on the kind of sources.
3. **Stage 2, "who plays when".** Run the diarizer unchanged and score DER against the reference. Then swap the
   speaker-fingerprint model for a general audio embedding, since WeSpeaker was trained on voices and should struggle
   with instruments. Spectral clustering and the timeline code stay the same: only the fingerprints change.
4. **Optional link to the emotion work.** Run the existing music-emotion model on the original flute, on the mixture and
   on the separated flute, against RaagaDhvani's raga-emotion labels. Does separation recover the mood the mixture
   hides?

**Expected outcome to test.** Demucs beats Conv-TasNet on this data; percussion-vs-melody separation is easier than
voice-vs-voice (the sources sound very different); the diarizer needs instrument embeddings to work at all.

## Considered and not chosen (for now)

| Dataset | Why not first |
|---|---|
| **Saraga Hindustani** | Real concerts, but its multitrack recordings are Carnatic only, so there are no Hindustani stems to score separation against†. Its annotations (sections, sama, phrases, tonic, mixed-track pitch) suit rhythm/melody work rather than this pipeline. A good **qualitative** test of the duo experiment's winner on real music |
| **Carnatic Varnam** | Small (28 recordings); raga and tala labels, no stems. *Later used in place of RaagaDhvani (status note above): the recordings are solo singing, so they serve as the voice stem* |
| **PMEmo, MERGE** (have lyrics) | Lyrics are English/Chinese; our Stage-3 recogniser is for Indian languages, so a lyrics experiment would need a different ASR model first |
| **MagnaTagATune, CAL500, Emotify, MEMO2496, MERP** | Tag or emotion labels only: nothing that scores "who" or "when" |
| **NRC-VAD, EmoBank** | Text lexicons, no audio |

## Order of work

1. DEAM background condition (smallest change, every metric reused).
2. The duo simulation, then separation (Conv-TasNet vs Demucs).
3. "Who plays when" with instrument embeddings.
4. Optional emotion link and the Saraga qualitative check.

## Sources

- DEAM: [Aljanaki, Yang & Soleymani (2017), *Developing a benchmark for emotional analysis of music*, PLoS ONE](https://pmc.ncbi.nlm.nih.gov/articles/PMC5345802/) · [dataset page and manual](https://cvml.unige.ch/databases/DEAM/manual.pdf)
- Mridangam Stroke: [CompMusic page](https://compmusic.upf.edu/mridangam-stroke-dataset) · [Zenodo record](https://zenodo.org/record/1265188) · [mirdata loader](https://mirdata.readthedocs.io/en/0.3.5/_modules/mirdata/datasets/mridangam_stroke.html)
- Saraga: [coverage and organisation](https://mtg.github.io/saraga/organization.html) · [mirdata Saraga Hindustani loader](https://mirdata.readthedocs.io/en/0.3.2/_modules/mirdata/datasets/saraga_hindustani.html)
