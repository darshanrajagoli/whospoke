# Datasets

What data the project uses, where it comes from, how much of it there is, and what each piece is for. We train no
model (D5), so each dataset is used to build test audio, to choose between models, or to tune settings.

## At a glance

| Dataset | Where from (licence) | What we take | Used for |
|---|---|---|---|
| **IndicVoices** (AI4Bharat) | Hugging Face [`ai4bharat/IndicVoices`](https://huggingface.co/datasets/ai4bharat/IndicVoices), gated | Hindi *valid* split (1 file) and *train* shard 0 of 82; only the phone-conversation recordings | The speech in every dev and test conversation; a side check of the ASR models (400 clips) |
| **Project Vaani**, transcribed part (IISc / ARTPARK) | Hugging Face [`ARTPARK-IISc/Vaani-transcription-part`](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part), gated | Hindi *test* shards 0–1 of 28 (with audio); the transcript column of 25 of the 193 Hindi *train* shards (no audio) | Choosing the Stage-3 model on real-world audio; measuring English words inside Hindi; building the Hinglish lexicon |
| **DEMAND** | [Zenodo 1227121](https://zenodo.org/records/1227121) (CC-BY-4.0) | Channel 1 of 8 environments, 16 kHz | Continuous background noise (the "bed") |
| **ESC-50** | [GitHub](https://github.com/karolpiczak/ESC-50) (CC-BY-NC-3.0) | Clips from 17 sound categories (animals, horns, engines, bells …) | Noise events on top of the bed |
| **Nirantar** (AI4Bharat) | Hindi extracted by a teammate on Colab (the full release is one ~198 GB archive) | A random 800-clip Hindi sample; 406 clips after removing our test speakers | Stage-3 cross-check only |
| AIR-RS-DB | — | Not used: no public source exists under this name (the proposal's citation is blank) | — |

Why Nirantar and AIR-RS-DB are not used more is in [DEVIATIONS.md](../DEVIATIONS.md) (row 2) and
[DECISIONS.md](../DECISIONS.md) D7. Vaani's main corpus and its Noise-Event release were also tried as a noise source and
dropped, because they hold about 1 s of speech-free audio per 500 s (D11).

## 1. The dev and test conversations (the main evaluation data)

Real broadcasts have no labels, so they can't be scored. We build conversations ourselves from real speech and real
noise, so we know exactly who said what and when (D3). `scripts/build_dataset.py` does it with `src/whospoke/synth.py`
and `src/whospoke/noise.py`.

**How one conversation is made**
1. Pick 2 or 3 speakers. Each brings their own side of a real IndicVoices phone call (the *Conversation* scenario,
   at least 4 chunks and 25 s), so the Hindi is spontaneous and naturally mixed with English.
2. Trim the silence at the edges of each utterance (D14), then let the speakers take turns, with short gaps, some turn
   changes overlapping, and loudness varied by up to ±3 dB.
3. Add a background soundscape at a set speech-to-noise ratio (SNR).

**Conditions.** Each group of speakers is rendered under all 9 conditions (D12), so a difference between conditions
comes from the condition, not from different voices:

| | Settings |
|---|---|
| Overlap | none · low · high |
| Noise | clean · village at 10 dB SNR · market at 5 dB SNR |
| *village* | DEMAND field, park, living room, kitchen + ESC-50 cows, roosters, hens, dogs, crows, sheep, insects, crickets, birds, frogs |
| *market* | DEMAND traffic, public square, station, bus + ESC-50 car horns, engines, sirens, trains, bells, dogs, crows, footsteps, alarms |

**The two sets**

| | Test | Dev (tuning) |
|---|---|---|
| Speakers from | IndicVoices Hindi *valid* split | IndicVoices Hindi *train* shard 0, minus every speaker who is also in *valid* (91 were; D13) |
| Speaker groups | 8 (four of 2 speakers, four of 3) | 4 (two of 2, two of 3) |
| Conversations | **72** (8 groups × 9 conditions) | **36** |
| Total audio | **82.9 min** (about 1 min each) | 38.9 min |
| Mean overlap: none / low / high | 0 / 4.4 % / 11.4 % of the time | 0 / 3.2 % / 13.3 % |
| Noise recordings | Second half of each DEMAND recording; ESC-50 folds 1–3 | First half; folds 4–5 |
| Used for | Final numbers, reported once | Every threshold and setting |

**What each conversation contains** (`<data>/synth/<split>/<id>/`):
`mixture.wav` (what the pipeline hears), `sources/S1.wav …` (each speaker alone, used for SI-SDR),
`reference.rttm` (who spoke when, for DER) and `reference.json` (every turn with its words, for WER and cpWER).
The names spell out the condition, e.g. `test_g02_2spk_ovl-high_market5` = test group 2, two speakers, high overlap,
market noise at 5 dB. That one is the example in [results/demo/](../results/demo/) and in the flash-talk clip.

**Stage 4 subset.** The LLM report is scored on the 63 test conversations of groups 0–6. Group 7 was used to revise the
prompts and is never scored (D32, D33).

## 2. Real-world speech for choosing the ASR model (Stage 3)

The pipeline's ASR model was chosen on audio that neither candidate had been trained on (D19, D24).
`scripts/eval_asr.py` and `scripts/eval_asr_nirantar.py`; single-speaker clips.

| Set | Clips | Hours | Role | IndicConformer WER | IndicWav2Vec WER |
|---|---|---|---|---|---|
| **Vaani** *test* | 400 | 0.75 | **The selection set.** Real phone recordings from across India; neither model saw them | **21.0 %** | 38.0 % |
| IndicVoices *valid* | 400 | 0.75 | Reported, with a caveat: IndicConformer was trained on IndicVoices, so it has a home advantage here | 13.9 % | 36.1 % |
| Nirantar Hindi sample | 406 | 0.91 | Cross-check, test speakers removed. Same collection as IndicVoices, so it flatters IndicConformer the same way | 11.0 % | 30.2 % |

All three give the same ranking.

## 3. Data for the Hinglish romaniser (Stage 3)

Vaani's annotators tag every English word spoken inside Hindi, e.g. `बिल्डिंग {building}`. We use these tags twice,
keeping train and test apart (D20):
- **Building the lexicon:** mined from Vaani *train* transcripts only (25 of 193 shards, about 66k transcripts), added
  to a hand-made list drawn from IndicVoices' 2,500 most frequent words. The result is
  `src/whospoke/resources/loanwords.tsv`, about 2,600 entries (`scripts/mine_loanwords.py`).
- **Testing it:** all 8,630 tagged English words in the two Vaani *test* shards. 82 % come out spelled correctly in
  Hinglish, and 78 % of the 775 English words in the 400 test clips are recognised by IndicConformer.

## 4. Keeping the test data honest

| Risk | What we did |
|---|---|
| Tuning on the test speakers | Dev speakers never appear in the test set (D13) |
| Tuning on the test noise | Different halves of each DEMAND recording and different ESC-50 folds for dev and test |
| IndicConformer has heard IndicVoices | The ASR model was chosen on Vaani instead (D19) |
| The lexicon has seen the test words | Mined from Vaani *train*; scored on Vaani *test* (D20) |
| Nirantar overlaps our test speakers | 165 of the 514 test-split speakers are also in Nirantar, and 923 of its clips are exact IndicVoices *valid* files. Every test speaker was removed before scoring (D7) |
| Tuning the LLM prompts on scored data | Test group 7 was used for prompt work and is never scored (D32) |

## 5. Data inside the pretrained models

We use these models as they are. What they were trained on matters for how far their numbers can be trusted:

| Model | Stage | Trained on |
|---|---|---|
| Conv-TasNet (`JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k`) | 1 | Libri2Mix, noisy: two-speaker *English* read speech with background noise, 16 kHz. Never heard Hindi; still gives +6.4 dB on our overlaps (D18) |
| WeSpeaker ResNet-34 (`pyannote/wespeaker-voxceleb-resnet34-LM`) | 2 | VoxCeleb (celebrity interviews, mostly English) |
| IndicConformer | 3 | Thousands of hours of Indian-language speech, including IndicVoices (hence D19) |
| IndicWav2Vec-Hindi | 3 | Self-supervised on 40 Indian languages, then fine-tuned on Hindi |

## 6. Getting the data

The data is git-ignored and lives outside the repository (`WHOSPOKE_DATA`, or a one-line `data_location.txt`, or
`./data` by default; see `src/whospoke/paths.py`). About 3 GB of downloads plus 0.9 GB of built conversations.

```bash
huggingface-cli login                 # IndicVoices and Vaani are gated: accept their terms on Hugging Face first
python scripts/fetch_data.py          # DEMAND, ESC-50, Vaani (~1.9 GB)
python scripts/build_dataset.py       # downloads IndicVoices (~1 GB), builds dev + test (seed 407)
python scripts/mine_loanwords.py      # Hinglish lexicon from Vaani train
```

The Nirantar sample (`<data>/raw/nirantar/`) is not downloadable by script: it comes from the teammate's Colab
extraction (D7).
