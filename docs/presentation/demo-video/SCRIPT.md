# Flash-talk script — audio version (the one we submit)

**When:** Tuesday 6 October 2026, 4:00 PM, D-LT10. **Time limit:** 3 to 4 minutes (this runs about 3:55; talk a little fast).
**Speakers:** Vismay (slide 1), Darshan (slides 2 and 3). Abhinav and Shrivaths stand with the team. All four must be there.
**The same text is in each slide's speaker notes.**
**What every line means:** [EXPLAINER.md](EXPLAINER.md).

Lines in [brackets] are cues. Don't read them out.

---

## Backup: the audio on Vismay's phone

Only if the PC's clips don't play (the file opened in a browser, or no sound). **Download all three to your phone
before 4 PM** (tap each link, then save), so they play without internet. On stage, play each one from the phone's
files or gallery and hold the phone's speaker close to the mic.

1. [Main clip, 18 s](https://github.com/darshanrajagoli/whospoke/raw/main/docs/presentation/demo-video/whospoke_demo.mp4): the noisy conversation
2. [Speaker A, 7.5 s](https://github.com/darshanrajagoli/whospoke/raw/main/docs/presentation/demo-video/speaker_a.mp4): first separated voice
3. [Speaker B, 7.5 s](https://github.com/darshanrajagoli/whospoke/raw/main/docs/presentation/demo-video/speaker_b.mp4): second separated voice

---

## Slide 1: whospoke — Vismay, about 95 seconds, including 33 seconds of audio

Good evening. We are team vismAI, and our project is whospoke: speaker-attributed transcription, meaning who spoke what, and when, in noisy Hindi–English conversations.

Here are eighteen seconds of one of our test conversations: two speakers overlapping, with market noise behind them. On screen: whospoke's speaker timeline and transcript, from the audio alone.

[CLICK THE BIG VIDEO. Let it play to the end, about 18 seconds. Say nothing.]
[If it won't play: phone file 1, see the backup section above.]

The pipeline is the proposal's four milestones, and each of us owned one. I built separation, using Conv-TasNet, a neural network that splits a recording into one track per speaker. This is the part of that clip where both of them talk at once, separated.

[CLICK "Speaker A". Let it play, about 7 seconds.]
[CLICK "Speaker B". Let it play, about 7 seconds.]
[If they won't play: phone files 2 and 3.]

Darshan built diarization, which uses speaker embeddings and our own clustering to work out who is speaking when. Abhinav built transcription, with AI4Bharat's IndicConformer and a Hinglish romaniser. Shrivaths built the report, with Airavata, a 7-billion-parameter Hindi LLM, running 4-bit on a laptop GPU.

Our main design choice is the order: diarize the original audio first, and separate only where voices overlap. Darshan will show you why.

[NEXT SLIDE. Hand over to Darshan.]

## Slide 2: results — Darshan, about 80 seconds

Thanks, Vismay. Everything here is measured on a held-out test set: 72 conversations, 83 minutes, built from real conversational Hindi and real village and market noise.

Our main metric is cpWER, from the CHiME-6 challenge. A word counts as wrong if it is misheard, or if it is given to the wrong speaker.

[Point to the bars, top to bottom.]
The ASR alone, on clean voices with the true speaker timeline: 17.7 percent. Add noise and overlap: 29.7. Replace the true timeline with our diarization: 49.5.

So diarization adds about 20 points, the largest share. Our clustering, spectral and GMM as the proposal asks, is implemented from scratch, and its diarization error, 20.3 percent, is within the confidence interval of pyannote 3.1, the standard tool.

[Point to the grey bar.]
And diarization works best on the original audio. On separated audio its error rises from 20 to 26 percent, so the proposal's order, separating everything first, ends at 55.8. Ours makes 11 percent fewer errors, and wins in 50 of the 72 conversations.

[Point to the M1 and M3 cards.]
Per stage: separation gains 9.6 dB SI-SDR where voices overlap, and IndicConformer gets 21 percent word error on real-world Hindi, against 38 for IndicWav2Vec, the proposal's alternative.

[NEXT SLIDE.]

## Slide 3: the report and what's next — Darshan, about 60 seconds

Finally, the report. Airavata, AI4Bharat's 7-billion-parameter Hindi LLM, runs locally and writes a summary, keywords and an English translation of every line, like these.

We learned three things.
First, it never touches who spoke or when. Its JSON output has no speaker or time fields; those are copied from diarization.
Second, its repair of the transcript does not help. It adds 0.9 points of cpWER, so the ASR transcript stays the record.
Third, errors cascade. From the true transcript, 99 percent of its keywords were really said. From real ASR output, 60.

Where next: who-spoke-when is the largest share of the error, mostly speaker confusion, so first we fine-tune the speaker embeddings on Hindi conversations. Then a larger LLM for action items, which came out empty in all 252 reports, and separation retrained on Indian voices and noise.

All of it runs with one command, from a recording to a report. Thank you.

---

## Rehearsal checklist

- Time it out loud. Aim for 3:50 to 3:58; talk a little fast.
- Vismay: click the big video first, then "Speaker A", then "Speaker B". Click on the clip itself, not next to it (a click beside it moves to the next slide).
- Practise the handover: "Darshan will show you why" is the cue for Darshan to step forward.
- Say "error", never "accuracy". Lower is better.
- Say "test conversation", not "real recording": the speech and noise are real, the mix is ours.
- Say the terms like this: cpWER "C-P-W-E-R", DER "D-E-R", SI-SDR "S-I-S-D-R", Conv-TasNet "conv-taz-net", GMM "G-M-M", Airavata "ai-raa-va-ta".
- Read numbers as written: "seventeen point seven percent", "fifty-five point eight".
