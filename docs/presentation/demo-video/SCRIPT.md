# Flash-talk script — demo-video version (backup, needs sound and video on the venue PC)

**When:** Tuesday 6 October 2026, 4:00 PM, D-LT10. **Time limit:** 3 to 4 minutes.
**Speakers:** Vismay (slide 1), Darshan (slides 2 and 3). Abhinav and Shrivaths stand with the team. All four must be there.
**The same text is in each slide's speaker notes.**
**What every line means, and the risks of this version:** [EXPLAINER.md](EXPLAINER.md).

Lines in [brackets] are cues. Don't read them out.

---

## Slide 1: whospoke — Vismay, about 75 seconds, including the 18-second clip

Good evening. We are team vismAI, and our project is whospoke: who spoke what, and when.

Two people talking over each other, switching between Hindi and English, with a busy market behind them. Most speech tools struggle with any one of those. Here are eighteen seconds of one of our test conversations, and what whospoke wrote from the audio alone.

[CLICK THE VIDEO. Let it play to the end, about 18 seconds. Say nothing.]

That is whospoke's own speaker timeline and transcript for that noisy, overlapping audio.

The pipeline is the proposal's four milestones, and each of us owned one. I built separation, which pulls overlapping voices apart. Darshan built diarization, which works out who is speaking when. Abhinav built transcription, which turns speech into text, and Shrivaths built the LLM report.

That order, diarization first and separation only where voices overlap, is our main result. Darshan will show you why.

[NEXT SLIDE. Hand over to Darshan.]

## Slide 2: results — Darshan, about 70 seconds

Thanks, Vismay. Everything here is measured on 72 test conversations, 83 minutes of real conversational Hindi with real village and market noise. None of the test speakers were used for tuning.

Our main measure is strict. A word counts as wrong if it is misheard, or if it is given to the wrong speaker.

[Point to the bars, top to bottom.]
This chart shows where the errors come from. The speech recogniser alone, on one clean voice: 17.7 percent. Add noise and overlapping speech: 29.7. Add fully automatic who-spoke-when: 49.5.

That last step, telling apart two people who talk over each other in a noisy market, is the largest source of error. Our clustering, both the spectral clustering and the Gaussian mixture the proposal asks for, is written from scratch, and it comes within error bars of pyannote, the standard tool.

[Point to the grey bar.]
And diarization works best on the original recording, before separation touches it. The proposal's order, separating everything first, gets 55.8. Ours makes 11 percent fewer errors, and wins in 50 of the 72 conversations.

(Leave the cards on the right for the audience to read; the clip used the time.)

[NEXT SLIDE.]

## Slide 3: the report and what's next — Darshan, about 60 seconds

Finally, the report. A local Hindi LLM, AI4Bharat's Airavata, writes a summary, keywords and an English translation of every line. Here it picks out that the speaker lost his ID card, and translates his question about office hours.

We tested the LLM as carefully as the rest, and learned three things.
First, it never touches who spoke or when. Those come straight from diarization.
Second, its attempt to repair the transcript does not help. It adds about one point of error, so the transcript stays the record.
Third, errors cascade. From a perfect transcript, 99 percent of its keywords were really said. From real speech recognition, 60.

The breakdown also tells us where to go next. Who-spoke-when is the largest share of the error, so first: voice fingerprints tuned on Hindi conversations. Then a larger LLM to pull out action items, and separation trained on Indian voices and noise.

All of it runs with one command, from a recording to a report. Thank you.

---

## Rehearsal checklist

- Time it out loud. Aim for 3:20 to 3:40.
- Practise the handover: "Darshan will show you why" is the cue for Darshan to step forward.
- Say "error", never "accuracy". Lower is better.
- Read numbers as written: "seventeen point seven percent", "fifty-five point eight".
