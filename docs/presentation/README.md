# Midsem flash talk — team vismAI

Tuesday 6 October 2026, 4:00 PM, D-LT10. 3 to 4 minutes, 3 slides, 10 % of the course grade. Vismay presents slide 1
and Darshan slides 2–3. Abhinav and Shrivaths stand with the team; all four must be there.

**Script to read on the day: [demo-video/SCRIPT.md](demo-video/SCRIPT.md).**

| | **Audio version (submitted)** | First version (not submitted) |
|---|---|---|
| File | [demo-video/vismAI.pptx](demo-video/vismAI.pptx) (and a [PDF](demo-video/vismAI.pdf)) | — |
| Slide 1 | Pipeline with owners and tools; an 18 s clip of a test conversation with whospoke's timeline and transcript in sync; Speaker A and Speaker B separated from the overlap, each playable | Pipeline, owners, one real output line |
| Script | [demo-video/SCRIPT.md](demo-video/SCRIPT.md), about 3:55 | [main/SCRIPT.md](main/SCRIPT.md), about 3:25 |
| What every line means | [demo-video/EXPLAINER.md](demo-video/EXPLAINER.md) | [main/EXPLAINER.md](main/EXPLAINER.md) |
| Slides' source | [demo-video/deck/](demo-video/deck/) | [main/deck/](main/deck/) |

Who built what (the contributions statement the FAQ asks for):

| Milestone | Owner |
|---|---|
| 1 · Separation | Vismay |
| 2 · Diarization | Darshan |
| 3 · Transcription | Abhinav |
| 4 · LLM report | Shrivaths |

## The submitted file

`vismAI.pptx` is built by [demo-video/build_pptx.py](demo-video/build_pptx.py): each slide is a picture of the
published slide (so it looks the same on any PC, whatever fonts it has) with the script in the speaker notes, and
slide 1 has the three clips embedded. A click on a clip plays it with sound; a second click pauses it. Clicking
anywhere else moves to the next slide.

On slide 1, click the big clip first, then "Speaker A", then "Speaker B". If the hall has no sound, the big clip still
works silently (the transcript appears in sync); skip the two voice clips and carry on.

The slides follow Anthropic's visual style: ivory `#FAF9F5`, near-black `#141413`, clay `#D97757` as the only strong
accent, Poppins for headings and Lora for body text.
