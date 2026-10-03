# Flash talk, demo-video version: what's different, and the risks

**Backup only. Submit the [main version](../main/).** This version swaps slide 1's static output line for an
18-second video of a real test recording with sound. It's more impressive, but whether it works **depends entirely on
the venue PC**, which we can't test beforehand (personal laptops are not allowed).

## (a) ELI5

On slide 1, Vismay clicks the video and everyone hears a real test conversation: two people talking over each other,
with a market behind them. While it plays, whospoke's speaker timeline (orange = Speaker A, blue = Speaker B) fills
in, and the transcript appears line by line in Hinglish. Hearing how messy the audio is makes the results on slide 2
land harder.

Slides 2 and 3 are the same as the main version, except that slide 2's "one number per milestone" sentence is cut to
make room for the clip. Everything in the [main explainer](../main/EXPLAINER.md) applies here too.

## (b) The clip, exactly

| | |
|---|---|
| Audio | `test_g02_2spk_ovl-high_market5`, 00:26.4–00:44.8: the same test conversation as `results/demo/`. Two speakers, heavy overlap, market noise at 5 dB. Unedited apart from a 0.3 s fade-in and 0.5 s fade-out. |
| Timeline bars | Speaker turns from `results/demo/timeline.json` (Stage 2 output), drawn up to the playhead |
| Transcript | Hinglish lines from `results/demo/transcript.json` (Stage 3 output), each shown when its line starts |
| How it was made | [make_demo_video.py](make_demo_video.py) replays the committed output against the audio; nothing is re-run. Re-running it produces the same file. |

**Say "what whospoke wrote", not "live".** The clip replays output the pipeline already produced for this recording;
it isn't processing the audio in real time. If anyone asks: the output is real and unedited, and the clip only times
it to the audio.

## How to run it on the day

The Slides page plays the clip with sound in Present mode, but it needs a browser, internet and our login, so it's
not an option at the venue. For PowerPoint:

1. Export the deck to PowerPoint from its artifact page. The export holds only the still picture of the video.
2. In PowerPoint, on slide 1, delete the picture and use **Insert → Video → This Device →
   `whospoke_demo.mp4`**. Resize it to the same box.
3. Under **Playback**, set **Start: On Click** and the volume to high.
4. Save as `vismAI.pptx`, open it on another PC, and play it once.

## What can go wrong (and the fallback)

| Risk | Why it's likely in a lecture hall | Fallback |
|---|---|---|
| No sound | Audio often isn't routed to the hall speakers, or is muted | The clip still works silently: the transcript appears in sync. Vismay says one line: "This is a noisy recording; the transcript is what whospoke wrote." |
| Video won't play | Old PowerPoint, missing codec, or the file wasn't embedded | Skip it. Slide 2 still makes the point. |
| Time overrun | The clip adds 18 s; the script runs about 3:26 with it | Slide 2's script is already shortened. If it runs long, cut slide 3's middle point. |
| Clicking in the wrong place | A click next to the video advances the slide | Practise clicking on the video itself |

## Key files

| File | What's in it |
|---|---|
| [SCRIPT.md](SCRIPT.md) | The words to say for this version, with the cue to start the clip |
| [whospoke_demo.mp4](whospoke_demo.mp4) | The clip (18 s, 1920×1080, H.264 + AAC, 0.7 MB) |
| [whospoke_demo_still.png](whospoke_demo_still.png) | Its last frame, used as the picture in exports |
| [make_demo_video.py](make_demo_video.py) | Rebuilds the clip from the test audio and `results/demo/` |
| [deck/](deck/) | The slides' source |
