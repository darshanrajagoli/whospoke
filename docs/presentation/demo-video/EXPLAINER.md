# Flash talk, audio version (submitted): what's different, and the risks

**This is the version we submitted ([vismAI.pptx](vismAI.pptx)).** Slide 1 has an 18-second clip of a test
conversation with sound, and the two voices separated from its overlap, each playable on its own. Whether the sound
works **depends on the venue PC**, which we can't test beforehand (personal laptops are not allowed); the fallbacks
are below.

**Update, 6 Oct:** the script was made more technical (model and metric names, which are now on the slides too), and
the test set is described as *built from* real speech and real noise. [SCRIPT.md](SCRIPT.md) is current; where this
explainer quotes older wording, the numbers and their sources still hold.

## The separated voices (Speaker A, Speaker B)

| | |
|---|---|
| Audio | The same test conversation, 00:28.0–00:35.5: the stretch where both people talk at once |
| How | Milestone 1's separator (Conv-TasNet, the pipeline's default) run on the recording, as `python -m whospoke separate` does, then cut to that stretch. Each track is named after the diarized speaker whose voice it matches (voice-fingerprint match over the whole 18 s), so the colours match the big clip |
| Check | Against the true voices: each track is one clean voice (SI-SDR +9.4 and +8.2 dB, against +1.6 and −3.9 dB for the raw mix). The true voices are used only for this check |
| Made by | [make_speaker_clips.py](make_speaker_clips.py) |

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
| Audio | Test conversation `test_g02_2spk_ovl-high_market5`, 00:26.4–00:44.8: the same test conversation as `results/demo/`. Two speakers, heavy overlap, market noise at 5 dB. Unedited apart from a 0.3 s fade-in and 0.5 s fade-out. |
| Timeline bars | Speaker turns from `results/demo/timeline.json` (Stage 2 output), drawn up to the playhead |
| Transcript | Hinglish lines from `results/demo/transcript.json` (Stage 3 output), each shown when its line starts |
| How it was made | [make_demo_video.py](make_demo_video.py) replays the committed output against the audio; nothing is re-run. Re-running it produces the same file. |

**Say "what whospoke wrote", not "live".** The clip replays output the pipeline already produced for this recording;
it isn't processing the audio in real time. If anyone asks: the output is real and unedited, and the clip only times
it to the audio.

## How to run it on the day

**Plan on the day:** Vismay plays all three clips from his phone, held to the mic (links and steps at the top of
[SCRIPT.md](SCRIPT.md)); the slide shows the clip's still, which already has the full timeline and transcript. The
embedded clips below are there if the PC plays them.

[vismAI.pptx](vismAI.pptx) already has all three clips embedded ([build_pptx.py](build_pptx.py)). A click on a clip
plays it with sound and a second click pauses it; a click anywhere else moves to the next slide. Order on slide 1:
the big clip, then "Speaker A", then "Speaker B".

## What can go wrong (and the fallback)

| Risk | Why it's likely in a lecture hall | Fallback |
|---|---|---|
| No sound | Audio often isn't routed to the hall speakers, or is muted | The big clip still works silently: the transcript appears in sync. Skip the two voice clips and say "each voice comes out on its own track". |
| Video won't play | Old PowerPoint, missing codec, or the file wasn't embedded | Skip it. Slide 2 still makes the point. |
| Time overrun | The clips add 33 s; the script runs about 3:55 | If it runs long, skip the "Per stage" line at the end of slide 2. |
| Clicking in the wrong place | A click next to the video advances the slide | Practise clicking on the video itself |

## Key files

| File | What's in it |
|---|---|
| [SCRIPT.md](SCRIPT.md) | The words to say for this version, with the cue to start the clip |
| [whospoke_demo.mp4](whospoke_demo.mp4) | The clip (18 s, 1920×1080, H.264 + AAC, 0.7 MB) |
| [whospoke_demo_still.png](whospoke_demo_still.png) | Its last frame, used as the picture in exports |
| [speaker_a.mp4](speaker_a.mp4), [speaker_b.mp4](speaker_b.mp4) | The separated voices (7.5 s each), with stills, made by [make_speaker_clips.py](make_speaker_clips.py) |
| [vismAI.pptx](vismAI.pptx), [vismAI.pdf](vismAI.pdf) | The submitted deck, built by [build_pptx.py](build_pptx.py) from [slides/](slides/) (the slides rendered at 1920×1080) |
| [make_demo_video.py](make_demo_video.py) | Rebuilds the clip from the test audio and `results/demo/` |
| [deck/](deck/) | The slides' source |
