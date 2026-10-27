# whospoke live

Two people talk into **one laptop microphone**. On the projector, the timeline of who speaks when and the
transcript build up while they talk. When they stop, the final result appears, each voice can be played on its own,
and (for Hindi) the LLM writes its report.

```powershell
# from C:\whospoke-stage4 (that copy has the 4-bit LLM in models\, so the report works with no extra flags)
C:\whospoke-data\venv\Scripts\python.exe demo\live.py
# from any other copy of the repo, point it at the LLM file:
C:\whospoke-data\venv\Scripts\python.exe demo\live.py --gguf C:\whospoke-stage4\models\airavata-q4_k_m.gguf
```

The browser opens at http://127.0.0.1:8765. Press F11 for full screen. The speech models take about half a minute
to load the first time. No internet is needed: every model loads from the local cache (`--allow-downloads` lifts that).

## What happens on screen

| On screen | What it is |
|---|---|
| Waveform | What the microphone heard. It turns dark once a pass has processed it. |
| A / B lanes | M2: who speaks when. Hatched stripes are stretches with two voices at once. M1 separates those before transcription. |
| Transcript | M3: Devanagari, then Hinglish. After the report, each line also gets the LLM's English translation. The "separated" tag means the line was transcribed from the separated voice. |
| Last pass | Time per stage, in pipeline order, and how much faster than real time it ran (about 20× on this GPU). |
| Hear each voice | After Stop: the whole recording through Conv-TasNet, each track matched to a speaker by voice fingerprint. |
| Report | M4, Hindi only: Airavata 7B writes a summary, key points and an English line for every line. |

While you talk, the recording so far goes through the whole pipeline again whenever ~1.5 s of new audio has arrived
(about every 2 s). It is not word-by-word captioning: who is who comes from clustering the whole recording, so the
first 10 to 20 seconds look rough and then settle as it hears more of each voice. Say that out loud; it is the
clustering at work.

**People: 2 / 3 / Count them.** Leave it on **2**. With the count given, our spectral clustering keeps its own
result when it found two speakers, and otherwise clusters again with two. On the 72 test conversations that lowers
DER from 20.3 % (count found automatically, as published) to **14.8 %**, and it never merges two people into one.
With "Count them" it said "1 speaker" for 13 of the 36 two-person test conversations
([scripts/eval_speaker_count.py](../scripts/eval_speaker_count.py), `results/speaker_count_test.json`).

**The report** needs the whole 6 GB GPU (D33), so the app closes the speech models, starts llama-server, writes the
report and loads the speech models again: about 2 minutes in all (on a 1-minute take: ~20 s to start the LLM,
~50 s to write, ~35 s to reload). Talk over it or skip it if time is short.

## Before the session

- Charger plugged in. On battery the GPU slows down.
- Nothing else on the GPU (`nvidia-smi` should show almost 0 MiB used before you start).
- Windows sound settings → input: Microphone Array, volume 80–100 %. If the mic has "Audio enhancements" or "Voice
  focus", turn them off: they can suppress whoever is not straight in front of the laptop.
- Start the app 2 minutes early, in the room, and do one full take with both of you. Check that the level bar
  moves for both voices.
- **Test recording** plays the flash-talk test conversation (two speakers, heavy overlap, market noise) through
  the speakers and feeds it in as if it came from the mic. It is the fallback if the room mic misbehaves.

## The take (about a minute)

1. Language **Hindi + English**, People **2**, source **Microphone**. Laptop between you, about half a metre away.
2. **Start** (or the space bar).
3. **First 15 seconds: two full sentences each, taking turns.** Clean samples of each voice are what the
   clustering needs. Short "haan / ok" replies at the start are what makes it wobble.
4. Then talk normally for 20 to 30 seconds. **Talk over each other twice, for 2 or 3 seconds each**: hatched
   stripes appear, and those lines get the "separated" tag.
5. **Stop** at 45 to 60 seconds. The final pass takes about 3 seconds.
6. Play **Speaker A alone**, then **Speaker B alone**.
7. Optional: **Write the report** (Hindi).
8. Optional second take in **Kannada**: switching language reloads the recogniser (about 10 seconds). Kannada lines
   appear in Kannada script only: no romanised line yet (the romaniser is Hindi-only until Abhinav's languages PR is
   merged), and no report (Airavata reads Hindi only).

Pure English also goes through the Hindi model. English words inside Hindi come out well, but whole English
sentences come out as Devanagari spellings of English. Keep it Hinglish.

## If something goes wrong

| Symptom | Fix |
|---|---|
| "Could not open the microphone" | `python demo\live.py --list-mics`, then restart with `--mic <number>` |
| The level bar does not move | Wrong input device, or the mic is muted in Windows |
| A take went badly | **Clear**, then start again |
| The report failed | The reason is in the status line; llama-server's log is `%TEMP%\whospoke_live_llm.log`. The speech models reload either way. |
| Anything stuck | Ctrl+C in the terminal and start it again (about 30 s) |

## How it is built

`live.py` is a small local web server. It records the microphone, re-runs the pipeline and serves `static/index.html`,
which polls it. The speech models (Order B: our spectral diarizer with the tuned settings, Conv-TasNet on the
overlaps, IndicConformer) run in a separate process, so they can be closed to give the LLM the GPU. Nothing in
`src/` is changed by the demo.
