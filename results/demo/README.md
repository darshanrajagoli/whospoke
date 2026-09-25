# Example output

`python -m whospoke run <data>/synth/test/test_g02_2spk_ovl-high_market5/mixture.wav --out results/demo`

The input is a 62 s test conversation: 2 speakers, heavy overlap, 5 dB market noise. It was chosen because its
who-said-what error (cpWER 49 %) is close to the test-set median (46.5 %), so this is a typical case, not the best
one. The audio is not in the repository; rebuild it with `scripts/build_dataset.py`.

| file | content |
|---|---|
| `timeline.json` | who spoke when (Stage 2) |
| `transcript.txt` | Devanagari transcript with speakers and times (Stage 3) |
| `transcript_hinglish.txt` | the same in Hinglish (Latin script) |
| `transcript.srt` | subtitles |
| `transcript.json` | everything above plus per-stage timings and GPU memory |
