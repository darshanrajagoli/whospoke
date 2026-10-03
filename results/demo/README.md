# Example output

`python -m whospoke run <data>/synth/test/test_g02_2spk_ovl-high_market5/mixture.wav --out results/demo`, then
`python -m whospoke postprocess results/demo/transcript.json` (Stage 4, with `scripts/serve_llm.py` running)

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
| `report.md` | Stage 4: summary, topic, keywords, action items, speakers, repaired and translated dialogue |
| `report.json` | the same, machine-readable, with the guardrail counts |

## How good is the Stage-4 report?

`report.md` was made on a CPU (4 cores) by the 4-bit Airavata: 4 LLM calls, 281 s for 63 s of audio (about 58 s on
the laptop GPU). The GPU build words the report a little differently, because its arithmetic differs in the last bits
([results/crosscheck_g00_gpu](../crosscheck_g00_gpu/README.md)). Read against the transcript, it is a rough reading
aid, not a trustworthy summary:

- **Right:** the lost ID card (line 6), the question about office hours, the CCTV cameras, the application to write.
  The only repair kept is a sensible one (`आप आप` → `आप`); the guardrails reverted two "repairs" that dropped words.
- **Wrong:** the summary's "e-pass" comes from mistranslating line 8 (एग्जाम पास आ रहे हैं, "the exams are near"),
  and "base card" is आधार (Aadhaar) translated literally. A key point gives Speaker A's question to Speaker B. Lines 6,
  8, 11 and 14 are mistranslated; lines 2 and 3 have no English (the model gave the same sentence for both, so both
  were dropped).
- **Keywords:** two are whole phrases rather than keywords, and "ई-पास" passed the "was it said?" check because पास
  does occur in line 8, with another meaning.

The Devanagari, Hinglish, speaker and time columns are the Stage 1–3 output (one word removed), so they can be
trusted as much as Stages 1–3. The English column and the summary need checking against them. Test-set numbers for
Stage 4 are in [docs/RESULTS.md](../../docs/RESULTS.md#stage-4--llm-post-processing).
