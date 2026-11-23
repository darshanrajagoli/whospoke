# test_m02_ovl-high_clean

sri-ramakrishnamurthy (raga sri, singer ramakrishnamurthy), mridangam tuned to D; overlap level ovl-high (65% of playing time both play), noise clean, 55 s. Picked as the **median** Demucs piece of this condition on the test set (a typical example, not the best).

SI-SDR of each output against the source it should match (higher is better):

| audio | vs true voice | vs true mridangam |
|---|---|---|
| mixture (no separation) | +1.1 dB | -1.1 dB |
| demucs | +14.3 dB | +12.8 dB |
| hpss | +4.1 dB | +2.8 dB |
| convtasnet (out1 = voice, out0 = drum) | -1.9 dB | -9.5 dB |

Audio sources (licences require attribution):
- Singing: Carnatic Varnam dataset (Koduri, Ishwar, Serrà & Serra, 2014, "Intonation analysis of ragas in Carnatic
  music", JNMR), recordings by gopalkoduri on Freesound (pack 14136), CC BY 3.0. Excerpts were cut and mixed.
- Mridangam: Mridangam Stroke dataset (Anantapadmanabhan, Bellur & Murthy, ICASSP 2013), CompMusic, CC BY-NC 4.0.
  Single strokes were arranged into patterns and mixed. Non-commercial use only.
- Noise (village/market pieces): DEMAND (CC BY 4.0) and ESC-50 (CC BY-NC 3.0).
These mixtures were made by the WhoSpoke project for evaluation; they are not real performances.
