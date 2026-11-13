# test_m00_ovl-high_market5

saveri-sreevidya (raga saveri, singer sreevidya), mridangam tuned to B; overlap level ovl-high (74% of playing time both play), noise market5, 56 s. Picked as the **median** Demucs piece of this condition on the test set (a typical example, not the best).

SI-SDR of each output against the source it should match (higher is better):

| audio | vs true voice | vs true mridangam |
|---|---|---|
| mixture (no separation) | -3.1 dB | -1.6 dB |
| demucs | +1.2 dB | +9.3 dB |
| hpss | -1.3 dB | +3.4 dB |
| convtasnet (out0 = voice, out1 = drum) | -8.9 dB | -0.4 dB |

Audio sources (licences require attribution):
- Singing: Carnatic Varnam dataset (Koduri, Ishwar, Serrà & Serra, 2014, "Intonation analysis of ragas in Carnatic
  music", JNMR), recordings by gopalkoduri on Freesound (pack 14136), CC BY 3.0. Excerpts were cut and mixed.
- Mridangam: Mridangam Stroke dataset (Anantapadmanabhan, Bellur & Murthy, ICASSP 2013), CompMusic, CC BY-NC 4.0.
  Single strokes were arranged into patterns and mixed. Non-commercial use only.
- Noise (village/market pieces): DEMAND (CC BY 4.0) and ESC-50 (CC BY-NC 3.0).
These mixtures were made by the WhoSpoke project for evaluation; they are not real performances.
