# test_m06_ovl-low_village10

begada-sreevidya (raga begada, singer sreevidya), mridangam tuned to B; overlap level ovl-low (16% of playing time both play), noise village10, 48 s. Picked as the **median** Demucs piece of this condition on the test set (a typical example, not the best).

SI-SDR of each output against the source it should match (higher is better):

| audio | vs true voice | vs true mridangam |
|---|---|---|
| mixture (no separation) | +2.6 dB | -4.7 dB |
| demucs | +7.5 dB | +10.1 dB |
| hpss | +5.1 dB | +1.2 dB |
| convtasnet (out1 = voice, out0 = drum) | -1.4 dB | +0.7 dB |

Audio sources (licences require attribution):
- Singing: Carnatic Varnam dataset (Koduri, Ishwar, Serrà & Serra, 2014, "Intonation analysis of ragas in Carnatic
  music", JNMR), recordings by gopalkoduri on Freesound (pack 14136), CC BY 3.0. Excerpts were cut and mixed.
- Mridangam: Mridangam Stroke dataset (Anantapadmanabhan, Bellur & Murthy, ICASSP 2013), CompMusic, CC BY-NC 4.0.
  Single strokes were arranged into patterns and mixed. Non-commercial use only.
- Noise (village/market pieces): DEMAND (CC BY 4.0) and ESC-50 (CC BY-NC 3.0).
These mixtures were made by the WhoSpoke project for evaluation; they are not real performances.
