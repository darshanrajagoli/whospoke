# ASR backends: notes

Module: `src/whospoke/asr_backends.py`. Tests: `tests/test_asr_backends.py`. Benchmark: `scripts/bench_asr_backends.py` → `results/asr_backend_bench.json`.

```python
from whospoke.asr_backends import IndicConformerASR, IndicWav2VecASR
asr = IndicConformerASR(decoding="rnnt")          # or "ctc"; device=None -> cuda if available
texts = asr.transcribe([wav1, wav2], sr=16000)    # mono float32 arrays -> Devanagari strings
asr.close()                                       # frees GPU memory
```

Both backends share the same input handling:
- Audio is resampled to 16 kHz if needed, and stereo is averaged to mono.
- Clips shorter than 0.1 s return `""` without calling the model.
- Clips longer than `max_chunk_s` (30 s for IndicConformer, 20 s for IndicWav2Vec) are cut at the quietest 25 ms frame within the last 5 s before each limit. The pieces' texts are joined back together. This was tested on 60 s inputs.
- Batches are sorted longest-first.

## IndicConformer (`ai4bharat/indic-conformer-600m-multilingual`)

**What the repo contains.** The model is a hybrid CTC + RNNT Conformer covering 22 languages. It is exported as:
- a TorchScript mel preprocessor (`assets/preprocessor.ts`)
- ONNX graphs: `encoder`, `ctc_decoder`, `rnnt_decoder`, `joint_enc`, `joint_pred`, `joint_pre_net`, and one `joint_post_net_<lang>` per language
- `vocab.json` and `language_masks.json`

The model card's way in is `AutoModel.from_pretrained(..., trust_remote_code=True)`, which loads `model_onnx.py`, followed by `model(wav, "hi", "ctc" | "rnnt")`.

**How we load it.** We skip that wrapper and load the same assets directly with `onnxruntime`. The wrapper has three problems here:
- no device control and no batching
- it loads every language head
- it creates CUDA sessions with default options, which is unusably slow on this GPU (see below)

The decoding logic mirrors `model_onnx.py`:
- **CTC:** the 5633-way output is masked to the 257 Hindi tokens; blank is local id 256.
- **RNNT (greedy):** decoding starts from SOS = 5632 (the global blank), and the local ids 0–255 from the Hindi joint head are fed back into the prediction network. The remote code uses the class default SOS=5632, not the `SOS: 256` in `config.json`.

Our transcripts are **identical** to the official wrapper's on 4 test utterances, for both decodings, on both CPU and GPU.

**Windows / GPU gotchas**

1. **onnxruntime-gpu version.** The global `onnxruntime` 1.27 is CPU-only. We installed `onnxruntime-gpu==1.19.2` into the venv with `--no-deps`, so numpy is untouched and the global install is shadowed only inside the venv. Other versions failed:
   - `onnxruntime-gpu` 1.27 is built for CUDA 13, but torch cu121 ships CUDA 12 DLLs.
   - 1.20.1, the version the model card pins, is no longer on PyPI.
   - 1.20.2 and 1.22 load, but their cuDNN-frontend Conv runs in "Fallback mode" against torch's cuDNN 9.1 and is about 15x slower.
   - 1.19.2 works. Import `torch` before `onnxruntime` so that ORT reuses torch's CUDA and cuDNN DLLs. The module does this.
2. **CUDA EP options matter a lot.** Timings are for 10 s of audio:
   - With the default `cudnn_conv_algo_search=EXHAUSTIVE`, cuDNN re-benchmarks every new input length, taking about 50 s each time.
   - With `HEURISTIC` or `DEFAULT` alone, cuDNN picks a slow kernel for the 24 depthwise Conv1d layers: about 0.9 s.
   - With `DEFAULT` plus `cudnn_conv1d_pad_to_nc1d=1`, it takes 0.05 s. This is the setting the module uses.
3. **The preprocessor's normalisation includes zero padding.** Batching raw audio therefore changes the features, and in one case changed a word (ज़रूर became जरूर). We compute features per utterance and pad only the features; the encoder masks padded frames correctly.
4. The first few calls include about 5–8 s of one-off CUDA/cuDNN warm-up. Model load takes about 10 s.
5. The RNNT per-step networks are tiny, so they run on CPU; only the encoder and `joint_enc` run on GPU. This avoids a host-to-device copy on every decoding step.
6. Weights are fp32, about 2.4 GB on GPU.

## IndicWav2Vec (`ai4bharat/indicwav2vec-hindi`)

The model is wav2vec2-large with a CTC head, decoded greedily.

**Windows / loading gotchas**

1. **transformers refuses the checkpoint.** transformers 4.57 will not `torch.load` a `pytorch_model.bin` on torch < 2.6 (CVE-2025-32434), and the repo has no safetensors file. We load the state dict ourselves with `torch.load(weights_only=True)`, which is safe, and pass it to `Wav2Vec2ForCTC.from_pretrained(None, config=..., state_dict=...)`. That call handles the weight-norm `weight_g`/`weight_v` renaming. An unmerged SFconvertbot safetensors PR (`revision="refs/pr/1"`) would be an alternative.
2. **Load the plain processor.** `preprocessor_config.json` names `Wav2Vec2ProcessorWithLM`, but the repo ships no LM, so `Wav2Vec2Processor` is loaded explicitly.
3. **fp16 is on by default on GPU.** It gives the same WER/CER as fp32 at about 2.4x the speed and half the memory.
4. **No KenLM language model.**
   - `kenlm` has no Windows wheel (sdist only), and building it needs MSVC plus CMake, which are not installed.
   - The HF repo ships no LM anyway: the model card says "doesn't support inference with Language Model".
   - `pyctcdecode` 0.5.0 is present, but without kenlm it would only add LM-free beam search, so it was skipped.

## Benchmark

**Setup:**
- Data: 100 random IndicVoices-Hindi *valid* utterances (seed 0, 1–20 s long, 680.9 s of audio in total).
- Metrics: corpus-level `jiwer` WER and CER.
- Normalisation: `<tags>` such as `<unintelligible>` removed, punctuation and symbols replaced by spaces, whitespace collapsed.
- RTF = processing time ÷ audio duration, measured after a 5-utterance warm-up; model load is excluded.
- Peak GPU memory is for the benchmark process only, taken from the Windows *GPU Process Memory* counter and including the CUDA context. Device-wide numbers were unusable because another process was using the GPU.
- Batch size is 4; RNNT decodes one utterance at a time.
- Hardware and software: RTX 3050 6 GB Laptop GPU, torch 2.5.1+cu121, onnxruntime-gpu 1.19.2.

| backend | decoding | device | WER | CER | RTF | peak GPU MB |
|---|---|---|---|---|---|---|
| IndicConformer | CTC | cuda | 15.9 % | 6.1 % | 0.006 | 2929 |
| IndicConformer | RNNT | cuda | **14.1 %** | **5.8 %** | 0.037 | 2631 |
| IndicWav2Vec | greedy CTC (fp16) | cuda | 35.1 % | 14.4 % | 0.007 | 2529 |
| IndicConformer | CTC | cpu | 15.9 % | 6.1 % | 0.127 | – |
| IndicConformer | RNNT | cpu | 14.1 % | 5.8 % | 0.185 | – |
| IndicWav2Vec | greedy CTC (fp32) | cpu | 35.2 % | 14.5 % | 0.265 | – |

**Notes on the numbers:**
- CTC peaks higher than RNNT because it encodes 4 utterances per batch.
- Sorting batches longest-first instead of shortest-first cut peak memory from 4.2 GB to 2.9 GB (CTC) and from 4.4 GB to 2.5 GB (wav2vec).
- Ignoring the nukta (ज़ vs ज) lowers WER by about 1 point for every backend; the ranking is unchanged.
- **Caveat:** IndicConformer was trained on IndicVoices (other utterances from the same corpus), so this test is in-domain for it and not for IndicWav2Vec. Expect a smaller gap on out-of-domain audio such as Vaani.

## Recommendation

**Make IndicConformer the default backend.**
- It is more accurate than IndicWav2Vec: WER is less than half (14–16 % vs 35 %) and CER is about 2.5x lower.
- IndicWav2Vec produces many malformed words, for example `दीजिेगा` and `पहन कर` for "पैन कार्ड".
- On GPU the speed is the same as IndicWav2Vec with CTC decoding, and it is the faster of the two on CPU.
- Its peak memory (about 2.6–2.9 GB) fits alongside other work on the 6 GB card.

**Decoding:**
- **CTC** is about 6x faster (RTF 0.006). Use it for tuning sweeps and CPU/Colab runs.
- **RNNT** is about 2 WER points better and still about 27x faster than real time on the GPU. Use `decoding="rnnt"` for the final reported results.
- The class default stays `"ctc"`; `pipeline.make_asr()` passes `decoding="rnnt"`, so the pipeline and every
  reported result use RNNT.
