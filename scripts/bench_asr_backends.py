"""Benchmark the Hindi ASR backends on IndicVoices-Hindi valid utterances.

Samples 100 utterances (seed 0, 1-20 s, non-empty normalised reference) and
reports corpus WER / CER (jiwer, after removing ``<tags>``, punctuation and
extra whitespace), real-time factor and peak GPU memory for each backend.
Each configuration runs in a fresh subprocess so memory numbers are isolated.

    python scripts/bench_asr_backends.py                 # all configs
    python scripts/bench_asr_backends.py --configs indicconformer-ctc@cuda
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from whospoke.paths import RAW  # noqa: E402

VALID = RAW / "indicvoices/hindi/valid-00000-of-00001.parquet"
OUT = ROOT / "results/asr_backend_bench.json"
CONFIGS = ["indicconformer-ctc@cuda", "indicconformer-rnnt@cuda", "indicwav2vec@cuda",
           "indicconformer-ctc@cpu", "indicconformer-rnnt@cpu", "indicwav2vec@cpu"]
N_UTT, N_WARMUP, SEED, MIN_S, MAX_S = 100, 5, 0, 1.0, 20.0


def normalize(text: str) -> str:
    text = re.sub(r"<[^>]*>", " ", text)  # e.g. <unintelligible>
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def load_eval_set():
    import pandas as pd
    import soundfile as sf

    df = pd.read_parquet(VALID)
    df = df[(df.duration >= MIN_S) & (df.duration <= MAX_S) & (df.text.map(normalize) != "")]
    evalset = df.sample(N_UTT, random_state=SEED)
    warmup = df.drop(evalset.index).sample(N_WARMUP, random_state=SEED)
    decode = lambda d: [sf.read(io.BytesIO(a["bytes"]), dtype="float32")[0] for a in d.audio_filepath]
    return evalset, decode(evalset), decode(warmup)


class GpuPeak:
    """Peak GPU memory of *this process* (ORT + torch + CUDA context), in MB.

    On Windows (WDDM) nvidia-smi/NVML cannot report per-process usage, so the
    ``GPU Process Memory`` performance counter is streamed at 1 Hz. Both the ORT
    arena and torch's caching allocator hold on to their peak, so 1 Hz sampling
    does not miss it. Elsewhere, device-wide used memory is polled instead
    (which includes other processes).
    """

    def __init__(self):
        import torch

        self._torch, self.samples = torch, []
        self._stop = threading.Event()
        if os.name == "nt":
            counter = rf"\GPU Process Memory(pid_{os.getpid()}_*)\Dedicated Usage"
            ps = (f"Get-Counter -Counter '{counter}' -Continuous -SampleInterval 1 | ForEach-Object "
                  "{ [Console]::WriteLine(($_.CounterSamples | Measure-Object CookedValue -Sum).Sum) }")
            self._proc = subprocess.Popen(["powershell", "-NoProfile", "-Command", ps],
                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            target, self.scope = self._read_counter, "process"
        else:
            self._proc, target, self.scope = None, self._poll_device, "device"
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()
        self.wait_first_sample()
        self.baseline_mb = self.samples[-1]

    def _read_counter(self):
        for line in self._proc.stdout:
            try:
                self.samples.append(float(line) / 2**20)
            except ValueError:
                pass

    def _poll_device(self):
        while not self._stop.is_set():
            free, total = self._torch.cuda.mem_get_info()
            self.samples.append((total - free) / 2**20)
            time.sleep(0.05)

    def wait_first_sample(self, timeout: float = 30.0):
        t0 = time.time()
        while not self.samples and time.time() - t0 < timeout:
            time.sleep(0.1)
        if not self.samples:
            raise RuntimeError("no GPU memory samples")

    def stop(self) -> float:
        n = len(self.samples)
        t0 = time.time()
        while len(self.samples) < n + 2 and time.time() - t0 < 10:  # let final samples land
            time.sleep(0.1)
        self._stop.set()
        if self._proc is not None:
            self._proc.kill()
        return max(self.samples)


def run_one(config: str) -> dict:
    import jiwer
    import torch

    from whospoke.asr_backends import IndicConformerASR, IndicWav2VecASR

    backend, device = config.split("@")
    evalset, wavs, warm = load_eval_set()
    if device == "cuda":
        torch.zeros(1, device="cuda")  # create the CUDA context before the baseline
        free, total = torch.cuda.mem_get_info()
        monitor = GpuPeak()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    if backend == "indicwav2vec":
        asr = IndicWav2VecASR(device=device)
    else:
        asr = IndicConformerASR(device=device, decoding=backend.split("-")[1])
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    asr.transcribe(warm)
    warmup_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    hyps = asr.transcribe(wavs)
    proc_s = time.perf_counter() - t0
    audio_s = sum(len(w) for w in wavs) / 16000

    refs_n, hyps_n = [normalize(t) for t in evalset.text], [normalize(h) for h in hyps]
    res = {
        "config": config, "backend": asr.name, "device": device,
        "decoding": getattr(asr, "decoding", "ctc"),
        "wer": jiwer.wer(refs_n, hyps_n), "cer": jiwer.cer(refs_n, hyps_n),
        "rtf": proc_s / audio_s, "proc_s": proc_s, "audio_s": audio_s,
        "load_s": load_s, "warmup_s": warmup_s,
        "n_empty_hyp": sum(h == "" for h in hyps_n),
        "hyps": hyps,
    }
    if device == "cuda":
        res["peak_gpu_mem_mb"] = monitor.stop()
        res["gpu_mem_scope"] = monitor.scope
        res["cuda_context_mb"] = monitor.baseline_mb
        res["torch_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
        res["torch_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
        res["device_used_at_start_mb"] = (total - free) / 2**20
    asr.close()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=CONFIGS, help="backend[-decoding]@device")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.one:  # child process: print a single JSON result
        print("@@RESULT@@" + json.dumps(run_one(args.one), ensure_ascii=False))
        return

    import onnxruntime
    import pandas as pd
    import torch
    import transformers

    evalset, wavs, _ = load_eval_set()
    results = []
    for cfg in args.configs:
        print(f"[bench] {cfg} ...", flush=True)
        p = subprocess.run([sys.executable, __file__, "--one", cfg], capture_output=True,
                           text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        line = next((l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")), None)
        if line is None:
            print(p.stdout[-2000:], p.stderr[-4000:], sep="\n")
            raise SystemExit(f"{cfg} failed")
        r = json.loads(line[len("@@RESULT@@"):])
        results.append(r)
        mem = f"{r['peak_gpu_mem_mb']:.0f} MB" if "peak_gpu_mem_mb" in r else "-"
        print(f"  WER {r['wer']:.4f}  CER {r['cer']:.4f}  RTF {r['rtf']:.4f}  peak GPU {mem}", flush=True)

    samples = [{"row": int(i), "duration": float(d), "speaker_id": str(s), "ref": t,
                "hyp": {r["config"]: r["hyps"][k] for r in results}}
               for k, (i, d, s, t) in enumerate(zip(evalset.index, evalset.duration, evalset.speaker_id, evalset.text))]
    summary = [{k: v for k, v in r.items() if k != "hyps"} for r in results]
    meta = {
        "data": "<data>/raw/indicvoices/hindi/valid-00000-of-00001.parquet", "n_utterances": N_UTT, "seed": SEED,
        "duration_range_s": [MIN_S, MAX_S], "total_audio_s": sum(len(w) for w in wavs) / 16000,
        "normalisation": "remove <tags>, replace Unicode punctuation/symbols with space, collapse whitespace",
        "metrics": "corpus-level jiwer WER/CER; RTF = processing time / audio duration (after a "
                   f"{N_WARMUP}-utterance warm-up, model load excluded); peak_gpu_mem_mb = peak dedicated GPU "
                   "memory of the benchmark process incl. CUDA context (Windows 'GPU Process Memory' "
                   "counter; gpu_mem_scope='device' means device-wide used memory instead)",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "onnxruntime": onnxruntime.__version__,
        "transformers": transformers.__version__, "pandas": pd.__version__,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"meta": meta, "results": summary, "samples": samples},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[bench] wrote {OUT}")


if __name__ == "__main__":
    main()
