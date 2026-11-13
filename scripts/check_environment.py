"""One-minute check that every model the A3 notebook needs loads and runs on this machine (run before long jobs).

Checks: torch + GPU, Conv-TasNet (separation), pyannote segmentation (speech/overlap detection), WeSpeaker (voice
fingerprints) and IndicConformer (speech-to-text), each on a few seconds of synthetic audio, and reports whether the
speech-to-text really runs on the GPU. Exit code 1 if any model fails.

    python scripts/check_environment.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import whospoke  # noqa: E402,F401  (applies the compatibility shims)


def main() -> None:
    import torch

    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(16000 * 3) * 0.05).astype(np.float32)
    failed = []

    def check(name, fn):
        t = time.perf_counter()
        try:
            msg = fn()
            print(f"  OK    {name:<34} {time.perf_counter() - t:5.1f} s  {msg or ''}")
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name:<34} {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc(limit=3)

    def sep():
        from whospoke.separation import Separator
        return f"output {Separator('convtasnet').separate(x).shape}"

    def seg():
        from whospoke import vad
        return f"{len(vad.speech_regions(x))} speech / {len(vad.overlap_regions(x))} overlap regions"

    def spk():
        from whospoke.diarization import embed
        return f"embedding {embed(x, [(0.0, 1.5)]).shape}"

    def asr():
        from whospoke.pipeline import make_asr
        a = make_asr("indicconformer")
        a.transcribe([x[:32000]])
        prov = a.encoder.get_providers()
        gpu = "CUDAExecutionProvider" in prov
        if torch.cuda.is_available() and not gpu:
            print("        WARNING: speech-to-text is on the CPU (onnxruntime found no CUDA): the transcript steps "
                  "will be much slower, but results are the same.")
        return f"encoder on {'GPU' if gpu else 'CPU'} ({prov[0]})"

    check("Conv-TasNet separation", sep)
    check("pyannote segmentation (VAD/overlap)", seg)
    check("WeSpeaker voice fingerprints", spk)
    check("IndicConformer speech-to-text", asr)
    print("ALL OK" if not failed else f"FAILED: {', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
