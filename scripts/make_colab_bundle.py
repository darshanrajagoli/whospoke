"""Pack the light upload for the A3 Colab notebook (notebooks/03_finetune_convtasnet_colab.ipynb).

The full data folder is ~6.5 GB; the notebook needs about a fifth of it. The bundle holds:

* the code (src/, scripts/, the notebook, requirements-colab.txt) and results/tuned_params.json, plus the cached dev
  diarization features (saves recomputing them on Colab);
* the speech dev and test conversations (Data/synth/{dev,test}: mixtures, clean sources, noise, references);
* the dev halves' noise sources: the 8 DEMAND recordings used, and ESC-50's meta file plus the clips of the
  village/market categories in dev folds 4-5 (training noise; the test folds are never needed: the test conversations
  already contain their noise).

Audio is stored as FLAC (lossless; every file is checked to decode back to the identical 16-bit samples), named
``*.wav.flac``; the notebook turns them back into ``.wav`` after unzipping. IndicVoices is NOT included: the notebook
downloads the two parquet files it needs on Colab, which is much faster than uploading them.

    python scripts/make_colab_bundle.py                 # -> ../whospoke_colab_bundle.zip (next to the repo)
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whospoke.noise import SOUNDSCAPES  # noqa: E402
from whospoke.paths import DATA, RAW, SYNTH  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
CODE = ["src", "scripts", "notebooks/03_finetune_convtasnet_colab.ipynb", "requirements-colab.txt", "pyproject.toml",
        "results/tuned_params.json"]


def add_flac(z: zipfile.ZipFile, src: Path, arc: str) -> int:
    pcm, sr = sf.read(src, dtype="int16", always_2d=False)
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="FLAC", subtype="PCM_16")
    back, _ = sf.read(io.BytesIO(buf.getvalue()), dtype="int16", always_2d=False)
    assert np.array_equal(back, pcm), f"FLAC round trip changed {src}"
    z.writestr(arc + ".flac", buf.getvalue(), compress_type=zipfile.ZIP_STORED)
    return len(buf.getvalue())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(PROJECT.parent / "whospoke_colab_bundle.zip"))
    args = ap.parse_args()
    files_audio: list[tuple[Path, str]] = []
    files_plain: list[tuple[Path, str]] = []
    for item in CODE:
        p = PROJECT / item
        for f in ([p] if p.is_file() else sorted(p.rglob("*"))):
            if f.is_file() and "__pycache__" not in f.parts:
                files_plain.append((f, f"whospoke/{f.relative_to(PROJECT).as_posix()}"))
    feats = PROJECT / "results" / "cache" / "features"
    for f in sorted(feats.glob("dev_*.pkl")):
        files_plain.append((f, f"whospoke/results/cache/features/{f.name}"))
    for split in ("dev", "test"):
        files_plain.append((SYNTH / split / "index.csv", f"whospoke/Data/synth/{split}/index.csv"))
        for d in sorted(p for p in (SYNTH / split).iterdir() if p.is_dir()):
            for f in sorted(d.rglob("*")):
                if f.is_file():
                    arc = f"whospoke/Data/{f.relative_to(DATA).as_posix()}"
                    (files_audio if f.suffix == ".wav" else files_plain).append((f, arc))
    beds = sorted({b for s in SOUNDSCAPES.values() for b in s.beds})
    for b in beds:
        files_audio.append((RAW / "demand" / b / "ch01.wav", f"whospoke/Data/raw/demand/{b}/ch01.wav"))
    esc = RAW / "esc50" / "ESC-50-master"
    meta = pd.read_csv(esc / "meta" / "esc50.csv")
    cats = {c for s in SOUNDSCAPES.values() for c in s.events}
    files_plain.append((esc / "meta" / "esc50.csv", "whospoke/Data/raw/esc50/ESC-50-master/meta/esc50.csv"))
    for fn in meta[meta.category.isin(cats) & meta.fold.isin([4, 5])].filename:
        files_audio.append((esc / "audio" / fn, f"whospoke/Data/raw/esc50/ESC-50-master/audio/{fn}"))

    out = Path(args.out)
    total = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f, arc in files_plain:
            z.write(f, arc)
        for i, (f, arc) in enumerate(files_audio, 1):
            total += add_flac(z, f, arc)
            if i % 200 == 0:
                print(f"  {i}/{len(files_audio)} audio files", flush=True)
    print(f"{out}: {out.stat().st_size / 2**20:.0f} MB ({len(files_plain)} files + {len(files_audio)} audio files as "
          f"FLAC, all checked bit-exact; ESC-50 clips: {sum('esc50/' in a for _, a in files_audio)}, "
          f"DEMAND recordings: {len(beds)})")


if __name__ == "__main__":
    main()
