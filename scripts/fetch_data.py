"""Download every external corpus the experiments need (skips anything already present).

    python scripts/fetch_data.py

* DEMAND (Zenodo 1227121, CC-BY-4.0): channel 1 of eight 16 kHz environments → background-noise beds (noise.py)
* ESC-50 (GitHub, CC-BY-NC): environmental sound events → village / market events (noise.py)
* Project Vaani transcription part (Hugging Face, gated — accept the terms first):
    - Hindi test shards 0–1 with audio → the ASR model choice (scripts/eval_asr.py)
    - the transcript column only of every 8th Hindi train shard → Hinglish lexicon mining (scripts/mine_loanwords.py)

IndicVoices is downloaded by scripts/build_dataset.py itself.
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whospoke.corpora import fetch  # noqa: E402
from whospoke.paths import RAW  # noqa: E402

DEMAND_ENVS = ("STRAFFIC", "SPSQUARE", "PSTATION", "TBUS", "NFIELD", "NPARK", "DKITCHEN", "DLIVING")
DEMAND_URL = "https://zenodo.org/records/1227121/files/{env}_16k.zip?download=1"
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
VAANI_TX = "ARTPARK-IISc/Vaani-transcription-part"
VAANI_TEST_SHARDS = (0, 1)          # of 28
VAANI_TRAIN_SHARDS = range(0, 193, 8)  # 25 of 193; transcripts only


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=600) as r:
        return r.read()


def demand() -> None:
    for env in DEMAND_ENVS:
        target = RAW / "demand" / env / "ch01.wav"
        if target.exists():
            continue
        with zipfile.ZipFile(io.BytesIO(_download(DEMAND_URL.format(env=env)))) as z:
            member = next(n for n in z.namelist() if n.endswith("/ch01.wav"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(member))
        print("DEMAND", env, "ok")


def esc50() -> None:
    root = RAW / "esc50"
    if (root / "ESC-50-master" / "meta" / "esc50.csv").exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(_download(ESC50_URL))) as z:
        z.extractall(root)
    print("ESC-50 ok")


def vaani() -> None:
    for i in VAANI_TEST_SHARDS:
        fetch(VAANI_TX, f"audio/Hindi/test-{i:05d}-of-00028.parquet", "vaani_transcribed")
    out = RAW / "vaani_transcribed" / "hindi_train_transcripts_sample.csv"
    if out.exists():
        return
    import pandas as pd
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs, frames = HfFileSystem(), []
    for i in VAANI_TRAIN_SHARDS:                 # read one column remotely instead of downloading ~1 GB shards
        with fs.open(f"datasets/{VAANI_TX}/audio/Hindi/train-{i:05d}-of-00193.parquet", "rb", block_size=2**20) as f:
            frames.append(pq.read_table(f, columns=["transcript"]).to_pandas())
        print("Vaani train shard", i, len(frames[-1]), flush=True)
    pd.concat(frames).to_csv(out, index=False)


if __name__ == "__main__":
    demand()
    esc50()
    vaani()
    print("all corpora present under", RAW)
