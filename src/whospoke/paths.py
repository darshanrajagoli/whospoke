"""Where the large, regenerable data lives (downloaded corpora + simulated conversations, ~5 GB).

It can sit outside the project folder (e.g. outside a synced OneDrive folder). Resolution order:

1. the ``WHOSPOKE_DATA`` environment variable,
2. a one-line file ``data_location.txt`` in the project folder containing the path,
3. ``<project>/data`` (the default, used on Colab and fresh clones).
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def _resolve() -> Path:
    env = os.environ.get("WHOSPOKE_DATA")
    if env:
        return Path(env)
    pointer = PROJECT / "data_location.txt"
    if pointer.exists():
        text = pointer.read_text(encoding="utf-8").strip()
        if text:
            return Path(text)
    return PROJECT / "data"


DATA = _resolve()
RAW = DATA / "raw"
SYNTH = DATA / "synth"
RESULTS = PROJECT / "results"


def conversation(rel: str | Path) -> Path:
    """Folder of a simulated conversation from its ``index.csv`` path (``synth/<split>/<id>``)."""
    parts = Path(str(rel).replace("\\", "/")).parts
    if parts and parts[0] == "data":          # older index files stored paths relative to the project
        parts = parts[1:]
    return DATA.joinpath(*parts)
