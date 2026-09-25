"""Grow the Hinglish loanword lexicon from Vaani's TRAIN transcripts.

Vaani annotators write every English word spoken inside Hindi as ``<Devanagari> {english}``
(e.g. ``बिल्डिंग {building}``). Mining those pairs from the *train* split gives a large, human-made
Devanagari → English lexicon; the *test* split is kept untouched for measuring the romaniser.

A pair is kept when the Devanagari form was seen ≥ 2 times, ≥ 70 % of its annotations agree on
the English spelling, and it is tagged as English in ≥ 50 % of all its appearances — so Hindi words
that merely *sound* like an English word (``बस`` "enough" vs "bus") are not romanised as English.
Hand-curated entries always win over mined ones, and the hand curation also acts as a block-list: it
covered the 2,500 most frequent words of IndicVoices' natural conversations, so any of those words that
was judged *Hindi* (e.g. ``बस``, which Vaani's picture descriptions mostly use as "bus") is never mined.

    python scripts/mine_loanwords.py
"""
from __future__ import annotations

import collections
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whospoke.corpora import _BRACE, _DEVANAGARI, read_indicvoices  # noqa: E402
from whospoke.metrics import normalise  # noqa: E402
from whospoke.paths import RAW  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
SRC = RAW / "vaani_transcribed" / "hindi_train_transcripts_sample.csv"
LEXICON = PROJECT / "src" / "whospoke" / "resources" / "loanwords.tsv"
MINED_HEADER = "# ---- mined automatically from Vaani train transcripts (scripts/mine_loanwords.py) ----"


def main() -> None:
    df = pd.read_csv(SRC)
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for raw in df.transcript.dropna():
        for m in _BRACE.finditer(raw):
            dev, eng = m.group(1), m.group(2).strip()
            dev = re.sub(r"[।.,!?\"'\-]", "", unicodedata.normalize("NFC", dev))
            if not dev or _DEVANAGARI.search(eng) or not _DEVANAGARI.search(dev):
                continue
            eng = eng.strip(" .,-").lower()
            if not re.fullmatch(r"[a-z][a-z .'-]*", eng) or len(eng) > 30:
                continue
            counts[dev][eng] += 1

    # how often each Devanagari form appears at all (tagged or not)
    occurrences: collections.Counter = collections.Counter()
    for raw in df.transcript.dropna():
        plain = re.sub(r"\{[^}]*\}", " ", raw)
        for tok in plain.split():
            occurrences[re.sub(r"[।.,!?\"'\-]", "", unicodedata.normalize("NFC", tok))] += 1

    text = LEXICON.read_text(encoding="utf-8")
    manual_part = text.split(MINED_HEADER)[0].rstrip("\n")
    manual = {line.split("\t")[0] for line in manual_part.splitlines() if line and not line.startswith("#")}
    iv = collections.Counter()
    for shard in ("valid-00000-of-00001", "train-00000-of-00082"):
        for t in read_indicvoices(RAW / "indicvoices" / "hindi" / f"{shard}.parquet").text:
            iv.update(normalise(t).split())
    hindi_blocklist = {w for w, _ in iv.most_common(2500)} - manual
    mined = []
    for dev, c in counts.items():
        if dev in hindi_blocklist:
            continue
        eng, n = c.most_common(1)[0]
        total = sum(c.values())
        tagged_share = total / max(occurrences[dev], total)
        if total >= 2 and n / total >= 0.7 and tagged_share >= 0.5 and dev not in manual:
            mined.append((dev, eng, total))
    mined.sort(key=lambda x: -x[2])
    lines = [manual_part, "", MINED_HEADER] + [f"{d}\t{e}" for d, e, _ in mined]
    LEXICON.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"transcripts: {len(df)} | distinct tagged forms: {len(counts)} | mined entries added: {len(mined)} "
          f"| manual entries: {len(manual)}")
    print("top mined:", ", ".join(f"{d}→{e}" for d, e, _ in mined[:25]))


if __name__ == "__main__":
    main()
