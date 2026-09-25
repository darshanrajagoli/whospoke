"""Readers for the speech corpora named in the proposal (IndicVoices, Project Vaani).

Both are distributed on Hugging Face as parquet shards whose ``audio`` column is a struct of
encoded bytes; we read them with pyarrow directly (no ``datasets`` audio decoding, which needs
FFmpeg/torchcodec on Windows).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .audio import SR, load
from .paths import PROJECT, RAW  # noqa: F401  (re-exported)

INDICVOICES_REPO = "ai4bharat/IndicVoices"
VAANI_REPO = "ARTPARK-IISc/Vaani"


def fetch(repo: str, filename: str, subdir: str) -> Path:
    """Download one dataset shard into ``<data>/raw/<subdir>`` (no-op when already present)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset", local_dir=RAW / subdir))


# ---------------------------------------------------------------- IndicVoices

_CHUNK_RE = re.compile(r"^(?P<rec>.+)_chunk_(?P<idx>\d+)\.\w+$")


def read_indicvoices(parquet: str | Path, with_audio: bool = False) -> pd.DataFrame:
    """One row per utterance. Adds ``recording`` and ``chunk`` parsed from the file name.

    Conversation-scenario files are named ``<recording>_chunk_<n>.flac``: consecutive chunks of one
    speaker's side of a real phone conversation, in order — which lets us rebuild natural turns.
    """
    cols = ["text", "duration", "speaker_id", "scenario", "task_name", "gender", "district", "audio_filepath"]
    table = pq.read_table(parquet, columns=cols)
    df = table.to_pandas()
    df["path"] = df.audio_filepath.map(lambda a: a["path"])
    if with_audio:
        df["audio_bytes"] = df.audio_filepath.map(lambda a: a["bytes"])
    df = df.drop(columns="audio_filepath")
    parsed = df.path.str.extract(_CHUNK_RE)
    df["recording"] = parsed["rec"].fillna(df.path)
    df["chunk"] = pd.to_numeric(parsed["idx"], errors="coerce").fillna(0).astype(int)
    df["text"] = df.text.fillna("").map(clean_transcript)
    return df


_TAGS = re.compile(r"<[^>]*>|\[[^\]]*\]|\b(?:unintelligible|inaudible|noise)\b", re.IGNORECASE)


def clean_transcript(text: str) -> str:
    """Drop annotation tags (``<noise>``, ``[breathing]``, ``unintelligible``) — they are not spoken words."""
    return " ".join(_TAGS.sub(" ", text).split())


@dataclass
class Utterance:
    speaker: str
    text: str
    wav: np.ndarray

    @property
    def duration(self) -> float:
        return len(self.wav) / SR


def conversation_sides(df: pd.DataFrame, min_chunks: int = 4, min_total_s: float = 25.0) -> pd.DataFrame:
    """Recordings from the *Conversation* scenario long enough to supply several turns.

    Returns one row per recording: speaker, gender, n_chunks, total duration.
    """
    conv = df[(df.scenario == "Conversation") & (df.text.str.len() > 0)]
    g = conv.groupby("recording").agg(
        speaker_id=("speaker_id", "first"),
        gender=("gender", "first"),
        n_chunks=("chunk", "size"),
        total_s=("duration", "sum"),
    )
    return g[(g.n_chunks >= min_chunks) & (g.total_s >= min_total_s)].reset_index()


def recording_utterances(df: pd.DataFrame, recording: str) -> list[Utterance]:
    """Decode all chunks of one recording, in order. ``df`` must be read ``with_audio=True``."""
    rows = df[df.recording == recording].sort_values("chunk")
    return [Utterance(r.speaker_id, r.text, load(r.audio_bytes)) for r in rows.itertuples() if r.text]


# ---------------------------------------------------------------- Vaani

def read_vaani(parquet: str | Path, with_audio: bool = False) -> pd.DataFrame:
    cols = ["duration", "speakerID", "gender", "district", "isTranscriptionAvailable", "transcript"]
    if with_audio:
        cols.append("audio")
    df = pq.read_table(parquet, columns=cols).to_pandas()
    if with_audio:
        df["audio_bytes"] = df.audio.map(lambda a: a["bytes"])
        df = df.drop(columns="audio")
    df["transcribed"] = df.isTranscriptionAvailable.astype(str).str.lower().isin(["yes", "true", "1"])
    return df


_BRACE = re.compile(r"(\S+)\s*\{([^}]*)\}")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


@dataclass
class VaaniTranscript:
    text: str                    # spoken words in Devanagari, tags removed
    english: dict[int, str]      # word index in ``text`` → English spelling, for code-switched words


def parse_vaani_transcript(raw: str) -> VaaniTranscript:
    """Vaani marks code-switching inline: ``स्मार्ट {smart}`` (English word + spelling) and
    ``बोहोत {बहुत}`` (dialect form + standard form). We keep the standard Devanagari form and
    remember which words are English."""
    english_spans: list[tuple[str, str]] = []

    def sub(m: re.Match) -> str:
        word, note = m.group(1), m.group(2).strip()
        if _DEVANAGARI.search(note):
            return note                      # dialect → standard spelling
        english_spans.append((word, note))
        return f"\x00{word}"                 # mark English word
    text = _BRACE.sub(sub, raw)
    text = _TAGS.sub(" ", text).replace("--", " ")
    words, english = [], {}
    for tok in text.split():
        if tok.startswith("\x00"):
            tok = tok[1:]
            eng = next((e for w, e in english_spans if w == tok), None)
            if eng:
                english[len(words)] = eng
        tok = re.sub(r"[।.,!?\"']", "", tok)
        if tok:
            words.append(tok)
    return VaaniTranscript(" ".join(words), english)


def read_vaani_transcribed(parquet: str | Path, with_audio: bool = False) -> pd.DataFrame:
    cols = ["transcript", "gender", "state", "district"] + (["audio"] if with_audio else [])
    df = pq.read_table(parquet, columns=cols).to_pandas()
    parsed = df.transcript.map(parse_vaani_transcript)
    df["text"] = parsed.map(lambda p: p.text)
    df["english"] = parsed.map(lambda p: p.english)
    if with_audio:
        df["audio_bytes"] = df.audio.map(lambda a: a["bytes"])
        df = df.drop(columns="audio")
    return df
