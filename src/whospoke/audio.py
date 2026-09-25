"""Small audio utilities shared by every stage (I/O, resampling, levels, mixing)."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000  # every stage of the pipeline works at 16 kHz mono


def load(path_or_bytes: str | Path | bytes, sr: int = SR) -> np.ndarray:
    """Read audio (file path or encoded bytes) as mono float32 at ``sr``."""
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else str(path_or_bytes)
    wav, file_sr = sf.read(src, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    return resample(wav, file_sr, sr)


def save(path: str | Path, wav: np.ndarray, sr: int = SR) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(wav, dtype=np.float32), sr, subtype="PCM_16")


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return np.asarray(wav, dtype=np.float32)
    from scipy.signal import resample_poly

    g = np.gcd(orig_sr, target_sr)
    return resample_poly(wav, target_sr // g, orig_sr // g).astype(np.float32)


def rms(wav: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.sqrt(np.mean(np.square(wav, dtype=np.float64)) + eps))


def active_rms(wav: np.ndarray, frame: int = 400, top_db: float = 40.0) -> float:
    """RMS over frames within ``top_db`` of the loudest frame, so silence does not dilute speech level."""
    n = len(wav) // frame
    if n == 0:
        return rms(wav)
    frames = wav[: n * frame].reshape(n, frame)
    e = np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12
    keep = 10 * np.log10(e) > 10 * np.log10(e.max()) - top_db
    return float(np.sqrt(e[keep].mean()))


def db_to_gain(db: float) -> float:
    return float(10 ** (db / 20))


def scale_to_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Return ``noise`` rescaled so that ``signal`` / noise has the requested SNR (active-speech level)."""
    return noise * (active_rms(signal) / (rms(noise) * db_to_gain(snr_db)))


def peak_normalise(wav: np.ndarray, peak: float = 0.9) -> tuple[np.ndarray, float]:
    """Scale so max |x| == ``peak``; returns (scaled, gain) so companion signals can share the gain."""
    m = float(np.max(np.abs(wav))) if len(wav) else 0.0
    g = peak / m if m > 0 else 1.0
    return (wav * g).astype(np.float32), g


def fmt_ts(seconds: float) -> str:
    """Seconds → ``MM:SS`` (or ``H:MM:SS``), the format used in the proposal's timeline example."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
