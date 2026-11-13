"""whospoke — a "who spoke what and when" pipeline for noisy, overlapping, code-switched Hindi audio,
ending in an LLM-written report (separation, diarization, Hindi/Hinglish ASR, LLM post-processing)."""

__version__ = "0.1.0"

from . import compat as _compat  # noqa: E402

_compat.apply()                  # no-op on the pinned torch 2.5.1; keeps newer torch/torchaudio (Colab) working
