"""whospoke — a "who spoke what and when" pipeline for noisy, overlapping, code-switched Hindi/Hinglish audio,
ending in an LLM-written report (separation, diarization, ASR, LLM post-processing). Stages 1–3 also run on the other
Indian languages IndicConformer covers (``language=``), but only Hindi is evaluated (DECISIONS D4, D40)."""

from . import compat as _compat

_compat.apply()                  # no-op on the pinned torch 2.5.1; keeps newer torch/torchaudio (e.g. Colab) working

from .languages import LanguageInfo, resolve_language  # noqa: E402

__version__ = "0.2.0"
__all__ = ["LanguageInfo", "Line", "Pipeline", "Result", "make_asr", "resolve_language", "__version__"]


def __getattr__(name: str):
    if name in ("Line", "Pipeline", "Result", "make_asr"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
