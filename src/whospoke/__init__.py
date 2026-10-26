"""whospoke — a "who spoke what and when" pipeline for noisy, overlapping, code-switched Indian audio
covering 22 regional languages, ending in an LLM-written report (separation, diarization, regional ASR, LLM post-processing)."""

from .languages import LanguageInfo, resolve_language

__version__ = "0.2.0"
__all__ = ["LanguageInfo", "Line", "Pipeline", "Result", "make_asr", "resolve_language", "__version__"]


def __getattr__(name: str):
    if name in ("Line", "Pipeline", "Result", "make_asr"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

