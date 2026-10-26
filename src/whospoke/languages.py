"""Regional languages supported by IndicVoices and AI4Bharat ASR models.

IndicVoices covers 22 official scheduled Indian languages:
Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani,
Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit,
Santali, Sindhi, Tamil, Telugu, and Urdu.

This module provides language metadata, script mappings, model repository links,
and alias resolution for all 4 milestones.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class LanguageInfo:
    """Metadata for an Indian language in IndicVoices and the ASR backends."""

    code: str                  # ISO 639-1 / 639-3 or AI4Bharat code (e.g. 'hi', 'bn', 'ta')
    name: str                  # English name (e.g. 'Hindi', 'Bengali', 'Tamil')
    indicvoices_name: str      # Subfolder/config in ai4bharat/IndicVoices (e.g. 'hindi', 'bengali')
    script: str                # Script family ('devanagari', 'bengali', 'gujarati', 'gurmukhi',
                               # 'odia', 'tamil', 'telugu', 'kannada', 'malayalam', 'arabic', 'ol_chiki')
    conformer_code: str        # Language key in ai4bharat/indic-conformer-600m-multilingual
    wav2vec_repo: str | None   # HF repo for IndicWav2Vec CTC, or None if no standalone checkpoint
    latin_name: str            # Romanised code-switch name (e.g. 'Hinglish', 'Benglish', 'Tanglish')


LANGUAGES: dict[str, LanguageInfo] = {
    "as": LanguageInfo(
        code="as",
        name="Assamese",
        indicvoices_name="assamese",
        script="bengali",
        conformer_code="as",
        wav2vec_repo=None,
        latin_name="Assamese (Latin)",
    ),
    "bn": LanguageInfo(
        code="bn",
        name="Bengali",
        indicvoices_name="bengali",
        script="bengali",
        conformer_code="bn",
        wav2vec_repo="ai4bharat/indicwav2vec_v1_bengali",
        latin_name="Benglish",
    ),
    "brx": LanguageInfo(
        code="brx",
        name="Bodo",
        indicvoices_name="bodo",
        script="devanagari",
        conformer_code="brx",
        wav2vec_repo=None,
        latin_name="Bodo (Latin)",
    ),
    "doi": LanguageInfo(
        code="doi",
        name="Dogri",
        indicvoices_name="dogri",
        script="devanagari",
        conformer_code="doi",
        wav2vec_repo=None,
        latin_name="Dogri (Latin)",
    ),
    "gu": LanguageInfo(
        code="gu",
        name="Gujarati",
        indicvoices_name="gujarati",
        script="gujarati",
        conformer_code="gu",
        wav2vec_repo="ai4bharat/indicwav2vec_v1_gujarati",
        latin_name="Gujlish",
    ),
    "hi": LanguageInfo(
        code="hi",
        name="Hindi",
        indicvoices_name="hindi",
        script="devanagari",
        conformer_code="hi",
        wav2vec_repo="ai4bharat/indicwav2vec-hindi",
        latin_name="Hinglish",
    ),
    "kn": LanguageInfo(
        code="kn",
        name="Kannada",
        indicvoices_name="kannada",
        script="kannada",
        conformer_code="kn",
        wav2vec_repo=None,
        latin_name="Kanglish",
    ),
    "ks": LanguageInfo(
        code="ks",
        name="Kashmiri",
        indicvoices_name="kashmiri",
        script="arabic",
        conformer_code="ks",
        wav2vec_repo=None,
        latin_name="Kashmiri (Latin)",
    ),
    "kok": LanguageInfo(
        code="kok",
        name="Konkani",
        indicvoices_name="konkani",
        script="devanagari",
        conformer_code="kok",
        wav2vec_repo=None,
        latin_name="Konkani (Latin)",
    ),
    "mai": LanguageInfo(
        code="mai",
        name="Maithili",
        indicvoices_name="maithili",
        script="devanagari",
        conformer_code="mai",
        wav2vec_repo=None,
        latin_name="Maithili (Latin)",
    ),
    "ml": LanguageInfo(
        code="ml",
        name="Malayalam",
        indicvoices_name="malayalam",
        script="malayalam",
        conformer_code="ml",
        wav2vec_repo=None,
        latin_name="Manglish",
    ),
    "mni": LanguageInfo(
        code="mni",
        name="Manipuri",
        indicvoices_name="manipuri",
        script="bengali",
        conformer_code="mni",
        wav2vec_repo=None,
        latin_name="Manipuri (Latin)",
    ),
    "mr": LanguageInfo(
        code="mr",
        name="Marathi",
        indicvoices_name="marathi",
        script="devanagari",
        conformer_code="mr",
        wav2vec_repo=None,
        latin_name="Marathi (Latin)",
    ),
    "ne": LanguageInfo(
        code="ne",
        name="Nepali",
        indicvoices_name="nepali",
        script="devanagari",
        conformer_code="ne",
        wav2vec_repo=None,
        latin_name="Nepali (Latin)",
    ),
    "or": LanguageInfo(
        code="or",
        name="Odia",
        indicvoices_name="odia",
        script="odia",
        conformer_code="or",
        wav2vec_repo="ai4bharat/indicwav2vec-odia",
        latin_name="Odia (Latin)",
    ),
    "pa": LanguageInfo(
        code="pa",
        name="Punjabi",
        indicvoices_name="punjabi",
        script="gurmukhi",
        conformer_code="pa",
        wav2vec_repo=None,
        latin_name="Punglish",
    ),
    "sa": LanguageInfo(
        code="sa",
        name="Sanskrit",
        indicvoices_name="sanskrit",
        script="devanagari",
        conformer_code="sa",
        wav2vec_repo=None,
        latin_name="Sanskrit (Latin)",
    ),
    "sat": LanguageInfo(
        code="sat",
        name="Santali",
        indicvoices_name="santali",
        script="ol_chiki",
        conformer_code="sat",
        wav2vec_repo=None,
        latin_name="Santali (Latin)",
    ),
    "sd": LanguageInfo(
        code="sd",
        name="Sindhi",
        indicvoices_name="sindhi",
        script="arabic",
        conformer_code="sd",
        wav2vec_repo=None,
        latin_name="Sindhi (Latin)",
    ),
    "ta": LanguageInfo(
        code="ta",
        name="Tamil",
        indicvoices_name="tamil",
        script="tamil",
        conformer_code="ta",
        wav2vec_repo="ai4bharat/indicwav2vec_v1_tamil",
        latin_name="Tanglish",
    ),
    "te": LanguageInfo(
        code="te",
        name="Telugu",
        indicvoices_name="telugu",
        script="telugu",
        conformer_code="te",
        wav2vec_repo="ai4bharat/indicwav2vec_v1_telugu",
        latin_name="Teluglish",
    ),
    "ur": LanguageInfo(
        code="ur",
        name="Urdu",
        indicvoices_name="urdu",
        script="arabic",
        conformer_code="ur",
        wav2vec_repo=None,
        latin_name="Roman Urdu",
    ),
}

# Common aliases mapping to canonical codes
ALIASES: dict[str, str] = {
    # Full names
    "assamese": "as",
    "bengali": "bn",
    "bangla": "bn",
    "bodo": "brx",
    "dogri": "doi",
    "gujarati": "gu",
    "gujrati": "gu",
    "hindi": "hi",
    "hinglish": "hi",
    "kannada": "kn",
    "kashmiri": "ks",
    "konkani": "kok",
    "gom": "kok",
    "maithili": "mai",
    "malayalam": "ml",
    "manipuri": "mni",
    "meitei": "mni",
    "marathi": "mr",
    "nepali": "ne",
    "odia": "or",
    "oriya": "or",
    "punjabi": "pa",
    "panjabi": "pa",
    "sanskrit": "sa",
    "santali": "sat",
    "sindhi": "sd",
    "tamil": "ta",
    "telugu": "te",
    "urdu": "ur",
}


def resolve_language(lang: str | None) -> LanguageInfo:
    """Resolve a language name, code, or alias to its canonical :class:`LanguageInfo`.

    Defaults to Hindi (``"hi"``) when ``lang`` is empty or None.
    Raises :class:`ValueError` with a list of available languages when unknown.
    """
    if not lang:
        return LANGUAGES["hi"]
    key = str(lang).strip().lower()
    if key in LANGUAGES:
        return LANGUAGES[key]
    if key in ALIASES:
        return LANGUAGES[ALIASES[key]]
    available = ", ".join(f"{v.name} ({k})" for k, v in sorted(LANGUAGES.items()))
    raise ValueError(f"Unknown language {lang!r}. Supported IndicVoices languages: {available}")


def all_languages() -> dict[str, LanguageInfo]:
    """Return a copy of the mapping from language code to :class:`LanguageInfo`."""
    return dict(LANGUAGES)


def supported_codes() -> list[str]:
    """List of all 22 supported language codes."""
    return sorted(LANGUAGES.keys())


def supported_names() -> list[str]:
    """List of English names of all 22 supported languages."""
    return sorted(info.name for info in LANGUAGES.values())


# Script detection regexes
_SCRIPT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("devanagari", re.compile(r"[\u0900-\u097F]")),
    ("bengali", re.compile(r"[\u0980-\u09FF]")),
    ("gurmukhi", re.compile(r"[\u0A00-\u0A7F]")),
    ("gujarati", re.compile(r"[\u0A80-\u0AFF]")),
    ("odia", re.compile(r"[\u0B00-\u0B7F]")),
    ("tamil", re.compile(r"[\u0B80-\u0BFF]")),
    ("telugu", re.compile(r"[\u0C00-\u0C7F]")),
    ("kannada", re.compile(r"[\u0C80-\u0CFF]")),
    ("malayalam", re.compile(r"[\u0D00-\u0D7F]")),
    ("arabic", re.compile(r"[\u0600-\u06FF\u0750-\u077F]")),
    ("ol_chiki", re.compile(r"[\u1C50-\u1C7F]")),
]


def detect_script(text: str) -> str | None:
    """Detect the predominant Indic script family in ``text``, or None if none found."""
    counts: dict[str, int] = {}
    for script_name, pattern in _SCRIPT_PATTERNS:
        matches = len(pattern.findall(text))
        if matches:
            counts[script_name] = matches
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]
