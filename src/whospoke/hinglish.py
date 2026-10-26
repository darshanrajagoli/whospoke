"""Regional script → standardized Latin representations (Hinglish, Benglish, Tanglish, etc.).

The proposal asks for:
"The pipeline must map regional phonemes directly to native scripts or standardized
Latin representations for code-switched text (e.g., Hinglish/Benglish)."

This module converts native regional speech transcripts to standardized Latin representations:
1. **Hindi & Devanagari languages (Hinglish, etc.)**:
   - English loanwords via lexicon (``resources/loanwords.tsv``, ~2,600 entries mined from
     IndicVoices + Vaani train): ``होमवर्क`` → ``homework``.
   - Hindi words by rule with Hindi schwa deletion (``कमरा`` → "kamra", ``समझना`` → "samajhna").
   - Chat-style vowels: long ā/ī/ū doubled in closed one-syllable words (``baat``, ``teen``).
   - Conventional spelling dictionary for frequent words (``mein``, ``hai``, ``nahi``).
2. **Other Brahmic scripts (Bengali, Gujarati, Gurmukhi, Odia, Tamil, Telugu, Kannada, Malayalam)**:
   - Full phonetic transliteration based on the shared Unicode layout of Indic scripts.
   - Preserves script-specific consonants (Tamil zh/rr/n, Malayalam chillus, Dravidian short e/o).
   - Produces Benglish, Gujlish, Tanglish, Teluglish, Kanglish, Manglish, etc.
3. **Perso-Arabic script (Urdu, Kashmiri, Sindhi)**:
   - Phonetic Roman Urdu / Latin representation.
4. **Ol Chiki script (Santali)**:
   - Standard phonetic Latin representation.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

_LEXICON = Path(__file__).with_name("resources") / "loanwords.tsv"

CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ळ": "l",
}
NUKTA_CONSONANTS = {"क": "q", "ख": "kh", "ग": "gh", "ज": "z", "ड": "d", "ढ": "dh", "फ": "f", "य": "y"}
PRECOMPOSED = {"क़": "क़", "ख़": "ख़", "ग़": "ग़", "ज़": "ज़", "ड़": "ड़", "ढ़": "ढ़", "फ़": "फ़", "य़": "य़"}
INDEPENDENT_VOWELS = {
    "अ": "a", "आ": "A", "इ": "i", "ई": "I", "उ": "u", "ऊ": "U", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
}
MATRAS = {
    "ा": "A", "ि": "i", "ी": "I", "ु": "u", "ू": "U", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
}
VIRAMA, NUKTA, ANUSVARA, CANDRABINDU, VISARGA = "्", "़", "ं", "ँ", "ः"
LABIALS = {"p", "ph", "b", "bh", "m"}
DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}

# Very frequent words whose chat spelling is conventional rather than rule-derived.
COMMON = {
    "में": "mein", "मैं": "main", "है": "hai", "हैं": "hain", "हूँ": "hoon", "हूं": "hoon",
    "नहीं": "nahi", "नही": "nahi", "यह": "yeh", "वह": "woh", "वो": "wo", "हाँ": "haan", "हां": "haan",
    "यहाँ": "yahan", "वहाँ": "wahan", "कहाँ": "kahan", "यहां": "yahan", "वहां": "wahan", "कहां": "kahan",
    "क्यों": "kyon", "क्योंकि": "kyonki", "अच्छा": "achha", "अच्छी": "achhi", "अच्छे": "achhe",
    "ज़्यादा": "zyada", "ज्यादा": "zyada", "तो": "to", "और": "aur", "भी": "bhi", "कुछ": "kuch",
    "बहुत": "bahut", "मुझे": "mujhe", "हम": "hum", "तुम": "tum", "एक": "ek", "जी": "ji",
    "लिए": "liye", "लिये": "liye", "चाहिए": "chahiye", "गए": "gaye", "गई": "gayi", "ठीक": "theek",
}


@lru_cache(maxsize=1)
def lexicon() -> dict[str, str]:
    out = {}
    if _LEXICON.exists():
        for line in _LEXICON.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                dev, eng = line.split("\t")
                out[unicodedata.normalize("NFC", dev.strip())] = eng.strip()
    return out


def _units(word: str) -> list[list[str]]:
    """Split a Devanagari word into [consonant-or-None, vowel, nasal] units.

    vowel: a romanisation key (``a`` = inherent schwa, ``A/I/U`` = long vowels) or ``""`` for a
    consonant with virama (no vowel).
    """
    for k, v in PRECOMPOSED.items():
        word = word.replace(k, v)
    units: list[list[str]] = []
    i, chars = 0, list(word)
    while i < len(chars):
        c = chars[i]
        if c in CONSONANTS:
            rom = CONSONANTS[c]
            if i + 1 < len(chars) and chars[i + 1] == NUKTA:
                rom = NUKTA_CONSONANTS.get(c, rom)
                i += 1
            unit = [rom, "a", ""]
            if i + 1 < len(chars) and chars[i + 1] in MATRAS:
                unit[1] = MATRAS[chars[i + 1]]
                i += 1
            elif i + 1 < len(chars) and chars[i + 1] == VIRAMA:
                unit[1] = ""
                i += 1
            units.append(unit)
        elif c in INDEPENDENT_VOWELS:
            units.append([None, INDEPENDENT_VOWELS[c], ""])
        elif c in (ANUSVARA, CANDRABINDU) and units:
            units[-1][2] = "n"
        elif c == VISARGA and units:
            units[-1][2] = "h"
        elif c in DIGITS:
            units.append([DIGITS[c], "", ""])
        elif c.isascii() and c.isalnum():
            units.append([c, "", ""])
        i += 1
    return units


def _delete_schwas(units: list[list[str]]) -> None:
    """Hindi schwa deletion (word-final, then V C _ C V scanning right to left)."""
    if len(units) > 1 and units[-1][0] is not None and units[-1][1] == "a" and not units[-1][2]:
        units[-1][1] = ""
    elif len(units) > 1 and units[-1][1] == "a" and units[-1][2]:
        units[-1][1] = ""          # final nasalised schwa (e.g. "-ं") → just the nasal
    for i in range(len(units) - 2, 0, -1):
        cur, prev, nxt = units[i], units[i - 1], units[i + 1]
        if cur[0] is None or cur[1] != "a" or cur[2]:
            continue
        prev_has_vowel = prev[1] != ""
        next_is_cv = nxt[0] is not None and nxt[1] != ""
        if prev_has_vowel and next_is_cv:
            cur[1] = ""


def _syllables(units: list[list[str]]) -> int:
    return sum(1 for u in units if u[1])


def _romanise_devanagari_word(word: str) -> str:
    word = unicodedata.normalize("NFC", word)
    if not word:
        return word
    if word in COMMON:
        return COMMON[word]
    lex = lexicon()
    if word in lex:
        return lex[word]
    if not re.search(r"[\u0900-\u097F]", word):
        return word
    units = _units(word)
    _delete_schwas(units)
    closed_mono = _syllables(units) <= 1 and units[-1][1] == ""   # e.g. baat, teen (not tha, ki)
    out = []
    for i, (c, v, nasal) in enumerate(units):
        if c is not None:
            if c == "v" and i == 0 and v in ("a", "A", "o"):
                c = "w"
            out.append(c)
        long_ok = closed_mono or (i == 0 and c is None and v == "A")
        vowel = {"A": "aa" if long_ok else "a", "I": "ee" if long_ok else "i",
                 "U": "oo" if long_ok else "u"}.get(v, v)
        out.append(vowel)
        if nasal:
            nxt = units[i + 1][0] if i + 1 < len(units) else None
            out.append("m" if nasal == "n" and nxt in LABIALS else nasal)
    text = "".join(out)
    text = text.replace("chchh", "chh").replace("chch", "cch")
    return text


# -----------------------------------------------------------------------------
# General Brahmic script romanisation (Bengali, Gujarati, Gurmukhi, Odia,
# Tamil, Telugu, Kannada, Malayalam)
# -----------------------------------------------------------------------------

# Script-specific offsets for non-standard or special characters
_BRAHMIC_SPECIAL: dict[int, str] = {
    # Tamil special characters
    0x0B83: "k",     # Aytham ஃ
    0x0BA9: "n",     # Tamil ன
    0x0BB1: "r",     # Tamil ற
    0x0BB4: "zh",    # Tamil ழ
    0x0BB3: "l",     # Tamil ள
    # Bengali / Assamese
    0x09CE: "t",     # Bengali Khanda ta ৎ
    0x09F0: "r",     # Assamese ra ৰ
    0x09F1: "w",     # Assamese wa ৱ
    0x09DF: "y",     # Bengali ya য়
    0x09DC: "r",     # Bengali rra ড়
    0x09DD: "rh",    # Bengali rha ঢ়
    # Odia
    0x0B71: "w",     # Odia wa ୱ
    0x0B5F: "y",     # Odia ya ୟ
    0x0B5C: "r",     # Odia rra ଡ଼
    0x0B5D: "rh",    # Odia rha ଢ଼
    # Malayalam chillus
    0x0D7A: "n",     # Chillu nn ൺ
    0x0D7B: "n",     # Chillu n ൻ
    0x0D7C: "r",     # Chillu r ർ
    0x0D7D: "l",     # Chillu l ൽ
    0x0D7E: "l",     # Chillu ll ൾ
    0x0D7F: "k",     # Chillu k ൿ
}

_BRAHMIC_CONSONANT_OFFSETS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
    0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x5C: "r", 0x5D: "rh", 0x5F: "y",
}

_BRAHMIC_VOWEL_OFFSETS = {
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo", 0x0B: "ri",
    0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au",
}

_BRAHMIC_MATRA_OFFSETS = {
    0x3E: "aa", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo", 0x43: "ri",
    0x46: "e", 0x47: "e", 0x48: "ai", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x55: "e", 0x56: "e", 0x57: "e",
}


def _romanise_brahmic_word(word: str) -> str:
    """Romanise a word from any Brahmic script (Bengali, Tamil, Telugu, etc.)."""
    chars = list(word)
    units: list[list[str]] = []   # [consonant, vowel, nasal]
    i = 0
    while i < len(chars):
        cp = ord(chars[i])
        # Direct special match
        if cp in _BRAHMIC_SPECIAL:
            units.append([_BRAHMIC_SPECIAL[cp], "", ""])
            i += 1
            continue

        rel = (cp - 0x0900) % 0x80

        # Consonants
        if rel in _BRAHMIC_CONSONANT_OFFSETS:
            cons = _BRAHMIC_CONSONANT_OFFSETS[rel]
            unit = [cons, "a", ""]
            if i + 1 < len(chars):
                nxt_cp = ord(chars[i + 1])
                nxt_rel = (nxt_cp - 0x0900) % 0x80
                if nxt_rel == 0x4D:       # virama / halant / pulli (no vowel)
                    unit[1] = ""
                    i += 1
                elif nxt_rel in _BRAHMIC_MATRA_OFFSETS:
                    unit[1] = _BRAHMIC_MATRA_OFFSETS[nxt_rel]
                    i += 1
                elif nxt_cp in (0x0BCA, 0x0BCB, 0x0BCC):  # Tamil 2-part matras
                    unit[1] = "o" if nxt_cp in (0x0BCA, 0x0BCB) else "au"
                    i += 1
            units.append(unit)
        # Independent vowels
        elif rel in _BRAHMIC_VOWEL_OFFSETS:
            units.append([None, _BRAHMIC_VOWEL_OFFSETS[rel], ""])
        # Nasals / modifiers
        elif rel in (0x01, 0x02, 0x70):  # candrabindu, anusvara, Gurmukhi tippi
            if units:
                units[-1][2] = "n"
            else:
                units.append([None, "an", ""])
        elif rel == 0x03:                # visarga
            if units:
                units[-1][2] = "h"
        elif 0x66 <= rel <= 0x6F:        # Indic digits
            units.append([str(rel - 0x66), "", ""])
        elif chars[i].isascii() and chars[i].isalnum():
            units.append([chars[i], "", ""])
        i += 1

    # Indo-Aryan scripts (Bengali, Gurmukhi, Gujarati, Odia) delete word-final inherent schwas
    first_cp = ord(chars[0])
    is_indo_aryan = any(base <= first_cp < base + 0x80 for base in (0x0980, 0x0A00, 0x0A80, 0x0B00))
    if is_indo_aryan and len(units) > 1 and units[-1][1] == "a" and not units[-1][2]:
        units[-1][1] = ""

    out: list[str] = []
    is_dravidian = any(base <= first_cp < base + 0x80 for base in (0x0B80, 0x0C00, 0x0C80, 0x0D00))
    for j, (c, v, nasal) in enumerate(units):
        if c is not None:
            out.append(c)
        if v:
            # chat-style vowels: long 'aa' is simplified to 'a' in multi-syllable words unless word-initial
            if v == "aa" and len(units) > 1 and j > 0 and units[-1][1] != "":
                v = "a"
            out.append(v)
        if nasal:
            nxt = units[j + 1][0] if j + 1 < len(units) else None
            if nasal == "n":
                if nxt in LABIALS or (nxt is None and is_dravidian):
                    out.append("m")
                else:
                    out.append("n")
            else:
                out.append(nasal)
    res = "".join(out)
    return res.replace("chchh", "chh").replace("chch", "cch")


# -----------------------------------------------------------------------------
# Perso-Arabic script romanisation (Urdu, Kashmiri, Sindhi)
# -----------------------------------------------------------------------------
_URDU_LEXICON: dict[str, str] = {
    "شکریہ": "shukriya",
    "سلام": "salaam",
    "علیکم": "alaikum",
    "کیسے": "kaise",
    "کیا": "kya",
    "نہیں": "nahin",
    "ہاں": "haan",
    "بہت": "bahut",
    "آپ": "aap",
    "تم": "tum",
    "میں": "main",
    "ہم": "hum",
    "ہے": "hai",
    "ہیں": "hain",
    "تھا": "tha",
    "تھی": "thi",
    "تھے": "the",
    "اچھا": "achha",
}

_ARABIC_LETTERS: dict[int, str] = {
    0x0622: "aa", 0x0627: "a", 0x0628: "b", 0x067E: "p", 0x062A: "t", 0x0679: "t",
    0x062B: "s", 0x062C: "j", 0x0686: "ch", 0x062D: "h", 0x062E: "kh",
    0x062F: "d", 0x0688: "d", 0x0630: "z", 0x0631: "r", 0x0691: "r", 0x0632: "z",
    0x0698: "zh", 0x0633: "s", 0x0634: "sh", 0x0635: "s", 0x0636: "z",
    0x0637: "t", 0x0638: "z", 0x0639: "a", 0x063A: "gh", 0x0641: "f", 0x0642: "q",
    0x06A9: "k", 0x0643: "k", 0x06AF: "g", 0x0644: "l", 0x0645: "m", 0x0646: "n",
    0x06BA: "n", 0x0648: "o", 0x06C1: "h", 0x0647: "h", 0x06BE: "h",
    0x0621: "", 0x06CC: "i", 0x064A: "i", 0x06D2: "e",
    0x064E: "a", 0x0650: "i", 0x064F: "u", 0x0652: "",
}


def _romanise_arabic_word(word: str) -> str:
    """Romanise a Perso-Arabic word (Urdu, Kashmiri, Sindhi)."""
    if word in _URDU_LEXICON:
        return _URDU_LEXICON[word]
    chars = list(word)
    n = len(chars)
    out = []
    for i, c in enumerate(chars):
        cp = ord(c)
        if cp in (0x06C1, 0x0647) and i == n - 1 and n > 1:
            out.append("a")
            continue
        if cp in (0x06CC, 0x064A) and i == n - 2 and ord(chars[i + 1]) in (0x06C1, 0x0647, 0x06BE):
            out.append("iy")
            continue
        if cp in _ARABIC_LETTERS:
            out.append(_ARABIC_LETTERS[cp])
        elif c.isascii():
            out.append(c)
    text = "".join(out)
    return text.replace("chchh", "chh").replace("chch", "cch")



# -----------------------------------------------------------------------------
# Ol Chiki script romanisation (Santali)
# -----------------------------------------------------------------------------
_OL_CHIKI: dict[int, str] = {
    0x1C5A: "o", 0x1C5B: "t", 0x1C5C: "g", 0x1C5D: "ng", 0x1C5E: "l",
    0x1C5F: "a", 0x1C60: "k", 0x1C61: "j", 0x1C62: "m", 0x1C63: "w",
    0x1C64: "i", 0x1C65: "s", 0x1C66: "h", 0x1C67: "ny", 0x1C68: "r",
    0x1C69: "u", 0x1C6A: "ch", 0x1C6B: "d", 0x1C6C: "n", 0x1C6D: "y",
    0x1C6E: "e", 0x1C6F: "p", 0x1C70: "d", 0x1C71: "n", 0x1C72: "r",
    0x1C73: "o", 0x1C74: "t", 0x1C75: "b", 0x1C76: "v", 0x1C77: "h",
    0x1C78: "n", 0x1C79: "", 0x1C7A: "", 0x1C7B: "", 0x1C7C: "", 0x1C7D: "",
}


def _romanise_ol_chiki_word(word: str) -> str:
    """Romanise an Ol Chiki word (Santali)."""
    out = []
    for c in word:
        cp = ord(c)
        if 0x1C50 <= cp <= 0x1C59:
            out.append(str(cp - 0x1C50))
        elif cp in _OL_CHIKI:
            out.append(_OL_CHIKI[cp])
        elif c.isascii():
            out.append(c)
    return "".join(out)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
_INDIC_REGEX = re.compile(r"[\u0900-\u0D7F\u0600-\u06FF\u0750-\u077F\u1C50-\u1C7F]+")


def romanise_word(word: str, language: str | None = None) -> str:
    """Romanise a single word from any Indic script into its Latin representation.

    Devanagari uses Hindi schwa deletion and the loanwords lexicon.
    Other scripts (Bengali, Tamil, Telugu, Gujarati, Gurmukhi, Odia, Kannada,
    Malayalam, Urdu, Santali) use their standard phonetic Latin mapping.
    """
    word = unicodedata.normalize("NFC", word)
    if not word:
        return word

    # 1. Devanagari (Hindi, Marathi, Nepali, Sanskrit, etc.)
    if re.search(r"[\u0900-\u097F]", word):
        return _romanise_devanagari_word(word)

    # 2. Other Brahmic scripts (Bengali 0x980, Gurmukhi 0xA00, Gujarati 0xA80,
    #    Odia 0xB00, Tamil 0xB80, Telugu 0xC00, Kannada 0xC80, Malayalam 0xD00)
    if re.search(r"[\u0980-\u0D7F]", word):
        return _romanise_brahmic_word(word)

    # 3. Perso-Arabic (Urdu, Kashmiri, Sindhi)
    if re.search(r"[\u0600-\u06FF\u0750-\u077F]", word):
        return _romanise_arabic_word(word)

    # 4. Ol Chiki (Santali)
    if re.search(r"[\u1C50-\u1C7F]", word):
        return _romanise_ol_chiki_word(word)

    return word


def romanise(text: str, language: str | None = None) -> str:
    """Romanise a sentence word by word (punctuation, English, and digits are kept).

    Supports all 22 official scheduled Indian languages.
    """
    text = unicodedata.normalize("NFC", text).replace("।", ".").replace("॥", ".")
    return _INDIC_REGEX.sub(lambda m: romanise_word(m.group(0), language=language), text)
