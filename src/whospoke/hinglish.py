"""Devanagari → Hinglish (romanised Hindi, as people type it in chats).

The proposal asks for "standardized Latin representations for code-switched text (e.g. Hinglish)".
The ASR models write everything — including English words — in Devanagari (``होमवर्क``). We convert:

1. **English loanwords** are looked up in a lexicon (``resources/loanwords.tsv``, ~430 entries built
   from the most frequent words in IndicVoices) and written in English: ``होमवर्क`` → ``homework``.
2. **Hindi words** are romanised by rule, with Hindi *schwa deletion*: the inherent "a" of a
   consonant is not pronounced at the end of a word or in the pattern V C _ C V (``कमरा`` is
   "kamra", not "kamara"; ``समझना`` is "samajhna").
3. Chat-style vowel spelling: long ``ā/ī/ū`` are doubled only in closed one-syllable words
   (``baat``, ``teen``, ``phool``) and in a word-initial ``आ`` (``aap``); elsewhere a single letter
   (``tha``, ``hamara``, ``nahi``). A few very common words use their conventional spelling (``mein``, ``hai``).
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


def romanise_word(word: str) -> str:
    word = unicodedata.normalize("NFC", word)
    if not word:
        return word
    if word in COMMON:
        return COMMON[word]
    lex = lexicon()
    if word in lex:
        return lex[word]
    if not re.search(r"[ऀ-ॿ]", word):
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


def romanise(text: str) -> str:
    """Romanise a Devanagari sentence word by word (punctuation and Latin text are kept)."""
    text = unicodedata.normalize("NFC", text).replace("।", ".").replace("॥", ".")
    return re.sub(r"[ऀ-ॿ]+", lambda m: romanise_word(m.group(0)), text)
