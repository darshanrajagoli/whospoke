import pytest

from whospoke.hinglish import lexicon, romanise, romanise_word


@pytest.mark.parametrize("dev,rom", [
    # schwa deletion
    ("कमरा", "kamra"), ("समझना", "samajhna"), ("मतलब", "matlab"), ("बचपन", "bachpan"),
    ("पढ़ते", "padhte"), ("आपके", "aapke"), ("करना", "karna"),
    # chat-style vowels
    ("बात", "baat"), ("तीन", "teen"), ("दूध", "doodh"), ("था", "tha"), ("क्या", "kya"), ("हमारा", "hamara"),
    # nukta consonants
    ("ज़मीन", "zamin"), ("फ़ायदा", "fayda"),
    # anusvara before a labial becomes m
    ("कंबल", "kambal"),
    # English loanwords
    ("होमवर्क", "homework"), ("होमबर्क", "homework"), ("एक्चुअली", "actually"), ("मोबाइल", "mobile"),
    # conventional spellings
    ("में", "mein"), ("नहीं", "nahi"), ("है", "hai"),
])
def test_words(dev, rom):
    assert romanise_word(dev) == rom


def test_sentence_keeps_punctuation_and_latin():
    assert romanise("आज रेट डाउन है। OK?") == "aaj rate down hai. OK?"


def test_lexicon_is_clean():
    lex = lexicon()
    assert len(lex) > 400
    assert all(k and v and "\t" not in v for k, v in lex.items())
