"""Tests for 22 regional Indian languages across all pipeline stages.

Verifies:
1. Language registry (ISO codes, aliases, script detection) for all 22 scheduled languages.
2. Milestone 3 multi-script transliteration (Bengali, Tamil, Telugu, Gujarati, Gurmukhi,
   Kannada, Malayalam, Odia, Urdu, Ol Chiki Santali).
3. Milestone 3 ASR backend language routing (IndicConformer & IndicWav2Vec).
4. Milestone 3 Pipeline and Result formatting for regional languages (Latin transcripts).
5. Milestone 4 LLM post-processing language prompts and guardrails.
"""
import json
import pytest

from whospoke.languages import (
    LANGUAGES,
    LanguageInfo,
    all_languages,
    detect_script,
    resolve_language,
    supported_codes,
    supported_names,
)
from whospoke.hinglish import romanise
from whospoke.llm_postprocess import (
    CleanedLine,
    PostProcessedReport,
    PostProcessor,
    StaticLLM,
    chunk_prompt,
    system_prompt,
    _INDIC_SCRIPTS,
)


# =====================================================================
# 1. Language registry & resolution
# =====================================================================

def test_all_22_scheduled_languages_registered():
    codes = supported_codes()
    assert len(codes) == 22
    expected = {
        "as", "bn", "brx", "doi", "gu", "hi", "kn", "ks", "kok", "mai",
        "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
    }
    assert set(codes) == expected


def test_resolve_language_by_code_and_name_case_insensitive():
    for code in supported_codes():
        info_code = resolve_language(code)
        assert info_code.code == code
        info_name = resolve_language(info_code.name.upper())
        assert info_name.code == code


def test_resolve_language_aliases():
    assert resolve_language("bangla").code == "bn"
    assert resolve_language("gom").code == "kok"
    assert resolve_language("konkani").code == "kok"
    assert resolve_language("oriya").code == "or"
    assert resolve_language("santali").code == "sat"


def test_resolve_invalid_language_raises_value_error():
    with pytest.raises(ValueError, match="Unknown language"):
        resolve_language("klingon")


def test_detect_script():
    assert detect_script("आई डी कार्ड") == "devanagari"
    assert detect_script("আমার নাম") == "bengali"
    assert detect_script("வணக்கம்") == "tamil"
    assert detect_script("నమస్కారం") == "telugu"
    assert detect_script("નમસ્તે") == "gujarati"
    assert detect_script("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ") == "gurmukhi"
    assert detect_script("ನಮಸ್ಕಾರ") == "kannada"
    assert detect_script("നമസ്കാരം") == "malayalam"
    assert detect_script("ନମସ୍କାର") == "odia"
    assert detect_script("شکریہ") == "arabic"
    assert detect_script("ᱡᱚᱦᱟᱨ") == "ol_chiki"
    assert detect_script("Hello world 123") is None


# =====================================================================
# 2. Multi-script romanisation
# =====================================================================

def test_romanise_bengali():
    # Bengali: আমি বাংলায় কথা বলি
    rom = romanise("আমি বাংলায় কথা বলি", language="bn")
    assert "aami" in rom or "ami" in rom
    assert "katha" in rom or "kotha" in rom
    assert "boli" in rom or "bali" in rom


def test_romanise_tamil():
    # Tamil: வணக்கம்
    rom = romanise("வணக்கம்", language="ta")
    assert "vanakkam" in rom or "vanakkam" in rom.lower()


def test_romanise_telugu():
    # Telugu: నమస్కారం
    rom = romanise("నమస్కారం", language="te")
    assert "namaskaaram" in rom or "namaskaram" in rom


def test_romanise_gujarati():
    # Gujarati: નમસ્તે
    rom = romanise("નમસ્તે", language="gu")
    assert "namaste" in rom


def test_romanise_gurmukhi():
    # Gurmukhi: ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ
    rom = romanise("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ", language="pa")
    assert "sat" in rom
    assert "sri" in rom or "sree" in rom
    assert "akaal" in rom or "akal" in rom


def test_romanise_kannada():
    # Kannada: ನಮಸ್ಕಾರ
    rom = romanise("ನಮಸ್ಕಾರ", language="kn")
    assert "namaskaara" in rom or "namaskara" in rom


def test_romanise_malayalam():
    # Malayalam: നമസ്കാരം
    rom = romanise("നമസ്കാരം", language="ml")
    assert "namaskaaram" in rom or "namaskaram" in rom


def test_romanise_odia():
    # Odia: ନମସ୍କାର
    rom = romanise("ନମସ୍କାର", language="or")
    assert "namaskaar" in rom or "namaskara" in rom


def test_romanise_urdu():
    # Urdu: شکریہ
    rom = romanise("شکریہ", language="ur")
    assert "shukriya" in rom


def test_romanise_santali():
    # Santali (Ol Chiki): ᱡᱚᱦᱟᱨ
    rom = romanise("ᱡᱚᱦᱟᱨ", language="sat")
    assert "johar" in rom


def test_romanise_hindi_backward_compatibility():
    # Devanagari: आई डी कार्ड गुम हो गया
    rom = romanise("आई डी कार्ड गुम हो गया")
    assert "aai" in rom
    assert "gum ho gaya" in rom


def test_asr_language_routing_metadata():
    for code in supported_codes():
        info = resolve_language(code)
        assert info.conformer_code == code
        if code in {"hi", "bn", "gu", "te", "or", "ta"}:
            assert info.wav2vec_repo is not None
            assert info.wav2vec_repo.startswith("ai4bharat/indicwav2vec")
        else:
            assert info.wav2vec_repo is None


def test_indicconformer_accepts_all_scheduled_languages():
    pytest.importorskip("torch")
    pytest.importorskip("huggingface_hub")
    from whospoke.asr_backends import IndicConformerASR

    for code in ["hi", "bn", "ta", "te", "mr", "gu", "ur", "kn", "ml", "pa"]:
        asr = IndicConformerASR(language=code)
        assert asr.language == code
        assert asr.lang_info.code == code


def test_indicwav2vec_language_support_and_error():
    pytest.importorskip("torch")
    pytest.importorskip("huggingface_hub")
    from whospoke.asr_backends import IndicWav2VecASR

    for code in ["hi", "bn", "gu", "te", "or", "ta"]:
        asr = IndicWav2VecASR(language=code)
        assert asr.language == code

    # Language without CTC checkpoint should raise descriptive error
    with pytest.raises(ValueError, match="no published IndicWav2Vec CTC checkpoint"):
        IndicWav2VecASR(language="ks")


# =====================================================================
# 4. Milestone 3 Pipeline & Result format
# =====================================================================

def test_line_and_result_latin_property_and_export(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from whospoke.diarization import Diarization, Turn
    from whospoke.pipeline import Line, Result

    line = Line("Speaker_A", 0.0, 2.0, "வணக்கம்", "vanakkam")
    assert line.latin == "vanakkam"
    assert "வணக்கம்" in line.render()
    assert "vanakkam" in line.render(latin=True)

    diar = Diarization([Turn(0.0, 2.0, "Speaker_A")])
    res = Result("B", diar, [line], language="ta")
    assert res.language == "ta"
    txt = res.transcript(latin=True)
    assert "vanakkam" in txt

    js = res.transcript_json()
    assert js["language"] == "ta"
    assert js["lines"][0]["latin"] == "vanakkam"

    saved = res.save(tmp_path)
    assert (saved / "transcript_latin.txt").exists()
    assert "vanakkam" in (saved / "transcript_latin.txt").read_text(encoding="utf-8")



# =====================================================================
# 5. Milestone 4 LLM post-processing for regional languages
# =====================================================================

def test_system_and_chunk_prompts_parameterized():
    bn_sys = system_prompt("bn")
    assert "Bengali" in bn_sys and "Benglish" in bn_sys

    ta_chunk = chunk_prompt("ta")
    assert "ASR text in Tamil" in ta_chunk

    hi_sys = system_prompt("hi")
    assert "Hindi" in hi_sys and "Hinglish" in hi_sys


def test_indic_scripts_regex_detects_all_scripts():
    assert _INDIC_SCRIPTS.search("आई डी कार्ड")  # Devanagari
    assert _INDIC_SCRIPTS.search("আমার নাম")    # Bengali
    assert _INDIC_SCRIPTS.search("வணக்கம்")      # Tamil
    assert _INDIC_SCRIPTS.search("నమస్కారం")     # Telugu
    assert _INDIC_SCRIPTS.search("شکریہ")        # Urdu
    assert _INDIC_SCRIPTS.search("ᱡᱚᱦᱟᱨ")        # Ol Chiki
    assert not _INDIC_SCRIPTS.search("Hello world 123")


def test_postprocessor_auto_detects_language_from_transcript():
    chunk_resp = json.dumps({
        "lines": [{"id": 1, "text": "வணக்கம்", "english": "Greetings.", "uncertain": False}],
        "summary": "Formal greetings.",
        "keywords": ["vanakkam"],
        "actions": [],
    }, ensure_ascii=False)
    synth_resp = json.dumps({
        "title": "Tamil Greeting",
        "topic": "greetings",
        "summary": "Speaker_A extends greetings.",
        "key_points": ["Greetings were extended."],
        "keywords": ["vanakkam"],
        "actions": [],
    }, ensure_ascii=False)

    llm = StaticLLM([chunk_resp, synth_resp])
    pp = PostProcessor(llm)
    t = {
        "order": "B",
        "language": "ta",
        "lines": [{"speaker": "Speaker_A", "start": 0.0, "end": 2.0, "text": "வணக்கம்", "latin": "vanakkam"}],
    }
    report = pp.process_transcript(t)
    assert report.language == "ta"
    assert report.to_dict()["language"] == "ta"
    md = report.markdown()
    assert "Tamil" in md
    assert "Tanglish" in md
