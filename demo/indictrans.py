"""English for each transcript line, with AI4Bharat's IndicTrans2 (Indic -> English, distilled, 200M parameters).

Not part of the four milestones: the live demo uses it so that an audience that reads neither Devanagari nor Kannada
can follow a take. Model: ai4bharat/indictrans2-indic-en-dist-200M (MIT), run in fp16 on the GPU next to Stages 1-3.

IndicTrans2 needs its own pre- and post-processing: punctuation and digit normalisation, placeholders for numbers,
URLs and e-mails, Indic normalisation and tokenisation, transliteration of every Indic script into Devanagari (the
model was trained on script-unified text), and "src_lang tgt_lang" tags. That logic is AI4Bharat's IndicTransToolkit
(MIT licence, Copyright (c) Varun Gumma). Its IndicProcessor is published only as a Cython extension, which needs a C
compiler to install, so the part used here (an Indic source, English target) is reproduced below in plain Python,
from IndicTransToolkit 1.1.1 (IndicTransToolkit/processor.pyx), with the placeholder queue replaced by explicit
maps, and the digit table also covering the few Indic digits upstream leaves out.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import regex as re

MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"     # gated on Hugging Face: accept its terms once
FALLBACK = "facebook/nllb-200-distilled-600M"          # not gated (CC-BY-NC): used until IndicTrans2 is available
FLORES = {"hi": "hin_Deva", "kn": "kan_Knda"}          # whospoke language -> FLORES code (both models use them)


def _local_weights(repo: str) -> str:
    """A folder with the model in safetensors format. transformers refuses ``pytorch_model.bin`` files with torch
    older than 2.6 (CVE-2025-32434), and these repositories publish only that. So the official file is read once with
    torch's safe loader (``weights_only=True``) and written next to the data as safetensors."""
    from huggingface_hub import snapshot_download
    from whospoke.paths import DATA

    src = Path(snapshot_download(repo))
    if any(src.glob("*.safetensors")):
        return str(src)
    dst = DATA / "models" / repo.replace("/", "--")
    if not (dst / "model.safetensors").exists():
        import torch
        from safetensors.torch import save_file

        state = torch.load(src / "pytorch_model.bin", map_location="cpu", weights_only=True)
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and f.name != "pytorch_model.bin":
                shutil.copy2(f, dst / f.name)
        # tied embeddings share memory, which safetensors cannot store; transformers ties them again on loading
        save_file({k: v.contiguous().clone() for k, v in state.items()}, str(dst / "model.safetensors"),
                  metadata={"format": "pt"})
    return str(dst)


def load_translator(device: str = "cpu"):
    """IndicTrans2 if it can be loaded, else NLLB-200. The CPU by default: Stages 1-3 fill the 6 GB GPU."""
    try:
        return IndicToEnglish(device)
    except Exception:
        return NLLBToEnglish(device)


class NLLBToEnglish:
    """Fallback: Meta's NLLB-200 (distilled, 600M), which reads Hindi and Kannada directly."""

    name = "NLLB-200"

    def __init__(self, device: str = "cpu", num_beams: int = 1):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch, self.device, self.num_beams = torch, device, num_beams
        path = _local_weights(FALLBACK)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device).eval()
        self.eng = self.tokenizer.convert_tokens_to_ids("eng_Latn")
        self.cache: dict[tuple[str, str], str] = {}

    def __call__(self, texts: list[str], language: str) -> list[str]:
        todo = [t for t in dict.fromkeys(texts) if t.strip() and (language, t) not in self.cache]
        if todo:
            self.tokenizer.src_lang = FLORES[language]
            enc = self.tokenizer(todo, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with self.torch.inference_mode():
                out = self.model.generate(**enc, forced_bos_token_id=self.eng, max_length=200, num_beams=self.num_beams)
            for text, eng in zip(todo, self.tokenizer.batch_decode(out, skip_special_tokens=True,
                                                                           clean_up_tokenization_spaces=True)):
                self.cache[(language, text)] = eng.strip()
        return [self.cache.get((language, t), "") for t in texts]


class IndicToEnglish:
    """Translate lines from an Indic language into English, caching every line already translated."""

    name = "IndicTrans2"

    def __init__(self, device: str | None = None, num_beams: int = 4):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = _local_weights(MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, trust_remote_code=True, torch_dtype=dtype)
        self.model = self.model.to(self.device).eval()
        self.num_beams = num_beams
        self.processor = _Processor()
        self.cache: dict[tuple[str, str], str] = {}

    def __call__(self, texts: list[str], language: str) -> list[str]:
        src = FLORES[language]
        todo = [t for t in dict.fromkeys(texts) if t.strip() and (language, t) not in self.cache]
        if todo:
            batch, maps = self.processor.preprocess(todo, src, "eng_Latn")
            enc = self.tokenizer(batch, truncation=True, padding="longest", return_tensors="pt",
                                 return_attention_mask=True).to(self.device)
            with self.torch.inference_mode():
                out = self.model.generate(**enc, use_cache=True, min_length=0, max_length=256,
                                          num_beams=self.num_beams, num_return_sequences=1)
            decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for text, eng, ph in zip(todo, decoded, maps):
                self.cache[(language, text)] = self.processor.postprocess_english(eng, ph)
        return [self.cache.get((language, t), "") for t in texts]


class _Processor:
    """IndicTransToolkit's IndicProcessor (inference mode), Indic source -> English target only."""

    _FLORES_ISO = {"hin_Deva": "hi", "kan_Knda": "kn", "eng_Latn": "en"}
    _INDIC_FAILURE_CASES = [
        "آی ڈی ", "ꯑꯥꯏꯗꯤ", "आईडी", "आई . डी . ", "आई . डी .", "आई. डी. ", "आई. डी.", "आय. डी. ", "आय. डी.",
        "आय . डी . ", "आय . डी .आइ . डी . ",   # (sic: a missing comma upstream joins these two entries)
        "आइ . डी .", "आइ. डी. ", "आइ. डी.", "ऐटि", "آئی ڈی ", "ᱟᱭᱰᱤ ᱾", "आयडी", "ऐडि", "आइडि", "ᱟᱭᱰᱤ",
    ]

    def __init__(self):
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        from indicnlp.tokenize import indic_tokenize
        from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
        from sacremoses import MosesDetokenizer

        self._normalizers = IndicNormalizerFactory()
        self._tokenize = indic_tokenize.trivial_tokenize
        self._xliterator = UnicodeIndicTransliterator()
        self._en_detok = MosesDetokenizer(lang="en")
        zeros = [0x09E6, 0x0AE6, 0x0CE6, 0x0966, 0x0660, 0xABF0, 0x0B66, 0x0A66, 0x1C50, 0x06F0, 0x0C66]
        self._digits = {z + d: str(d) for z in zeros for d in range(10)}
        self._multispace = re.compile(r"[ ]{2,}")
        self._digit_space_percent = re.compile(r"(\d) %")
        self._double_quot_punc = re.compile(r"\"([,\.]+)")
        self._digit_nbsp_digit = re.compile(r"(\d) (\d)")
        self._end_bracket_space_punc = re.compile(r"\) ([\.!:?;,])")
        self._url = re.compile(
            r"\b(?<![\w/.])(?:(?:https?|ftp)://)?(?:(?:[\w-]+\.)+(?!\.))(?:[\w/\-?#&=%.]+)+(?!\.\w+)\b")
        self._numeral = re.compile(
            r"(~?\d+\.?\d*\s?%?\s?-?\s?~?\d+\.?\d*\s?%|~?\d+%|\d+[-\/.,:']\d+[-\/.,:'+]\d+(?:\.\d+)?|\d+[-\/.:'+]\d+(?:\.\d+)?)")
        self._email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}")
        self._other = re.compile(r"[A-Za-z0-9]*[#|@]\w+")
        self._punc = [
            (re.compile(r"\r"), ""), (re.compile(r"\(\s*"), "("), (re.compile(r"\s*\)"), ")"),
            (re.compile(r"\s:\s?"), ":"), (re.compile(r"\s;\s?"), ";"), (re.compile(r"[`´‘‚’]"), "'"),
            (re.compile(r"[„“”«»]"), '"'), (re.compile(r"[–—]"), "-"), (re.compile(r"\.\.\."), "..."),
            (re.compile(r" %"), "%"), (re.compile(r"nº "), "nº "), (re.compile(r" ºC"), " ºC"),
            (re.compile(r" [?!;]"), lambda m: m.group(0).strip()), (re.compile(r", "), ", "),
        ]

    def _punc_norm(self, text: str) -> str:
        for pattern, repl in self._punc:
            text = pattern.sub(repl, text)
        text = self._multispace.sub(" ", text)
        text = self._end_bracket_space_punc.sub(r")\1", text)
        text = self._digit_space_percent.sub(r"\1%", text)
        text = self._double_quot_punc.sub(r'\1"', text)
        text = self._digit_nbsp_digit.sub(r"\1.\2", text)
        return text.strip()

    def _wrap_with_placeholders(self, text: str) -> tuple[str, dict[str, str]]:
        serial, ph = 1, {}
        for pattern in (self._email, self._url, self._numeral, self._other):
            for match in set(pattern.findall(text)):
                if pattern is self._url and len(match.replace(".", "")) < 4:
                    continue
                if pattern is self._numeral and len(match.replace(" ", "").replace(".", "").replace(":", "")) < 4:
                    continue
                for tag in ("ID", "id"):
                    for form in ("<{t}{n}>", "< {t}{n} >", "[{t}{n}]", "[ {t}{n} ]", "[{t} {n}]", "<{t}{n}]",
                                 "< {t}{n}]", "<{t}{n} ]"):
                        ph[form.format(t=tag, n=serial)] = match
                for case in self._INDIC_FAILURE_CASES:
                    for form in ("<{c}{n}>", "< {c}{n} >", "< {c} {n} >", "<{c} {n}]", "< {c} {n} ]", "[{c}{n}]",
                                 "[{c} {n}]", "[ {c}{n} ]", "[ {c} {n} ]", "{c} {n}", "{c}{n}"):
                        ph[form.format(c=case, n=serial)] = match
                text = text.replace(match, f"<ID{serial}>")
                serial += 1
        return re.sub(r"\s+", " ", text).replace(">/", ">").replace("]/", "]"), ph

    def preprocess(self, batch: list[str], src_lang: str, tgt_lang: str) -> tuple[list[str], list[dict[str, str]]]:
        iso = self._FLORES_ISO[src_lang]
        normalizer = self._normalizers.get_normalizer(iso)
        out, maps = [], []
        for sent in batch:
            sent = self._punc_norm(sent).translate(self._digits)
            sent, ph = self._wrap_with_placeholders(sent)
            tokens = " ".join(self._tokenize(normalizer.normalize(sent.strip()), iso))
            xlated = self._xliterator.transliterate(tokens, iso, "hi").replace(" ् ", "्")
            out.append(f"{src_lang} {tgt_lang} {xlated.strip()}")
            maps.append(ph)
        return out, maps

    def postprocess_english(self, sent: str, ph: dict[str, str]) -> str:
        for k, v in ph.items():
            sent = sent.replace(k, v)
        return self._en_detok.detokenize(sent.split(" "))
