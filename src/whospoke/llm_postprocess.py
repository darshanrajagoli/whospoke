"""Stage 4 — LLM post-processing: clean, translate and summarise the Stage-3 transcript.

The Stage-3 transcript is treated as *evidence*, not as free text the LLM may rewrite:

1. **Speaker labels and timestamps never pass through the LLM.** It sees numbered lines and returns, for each
   line number, a repaired Devanagari line, an English translation and an "uncertain" flag. Speaker and time
   are copied from Stage 3.
2. **The reply is checked in Python.** When the server supports it, the reply is constrained to a JSON schema
   that lists every line number in order. Either way, line numbers that do not exist are dropped, missing lines
   are restored from Stage 3 and flagged, and a "repaired" line that differs from the ASR line in more than half
   of its characters is reverted (the model rewrote or translated it instead of repairing it).
3. **The summary must be grounded.** Keywords are kept only if they occur in the transcript or its line
   translations. Action items must name a real speaker (or "unspecified") and cite real line numbers.

The Hinglish column is made by the Stage-3 romaniser from the repaired Devanagari, so both scripts agree and
the LLM never has to transliterate.

The backend is any OpenAI-compatible ``/v1/chat/completions`` server; by default llama.cpp's ``llama-server``
running a 4-bit AI4Bharat Airavata (``scripts/serve_llm.py``). Long transcripts are cut into chunks that fit
the model's context window. A second call turns the checked chunks into the report (title, topic, summary,
key points, keywords, action items).
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .audio import fmt_ts
from .hinglish import romanise
from .metrics import error_counts, normalise

DEFAULT_MODEL = os.getenv("WHOSPOKE_LLM_MODEL", "ai4bharat/Airavata")
DEFAULT_BASE_URL = os.getenv("WHOSPOKE_LLM_URL", "http://127.0.0.1:8080/v1")
DEFAULT_CONTEXT = int(os.getenv("WHOSPOKE_LLM_CONTEXT", "4096"))      # Airavata is Llama-2 based: 4,096 tokens
DEFAULT_TEMPERATURE = float(os.getenv("WHOSPOKE_LLM_TEMPERATURE", "0"))  # greedy: the same input gives the same report
DEFAULT_TIMEOUT = int(os.getenv("WHOSPOKE_LLM_TIMEOUT", "900"))      # a 7B model on a laptop CPU is slow
MAX_LINES_PER_CHUNK = 12        # a 7B model loses track of longer numbered lists
MAX_EDIT_RATIO = 0.5            # a repair may change at most half of a line's characters (D29)
UNSPECIFIED = "unspecified"     # action-item owner when the transcript does not say who must act

SYSTEM_PROMPT = """You clean up automatic speech recognition (ASR) transcripts of Hindi and Hinglish \
conversations recorded in noisy places. The transcript is the only source of truth. Never add facts, names, \
numbers, dates, places, events or actions that are not in it. Reply with JSON only."""

CHUNK_PROMPT = """Below are numbered lines of a conversation: line number | speaker | time | ASR text in Devanagari.
The ASR text has mistakes caused by noise and people talking over each other.

For every line, in the same order, return an object with:
- "id": the line number.
- "text": the same line in Devanagari, with only obvious ASR mistakes repaired (a repeated word, a broken word, \
missing punctuation). Keep English words that were spoken in English. If you are not sure, copy the line unchanged.
- "en": a faithful English translation of the line. Do not add anything that is not said.
- "uncertain": true if the line is too garbled to understand, otherwise false.

Also return:
- "summary": one or two English sentences saying what these lines are about. Only what is said.
- "keywords": up to 6 words or short phrases that actually occur in these lines.
- "actions": explicit requests, instructions or promises only. "owner" is the speaker who has to act \
(or "unspecified"), "action" says what in English, "lines" lists the supporting line numbers. Empty list if none.

LINES:
"""

SYNTHESIS_PROMPT = """Below is the evidence from a transcribed conversation: summaries of its parts, its lines \
translated into English, and candidate keywords and actions. Write the final report.

Return:
- "title": a short factual title (at most 10 words).
- "topic": the main topic in a few words (for example "lost ID card", "crop prices", "weather update").
- "summary": an executive summary in 2 to 4 English sentences. Only what the evidence supports.
- "key_points": up to 5 short factual points.
- "keywords": up to 8 keywords that occur in the evidence.
- "actions": explicit requests, instructions or promises only, with "owner" (a speaker label or "unspecified"), \
"action" and supporting "lines". Empty list if none.

EVIDENCE:
"""

MERGE_PROMPT = """Below are summaries of consecutive parts of one conversation. Merge them into one factual summary of \
at most 4 English sentences. Only what the summaries say.
Return: "summary".

SUMMARIES:
"""


# ====================================================================== LLM backends
class LLMError(RuntimeError):
    """The LLM call failed or its reply could not be used."""


class LLMConnectionError(LLMError):
    """The server cannot be reached: nothing will work, so the run stops."""


class LLMContextError(LLMError):
    """The request did not fit the model's context window, or the server timed out on it."""


class LLMHTTPError(LLMError):
    """The server answered with an HTTP error status."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"LLM server returned HTTP {status}: {detail}")
        self.status, self.detail = status, detail


@dataclass
class LLMReply:
    text: str
    finish_reason: str | None = None     # "length" means the reply was cut off by max_tokens
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    seconds: float = 0.0


class LLMBackend(Protocol):
    model_name: str
    schema_enforced: bool

    def ping(self) -> None: ...

    def complete(self, system: str, user: str, *, temperature: float, max_tokens: int,
                 schema: dict | None = None) -> LLMReply: ...


_CONTEXT_ERROR = re.compile(r"context|n_ctx|too long|exceed|maximum.*tokens", re.I)
_SCHEMA_ERROR = re.compile(r"schema|grammar|response_format|json", re.I)
RETRY_WAIT_S = 5.0              # wait between retries while the server answers HTTP 503 (loading / busy)
_START_HINT = "Start it first: python scripts/serve_llm.py (see docs/MILESTONE4.md)."


class OpenAICompatibleLLM:
    """Client for an OpenAI-compatible ``/v1/chat/completions`` server (llama.cpp, llama-cpp-python, vLLM, Ollama).

    ``schema`` is sent as ``response_format={"type": "json_object", "schema": ...}``, which llama.cpp turns into
    a grammar so that only schema-valid JSON can be generated. A server that rejects the field is asked again
    without it, and the schema is not sent again; the Python checks still apply.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 timeout_s: int = DEFAULT_TIMEOUT, json_schema: bool = True, seed: int = 0):
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.timeout_s = timeout_s
        self.schema_enforced = json_schema
        self.seed = seed

    def complete(self, system: str, user: str, *, temperature: float, max_tokens: int,
                 schema: dict | None = None) -> LLMReply:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature, "max_tokens": max_tokens, "seed": self.seed, "stream": False,
        }
        if schema is not None and self.schema_enforced:
            payload["response_format"] = {"type": "json_object", "schema": schema}
        try:
            return self._post(payload)
        except LLMHTTPError as exc:
            # Only a server that rejects the schema itself loses it; a transient error must not switch it off.
            if "response_format" not in payload or exc.status not in (400, 422, 500) \
                    or not _SCHEMA_ERROR.search(exc.detail):
                raise
            payload.pop("response_format")
            reply = self._post(payload)
            self.schema_enforced = False      # the server does not support it; stop sending it
            # loud, because it can also mean the server failed to build the grammar (see the chat template)
            warnings.warn(f"The LLM server rejected the JSON schema, so replies are no longer constrained ({exc}).",
                          stacklevel=2)
            return reply

    def ping(self, wait_s: float = 120) -> None:
        """Raise LLMConnectionError unless the server is up. A server still loading its model (HTTP 503) is
        given up to ``wait_s`` seconds."""
        deadline = time.monotonic() + wait_s
        while True:
            try:
                with urllib.request.urlopen(self.base_url + "/models", timeout=10):
                    return
            except urllib.error.HTTPError as exc:
                if exc.code != 503:
                    return                        # it answered; some servers do not implement /models
                if time.monotonic() > deadline:
                    raise LLMConnectionError(f"The Stage-4 LLM server at {self.base_url} is still not ready "
                                             f"(HTTP 503, usually: loading the model) after {wait_s:.0f} s.") from exc
                time.sleep(RETRY_WAIT_S)
            except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
                raise LLMConnectionError(f"Cannot reach the Stage-4 LLM server at {self.base_url} "
                                         f"({getattr(exc, 'reason', exc)}). {_START_HINT}") from exc

    def _post(self, payload: dict, retries_503: int = 6) -> LLMReply:
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            if exc.code == 503:                   # busy or still loading the model: wait, then give up cleanly
                if retries_503 > 0:
                    time.sleep(RETRY_WAIT_S)
                    return self._post(payload, retries_503 - 1)
                raise LLMConnectionError(f"The Stage-4 LLM server keeps answering HTTP 503: {detail}") from exc
            if _CONTEXT_ERROR.search(detail):
                raise LLMContextError(f"LLM server returned HTTP {exc.code}: {detail}") from exc
            raise LLMHTTPError(exc.code, detail) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise LLMContextError(f"LLM server did not answer within {self.timeout_s} s") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise LLMContextError(f"LLM server did not answer within {self.timeout_s} s") from exc
            raise LLMConnectionError(
                f"Cannot reach the Stage-4 LLM server at {self.base_url} ({exc.reason}). {_START_HINT}") from exc
        except (http.client.HTTPException, OSError) as exc:   # connection dropped mid-reply (server crashed?)
            raise LLMConnectionError(f"Lost the connection to the Stage-4 LLM server at {self.base_url} "
                                     f"({type(exc).__name__}: {exc}). {_START_HINT}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LLMError("LLM server returned a body that is not JSON") from exc
        try:
            choice = body["choices"][0]
            content = (choice.get("message") or {}).get("content")
            if content is None:
                content = choice.get("text")
            if not isinstance(content, str):
                raise TypeError
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {str(body)[:500]}") from exc
        usage = body.get("usage") or {}
        return LLMReply(content.strip(), choice.get("finish_reason"), usage.get("prompt_tokens"),
                        usage.get("completion_tokens"), round(time.perf_counter() - t0, 3))


class StaticLLM:
    """Scripted backend for tests: returns the given replies in order (a string, an LLMReply, or an exception)."""

    def __init__(self, responses: list | str, model_name: str = "static-test", schema_enforced: bool = False):
        self.responses = [responses] if isinstance(responses, str) else list(responses)
        self.model_name = model_name
        self.schema_enforced = schema_enforced
        self.calls: list[dict] = []

    def ping(self) -> None:
        pass

    def complete(self, system: str, user: str, *, temperature: float, max_tokens: int,
                 schema: dict | None = None) -> LLMReply:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens, "schema": schema})
        if len(self.calls) > len(self.responses):
            raise LLMError("StaticLLM has no reply left")
        r = self.responses[len(self.calls) - 1]
        if isinstance(r, BaseException):
            raise r
        return r if isinstance(r, LLMReply) else LLMReply(r, "stop")


# ====================================================================== report data
@dataclass
class CleanedLine:
    line_id: int              # 1-based position in transcript.json "lines"
    speaker: str              # from Stage 2, never changed
    start: float              # from Stage 2, never changed
    end: float
    text: str                 # repaired Devanagari
    hinglish: str             # romanised from ``text`` by the Stage-3 romaniser
    translation: str          # English
    source_text: str          # the ASR line as Stage 3 wrote it
    changed: bool = False
    uncertain: bool = False
    note: str = ""


@dataclass
class ActionItem:
    owner: str
    action: str
    evidence_line_ids: list[int] = field(default_factory=list)


@dataclass
class Diagnostics:
    """What the guardrails did during one report; used by the report's last section and by the evaluation."""
    llm_calls: int = 0
    llm_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    chunks: int = 0
    chunk_splits: int = 0             # a chunk was too long or its reply unusable, so it was halved and retried
    invalid_replies: int = 0          # replies that were not valid JSON or were cut off
    failed_chunks: int = 0            # single lines the LLM could not process at all (ASR wording kept)
    synthesis_failed: bool = False
    lines_changed: int = 0
    lines_reverted: int = 0           # repair changed > MAX_EDIT_RATIO of the characters, or added words: reverted
    lines_restored: int = 0           # the reply skipped the line: restored from Stage 3
    translations_too_long: int = 0    # flagged: may add content (> 2 × the line's length + 30 characters)
    lines_flagged_by_llm: int = 0
    missing_translations: int = 0
    unknown_line_ids: int = 0         # line numbers in the reply that do not exist
    keywords_dropped: int = 0         # not found in the transcript or its translations
    actions_dropped: int = 0          # unknown owner or no valid evidence line
    empty_lines_skipped: int = 0      # Stage-3 lines with no recognised words
    schema_enforced: bool = False


@dataclass
class PostProcessedReport:
    title: str
    topic: str
    executive_summary: str
    key_points: list[str]
    keywords: list[str]
    action_items: list[ActionItem]
    cleaned_dialogue: list[CleanedLine]
    speakers: list[dict]
    source_sha256: str
    source_line_count: int
    llm_model: str
    generated_at_utc: str
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    pipeline_order: str | None = None
    source_audio_duration_s: float | None = None

    @property
    def uncertain_lines(self) -> list[int]:
        return [x.line_id for x in self.cleaned_dialogue if x.uncertain]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "topic": self.topic,
            "executive_summary": self.executive_summary,
            "key_points": self.key_points,
            "keywords": self.keywords,
            "action_items": [asdict(x) for x in self.action_items],
            "speakers": self.speakers,
            "cleaned_dialogue": [asdict(x) for x in self.cleaned_dialogue],
            "uncertain_lines": self.uncertain_lines,
            "diagnostics": asdict(self.diagnostics),
            "provenance": {
                "source_sha256": self.source_sha256,
                "source_line_count": self.source_line_count,
                "llm_model": self.llm_model,
                "generated_at_utc": self.generated_at_utc,
                "pipeline_order": self.pipeline_order,
                "source_audio_duration_s": self.source_audio_duration_s,
            },
        }

    def markdown(self) -> str:
        d = self.diagnostics
        dur = f"{fmt_ts(self.source_audio_duration_s)} of audio · " if self.source_audio_duration_s else ""
        out = [f"# {self.title}", "",
               f"*{dur}{len(self.speakers)} speakers · {len(self.cleaned_dialogue)} lines · "
               f"post-processed by `{self.llm_model}`*", "",
               "## Executive summary", "",
               self.executive_summary or "_No summary could be generated; see the dialogue below._", ""]
        if self.topic:
            out += [f"**Topic:** {self.topic}", ""]
        out += ["## Key points", ""]
        out += [f"- {p}" for p in self.key_points] or ["- None extracted."]
        out += ["", "## Keywords", "",
                " · ".join(f"`{k}`" for k in self.keywords) if self.keywords else "None extracted.", "",
                "## Action items", ""]
        if self.action_items:
            out += ["| who | what | said in line |", "|---|---|---|"]
            out += [f"| {_md(a.owner)} | {_md(a.action)} | {', '.join(map(str, a.evidence_line_ids))} |"
                    for a in self.action_items]
        else:
            out.append("None stated in the conversation.")
        out += ["", "## Speakers", "", "| speaker | talk time | turns |", "|---|---|---|"]
        out += [f"| {s['speaker']} | {s['talk_time_s']:.0f} s ({100 * s['share']:.0f} %) | {s['turns']} |"
                for s in self.speakers]
        out += ["", "## Dialogue", "",
                "Speaker labels and times come from Stage 2 unchanged. ⚠ marks a line the model could not "
                "repair or translate with confidence (reasons below).", "",
                "| # | time | speaker | transcript (repaired) | Hinglish | English |", "|---|---|---|---|---|---|"]
        for x in self.cleaned_dialogue:
            flag = " ⚠" if x.uncertain else ""
            out.append(f"| {x.line_id} | {fmt_ts(x.start)}–{fmt_ts(x.end)} | {x.speaker} | {_md(x.text)}{flag} | "
                       f"{_md(x.hinglish)} | {_md(x.translation) or '—'} |")
        changed = [x for x in self.cleaned_dialogue if x.changed]
        out += ["", "## What Stage 4 changed", ""]
        if changed:
            out += ["| # | ASR (Stage 3) | repaired |", "|---|---|---|"]
            out += [f"| {x.line_id} | {_md(x.source_text)} | {_md(x.text)} |" for x in changed]
        else:
            out.append("No line was changed; the transcript column is the ASR output.")
        flagged = [x for x in self.cleaned_dialogue if x.uncertain]
        out += ["", "## Uncertain lines", ""]
        out += [f"- Line {x.line_id} ({fmt_ts(x.start)}, {x.speaker}): {x.note}" for x in flagged] \
            or ["None."]
        out += ["", "## How this report was made", "",
                f"- Source: Stage-3 `transcript.json`, {self.source_line_count} lines "
                f"(SHA-256 `{self.source_sha256[:16]}…`), pipeline order {self.pipeline_order or 'unknown'}.",
                f"- Model: `{self.llm_model}`, {d.llm_calls} calls, {d.llm_seconds:.0f} s"
                f"{', JSON schema enforced by the server' if d.schema_enforced else ''}.",
                f"- Guardrails: {d.lines_changed} lines repaired, {d.lines_reverted} over-edited repairs reverted, "
                f"{d.lines_restored} skipped lines restored, {d.keywords_dropped} ungrounded keywords and "
                f"{d.actions_dropped} unsupported action items dropped.",
                f"- Generated {self.generated_at_utc}.", ""]
        return "\n".join(out)

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "report.md").write_text(self.markdown(), encoding="utf-8")
        return out


def _md(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


# ====================================================================== JSON parsing and schemas
def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating code fences or a few stray words around it."""
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I).strip()
    try:
        obj = json.loads(clean)
    except json.JSONDecodeError:
        start, end = clean.find("{"), clean.rfind("}")
        if start < 0 or end <= start:
            raise LLMError("LLM reply contains no JSON object") from None
        try:
            obj = json.loads(clean[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM reply is not valid JSON: {clean[:300]!r}") from exc
    if not isinstance(obj, dict):
        raise LLMError("LLM reply must be a JSON object")
    return obj


_ACTION_SCHEMA = {
    "type": "object",
    "properties": {"owner": {"type": "string"}, "action": {"type": "string", "maxLength": 200},
                   "lines": {"type": "array", "items": {"type": "integer"}, "maxItems": 6}},
    "required": ["owner", "action", "lines"], "additionalProperties": False,
}


def chunk_schema(chunk: list[dict]) -> dict:
    """JSON schema for one chunk reply: exactly these line numbers, in this order (llama.cpp ``prefixItems``)."""
    def line(src: dict) -> dict:
        n = len(src["text"])
        return {"type": "object",
                "properties": {"id": {"const": src["line_id"]},
                               "text": {"type": "string", "maxLength": 2 * n + 20},
                               "en": {"type": "string", "maxLength": _max_en(src["text"])},
                               "uncertain": {"type": "boolean"}},
                "required": ["id", "text", "en", "uncertain"], "additionalProperties": False}
    return {"type": "object",
            "properties": {
                # a tuple: exactly these lines, in order (llama.cpp lets "items" override "prefixItems", so no "items")
                "lines": {"type": "array", "prefixItems": [line(x) for x in chunk]},
                "summary": {"type": "string", "maxLength": 400},
                "keywords": {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 6},
                "actions": {"type": "array", "items": _ACTION_SCHEMA, "maxItems": 4}},
            "required": ["lines", "summary", "keywords", "actions"], "additionalProperties": False}


SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "maxLength": 90},
        "topic": {"type": "string", "maxLength": 80},
        "summary": {"type": "string", "maxLength": 800},
        "key_points": {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 5},
        "keywords": {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 8},
        "actions": {"type": "array", "items": _ACTION_SCHEMA, "maxItems": 6}},
    "required": ["title", "topic", "summary", "key_points", "keywords", "actions"], "additionalProperties": False,
}


MERGE_SCHEMA = {"type": "object", "properties": {"summary": {"type": "string", "maxLength": 600}},
                "required": ["summary"], "additionalProperties": False}


def _max_en(text: str) -> int:
    """Longest acceptable English translation of a line: English is rarely more than ~1.5× the Devanagari."""
    return 2 * len(text) + 30


# ====================================================================== token budget
def estimate_tokens(text: str) -> int:
    """Conservative token estimate without a tokenizer: Devanagari ≈ 2 characters per token, other text ≈ 3."""
    deva = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    return int((deva / 2 + (len(text) - deva) / 3) * 1.1) + 1


def _line_prompt(src: dict) -> str:
    return f"{src['line_id']} | {src['speaker']} | {fmt_ts(src['start'])}-{fmt_ts(src['end'])} | {src['text']}"


def _reply_tokens(chunk: list[dict]) -> int:
    """Expected reply length: each line comes back once in Devanagari and once in English, plus JSON and summary."""
    return sum(2 * estimate_tokens(x["text"]) + 24 for x in chunk) + 220


# ====================================================================== the post-processor
class PostProcessor:
    """Stage 4. ``process_transcript`` takes a Stage-3 ``transcript.json`` dict and returns a checked report."""

    def __init__(self, backend: LLMBackend | None = None, *, model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL, temperature: float = DEFAULT_TEMPERATURE,
                 context_tokens: int = DEFAULT_CONTEXT, timeout_s: int = DEFAULT_TIMEOUT,
                 max_lines_per_chunk: int = MAX_LINES_PER_CHUNK, max_edit_ratio: float = MAX_EDIT_RATIO,
                 json_schema: bool = True):
        self.backend = backend or OpenAICompatibleLLM(base_url, model, timeout_s, json_schema)
        self.temperature = temperature
        self.context_tokens = context_tokens
        self.max_lines_per_chunk = max_lines_per_chunk
        self.max_edit_ratio = max_edit_ratio
        self.diag = Diagnostics()

    # ------------------------------------------------------------------ public entry points
    def process_file(self, path: str | Path) -> PostProcessedReport:
        return self.process_transcript(json.loads(Path(path).read_text(encoding="utf-8")))

    def process_transcript(self, transcript: dict[str, Any]) -> PostProcessedReport:
        self.diag = Diagnostics()
        all_lines = _line_records(transcript)
        source = [x for x in all_lines if normalise(x["text"])]
        self.diag.empty_lines_skipped = len(all_lines) - len(source)
        speakers = _speaker_stats(source)
        valid_owners = {s["speaker"] for s in speakers} | {UNSPECIFIED}

        cleaned: list[CleanedLine] = []
        chunk_notes: list[dict] = []
        for chunk in self._plan_chunks(source):
            for lines, note in self._run_chunk(chunk):
                cleaned += lines
                chunk_notes.append(note)
        self.diag.chunks = len(chunk_notes)

        haystack = _token_set([x.source_text for x in cleaned] + [x.text for x in cleaned]
                              + [x.hinglish for x in cleaned] + [x.translation for x in cleaned]
                              + [romanise(x.source_text) for x in cleaned])
        by_id = {x.line_id: x for x in cleaned}
        synth = self._synthesise(cleaned, chunk_notes) if cleaned else {}

        keywords = self._grounded(synth.get("keywords"), haystack)
        if not keywords:
            keywords = self._grounded([k for c in chunk_notes for k in c["keywords"]], haystack)
        actions = self._actions(synth.get("actions"), by_id, valid_owners)
        if not actions:
            actions = self._actions([a for c in chunk_notes for a in c["actions"]], by_id, valid_owners)
        summary = _text(synth.get("summary"))
        if not summary:                       # synthesis failed: the part summaries, merged until they are short
            summary = " ".join(self._condense([c["summary"] for c in chunk_notes]))[:1500]

        d = self.diag
        d.lines_changed = sum(x.changed for x in cleaned)
        d.schema_enforced = bool(getattr(self.backend, "schema_enforced", False))
        if not source:
            summary = "No speech was recognised in this recording."
        return PostProcessedReport(
            title=_text(synth.get("title"))[:90] or "Conversation report",
            topic=_text(synth.get("topic"))[:80],
            executive_summary=summary,
            key_points=_dedupe(_text(p) for p in _as_list(synth.get("key_points")))[:5],
            keywords=keywords[:8],
            action_items=actions[:6],
            cleaned_dialogue=cleaned,
            speakers=speakers,
            source_sha256=_fingerprint(transcript),
            source_line_count=len(all_lines),
            llm_model=self.backend.model_name,
            generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            diagnostics=d,
            pipeline_order=transcript.get("order"),
            source_audio_duration_s=transcript.get("duration_s"),
        )

    # ------------------------------------------------------------------ LLM calls
    def _call(self, user: str, schema: dict, max_tokens: int) -> LLMReply:
        reply = self.backend.complete(SYSTEM_PROMPT, user, temperature=self.temperature,
                                      max_tokens=max_tokens, schema=schema)
        d = self.diag
        d.llm_calls += 1
        d.llm_seconds = round(d.llm_seconds + reply.seconds, 3)
        d.prompt_tokens += reply.prompt_tokens or 0
        d.completion_tokens += reply.completion_tokens or 0
        return reply

    def _budget(self) -> int:
        return self.context_tokens - estimate_tokens(SYSTEM_PROMPT) - 64

    def _plan_chunks(self, source: list[dict]) -> list[list[dict]]:
        """Consecutive lines, as many as fit the context window (prompt + expected reply), at most N lines."""
        budget = self._budget() - estimate_tokens(CHUNK_PROMPT)
        chunks: list[list[dict]] = []
        cur: list[dict] = []
        for line in source:
            trial = cur + [line]
            cost = sum(estimate_tokens(_line_prompt(x)) for x in trial) + _reply_tokens(trial)
            if cur and (cost > budget or len(trial) > self.max_lines_per_chunk):
                chunks.append(cur)
                trial = [line]
            cur = trial
        return chunks + ([cur] if cur else [])

    def _run_chunk(self, chunk: list[dict]) -> list[tuple[list[CleanedLine], dict]]:
        """Process one chunk. If the reply is unusable, halve the chunk and retry; one line that fails keeps its ASR text."""
        user = CHUNK_PROMPT + "\n".join(_line_prompt(x) for x in chunk)
        max_tokens = max(_reply_tokens(chunk), self._budget() - estimate_tokens(user))
        reason = "the model's reply could not be used"
        try:
            reply = self._call(user, chunk_schema(chunk), max_tokens)
            if reply.finish_reason == "length":
                raise LLMError("reply cut off at the token limit")
            return [self._check_chunk(parse_json_object(reply.text), chunk)]
        except LLMConnectionError:
            raise
        except LLMContextError:
            reason = "the line did not fit the model's context window"
        except LLMError:
            self.diag.invalid_replies += 1
        if len(chunk) > 1:
            self.diag.chunk_splits += 1
            mid = len(chunk) // 2
            return self._run_chunk(chunk[:mid]) + self._run_chunk(chunk[mid:])
        self.diag.failed_chunks += 1
        return [([_kept(chunk[0], f"{reason}; ASR wording kept.")], {"summary": "", "keywords": [], "actions": []})]

    def _check_chunk(self, obj: dict, chunk: list[dict]) -> tuple[list[CleanedLine], dict]:
        by_id = {x["line_id"]: x for x in chunk}
        got: dict[int, dict] = {}
        for item in _as_list(obj.get("lines", obj.get("cleaned_dialogue"))):
            if not isinstance(item, dict):
                continue
            try:
                i = int(item.get("id", item.get("line_id")))
            except (TypeError, ValueError, OverflowError):
                continue
            if i not in by_id:
                self.diag.unknown_line_ids += 1
            elif i not in got:
                got[i] = item
        lines = []
        for src in chunk:
            item = got.get(src["line_id"])
            if item is None:
                self.diag.lines_restored += 1
                lines.append(_kept(src, "the model skipped this line; ASR wording kept."))
                continue
            text = _text(item.get("text", item.get("cleaned"))) or src["text"]
            en = _text(item.get("en", item.get("translation")))
            notes = []
            if normalise(text) != normalise(src["text"]):      # punctuation-only repairs are always accepted
                edits, n = error_counts(src["text"], text, "char")
                n_src = len(_tokens(src["text"]))
                if edits / max(n, 1) > self.max_edit_ratio:
                    self.diag.lines_reverted += 1
                    notes.append("the model rewrote this line instead of repairing it; ASR wording kept")
                    text = src["text"]
                elif len(_tokens(text)) - n_src > max(1, round(0.2 * n_src)):
                    self.diag.lines_reverted += 1      # a repair removes or fixes words; it does not add sentences
                    notes.append("the model added words that the ASR line does not contain; ASR wording kept")
                    text = src["text"]
            if len(en) > _max_en(src["text"]):
                self.diag.translations_too_long += 1
                notes.append("the English is much longer than the line, so it may add things that were not said")
            uncertain = bool(notes) or item.get("uncertain") is True
            if item.get("uncertain") is True:
                self.diag.lines_flagged_by_llm += 1
                notes.append("the model found this line hard to understand")
            if not en:
                self.diag.missing_translations += 1
                uncertain = True
                notes.append("no English translation was returned")
            changed = text != src["text"]
            lines.append(CleanedLine(
                src["line_id"], src["speaker"], src["start"], src["end"], text,
                src["hinglish"] if not changed and src["hinglish"] else romanise(text), en, src["text"],
                changed, uncertain, _sentence("; ".join(notes))))
        note = {"summary": _text(obj.get("summary", obj.get("chunk_summary"))),
                "keywords": [_text(k) for k in _as_list(obj.get("keywords")) if _text(k)],
                "actions": _as_list(obj.get("actions", obj.get("action_items")))}
        return lines, note

    def _synthesise(self, cleaned: list[CleanedLine], notes: list[dict]) -> dict:
        """Second call: title, topic, summary, key points, keywords and actions from the checked chunks."""
        parts = [f"Part {i}: {s}" for i, s in enumerate(self._condense([n["summary"] for n in notes]), 1)]
        lines = [f"{x.line_id} {x.speaker}: {x.translation or x.hinglish}" for x in cleaned]
        kw = _dedupe(k for n in notes for k in n["keywords"])[:30]
        acts = [a for n in notes for a in n["actions"] if isinstance(a, dict)][:12]
        head = SYNTHESIS_PROMPT + "SUMMARIES OF THE PARTS:\n" + ("\n".join(parts) or "(none)") + "\n\n"
        tail = (f"\nCANDIDATE KEYWORDS: {', '.join(kw) or '(none)'}\n"
                f"CANDIDATE ACTIONS: {json.dumps(acts, ensure_ascii=False) if acts else '(none)'}\n")
        reply_budget = 450
        room = self._budget() - reply_budget - estimate_tokens(head + tail)
        body, used = [], 0
        for ln in lines:                      # as many translated lines as fit; the part summaries cover the rest
            t = estimate_tokens(ln) + 1
            if used + t > room:
                body.append("(later lines omitted for length; see the part summaries)")
                break
            body.append(ln)
            used += t
        user = head + "LINES (English):\n" + "\n".join(body) + "\n" + tail
        try:
            reply = self._call(user, SYNTHESIS_SCHEMA, max(reply_budget, self._budget() - estimate_tokens(user)))
            if reply.finish_reason == "length":
                raise LLMError("reply cut off at the token limit")
            return parse_json_object(reply.text)
        except LLMConnectionError:
            raise
        except LLMError:
            self.diag.synthesis_failed = True
            return {}

    def _condense(self, summaries: list[str]) -> list[str]:
        """Part summaries short enough to leave half the context window for the rest of the synthesis prompt.

        A long recording has one summary per chunk (a 30-minute one about 30), more than fits. Consecutive summaries
        are merged by the LLM in groups, repeatedly, until they fit. A group the LLM cannot merge keeps its first
        summary only.
        """
        parts = [s for s in summaries if s]
        room = self._budget() // 2
        while len(parts) > 1 and estimate_tokens("\n".join(parts)) > room:
            groups, cur = [], []
            for s in parts:
                if cur and estimate_tokens("\n".join(cur + [s])) > room:
                    groups.append(cur)
                    cur = []
                cur.append(s)
            groups.append(cur)
            if len(groups) == len(parts):            # every summary alone is already too long: shorten them
                return [s[:400] for s in parts][: max(1, room // 120)]
            merged = []
            for g in groups:
                if len(g) == 1:
                    merged.append(g[0])
                    continue
                try:
                    reply = self._call(MERGE_PROMPT + "\n".join(g), MERGE_SCHEMA, 300)
                    merged.append(_text(parse_json_object(reply.text).get("summary")) or g[0])
                except LLMConnectionError:
                    raise
                except LLMError:
                    merged.append(g[0])
            parts = merged
        return parts

    # ------------------------------------------------------------------ grounding checks
    def _grounded(self, keywords: Any, haystack: set[str]) -> list[str]:
        out = []
        for k in _dedupe(_text(x) for x in _as_list(keywords)):
            toks = _content_tokens(k)
            if toks and all(t in haystack for t in toks):
                if k.lower() not in (o.lower() for o in out):
                    out.append(k)
            else:
                self.diag.keywords_dropped += 1
        return out

    def _actions(self, value: Any, lines: dict[int, CleanedLine], owners: set[str]) -> list[ActionItem]:
        """Keep an action only if its owner exists, it cites real lines, and it shares a word with those lines."""
        out: list[ActionItem] = []
        for item in _as_list(value):
            if not isinstance(item, dict):
                continue
            owner = _text(item.get("owner"))
            owner = UNSPECIFIED if owner.lower() in ("", UNSPECIFIED, "unknown", "none") else owner
            action = _text(item.get("action"))
            ids = []
            for x in _as_list(item.get("lines", item.get("evidence_line_ids"))):
                try:
                    i = int(x)
                except (TypeError, ValueError, OverflowError):
                    continue
                if i in lines and i not in ids:
                    ids.append(i)
            evidence = _token_set([t for i in ids for t in (lines[i].source_text, lines[i].text, lines[i].hinglish,
                                                            lines[i].translation, romanise(lines[i].source_text))])
            if owner not in owners or not action or not ids or not (set(_content_tokens(action)) & evidence):
                self.diag.actions_dropped += 1
                continue
            if all((a.owner, a.action.lower()) != (owner, action.lower()) for a in out):
                out.append(ActionItem(owner, action, ids))
        return out


# ====================================================================== helpers
# Words that occur in almost every conversation, so they prove nothing about a keyword or an action.
_STOP = {
    "the", "and", "for", "with", "this", "that", "from", "are", "was", "has", "have", "not", "you", "your", "will",
    "about", "they", "them", "their", "there", "what", "which", "would", "been", "were", "into", "some", "then",
    "than", "also", "said", "says", "asks", "speaker", "speakers", "unspecified",
    "हाँ", "हां", "में", "है", "हैं", "और", "हम", "चाहिए", "नहीं", "तो", "का", "की", "के", "को", "से", "पर", "भी",
    "यह", "वह", "वो", "क्या", "कि", "ये", "था", "थे", "थी", "हो", "एक", "कर", "रहे", "रहा", "रही", "गया", "जी", "सर",
    "haan", "han", "hai", "hain", "mein", "aur", "hum", "chahiye", "nahi", "nahin", "kya", "yeh", "the", "tha",
    "thi", "rahe", "raha", "rahi", "gaya", "sir",
}


def _content_tokens(text: str) -> list[str]:
    """Tokens of at least 3 characters that are not stop words: the words that carry content."""
    return [t for t in _tokens(text) if len(t) >= 3 and t not in _STOP]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u0900-\u097f]+", normalise(text))


def _token_set(texts: list[str]) -> set[str]:
    return {t for s in texts for t in _tokens(s or "")}


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip() if isinstance(value, (str, int, float)) else ""


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _dedupe(items) -> list[str]:
    out: list[str] = []
    for s in items:
        if s and s not in out:
            out.append(s)
    return out


def _sentence(s: str) -> str:
    s = s.strip().rstrip(".")
    return f"{s[:1].upper()}{s[1:]}." if s else ""


def _kept(src: dict, note: str) -> CleanedLine:
    """The Stage-3 line, unchanged and flagged (used when the LLM gave nothing usable for it)."""
    return CleanedLine(src["line_id"], src["speaker"], src["start"], src["end"], src["text"],
                       src["hinglish"] or romanise(src["text"]), "", src["text"], False, True, _sentence(note))


def _fingerprint(source: dict[str, Any]) -> str:
    """SHA-256 of the transcript lines (the evidence Stage 4 used; timings are left out because they change)."""
    canonical = json.dumps(source.get("lines", []), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _line_records(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    raw = transcript.get("lines", [])
    if not isinstance(raw, list):
        raise ValueError("transcript JSON 'lines' must be a list")
    out = []
    for i, line in enumerate(raw, 1):
        if not isinstance(line, dict):
            raise ValueError(f"transcript line {i} is not an object")
        out.append({"line_id": i, "speaker": str(line.get("speaker", "Speaker_?")),
                    "start": float(line.get("start", 0.0)), "end": float(line.get("end", 0.0)),
                    "text": str(line.get("text", "")).strip(), "hinglish": str(line.get("hinglish", "")).strip()})
    return out


def _speaker_stats(lines: list[dict]) -> list[dict]:
    stats: dict[str, dict] = {}
    for x in lines:
        s = stats.setdefault(x["speaker"], {"speaker": x["speaker"], "talk_time_s": 0.0, "turns": 0})
        s["talk_time_s"] = round(s["talk_time_s"] + max(0.0, x["end"] - x["start"]), 2)
        s["turns"] += 1
    total = sum(s["talk_time_s"] for s in stats.values()) or 1.0
    for s in stats.values():
        s["share"] = round(s["talk_time_s"] / total, 3)
    return sorted(stats.values(), key=lambda s: s["speaker"])


__all__ = [
    "DEFAULT_MODEL", "DEFAULT_BASE_URL", "DEFAULT_CONTEXT", "LLMError", "LLMConnectionError", "LLMContextError",
    "LLMReply", "LLMBackend", "OpenAICompatibleLLM", "StaticLLM", "CleanedLine", "ActionItem", "Diagnostics",
    "PostProcessedReport", "PostProcessor", "parse_json_object", "chunk_schema", "estimate_tokens",
]
