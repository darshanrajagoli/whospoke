"""Stage 4: the guardrails around the LLM, with scripted replies (no model needed).

``pytest -m slow`` also runs the real local LLM server on the demo transcript, if one is running.
"""
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from whospoke.llm_postprocess import (
    LLMConnectionError, LLMContextError, LLMError, LLMReply, OpenAICompatibleLLM, PostProcessor, StaticLLM,
    chunk_schema, estimate_tokens, parse_json_object,
)

DEMO = Path(__file__).resolve().parents[1] / "results" / "demo" / "transcript.json"


def transcript():
    return {
        "order": "B",
        "duration_s": 9.0,
        "lines": [
            {"speaker": "Speaker_A", "start": 0.0, "end": 2.0,
             "text": "आई डी कार्ड गुम हो गया है सर सर", "hinglish": "aai D card gum ho gaya hai sir sir"},
            {"speaker": "Speaker_B", "start": 2.1, "end": 4.0,
             "text": "आधार कार्ड की फोटो कॉपी लाना", "hinglish": "aadhar card ki photo copy lana"},
            {"speaker": "Speaker_A", "start": 4.2, "end": 4.5, "text": "", "hinglish": ""},
            {"speaker": "Speaker_A", "start": 5.0, "end": 9.0,
             "text": "ठीक है कल ले आऊंगा", "hinglish": "theek hai kal le aaunga"},
        ],
    }


def chunk_reply(**changes):
    lines = [
        {"id": 1, "text": "आई डी कार्ड गुम हो गया है सर।", "en": "My ID card has been lost, sir.", "uncertain": False},
        {"id": 2, "text": "आधार कार्ड की फोटो कॉपी लाना।", "en": "Bring a photocopy of the Aadhaar card.",
         "uncertain": False},
        {"id": 4, "text": "ठीक है, कल ले आऊंगा।", "en": "Okay, I will bring it tomorrow.", "uncertain": False},
    ]
    obj = {"lines": lines, "summary": "A lost ID card and the documents needed to replace it.",
           "keywords": ["ID card", "Aadhaar card"],
           "actions": [{"owner": "Speaker_A", "action": "Bring a photocopy of the Aadhaar card", "lines": [2, 4]}]}
    obj.update(changes)
    return json.dumps(obj, ensure_ascii=False)


def synthesis_reply(**changes):
    obj = {"title": "Lost ID card", "topic": "replacing a lost ID card",
           "summary": "Speaker_A has lost their ID card. Speaker_B asks for a photocopy of the Aadhaar card.",
           "key_points": ["The ID card is lost.", "An Aadhaar photocopy is needed."],
           "keywords": ["ID card", "Aadhaar card", "photocopy"],
           "actions": [{"owner": "Speaker_A", "action": "Bring a photocopy of the Aadhaar card", "lines": [2, 4]}]}
    obj.update(changes)
    return json.dumps(obj, ensure_ascii=False)


def run(*replies, **kw):
    llm = StaticLLM(list(replies))
    return PostProcessor(llm, **kw).process_transcript(transcript()), llm


# ---------------------------------------------------------------- parsing and schema
def test_parse_json_object_tolerates_fences_and_stray_words():
    assert parse_json_object("```json\n{\"x\": 1}\n```") == {"x": 1}
    assert parse_json_object("Here it is: {\"x\": [1, 2]} hope that helps") == {"x": [1, 2]}
    with pytest.raises(LLMError):
        parse_json_object("no json here")


def test_chunk_schema_pins_every_line_number_in_order():
    src = [{"line_id": 3, "text": "क"}, {"line_id": 7, "text": "ख"}]
    lines = chunk_schema(src)["properties"]["lines"]
    assert [x["properties"]["id"]["const"] for x in lines["prefixItems"]] == [3, 7]
    assert "items" not in lines        # llama.cpp would let "items" override the tuple


def test_token_estimate_is_conservative_for_devanagari():
    assert estimate_tokens("आधार कार्ड की फोटो कॉपी") > estimate_tokens("aadhar card ki photo copy")


# ---------------------------------------------------------------- the guardrails
def test_happy_path_report():
    report, llm = run(chunk_reply(), synthesis_reply())
    assert len(llm.calls) == 2                              # one chunk + the synthesis
    assert [x.line_id for x in report.cleaned_dialogue] == [1, 2, 4]   # the empty line 3 is skipped
    assert report.diagnostics.empty_lines_skipped == 1
    first = report.cleaned_dialogue[0]
    assert (first.speaker, first.start, first.end) == ("Speaker_A", 0.0, 2.0)
    assert first.changed and first.source_text.endswith("सर सर")
    assert first.hinglish.startswith("aai D card") or first.hinglish.startswith("aai")   # from the romaniser
    assert report.title == "Lost ID card"
    assert report.keywords == ["ID card", "Aadhaar card", "photocopy"]
    assert report.action_items[0].owner == "Speaker_A"
    assert report.uncertain_lines == []
    assert {s["speaker"] for s in report.speakers} == {"Speaker_A", "Speaker_B"}
    # speaker and time never reach the model's output format, only its input
    assert "speaker" not in json.dumps(llm.calls[0]["schema"]["properties"]["lines"])


def test_unknown_line_ids_are_dropped_and_missing_lines_restored():
    obj = json.loads(chunk_reply())
    obj["lines"] = [obj["lines"][0], {"id": 99, "text": "invented", "en": "Invented.", "uncertain": False}]
    report, _ = run(json.dumps(obj, ensure_ascii=False), synthesis_reply())
    assert [x.line_id for x in report.cleaned_dialogue] == [1, 2, 4]
    assert report.diagnostics.unknown_line_ids == 1
    assert report.diagnostics.lines_restored == 2
    restored = report.cleaned_dialogue[1]
    assert restored.uncertain and not restored.changed and "skipped" in restored.note
    assert restored.text == "आधार कार्ड की फोटो कॉपी लाना"


def test_over_edited_line_is_reverted():
    obj = json.loads(chunk_reply())
    obj["lines"][1]["text"] = "My ID card was lost"          # translated instead of repaired
    report, _ = run(json.dumps(obj, ensure_ascii=False), synthesis_reply())
    line = report.cleaned_dialogue[1]
    assert line.text == line.source_text and not line.changed and line.uncertain
    assert report.diagnostics.lines_reverted == 1


def test_missing_translation_and_model_flag_are_reported():
    obj = json.loads(chunk_reply())
    obj["lines"][0]["en"] = ""
    obj["lines"][2]["uncertain"] = True
    report, _ = run(json.dumps(obj, ensure_ascii=False), synthesis_reply())
    assert report.uncertain_lines == [1, 4]
    assert report.diagnostics.missing_translations == 1
    assert report.diagnostics.lines_flagged_by_llm == 1
    assert "translation" in report.cleaned_dialogue[0].note


def test_ungrounded_keywords_and_unsupported_actions_are_dropped():
    synth = synthesis_reply(
        keywords=["ID card", "weather forecast", "election"],
        actions=[{"owner": "Speaker_Z", "action": "x", "lines": [1]},           # unknown speaker
                 {"owner": "Speaker_B", "action": "Check the CCTV", "lines": [42]},   # no valid evidence
                 {"owner": "unknown", "action": "Bring the photocopy", "lines": [2]}])
    report, _ = run(chunk_reply(), synth)
    assert report.keywords == ["ID card"]
    assert report.diagnostics.keywords_dropped == 2
    assert [(a.owner, a.evidence_line_ids) for a in report.action_items] == [("unspecified", [2])]


def test_invalid_reply_splits_the_chunk_then_keeps_failed_line():
    half1 = json.loads(chunk_reply())
    half1["lines"] = half1["lines"][:1]
    half2 = json.loads(chunk_reply())
    half2["lines"] = half2["lines"][1:]
    # 3 lines: garbage → split into [1] and [2, 4]; [1] fails again → kept; [2, 4] succeeds
    report, llm = run("not json", "still not json", json.dumps(half2, ensure_ascii=False), synthesis_reply())
    assert len(llm.calls) == 4
    d = report.diagnostics
    assert (d.invalid_replies, d.chunk_splits, d.failed_chunks) == (2, 1, 1)
    assert report.cleaned_dialogue[0].uncertain and not report.cleaned_dialogue[0].changed
    assert report.cleaned_dialogue[1].translation.startswith("Bring")


def test_truncated_reply_and_context_error_split_the_chunk():
    one = json.loads(chunk_reply())
    one["lines"] = one["lines"][:1]
    two = json.loads(chunk_reply())
    two["lines"] = two["lines"][1:]
    replies = [LLMReply('{"lines": [', "length"), LLMContextError("too long"),
               json.dumps(two, ensure_ascii=False), synthesis_reply()]
    report, llm = run(*replies)
    assert report.diagnostics.chunk_splits == 1
    assert report.cleaned_dialogue[0].uncertain and "context" in report.cleaned_dialogue[0].note


def test_synthesis_failure_falls_back_to_chunk_results():
    report, _ = run(chunk_reply(), "garbage")
    assert report.diagnostics.synthesis_failed
    assert report.executive_summary.startswith("A lost ID card")
    assert report.keywords == ["ID card", "Aadhaar card"]
    assert report.action_items and report.title == "Conversation report"


def test_connection_error_stops_the_run():
    with pytest.raises(LLMConnectionError):
        run(LLMConnectionError("down"))


def test_long_transcripts_are_chunked_to_fit_the_context():
    lines = [{"speaker": f"Speaker_{'AB'[i % 2]}", "start": float(i), "end": i + 0.9,
              "text": "यह एक लंबी बातचीत की पंक्ति है जिसमें कई शब्द हैं", "hinglish": ""} for i in range(40)]
    post = PostProcessor(StaticLLM([]), context_tokens=1024)
    chunks = post._plan_chunks([dict(x, line_id=i + 1) for i, x in enumerate(lines)])
    assert sum(map(len, chunks)) == 40 and len(chunks) > 3
    assert max(map(len, chunks)) <= 12


def test_empty_transcript_needs_no_llm():
    llm = StaticLLM([])
    report = PostProcessor(llm).process_transcript({"lines": [{"speaker": "A", "start": 0, "end": 1, "text": ""}]})
    assert llm.calls == [] and report.cleaned_dialogue == []
    assert "No speech" in report.executive_summary


def test_report_files(tmp_path):
    report, _ = run(chunk_reply(), synthesis_reply())
    report.save(tmp_path)
    data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert {"executive_summary", "keywords", "action_items", "cleaned_dialogue", "speakers", "diagnostics",
            "provenance"} <= set(data)
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    for heading in ("## Executive summary", "## Keywords", "## Action items", "## Speakers", "## Dialogue",
                    "## What Stage 4 changed", "## How this report was made"):
        assert heading in md
    assert "`ID card`" in md


# ---------------------------------------------------------------- the HTTP client, against a fake server
class FakeServer:
    """Minimal OpenAI-compatible server: records requests, answers with a scripted list of (status, body)."""

    def __init__(self, answers):
        self.answers, self.requests = list(answers), []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data": []}')

            def do_POST(self):
                outer.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                status, body = outer.answers.pop(0)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def ok(text, finish="stop"):
    return 200, {"choices": [{"message": {"content": text}, "finish_reason": finish}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def test_client_sends_schema_and_reads_usage():
    srv = FakeServer([ok('{"a": 1}')])
    try:
        llm = OpenAICompatibleLLM(srv.url, "m")
        llm.ping()
        r = llm.complete("s", "u", temperature=0, max_tokens=50, schema={"type": "object"})
        assert (r.text, r.finish_reason, r.prompt_tokens, r.completion_tokens) == ('{"a": 1}', "stop", 10, 5)
        assert srv.requests[0]["response_format"] == {"type": "json_object", "schema": {"type": "object"}}
        assert srv.requests[0]["temperature"] == 0 and srv.requests[0]["max_tokens"] == 50
    finally:
        srv.close()


def test_client_drops_schema_when_server_rejects_it():
    srv = FakeServer([(400, {"error": "response_format not supported"}), ok("{}"), ok("{}")])
    try:
        llm = OpenAICompatibleLLM(srv.url, "m")
        llm.complete("s", "u", temperature=0, max_tokens=5, schema={"type": "object"})
        llm.complete("s", "u", temperature=0, max_tokens=5, schema={"type": "object"})
        assert not llm.schema_enforced
        assert "response_format" not in srv.requests[1] and "response_format" not in srv.requests[2]
    finally:
        srv.close()


def test_client_reports_context_errors():
    srv = FakeServer([(400, {"error": {"message": "the request exceeds the available context size"}})])
    try:
        with pytest.raises(LLMContextError):
            OpenAICompatibleLLM(srv.url, "m").complete("s", "u", temperature=0, max_tokens=5, schema={"type": "object"})
    finally:
        srv.close()


def test_client_reports_unreachable_server():
    llm = OpenAICompatibleLLM("http://127.0.0.1:9/v1", "m", timeout_s=2)
    with pytest.raises(LLMConnectionError, match="serve_llm.py"):
        llm.ping()
    with pytest.raises(LLMConnectionError):
        llm.complete("s", "u", temperature=0, max_tokens=5)


def test_postprocessor_through_http(tmp_path):
    srv = FakeServer([ok(chunk_reply()), ok(synthesis_reply())])
    try:
        report = PostProcessor(base_url=srv.url).process_transcript(transcript())
        assert report.diagnostics.schema_enforced and report.diagnostics.llm_calls == 2
        assert report.diagnostics.prompt_tokens == 20
        prompt = srv.requests[0]["messages"][1]["content"]
        assert "1 | Speaker_A | 00:00-00:02 | आई डी कार्ड" in prompt
    finally:
        srv.close()


# ---------------------------------------------------------------- the pipeline runs Stage 4 after Stage 3
class _Diarizer:
    name = "stub"

    def __call__(self, wav, n_speakers=None):
        from whospoke.diarization import Diarization, Turn

        return Diarization([Turn(0.0, 2.0, "Speaker_A"), Turn(2.1, 4.0, "Speaker_B"), Turn(5.0, 9.0, "Speaker_A")])


class _ASR:
    name = "stub"

    def transcribe(self, pieces):
        texts = ["आई डी कार्ड गुम हो गया है सर सर", "आधार कार्ड की फोटो कॉपी लाना", "ठीक है कल ले आऊंगा"]
        return texts[: len(pieces)]


def _stub_pipeline(postprocessor):
    pytest.importorskip("torch")
    from whospoke.pipeline import Pipeline

    return Pipeline("B", separator=None, diarizer=_Diarizer(), asr_model=_ASR(), postprocessor=postprocessor)


def test_pipeline_writes_the_stage4_report(tmp_path):
    llm = StaticLLM([chunk_reply().replace('"id": 4', '"id": 3'), synthesis_reply(actions=[])])
    res = _stub_pipeline(PostProcessor(llm)).run(np.zeros(9 * 16000, np.float32))
    assert "llm" in res.timings and res.report["title"] == "Lost ID card"
    out = res.save(tmp_path)
    assert {"report.json", "report.md", "transcript.json", "timeline.json"} <= {p.name for p in out.iterdir()}
    saved = json.loads((out / "transcript.json").read_text(encoding="utf-8"))
    canonical = json.dumps(saved["lines"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # the provenance hash matches the saved transcript, although its timings changed after Stage 4
    assert res.report["provenance"]["source_sha256"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_pipeline_keeps_stage3_when_stage4_fails(tmp_path):
    res = _stub_pipeline(PostProcessor(StaticLLM([LLMConnectionError("down")]))).run(np.zeros(9 * 16000, np.float32))
    assert res.report is None and "down" in res.report_error
    out = res.save(tmp_path)
    assert (out / "transcript.json").exists() and not (out / "report.json").exists()


def test_pipeline_without_postprocessor_is_unchanged():
    res = _stub_pipeline(None).run(np.zeros(9 * 16000, np.float32))
    assert res.report is None and "llm" not in res.timings


# ---------------------------------------------------------------- the real model (pytest -m slow)
@pytest.mark.slow
def test_real_llm_on_demo_transcript(tmp_path):
    llm = OpenAICompatibleLLM()
    try:
        llm.ping()
    except LLMConnectionError:
        pytest.skip("no Stage-4 LLM server running (python scripts/serve_llm.py)")
    report = PostProcessor(llm).process_file(DEMO)
    src = json.loads(DEMO.read_text(encoding="utf-8"))
    kept = [x for x in src["lines"] if x["text"].strip()]
    assert [(x.speaker, x.start, x.end) for x in report.cleaned_dialogue] == \
        [(x["speaker"], x["start"], x["end"]) for x in kept]
    assert report.executive_summary and report.keywords
    assert report.diagnostics.failed_chunks <= len(kept) // 4
    report.save(tmp_path)
