"""Stage-4 Task B probe: reproduce F2 (action-item extraction) on GROUP-7 DEV examples only.

Runs the REAL local Airavata server over each group-7 *reference* transcript, records the
actions the model PROPOSES (before the Python gate), and reports how many survived, how many
were dropped, and the raw proposals so the cause (model emits nothing vs. gate rejects) is visible.
Group 7 only (D32): scored groups 0-6 are never touched here.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import whospoke.llm_postprocess as lp                                    # noqa: E402
from whospoke.llm_postprocess import PostProcessor, OpenAICompatibleLLM, LLMConnectionError  # noqa: E402

IDS = [f"test_g07_3spk_ovl-{o}_{n}" for o in ("high", "low", "none") for n in ("clean", "market5", "village10")]
BASE = Path("results/eval_test_indicconformer")


def stage3(conv, system):
    if system == "reference":
        lines = [{"speaker": s["speaker"], "start": s["start"], "end": s["end"], "text": s["text"],
                  "hinglish": lp.romanise(s["text"])} for s in sorted(conv["reference"], key=lambda s: s["start"])]
    else:
        lines = conv["systems"][system]
    return {"order": "oracle" if system != "B-spectral" else "B",
            "duration_s": conv.get("duration_s", 0.0), "lines": lines}


class Recorder(OpenAICompatibleLLM):
    def complete(self, system, user, **kw):
        r = super().complete(system, user, **kw)
        self.raw.append({"user": user, "reply": r.text})
        return r


def classify(a, by_id, valid_owners):
    """Why would _actions keep/reject this proposed action (mirror of its gate, for diagnosis only)."""
    owner = (a.get("owner") or "").strip() or "unspecified"
    ids = [int(x) for x in (a.get("lines") or []) if str(x).lstrip("-").isdigit()]
    real = [i for i in ids if i in by_id]
    reason = []
    if owner not in valid_owners and owner not in ("unspecified",):
        reason.append("owner_not_a_speaker")
    if not real:
        reason.append("no_real_evidence_line")
    return owner, ids, real, reason


print("PROBE_START rev=%s" % lp.PROMPT_SCHEMA_REVISION, flush=True)
try:
    OpenAICompatibleLLM().ping()
    print("SERVER_OK", flush=True)
except LLMConnectionError as e:
    print("SERVER_DOWN %s" % e, flush=True)
    sys.exit(3)

for cid in IDS:
    conv = json.loads((BASE / f"{cid}.json").read_text(encoding="utf-8"))
    src = stage3(conv, "reference")
    rec = Recorder()
    rec.raw = []
    rep = PostProcessor(rec).process_transcript(src)
    proposed, diag = [], []
    valid_owners = {s["speaker"] for s in rep.speakers} | {lp.UNSPECIFIED}
    by_id = {x.line_id: x for x in rep.cleaned_dialogue}
    for e in rec.raw:
        try:
            obj = lp.parse_json_object(e["reply"])
        except Exception:
            continue
        for a in (obj.get("actions") or []):
            if isinstance(a, dict):
                proposed.append(a)
                diag.append({"owner": a.get("owner"), "action": (a.get("action") or "")[:120],
                             "lines": a.get("lines"), "why": classify(a, by_id, valid_owners)[3]})
    kept = [{"owner": x.owner, "action": x.action, "lines": x.evidence_line_ids} for x in rep.action_items]
    print(json.dumps({"cid": cid, "lines": len(rep.cleaned_dialogue), "llm_calls": rep.diagnostics.llm_calls,
                      "model_proposed": len(proposed), "dropped": rep.diagnostics.actions_dropped,
                      "kept": len(kept), "kept_items": kept, "proposed_detail": diag}, ensure_ascii=False), flush=True)
print("DONE", flush=True)
