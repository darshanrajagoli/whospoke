"""Stage-4 Task B experiment: does a MORE EXPLICIT action instruction (group-7 dev only) make the
model actually emit action items, and do those pass the (unchanged) Python grounding gate?

Same 2 clearest group-7 references as the baseline probe. Only the ACTION wording in the two
prompts is strengthened (appended clarification); every guardrail, schema and the evidence rules
stay identical. We record what the model PROPOSES vs what survives, to decide if prompt wording is
a demonstrated cause of F2. Scored groups 0-6 are never used (D32).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import whospoke.llm_postprocess as lp                                    # noqa: E402
from whospoke.llm_postprocess import PostProcessor, OpenAICompatibleLLM, LLMConnectionError  # noqa: E402

EXTRA = (" Pay special attention to ACTION ITEMS: actively look for any instruction, request or commitment "
         "\u2014 anything a speaker is told to do, asked to do, or agrees to do (bring something, come somewhere, "
         "take a test, stop doing something). For each, set \"owner\" to the speaker who must act (or \"unspecified\"), "
         "\"action\" to what they must do in English, and \"lines\" to the exact line numbers where it is said. "
         "Do not leave this list empty if the dialogue contains anything someone should do.")

IDS = ["test_g07_3spk_ovl-high_clean", "test_g07_3spk_ovl-high_market5"]
BASE = Path("results/eval_test_indicconformer")


def stage3(conv, system):
    lines = [{"speaker": s["speaker"], "start": s["start"], "end": s["end"], "text": s["text"],
              "hinglish": lp.romanise(s["text"])} for s in sorted(conv["reference"], key=lambda s: s["start"])]
    return {"order": "oracle", "duration_s": conv.get("duration_s", 0.0), "lines": lines}


class Recorder(OpenAICompatibleLLM):
    def complete(self, system, user, **kw):
        r = super().complete(system, user, **kw)
        self.raw.append({"user": user, "reply": r.text})
        return r


print("VARIANT_B_START same_rev=%s" % lp.PROMPT_SCHEMA_REVISION, flush=True)
try:
    OpenAICompatibleLLM().ping()
    print("SERVER_OK", flush=True)
except LLMConnectionError as e:
    print("SERVER_DOWN %s" % e, flush=True)
    sys.exit(3)

lp.CHUNK_PROMPT = lp.CHUNK_PROMPT + EXTRA          # only the wording of the action instruction changes
lp.SYNTHESIS_PROMPT = lp.SYNTHESIS_PROMPT + EXTRA

for cid in IDS:
    conv = json.loads((BASE / f"{cid}.json").read_text(encoding="utf-8"))
    src = stage3(conv, "reference")
    rec = Recorder()
    rec.raw = []
    rep = PostProcessor(rec).process_transcript(src)
    proposed, detail = [], []
    valid_owners = {s["speaker"] for s in rep.speakers} | {lp.UNSPECIFIED}
    by_id = {x.line_id: x for x in rep.cleaned_dialogue}
    for e in rec.raw:
        try:
            obj = lp.parse_json_object(e["reply"])
        except Exception:
            continue
        for a in (obj.get("actions") or []):
            if isinstance(a, dict):
                ids = [int(x) for x in (a.get("lines") or []) if str(x).lstrip("-").isdigit()]
                real = [i for i in ids if i in by_id]
                why = ("owner" if (a.get("owner") or "").strip() not in valid_owners and
                       (a.get("owner") or "").strip() not in ("", "unspecified") else "") + \
                      ("no_real_line" if not real else "")
                proposed.append(a)
                detail.append({"owner": a.get("owner"), "action": (a.get("action") or "")[:140],
                               "lines": ids, "reject_if": why})
    kept = [{"owner": x.owner, "action": x.action, "lines": x.evidence_line_ids} for x in rep.action_items]
    print(json.dumps({"cid": cid, "variant": "B", "calls": rep.diagnostics.llm_calls,
                      "model_proposed": len(proposed), "dropped": rep.diagnostics.actions_dropped,
                      "kept": len(kept), "kept_items": kept, "proposed_detail": detail}, ensure_ascii=False), flush=True)
print("DONE", flush=True)
