"""End-to-end: real IndicVoices speech → simulated conversation → both pipeline orders.

Slow (loads every model); run with ``pytest -m slow``.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from whospoke.corpora import RAW, conversation_sides, read_indicvoices, recording_utterances
from whospoke.metrics import cp_error, der
from whospoke.synth import SynthConfig, simulate

VALID = RAW / "indicvoices" / "hindi" / "valid-00000-of-00001.parquet"
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not VALID.exists(), reason="IndicVoices valid shard not downloaded")]


@pytest.fixture(scope="module")
def conversation():
    df = read_indicvoices(VALID, with_audio=True)
    sides = conversation_sides(df).sample(frac=1.0, random_state=11)
    spk = {f"S{i + 1}": recording_utterances(df, rec) for i, rec in enumerate(sides.recording.iloc[:2])}
    return simulate("it", spk, SynthConfig(target_s=35, overlap_prob=0.5, overlap_s=(1.0, 2.5), snr_db=15, seed=5),
                    noise=np.random.default_rng(0).standard_normal(16000 * 120).astype(np.float32) * 0.01)


@pytest.fixture(scope="module")
def asr():
    from whospoke.pipeline import make_asr

    return make_asr("indicconformer")


@pytest.mark.parametrize("order", ["A", "B"])
def test_pipeline_end_to_end(order, conversation, asr, tmp_path):
    from whospoke.pipeline import Pipeline

    res = Pipeline(order, asr_model=asr).run(conversation.mixture)
    ref = [(s.start, s.end, s.speaker) for s in conversation.activity()]
    assert der(ref, res.diarization.tuples())["der"] < 0.5
    ref_text = {}
    for s in conversation.segments:
        ref_text.setdefault(s.speaker, []).append(s.text)
    score = cp_error({k: " ".join(v) for k, v in ref_text.items()}, res.text_by_speaker())
    assert score["rate"] < 0.8
    assert set(res.timings) >= {"separation", "diarization", "asr", "total"}

    out = res.save(tmp_path / order)
    timeline = json.loads((out / "timeline.json").read_text(encoding="utf-8"))
    assert timeline and {"start", "end", "speaker"} <= set(timeline[0])
    assert (out / "transcript_hinglish.txt").read_text(encoding="utf-8").strip()
    assert "-->" in (out / "transcript.srt").read_text(encoding="utf-8")


def test_order_b_separates_only_overlapping_stretches(conversation, asr):
    from whospoke.audio import SR
    from whospoke.diarization import Turn
    from whospoke.pipeline import Pipeline, overlap_intervals

    pipe = Pipeline("B", asr_model=asr)                       # default sep_mode="splice"
    diar = pipe.diarizer(conversation.mixture)
    assert "overlap" in diar.info, "overlap detection must always run in Order B"
    # Make the splice logic testable whatever the detector finds on this synthetic clip: add the true overlaps.
    truth = overlap_intervals([Turn(s.start, s.end, s.speaker) for s in conversation.activity()])
    assert truth, "fixture must contain overlapping speech"
    diar.info["overlap"] = sorted(list(diar.info["overlap"]) + truth)
    regions = pipe._overlap_regions(diar)
    units = pipe._targeted_separation(conversation.mixture, diar)
    assert any(src == "separated" for _, _, src in units)
    for turn, audio, source in units:
        hits = [(a, b) for a, b in regions if a < turn.end and b > turn.start]
        if source == "mixture":
            assert audio is conversation.mixture
            continue
        assert hits
        # outside every overlap region the turn's audio is untouched mixture
        keep = np.ones(len(audio), bool)
        for a, b in hits:
            keep[int(a * SR): int(b * SR)] = False
        assert np.array_equal(audio[keep], conversation.mixture[keep])
