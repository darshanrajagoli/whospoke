"""Timeline-building logic of the diarizer (no neural models needed)."""
import numpy as np

from whospoke.diarization import (FRAME, Diarization, Turn, _frame_labels, _smooth, _turns_from_frames,
                                  add_overlap_speakers, merge_turns, sliding_windows)
from whospoke.vad import complement, intersect, total


def test_sliding_windows_cover_regions():
    wins = sliding_windows([(0.0, 5.0), (6.0, 6.8), (7.0, 7.2)], win=1.5, hop=0.75)
    assert wins[0] == (0.0, 1.5) and wins[-2][1] == 5.0
    assert (6.0, 6.8) in wins                   # short region → one window
    assert all(e - s >= 0.4 for s, e in wins)   # 0.2 s region dropped


def test_frame_labels_and_turns():
    wins = [(0.0, 1.5), (0.75, 2.25), (3.0, 4.5)]
    lab = _frame_labels(wins, np.array([0, 0, 1]), [(0.0, 2.25), (3.0, 4.5)], 5.0, 2)
    turns = _turns_from_frames(lab, {0: "Speaker_A", 1: "Speaker_B"})
    assert [t.speaker for t in turns] == ["Speaker_A", "Speaker_B"]
    assert abs(turns[0].end - 2.25) <= FRAME and abs(turns[1].start - 3.0) <= FRAME


def test_smoothing_absorbs_flicker():
    lab = np.array([0] * 20 + [1] * 2 + [0] * 20)
    assert set(_smooth(lab, 6).tolist()) == {0}
    keep = np.array([0] * 20 + [1] * 10)
    assert set(_smooth(keep, 6).tolist()) == {0, 1}


def test_overlap_second_speaker_is_nearest_other():
    turns = [Turn(0, 5, "Speaker_A"), Turn(5, 9, "Speaker_B"), Turn(20, 25, "Speaker_C")]
    out = add_overlap_speakers(turns, [(4.2, 5.0)])
    b = [t for t in out if t.speaker == "Speaker_B"]
    assert b[0].start == 4.2                     # B extended back into the overlap
    assert not any(t.speaker == "Speaker_C" and t.start < 10 for t in out)


def test_merge_turns():
    out = merge_turns([Turn(0, 2, "A"), Turn(1.5, 3, "A"), Turn(3.1, 4, "A"), Turn(2, 3, "B")], gap=0.2)
    assert [(t.start, t.end, t.speaker) for t in out] == [(0, 4, "A"), (2, 3, "B")]


def test_region_helpers():
    assert complement([(1, 2), (3, 4)], 5) == [(0, 1), (2, 3), (4, 5)]
    assert intersect([(0, 2), (3, 5)], [(1, 4)]) == [(1, 2), (3, 4)]
    assert total([(0, 1), (2, 4)]) == 3


def test_timeline_json_format():
    d = Diarization([Turn(12.2, 45.4, "Speaker_A"), Turn(46.0, 75.1, "Speaker_B")])
    tl = d.timeline_json()
    assert tl[0]["start"] == "00:12" and tl[0]["end"] == "00:45" and tl[1]["speaker"] == "Speaker_B"
