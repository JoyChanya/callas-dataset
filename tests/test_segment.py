"""ASR<->diarization alignment tests (acceptance criterion 4)."""
from pathlib import Path

from src.segment import assign_speaker, merge_turns, parse_rttm

DIAR = [
    {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
    {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
]


def test_assign_by_max_overlap():
    assert assign_speaker({"start": 0.5, "end": 4.0, "text": "a"}, DIAR) == "SPEAKER_00"
    assert assign_speaker({"start": 6.0, "end": 9.0, "text": "b"}, DIAR) == "SPEAKER_01"


def test_assign_straddle_picks_larger():
    # 4.0-7.0 overlaps SPK0 by 1s, SPK1 by 2s -> SPK1
    assert assign_speaker({"start": 4.0, "end": 7.0, "text": "c"}, DIAR) == "SPEAKER_01"


def test_assign_zero_length_nearest():
    assert assign_speaker({"start": 8.0, "end": 8.0, "text": "d"}, DIAR) == "SPEAKER_01"


def test_merge_consecutive_same_speaker():
    labeled = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "สวัสดี"},
        {"speaker": "SPEAKER_00", "start": 1.2, "end": 2.0, "text": "ครับ"},
        {"speaker": "SPEAKER_01", "start": 2.5, "end": 4.0, "text": "ว่าไง"},
    ]
    turns = merge_turns(labeled, merge_gap_s=0.75, min_turn_s=0.5)
    assert len(turns) == 2
    assert turns[0]["text"] == "สวัสดี ครับ"
    assert turns[0]["turn_index"] == 0 and turns[1]["turn_index"] == 1


def test_merge_gap_too_big_splits():
    labeled = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "a"},
        {"speaker": "SPEAKER_00", "start": 5.0, "end": 6.0, "text": "b"},
    ]
    assert len(merge_turns(labeled, merge_gap_s=0.75, min_turn_s=0.1)) == 2


def test_empty_text_dropped():
    labeled = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "   "}]
    assert merge_turns(labeled, 0.75, 0.5) == []


def test_parse_rttm(tmp_path: Path):
    rttm = ("SPEAKER f 1 0.000 5.000 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
            "SPEAKER f 1 5.000 3.000 <NA> <NA> SPEAKER_01 <NA> <NA>\n")
    p = tmp_path / "f.rttm"
    p.write_text(rttm, encoding="utf-8")
    got = parse_rttm(p)
    assert got[0] == {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}
    assert got[1]["end"] == 8.0
