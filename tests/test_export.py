"""CSV schema + label-coalescing tests (acceptance criterion 4)."""
import pandas as pd

from src.export import TURNS_COLUMNS, _effective_rows
from src.prelabel import REVIEW_COLUMNS, classify_rules, load_keywords


def _row(**kw):
    base = {c: "" for c in REVIEW_COLUMNS}
    base.update(kw)
    return base


def test_turns_schema_matches_spec():
    assert TURNS_COLUMNS == [
        "id", "video_id", "url", "start", "end", "turn_index", "speaker_role",
        "text", "is_scam", "scam_type", "is_after_onset", "transcript_source",
        "label_source"]


def test_human_label_overrides_prelabel():
    df = pd.DataFrame([_row(
        id="v_000", video_id="v", url="u", turn_index=0, text="hi",
        pre_is_scam=1, pre_scam_type="phishing_otp", pre_speaker_role="scammer",
        is_scam=0, scam_type="none", speaker_role="receiver")])
    out = _effective_rows(df)
    assert out.loc[0, "is_scam"] == 0
    assert out.loc[0, "scam_type"] == "none"
    assert out.loc[0, "label_source"] == "human"


def test_prelabel_used_when_no_human_edit():
    df = pd.DataFrame([_row(
        id="v_000", video_id="v", url="u", turn_index=0, text="hi",
        pre_is_scam=1, pre_scam_type="investment", pre_speaker_role="scammer",
        label_source="rule")])
    out = _effective_rows(df)
    assert out.loc[0, "is_scam"] == 1
    assert out.loc[0, "scam_type"] == "investment"
    assert out.loc[0, "label_source"] == "rule"


def test_narrator_and_keep0_dropped():
    df = pd.DataFrame([
        _row(id="a", video_id="v", turn_index=0, text="x", pre_speaker_role="narrator"),
        _row(id="b", video_id="v", turn_index=1, text="y", keep=0,
             pre_speaker_role="scammer"),
        _row(id="c", video_id="v", turn_index=2, text="z", pre_speaker_role="victim",
             pre_is_scam=1),
    ])
    out = _effective_rows(df)
    assert list(out["id"]) == ["c"]


def test_classify_rules_detects_type():
    kw = load_keywords("labels/keywords.yaml")
    turns = [{"turn_index": 0, "speaker": "SPEAKER_00", "text": "ผมเป็นตำรวจ คุณมีคดีฟอกเงิน"},
             {"turn_index": 1, "speaker": "SPEAKER_01", "text": "จริงเหรอ ไม่เข้าใจ"}]
    info = classify_rules(turns, kw)
    assert info["is_scam"] == 1
    assert info["scam_type"] == "fake_authority"
    assert info["onset"] == 0
    assert info["scammer_speaker"] == "SPEAKER_00"
