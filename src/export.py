"""Stage 9 — build the released CSVs from the reviewed labels.

Reads data/review/review_done.csv (human-edited). If that is missing it falls
back to review.csv and WARNS that labels are unreviewed. Human columns win
over the machine pre_* columns. Rows with speaker_role=narrator or keep=0 are
dropped. Output: data/final/{turns,conversations,simple}.csv + stats.md.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.utils import Paths, get_logger, paths_from_config

log = get_logger("export")

SCAM_TYPES = ["fake_authority", "romance", "investment", "phishing_otp",
              "lottery_prize", "other_scam", "none"]

TURNS_COLUMNS = ["id", "video_id", "url", "start", "end", "turn_index",
                 "speaker_role", "text", "is_scam", "scam_type",
                 "is_after_onset", "transcript_source", "label_source"]


def _blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return str(v).strip() == ""


def _coalesce(human, pre, default=""):
    return human if not _blank(human) else (pre if not _blank(pre) else default)


def _review_df(paths: Paths) -> tuple[pd.DataFrame, bool]:
    done_path = paths.review / "review_done.csv"
    if done_path.exists():
        return pd.read_csv(done_path, dtype={"video_id": str}), True
    rev = paths.review / "review.csv"
    if not rev.exists():
        raise FileNotFoundError("no review.csv / review_done.csv — run prelabel first")
    log.warning("review_done.csv not found — exporting UNREVIEWED pre-labels")
    return pd.read_csv(rev, dtype={"video_id": str}), False


def _effective_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in df.iterrows():
        role = _coalesce(r.get("speaker_role"), r.get("pre_speaker_role"))
        keep = r.get("keep")
        keep = 1 if _blank(keep) else int(float(keep))
        if keep == 0 or role == "narrator":
            continue
        human_touched = any(not _blank(r.get(c)) for c in
                            ("speaker_role", "is_scam", "scam_type", "text_corrected"))
        out.append({
            "id": r.get("id"), "video_id": r.get("video_id"), "url": r.get("url"),
            "start": r.get("start"), "end": r.get("end"),
            "turn_index": r.get("turn_index"),
            "speaker_role": role,
            "text": _coalesce(r.get("text_corrected"), r.get("text")),
            "is_scam": int(float(_coalesce(r.get("is_scam"), r.get("pre_is_scam"), 0))),
            "scam_type": _coalesce(r.get("scam_type"), r.get("pre_scam_type"), "none"),
            "is_after_onset": int(float(_coalesce(r.get("is_after_onset"),
                                                  r.get("pre_is_after_onset"), 0))),
            "transcript_source": r.get("transcript_source") or "asr",
            "label_source": "human" if human_touched else r.get("label_source", "rule"),
        })
    return pd.DataFrame(out, columns=TURNS_COLUMNS)


def _conversations(turns: pd.DataFrame, durations: dict) -> pd.DataFrame:
    convs = []
    for vid, g in turns.groupby("video_id", sort=False):
        g = g.sort_values("turn_index")
        text = "\n".join(f"[{row.speaker_role or '?'}] {row.text}"
                         for row in g.itertuples())
        is_scam = int((g["is_scam"] == 1).any())
        nz = [t for t in g["scam_type"] if t and t != "none"]
        scam_type = max(set(nz), key=nz.count) if nz else "none"
        onset_rows = g[g["is_after_onset"] == 1]["turn_index"]
        onset = int(onset_rows.min()) if len(onset_rows) else None
        convs.append({"video_id": vid, "url": g["url"].iloc[0], "text": text,
                      "is_scam": is_scam, "scam_type": scam_type,
                      "n_turns": len(g), "scam_onset_turn": onset,
                      "duration_s": durations.get(vid)})
    return pd.DataFrame(convs)


def _stats_md(videos_n, conv: pd.DataFrame, turns: pd.DataFrame,
              durations: dict, reviewed: bool) -> str:
    hours = sum(d for d in durations.values() if d) / 3600.0
    lines = ["# Dataset stats", ""]
    lines.append(f"- reviewed labels: {'yes' if reviewed else 'NO (pre-labels)'}")
    lines.append(f"- videos with audio: {videos_n}")
    lines.append(f"- conversations: {len(conv)}")
    lines.append(f"- turns: {len(turns)}")
    lines.append(f"- audio hours (downloaded): {hours:.2f}")
    lines.append("")
    lines.append("## conversations by scam_type")
    vc = conv["scam_type"].value_counts()
    for t in SCAM_TYPES:
        if t in vc:
            lines.append(f"- {t}: {int(vc[t])}")
    lines.append("")
    lines.append("## scam vs not-scam (conversations)")
    lines.append(f"- scam: {int((conv['is_scam'] == 1).sum())}")
    lines.append(f"- not scam: {int((conv['is_scam'] == 0).sum())}")
    return "\n".join(lines) + "\n"


def run(cfg: dict, paths: Paths) -> None:
    enc = "utf-8-sig" if cfg.get("export", {}).get("bom", True) else "utf-8"
    df, reviewed = _review_df(paths)

    durations: dict = {}
    if paths.videos_csv.exists():
        vdf = pd.read_csv(paths.videos_csv, dtype={"video_id": str})
        for _, r in vdf.iterrows():
            d = r.get("duration_s")
            durations[r["video_id"]] = None if _blank(d) else float(d)

    turns = _effective_rows(df)
    conv = _conversations(turns, durations)
    simple = conv[["text", "is_scam", "scam_type"]].copy()

    turns.to_csv(paths.final / "turns.csv", index=False, encoding=enc)
    conv.to_csv(paths.final / "conversations.csv", index=False, encoding=enc)
    simple.to_csv(paths.final / "simple.csv", index=False, encoding=enc)

    stats = _stats_md(len(durations), conv, turns, durations, reviewed)
    (paths.final / "stats.md").write_text(stats, encoding="utf-8")
    log.info("export done: %d turns, %d conversations", len(turns), len(conv))
    print(stats)


if __name__ == "__main__":
    from src.utils import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg.get("log_level", "INFO"))
    run(cfg, paths_from_config(cfg))
