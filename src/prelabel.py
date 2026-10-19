"""Stage 7 — pre-label turns to speed up human review -> data/review/review.csv

Default: keyword rules from labels/keywords.yaml (fast, transparent).
Optional (--llm): a LOCAL open-source instruct model served by vLLM on an
OpenAI-compatible endpoint. All pre-labels MUST be checked by a human; the
search query group (scam/legit) is only a weak hint, never a label.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.utils import Paths, get_logger, paths_from_config, read_json, video_url

log = get_logger("prelabel")

# priority order for tie-breaking a conversation's scam_type
TYPE_ORDER = ["fake_authority", "phishing_otp", "investment",
              "lottery_prize", "romance"]

REVIEW_COLUMNS = [
    # identity + content
    "id", "video_id", "url", "turn_index", "speaker", "start", "end",
    "text", "text_raw",
    # machine pre-labels
    "pre_speaker_role", "pre_is_scam", "pre_scam_type", "pre_is_after_onset",
    "label_source",
    # empty columns for the human reviewer to fill / correct
    "speaker_role", "is_scam", "scam_type", "keep", "text_corrected", "notes",
]


def load_keywords(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _hits(text: str, words: list[str]) -> int:
    return sum(1 for w in words if w and w in text)


def classify_rules(turns: list[dict], kw: dict) -> dict:
    """Return conversation-level + per-turn rule labels."""
    types = kw.get("scam_types", {})
    generic = kw.get("generic_scam", [])
    roles = kw.get("roles", {})

    type_scores = {t: 0 for t in types}
    speaker_scam_hits: dict[str, int] = {}
    onset = None
    per_turn = []
    for t in turns:
        txt = t.get("text", "")
        spk = t.get("speaker")
        t_type_hits = {ty: _hits(txt, words) for ty, words in types.items()}
        for ty, h in t_type_hits.items():
            type_scores[ty] += h
        gen = _hits(txt, generic)
        turn_has_scam = any(t_type_hits.values()) or gen > 0
        if turn_has_scam and onset is None:
            onset = t.get("turn_index", len(per_turn))
        # the speaker who voices scam content is the likely scammer
        role_bias = _hits(txt, roles.get("scammer", []))
        speaker_scam_hits[spk] = speaker_scam_hits.get(spk, 0) + \
            sum(t_type_hits.values()) + gen + role_bias
        per_turn.append({"has_scam": turn_has_scam, "type_hits": t_type_hits})

    best_type, best = None, 0
    for ty in TYPE_ORDER:
        if type_scores.get(ty, 0) > best:
            best, best_type = type_scores[ty], ty
    any_scam = best_type is not None or onset is not None
    if any_scam and best_type is None:
        best_type = "other_scam"
    conv_type = best_type if any_scam else "none"
    is_scam = 1 if any_scam else 0

    voted = {s: h for s, h in speaker_scam_hits.items() if s and h > 0}
    scammer_spk = max(voted, key=voted.get) if voted else None

    return {"is_scam": is_scam, "scam_type": conv_type, "onset": onset,
            "scammer_speaker": scammer_spk, "per_turn": per_turn}


def _role_for(speaker: str, info: dict, is_scam: int) -> str:
    if info["scammer_speaker"] is not None:
        if speaker == info["scammer_speaker"]:
            return "scammer" if is_scam else "caller_legit"
        return "victim" if is_scam else "receiver"
    return ""


def run(cfg: dict, paths: Paths, use_llm: bool = False,
        limit: int | None = None) -> pd.DataFrame:
    p = cfg.get("prelabel", {})
    kw = load_keywords(p.get("keywords_file", "labels/keywords.yaml"))

    url_map = {}
    if paths.videos_csv.exists():
        vdf = pd.read_csv(paths.videos_csv, dtype={"video_id": str})
        url_map = dict(zip(vdf["video_id"], vdf["url"]))

    clean_files = sorted(paths.clean.glob("*.json"))
    if limit:
        clean_files = clean_files[:limit]

    rows = []
    for cf in clean_files:
        data = read_json(cf)
        vid = data.get("video_id", cf.stem)
        turns = data.get("turns", [])
        if use_llm:
            info = _classify_llm(turns, cfg) or classify_rules(turns, kw)
            src = "llm"
        else:
            info = classify_rules(turns, kw)
            src = "rule"

        onset = info.get("onset")
        for t in turns:
            ti = t.get("turn_index")
            rows.append({
                "id": f"{vid}_{ti:03d}" if ti is not None else vid,
                "video_id": vid, "url": url_map.get(vid, video_url(vid)),
                "turn_index": ti, "speaker": t.get("speaker"),
                "start": t.get("start"), "end": t.get("end"),
                "text": t.get("text"), "text_raw": t.get("text_raw"),
                "pre_speaker_role": _role_for(t.get("speaker"), info, info["is_scam"]),
                "pre_is_scam": info["is_scam"],
                "pre_scam_type": info["scam_type"],
                "pre_is_after_onset": 1 if (onset is not None and ti is not None
                                            and ti >= onset) else 0,
                "label_source": src,
                "speaker_role": "", "is_scam": "", "scam_type": "",
                "keep": "", "text_corrected": "", "notes": "",
            })

    df = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    if not df.empty:
        df = df.sort_values(["video_id", "turn_index"]).reset_index(drop=True)
    out = paths.review / "review.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log.info("review.csv -> %d turns from %d videos", len(df), len(clean_files))
    return df


def _classify_llm(turns: list[dict], cfg: dict):
    """Prompt a local vLLM endpoint for {is_scam, scam_type, ...}. Best-effort."""
    import json

    try:
        from openai import OpenAI
    except Exception as exc:
        log.warning("openai client missing (%s) — falling back to rules", exc)
        return None

    llm = cfg.get("prelabel", {}).get("llm", {})
    convo = "\n".join(f"[{t.get('speaker')}] {t.get('text')}" for t in turns)
    prompt = (
        "You label Thai phone-call transcripts. Reply ONLY with JSON: "
        '{"is_scam":0|1,"scam_type":"fake_authority|romance|investment|'
        'phishing_otp|lottery_prize|other_scam|none",'
        '"scammer_speaker":"SPEAKER_xx|null","scam_onset_turn":int|null}.\n\n'
        f"Transcript:\n{convo}"
    )
    try:
        client = OpenAI(base_url=llm.get("base_url"), api_key=llm.get("api_key", "EMPTY"))
        resp = client.chat.completions.create(
            model=llm.get("model") or "local", temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(resp.choices[0].message.content)
        return {"is_scam": int(data.get("is_scam", 0)),
                "scam_type": data.get("scam_type", "none"),
                "onset": data.get("scam_onset_turn"),
                "scammer_speaker": None if data.get("scammer_speaker") in (None, "null")
                else data.get("scammer_speaker"), "per_turn": []}
    except Exception as exc:
        log.warning("llm prelabel failed (%s) — falling back to rules", exc)
        return None


if __name__ == "__main__":
    from src.utils import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg.get("log_level", "INFO"))
    run(cfg, paths_from_config(cfg))
