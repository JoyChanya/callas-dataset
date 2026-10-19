"""Stage 5 — align ASR segments to diarization, merge into speaker turns.

data/interim/asr/{id}.json + data/interim/diar/{id}.rttm
    -> data/interim/turns/{id}.json  (one item per speaker turn)

The alignment helpers are pure functions (see tests/test_segment.py).
"""
from __future__ import annotations

from pathlib import Path

from src.utils import Paths, done, get_logger, paths_from_config, read_json, write_json

log = get_logger("segment")


def parse_rttm(path: Path) -> list[dict]:
    """RTTM 'SPEAKER file 1 start dur ...' -> [{start,end,speaker}]."""
    out: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 8 or p[0] != "SPEAKER":
            continue
        start, dur, spk = float(p[3]), float(p[4]), p[7]
        out.append({"start": start, "end": start + dur, "speaker": spk})
    return out


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speaker(seg: dict, diar: list[dict]) -> str:
    """Speaker whose diarization intervals overlap this segment the most."""
    start = seg.get("start") or 0.0
    end = seg.get("end")
    if end is None:
        end = start
    best, best_ov = "SPEAKER_00", 0.0
    for d in diar:
        ov = _overlap(start, end, d["start"], d["end"])
        if ov > best_ov:
            best, best_ov = d["speaker"], ov
    if best_ov == 0.0 and diar:  # zero-length / gap: nearest by midpoint
        mid = (start + end) / 2.0
        best = min(diar, key=lambda d: abs((d["start"] + d["end"]) / 2.0 - mid))["speaker"]
    return best


def merge_turns(labeled: list[dict], merge_gap_s: float, min_turn_s: float) -> list[dict]:
    """Merge consecutive same-speaker segments; drop tiny/empty turns."""
    labeled = [s for s in labeled if (s.get("text") or "").strip()]
    labeled.sort(key=lambda s: (s.get("start") or 0.0))

    turns: list[dict] = []
    for s in labeled:
        start = s.get("start") or 0.0
        end = s.get("end") if s.get("end") is not None else start
        text = s["text"].strip()
        if turns and turns[-1]["speaker"] == s["speaker"] and \
                start - turns[-1]["end"] <= merge_gap_s:
            turns[-1]["end"] = max(turns[-1]["end"], end)
            turns[-1]["text"] += " " + text
        else:
            turns.append({"speaker": s["speaker"], "start": start,
                          "end": end, "text": text})

    out = []
    for t in turns:
        if (t["end"] - t["start"]) < min_turn_s and len(t["text"]) < 2:
            continue
        t["text"] = " ".join(t["text"].split())
        out.append(t)
    for i, t in enumerate(out):
        t["turn_index"] = i
    return out


def segment_one(asr_json: dict, diar: list[dict], cfg: dict) -> list[dict]:
    seg_cfg = cfg.get("segment", {})
    asr_segs = (asr_json.get("segments") or {}).get("asr") or []
    labeled = [{**s, "speaker": assign_speaker(s, diar)} for s in asr_segs]
    return merge_turns(labeled, seg_cfg.get("merge_gap_s", 0.75),
                       seg_cfg.get("min_turn_s", 0.5))


def run(cfg: dict, paths: Paths, limit: int | None = None) -> None:
    asr_files = sorted(paths.asr.glob("*.json"))
    if limit:
        asr_files = asr_files[:limit]
    for aj in asr_files:
        vid = aj.stem
        out_path = paths.turns / f"{vid}.json"
        if done(out_path):
            continue
        rttm = paths.diar / f"{vid}.rttm"
        if rttm.exists():
            diar = parse_rttm(rttm)
        else:
            # no diarization yet (needs HF token) — single-speaker fallback so
            # the pipeline still runs end to end. Speaker roles stay unassigned.
            log.warning("no rttm for %s — single-speaker fallback "
                        "(set HF_TOKEN + run diarize for scammer/victim roles)", vid)
            diar = []
        turns = segment_one(read_json(aj), diar, cfg)
        write_json(out_path, {"video_id": vid, "turns": turns})
        log.info("segmented %s -> %d turns", vid, len(turns))


if __name__ == "__main__":
    from src.utils import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg.get("log_level", "INFO"))
    run(cfg, paths_from_config(cfg))
