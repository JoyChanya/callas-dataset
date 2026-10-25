#!/usr/bin/env python3
"""Process ONE clip → speaker-labeled transcript (who speaks to whom).

Give it a single YouTube URL. It downloads the audio, transcribes it (Thai
Whisper), finds who-spoke-when (pyannote diarization), and prints the
conversation as turns:

    [12.3-15.0] Speaker A: สวัสดีค่ะ ที่นี่ไปรษณีย์ไทย ...
    [15.0-18.4] Speaker B: มีอะไรให้ช่วยครับ ...

Usage:
    python process_one.py "https://www.youtube.com/watch?v=XXXX"
    python process_one.py "<url>" --device cuda --model biodatlab/whisper-th-large-v3-combined
    python process_one.py "<url>" --no-diarize      # transcript only (1 speaker)

Outputs (data/out/):
    {id}.txt   human-readable transcript
    {id}.json  structured turns + summary
"""
from __future__ import annotations

import argparse
import os
import time

from src import diarize, download, segment, transcribe
from src.prelabel import classify_rules, load_keywords
from src.utils import (extract_video_id, get_device, get_logger, load_config,
                       paths_from_config, setup_logging, write_json)

log = get_logger("process_one")


def _fmt(t) -> str:
    a, b = t.get("start"), t.get("end")
    if a is None:
        return ""
    return f"[{a:.1f}-{b:.1f}] " if b is not None else f"[{a:.1f}] "


def process(url: str, cfg: dict, paths, do_diarize: bool = True) -> dict:
    vid = extract_video_id(url) or "clip"

    # 1. download audio -> 16 kHz mono wav
    log.info("downloading %s", url)
    t0 = time.perf_counter()
    try:
        download.download_one(vid, url, cfg, paths)
    except Exception as exc:
        m = str(exc)
        if "not a bot" in m or "Sign in to confirm" in m:
            raise SystemExit(
                "\nYouTube blocked the download with a bot-check (common on "
                "Colab IPs).\nFix: export a cookies.txt from a browser logged "
                "in to youtube.com, upload it, and set youtube.cookiefile in "
                "config.yaml.\nIn the Colab notebook, run the 'YouTube cookies' "
                "cell.\n")
        raise
    wav = paths.raw_audio / f"{vid}.wav"
    if not wav.exists():  # id mismatch (non-standard url) -> take newest wav
        got = sorted(paths.raw_audio.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        if not got:
            raise SystemExit("download produced no audio")
        wav = got[-1]
        vid = wav.stem
    log.info("audio ready (%.1fs): %s", time.perf_counter() - t0, wav.name)

    # 2. transcribe
    device = get_device(cfg.get("device", "auto"))
    asr = transcribe._build_pipeline(cfg, device)
    log.info("transcribing...")
    t0 = time.perf_counter()
    asr_segs = transcribe._asr_segments(asr, wav, cfg.get("transcribe", {}).get("language", "th"))
    log.info("transcribed in %.1fs (%d segments)", time.perf_counter() - t0, len(asr_segs))
    asr_json = {"video_id": vid, "segments": {"asr": asr_segs}}

    # 3. diarize (who spoke when) — needs HF token
    diar = []
    if do_diarize and os.environ.get("HF_TOKEN"):
        log.info("diarizing...")
        t0 = time.perf_counter()
        rttm = paths.diar / f"{vid}.rttm"
        pipe = diarize._load_pipeline(cfg)
        diarize.diarize_one(pipe, wav, cfg, rttm)
        diar = segment.parse_rttm(rttm)
        log.info("diarized in %.1fs", time.perf_counter() - t0)
    elif do_diarize:
        log.warning("no HF_TOKEN set -> skipping diarization (single speaker). "
                    "Set HF_TOKEN to separate who-speaks-to-whom.")

    # 4. merge into speaker turns
    turns = segment.segment_one(asr_json, diar, cfg)

    # 5. map pyannote SPEAKER_xx -> readable A / B / C (order of first speech)
    labels: dict[str, str] = {}
    for t in turns:
        spk = t["speaker"]
        if spk not in labels:
            labels[spk] = f"Speaker {chr(ord('A') + len(labels))}"
        t["who"] = labels[spk]

    # 6. weak scam hint (keyword rules) — who looks like the scammer
    kw = load_keywords(cfg.get("prelabel", {}).get("keywords_file", "labels/keywords.yaml"))
    info = classify_rules(turns, kw)
    scammer_who = labels.get(info.get("scammer_speaker"))

    return {"video_id": vid, "url": url, "turns": turns, "labels": labels,
            "n_speakers": len(labels), "guess": info, "scammer_who": scammer_who}


def render(res: dict) -> str:
    g = res["guess"]
    head = [
        f"# {res['url']}",
        f"speakers: {res['n_speakers']}   turns: {len(res['turns'])}",
        f"scam guess: is_scam={g['is_scam']} type={g['scam_type']}"
        + (f"   likely scammer: {res['scammer_who']}" if res['scammer_who'] else ""),
        "(labels are a weak keyword guess — verify by reading)",
        "",
    ]
    body = [f"{_fmt(t)}{t['who']}: {t['text']}" for t in res["turns"]]
    return "\n".join(head + body)


def main() -> None:
    ap = argparse.ArgumentParser(description="One clip -> speaker-labeled transcript")
    ap.add_argument("url", help="YouTube URL (or 11-char video id)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu")
    ap.add_argument("--model", default=None, help="override transcribe model")
    ap.add_argument("--no-diarize", action="store_true", help="skip speaker split")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if args.model:
        cfg.setdefault("transcribe", {})["model"] = args.model
    setup_logging(cfg.get("log_level", "INFO"))
    paths = paths_from_config(cfg)

    res = process(args.url, cfg, paths, do_diarize=not args.no_diarize)

    out_dir = paths.data / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    text = render(res)
    (out_dir / f"{res['video_id']}.txt").write_text(text, encoding="utf-8-sig")
    write_json(out_dir / f"{res['video_id']}.json", res)

    print("\n" + text + "\n")
    print(f"saved -> {out_dir / (res['video_id'] + '.txt')}")


if __name__ == "__main__":
    main()
