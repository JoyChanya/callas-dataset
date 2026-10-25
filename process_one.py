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
import re
import time
from pathlib import Path

import pandas as pd

from src import clean, diarize, download, segment, transcribe
from src.prelabel import classify_rules, load_keywords
from src.utils import (done, extract_video_id, get_device, get_logger,
                       load_config, paths_from_config, setup_logging, write_json)

log = get_logger("process_one")


def _fmt(t) -> str:
    a, b = t.get("start"), t.get("end")
    if a is None:
        return ""
    return f"[{a:.1f}-{b:.1f}] " if b is not None else f"[{a:.1f}] "


def _after_audio(vid, wav, url, cfg, paths, do_diarize) -> dict:
    """Shared tail: transcribe -> diarize -> turns -> speaker labels -> hint."""
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

    # 6. scam hint + roles (keyword rules) — who is the caller/scammer vs victim
    kw = load_keywords(cfg.get("prelabel", {}).get("keywords_file", "labels/keywords.yaml"))
    info = classify_rules(turns, kw)
    scammer_spk = info.get("scammer_speaker")
    scammer_who = labels.get(scammer_spk)

    # 7. preprocess: mask PII + tag each turn with a role
    cc = cfg.get("clean", {})
    masker = clean._NameMasker(cc.get("ner_engine", "thainer")) \
        if cc.get("mask_person_name", True) else None
    for t in turns:
        t["text_clean"] = clean.clean_text(t["text"], masker, cc)
        if len(labels) >= 2 and scammer_spk:
            t["role"] = "scammer" if t["speaker"] == scammer_spk else "victim"
        else:
            t["role"] = ""  # single speaker or no guess -> leave for human

    return {"video_id": vid, "url": url, "turns": turns, "labels": labels,
            "n_speakers": len(labels), "guess": info, "scammer_who": scammer_who}


def process(url: str, cfg: dict, paths, do_diarize: bool = True,
            audio: str | None = None) -> dict:
    # 1. get a 16 kHz mono wav — from a local file (--audio) or by downloading
    if audio:
        src = Path(audio)
        if not src.exists():
            raise SystemExit(f"audio file not found: {audio}")
        vid = re.sub(r"[^A-Za-z0-9_-]", "_", src.stem)[:40] or "clip"
        wav = paths.raw_audio / f"{vid}.wav"
        if not done(wav):
            dlc = cfg.get("download", {})
            download._to_wav(src, wav, dlc.get("audio_sample_rate", 16000),
                             dlc.get("audio_channels", 1),
                             dlc.get("ffmpeg_path", "ffmpeg"))
        url = url or src.name
        log.info("using local audio: %s -> %s", src.name, wav.name)
        return _after_audio(vid, wav, url, cfg, paths, do_diarize)

    vid = extract_video_id(url) or "clip"
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
        if "Requested format is not available" in m:
            raise SystemExit(
                "\nYouTube didn't offer a downloadable audio format to this "
                "client.\nTry: (1) make sure you ran the cookies cell, "
                "(2) update yt-dlp (`pip install -U yt-dlp`), "
                "(3) try another clip.\n"
                "config.yaml youtube.player_client already tries several "
                "clients.\n")
        raise
    wav = paths.raw_audio / f"{vid}.wav"
    if not wav.exists():  # id mismatch (non-standard url) -> take newest wav
        got = sorted(paths.raw_audio.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        if not got:
            raise SystemExit("download produced no audio")
        wav = got[-1]
        vid = wav.stem
    log.info("audio ready (%.1fs): %s", time.perf_counter() - t0, wav.name)
    return _after_audio(vid, wav, url, cfg, paths, do_diarize)


def render(res: dict) -> str:
    g = res["guess"]
    roles = {t["who"]: t["role"] for t in res["turns"] if t.get("role")}
    rolemap = "   ".join(f"{who} = {r}" for who, r in roles.items())
    head = [
        f"# {res['url']}",
        f"speakers: {res['n_speakers']}   turns: {len(res['turns'])}",
        f"scam guess: is_scam={g['is_scam']} type={g['scam_type']}"
        + (f"   caller/scammer: {res['scammer_who']}" if res['scammer_who'] else ""),
        f"roles: {rolemap}" if rolemap else "roles: (needs HF token + 2 speakers)",
        "(auto guess — verify by reading)",
        "",
    ]
    body = [f"{_fmt(t)}{t['who']}"
            + (f" ({t['role']})" if t.get("role") else "")
            + f": {t['text']}" for t in res["turns"]]
    return "\n".join(head + body)


def save_dataset(res: dict, paths) -> "object":
    """Append this clip's preprocessed (masked) rows to a growing dataset.

    data/out/dataset.csv       one row per turn (text_masked, role, labels)
    data/out/conversations.csv one row per clip (turns joined)
    Re-processing a clip refreshes its rows (keyed by clip id).
    """
    out = paths.data / "out"
    out.mkdir(parents=True, exist_ok=True)
    vid, g = res["video_id"], res["guess"]

    rows = [{
        "id": f"{vid}_{t['turn_index']:03d}", "clip": res["url"],
        "turn_index": t["turn_index"], "speaker": t["who"],
        "role": t.get("role", ""), "text": t["text_clean"],
        "is_scam": g["is_scam"], "scam_type": g["scam_type"],
    } for t in res["turns"]]
    df = pd.DataFrame(rows)
    p = out / "dataset.csv"
    if p.exists():
        old = pd.read_csv(p, dtype={"id": str})
        old = old[~old["id"].str.startswith(f"{vid}_")]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(p, index=False, encoding="utf-8-sig")

    conv_text = "\n".join(f"[{t.get('role') or t['who']}] {t['text_clean']}"
                          for t in res["turns"])
    crow = {"clip": res["url"], "is_scam": g["is_scam"],
            "scam_type": g["scam_type"], "n_turns": len(res["turns"]),
            "scammer": res.get("scammer_who") or "", "text": conv_text}
    cp = out / "conversations.csv"
    if cp.exists():
        oc = pd.read_csv(cp)
        oc = oc[oc["clip"] != res["url"]]
        cdf = pd.concat([oc, pd.DataFrame([crow])], ignore_index=True)
    else:
        cdf = pd.DataFrame([crow])
    cdf.to_csv(cp, index=False, encoding="utf-8-sig")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="One clip -> speaker-labeled transcript")
    ap.add_argument("url", nargs="?", default=None, help="YouTube URL (or 11-char video id)")
    ap.add_argument("--audio", default=None,
                    help="use a local audio/video file instead of downloading "
                         "(sidesteps YouTube blocks on Colab)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu")
    ap.add_argument("--model", default=None, help="override transcribe model")
    ap.add_argument("--no-diarize", action="store_true", help="skip speaker split")
    args = ap.parse_args()
    if not args.url and not args.audio:
        ap.error("give a YouTube URL, or --audio PATH to use a local file")

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if args.model:
        cfg.setdefault("transcribe", {})["model"] = args.model
    setup_logging(cfg.get("log_level", "INFO"))
    paths = paths_from_config(cfg)

    res = process(args.url, cfg, paths, do_diarize=not args.no_diarize, audio=args.audio)

    out_dir = paths.data / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    text = render(res)
    (out_dir / f"{res['video_id']}.txt").write_text(text, encoding="utf-8-sig")
    write_json(out_dir / f"{res['video_id']}.json", res)

    ds = save_dataset(res, paths)

    print("\n" + text + "\n")
    print(f"transcript -> {out_dir / (res['video_id'] + '.txt')}")
    print(f"dataset    -> {ds}  (appends one row per turn; grows as you add clips)")


if __name__ == "__main__":
    main()
