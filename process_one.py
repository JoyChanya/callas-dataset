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
import subprocess
import tempfile
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


def _merge_diar(diar: list, gap: float, min_dur: float) -> list:
    """Merge adjacent same-speaker diarization segments into coherent turns."""
    out: list[dict] = []
    for d in sorted(diar, key=lambda x: x["start"]):
        if out and d["speaker"] == out[-1]["speaker"] and \
                d["start"] - out[-1]["end"] <= gap:
            out[-1]["end"] = max(out[-1]["end"], d["end"])
        else:
            out.append(dict(d))
    return [d for d in out if d["end"] - d["start"] >= min_dur]


def _slice(wav: Path, start: float, end: float, out: Path, ffmpeg: str) -> None:
    # -ss/-to AFTER -i => accurate cut on the input timeline
    subprocess.run([ffmpeg, "-y", "-i", str(wav), "-ss", str(start),
                    "-to", str(end), "-ac", "1", "-ar", "16000", str(out)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _transcribe_by_turns(asr, wav, diar, cfg, lang) -> list:
    """Diarize-first: one ASR pass per speaker turn (no Whisper-timestamp
    dependency, text maps cleanly to a speaker)."""
    sc = cfg.get("segment", {})
    merged = _merge_diar(diar, sc.get("merge_gap_s", 0.75), sc.get("min_turn_s", 0.5))
    ff = cfg.get("download", {}).get("ffmpeg_path", "ffmpeg")
    log.info("transcribing %d speaker turns...", len(merged))
    t0 = time.perf_counter()
    turns = []
    for i, seg in enumerate(merged):
        tmp = Path(tempfile.gettempdir()) / f"_callas_slice_{i}.wav"
        _slice(wav, seg["start"], seg["end"], tmp, ff)
        try:
            text = transcribe._asr_text(asr, str(tmp), lang)
        finally:
            tmp.unlink(missing_ok=True)
        if text:
            turns.append({"turn_index": len(turns), "speaker": seg["speaker"],
                          "start": seg["start"], "end": seg["end"], "text": text})
        if (i + 1) % 5 == 0 or i + 1 == len(merged):
            log.info("  %d/%d turns (%.0fs)", i + 1, len(merged),
                     time.perf_counter() - t0)
    log.info("transcribed %d turns in %.1fs", len(turns), time.perf_counter() - t0)
    return turns


def _after_audio(vid, wav, url, cfg, paths, do_diarize) -> dict:
    """Diarize first (reliable speaker turns), then transcribe each turn."""
    device = get_device(cfg.get("device", "auto"))
    lang = cfg.get("transcribe", {}).get("language", "th")

    # 2. diarize (who spoke when) — needs HF token
    diar = []
    if do_diarize and os.environ.get("HF_TOKEN"):
        log.info("diarizing...")
        t0 = time.perf_counter()
        rttm = paths.diar / f"{vid}.rttm"
        pipe = diarize._load_pipeline(cfg)
        diarize.diarize_one(pipe, wav, cfg, rttm)
        diar = segment.parse_rttm(rttm)
        log.info("diarized in %.1fs (%d raw segments)", time.perf_counter() - t0, len(diar))
    elif do_diarize:
        log.warning("no HF_TOKEN set -> single speaker, no caller/victim split. "
                    "Set HF_TOKEN to separate who-speaks-to-whom.")

    # 3. transcribe
    asr = transcribe._build_pipeline(cfg, device)
    if diar:
        turns = _transcribe_by_turns(asr, wav, diar, cfg, lang)
    else:
        log.info("transcribing whole clip (single speaker)...")
        t0 = time.perf_counter()
        asr_segs = transcribe._asr_segments(asr, wav, lang)
        turns = segment.segment_one({"video_id": vid, "segments": {"asr": asr_segs}}, [], cfg)
        log.info("transcribed in %.1fs", time.perf_counter() - t0)

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
