"""Stage 2 — download best audio -> 16 kHz mono WAV, fetch Thai subtitles.

Resumable: a video whose .wav already exists is skipped. Failures are logged
and do not stop the run. Subtitle availability is written back to videos.csv.
Raw media stays in data/raw/ (git-ignored) — never commit it.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pandas as pd
import yt_dlp
from tqdm import tqdm

from src.utils import (
    Paths, cookiefile, done, get_logger, paths_from_config, video_url,
)

log = get_logger("download")


def _ydl_opts(dl: dict, paths: Paths, cookie: str | None = None) -> dict:
    opts = {
        "format": "bestaudio/best",
        # outtmpl is just the name; `paths` sets the dir per output type so
        # audio lands in raw/audio and subs in raw/subs (no nested dirs).
        "outtmpl": "%(id)s.%(ext)s",
        "writesubtitles": dl.get("write_manual_subs", True),
        "writeautomaticsub": dl.get("write_auto_subs", True),
        "subtitleslangs": dl.get("subtitle_langs", ["th"]),
        "subtitlesformat": "vtt",
        "paths": {"home": str(paths.raw_audio), "subtitle": str(paths.raw_subs)},
        "retries": dl.get("retries", 3),
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": False,
        "noplaylist": True,
    }
    if cookie:
        opts["cookiefile"] = cookie
    return opts


def _to_wav(src: Path, dst: Path, sr: int, ch: int, ffmpeg: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-i", str(src), "-ac", str(ch), "-ar", str(sr),
           "-vn", str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def download_one(video_id: str, url: str, cfg: dict, paths: Paths) -> dict:
    dl = cfg.get("download", {})
    wav = paths.raw_audio / f"{video_id}.wav"
    result = {"status": "done", "has_manual_subs": False, "has_auto_subs": False}

    if done(wav):
        result["status"] = "exists"
        return result

    target = url or video_url(video_id)  # TikTok/YouTube URL from videos.csv
    with yt_dlp.YoutubeDL(_ydl_opts(dl, paths, cookiefile(cfg))) as ydl:
        info = ydl.extract_info(target, download=True)

    # subtitle availability (from metadata, independent of what downloaded)
    result["has_manual_subs"] = "th" in (info.get("subtitles") or {})
    result["has_auto_subs"] = "th" in (info.get("automatic_captions") or {})

    # locate the audio file yt-dlp actually wrote, then transcode
    src: Path | None = None
    reqs = info.get("requested_downloads") or []
    if reqs and reqs[0].get("filepath"):
        src = Path(reqs[0]["filepath"])
    if src is None or not src.exists():
        cands = [p for p in paths.raw_audio.glob(f"{video_id}.*")
                 if p.suffix != ".wav"]
        src = cands[0] if cands else None
    if src is None:
        raise FileNotFoundError(f"no audio file produced for {video_id}")

    _to_wav(src, wav, dl.get("audio_sample_rate", 16000),
            dl.get("audio_channels", 1), dl.get("ffmpeg_path", "ffmpeg"))
    try:
        src.unlink()  # drop the original container, keep only the wav
    except OSError:
        pass
    return result


def run(cfg: dict, paths: Paths, limit: int | None = None) -> None:
    if not paths.videos_csv.exists():
        raise FileNotFoundError("run the search stage first (no videos.csv)")

    df = pd.read_csv(paths.videos_csv, dtype={"video_id": str})
    dl = cfg.get("download", {})
    sleep_s = dl.get("sleep_s", 2.0)

    todo = df if limit is None else df.head(limit)
    log.info("download: %d video(s) to check", len(todo))
    for i, row in tqdm(todo.iterrows(), total=len(todo), desc="download", unit="vid"):
        vid = row["video_id"]
        try:
            res = download_one(vid, row.get("url") or "", cfg, paths)
            df.at[i, "status"] = res["status"]
            df.at[i, "has_manual_subs"] = res["has_manual_subs"]
            df.at[i, "has_auto_subs"] = res["has_auto_subs"]
            if res["status"] != "exists":
                log.info("downloaded %s (manual_subs=%s auto_subs=%s)", vid,
                         res["has_manual_subs"], res["has_auto_subs"])
                time.sleep(sleep_s)  # be polite only on real downloads
        except Exception as exc:
            df.at[i, "status"] = "failed"
            log.warning("download failed %s: %s", vid, exc)
        df.to_csv(paths.videos_csv, index=False)  # checkpoint each item


if __name__ == "__main__":
    from src.utils import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg.get("log_level", "INFO"))
    run(cfg, paths_from_config(cfg))
