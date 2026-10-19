"""Stage 1 — find candidate clips -> data/meta/videos.csv

Two sources:
- **YouTube**: keyword search via yt-dlp `ytsearchN:<query>`.
- **TikTok**: yt-dlp has NO keyword search, so clips come from
  `search.tiktok_sources` (hashtag `/tag/...`, user `/@...`, or explicit video
  URLs, flat-expanded) and/or tiktok URLs in `manual_urls.txt`. TikTok on cloud
  IPs (Colab) usually needs `youtube.cookiefile` set.

Per-source caps come from `search.targets` (e.g. youtube: 20, tiktok: 20).
Resumable: existing rows are kept; only new video_ids are appended.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yt_dlp

from src.utils import (
    Paths, cookiefile, extract_video_id, get_logger, paths_from_config,
    video_url,
)

log = get_logger("search")

COLUMNS = [
    "video_id", "url", "title", "channel", "duration_s", "upload_date",
    "query", "query_group", "source", "status", "has_manual_subs",
    "has_auto_subs",
]


def _flat_opts(cookie: str | None) -> dict:
    o = {"quiet": True, "no_warnings": True, "extract_flat": True,
         "skip_download": True}
    if cookie:
        o["cookiefile"] = cookie
    return o


def _source_of(url: str) -> str:
    return "tiktok" if "tiktok.com" in (url or "") else "youtube"


def _entry_row(e: dict) -> dict:
    return {
        "video_id": e.get("id"),
        "url": e.get("url") or e.get("webpage_url") or "",
        "title": e.get("title") or "",
        "channel": e.get("channel") or e.get("uploader") or "",
        "duration_s": e.get("duration"),
        "upload_date": e.get("upload_date"),
    }


def _extract(spec: str, cookie: str | None) -> list[dict]:
    """Flat-extract a search spec or page URL into candidate rows."""
    with yt_dlp.YoutubeDL(_flat_opts(cookie)) as ydl:
        info = ydl.extract_info(spec, download=False)
    entries = info.get("entries") if isinstance(info, dict) else None
    if entries is None and info:
        entries = [info]
    return [_entry_row(e) for e in (entries or []) if e and e.get("id")]


def _within_duration(dur, lo: int, hi: int) -> bool:
    if dur is None:  # flat results often omit duration — keep, filter later
        return True
    return lo <= float(dur) <= hi


def _dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        vid = r.get("video_id")
        if vid and vid not in seen:
            seen.add(vid)
            out.append(r)
    return out


def run(cfg: dict, paths: Paths, max_per_query: int | None = None,
        limit: int | None = None) -> pd.DataFrame:
    s = cfg.get("search", {})
    n = max_per_query or s.get("max_per_query", 50)
    lo, hi = s.get("duration_min_s", 30), s.get("duration_max_s", 1800)
    targets = s.get("targets", {}) or {}
    yt_target, tk_target = targets.get("youtube"), targets.get("tiktok")
    cookie = cookiefile(cfg)

    yt_rows: list[dict] = []
    tk_rows: list[dict] = []

    # ---- YouTube: keyword search ----
    for group, queries in (s.get("queries") or {}).items():
        for q in queries:
            log.info("youtube search [%s] %s (n=%d)", group, q, n)
            try:
                hits = _extract(f"ytsearch{n}:{q}", cookie)
            except Exception as exc:
                log.warning("yt search failed %r: %s", q, exc)
                continue
            for r in hits:
                if not _within_duration(r["duration_s"], lo, hi):
                    continue
                r.update(query=q, query_group=group, source="youtube",
                         url=video_url(r["video_id"]))
                yt_rows.append(r)

    # ---- TikTok: hashtag / user / video pages ----
    for url in (s.get("tiktok_sources") or []):
        log.info("tiktok source %s", url)
        try:
            hits = _extract(url, cookie)
        except Exception as exc:
            log.warning("tiktok source failed (%s). TikTok usually needs login "
                        "— set youtube.cookiefile (cookies.txt), or paste tiktok "
                        "video URLs into manual_urls.txt. err: %s", url, exc)
            continue
        for r in hits:
            if not r["url"]:
                continue  # can't download without a resolvable URL
            r.update(query=url, query_group="scam", source="tiktok")
            tk_rows.append(r)

    # ---- manual URLs (any platform), one per line ----
    mf = Path(s.get("manual_urls_file", "data/meta/manual_urls.txt"))
    if mf.exists():
        for line in mf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if _source_of(line) == "youtube":
                vid = extract_video_id(line)
                if not vid:
                    log.warning("manual: cannot parse youtube id from %r", line)
                    continue
                yt_rows.append({"video_id": vid, "url": video_url(vid),
                                "title": "", "channel": "", "duration_s": None,
                                "upload_date": None, "query": "manual",
                                "query_group": "manual", "source": "youtube"})
            else:
                try:
                    hits = _extract(line, cookie)
                except Exception as exc:
                    log.warning("manual tiktok failed %s: %s", line, exc)
                    continue
                for r in hits:
                    r.update(query="manual", query_group="manual", source="tiktok")
                    if r["url"]:
                        tk_rows.append(r)

    yt_rows = _dedupe(yt_rows)[:yt_target] if yt_target else _dedupe(yt_rows)
    tk_rows = _dedupe(tk_rows)[:tk_target] if tk_target else _dedupe(tk_rows)
    log.info("collected: %d youtube, %d tiktok", len(yt_rows), len(tk_rows))

    new = pd.DataFrame(yt_rows + tk_rows)
    if not new.empty:
        new["status"] = "new"
        new["has_manual_subs"] = pd.NA
        new["has_auto_subs"] = pd.NA
        new = new.drop_duplicates(subset="video_id", keep="first")

    if paths.videos_csv.exists():
        old = pd.read_csv(paths.videos_csv, dtype={"video_id": str})
        merged = pd.concat([old, new], ignore_index=True) \
                   .drop_duplicates(subset="video_id", keep="first")
    else:
        merged = new

    for c in COLUMNS:
        if c not in merged.columns:
            merged[c] = pd.NA
    merged = merged[COLUMNS]
    if limit:
        merged = merged.head(limit)

    paths.videos_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(paths.videos_csv, index=False)
    log.info("videos.csv -> %d rows (%d new this run)", len(merged), len(new))
    return merged


if __name__ == "__main__":
    from src.utils import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg.get("log_level", "INFO"))
    run(cfg, paths_from_config(cfg))
