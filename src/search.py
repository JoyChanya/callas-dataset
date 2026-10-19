"""Stage 1 — search YouTube for candidate clips -> data/meta/videos.csv

Uses yt-dlp flat search (`ytsearchN:<query>`); no audio downloaded here.
Resumable: existing rows are kept, only new video_ids are appended.
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
    "query", "query_group", "status", "has_manual_subs", "has_auto_subs",
]

_FLAT_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "extract_flat": True,
    "skip_download": True,
}


def _search_query(query: str, n: int, cookie: str | None = None) -> list[dict]:
    opts = dict(_FLAT_OPTS)
    if cookie:
        opts["cookiefile"] = cookie
    rows: list[dict] = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    for e in (info or {}).get("entries") or []:
        if not e:
            continue
        vid = e.get("id")
        if not vid:
            continue
        rows.append(
            {
                "video_id": vid,
                "url": e.get("url") or video_url(vid),
                "title": e.get("title") or "",
                "channel": e.get("channel") or e.get("uploader") or "",
                "duration_s": e.get("duration"),
                "upload_date": e.get("upload_date"),
            }
        )
    return rows


def _within_duration(dur, lo: int, hi: int) -> bool:
    # keep when duration unknown (flat search often omits it) — filtered later
    if dur is None:
        return True
    return lo <= float(dur) <= hi


def _load_manual(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vid = extract_video_id(line)
        if vid:
            out.append({"video_id": vid, "url": video_url(vid), "title": "",
                        "channel": "", "duration_s": None, "upload_date": None,
                        "query": "manual", "query_group": "manual"})
        else:
            log.warning("manual_urls: cannot parse id from %r", line)
    return out


def run(cfg: dict, paths: Paths, max_per_query: int | None = None,
        limit: int | None = None) -> pd.DataFrame:
    s = cfg.get("search", {})
    n = max_per_query or s.get("max_per_query", 50)
    lo = s.get("duration_min_s", 30)
    hi = s.get("duration_max_s", 1800)
    cookie = cookiefile(cfg)

    collected: list[dict] = []
    for group, queries in (s.get("queries") or {}).items():
        for q in queries:
            log.info("search [%s] %s (n=%d)", group, q, n)
            try:
                hits = _search_query(q, n, cookie)
            except Exception as exc:
                log.warning("search failed for %r: %s", q, exc)
                continue
            for r in hits:
                if not _within_duration(r["duration_s"], lo, hi):
                    continue
                r["query"] = q
                r["query_group"] = group
                collected.append(r)

    collected += _load_manual(Path(s.get("manual_urls_file", "data/meta/manual_urls.txt")))

    new = pd.DataFrame(collected)
    if not new.empty:
        new["status"] = "new"
        new["has_manual_subs"] = pd.NA
        new["has_auto_subs"] = pd.NA
        new = new.drop_duplicates(subset="video_id", keep="first")

    # merge with existing (resumable): keep existing rows/status untouched
    if paths.videos_csv.exists():
        old = pd.read_csv(paths.videos_csv, dtype={"video_id": str})
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset="video_id", keep="first")
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
