"""Shared helpers: config, paths, logging, device, resumable IO, id parsing."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

# load .env (HF_TOKEN etc.) if present — optional
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


# ---------------- logging ----------------
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ---------------- config ----------------
def load_config(path: str | Path = "config.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------- paths ----------------
class Paths:
    """Canonical dataset directory layout, derived from one data_dir."""

    def __init__(self, data_dir: str | Path):
        self.data = Path(data_dir)
        self.meta = self.data / "meta"
        self.raw_audio = self.data / "raw" / "audio"
        self.raw_subs = self.data / "raw" / "subs"
        self.asr = self.data / "interim" / "asr"
        self.diar = self.data / "interim" / "diar"
        self.turns = self.data / "interim" / "turns"
        self.clean = self.data / "interim" / "clean"
        self.review = self.data / "review"
        self.final = self.data / "final"

    def _all(self) -> list[Path]:
        return [
            self.meta, self.raw_audio, self.raw_subs, self.asr, self.diar,
            self.turns, self.clean, self.review, self.final,
        ]

    def ensure(self) -> "Paths":
        for d in self._all():
            d.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def videos_csv(self) -> Path:
        return self.meta / "videos.csv"


def paths_from_config(cfg: dict) -> Paths:
    return Paths(cfg.get("paths", {}).get("data_dir", "data")).ensure()


# ---------------- device ----------------
def get_device(prefer: str = "auto") -> str:
    """Resolve compute device. auto -> cuda > mps > cpu."""
    if prefer and prefer != "auto":
        return prefer
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---------------- resumable IO ----------------
def done(path: str | Path) -> bool:
    """True if a stage output already exists and is non-empty (skip it)."""
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def write_json(path: str | Path, obj: Any) -> None:
    """Atomic JSON write (UTF-8, Thai preserved)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- youtube id helpers ----------------
def video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def cookiefile(cfg: dict) -> str | None:
    """Optional cookies.txt for YouTube bot-checks (common on cloud IPs)."""
    c = (cfg.get("youtube") or {}).get("cookiefile") or ""
    return c or None


_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/|/live/)([A-Za-z0-9_-]{11})")


def extract_video_id(url: str) -> str | None:
    """Pull an 11-char YouTube id from a URL, or return it if already an id."""
    url = (url or "").strip()
    if not url:
        return None
    m = _YT_ID_RE.search(url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    return None
