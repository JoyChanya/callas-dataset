"""Stage 6 — normalize Thai text and mask PII (Thai PDPA).

data/interim/turns/{id}.json -> data/interim/clean/{id}.json

Regex masks are pure functions (tested in tests/test_clean.py). Person-name
masking uses a PyThaiNLP NER model; if it is unavailable the stage still runs
(regex masking only) and logs a warning. Organisation names are KEPT on
purpose — "ธนาคาร", "สภ.", "DHL" are strong scam signals.
"""
from __future__ import annotations

import re
from pathlib import Path

from src.utils import Paths, done, get_logger, paths_from_config, read_json, write_json

log = get_logger("clean")

# order matters: specific/long patterns first so they win
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+66|0)\d{1,2}[-\s]?\d{3}[-\s]?\d{3,4}")
ID_CARD_RE = re.compile(r"(?<!\d)\d{13}(?!\d)")
ACCOUNT_RE = re.compile(r"(?<!\d)\d[\d-]{8,13}\d(?!\d)")  # 10–12 digits, opt dashes


def mask_pii(text: str, flags: dict | None = None) -> str:
    """Replace phone / id / account / url with placeholder tokens."""
    f = flags or {}
    if f.get("mask_url", True):
        text = URL_RE.sub("[URL]", text)
    if f.get("mask_phone", True):
        text = PHONE_RE.sub("[PHONE]", text)
    if f.get("mask_id_card", True):
        text = ID_CARD_RE.sub("[ID_CARD]", text)
    if f.get("mask_account", True):
        text = ACCOUNT_RE.sub("[ACCOUNT]", text)
    return text


class _NameMasker:
    """Lazy wrapper around PyThaiNLP NER; no-op if the model can't load."""

    def __init__(self, engine: str = "thainer"):
        self.ner = None
        try:
            from pythainlp.tag import NER

            self.ner = NER(engine=engine)
        except Exception as exc:  # model/torch missing -> skip name masking
            log.warning("PyThaiNLP NER unavailable (%s) — names NOT masked", exc)

    def mask(self, text: str) -> str:
        if self.ner is None:
            return text
        try:
            tagged = self.ner.tag(text)
        except Exception:
            return text
        out, prev_person = [], False
        for tok, tag in tagged:
            if "PERSON" in tag:
                if not prev_person:
                    out.append("[NAME]")
                prev_person = True
            else:
                out.append(tok)
                prev_person = False
        return "".join(out)


def clean_text(text: str, masker: _NameMasker | None, cfg_clean: dict) -> str:
    if cfg_clean.get("normalize", True):
        try:
            from pythainlp.util import normalize

            text = normalize(text)
        except Exception:
            pass
    text = " ".join(text.split())
    if cfg_clean.get("mask_person_name", True) and masker is not None:
        text = masker.mask(text)
    text = mask_pii(text, cfg_clean)
    return " ".join(text.split())


def run(cfg: dict, paths: Paths, limit: int | None = None) -> None:
    cc = cfg.get("clean", {})
    turn_files = sorted(paths.turns.glob("*.json"))
    if limit:
        turn_files = turn_files[:limit]
    pending = [t for t in turn_files if not done(paths.clean / t.name)]
    masker = _NameMasker(cc.get("ner_engine", "thainer")) \
        if (pending and cc.get("mask_person_name", True)) else None

    for tf in turn_files:
        out_path = paths.clean / tf.name
        if done(out_path):
            continue
        data = read_json(tf)
        for t in data.get("turns", []):
            t["text_raw"] = t["text"]
            t["text"] = clean_text(t["text"], masker, cc)
        write_json(out_path, data)
        log.info("cleaned %s (%d turns)", data.get("video_id"),
                 len(data.get("turns", [])))


if __name__ == "__main__":
    from src.utils import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg.get("log_level", "INFO"))
    run(cfg, paths_from_config(cfg))
