"""Stage 4 — speaker diarization with pyannote -> data/interim/diar/{id}.rttm

Needs HF_TOKEN in env and acceptance of the model terms on Hugging Face:
  https://hf.co/pyannote/speaker-diarization-3.1
  https://hf.co/pyannote/segmentation-3.0
Resumable: skips ids whose .rttm already exists. MPS is flaky for pyannote,
so we fall back to CPU automatically if the GPU path raises.
"""
from __future__ import annotations

import os
from pathlib import Path

from src.utils import Paths, done, get_device, get_logger, paths_from_config

log = get_logger("diarize")


def _load_pipeline(cfg: dict):
    import torch
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set (needed for pyannote). See .env.example")

    d = cfg.get("diarize", {})
    pipe = Pipeline.from_pretrained(
        d.get("model", "pyannote/speaker-diarization-3.1"),
        use_auth_token=token,
    )
    device = get_device(cfg.get("device", "auto"))
    try:
        pipe.to(torch.device(device))
    except Exception as exc:
        log.warning("could not move pipeline to %s (%s) — using cpu", device, exc)
    return pipe


def diarize_one(pipe, wav: Path, cfg: dict, out_path: Path) -> None:
    d = cfg.get("diarize", {})
    kw = {}
    if d.get("min_speakers") is not None:
        kw["min_speakers"] = d["min_speakers"]
    if d.get("max_speakers") is not None:
        kw["max_speakers"] = d["max_speakers"]

    try:
        diar = pipe(str(wav), **kw)
    except Exception as exc:  # e.g. MPS op not implemented -> retry on cpu
        import torch

        log.warning("diarize on gpu failed (%s) — retrying on cpu", exc)
        pipe.to(torch.device("cpu"))
        diar = pipe(str(wav), **kw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        diar.write_rttm(f)


def run(cfg: dict, paths: Paths, limit: int | None = None) -> None:
    wavs = sorted(paths.raw_audio.glob("*.wav"))
    if limit:
        wavs = wavs[:limit]
    pending = [w for w in wavs if not done(paths.diar / f"{w.stem}.rttm")]
    if not pending:
        log.info("nothing to diarize")
        return

    pipe = _load_pipeline(cfg)
    for wav in pending:
        vid = wav.stem
        try:
            diarize_one(pipe, wav, cfg, paths.diar / f"{vid}.rttm")
            log.info("diarized %s", vid)
        except Exception as exc:
            log.warning("diarize failed %s: %s", vid, exc)


if __name__ == "__main__":
    from src.utils import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg.get("log_level", "INFO"))
    run(cfg, paths_from_config(cfg))
