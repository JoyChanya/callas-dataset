"""Stage 3 — transcribe WAV with Thonburian Whisper -> data/interim/asr/{id}.json

Also parses any YouTube .vtt into the same segment format so ASR and the
platform captions can be compared later. Resumable: skips ids already done.
Heavy deps (torch/transformers) are imported lazily.

Low-memory aware: on MPS (Apple Silicon unified memory) it uses fp16 +
batch_size 1, and if a run still hits an out-of-memory error it rebuilds the
pipeline on CPU and retries.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import (
    Paths, done, get_device, get_logger, paths_from_config, read_json,
    write_json,
)

log = get_logger("transcribe")


def _parse_vtt(path: Path) -> list[dict]:
    import webvtt

    out: list[dict] = []
    for cap in webvtt.read(str(path)):
        txt = " ".join(cap.text.split())
        if txt:
            out.append({"start": cap.start_in_seconds,
                        "end": cap.end_in_seconds, "text": txt})
    return out


def _build_pipeline(cfg: dict, device: str):
    import torch
    import transformers
    from transformers import pipeline

    t = cfg.get("transcribe", {})
    # fp16 on gpu/mps halves memory; cpu stays fp32 for numerical safety
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    # 8 GB unified memory can't hold a big batch of whisper-medium activations
    batch = 1 if device == "mps" else t.get("batch_size", 8)
    chunk = t.get("chunk_length_s") or 0
    mode = f"chunked({chunk}s)" if chunk else "sequential long-form"
    log.info("loading ASR model %s on %s (dtype=%s, %s)",
             t.get("model"), device, dtype, mode)
    kw = dict(
        model=t.get("model", "biodatlab/whisper-th-medium-combined"),
        device=device,
    )
    # chunk_length_s>0 = fast chunked (hallucination-prone on noisy audio, poor
    # timestamps). 0 = Whisper sequential long-form: accurate segment
    # timestamps + built-in temperature/compression fallback. Default 0.
    if chunk:
        kw["chunk_length_s"] = chunk
        kw["batch_size"] = batch
    if int(transformers.__version__.split(".")[0]) >= 5:
        kw["dtype"] = dtype
    else:
        kw["torch_dtype"] = dtype
    return pipeline("automatic-speech-recognition", **kw)


def _asr_segments(asr, wav: Path, language: str) -> list[dict]:
    # anti-hallucination: no_repeat_ngram kills "โอ้ โอ้ โอ้" loops; temperature
    # fallback + compression/logprob thresholds retry bad windows; not
    # conditioning on prev tokens stops error propagation across windows.
    gen = {
        "language": language, "task": "transcribe",
        "condition_on_prev_tokens": False,
        "no_repeat_ngram_size": 3,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    }
    res = asr(str(wav), return_timestamps=True, generate_kwargs=gen)
    out: list[dict] = []
    for ch in res.get("chunks", []):
        ts = ch.get("timestamp") or (None, None)
        txt = (ch.get("text") or "").strip()
        if txt:
            out.append({"start": ts[0], "end": ts[1], "text": txt})
    if not out and res.get("text"):
        out.append({"start": 0.0, "end": None, "text": res["text"].strip()})
    return out


def _is_oom(exc: Exception) -> bool:
    s = str(exc).lower()
    return "out of memory" in s or "mps backend out of memory" in s


def run(cfg: dict, paths: Paths, limit: int | None = None) -> None:
    t = cfg.get("transcribe", {})
    wavs = sorted(paths.raw_audio.glob("*.wav"))
    if limit:
        wavs = wavs[:limit]
    if not wavs:
        log.warning("no WAV files in %s — run download first", paths.raw_audio)
        return

    subs_flags = {}
    if paths.videos_csv.exists():
        vdf = pd.read_csv(paths.videos_csv, dtype={"video_id": str})
        for _, r in vdf.iterrows():
            subs_flags[r["video_id"]] = (bool(r.get("has_manual_subs")),
                                         bool(r.get("has_auto_subs")))

    pending = [w for w in wavs if not done(paths.asr / f"{w.stem}.json")]
    if not pending:
        log.info("nothing to transcribe")
        return

    device = get_device(cfg.get("device", "auto"))
    asr = _build_pipeline(cfg, device)
    lang = t.get("language", "th")

    for wav in wavs:
        vid = wav.stem
        out_path = paths.asr / f"{vid}.json"
        if done(out_path):
            continue
        segments: dict[str, list] = {}
        try:
            segments["asr"] = _asr_segments(asr, wav, lang)
        except Exception as exc:
            if device != "cpu" and _is_oom(exc):
                log.warning("%s OOM on %s — rebuilding on cpu and retrying",
                            vid, device)
                device = "cpu"
                asr = _build_pipeline(cfg, "cpu")
                try:
                    segments["asr"] = _asr_segments(asr, wav, lang)
                except Exception as e2:
                    log.warning("ASR failed %s on cpu: %s", vid, e2)
                    continue
            else:
                log.warning("ASR failed %s: %s", vid, exc)
                continue

        if t.get("also_parse_youtube_subs", True):
            vtt = paths.raw_subs / f"{vid}.th.vtt"
            if vtt.exists():
                has_manual, _ = subs_flags.get(vid, (False, False))
                key = "youtube_manual" if has_manual else "youtube_auto"
                try:
                    segments[key] = _parse_vtt(vtt)
                except Exception as exc:
                    log.warning("vtt parse failed %s: %s", vid, exc)

        write_json(out_path, {"video_id": vid, "model": t.get("model"),
                              "segments": segments})
        log.info("transcribed %s (asr=%d segs)", vid,
                 len(segments.get("asr", [])))


if __name__ == "__main__":
    from src.utils import load_config, setup_logging

    cfg = load_config()
    setup_logging(cfg.get("log_level", "INFO"))
    run(cfg, paths_from_config(cfg))
