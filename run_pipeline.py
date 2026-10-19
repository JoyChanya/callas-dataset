#!/usr/bin/env python3
"""Callas dataset pipeline — run all stages or one stage.

Examples:
    python run_pipeline.py --stage search --max-per-query 50
    python run_pipeline.py --stage download
    python run_pipeline.py --stage all --limit 5      # quick test on 5 videos
    python run_pipeline.py --stage prelabel --llm
    python run_pipeline.py --stage export
"""
from __future__ import annotations

import argparse

from src import (clean, diarize, download, export, prelabel, search, segment,
                 transcribe)
from src.utils import get_logger, load_config, paths_from_config, setup_logging

log = get_logger("pipeline")

STAGES = ["search", "download", "transcribe", "diarize", "segment", "clean",
          "prelabel", "export"]


def run_stage(name: str, cfg: dict, paths, args) -> None:
    log.info("=== stage: %s ===", name)
    if name == "search":
        search.run(cfg, paths, max_per_query=args.max_per_query, limit=args.limit)
    elif name == "download":
        download.run(cfg, paths, limit=args.limit)
    elif name == "transcribe":
        transcribe.run(cfg, paths, limit=args.limit)
    elif name == "diarize":
        diarize.run(cfg, paths, limit=args.limit)
    elif name == "segment":
        segment.run(cfg, paths, limit=args.limit)
    elif name == "clean":
        clean.run(cfg, paths, limit=args.limit)
    elif name == "prelabel":
        prelabel.run(cfg, paths, use_llm=args.llm, limit=args.limit)
    elif name == "export":
        export.run(cfg, paths)
    else:
        raise ValueError(f"unknown stage {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Callas dataset pipeline")
    ap.add_argument("--stage", required=True, choices=STAGES + ["all"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N items (quick tests)")
    ap.add_argument("--max-per-query", type=int, default=None,
                    help="override search.max_per_query")
    ap.add_argument("--llm", action="store_true",
                    help="use local vLLM for prelabel instead of keyword rules")
    ap.add_argument("--device", default=None,
                    help="override device: cuda | mps | cpu (e.g. cuda on Colab)")
    ap.add_argument("--model", default=None,
                    help="override transcribe.model (e.g. a large Thai Whisper)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if args.model:
        cfg.setdefault("transcribe", {})["model"] = args.model
    setup_logging(cfg.get("log_level", "INFO"))
    paths = paths_from_config(cfg)

    stages = STAGES if args.stage == "all" else [args.stage]
    for name in stages:
        run_stage(name, cfg, paths, args)


if __name__ == "__main__":
    main()
