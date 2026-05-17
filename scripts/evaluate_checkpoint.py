#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from energy_coding.config import load_config
from energy_coding.evaluation import append_metrics, evaluate_core_metric, evaluate_humaneval
from energy_coding.modeling import build_model, load_checkpoint


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _apply_overlay(config, overlay_path: str) -> None:
    with open(overlay_path, "r", encoding="utf-8") as f:
        overlay = json.load(f)
    for section_name, section_values in overlay.items():
        section = getattr(config, section_name, None)
        if section is None:
            continue
        for key, value in section_values.items():
            if hasattr(section, key):
                setattr(section, key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an EBT checkpoint on CORE and/or HumanEval.")
    parser.add_argument("--config", default="configs/ebt_1b_climbmix.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--overlay", default=None, help="optional JSON overlay applied on top of --config")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-core", action="store_true")
    parser.add_argument("--skip-humaneval", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.overlay:
        _apply_overlay(config, args.overlay)
    device = resolve_device(args.device)
    model, _ = build_model(config, device=device, execution_mode="inference")
    step, tokens_seen, metrics = load_checkpoint(args.checkpoint, model, strict=True)
    metrics = {"step": step, "tokens_seen": tokens_seen, **metrics}

    if not args.skip_core:
        metrics.update(evaluate_core_metric(model, config, device))
    if not args.skip_humaneval:
        metrics.update(evaluate_humaneval(model, config, device))

    out_dir = Path(config.train.out_dir)
    append_metrics(out_dir / "eval_metrics.jsonl", metrics)
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(f"{key}/{sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
