#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from energy_coding.config import load_config
from energy_coding.data import prepare_sft_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare conversational/code SFT JSONL splits.")
    parser.add_argument("--config", default="configs/ebt_1b_climbmix.json")
    parser.add_argument(
        "--source-caps-json",
        default=None,
        help=(
            "Optional JSON file overriding per-source row caps. Recognized keys: "
            "smoltalk, mmlu, gsm8k, codealpaca, mbpp, open_codereasoning."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    source_caps = None
    if args.source_caps_json:
        with open(args.source_caps_json, "r", encoding="utf-8") as f:
            source_caps = json.load(f)

    manifest_path = prepare_sft_jsonl(
        output_dir=config.post_train.data_dir,
        max_train_examples=config.post_train.max_train_examples,
        max_val_examples=config.post_train.max_val_examples,
        max_test_examples=config.post_train.max_test_examples,
        val_fraction=config.post_train.val_fraction,
        test_fraction=config.post_train.test_fraction,
        seed=config.train.seed,
        source_caps=source_caps,
    )
    print(f"Prepared SFT manifest: {manifest_path}")


if __name__ == "__main__":
    main()
