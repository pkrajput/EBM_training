#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from energy_coding.config import load_config
from energy_coding.data import prepare_climbmix


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ClimbMix shards and create train/val/test manifest.")
    parser.add_argument("--config", default="configs/ebt_1b_climbmix.json")
    parser.add_argument("--num-train-shards", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    num_train_shards = args.num_train_shards or config.data.num_train_shards
    manifest_path = prepare_climbmix(
        data_dir=Path(config.data.data_dir),
        num_train_shards=num_train_shards,
        num_workers=args.num_workers,
    )
    print(f"Prepared ClimbMix manifest: {manifest_path}")
    print(f"Train shards: {num_train_shards}")
    print("Validation shard: shard_06542.parquet")
    print("Test shard: shard_06541.parquet")


if __name__ == "__main__":
    main()
