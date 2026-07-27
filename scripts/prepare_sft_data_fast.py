#!/usr/bin/env python3
"""Fast SFT data prep for bandwidth-OK / high-latency boxes.

The default `prepare_sft_data.py` streams SmolTalk row-by-row, which on a
high-latency link is dominated by per-request latency (~4 rows/s). This script
instead downloads a few SmolTalk parquet *shards* in bulk (bandwidth-bound,
~5 MB/s -> ~116k rows in ~47s per shard) and reads them locally, then pulls the
remaining (small) sources with non-streaming `load_dataset`. It reuses the exact
same normalizers as the canonical path, so the rendered chat format is identical.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from energy_coding.config import load_config
from energy_coding.data import (
    _normalize_codealpaca,
    _normalize_gsm8k,
    _normalize_mbpp,
    _normalize_mmlu_aux,
    _normalize_smoltalk,
)


def _smoltalk_from_parquet(max_rows: int, num_shards: int) -> list[str]:
    api = HfApi()
    files = [
        f
        for f in api.list_repo_files("HuggingFaceTB/smoltalk", repo_type="dataset")
        if f.startswith("data/all/train-") and f.endswith(".parquet")
    ]
    files.sort()
    out: list[str] = []
    for fname in files[:num_shards]:
        path = hf_hub_download(
            repo_id="HuggingFaceTB/smoltalk", repo_type="dataset", filename=fname
        )
        table = pq.read_table(path, columns=["messages"])
        for messages in table.column("messages").to_pylist():
            text = _normalize_smoltalk({"messages": messages})
            if text:
                out.append(text)
            if len(out) >= max_rows:
                print(f"smoltalk: {len(out):,} (stopped at shard {fname})", flush=True)
                return out
        print(f"smoltalk: {len(out):,} after {fname}", flush=True)
    return out


def _bulk(loader, config, split, normalize, limit, trust_remote_code=False):
    from datasets import load_dataset

    try:
        ds = load_dataset(
            loader,
            config,
            split=split,
            streaming=False,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        print(f"skip {loader}: {exc}", flush=True)
        return []
    out: list[str] = []
    for rec in ds:
        text = normalize(rec)
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    print(f"{loader}: {len(out):,}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ebt_191m_randmcmc_8x5090.json")
    ap.add_argument("--smoltalk", type=int, default=150_000)
    ap.add_argument("--smoltalk-shards", type=int, default=2)
    ap.add_argument("--mmlu", type=int, default=40_000)
    ap.add_argument("--gsm8k", type=int, default=8_000)
    ap.add_argument("--codealpaca", type=int, default=20_000)
    ap.add_argument("--mbpp", type=int, default=974)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.post_train.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.train.seed
    rng = random.Random(seed)

    examples: list[str] = []
    counts: dict[str, int] = {}

    st = _smoltalk_from_parquet(args.smoltalk, args.smoltalk_shards)
    counts["smoltalk"] = len(st)
    examples += st

    # cais/mmlu auxiliary_train (parquet) nests every field under a single
    # struct column named "train": {"train": {"question":.., "choices":.., ..}}.
    # Unwrap it before handing to the standard MMLU normalizer.
    def _mmlu_unwrap(rec):
        inner = rec.get("train") if isinstance(rec.get("train"), dict) else rec
        return _normalize_mmlu_aux(inner)

    for name, loader, conf, split, norm, lim, trc in [
        ("mmlu-aux", "cais/mmlu", "auxiliary_train", "train", _mmlu_unwrap, args.mmlu, False),
        ("gsm8k", "openai/gsm8k", "main", "train", _normalize_gsm8k, args.gsm8k, False),
        ("codealpaca", "sahil2801/CodeAlpaca-20k", None, "train", _normalize_codealpaca, args.codealpaca, False),
        ("mbpp", "google-research-datasets/mbpp", "full", "train", _normalize_mbpp, args.mbpp, False),
    ]:
        rows = _bulk(loader, conf, split, norm, lim, trc)
        counts[name] = len(rows)
        examples += rows

    if len(examples) < 100:
        raise RuntimeError("Too few SFT examples prepared; check HF access.")

    rng.shuffle(examples)
    n = len(examples)
    test_n = min(cfg.post_train.max_test_examples, max(1, int(n * cfg.post_train.test_fraction)))
    val_n = min(cfg.post_train.max_val_examples, max(1, int(n * cfg.post_train.val_fraction)))
    test_ex = examples[:test_n]
    val_ex = examples[test_n : test_n + val_n]
    train_ex = examples[test_n + val_n : test_n + val_n + cfg.post_train.max_train_examples]

    for split, rows in {"train": train_ex, "val": val_ex, "test": test_ex}.items():
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for text in rows:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": "sft-mixture-fast",
        "source_counts": counts,
        "train": len(train_ex),
        "val": len(val_ex),
        "test": len(test_ex),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("SFT manifest:", json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
