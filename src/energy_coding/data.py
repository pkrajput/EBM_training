from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Iterator

import pyarrow.parquet as pq
import requests
import torch


CLIMBMIX_BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
CLIMBMIX_MAX_SHARD = 6542


def shard_name(index: int) -> str:
    return f"shard_{index:05d}.parquet"


@dataclass(frozen=True)
class DownloadJob:
    index: int
    data_dir: str
    base_url: str = CLIMBMIX_BASE_URL


def _download_one(job: DownloadJob) -> bool:
    data_dir = Path(job.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = shard_name(job.index)
    output_path = data_dir / filename
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"skip {filename}: already present", flush=True)
        return True

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    url = f"{job.base_url}/{filename}"
    for attempt in range(1, 6):
        try:
            print(f"download {filename} (attempt {attempt}/5)", flush=True)
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, output_path)
            return True
        except Exception as exc:
            print(f"download failed for {filename}: {exc}", flush=True)
            if tmp_path.exists():
                tmp_path.unlink()
            time.sleep(min(60, 2**attempt))
    return False


def prepare_climbmix(
    data_dir: str | Path,
    num_train_shards: int,
    num_workers: int,
    val_shard: int = CLIMBMIX_MAX_SHARD,
    test_shard: int = CLIMBMIX_MAX_SHARD - 1,
) -> Path:
    if num_train_shards < 1:
        raise ValueError("num_train_shards must be >= 1")
    if num_train_shards >= test_shard:
        raise ValueError(
            f"num_train_shards={num_train_shards} would collide with held-out test shard {test_shard}"
        )

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_indices = list(range(num_train_shards))
    all_indices = train_indices + [test_shard, val_shard]
    jobs = [DownloadJob(index=i, data_dir=str(data_dir)) for i in all_indices]
    with Pool(processes=num_workers) as pool:
        results = pool.map(_download_one, jobs)
    if not all(results):
        failed = [idx for idx, ok in zip(all_indices, results) if not ok]
        raise RuntimeError(f"Failed to download shards: {failed}")

    manifest = {
        "dataset": "karpathy/climbmix-400b-shuffle",
        "train": [str(data_dir / shard_name(i)) for i in train_indices],
        "val": [str(data_dir / shard_name(val_shard))],
        "test": [str(data_dir / shard_name(test_shard))],
        "num_train_shards": num_train_shards,
        "val_shard": val_shard,
        "test_shard": test_shard,
    }
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def load_manifest(data_dir: str | Path) -> dict[str, list[str]]:
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run `python scripts/prepare_data.py` first."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


class ParquetDocumentStream:
    def __init__(
        self,
        parquet_paths: list[str],
        rank: int,
        world_size: int,
        tokenizer_batch_size: int,
        repeat: bool,
    ) -> None:
        if not parquet_paths:
            raise ValueError("parquet_paths cannot be empty")
        self.parquet_paths = parquet_paths
        self.rank = rank
        self.world_size = world_size
        self.tokenizer_batch_size = tokenizer_batch_size
        self.repeat = repeat

    def __iter__(self) -> Iterator[list[str]]:
        while True:
            yielded = False
            for path in self.parquet_paths:
                parquet = pq.ParquetFile(path)
                for row_group_idx in range(self.rank, parquet.num_row_groups, self.world_size):
                    row_group = parquet.read_row_group(row_group_idx, columns=["text"])
                    docs = row_group.column("text").to_pylist()
                    for offset in range(0, len(docs), self.tokenizer_batch_size):
                        batch = [doc for doc in docs[offset : offset + self.tokenizer_batch_size] if doc]
                        if batch:
                            yielded = True
                            yield batch
            if not self.repeat or not yielded:
                break


class BestFitCausalLoader:
    """BOS-aligned best-fit packer for EBT's `[B, 1, T+1]` training format."""

    def __init__(
        self,
        tokenizer,
        parquet_paths: list[str],
        batch_size: int,
        sequence_length: int,
        device: str,
        rank: int = 0,
        world_size: int = 1,
        tokenizer_batch_size: int = 128,
        buffer_documents: int = 2048,
        repeat: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.row_capacity = sequence_length + 1
        self.device = device
        self.buffer_documents = buffer_documents
        self.document_batches = iter(
            ParquetDocumentStream(
                parquet_paths=parquet_paths,
                rank=rank,
                world_size=world_size,
                tokenizer_batch_size=tokenizer_batch_size,
                repeat=repeat,
            )
        )
        self.doc_buffer: list[list[int]] = []
        self.bos_token_id = self._special_token("bos_token_id")
        self.eos_token_id = self._special_token("eos_token_id")

    def _special_token(self, name: str) -> int:
        token_id = getattr(self.tokenizer, name, None)
        if token_id is not None:
            return int(token_id)
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            return int(self.tokenizer.eos_token_id)
        return 0

    def _refill(self) -> None:
        docs = next(self.document_batches)
        encoded = self.tokenizer(
            docs,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        for ids in encoded:
            if not ids:
                continue
            tokens = [self.bos_token_id] + ids + [self.eos_token_id]
            while len(tokens) > self.row_capacity:
                self.doc_buffer.append(tokens[: self.row_capacity])
                tokens = [self.bos_token_id] + tokens[self.row_capacity :]
            if len(tokens) > 1:
                self.doc_buffer.append(tokens)

    def __iter__(self) -> "BestFitCausalLoader":
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        rows = torch.empty((self.batch_size, self.row_capacity), dtype=torch.long)
        for row_idx in range(self.batch_size):
            pos = 0
            while pos < self.row_capacity:
                while len(self.doc_buffer) < self.buffer_documents:
                    self._refill()

                remaining = self.row_capacity - pos
                best_idx = -1
                best_len = 0
                for idx, doc in enumerate(self.doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = idx
                        best_len = doc_len

                if best_idx >= 0:
                    doc = self.doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    rows[row_idx, pos : pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    shortest_idx = min(range(len(self.doc_buffer)), key=lambda i: len(self.doc_buffer[i]))
                    doc = self.doc_buffer.pop(shortest_idx)
                    rows[row_idx, pos : pos + remaining] = torch.tensor(
                        doc[:remaining], dtype=torch.long
                    )
                    pos += remaining

        if self.device.startswith("cuda"):
            rows = rows.pin_memory().to(self.device, non_blocking=True)
        else:
            rows = rows.to(self.device)
        return {"input_ids": rows.unsqueeze(1)}


def render_conversation(messages: list[dict[str, str]]) -> str:
    """Render a multi-turn conversation into the format EBT expects in finetune
    mode. EBT's `mask_q_tokens` (model/model_utils.py:399) masks loss on every
    token *before* the first occurrence of the literal string `"[[Answer]]:"`,
    so the convention here is to put exactly one such marker right before the
    first assistant turn. Later turns (user follow-ups, more assistant turns)
    are included after the marker — loss IS computed on them, which mirrors how
    nanochat trains on the entire post-first-assistant span.
    """
    user_buffer: list[str] = []
    rendered_blocks: list[str] = []
    seen_first_assistant = False
    system_prefix = ""

    for message in messages:
        role = (message.get("role") or "user").strip().lower()
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if role in {"assistant", "model", "gpt"}:
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            role = "user"

        if role == "system" and not rendered_blocks and not user_buffer and not seen_first_assistant:
            system_prefix = f"System: {content}\n\n"
            continue

        if not seen_first_assistant:
            if role == "assistant":
                user_text = "\n\n".join(user_buffer).strip()
                user_buffer = []
                if user_text:
                    rendered_blocks.append(f"User: {user_text}")
                rendered_blocks.append(f"[[Answer]]: {content}")
                seen_first_assistant = True
            else:
                user_buffer.append(content)
        else:
            if role == "user":
                rendered_blocks.append(f"\n\nUser: {content}")
            else:
                rendered_blocks.append(f"\n\nAssistant: {content}")

    if not seen_first_assistant:
        return ""
    return (system_prefix + "\n".join(rendered_blocks)).strip()


# --- per-source normalizers ---------------------------------------------------


def _normalize_smoltalk(record: dict) -> str | None:
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        return render_conversation(messages) or None
    conversations = record.get("conversations")
    if isinstance(conversations, list) and conversations:
        converted = [
            {
                "role": item.get("from") or item.get("role") or "user",
                "content": item.get("value") or item.get("content") or "",
            }
            for item in conversations
        ]
        return render_conversation(converted) or None
    return None


def _normalize_mmlu_aux(record: dict) -> str | None:
    """Format MMLU auxiliary_train as a multiple choice teaching example."""
    question = (record.get("question") or "").strip()
    choices = record.get("choices") or []
    answer = record.get("answer")
    if not question or not isinstance(choices, list) or not choices:
        return None
    if answer is None:
        return None
    try:
        gold_idx = int(answer)
    except (TypeError, ValueError):
        return None
    if not 0 <= gold_idx < len(choices):
        return None
    letters = ["A", "B", "C", "D", "E", "F"][: len(choices)]
    rendered_choices = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    user = f"{question}\n\n{rendered_choices}\n\nAnswer with a single letter."
    assistant = f"{letters[gold_idx]}"
    return render_conversation(
        [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    )


def _normalize_gsm8k(record: dict) -> str | None:
    question = (record.get("question") or "").strip()
    answer = (record.get("answer") or "").strip()
    if not question or not answer:
        return None
    return render_conversation(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )


def _normalize_codealpaca(record: dict) -> str | None:
    instruction = (record.get("instruction") or "").strip()
    input_text = (record.get("input") or "").strip()
    output = (record.get("output") or "").strip()
    if not instruction or not output:
        return None
    user = instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"
    return render_conversation(
        [{"role": "user", "content": user}, {"role": "assistant", "content": output}]
    )


def _normalize_mbpp(record: dict) -> str | None:
    problem = (record.get("text") or record.get("prompt") or "").strip()
    code = (record.get("code") or "").strip()
    tests = record.get("test_list") or []
    if not problem or not code:
        return None
    extra = ""
    if isinstance(tests, list) and tests:
        extra = "\n\nExample assertions:\n" + "\n".join(tests[:3])
    return render_conversation(
        [
            {"role": "user", "content": problem + extra},
            {"role": "assistant", "content": f"```python\n{code}\n```"},
        ]
    )


def _normalize_open_codereasoning(record: dict) -> str | None:
    question = (record.get("input") or record.get("problem") or record.get("question") or "").strip()
    answer = (record.get("output") or record.get("solution") or "").strip()
    if not question or not answer:
        return None
    return render_conversation(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )


# --- SFT preparation entry point ---------------------------------------------


@dataclass
class SFTSource:
    name: str
    loader: str  # huggingface dataset name
    config: str | None
    split: str
    streaming: bool
    limit: int
    normalize: Callable[[dict], "str | None"]
    trust_remote_code: bool = False


def _default_sft_sources(
    max_smoltalk: int,
    max_mmlu: int,
    max_gsm8k: int,
    max_codealpaca: int,
    max_mbpp: int,
    max_open_codereasoning: int,
) -> list[SFTSource]:
    """The default SFT mixture aligns with nanochat (SmolTalk + MMLU + GSM8K) plus a
    code emphasis (CodeAlpaca + MBPP) and a sprinkle of LoopLM-style reasoning data
    from OpenCodeReasoning when accessible. Each source has its own row cap so the
    early sources never starve the later sources."""
    return [
        SFTSource(
            name="HuggingFaceTB/smoltalk",
            loader="HuggingFaceTB/smoltalk",
            config="all",
            split="train",
            streaming=True,
            limit=max_smoltalk,
            normalize=_normalize_smoltalk,
        ),
        SFTSource(
            name="cais/mmlu-aux",
            loader="cais/mmlu",
            config="auxiliary_train",
            split="train",
            streaming=False,
            limit=max_mmlu,
            normalize=_normalize_mmlu_aux,
            trust_remote_code=True,
        ),
        SFTSource(
            name="openai/gsm8k",
            loader="openai/gsm8k",
            config="main",
            split="train",
            streaming=False,
            limit=max_gsm8k,
            normalize=_normalize_gsm8k,
        ),
        SFTSource(
            name="sahil2801/CodeAlpaca-20k",
            loader="sahil2801/CodeAlpaca-20k",
            config=None,
            split="train",
            streaming=False,
            limit=max_codealpaca,
            normalize=_normalize_codealpaca,
        ),
        SFTSource(
            name="google-research-datasets/mbpp",
            loader="google-research-datasets/mbpp",
            config="full",
            split="train",
            streaming=False,
            limit=max_mbpp,
            normalize=_normalize_mbpp,
            trust_remote_code=True,
        ),
        SFTSource(
            name="nvidia/OpenCodeReasoning",
            loader="nvidia/OpenCodeReasoning",
            config="split_0",
            split="split_0",
            streaming=True,
            limit=max_open_codereasoning,
            normalize=_normalize_open_codereasoning,
        ),
    ]


def _load_dataset_safely(source: SFTSource):
    from datasets import load_dataset

    last_err: Exception | None = None
    for try_config in (source.config, None):
        try:
            return load_dataset(
                source.loader,
                try_config,
                split=source.split,
                streaming=source.streaming,
                trust_remote_code=source.trust_remote_code,
            )
        except Exception as exc:
            last_err = exc
    raise last_err  # type: ignore[misc]


def prepare_sft_jsonl(
    output_dir: str | Path,
    max_train_examples: int,
    max_val_examples: int,
    max_test_examples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 1337,
    source_caps: dict[str, int] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    caps = source_caps or {}
    sources = _default_sft_sources(
        max_smoltalk=caps.get("smoltalk", 460_000),
        max_mmlu=caps.get("mmlu", 100_000),
        max_gsm8k=caps.get("gsm8k", 8_000),
        max_codealpaca=caps.get("codealpaca", 20_000),
        max_mbpp=caps.get("mbpp", 974),
        max_open_codereasoning=caps.get("open_codereasoning", 25_000),
    )

    per_source_counts: dict[str, int] = {}
    examples: list[str] = []
    for source in sources:
        try:
            dataset = _load_dataset_safely(source)
        except Exception as exc:
            print(f"Skipping {source.name}: {exc}", flush=True)
            per_source_counts[source.name] = 0
            continue
        kept = 0
        for idx, record in enumerate(dataset):
            if kept >= source.limit:
                break
            text = source.normalize(record)
            if text:
                examples.append(text)
                kept += 1
            if len(examples) >= max_train_examples + max_val_examples + max_test_examples:
                break
        per_source_counts[source.name] = kept
        print(f"loaded {kept:,} examples from {source.name}", flush=True)
        if len(examples) >= max_train_examples + max_val_examples + max_test_examples:
            break

    if len(examples) < 100:
        raise RuntimeError(
            "Too few SFT examples prepared. Check HF dataset access (set HF_TOKEN if needed)."
        )

    rng.shuffle(examples)
    test_count = min(max_test_examples, max(1, int(len(examples) * test_fraction)))
    val_count = min(max_val_examples, max(1, int(len(examples) * val_fraction)))
    test_examples = examples[:test_count]
    val_examples = examples[test_count : test_count + val_count]
    train_examples = examples[test_count + val_count : test_count + val_count + max_train_examples]

    splits = {"train": train_examples, "val": val_examples, "test": test_examples}
    for split, split_examples in splits.items():
        with open(output_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for text in split_examples:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": "sft-mixture",
        "sources": [source.name for source in sources],
        "source_counts": per_source_counts,
        "train": str(output_dir / "train.jsonl"),
        "val": str(output_dir / "val.jsonl"),
        "test": str(output_dir / "test.jsonl"),
        "counts": {split: len(items) for split, items in splits.items()},
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


class JsonlSFTLoader:
    """Packed SFT loader. Conversations longer than `sequence_length` are
    truncated (they keep the user prefix + the start of the assistant). This is
    safe because we use EBT's "pretrain" execution_mode for SFT and train on
    the full sequence (no [[Answer]]: based masking)."""

    def __init__(
        self,
        tokenizer,
        path: str | Path,
        batch_size: int,
        sequence_length: int,
        device: str,
        rank: int = 0,
        world_size: int = 1,
        repeat: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.path = Path(path)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.row_capacity = sequence_length + 1
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.repeat = repeat
        self.bos_token_id = getattr(tokenizer, "bos_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 0
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None) or self.bos_token_id
        self._rows = self._load_rows()
        if not self._rows:
            raise RuntimeError(f"No usable rows found in {self.path}")
        self._idx = 0

    def _load_rows(self) -> list[list[int]]:
        rows = []
        truncated = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx % self.world_size != self.rank:
                    continue
                record = json.loads(line)
                text = record.get("text", "")
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if not ids:
                    continue
                tokens = [self.bos_token_id] + ids + [self.eos_token_id]
                if len(tokens) > self.row_capacity:
                    tokens = tokens[: self.row_capacity - 1] + [self.eos_token_id]
                    truncated += 1
                if len(tokens) >= 2:
                    rows.append(tokens)
        if truncated:
            print(
                f"[SFT loader, rank {self.rank}] kept={len(rows):,} truncated={truncated:,}",
                flush=True,
            )
        return rows

    def __iter__(self) -> "JsonlSFTLoader":
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        rows = torch.full(
            (self.batch_size, self.row_capacity),
            fill_value=int(self.eos_token_id),
            dtype=torch.long,
        )
        for row_idx in range(self.batch_size):
            if self._idx >= len(self._rows):
                if not self.repeat:
                    raise StopIteration
                self._idx = 0
            row = self._rows[self._idx]
            self._idx += 1
            rows[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long)
        if self.device.startswith("cuda"):
            rows = rows.pin_memory().to(self.device, non_blocking=True)
        else:
            rows = rows.to(self.device)
        return {"input_ids": rows.unsqueeze(1)}
