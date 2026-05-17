#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from energy_coding.config import asdict_nested, load_config
from energy_coding.data import JsonlSFTLoader
from energy_coding.evaluation import append_metrics, evaluate_humaneval, evaluate_loss
from energy_coding.modeling import (
    build_model,
    get_uncompiled_model,
    load_checkpoint,
    lr_multiplier,
    make_optimizer,
    save_checkpoint,
)
from train_1b import EBTLossWrapper, init_distributed, precision_context, print0


def latest_checkpoint(out_dir: Path) -> Path | None:
    checkpoints = sorted(out_dir.glob("ckpt_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if checkpoints:
        return checkpoints[-1]
    ckpt = out_dir / "ckpt_latest.pt"
    return ckpt if ckpt.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT/post-train an EBT checkpoint into a conversational model.")
    parser.add_argument("--config", default="configs/ebt_1b_climbmix.json")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config.train.device_batch_size = config.post_train.device_batch_size
    config.train.gradient_accumulation_steps = config.post_train.gradient_accumulation_steps
    config.train.max_steps = config.post_train.max_steps
    config.train.precision = config.post_train.precision
    config.train.compile_model = config.post_train.compile_model
    config.optim.learning_rate = config.post_train.learning_rate
    config.optim.weight_decay = config.post_train.weight_decay
    config.optim.warmup_steps = config.post_train.warmup_steps
    config.optim.warmdown_ratio = config.post_train.warmdown_ratio
    config.optim.min_lr_fraction = config.post_train.min_lr_fraction

    ddp, rank, local_rank, world_size, device = init_distributed()
    master = rank == 0
    torch.manual_seed(config.train.seed + rank)
    np.random.seed(config.train.seed + rank)
    random.seed(config.train.seed + rank)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    out_dir = Path(config.post_train.out_dir)
    if master:
        out_dir.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()

    wandb_run = None
    if master and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wandb_run = wandb.init(
                project=config.post_train.wandb_project,
                entity=config.train.wandb_entity,
                name=config.post_train.run_name,
                config=asdict_nested(config),
            )
            wandb_run.define_metric("step")
            wandb_run.define_metric("tokens_seen")
            wandb_run.define_metric("*", step_metric="step")
        except Exception as exc:
            print0(rank, f"W&B init failed, continuing with JSONL metrics only: {exc}")

    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer, clean_up_tokenization_spaces=False)
    train_loader = iter(
        JsonlSFTLoader(
            tokenizer=tokenizer,
            path=Path(config.post_train.data_dir) / "train.jsonl",
            batch_size=config.post_train.device_batch_size,
            sequence_length=config.data.sequence_length,
            device=device,
            rank=rank,
            world_size=world_size,
            repeat=True,
        )
    )

    def build_val_loader():
        return iter(
            JsonlSFTLoader(
                tokenizer=tokenizer,
                path=Path(config.post_train.data_dir) / "val.jsonl",
                batch_size=config.post_train.device_batch_size,
                sequence_length=config.data.sequence_length,
                device=device,
                rank=rank,
                world_size=world_size,
                repeat=True,
            )
        )

    # We intentionally pass execution_mode="pretrain" instead of "finetune".
    # Two reasons:
    #   1. EBT's `forward_loss_wrapper` references `self.tokenizer` in the
    #      finetune branch but never sets it in `__init__` (EBT bug -> AttributeError).
    #   2. Their `mask_q_tokens` fills masked positions with `tokenizer.pad_token_id`
    #      (=1 for gpt-neox) but the loss ignore_index is `tokenizer.eos_token_id`
    #      (=0). So masked user tokens would still contribute to loss.
    # In pretrain mode we train on the entire conversation (nanochat-style); since
    # our renderer makes the assistant span dominate, this is a close approximation
    # of true SFT and is what's actually proven to work in this codebase.
    model, _ = build_model(config, device=device, execution_mode="pretrain")
    raw_model = model
    start_step, tokens_seen, _ = load_checkpoint(args.base_checkpoint, raw_model, strict=True)
    print0(rank, f"Loaded base checkpoint {args.base_checkpoint} (base_step={start_step}, base_tokens={tokens_seen:,})")

    loss_module: torch.nn.Module = EBTLossWrapper(raw_model)
    if config.post_train.compile_model:
        print0(rank, "Compiling SFT loss wrapper...")
        loss_module = torch.compile(loss_module, dynamic=False)
    train_model = DDP(loss_module, device_ids=[local_rank]) if ddp else loss_module
    optimizer = make_optimizer(raw_model, config.optim, model_cfg=config.model)
    autocast_context, scaler = precision_context(config.post_train.precision, device)

    resume_path = Path(args.resume_from) if args.resume_from else None
    if resume_path is None and config.post_train.resume and not args.no_resume:
        resume_path = latest_checkpoint(out_dir)
    step0 = 0
    sft_tokens_seen = 0
    last_metrics: dict = {}
    if resume_path is not None and resume_path.exists():
        step0, sft_tokens_seen, last_metrics = load_checkpoint(
            resume_path,
            raw_model,
            optimizer=optimizer,
            scaler=scaler,
            strict=True,
        )
        print0(rank, f"Resumed SFT from {resume_path} at step={step0}")

    global_tokens_per_step = (
        config.post_train.device_batch_size
        * config.data.sequence_length
        * config.post_train.gradient_accumulation_steps
        * world_size
    )
    print0(rank, f"SFT tokens/step: {global_tokens_per_step:,}")
    batch = next(train_loader)
    smooth_loss = None
    t0 = time.time()

    for step in range(step0, config.post_train.max_steps):
        train_model.train()
        mult = lr_multiplier(
            step=step,
            max_steps=config.post_train.max_steps,
            warmup_steps=config.post_train.warmup_steps,
            warmdown_ratio=config.post_train.warmdown_ratio,
            final_frac=config.post_train.min_lr_fraction,
        )
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * mult

        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(config.post_train.gradient_accumulation_steps):
            if ddp:
                train_model.require_backward_grad_sync = (
                    micro_step == config.post_train.gradient_accumulation_steps - 1
                )
            with autocast_context:
                loss = train_model(batch, phase="train")
                scaled_loss = loss / config.post_train.gradient_accumulation_steps
            batch = next(train_loader)
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        if config.optim.grad_clip > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.optim.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        sft_tokens_seen += global_tokens_per_step
        loss_value = float(loss.detach().item())
        smooth_loss = loss_value if smooth_loss is None else 0.9 * smooth_loss + 0.1 * loss_value

        if master and step % config.post_train.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            metrics = {
                "step": step,
                "tokens_seen": sft_tokens_seen,
                "sft/train_loss": loss_value,
                "sft/train_loss_ema": smooth_loss,
                "sft/lr": optimizer.param_groups[0]["lr"],
                "sft/tok_per_sec": global_tokens_per_step * config.post_train.log_interval / max(dt, 1e-6),
            }
            append_metrics(out_dir / "train_metrics.jsonl", metrics)
            if wandb_run is not None:
                wandb_run.log(metrics)
            print0(rank, f"sft step {step:06d} | loss {loss_value:.4f} | ema {smooth_loss:.4f}")

        should_eval = (
            config.post_train.eval_interval > 0
            and step > 0
            and step % config.post_train.eval_interval == 0
        )
        should_save = (
            config.post_train.save_interval > 0
            and step > 0
            and step % config.post_train.save_interval == 0
        )

        if should_eval:
            val_metrics = evaluate_loss(raw_model, build_val_loader(), config.post_train.eval_steps, autocast_context)
            if master:
                payload = {
                    "step": step,
                    "tokens_seen": sft_tokens_seen,
                    "sft/val_loss": val_metrics["val_loss"],
                    "sft/val_ppl": val_metrics["val_ppl"],
                }
                last_metrics.update(payload)
                append_metrics(out_dir / "eval_metrics.jsonl", payload)
                if wandb_run is not None:
                    wandb_run.log(payload)
                print0(rank, f"sft eval step {step}: {payload}")

        if master and should_save:
            save_checkpoint(
                out_dir / "ckpt_latest.pt",
                raw_model,
                optimizer,
                scaler,
                step,
                sft_tokens_seen,
                asdict_nested(config),
                last_metrics,
            )
            save_checkpoint(
                out_dir / f"ckpt_step_{step}.pt",
                raw_model,
                optimizer,
                scaler,
                step,
                sft_tokens_seen,
                asdict_nested(config),
                last_metrics,
            )
            print0(rank, f"saved SFT checkpoint at step {step}")

        if ddp and (should_eval or should_save):
            dist.barrier()

    if master:
        code_metrics = evaluate_humaneval(raw_model, config, device)
        append_metrics(out_dir / "eval_metrics.jsonl", {"step": config.post_train.max_steps, **code_metrics})
        if wandb_run is not None:
            wandb_run.log({"step": config.post_train.max_steps, **code_metrics})
        save_checkpoint(
            out_dir / "ckpt_final.pt",
            raw_model,
            optimizer,
            scaler,
            config.post_train.max_steps,
            sft_tokens_seen,
            asdict_nested(config),
            {**last_metrics, **code_metrics},
        )
    if ddp:
        dist.destroy_process_group()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
