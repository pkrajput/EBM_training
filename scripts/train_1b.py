#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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
from energy_coding.data import BestFitCausalLoader, load_manifest
from energy_coding.evaluation import (
    append_metrics,
    evaluate_core_metric,
    evaluate_humaneval,
    evaluate_loss,
)
from energy_coding.modeling import (
    build_model,
    count_parameters,
    get_uncompiled_model,
    load_checkpoint,
    lr_multiplier,
    make_optimizer,
    save_checkpoint,
)


class EBTLossWrapper(torch.nn.Module):
    """Expose EBT's custom loss method as a normal DDP-compatible forward."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: dict[str, torch.Tensor], phase: str = "train") -> torch.Tensor:
        return self.model.forward_loss_wrapper(batch, phase=phase)["loss"]


@torch.no_grad()
def _ebt_pretrain_aux_snapshot(raw_model, batch) -> dict[str, float]:
    """Run an extra uncompiled forward on a tiny batch slice to grab EBT's
    pretrain-time diagnostics: `initial_loss`, `final_step_loss`, and
    `initial_final_pred_energies_gap`. This must be uncompiled and called
    sparingly (every log_interval steps) so we don't add measurable overhead.
    The "energies gap" is the EBT paper's central diagnostic that MCMC is
    actually descending the learned energy landscape."""
    try:
        slim_batch = {"input_ids": batch["input_ids"][:1]}
        with torch.enable_grad():
            metrics = raw_model.forward_loss_wrapper(slim_batch, phase="valid")
    except Exception:
        return {}
    out: dict[str, float] = {}
    for key in ("initial_loss", "final_step_loss", "perplexity", "initial_final_pred_energies_gap"):
        value = metrics.get(key)
        if isinstance(value, torch.Tensor):
            try:
                out[f"ebt/{key}"] = float(value.detach().item())
            except Exception:
                continue
        elif isinstance(value, (int, float)):
            out[f"ebt/{key}"] = float(value)
    return out


def init_distributed() -> tuple[bool, int, int, int, str]:
    ddp = int(os.environ.get("RANK", -1)) != -1
    if not ddp:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return False, 0, 0, 1, device

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=12))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    return True, rank, local_rank, world_size, device


def print0(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def precision_context(precision: str, device: str):
    if not device.startswith("cuda"):
        return nullcontext(), None
    if precision == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16), None
    if precision == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16), torch.amp.GradScaler("cuda")
    if precision == "fp32":
        return nullcontext(), None
    raise ValueError(f"Unsupported precision: {precision}")


def latest_checkpoint(out_dir: Path) -> Path | None:
    checkpoints = sorted(out_dir.glob("ckpt_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if checkpoints:
        return checkpoints[-1]
    ckpt = out_dir / "ckpt_latest.pt"
    return ckpt if ckpt.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 1B EBT on ClimbMix.")
    parser.add_argument("--config", default="configs/ebt_1b_climbmix.json")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    ddp, rank, local_rank, world_size, device = init_distributed()
    master = rank == 0
    torch.manual_seed(config.train.seed + rank)
    np.random.seed(config.train.seed + rank)
    random.seed(config.train.seed + rank)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    out_dir = Path(config.train.out_dir)
    if master:
        out_dir.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()
    wandb_run = None
    if master and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wandb_run = wandb.init(
                project=config.train.wandb_project,
                entity=config.train.wandb_entity,
                name=config.train.run_name,
                config=asdict_nested(config),
            )
            wandb_run.define_metric("step")
            wandb_run.define_metric("tokens_seen")
            wandb_run.define_metric("train_loss", step_metric="step")
            wandb_run.define_metric("train_loss_ema", step_metric="step")
            wandb_run.define_metric("val_loss", step_metric="step")
            wandb_run.define_metric("val_ppl", step_metric="step")
            wandb_run.define_metric("core_metric", step_metric="step")
            wandb_run.define_metric("humaneval_pass_at_1", step_metric="step")
            wandb_run.define_metric("tok_per_sec", step_metric="step")
        except Exception as exc:
            print0(rank, f"W&B init failed, continuing with JSONL metrics only: {exc}")

    manifest = load_manifest(config.data.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer, clean_up_tokenization_spaces=False)
    train_loader = iter(
        BestFitCausalLoader(
            tokenizer=tokenizer,
            parquet_paths=manifest["train"],
            batch_size=config.train.device_batch_size,
            sequence_length=config.data.sequence_length,
            device=device,
            rank=rank,
            world_size=world_size,
            tokenizer_batch_size=config.data.tokenizer_batch_size,
            buffer_documents=config.data.train_buffer_documents,
            repeat=True,
        )
    )

    def build_val_loader():
        return iter(
            BestFitCausalLoader(
                tokenizer=tokenizer,
                parquet_paths=manifest["val"],
                batch_size=config.train.device_batch_size,
                sequence_length=config.data.sequence_length,
                device=device,
                rank=rank,
                world_size=world_size,
                tokenizer_batch_size=config.data.tokenizer_batch_size,
                buffer_documents=config.data.eval_buffer_documents,
                repeat=True,
            )
        )

    model, hparams = build_model(config, device=device, execution_mode="pretrain")
    raw_model = model
    param_count = count_parameters(raw_model)
    global_tokens_per_step = (
        config.train.device_batch_size
        * config.data.sequence_length
        * config.train.gradient_accumulation_steps
        * world_size
    )
    if config.train.max_steps < 1:
        config.train.max_steps = math.ceil(config.train.target_tokens / global_tokens_per_step)

    print0(rank, f"Run: {config.train.run_name}")
    print0(rank, f"Device: {device} | world_size={world_size}")
    print0(rank, f"Parameters: {param_count:,} ({param_count / 1e9:.2f}B)")
    print0(rank, f"Global tokens/step: {global_tokens_per_step:,}")
    print0(rank, f"Max steps: {config.train.max_steps:,}")
    print0(rank, f"Target tokens: {global_tokens_per_step * config.train.max_steps:,}")
    print0(rank, f"Precision: {config.train.precision}")
    print0(rank, f"MCMC train steps: {config.model.mcmc_num_steps}; randomized up to {config.model.randomize_mcmc_num_steps}")

    loss_module: torch.nn.Module = EBTLossWrapper(raw_model)
    if config.train.compile_model:
        print0(rank, "Compiling model...")
        loss_module = torch.compile(loss_module, dynamic=False)

    optimizer = make_optimizer(raw_model, config.optim, model_cfg=config.model)
    autocast_context, scaler = precision_context(config.train.precision, device)

    start_step = 0
    tokens_seen = 0
    resume_path = Path(args.resume_from) if args.resume_from else None
    if resume_path is None and config.train.resume and not args.no_resume:
        resume_path = latest_checkpoint(out_dir)
    if resume_path is not None and resume_path.exists():
        start_step, tokens_seen, _ = load_checkpoint(
            resume_path,
            raw_model,
            optimizer=optimizer,
            scaler=scaler,
            strict=True,
        )
        print0(rank, f"Resumed from {resume_path} at step={start_step}, tokens={tokens_seen:,}")

    train_model = DDP(loss_module, device_ids=[local_rank]) if ddp else loss_module
    raw_for_save = raw_model
    batch = next(train_loader)
    smooth_loss = None
    t0 = time.time()
    last_metrics: dict = {}

    for step in range(start_step, config.train.max_steps):
        train_model.train()
        for group in optimizer.param_groups:
            mult = lr_multiplier(
                step=step,
                max_steps=config.train.max_steps,
                warmup_steps=config.optim.warmup_steps,
                warmdown_ratio=config.optim.warmdown_ratio,
                final_frac=config.optim.min_lr_fraction,
            )
            group["lr"] = group["initial_lr"] * mult

        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(config.train.gradient_accumulation_steps):
            if ddp:
                train_model.require_backward_grad_sync = (
                    micro_step == config.train.gradient_accumulation_steps - 1
                )
            with autocast_context:
                loss = train_model(batch, phase="train")
                scaled_loss = loss / config.train.gradient_accumulation_steps
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

        tokens_seen += global_tokens_per_step
        loss_value = float(loss.detach().item())
        smooth_loss = loss_value if smooth_loss is None else 0.9 * smooth_loss + 0.1 * loss_value

        if master and step % config.train.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_per_sec = global_tokens_per_step * config.train.log_interval / max(dt, 1e-6)
            lrs_by_group = {f"lr/{g.get('name', 'g')}": g["lr"] for g in optimizer.param_groups}
            try:
                alpha_value = float(raw_model.alpha.detach().item())
            except Exception:
                alpha_value = float("nan")
            aux = _ebt_pretrain_aux_snapshot(raw_model, batch)
            metrics = {
                "step": step,
                "tokens_seen": tokens_seen,
                "train_loss": loss_value,
                "train_loss_ema": smooth_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "tok_per_sec": tok_per_sec,
                "ebt/alpha_step_size": alpha_value,
                **lrs_by_group,
                **aux,
            }
            append_metrics(out_dir / "train_metrics.jsonl", metrics)
            if wandb_run is not None:
                wandb_run.log(metrics)
            print(
                f"step {step:07d} | loss {loss_value:.4f} | ema {smooth_loss:.4f} | "
                f"lr {metrics['lr']:.2e} | alpha {alpha_value:.3f} | "
                f"tok/s {tok_per_sec:,.0f} | tokens {tokens_seen:,}",
                flush=True,
            )

        should_eval = (
            config.train.eval_interval > 0
            and step > 0
            and step % config.train.eval_interval == 0
        )
        should_save = (
            config.train.save_interval > 0
            and step > 0
            and step % config.train.save_interval == 0
        )
        should_core = (
            config.train.core_eval_interval > 0
            and step > 0
            and step % config.train.core_eval_interval == 0
        )
        should_humaneval = (
            config.train.humaneval_interval > 0
            and step > 0
            and step % config.train.humaneval_interval == 0
        )

        if should_eval:
            val_metrics = evaluate_loss(raw_model, build_val_loader(), config.train.eval_steps, autocast_context)
            if master:
                last_metrics.update(val_metrics)
                payload = {"step": step, "tokens_seen": tokens_seen, **val_metrics}
                append_metrics(out_dir / "eval_metrics.jsonl", payload)
                if wandb_run is not None:
                    wandb_run.log(payload)
                print0(rank, f"eval step {step}: {val_metrics}")

        if master and should_save:
            config_dict = asdict_nested(config)
            save_checkpoint(
                out_dir / "ckpt_latest.pt",
                raw_for_save,
                optimizer,
                scaler,
                step,
                tokens_seen,
                config_dict,
                last_metrics,
            )
            save_checkpoint(
                out_dir / f"ckpt_step_{step}.pt",
                raw_for_save,
                optimizer,
                scaler,
                step,
                tokens_seen,
                config_dict,
                last_metrics,
            )
            print(f"saved checkpoint at step {step}", flush=True)

        core_metrics = None
        if should_core:
            core_metrics = evaluate_core_metric(raw_model, config, device)
            if master:
                last_metrics.update(core_metrics)
                payload = {"step": step, "tokens_seen": tokens_seen, **core_metrics}
                append_metrics(out_dir / "eval_metrics.jsonl", payload)
                if wandb_run is not None:
                    wandb_payload = {
                        k: v for k, v in payload.items() if not isinstance(v, dict)
                    }
                    for task_name, task_value in core_metrics.get("core_centered_results", {}).items():
                        wandb_payload[f"core/{task_name}"] = task_value
                    for task_name, task_value in core_metrics.get("core_raw_results", {}).items():
                        wandb_payload[f"core_raw/{task_name}"] = task_value
                    wandb_run.log(wandb_payload)
                print(f"CORE eval step {step}: {core_metrics['core_metric']:.4f}", flush=True)

        if master and should_humaneval:
            code_metrics = evaluate_humaneval(raw_model, config, device)
            last_metrics.update(code_metrics)
            payload = {"step": step, "tokens_seen": tokens_seen, **code_metrics}
            append_metrics(out_dir / "eval_metrics.jsonl", payload)
            if wandb_run is not None:
                wandb_run.log(payload)
            print(f"HumanEval step {step}: pass@1={code_metrics['humaneval_pass_at_1']:.4f}", flush=True)

        if ddp and (should_eval or should_save or should_core or should_humaneval):
            dist.barrier()

        stop_now = False
        if master and last_metrics:
            core_ok = last_metrics.get("core_metric", -1.0) >= config.train.early_stop_core
            if config.train.require_humaneval_for_stop:
                humaneval_ok = (
                    last_metrics.get("humaneval_pass_at_1", -1.0) >= config.train.early_stop_humaneval
                )
            else:
                humaneval_ok = True
            enough_tokens = tokens_seen >= config.train.minimum_tokens_before_stop
            stop_now = bool(core_ok and humaneval_ok and enough_tokens)
            if stop_now:
                print(
                    "Early stop target reached: "
                    f"CORE={last_metrics.get('core_metric')}, "
                    f"HumanEval={last_metrics.get('humaneval_pass_at_1', 'tracked-only')}, "
                    f"tokens={tokens_seen:,}",
                    flush=True,
                )

        if ddp:
            flag = torch.tensor(int(stop_now), device=device)
            dist.broadcast(flag, src=0)
            stop_now = bool(flag.item())
            dist.barrier()
        if stop_now:
            break

    if master:
        save_checkpoint(
            out_dir / "ckpt_final.pt",
            raw_for_save,
            optimizer,
            scaler,
            step,
            tokens_seen,
            asdict_nested(config),
            last_metrics,
        )
    if ddp:
        dist.destroy_process_group()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
