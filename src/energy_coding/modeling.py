from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from energy_coding.config import ModelConfig, OptimConfig, RunConfig
from energy_coding.paths import add_dependency_paths


def build_hparams(config: RunConfig, batch_size_per_device: int, execution_mode: str) -> SimpleNamespace:
    model = config.model
    data = config.data
    train = config.train
    optim = config.optim
    return SimpleNamespace(
        modality="NLP",
        model_name=model.model_name,
        tokenizer=data.tokenizer,
        context_length=data.sequence_length,
        num_transformer_blocks=model.num_transformer_blocks,
        multiheaded_attention_heads=model.multiheaded_attention_heads,
        embedding_dim=model.embedding_dim,
        ffn_dim_multiplier=model.ffn_dim_multiplier,
        batch_size_per_device=batch_size_per_device,
        ebt_type=model.ebt_type,
        ebt_norm=model.ebt_norm,
        ebt_act_func=model.ebt_act_func,
        dyt_alpha_init=model.dyt_alpha_init,
        weight_initialization_method=model.weight_initialization_method,
        weight_initialization_gain=model.weight_initialization_gain,
        peak_learning_rate=optim.learning_rate,
        accumulate_grad_batches=train.gradient_accumulation_steps,
        max_steps=train.max_steps,
        max_scheduling_steps=train.max_steps,
        warm_up_steps=optim.warmup_steps,
        gradient_clip_val=optim.grad_clip,
        weight_decay=optim.weight_decay,
        beta1=optim.beta1,
        beta2=optim.beta2,
        mcmc_num_steps=model.mcmc_num_steps,
        mcmc_step_size=model.mcmc_step_size,
        mcmc_step_size_learnable=model.mcmc_step_size_learnable,
        mcmc_step_size_lr_multiplier=model.mcmc_step_size_lr_multiplier,
        langevin_dynamics_noise=model.langevin_dynamics_noise,
        langevin_dynamics_noise_learnable=model.langevin_dynamics_noise_learnable,
        randomize_mcmc_step_size_scale=model.randomize_mcmc_step_size_scale,
        randomize_mcmc_num_steps=model.randomize_mcmc_num_steps,
        randomize_mcmc_num_steps_final_landscape=model.randomize_mcmc_num_steps_final_landscape,
        randomize_mcmc_num_steps_min=model.randomize_mcmc_num_steps_min,
        denoising_initial_condition=model.denoising_initial_condition,
        gaussian_random_noise_scaling=model.gaussian_random_noise_scaling,
        normalize_initial_condition=model.normalize_initial_condition,
        normalize_initial_condition_only_first_step=model.normalize_initial_condition_only_first_step,
        vocab_to_embed_uses_prob_dist=model.vocab_to_embed_uses_prob_dist,
        num_modality_processing_mlp_layers=model.num_modality_processing_mlp_layers,
        learnable_process_memory=model.learnable_process_memory,
        process_memory_type=model.process_memory_type,
        process_memory_linear_layer=model.process_memory_linear_layer,
        gpus="8",
        distributed_strategy="ddp",
        clamp_futures_grad=model.clamp_futures_grad,
        clamp_futures_grad_max_change=model.clamp_futures_grad_max_change,
        absolute_clamp=model.absolute_clamp,
        clamp_max_after_warm_up=model.clamp_max_after_warm_up,
        sharpen_predicted_distribution=model.sharpen_predicted_distribution,
        mcmc_replay_buffer=model.mcmc_replay_buffer,
        mcmc_replay_buffer_size=model.mcmc_replay_buffer_size,
        mcmc_replay_buffer_sample_bs_percent=model.mcmc_replay_buffer_sample_bs_percent,
        truncate_mcmc=model.truncate_mcmc,
        no_mcmc_detach=model.no_mcmc_detach,
        contrastive_loss=model.contrastive_loss,
        contrastive_loss_coeff=model.contrastive_loss_coeff,
        discrete_contrastive_loss_true_logit_val=model.discrete_contrastive_loss_true_logit_val,
        soften_target_prob_dist=model.soften_target_prob_dist,
        reconstruction_coeff=model.reconstruction_coeff,
        norm_pred=model.norm_pred,
        norm_pred_not_final_step=model.norm_pred_not_final_step,
        scale_alpha_with_energy=model.scale_alpha_with_energy,
        scale_alpha_with_energy_temp=model.scale_alpha_with_energy_temp,
        execution_mode=execution_mode,
        debug_unused_parameters=False,
    )


def build_model(config: RunConfig, device: str, execution_mode: str = "pretrain"):
    add_dependency_paths(require_nanochat=False)
    from model.nlp.ebt import EBT_NLP

    hparams = build_hparams(
        config=config,
        batch_size_per_device=config.train.device_batch_size,
        execution_mode=execution_mode,
    )
    model = EBT_NLP(hparams).to(device)
    return model, hparams


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def get_uncompiled_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def make_optimizer(
    model: torch.nn.Module,
    optim: OptimConfig,
    model_cfg: ModelConfig | None = None,
) -> torch.optim.Optimizer:
    """Param-group layout:

    - **alpha** (the EBT learnable MCMC step size). Dedicated group with no weight
      decay and LR scaled by `mcmc_step_size_lr_multiplier × peak_lr`, exactly
      mirroring `EBT/base_model_trainer.py::configure_optimizers_nlp`. Without
      this group, the learnable alpha gets the global LR which is far too small
      for the energy-landscape scale, breaking the System 2 training recipe.
    - matrix decay, no-decay, embedding, head: standard LLM groups.
    """
    alpha_params = []
    embedding_params = []
    head_params = []
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower = name.lower()
        if name == "alpha" or lower.endswith(".alpha"):
            alpha_params.append(param)
        elif param.ndim < 2 or "norm" in lower or lower.endswith(".bias"):
            no_decay_params.append(param)
        elif "embed" in lower or "tok" in lower or "wte" in lower:
            embedding_params.append(param)
        elif "lm_head" in lower or "output" in lower or "unembed" in lower:
            head_params.append(param)
        else:
            decay_params.append(param)

    alpha_lr_mult = (
        model_cfg.mcmc_step_size_lr_multiplier if model_cfg is not None else 1.5
    )

    groups = [
        {
            "params": decay_params,
            "lr": optim.learning_rate,
            "weight_decay": optim.weight_decay,
            "name": "matrix_decay",
        },
        {
            "params": no_decay_params,
            "lr": optim.learning_rate,
            "weight_decay": 0.0,
            "name": "no_decay",
        },
        {
            "params": embedding_params,
            "lr": optim.learning_rate * optim.embedding_lr_scale,
            "weight_decay": optim.weight_decay * 0.1,
            "name": "embedding",
        },
        {
            "params": head_params,
            "lr": optim.learning_rate * optim.head_lr_scale,
            "weight_decay": optim.weight_decay * 0.1,
            "name": "head",
        },
        {
            "params": alpha_params,
            "lr": optim.learning_rate * alpha_lr_mult,
            "weight_decay": 0.0,
            "name": "alpha_mcmc_step_size",
        },
    ]
    groups = [group for group in groups if group["params"]]
    for group in groups:
        group["initial_lr"] = group["lr"]
    return torch.optim.AdamW(groups, betas=(optim.beta1, optim.beta2))


def lr_multiplier(step: int, max_steps: int, warmup_steps: int, warmdown_ratio: float, final_frac: float) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    warmdown_steps = int(max_steps * warmdown_ratio)
    warmdown_start = max_steps - warmdown_steps
    if step < warmdown_start:
        return 1.0
    progress = (max_steps - step) / float(max(1, warmdown_steps))
    progress = max(0.0, min(1.0, progress))
    return final_frac + (1.0 - final_frac) * progress


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    tokens_seen: int,
    config_dict: dict,
    metrics: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = get_uncompiled_model(model)
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "tokens_seen": tokens_seen,
        "config": config_dict,
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    strict: bool = True,
) -> tuple[int, int, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw_model = get_uncompiled_model(model)
    raw_model.load_state_dict(checkpoint["model"], strict=strict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("step", 0)), int(checkpoint.get("tokens_seen", 0)), checkpoint.get("metrics", {})
