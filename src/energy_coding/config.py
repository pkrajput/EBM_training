from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    dataset_name: str = "climbmix"
    data_dir: str = "data/climbmix"
    num_train_shards: int = 170
    tokenizer: str = "EleutherAI/gpt-neox-20b"
    sequence_length: int = 2048
    validation_split_name: str = "val"
    test_split_name: str = "test"
    train_buffer_documents: int = 2048
    eval_buffer_documents: int = 512
    tokenizer_batch_size: int = 128


@dataclass
class ModelConfig:
    model_name: str = "ebt"
    model_size_label: str = "1b"
    num_transformer_blocks: int = 28
    multiheaded_attention_heads: int = 32
    embedding_dim: int = 2048
    ffn_dim_multiplier: float | None = None
    ebt_type: str = "time_embed"
    ebt_norm: str = "rms"
    ebt_act_func: str = "silu"
    dyt_alpha_init: float = 0.5
    weight_initialization_method: str = "xavier"
    weight_initialization_gain: float = 1.0
    denoising_initial_condition: str = "random_noise"
    gaussian_random_noise_scaling: float = 1.0
    normalize_initial_condition: bool = True
    normalize_initial_condition_only_first_step: bool = False
    vocab_to_embed_uses_prob_dist: bool = False
    num_modality_processing_mlp_layers: int = 1
    learnable_process_memory: bool = False
    process_memory_type: str | None = None
    process_memory_linear_layer: bool = False
    # MCMC / energy landscape regularization. Defaults follow the paper's
    # S2 (System 2 capable) recipe from EBT/job_scripts/nlp/pretrain/ebt_s2.sh:
    #   - mcmc_step_size 0.5 with learnable=True (paired with scale_alpha_with_energy)
    #   - randomize_mcmc_num_steps 3 (min 2) drives the path-randomization that the
    #     paper's Table 2 shows is critical for thinking gains
    #   - norm_pred + scale_alpha_with_energy stabilize the optimizer at 1B scale
    mcmc_num_steps: int = 1
    mcmc_step_size: float = 0.5
    mcmc_step_size_learnable: bool = True
    mcmc_step_size_lr_multiplier: float = 1.5
    langevin_dynamics_noise: float = 3.0
    langevin_dynamics_noise_learnable: bool = False
    randomize_mcmc_step_size_scale: float = 2.0
    randomize_mcmc_num_steps: int = 3
    randomize_mcmc_num_steps_min: int = 2
    randomize_mcmc_num_steps_final_landscape: bool = False
    mcmc_replay_buffer: bool = True
    mcmc_replay_buffer_size: int = 48
    mcmc_replay_buffer_sample_bs_percent: float = 0.25
    truncate_mcmc: bool = True
    no_mcmc_detach: bool = True
    clamp_futures_grad: bool = True
    clamp_futures_grad_max_change: float = 9.0
    absolute_clamp: float = 0.0
    clamp_max_after_warm_up: float = 0.0
    sharpen_predicted_distribution: float = 0.0
    contrastive_loss: bool = False
    contrastive_loss_coeff: float = 0.0005
    discrete_contrastive_loss_true_logit_val: float = 0.0
    soften_target_prob_dist: float = 0.0
    reconstruction_coeff: float = 1.0
    norm_pred: bool = True
    norm_pred_not_final_step: bool = False
    scale_alpha_with_energy: bool = True
    scale_alpha_with_energy_temp: float = 1.0


@dataclass
class OptimConfig:
    learning_rate: float = 2.0e-4
    min_lr_fraction: float = 0.05
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 10_000
    warmdown_ratio: float = 0.65
    embedding_lr_scale: float = 0.5
    head_lr_scale: float = 1.0


@dataclass
class TrainConfig:
    run_name: str = "ebt-1b-climbmix"
    out_dir: str = "runs/ebt_1b_climbmix"
    precision: str = "bf16"
    compile_model: bool = True
    seed: int = 1337
    device_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    target_tokens: int = 25_000_000_000
    max_steps: int = -1
    log_interval: int = 10
    eval_interval: int = 1000
    save_interval: int = 1000
    eval_steps: int = 64
    core_eval_interval: int = 5000
    core_max_per_task: int = 200
    eval_mcmc_depths: list[int] = field(default_factory=lambda: [1, 2])
    humaneval_interval: int = 5000
    humaneval_max_problems: int = 164
    humaneval_max_new_tokens: int = 256
    humaneval_temperature: float = 0.2
    humaneval_top_p: float = 0.95
    humaneval_top_k: int = 50
    humaneval_num_samples: int = 1
    humaneval_self_verify_samples: int = 1
    early_stop_core: float = 0.257
    early_stop_humaneval: float = 0.0
    require_humaneval_for_stop: bool = False
    minimum_tokens_before_stop: int = 8_000_000_000
    wandb_project: str = "energy-coding-pretrain"
    wandb_entity: str | None = None
    resume: bool = True


@dataclass
class PostTrainConfig:
    data_dir: str = "data/sft"
    out_dir: str = "runs/ebt_1b_sft"
    run_name: str = "ebt-1b-sft"
    wandb_project: str = "energy-coding-sft"
    max_train_examples: int = 750_000
    max_val_examples: int = 25_000
    max_test_examples: int = 25_000
    val_fraction: float = 0.02
    test_fraction: float = 0.02
    max_steps: int = 20_000
    learning_rate: float = 5.0e-5
    min_lr_fraction: float = 0.0
    weight_decay: float = 0.0
    warmup_steps: int = 500
    warmdown_ratio: float = 0.5
    device_batch_size: int = 2
    gradient_accumulation_steps: int = 32
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 500
    eval_steps: int = 64
    precision: str = "bf16"
    compile_model: bool = True
    resume: bool = True


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    post_train: PostTrainConfig = field(default_factory=PostTrainConfig)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__"):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> RunConfig:
    config = RunConfig()
    with open(path, "r", encoding="utf-8") as f:
        values = json.load(f)
    return _merge_dataclass(config, values)


def asdict_nested(config: RunConfig) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if hasattr(value, "__dataclass_fields__"):
            return {key: convert(getattr(value, key)) for key in value.__dataclass_fields__}
        return value

    return convert(config)
