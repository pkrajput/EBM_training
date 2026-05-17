# energy-coding

Train-ready wrapper for a 1B Energy-Based Transformer (EBT) language model.

We pre-train an EBT on NVIDIA's ClimbMix dataset (the same shuffled-by-Karpathy
mirror that nanochat uses), gate early-stop on CORE, and track HumanEval pass@1
as a diagnostic. Once CORE crosses the GPT-2 threshold we hand the checkpoint to
a conversational + code SFT post-training stage modeled after both nanochat
(SmolTalk + MMLU + GSM8K) and ByteDance's LoopLM paper (`Loop_LM.pdf`).

## Why This Setup

- **EBT paper (Gladstone et al. 2025).** The architectural and MCMC defaults
  follow the paper's **S2 (System-2 capable) recipe** from
  `EBT/job_scripts/nlp/pretrain/ebt_s2.sh`: `mcmc_step_size=0.5` with
  `mcmc_step_size_learnable=True` and `mcmc_step_size_lr_multiplier=1.5`,
  randomized step size (scale=2.0), randomized num-steps (1..3),
  `langevin_dynamics_noise=3.0`, replay buffer, `truncate_mcmc`,
  `no_mcmc_detach`, `norm_pred`, `scale_alpha_with_energy`. Per the paper's
  Table 2, every one of these regularizers is needed for thinking gains to
  emerge — removing any of them either kills the gain or shifts the
  thinking-longer vs. self-verification trade-off. We also expose a dedicated
  optimizer param group for the learnable `α` (mirroring
  `EBT/base_model_trainer.py::configure_optimizers_nlp`) so `α` actually learns.
- **Karpathy / nanochat motivation.** GPT-2-grade quality comes primarily from
  strong pretraining: good data, large token batches, 2048 context, low
  precision, and frequent CORE evaluation. We mirror nanochat's data path
  (ClimbMix-400B from `karpathy/climbmix-400b-shuffle`, BOS-aligned best-fit
  packing) so our gains track theirs.
- **ByteDance LoopLM motivation.** Compute-dependent models should learn useful
  latent/iterative computation in pretraining, not only in post-training. For
  EBTs this maps to training with MCMC refinement and evaluating whether extra
  MCMC steps help at inference. We track this with the **multi-depth CORE
  sweep** (`eval_mcmc_depths`) and report `core_thinking_gain` at every CORE
  eval — that's the paper's `Figure 6b` reproduced live, on our run.
- **Current target.** ~1B-parameter EBT (28 layers / 32 heads / 2048 embed),
  ClimbMix-400B, 2048 context, bf16, ~524K tokens per optimizer step on 8 GPUs.

## Datasets (Pre + Post)

| Stage | Source | HF repo | Why |
| --- | --- | --- | --- |
| Pretrain | NVIDIA ClimbMix-400B (Karpathy-shuffled mirror) | `karpathy/climbmix-400b-shuffle` | Same dataset that delivered the recent nanochat speedup (~27% faster to GPT-2 capability vs FineWeb-Edu). |
| Pretrain val | Last shard | `karpathy/climbmix-400b-shuffle` `shard_06542.parquet` | Pinned held-out shard, matches nanochat convention. |
| Pretrain test | Second-to-last shard | `karpathy/climbmix-400b-shuffle` `shard_06541.parquet` | Pinned held-out shard for post-hoc reporting. |
| SFT chat | SmolTalk | `HuggingFaceTB/smoltalk` | Conversational backbone, same as nanochat. |
| SFT MC | MMLU `auxiliary_train` | `cais/mmlu` | Teaches multiple-choice, same as nanochat. |
| SFT math | GSM8K `main` | `openai/gsm8k` | Teaches math + tool use, same as nanochat. |
| SFT code | CodeAlpaca-20k | `sahil2801/CodeAlpaca-20k` | Boosts HumanEval pass@1 prior. |
| SFT code | MBPP `full` | `google-research-datasets/mbpp` | Adds problem/solution pairs. |
| SFT reasoning | OpenCodeReasoning | `nvidia/OpenCodeReasoning` | LoopLM-style code reasoning when available; auto-skipped if gated. |

ClimbMix shards are downloaded directly from the HF resolve URL (no `datasets`
load-on-disk explosion). The SFT mixture is built with per-source row caps so
SmolTalk never starves MMLU/GSM8K/CodeAlpaca/MBPP.

## Repo Layout

```text
configs/                  # JSON run configs (pretrain + post-train)
run/                      # shell entry points for a fresh GPU box
scripts/                  # data, training, and checkpoint evaluation scripts
src/energy_coding/        # reusable data/model/eval utilities
analysis/                 # earlier written analysis docs
Loop_LM.pdf               # reference paper
```

`EBT/` and `nanochat/` are cloned on the GPU machine by `run/setup_gpu.sh`. We
do NOT `pip install -e nanochat` (its pyproject pins `torch==2.9.1` which would
collide with EBT's `torch==2.4.0`). Instead we add `nanochat/` to `PYTHONPATH`
and only import `nanochat.core_eval` and `nanochat.execution`.

## Recommended GPU

All configs target the **same 524,288 tokens-per-optimizer-step** so the LR
warmup/warmdown schedule doesn't need re-tuning across hardware. Pick the row
that matches your box:

| Tier | Hardware | Config | Setup script |
| --- | --- | --- | --- |
| Recommended | 8 × H200 SXM (141 GB) | `configs/ebt_1b_climbmix.json` | `run/setup_gpu.sh` |
| Datacenter Blackwell | 2 × B200 SXM (179-192 GB) | `configs/ebt_1b_climbmix_b200.json` | `run/setup_gpu_blackwell.sh` |
| Workstation Blackwell | 4 × RTX PRO 6000 (96 GB) | `configs/ebt_1b_climbmix_rtx6000.json` | `run/setup_gpu_blackwell.sh` |
| Solid | 8 × H100 SXM (80 GB) | `configs/ebt_1b_climbmix_h100.json` | `run/setup_gpu.sh` |
| Minimum | 4 × H100 SXM (80 GB) | `configs/ebt_1b_climbmix_h100.json` + `GPUS=4` | `run/setup_gpu.sh` |

> **Important: Blackwell needs a different torch wheel.** The Hopper setup pins
> `torch==2.4.0+cu121`, which has no Blackwell kernels and will fail at the
> first matmul. Use `run/setup_gpu_blackwell.sh` (installs `torch==2.7.1+cu128`)
> for any Blackwell card (B200, RTX PRO 6000, RTX 5090, GB200).

Per-config tokens-per-step:

```text
H200/H100 (8 GPUs):       8 * 2048 *  4 * 8 = 524,288
B200      (2 GPUs):      16 * 2048 *  8 * 2 = 524,288
RTX 6000  (4 GPUs):       8 * 2048 *  8 * 4 = 524,288
```

## Ready-To-Run (Copy/Paste)

```bash
tar -xzf energy-coding-train-ready.tar.gz && cd energy-coding

wandb login
export WANDB_API_KEY=...
huggingface-cli login          # optional, helps with SmolTalk/MMLU rate limits
```

Then pick **one** of the four hardware paths:

```bash
# Path A: 8 x H200 SXM (recommended, default).
bash run/setup_gpu.sh
source .venv/bin/activate
bash run/smoke_test.sh
bash run/prepare_climbmix.sh
bash run/train_1b_h200.sh
# When early-stop fires:
bash run/prepare_sft.sh
bash run/train_sft_h200.sh runs/ebt_1b_climbmix/ckpt_latest.pt
```

```bash
# Path B: 2 x B200 SXM (Blackwell datacenter).
bash run/setup_gpu_blackwell.sh
source .venv/bin/activate
bash run/smoke_test.sh
bash run/prepare_climbmix.sh
bash run/train_1b_b200.sh
# When early-stop fires:
bash run/prepare_sft.sh
bash run/train_sft_b200.sh runs/ebt_1b_climbmix_b200/ckpt_latest.pt
```

```bash
# Path C: 4 x RTX PRO 6000 (Blackwell workstation).
bash run/setup_gpu_blackwell.sh
source .venv/bin/activate
bash run/smoke_test.sh
bash run/prepare_climbmix.sh
bash run/train_1b_rtx6000.sh
# When early-stop fires:
bash run/prepare_sft.sh
bash run/train_sft_rtx6000.sh runs/ebt_1b_climbmix_rtx6000/ckpt_latest.pt
```

```bash
# Path D: 8 x H100 SXM (or 4 x H100, set GPUS=4).
bash run/setup_gpu.sh
source .venv/bin/activate
bash run/smoke_test.sh
bash run/prepare_climbmix.sh
bash run/train_1b_h100.sh
# When early-stop fires:
bash run/prepare_sft.sh
bash run/train_sft_h100.sh runs/ebt_1b_climbmix_h100/ckpt_latest.pt
```

## What The Trainer Tracks

JSONL on disk + W&B (`energy-coding-pretrain` and `energy-coding-sft`):

Training-time signals (logged every `log_interval` steps):

- `train_loss`, `train_loss_ema`, `lr`, `tok_per_sec`, `tokens_seen`
- `lr/alpha_mcmc_step_size`, `lr/matrix_decay`, `lr/no_decay`, ...
- `ebt/alpha_step_size` — current learned value of the MCMC step size `α`
- `ebt/initial_loss`, `ebt/final_step_loss`, `ebt/perplexity`
- `ebt/initial_final_pred_energies_gap` — **the paper's central pretraining
  diagnostic** that MCMC is actually descending the learned energy landscape.
  Should be positive and grow during training. If it stays ~0 or flips negative,
  energy landscape isn't being learned and you should stop and inspect.

Eval signals (every `core_eval_interval` / `humaneval_interval` steps):

- `val_loss`, `val_ppl`
- `core_metric` — CORE at the **deepest** `eval_mcmc_depths` (currently 4 sweeps)
- `core_metric_at_1`, `core_metric_at_2`, `core_metric_at_4` — per-depth read so
  you can watch the System-2 thinking curve emerge during pretraining
- `core_thinking_gain` = `max_d core_metric_at_d − min_d core_metric_at_d`. The
  EBT paper (Section 4.1.2, Figure 6b) shows this gain *grows* with training;
  if it doesn't, the energy landscape isn't learning to be smoother with more
  thinking
- per-task `core/<task>` and `core_raw/<task>`
- `humaneval_pass_at_1` (single-sample, T=0.2)
- `humaneval_pass_at_1_bon` — Best-of-`humaneval_self_verify_samples` selected
  by **minimum energy of the EBT verifier** (the paper's Self-Verification mode)
- `humaneval_pass_at_n` — pass if *any* candidate passes (sanity upper bound)

Early stop (pretraining) fires when **both**:

- `core_metric >= 0.257` (configurable as `early_stop_core`) — this is CORE at
  the deepest thinking depth, i.e. the model's best achievable score
- `tokens_seen >= 8e9` (configurable as `minimum_tokens_before_stop`)

HumanEval pass@1 is logged but is **not** a hard pretraining stop gate; it is
the coding-emergence diagnostic. The hard HumanEval target should be applied
during SFT once we have code-specific training signal.

## Manual Checkpoint Evaluation

```bash
bash run/evaluate_checkpoint.sh runs/ebt_1b_climbmix/ckpt_latest.pt
```

## Packaging For Upload

```bash
bash run/package_for_gpu.sh
```

This creates `energy-coding-train-ready.tar.gz` excluding `data/`, `runs/`,
checkpoints, venvs, and git metadata.

## Main Hyperparameters

```text
model:                28 layers, 32 heads, 2048 dim (~1B target)
context:              2048
precision:            bf16
dataset:              karpathy/climbmix-400b-shuffle  (170 train shards by default)
val shard:            shard_06542
test shard:           shard_06541
peak lr:              2e-4 (matrix), 1e-4 (embedding), 3e-4 (alpha = 1.5 * peak)
schedule:             warmup -> constant -> long warmdown
warmup:               10,000 steps
warmdown:             65% of training

# EBT-specific (paper S2 recipe, see ebt_s2.sh):
mcmc_step_size:       0.5  (learnable, lr=1.5*peak)
randomize_alpha:      x2.0 scale
randomize_num_steps:  min 2, max 3 (final landscape gets the extra sweeps)
langevin_noise:       3.0
truncate_mcmc:        true   (only final-step loss back-props)
no_mcmc_detach:       true   (proper 2nd-order training)
norm_pred:            true   (RMSNorm on prediction)
scale_alpha_with_E:   true   (alpha *= exp(E / temp))
replay_buffer:        size 48, 25% of batch sourced from buffer
clamp_futures_grad:   ±9.0 / alpha
```

Inference-time thinking knobs (in `train.eval_mcmc_depths` /
`train.humaneval_self_verify_samples`):

- CORE evaluated at depths `[1, 2, 4]` to chart the System-2 scaling curve
- HumanEval scored both single-sample and as Best-of-4 by lowest energy

## Notes

- The CORE eval bundle is downloaded once into
  `data/eval_bundle_root/eval_bundle/` (set `CORE_EVAL_DIR` to relocate).
- HumanEval evaluation uses the nanochat sandbox (`nanochat.execution.execute_code`)
  with an 8s timeout per problem.
- This repo does **not** fine-tune first. The stop target is base-model quality.
- Code fine-tuning should start only after CORE improves meaningfully.
