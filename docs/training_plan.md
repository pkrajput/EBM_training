# 1B EBT Training Plan

## Pretraining Dataset

- Source: `karpathy/climbmix-400b-shuffle` (NVIDIA ClimbMix-400B, shuffled mirror).
- Train: first `170` shards (~10B tokens) by default; configurable via
  `--num-train-shards` on `scripts/prepare_data.py`.
- Validation: `shard_06542.parquet` (pinned).
- Test: `shard_06541.parquet` (pinned, reported post-hoc).
- Loader: BOS-aligned best-fit packing into EBT's `[B, 1, T+1]` format.

ClimbMix was selected to mirror Karpathy's recent nanochat speedup (FineWeb-Edu
-> ClimbMix gave ~27% wallclock improvement to GPT-2 capability) while still
matching the corpus profile that LoopLM trains on (FineWeb / DCLM / Nemotron
mixtures totalling 7.7T tokens).

## Metrics

JSONL on disk:

- `train_loss`, `train_loss_ema`, `lr`, `tok_per_sec`, `tokens_seen`
- `val_loss`, `val_ppl`
- `core_metric`, plus per-task `core_centered_results[*]` and `core_raw_results[*]`
- `humaneval_pass_at_1` (diagnostic during base pretraining)

W&B (when `WANDB_API_KEY` is set):

- pretraining loss / validation curves
- CORE curve (overall + per task)
- HumanEval pass@1
- SFT loss / validation curves

Stop criteria for pretraining:

- `CORE >= 0.257`
- at least `8B` tokens seen

HumanEval pass@1 is **not** a hard pretraining stop condition. For a base model
it is unrealistically noisy; we use it to detect coding emergence, then make it
a hard target during SFT.

## Post-Training (SFT) Data

Conversational + reasoning + code mix, prepared by `scripts/prepare_sft_data.py`:

| Source | HF repo | Default cap | Role |
| --- | --- | --- | --- |
| SmolTalk train | `HuggingFaceTB/smoltalk` (config `all`) | 460,000 | General conversation backbone. |
| MMLU `auxiliary_train` | `cais/mmlu` | 100,000 | Multiple-choice reasoning. |
| GSM8K `main` train | `openai/gsm8k` | 8,000 | Math word problems. |
| CodeAlpaca-20k | `sahil2801/CodeAlpaca-20k` | 20,000 | Instruction-following Python. |
| MBPP full train | `google-research-datasets/mbpp` | 974 | Code synthesis. |
| OpenCodeReasoning split_0 | `nvidia/OpenCodeReasoning` (streaming) | 25,000 | LoopLM-style reasoning data (skipped if gated). |

Splits written to `data/sft/train.jsonl`, `data/sft/val.jsonl`,
`data/sft/test.jsonl`. Sources whose download fails (auth gate, network, etc.)
are skipped with a warning instead of crashing.

## Hyperparameters

- 28 transformer blocks
- 32 attention heads
- 2048 embedding dim
- 2048 context
- bf16
- 2e-4 peak LR
- 10K warmup steps
- 65% warmdown
- 0.1 weight decay
- global batch: 524,288 tokens on 8 GPUs

## Compute-Dependent Training

LoopLM motivates learning iterative computation during pretraining. For EBT this
repo uses:

- `ebt_type = time_embed`
- `mcmc_num_steps = 1`
- randomized MCMC depth up to 2
- Langevin noise = 3.0
- replay buffer enabled
- truncation and `no_mcmc_detach` enabled, following the original XL EBT script

The core research test is whether later (test-time) MCMC sweeps improve CORE and
HumanEval. The trainer can be re-evaluated at higher MCMC depth by reloading the
checkpoint and overriding `mcmc_num_steps` in the config.
