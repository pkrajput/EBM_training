#!/usr/bin/env bash
# 191M EBT pretrain WITH randomized MCMC on a single RTX 5000 Ada (GPU 0).
# (GPU 1 on this box is shared with another user's job, so we pin to GPU 0.)
# Stops when CORE matches the old fixed-depth model (early_stop_core in config).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

# Single GPU -> no torchrun/DDP needed. Randomized MCMC is ON via the config
# (randomize_mcmc_num_steps=3); pass --randomize-mcmc-steps 0 to disable.
python scripts/train_191m.py \
  --config configs/ebt_191m_randmcmc_2gpu.json "$@"
