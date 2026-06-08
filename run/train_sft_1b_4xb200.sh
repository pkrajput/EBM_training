#!/usr/bin/env bash
# SFT phase for the 1.05B EBT. Run AFTER pretraining (run/train_1b_4xb200.sh)
# reaches its target tokens. Code-heavy mixture is configured in
# src/energy_coding/data.py (SmolTalk 250K + MMLU 60K + GSM8K 8K +
# CodeAlpaca 20K + MBPP 974 + OpenCodeReasoning 80K -> ~420K examples,
# 35-40% code).
#
# Expected cost: ~$340 (~17 hours on 4 x B200 at 4000 max_steps).
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash run/train_sft_1b_4xb200.sh runs/ebt_1b_climbmix_4xb200/ckpt_latest.pt"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

GPUS="${GPUS:-4}"

torchrun --standalone --nproc_per_node="$GPUS" scripts/train_sft.py \
  --config configs/ebt_1b_climbmix_4xb200.json \
  --base-checkpoint "$1"
