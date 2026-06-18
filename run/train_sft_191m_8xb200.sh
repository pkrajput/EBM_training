#!/usr/bin/env bash
# SFT phase for the 191M EBT on 8 x B200. Run AFTER pretraining
# (run/train_191m_8xb200.sh) reaches its target tokens (or is called directly by
# run/train_full_pipeline_8xb200.sh on the latest pretrain checkpoint).
# Code-heavy mixture is configured in src/energy_coding/data.py.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash run/train_sft_191m_8xb200.sh runs/ebt_191m_climbmix_8xb200/ckpt_latest.pt"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

GPUS="${GPUS:-8}"

torchrun --standalone --nproc_per_node="$GPUS" scripts/train_sft.py \
  --config configs/ebt_191m_climbmix_8xb200.json \
  --base-checkpoint "$1"
