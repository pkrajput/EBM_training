#!/usr/bin/env bash
# Train the 1B EBT on 4 x RTX PRO 6000 (Blackwell, 96 GB GDDR7). Global
# tokens/step ~ 524K, matching the default 8xH200 schedule.
set -euo pipefail

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

# RTX PRO 6000 is a workstation card without NVLink. PCIe-only NCCL works
# best with these flags; tune to your machine if NCCL warns.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

GPUS="${GPUS:-4}"

torchrun --standalone --nproc_per_node="$GPUS" scripts/train_1b.py \
  --config configs/ebt_1b_climbmix_rtx6000.json
