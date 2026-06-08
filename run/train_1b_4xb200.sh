#!/usr/bin/env bash
# 1.05B EBT (20L * 16H * 2048E) on 4 x B200 SXM. Designed for the $2.5K budget
# scenario: train to ~5B tokens of ClimbMix, then SFT on code-heavy mixture.
#
#   tokens/step = 4 * 1024 * 32 * 4 = 524,288  (same schedule as small runs)
#   estimated throughput ~12-16K tok/s on 4 x B200
#   -> ~5B tokens in ~85-95 hours ~$1700-1900
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
# B200 has NVLink so leave P2P on. Single-node so IB off.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

GPUS="${GPUS:-4}"

torchrun --standalone --nproc_per_node="$GPUS" scripts/train_1b.py \
  --config configs/ebt_1b_climbmix_4xb200.json
