#!/usr/bin/env bash
# 8 x RTX PRO 6000 Blackwell Workstation (96 GB each) launch for the 127M
# 'small' EBT. Same paper-faithful S2 recipe as the 2 x B200 small config;
# seq=512 + bs=16 + grad_accum=8 -> 524,288 tokens/optimizer step.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
# RTX PRO 6000 WS has no NVLink, only PCIe 5.0 16x. P2P over PCIe still helps;
# IB always disabled (we're single-node).
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

GPUS="${GPUS:-8}"

torchrun --standalone --nproc_per_node="$GPUS" scripts/train_1b.py \
  --config configs/ebt_small_climbmix_8x6000.json
