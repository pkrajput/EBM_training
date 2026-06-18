#!/usr/bin/env bash
# 191M EBT pretrain on 8 x B200 (Blackwell, 183 GB each).
# bs=8 * seq=512 * accum=8 * world=8 = 262,144 tokens/optimizer step.
# Observed ~78K tok/s on 8xB200.
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

GPUS="${GPUS:-8}"

torchrun --standalone --nproc_per_node="$GPUS" scripts/train_191m.py \
  --config configs/ebt_191m_climbmix_8xb200.json
