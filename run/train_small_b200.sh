#!/usr/bin/env bash
# Train the ~160M ("small") EBT on 2 x B200 SXM. Same paper-faithful S2 recipe
# as the 1B config; only the model spec is smaller (12 layers x 12 heads x
# 768 dim, GPT-2-124M-ish). Picks per-step token budget that lets first CORE
# eval fire within a few hundred wall-clock seconds.
#
# Global tokens/step = 4 * 2048 * 64 * 2 = 524,288 (same schedule as 1B).
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

GPUS="${GPUS:-2}"

torchrun --standalone --nproc_per_node="$GPUS" scripts/train_1b.py \
  --config configs/ebt_small_climbmix_b200.json
