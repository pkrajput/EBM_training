#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$ROOT/src:$PYTHONPATH"

python scripts/prepare_data.py \
  --config configs/smoke_test.json \
  --num-train-shards 1 \
  --num-workers "${NUM_WORKERS:-4}"

torchrun --standalone --nproc_per_node="${GPUS:-1}" scripts/train_1b.py \
  --config configs/smoke_test.json \
  --no-resume
