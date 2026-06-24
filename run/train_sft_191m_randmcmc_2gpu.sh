#!/usr/bin/env bash
# SFT for the randomized-MCMC 191M EBT on a single RTX 5000 Ada (GPU 0).
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash run/train_sft_191m_randmcmc_2gpu.sh <base_checkpoint.pt>"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

python scripts/train_sft.py \
  --config configs/ebt_191m_randmcmc_2gpu.json \
  --base-checkpoint "$1"
