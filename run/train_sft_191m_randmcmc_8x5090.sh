#!/usr/bin/env bash
# SFT the randomized-MCMC 191M EBT into a chat model on 8x RTX 5090.
# Budget: 3000 steps * 131,072 tok/step = ~393M SFT tokens (~5 h at ~23k tok/s).
# Depth is preserved automatically: the model config (randomize_mcmc_num_steps,
# Langevin noise, learnable step size) is identical to pretraining, so SFT keeps
# training the model with variable MCMC depth.
#
# Usage:
#   export WANDB_API_KEY=...   # optional but recommended
#   bash run/train_sft_191m_randmcmc_8x5090.sh runs/ebt_191m_randmcmc_8x5090/ckpt_step_52000.pt
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash run/train_sft_191m_randmcmc_8x5090.sh <base_checkpoint.pt>"
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
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

GPUS="${GPUS:-8}"

torchrun --standalone --nproc_per_node="$GPUS" scripts/train_sft.py \
  --config configs/ebt_191m_randmcmc_8x5090.json \
  --base-checkpoint "$1"
