#!/usr/bin/env bash
# 191M EBT pretrain WITH randomized MCMC on an 8-GPU vast.ai box.
# This is the "System-2" arm of the depth-scaling study: it trains with a
# RANDOMIZED number of MCMC steps (randomize_mcmc_num_steps=3, min=2) so the
# learned energy landscape stays descendable past the nominal step count and
# accuracy IMPROVES with more inference-time "thinking" depth.
#
# Global batch = device_batch_size(4) * seq(512) * accum(16) * world(8)
#              = 262,144 tokens / optimizer step  (same proven batch as the old
#                fixed-depth 8xB200 run, so the LR schedule transfers).
#
# Memory: bs=4/accum=16 fits 80 GB cards (H100/A100-80GB). On B200/141GB edit the
#         config to device_batch_size=8 / gradient_accumulation_steps=8 (same
#         global batch, ~2x fewer accum micro-steps, faster).
#
# CONTROLLED STUDY (for the paper):
#   System-2 (this file, default):     randomized MCMC steps  -> depth helps
#   System-1 control (same config):    pass  --randomize-mcmc-steps 0  -> fixed depth
#     e.g.:  bash run/train_191m_randmcmc_8gpu.sh --randomize-mcmc-steps 0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

GPUS="${GPUS:-8}"

torchrun --standalone --nproc_per_node="$GPUS" scripts/train_191m.py \
  --config configs/ebt_191m_randmcmc_8gpu.json "$@"
