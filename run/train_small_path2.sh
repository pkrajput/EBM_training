#!/usr/bin/env bash
# Path-2 (FULL paper-faithful S2 recipe) run on 8 x RTX PRO 6000 Blackwell
# Workstation cards. Differences vs the prior `train_small_8x6000.sh`:
#
#   model.mcmc_num_steps      = 4      (was 1)
#   model.no_mcmc_detach      = true   (was false)
#   model.mcmc_replay_buffer  = true   (was false)
#
# Together these restore the three S2-recipe regularizers that the EBT paper's
# Table 2 says are required for thinking-scaling to emerge. Memory cost: ~3x
# more activation memory and ~2.5-3x slower per optimizer step than the
# relaxed-S2 run. With seq=512 + bs=4 + accum=32 on 8 RTX 6000s we stay under
# ~30-40 GB per GPU.
#
# Global tokens per optimizer step: 4 * 512 * 32 * 8 = 524,288 (unchanged).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
# RTX PRO 6000 WS has no NVLink, only PCIe 5.0 16x. P2P over PCIe still helps.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

GPUS="${GPUS:-8}"

torchrun --standalone --nproc_per_node="$GPUS" scripts/train_1b.py \
  --config configs/ebt_small_path2_8x6000.json
