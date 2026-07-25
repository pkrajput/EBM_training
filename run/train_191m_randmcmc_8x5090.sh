#!/usr/bin/env bash
# 191M EBT pretrain WITH randomized MCMC on 8x RTX 5090 (32 GB, Blackwell sm_120).
# Budget box: this is the money-constrained alternative to 8x H100.
#
# Memory: RTX 5090 has only 32 GB, and randomized MCMC + no_mcmc_detach is
#   activation-heavy (we OOM'd at bs=2 even on 48 GB). So device_batch_size=1.
#   Global batch = 1 * 512 * 16 * 8 = 65,536 tokens / optimizer step.
#   -> smaller batch than the H100 config on purpose: more optimizer updates per
#      token, which is better when the TOKEN BUDGET (not compute) is the limit.
#   If you still OOM: first drop core_max_per_task, then set no_mcmc_detach=false
#   in the config (first-order MCMC, ~half the activation memory).
#
# Prereq on the box:
#   bash run/setup_gpu_blackwell.sh     # torch 2.7.1+cu128 + NCCL hang fix
#   export WANDB_API_KEY=...
#
# CONTROLLED STUDY: the System-1 control does NOT need a paid retrain — use the
# existing fixed-depth checkpoint (models/ebt_191m/ebt_191m_pretrain_base_step7000.pt)
# as the baseline and compare against this run's ~1.8B-token checkpoint (matched
# tokens) plus the final one. To retrain a matched control anyway:
#   bash run/train_191m_randmcmc_8x5090.sh --randomize-mcmc-steps 0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# RTX 5090 has no NVLink; DDP all-reduce goes over PCIe. For a 191M model the
# gradient bucket is tiny (~0.4 GB) so PCIe is fine. Keep NCCL on the loopback
# for the single-node --standalone case.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

GPUS="${GPUS:-8}"

torchrun --standalone --nproc_per_node="$GPUS" scripts/train_191m.py \
  --config configs/ebt_191m_randmcmc_8x5090.json "$@"
