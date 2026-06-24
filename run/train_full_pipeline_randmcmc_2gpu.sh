#!/usr/bin/env bash
# End-to-end on the uni 2-GPU box (single RTX 5000 Ada, GPU 0):
#   Phase A: pretrain 191M EBT WITH randomized MCMC, stop when CORE matches the
#            old fixed-depth model (early_stop_core in config).
#   Phase B: SFT (same recipe/size) on the code-heavy mixture.
# The System-1 vs System-2 MCQ eval is run separately afterwards via
# scripts/mcq_eval.py (no time pressure).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PRETRAIN_OUT="runs/ebt_191m_randmcmc_2gpu"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo; echo "================================================================"; echo "[$(ts)] $1"; echo "================================================================"; }

banner "PHASE A: PRETRAIN 191M EBT + randomized MCMC (stop at CORE match)"
bash run/train_191m_randmcmc_2gpu.sh

if [ -f "${PRETRAIN_OUT}/pretrain_status.json" ] && grep -q '"collapsed": true' "${PRETRAIN_OUT}/pretrain_status.json"; then
  echo "[$(ts)] ERROR: pretrain collapsed; refusing to SFT. Inspect ${PRETRAIN_OUT}."
  exit 1
fi

LATEST_CKPT="$(ls -t ${PRETRAIN_OUT}/ckpt_step_*.pt 2>/dev/null | head -1 || true)"
[ -z "$LATEST_CKPT" ] && LATEST_CKPT="${PRETRAIN_OUT}/ckpt_latest.pt"
if [ ! -f "$LATEST_CKPT" ]; then
  echo "[$(ts)] ERROR: no pretrain checkpoint in ${PRETRAIN_OUT}. Aborting."
  exit 1
fi
banner "Pretrain checkpoint: $LATEST_CKPT"

if [ ! -f data/sft/train.jsonl ]; then
  banner "PHASE B-PREP: build SFT JSONL (code-heavy mixture)"
  bash run/prepare_sft.sh
fi

banner "PHASE B: SFT from $LATEST_CKPT"
bash run/train_sft_191m_randmcmc_2gpu.sh "$LATEST_CKPT"

banner "PIPELINE COMPLETE — run scripts/mcq_eval.py for the System-1/2 comparison"
