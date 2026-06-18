#!/usr/bin/env bash
# End-to-end pipeline for the 8xB200 / 191M EBT / MBPP run.
#
# Phase A: pretrain on ClimbMix (resumes from the latest checkpoint; stops on
#          CORE>=early_stop_core + min_tokens, OR on target_tokens, OR on
#          loss_patience divergence, OR on uniform collapse).
# Phase B: SFT on code-heavy mixture using the best pretrain checkpoint.
# Phase C: CHEAP eval (10 random HumanEval + 10 random MBPP) on the SFT
#          checkpoint, for a quick MBPP signal within budget.
#
# Each phase logs to its own W&B run; everything streams to pipeline.log.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PRETRAIN_OUT="runs/ebt_191m_climbmix_8xb200"
SFT_OUT="runs/ebt_191m_sft_8xb200"
CONFIG="configs/ebt_191m_climbmix_8xb200.json"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo; echo "================================================================"; echo "[$(ts)] $1"; echo "================================================================"; }

# ----------------------------------------------------------------
banner "PHASE A: PRETRAIN (191M EBT on ClimbMix, resumes from checkpoint)"
# ----------------------------------------------------------------
bash run/train_191m_8xb200.sh

# ORCHESTRATION SAFETY: never run SFT if pretrain ended in a uniform collapse.
if [ -f "${PRETRAIN_OUT}/pretrain_status.json" ]; then
  if grep -q '"collapsed": true' "${PRETRAIN_OUT}/pretrain_status.json"; then
    echo "[$(ts)] ERROR: pretrain ended in COLLAPSE (see ${PRETRAIN_OUT}/pretrain_status.json)."
    echo "[$(ts)] Refusing to run SFT on a collapsed model. Inspect/fix and rerun pretrain."
    exit 1
  fi
fi

LATEST_PRETRAIN_CKPT="$(ls -t ${PRETRAIN_OUT}/ckpt_step_*.pt 2>/dev/null | head -1 || true)"
if [ -z "$LATEST_PRETRAIN_CKPT" ]; then
  LATEST_PRETRAIN_CKPT="${PRETRAIN_OUT}/ckpt_latest.pt"
fi
if [ ! -f "$LATEST_PRETRAIN_CKPT" ]; then
  echo "[$(ts)] ERROR: no pretrain checkpoint found in ${PRETRAIN_OUT}. Aborting."
  exit 1
fi
banner "Pretrain produced checkpoint: $LATEST_PRETRAIN_CKPT"

# ----------------------------------------------------------------
banner "PHASE B-PREP: build SFT JSONL (code-heavy mixture)"
# ----------------------------------------------------------------
if [ ! -f data/sft/train.jsonl ]; then
  bash run/prepare_sft.sh
else
  echo "[$(ts)] data/sft/train.jsonl already exists, skipping SFT prep."
fi

# ----------------------------------------------------------------
banner "PHASE B: SFT (code-heavy, 3000 steps) starting from $LATEST_PRETRAIN_CKPT"
# ----------------------------------------------------------------
bash run/train_sft_191m_8xb200.sh "$LATEST_PRETRAIN_CKPT"

# Prefer the BEST (lowest-ema) SFT checkpoint — robust if SFT collapsed late.
LATEST_SFT_CKPT="${SFT_OUT}/ckpt_best.pt"
if [ ! -f "$LATEST_SFT_CKPT" ]; then
  LATEST_SFT_CKPT="$(ls -t ${SFT_OUT}/ckpt_step_*.pt 2>/dev/null | head -1 || true)"
fi
if [ -z "$LATEST_SFT_CKPT" ]; then
  LATEST_SFT_CKPT="${SFT_OUT}/ckpt_latest.pt"
fi
if [ ! -f "$LATEST_SFT_CKPT" ]; then
  echo "[$(ts)] ERROR: no SFT checkpoint found in ${SFT_OUT}. Aborting before final eval."
  exit 1
fi
banner "SFT produced checkpoint: $LATEST_SFT_CKPT"

# ----------------------------------------------------------------
banner "PHASE C: CHEAP eval (10 random HumanEval + 10 random MBPP) on $LATEST_SFT_CKPT"
# ----------------------------------------------------------------
CONFIG="$CONFIG" \
HUMANEVAL_MAX_PROBLEMS="${HUMANEVAL_MAX_PROBLEMS:-10}" \
HUMANEVAL_SELF_VERIFY="${HUMANEVAL_SELF_VERIFY:-1}" \
CORE_MAX_PER_TASK="${CORE_MAX_PER_TASK:-100}" \
EVAL_MCMC_DEPTHS="${EVAL_MCMC_DEPTHS:-1,2}" \
MBPP_MAX_PROBLEMS="${MBPP_MAX_PROBLEMS:-10}" \
MBPP_SELF_VERIFY="${MBPP_SELF_VERIFY:-1}" \
bash run/evaluate_checkpoint.sh "$LATEST_SFT_CKPT"

banner "PIPELINE COMPLETE"
echo "[$(ts)] Pretrain ckpt: $LATEST_PRETRAIN_CKPT"
echo "[$(ts)] SFT ckpt:      $LATEST_SFT_CKPT"
echo "[$(ts)] Eval JSONL:    ${PRETRAIN_OUT}/eval_metrics.jsonl + ${SFT_OUT}/eval_metrics.jsonl"
