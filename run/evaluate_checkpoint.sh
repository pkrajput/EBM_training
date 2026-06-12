#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash run/evaluate_checkpoint.sh runs/.../ckpt_latest.pt"
  echo ""
  echo "Optional env vars:"
  echo "  CONFIG                     (default configs/ebt_1b_climbmix.json)"
  echo "  HUMANEVAL_MAX_PROBLEMS     (default 164 - full HumanEval)"
  echo "  HUMANEVAL_SELF_VERIFY      (default 4)"
  echo "  MBPP_MAX_PROBLEMS          (default 257 - full sanitized test)"
  echo "  MBPP_SELF_VERIFY           (default 4)"
  echo "  CORE_MAX_PER_TASK          (default 500)"
  echo "  EVAL_MCMC_DEPTHS           (default '1,2,4,8')"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/ebt_1b_climbmix.json}"
HUMANEVAL_MAX_PROBLEMS="${HUMANEVAL_MAX_PROBLEMS:-164}"
HUMANEVAL_SELF_VERIFY="${HUMANEVAL_SELF_VERIFY:-4}"
MBPP_MAX_PROBLEMS="${MBPP_MAX_PROBLEMS:-257}"
MBPP_SELF_VERIFY="${MBPP_SELF_VERIFY:-4}"
CORE_MAX_PER_TASK="${CORE_MAX_PER_TASK:-500}"
EVAL_MCMC_DEPTHS="${EVAL_MCMC_DEPTHS:-1,2,4,8}"

OVERLAY=$(mktemp -t energy_coding_eval_overlay.XXXXXX.json)
python - "$OVERLAY" <<EOF
import json, sys
overlay = {"train": {
    "humaneval_max_problems": ${HUMANEVAL_MAX_PROBLEMS},
    "humaneval_self_verify_samples": ${HUMANEVAL_SELF_VERIFY},
    "mbpp_max_problems": ${MBPP_MAX_PROBLEMS},
    "mbpp_self_verify_samples": ${MBPP_SELF_VERIFY},
    "core_max_per_task": ${CORE_MAX_PER_TASK},
    "eval_mcmc_depths": [int(d) for d in "${EVAL_MCMC_DEPTHS}".split(",") if d.strip()],
}}
open(sys.argv[1], "w").write(json.dumps(overlay))
EOF

trap 'rm -f "$OVERLAY"' EXIT

python scripts/evaluate_checkpoint.py \
  --config "$CONFIG" \
  --overlay "$OVERLAY" \
  --checkpoint "$1"
