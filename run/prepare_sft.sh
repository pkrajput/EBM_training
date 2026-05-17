#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT/src:$PYTHONPATH"

python scripts/prepare_sft_data.py \
  --config configs/ebt_1b_climbmix.json
