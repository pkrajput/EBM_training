#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/ebt_1b_climbmix.json}"
NUM_WORKERS="${NUM_WORKERS:-16}"

CMD=(python scripts/prepare_data.py --config "$CONFIG" --num-workers "$NUM_WORKERS")

# Optional override for the shard count (e.g. NUM_TRAIN_SHARDS=300 to download more).
if [ -n "${NUM_TRAIN_SHARDS:-}" ]; then
  CMD+=(--num-train-shards "$NUM_TRAIN_SHARDS")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
