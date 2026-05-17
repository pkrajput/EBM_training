#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='data' \
  --exclude='runs' \
  --exclude='*.pt' \
  --exclude='*.ckpt' \
  --exclude='__pycache__' \
  --exclude='energy-coding-train-ready.tar.gz' \
  -czf energy-coding-train-ready.tar.gz .

echo "Created: $ROOT/energy-coding-train-ready.tar.gz"
