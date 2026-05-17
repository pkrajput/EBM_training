#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# Our project requirements (pinned to match EBT's torch==2.4.0 cu121 stack).
python -m pip install -r requirements.txt

# Clone EBT (the upstream Energy-Based Transformer code we wrap).
if [ ! -d EBT ] || [ -z "$(ls -A EBT 2>/dev/null)" ]; then
  rm -rf EBT
  git clone https://github.com/alexiglad/EBT.git EBT
fi

# Clone nanochat for its CORE evaluator and HumanEval sandbox only.
# IMPORTANT: we do NOT `pip install -e nanochat` because its pyproject pins
# torch==2.9.1 which would break EBT's torch==2.4.0 environment.
if [ ! -d nanochat ] || [ -z "$(ls -A nanochat 2>/dev/null)" ]; then
  rm -rf nanochat
  git clone https://github.com/karpathy/nanochat.git nanochat
fi

# Install EBT's pinned deps. Our requirements.txt is a superset, so this should
# be a no-op. Allow it to keep failing gracefully if EBT bumps a pin.
python -m pip install -r EBT/requirements.txt || true

# Upgrade NCCL. PyTorch wheels bundle NCCL 2.26.2 which hangs on init in some
# containerized envs (verified on vast.ai Blackwell + cu128, likely safer to
# upgrade everywhere). 2.30.4+ fixes the hang. Torch will print a version-pin
# warning but the ABI is compatible.
python -m pip install --upgrade "nvidia-nccl-cu12>=2.30,<3" 2>/dev/null || true

# Make sure the (small) runtime deps that nanochat's core_eval needs are present.
# We only import `nanochat.core_eval.evaluate_task` and `nanochat.execution.execute_code`.
python -m pip install --upgrade \
  "jinja2>=3.1" \
  "PyYAML>=6.0" \
  "filelock>=3.16" \
  "sentencepiece>=0.2.0" \
  "protobuf>=4.25"

# Optional: evalplus for offline HumanEval+/MBPP+ scoring. Don't fail setup if unavailable.
python -m pip install "evalplus>=0.3" 2>/dev/null || true

# Pre-download the CORE eval bundle so the first training-time eval is fast.
mkdir -p data/eval_bundle_root
if [ ! -f data/eval_bundle_root/eval_bundle/core.yaml ]; then
  echo "Downloading CORE eval bundle (one-time)..."
  curl -L --fail -o data/eval_bundle_root/eval_bundle.zip \
    https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
  python - <<'EOF'
import zipfile, shutil, os, tempfile
src = "data/eval_bundle_root/eval_bundle.zip"
with tempfile.TemporaryDirectory() as td:
    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(td)
    extracted = os.path.join(td, "eval_bundle")
    target = "data/eval_bundle_root/eval_bundle"
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.move(extracted, target)
print("CORE eval bundle installed at", os.path.abspath("data/eval_bundle_root/eval_bundle"))
EOF
fi

echo "Setup complete."
echo "Activate with: source .venv/bin/activate"
