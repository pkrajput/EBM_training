#!/usr/bin/env bash
# Setup for NVIDIA Blackwell GPUs (B200, RTX PRO 6000, RTX 5090, GB200).
# Blackwell requires CUDA 12.8 wheels; the standard setup_gpu.sh pins
# torch==2.4.0 (cu121) which has NO sm_100/sm_120 kernels and will not run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# Blackwell-compatible deps (torch 2.7 + cu128).
python -m pip install -r requirements_blackwell.txt

# Clone EBT (the upstream Energy-Based Transformer code we wrap).
if [ ! -d EBT ] || [ -z "$(ls -A EBT 2>/dev/null)" ]; then
  rm -rf EBT
  git clone https://github.com/alexiglad/EBT.git EBT
fi

# Clone nanochat for its CORE evaluator and HumanEval sandbox only.
if [ ! -d nanochat ] || [ -z "$(ls -A nanochat 2>/dev/null)" ]; then
  rm -rf nanochat
  git clone https://github.com/karpathy/nanochat.git nanochat
fi

# We intentionally do NOT `pip install -r EBT/requirements.txt` here — those
# pins (torch==2.4.0, xformers==0.0.27.post2) would downgrade torch and break
# Blackwell support. The libraries EBT actually imports (pytorch_lightning,
# torchmetrics, transformers, diffusers, torchvision) are already installed
# above at Blackwell-compatible versions.

# Optional: evalplus for offline HumanEval+ scoring. Don't fail setup if unavailable.
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

# Sanity-check we can actually see the GPUs as Blackwell.
python - <<'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA: {torch.version.cuda}")
    n = torch.cuda.device_count()
    print(f"Visible devices: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {props.name}  sm_{props.major}{props.minor}  "
              f"{props.total_memory / 1024**3:.1f} GB")
EOF

echo "Setup complete."
echo "Activate with: source .venv/bin/activate"
