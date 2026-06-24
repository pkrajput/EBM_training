#!/usr/bin/env python3
"""
Manually test the trained EBT model from the command line.

Examples:
  PYTHONPATH=src python scripts/chat_ebt.py \
      --question "Write a Python function that returns the nth Fibonacci number." \
      --depth 2 --max-new-tokens 128

  PYTHONPATH=src python scripts/chat_ebt.py -q "What is the capital of France?" -d 4 -n 64

Args:
  --question / -q        the prompt to ask the model
  --depth / -d           ENERGY depth = number of MCMC "thinking" steps at
                         inference (1 = fast/shallow, higher = more refinement).
                         This is the EBT System-2 knob.
  --max-new-tokens / -n  OUTPUT length (max tokens to generate)
  --temperature / -t     0 = greedy (default), >0 = sampling
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Make `energy_coding` importable when run as `python scripts/chat_ebt.py`
# from the repo root (no need to set PYTHONPATH=src manually).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

import torch
from transformers import AutoTokenizer

from energy_coding.config import load_config
from energy_coding.evaluation import _generate_one, _override_mcmc_depth
from energy_coding.modeling import build_model, get_uncompiled_model, load_checkpoint

DEFAULT_CKPT = "models/ebt_191m/ebt_191m_sft_best.pt"
DEFAULT_CONFIG = "models/ebt_191m/ebt_191m_climbmix_8xb200.json"


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="Manually test the trained EBT model.")
    ap.add_argument("--question", "-q", required=True, help="the prompt/question to ask")
    ap.add_argument("--depth", "-d", type=int, default=2,
                    help="energy/MCMC thinking depth at inference (>=1; higher = more refinement)")
    ap.add_argument("--max-new-tokens", "-n", type=int, default=128, help="output length (tokens)")
    ap.add_argument("--temperature", "-t", type=float, default=0.0, help="0=greedy, >0=sampling")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--raw", action="store_true",
                    help="send the question verbatim instead of wrapping in the SFT 'User:/[[Answer]]:' template")
    args = ap.parse_args()

    device = resolve_device(args.device)
    config = load_config(args.config)
    # route generation knobs through the fields _generate_one reads
    config.train.humaneval_temperature = args.temperature

    print(f"[load] config={args.config}")
    print(f"[load] checkpoint={args.checkpoint}")
    print(f"[device] {device}")

    model, _ = build_model(config, device=device, execution_mode="inference")
    raw_model = get_uncompiled_model(model)
    step, tokens_seen, _ = load_checkpoint(args.checkpoint, raw_model, strict=True)
    raw_model.eval()
    print(f"[ckpt] step={step} tokens={tokens_seen:,}")

    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer, clean_up_tokenization_spaces=False)

    # Match the SFT training format unless --raw.
    if args.raw:
        prompt = args.question
    else:
        prompt = f"User: {args.question}\n[[Answer]]: "

    print("\n" + "=" * 70)
    print(f"PROMPT (energy depth={args.depth}, max_new_tokens={args.max_new_tokens}, temp={args.temperature}):")
    print(prompt)
    print("-" * 70)

    t0 = time.time()
    with _override_mcmc_depth(raw_model, args.depth):
        text, mean_energy = _generate_one(
            raw_model, tokenizer, prompt, config, device, max_new_tokens=args.max_new_tokens
        )
    dt = time.time() - t0

    print("COMPLETION:")
    print(text)
    print("=" * 70)
    print(f"[stats] mean final-step energy = {mean_energy:.4f} "
          f"(lower = model more 'confident' the answer fits) | {dt:.1f}s")


if __name__ == "__main__":
    main()
