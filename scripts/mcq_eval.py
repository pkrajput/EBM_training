#!/usr/bin/env python3
"""
System-1 vs System-2 MCQ experiment.

Scores each answer choice by the model's (length-normalized) log-likelihood and
picks the best one. For the EBT we sweep the MCMC "thinking depth" (System-1 =
depth 1, System-2 = deeper) to test whether more test-time compute reliably
improves multiple-choice accuracy. A standard small LM (GPT-2) is the System-1
baseline (no depth knob).

Usage:
  python scripts/mcq_eval.py --dataset arc_easy --limit 200 --depths 1,2,4
  python scripts/mcq_eval.py --dataset arc_easy --limit 200 --slm gpt2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from energy_coding.config import load_config
from energy_coding.evaluation import _override_mcmc_depth
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


# ---------------------------------------------------------------- datasets
def load_mcq(name: str, limit: int):
    items = []
    if name == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
        for ex in ds:
            texts = ex["choices"]["text"]
            labels = ex["choices"]["label"]
            if ex["answerKey"] not in labels:
                continue
            items.append({"q": ex["question"], "choices": texts,
                          "gold": labels.index(ex["answerKey"])})
            if len(items) >= limit:
                break
    elif name == "piqa":
        ds = load_dataset("piqa", split="validation", trust_remote_code=True)
        for ex in ds:
            items.append({"q": ex["goal"], "choices": [ex["sol1"], ex["sol2"]],
                          "gold": int(ex["label"])})
            if len(items) >= limit:
                break
    elif name == "sciq":
        ds = load_dataset("sciq", split="validation")
        for ex in ds:
            choices = [ex["distractor1"], ex["distractor2"], ex["distractor3"], ex["correct_answer"]]
            items.append({"q": ex["question"], "choices": choices, "gold": 3})
            if len(items) >= limit:
                break
    else:
        raise ValueError(f"unknown dataset {name}")
    return items


def fmt_context(q: str) -> str:
    return f"Question: {q}\nAnswer:"


# ---------------------------------------------------------------- EBT scoring
@torch.enable_grad()  # EBT MCMC needs grad
def ebt_choice_logprob(raw_model, tokenizer, device, context: str, choice: str):
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(context + choice, add_special_tokens=False)
    n_choice = len(full_ids) - len(ctx_ids)
    if n_choice <= 0:
        return -1e9, 1
    max_len = raw_model.hparams.context_length if hasattr(raw_model.hparams, "context_length") else 512
    inp = full_ids[:-1][-(max_len):]
    offset = len(full_ids) - 1 - len(inp)  # if truncated from the left
    input_ids = torch.tensor([inp], device=device)
    pd, _ = raw_model.forward(input_ids, start_pos=0, learning=False,
                              return_raw_logits=True, no_randomness=True)
    logits = pd[-1][0].float()  # (S, V)
    logprobs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for j in range(len(ctx_ids), len(full_ids)):
        pos = j - 1 - offset
        if 0 <= pos < logprobs.shape[0]:
            total += float(logprobs[pos, full_ids[j]])
    return total, n_choice


def eval_ebt(items, raw_model, tokenizer, device, depth):
    raw_model.eval()
    correct = correct_norm = 0
    t0 = time.time()
    with _override_mcmc_depth(raw_model, depth):
        for k, it in enumerate(items):
            ctx = fmt_context(it["q"])
            scores, norms = [], []
            for ch in it["choices"]:
                s, n = ebt_choice_logprob(raw_model, tokenizer, device, ctx, " " + ch.strip())
                scores.append(s)
                norms.append(s / max(n, 1))
            if max(range(len(scores)), key=lambda i: scores[i]) == it["gold"]:
                correct += 1
            if max(range(len(norms)), key=lambda i: norms[i]) == it["gold"]:
                correct_norm += 1
            if (k + 1) % 25 == 0:
                print(f"    depth {depth}: {k+1}/{len(items)} done ({time.time()-t0:.0f}s)", flush=True)
    n = len(items)
    return correct / n, correct_norm / n


# ---------------------------------------------------------------- SLM baseline
def eval_slm(items, model_name, device):
    from transformers import AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    correct = correct_norm = 0
    with torch.no_grad():
        for it in items:
            ctx = fmt_context(it["q"])
            scores, norms = [], []
            for ch in it["choices"]:
                ctx_ids = tok.encode(ctx, add_special_tokens=False)
                full_ids = tok.encode(ctx + " " + ch.strip(), add_special_tokens=False)
                n_choice = max(1, len(full_ids) - len(ctx_ids))
                inp = torch.tensor([full_ids], device=device)
                logits = mdl(inp).logits[0].float()
                lp = torch.log_softmax(logits, dim=-1)
                total = sum(float(lp[j - 1, full_ids[j]]) for j in range(len(ctx_ids), len(full_ids)))
                scores.append(total)
                norms.append(total / n_choice)
            if max(range(len(scores)), key=lambda i: scores[i]) == it["gold"]:
                correct += 1
            if max(range(len(norms)), key=lambda i: norms[i]) == it["gold"]:
                correct_norm += 1
    n = len(items)
    return correct / n, correct_norm / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="arc_easy", choices=["arc_easy", "piqa", "sciq"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--depths", default="1,2,4")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--slm", default=None, help="optional HF causal LM baseline, e.g. gpt2 / EleutherAI/pythia-160m")
    ap.add_argument("--skip-ebt", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    print(f"[device] {device} | dataset {args.dataset} | limit {args.limit}")
    items = load_mcq(args.dataset, args.limit)
    n_choices = len(items[0]["choices"])
    print(f"[data] {len(items)} questions, {n_choices} choices -> random = {100.0/n_choices:.1f}%\n")

    results = {}
    if not args.skip_ebt:
        config = load_config(args.config)
        model, _ = build_model(config, device=device, execution_mode="inference")
        raw_model = get_uncompiled_model(model)
        step, _, _ = load_checkpoint(args.checkpoint, raw_model, strict=True)
        tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer, clean_up_tokenization_spaces=False)
        print(f"[ebt] loaded {args.checkpoint} (step {step})")
        for d in depths:
            acc, acc_norm = eval_ebt(items, raw_model, tokenizer, device, d)
            results[f"EBT depth={d}"] = (acc, acc_norm)
            print(f"  >> EBT depth={d}: acc={acc*100:.1f}%  acc_norm={acc_norm*100:.1f}%", flush=True)

    if args.slm:
        acc, acc_norm = eval_slm(items, args.slm, device)
        results[f"SLM {args.slm}"] = (acc, acc_norm)
        print(f"  >> SLM {args.slm}: acc={acc*100:.1f}%  acc_norm={acc_norm*100:.1f}%", flush=True)

    print("\n================ SUMMARY ================")
    print(f"{'model':24} {'acc':>7} {'acc_norm':>9}")
    for name, (a, an) in results.items():
        print(f"{name:24} {a*100:6.1f}% {an*100:8.1f}%")
    print(f"random baseline:         {100.0/n_choices:6.1f}%")


if __name__ == "__main__":
    main()
