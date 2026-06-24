# EBT 191M — trained artifacts (8×B200 run, Jun 2026)

Local snapshot of the 191M Energy-Based Transformer run so it can be tested
offline. Safe to shut down the cloud instance once this folder is complete.

## Files

| File | What it is |
|---|---|
| `ebt_191m_pretrain_base_step7000.pt` | Pretrain base — step 7000, **1.84B ClimbMix tokens**, val_loss 3.35 / ppl 28.6, CORE 0.050 (thinking-gain +0.017) |
| `ebt_191m_sft_best.pt` | **Best SFT checkpoint** (step 400, code-heavy mixture), val_loss 2.02 / ppl 7.5, CORE 0.076 (thinking-gain +0.026) |
| `ebt_191m_climbmix_8xb200.json` | The exact config used (architecture + MCMC + optim) |
| `pretrain_train_metrics.jsonl` / `pretrain_eval_metrics.jsonl` | Pretrain loss / energy-gap / CORE curves |
| `sft_train_metrics.jsonl` / `sft_eval_metrics.jsonl` | SFT loss curves + final eval |
| `pipeline.log`, `sft.log` | Full run logs (incl. the collapse + kill-switch events) |

## Model
- 12 layers × 16 heads × 1024 dim, vocab 50,257 (gpt-neox-20b tokenizer) ≈ **191M params**
- EBT: k=2 MCMC steps, alpha=0.25 (pinned), norm_pred, scale_alpha_with_energy, no_mcmc_detach (full Hessian)

## Final results (cheap 10-question eval on the SFT model)
- **MBPP pass@1: 0/10 (0%)**, **HumanEval pass@1: 0/10 (0%)** — 191M is below the scale needed for working code generation.
- The model works as a language model (ppl 7.5) and **reproduces the EBT System-2 thinking-gain** (CORE at depth-2 > depth-1, consistently).

## How to test locally
The EBT model needs the heavy deps (torch, pytorch_lightning, transformers, diffusers,
torchvision) plus the `EBT/` and `nanochat/` folders present in the repo root.

```bash
# from the repo root (energy-coding/)
source .venv/bin/activate            # a CUDA-or-CPU venv with the deps installed
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python scripts/evaluate_checkpoint.py \
  --config models/ebt_191m/ebt_191m_climbmix_8xb200.json \
  --checkpoint models/ebt_191m/ebt_191m_sft_best.pt \
  --skip-core                         # add flags to taste; MBPP/HumanEval need the eval deps
```

Note: checkpoints (`*.pt`) are git-ignored by the repo `.gitignore`, so they stay local.
