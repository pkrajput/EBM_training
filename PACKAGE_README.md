# EBT 191M — Energy-Based Transformer (test + reproduce package)

Self-contained snapshot to **load, test, and reproduce** a 191M-parameter
Energy-Based Transformer (EBT) language model trained on NVIDIA ClimbMix and
SFT'd on a code-heavy mixture.

> TL;DR honest results: the EBT *mechanism* works (perplexity ~7.5, and a
> measurable System-2 "thinking-gain": more MCMC steps → higher CORE), but at
> 191M params / 1.84B tokens the model is small and undertrained, so MBPP and
> HumanEval are 0% and free-form answers are weak. See "Results" below.

## What's in here
```
src/energy_coding/      core training/eval/inference code
scripts/                chat_ebt.py (manual test), evaluate_checkpoint.py, train_*.py
EBT/                    the upstream Energy-Based Transformer model code (required to load)
nanochat/               Karpathy's nanochat (required for CORE/HumanEval eval)
configs/                training/eval configs (active: ebt_191m_climbmix_8xb200.json)
run/                    launch scripts used for the GPU run (reproduction reference)
models/ebt_191m/        the trained checkpoints + metrics + logs (see its own README)
requirements_local.txt  CPU/MPS deps to test locally
requirements_blackwell.txt / requirements.txt  GPU deps to reproduce training
```

## 1. Setup (local CPU/MPS test)
```bash
python3 -m venv .venv && source .venv/bin/activate     # Python 3.11–3.13
pip install -r requirements_local.txt
```

## 2. Manually test the model
```bash
python scripts/chat_ebt.py \
  -q "Explain what a binary search is." \
  -d 2 \           # energy/MCMC "thinking" depth (1=shallow, higher=more refinement)
  -n 64 \          # output length (tokens)
  --device cpu     # or mps (Apple Silicon) / cuda
```
- Defaults load `models/ebt_191m/ebt_191m_sft_best.pt`.
- Try `-d 1` vs `-d 4` on the same prompt to see the System-2 thinking-depth effect.
- It prints the completion + the mean final-step **energy** (lower = more confident).
- `--raw` sends the prompt verbatim (skips the `User:/[[Answer]]:` SFT template);
  useful for plain completion tests on the base model.

Test the pretrain base (often better at plain factual completion than the
code-heavy SFT model):
```bash
python scripts/chat_ebt.py --raw -q "The capital of France is" -d 2 -n 8 \
  --checkpoint models/ebt_191m/ebt_191m_pretrain_base_step7000.pt
```

## 3. Reproduce the eval numbers
```bash
PYTHONPATH=src python scripts/evaluate_checkpoint.py \
  --config configs/ebt_191m_climbmix_8xb200.json \
  --checkpoint models/ebt_191m/ebt_191m_sft_best.pt
# (add --skip-core / --skip-mbpp / --skip-humaneval to taste)
```

## 4. Reproduce training (GPU)
On an 8×B200 (Blackwell) box:
```bash
bash run/setup_gpu_blackwell.sh          # venv + deps + clones EBT/nanochat + CORE bundle
bash run/prepare_climbmix.sh             # downloads ClimbMix shards
bash run/train_full_pipeline_8xb200.sh   # pretrain -> SFT -> cheap eval
```
Key knobs are in `configs/ebt_191m_climbmix_8xb200.json` (model size, MCMC depth,
LR, grad-spike guard, token target).

## Results (final, honest)
| Metric | Value |
|---|---|
| Params | 191M (12L × 16H × 1024D) |
| Pretrain tokens | 1.84B (ClimbMix) |
| SFT | code-heavy mixture, best ckpt at step 400 |
| Pretrain val ppl | 28.6 ; SFT val ppl | **7.5** |
| CORE | 0.076 (depth-1: 0.050 → depth-2: 0.076 → **+0.026 thinking-gain**) |
| MBPP pass@1 | **0/10 (0%)** |
| HumanEval pass@1 | **0/10 (0%)** |

The thinking-gain (CORE rising with MCMC depth) is the EBT paper's central claim,
reproduced here. Code/QA performance is limited by scale: ~191M params and ~1.84B
tokens (Chinchilla-optimal is ~4B; comparable coherent small LMs see 10–300B).

## Notes
- Checkpoints are PyTorch `.pt` files (~2.3 GB each).
- EBT inference uses gradient-based MCMC, so generation needs `torch.enable_grad`
  (handled by the scripts) and is slower than a normal LM, especially on CPU.
