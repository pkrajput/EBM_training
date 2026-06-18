"""
Standalone, free (CPU/MPS) reproduction of the EBT-NLP training dynamics.

Goal: figure out WHY pretraining loss gets stuck near uniform (ln(vocab)),
without spending a cent on GPUs.

- Uses the EXACT EBTDefault energy-transformer + custom attention copied
  verbatim from EBT/model/ar_ebt_default.py (only the heavy diffusers/
  torchvision import chain is dropped).
- Implements the EXACT MCMC loss loop from EBT/model/nlp/ebt.py.
- Trains on a fully-learnable synthetic bigram task: next = perm[cur].
  Uniform loss = ln(V). A WORKING model must drive loss toward 0.
- Compares EBT configs + a standard-LM baseline (same backbone, softmax head)
  to isolate whether the EBT MCMC mechanism is what blocks learning.
"""
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


# ----------------------------------------------------------------------------
# helpers copied from model_utils.py
# ----------------------------------------------------------------------------
@dataclass
class EBTModelArgs:
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 64
    max_seq_len: int = 16
    weight_initialization: str = "xavier"
    weight_initialization_gain: float = 1.0


def init_whole_model_weights(model, weight_initialization_method, nonlinearity="linear", weight_initialization_gain=1.0):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if weight_initialization_method == "xavier":
                nn.init.xavier_normal_(m.weight)
                if weight_initialization_gain != 1.0:
                    m.weight.data *= weight_initialization_gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif weight_initialization_method == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    model.apply(init_weights)


# ----------------------------------------------------------------------------
# architecture copied verbatim from ar_ebt_default.py
# ----------------------------------------------------------------------------
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: EBTModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wq, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wk, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wv, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        init_whole_model_weights(self.wo, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)

    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, full_seqlen, _ = x.shape
        original_seqlen = full_seqlen // 2
        context_length = original_seqlen + 1
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, full_seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        xq_o = xq[:, :original_seqlen]
        xk_o = xk[:, :original_seqlen]
        xv_o = xv[:, :original_seqlen]
        xq_p = xq[:, original_seqlen:]
        xk_p = xk[:, original_seqlen:]
        xv_p = xv[:, original_seqlen:]
        xq_o, xk_o = apply_rotary_emb(xq_o, xk_o, freqs_cis=freqs_cis[:original_seqlen])
        xq_p, xk_p = apply_rotary_emb(xq_p, xk_p, freqs_cis=freqs_cis[1:context_length])
        xq_o = xq_o.transpose(1, 2)
        keys_o = xk_o.transpose(1, 2)
        values_o = xv_o.transpose(1, 2)
        scores_o = torch.matmul(xq_o, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores_o = scores_o + mask[:-1, :-1]
        scores_o = F.softmax(scores_o.float(), dim=-1).type_as(xq_o)
        output_o = torch.matmul(scores_o, values_o)
        output_o = output_o.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1)
        xq_p = xq_p.transpose(1, 2)
        keys_p = xk_p.transpose(1, 2)
        values_p = xv_p.transpose(1, 2)
        scores_p = torch.matmul(xq_p, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)
        temp_append = torch.zeros((scores_p.shape[0], scores_p.shape[1], scores_p.shape[2], 1), dtype=scores_p.dtype, device=scores_p.device)
        scores_p = torch.cat((scores_p, temp_append), dim=-1)
        insertion_superdiagonal = (xq_p * keys_p).sum(dim=3) / math.sqrt(self.head_dim)
        insertion_superdiagonal = insertion_superdiagonal.to(scores_p.dtype)
        superdiag_rows = torch.arange(scores_p.shape[2])
        superdiag_cols = torch.arange(1, scores_p.shape[3])
        zero_superdiag = torch.zeros_like(insertion_superdiagonal, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_removal_mask = torch.ones_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_removal_mask[:, :, superdiag_rows, superdiag_cols] = zero_superdiag
        scores_p = scores_p * diagonal_removal_mask
        diagonal_addition_mask = torch.zeros_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_addition_mask[:, :, superdiag_rows, superdiag_cols] = insertion_superdiagonal
        scores_p = scores_p + diagonal_addition_mask
        if mask is not None:
            scores_p = scores_p + mask[1:, :]
        scores_p = F.softmax(scores_p.float(), dim=-1).type_as(xq_p)
        scores_p_superdiagonal = scores_p.diagonal(offset=1, dim1=2, dim2=3).clone()
        scores_p = scores_p * diagonal_removal_mask
        scores_p = scores_p[:, :, :, :-1]
        output_p = torch.matmul(scores_p, values_o)
        next_pred_self_attention = values_p * scores_p_superdiagonal.unsqueeze(dim=-1)
        output_p = output_p + next_pred_self_attention
        output_p = output_p.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1)
        output = torch.cat((output_o, output_p), dim=1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_dim_multiplier, weight_initialization, weight_initialization_gain):
        super().__init__()
        hidden_dim = dim if ffn_dim_multiplier is None else int(dim * ffn_dim_multiplier)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        init_whole_model_weights(self.w1, weight_initialization, weight_initialization_gain=weight_initialization_gain)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        init_whole_model_weights(self.w2, weight_initialization, weight_initialization_gain=weight_initialization_gain)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        init_whole_model_weights(self.w3, weight_initialization, weight_initialization_gain=weight_initialization_gain)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args.dim, args.ffn_dim_multiplier, args.weight_initialization, args.weight_initialization_gain)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h))


class EBTDefault(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers
        self.layers = torch.nn.ModuleList([TransformerBlock(i, params) for i in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.freqs_cis = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len)
        self.final_layer = nn.Linear(params.dim, 1, bias=False)
        init_whole_model_weights(self.final_layer, params.weight_initialization)

    def forward(self, embeddings, start_pos=0, mcmc_step=None):
        _bsz, seqlen = embeddings.shape[:2]
        seqlen = (seqlen + 2) // 2
        self.freqs_cis = self.freqs_cis.to(embeddings.device)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        mask = torch.full((seqlen, seqlen), float("-inf"), device=embeddings.device)
        mask = torch.triu(mask, diagonal=1)
        mask = torch.hstack([torch.zeros((seqlen, start_pos), device=embeddings.device), mask]).type_as(embeddings)
        for layer in self.layers:
            embeddings = layer(embeddings, start_pos, freqs_cis, mask)
        embeddings = self.norm(embeddings)
        energies = self.final_layer(embeddings)
        energies = energies[:, embeddings.shape[1] // 2:]
        return energies


# ----------------------------------------------------------------------------
# EBT-NLP model: faithful port of nlp/ebt.py forward + forward_loss_wrapper
# ----------------------------------------------------------------------------
class EBT_NLP(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.alpha = nn.Parameter(torch.tensor(float(cfg["mcmc_step_size"])), requires_grad=cfg["mcmc_step_size_learnable"])
        self.langevin_dynamics_noise_std = nn.Parameter(torch.tensor(float(cfg["langevin_dynamics_noise"])), requires_grad=False)
        self.embeddings = nn.Embedding(vocab_size, cfg["embedding_dim"])
        init_whole_model_weights(self.embeddings, "xavier")
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.vocab_to_embed = nn.Linear(vocab_size, cfg["embedding_dim"], bias=False)
        init_whole_model_weights(self.vocab_to_embed, "xavier")
        args = EBTModelArgs(dim=cfg["embedding_dim"], n_layers=cfg["num_transformer_blocks"],
                            n_heads=cfg["multiheaded_attention_heads"], max_seq_len=cfg["max_seq_len"] + 2,
                            ffn_dim_multiplier=cfg.get("ffn_dim_multiplier"))
        self.transformer = EBTDefault(args)
        if cfg["norm_pred"]:
            self.pred_norm = nn.RMSNorm(vocab_size)

    def forward(self, x, learning=True):
        c = self.cfg
        real_embeddings_input = self.embeddings(x)
        batch_size, seq_length = x.shape
        alpha = torch.clamp(self.alpha, min=0.0001)
        ld_std = torch.clamp(self.langevin_dynamics_noise_std, min=1e-6)
        predicted_tokens = torch.randn(batch_size, seq_length, self.vocab_size, device=x.device) * c["gaussian_random_noise_scaling"]
        k = c["mcmc_num_steps"]
        mcmc_steps = list(range(k))
        predicted_distributions, predicted_energies = [], []
        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(mcmc_steps):
                if c["no_mcmc_detach"]:
                    predicted_tokens.requires_grad_()
                else:
                    predicted_tokens = predicted_tokens.detach().requires_grad_()
                if c["langevin_dynamics_noise"] != 0:
                    predicted_tokens = predicted_tokens + torch.randn_like(predicted_tokens.detach()) * ld_std
                if c["normalize_initial_condition"]:
                    if c["normalize_initial_condition_only_first_step"]:
                        if mcmc_step == 0:
                            predicted_tokens = self.softmax(predicted_tokens)
                    else:
                        predicted_tokens = self.softmax(predicted_tokens)
                    predicted_embeddings = self.vocab_to_embed(predicted_tokens)
                else:
                    predicted_embeddings = self.vocab_to_embed(predicted_tokens)
                all_embeddings = torch.cat((real_embeddings_input, predicted_embeddings), dim=1)
                energy_preds = self.transformer(all_embeddings, start_pos=0)
                energy_preds = energy_preds.reshape(-1, 1)
                predicted_energies.append(energy_preds)
                grad = torch.autograd.grad([energy_preds.sum()], [predicted_tokens], create_graph=learning)[0]
                if c["scale_alpha_with_energy"]:
                    exped = torch.exp(energy_preds.reshape(batch_size, seq_length, 1) / c["scale_alpha_with_energy_temp"])
                    predicted_tokens = predicted_tokens - alpha * exped * grad
                else:
                    predicted_tokens = predicted_tokens - alpha * grad
                if c["norm_pred"] and not (c["norm_pred_not_final_step"] and i == (len(mcmc_steps) - 1)):
                    predicted_tokens = self.pred_norm(predicted_tokens)
                predicted_distributions.append(predicted_tokens)
        return predicted_distributions, predicted_energies

    def loss(self, x, learning=True):
        input_ids = x[:, :-1]
        targets = x[:, 1:].reshape(-1)
        preds, energies = self.forward(input_ids, learning=learning)
        total = len(preds)
        recon = 0.0
        for step, pred in enumerate(preds):
            logp = self.log_softmax(pred).reshape(-1, self.vocab_size)
            cce = F.nll_loss(logp, targets)
            recon = recon + cce
            if step == 0:
                initial_loss = cce.detach()
                init_e = energies[0].mean().detach()
            if step == total - 1:
                final_loss = cce.detach()
                final_e = energies[-1].mean().detach()
        recon = recon / total
        gap = (init_e - final_e)
        # energy-magnitude diagnostics (suspected exp(energy) explosion driver)
        all_e = torch.cat([e.detach().reshape(-1) for e in energies])
        emax = float(all_e.abs().max())
        emean = float(all_e.abs().mean())
        return recon, dict(initial_loss=float(initial_loss), final_step_loss=float(final_loss),
                           energy_gap=float(gap), alpha=float(self.alpha.detach()),
                           e_absmax=emax, e_absmean=emean)


# ----------------------------------------------------------------------------
# standard LM baseline (same backbone-ish, direct softmax head)
# ----------------------------------------------------------------------------
class StdLM(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        d = cfg["embedding_dim"]
        self.embeddings = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.embeddings.weight, std=0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, cfg["multiheaded_attention_heads"], dim_feedforward=d * 4,
                                       batch_first=True, activation="gelu", norm_first=True)
            for _ in range(cfg["num_transformer_blocks"])
        ])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)

    def loss(self, x, learning=True):
        input_ids = x[:, :-1]
        targets = x[:, 1:].reshape(-1)
        h = self.embeddings(input_ids)
        S = input_ids.shape[1]
        cmask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
        for layer in self.layers:
            h = layer(h, src_mask=cmask)
        logits = self.head(self.norm(h))
        cce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets)
        return cce, dict(initial_loss=float(cce.detach()), final_step_loss=float(cce.detach()), energy_gap=0.0, alpha=0.0)


# ----------------------------------------------------------------------------
# synthetic fully-learnable task: next = perm[cur]   (uniform loss = ln(V))
# ----------------------------------------------------------------------------
def make_batch(perm, batch, seq, vocab, device):
    starts = torch.randint(0, vocab, (batch, 1), device=device)
    seqs = [starts]
    cur = starts
    for _ in range(seq):
        cur = perm[cur]
        seqs.append(cur)
    return torch.cat(seqs, dim=1)


def base_cfg(**over):
    cfg = dict(
        embedding_dim=128, num_transformer_blocks=2, multiheaded_attention_heads=4,
        max_seq_len=32, ffn_dim_multiplier=4.0,
        mcmc_num_steps=2, mcmc_step_size=0.5, mcmc_step_size_learnable=True,
        langevin_dynamics_noise=1.0, no_mcmc_detach=True,
        gaussian_random_noise_scaling=1.0,
        normalize_initial_condition=True, normalize_initial_condition_only_first_step=False,
        norm_pred=True, norm_pred_not_final_step=False,
        scale_alpha_with_energy=True, scale_alpha_with_energy_temp=1.0,
    )
    cfg.update(over)
    return cfg


def run(name, model_kind, cfg, steps=400, lr=6e-5, vocab=64, seq=16, batch=32, device="cpu", seed=0, precision="fp32", grad_clip=0.5):
    torch.manual_seed(seed)
    perm = torch.randperm(vocab, device=device)
    if model_kind == "ebt":
        model = EBT_NLP(cfg, vocab).to(device)
    else:
        model = StdLM(cfg, vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    uniform = math.log(vocab)
    autocast = (torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16)
                if precision == "bf16" else None)
    log = []
    for step in range(steps):
        x = make_batch(perm, batch, seq, vocab, device)
        if autocast is not None:
            with autocast:
                loss, aux = model.loss(x, learning=True)
        else:
            loss, aux = model.loss(x, learning=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            log.append((step, float(loss), aux))
    print(f"\n=== {name} (uniform={uniform:.3f}) ===")
    for step, l, aux in log:
        print(f"  step {step:4d} | loss {l:6.3f} | final_L {aux['final_step_loss']:6.3f} | "
              f"gap {aux['energy_gap']:7.3f} | alpha {aux['alpha']:.3f}")
    best = min(l for _, l, _ in log)
    verdict = "LEARNS" if best < uniform - 0.5 else "STUCK near uniform"
    print(f"  -> best {best:.3f}  [{verdict}]")
    return best, uniform


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    dev = args.device

    experiments = {
        "baseline_stdLM": ("std", base_cfg(), 3e-3),
        "ebt_CURRENT(softmax_every_step)": ("ebt", base_cfg(), 6e-5),
        "ebt_softmax_FIRST_step_only": ("ebt", base_cfg(normalize_initial_condition_only_first_step=True), 6e-5),
        "ebt_no_normalize_init": ("ebt", base_cfg(normalize_initial_condition=False), 6e-5),
        "ebt_no_norm_pred": ("ebt", base_cfg(norm_pred=False), 6e-5),
        "ebt_higher_LR_3e-4": ("ebt", base_cfg(), 3e-4),
        "ebt_more_steps_k5": ("ebt", base_cfg(mcmc_num_steps=5), 6e-5),
    }
    results = {}
    for nm, (kind, cfg, lr) in experiments.items():
        if args.only and args.only not in nm:
            continue
        best, uni = run(nm, kind, cfg, steps=args.steps, lr=lr, device=dev)
        results[nm] = best
    print("\n================ SUMMARY ================")
    for nm, best in results.items():
        print(f"  {nm:40s} best_loss={best:.3f}")
