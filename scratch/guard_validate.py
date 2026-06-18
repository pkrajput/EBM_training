import sys, os, math, statistics; sys.argv=["x"]
sys.path.insert(0, os.path.dirname(__file__))
import ebt_repro as R, torch
torch.manual_seed(0); dev="mps"; vocab=4000
perm = torch.randperm(vocab, device=dev)
cfg = R.base_cfg(embedding_dim=512, num_transformer_blocks=12, multiheaded_attention_heads=8, max_seq_len=32, scale_alpha_with_energy=True)
model = R.EBT_NLP(cfg, vocab).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9,0.95), weight_decay=0.1)
hist=[]; skips=0
print("=== guard validate (lr6e-4, spiky regime) ===", flush=True)
for step in range(300):
    x = R.make_batch(perm, 16, 16, vocab, dev)
    loss, aux = model.loss(x, learning=True)
    opt.zero_grad(set_to_none=True); loss.backward()
    tn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5))
    skip=False
    if not math.isfinite(tn): skip=True
    elif len(hist)>=30:
        med=sorted(hist)[len(hist)//2]
        if tn > 8.0*max(med,1e-6): skip=True
    if skip: skips+=1; opt.zero_grad(set_to_none=True)
    else:
        if math.isfinite(tn): hist.append(tn)
        if len(hist)>100: hist.pop(0)
        opt.step()
    if step%40==0 or step==299:
        print(f"  step {step:3d} | loss {float(loss):.3f} | gradnorm {tn:8.1f} | skips {skips}", flush=True)
print("DONE", flush=True)
