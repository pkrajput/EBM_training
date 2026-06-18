import sys, os, math; sys.argv=["x"]
sys.path.insert(0, os.path.dirname(__file__))
import ebt_repro as R, torch
def run(name, vocab, prec, lr, steps, dev, gc=0.5, batch=8, seq=32):
    torch.manual_seed(0)
    perm = torch.randperm(vocab, device=dev)
    cfg = R.base_cfg(embedding_dim=512, num_transformer_blocks=6, multiheaded_attention_heads=8, max_seq_len=seq*2+4, scale_alpha_with_energy=True)
    model = R.EBT_NLP(cfg, vocab).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95), weight_decay=0.1)
    uni = math.log(vocab)
    ac = torch.autocast("mps", dtype=torch.bfloat16) if prec=="bf16" else None
    print(f"=== {name} vocab={vocab} prec={prec} lr={lr} uniform={uni:.3f} ===", flush=True)
    collapsed=None
    for step in range(steps):
        x = R.make_batch(perm, batch, seq, vocab, dev)
        if ac:
            with ac: loss, aux = model.loss(x, learning=True)
        else: loss, aux = model.loss(x, learning=True)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), gc); opt.step()
        if abs(float(loss)-uni)<0.05 and step>20 and collapsed is None: collapsed=step
        if step%30==0 or step==steps-1 or (collapsed and step<collapsed+3):
            print(f"  step {step:4d} | loss {float(loss):7.3f} | gn {float(gn):8.1f} | e_absmax {aux['e_absmax']:7.1f}", flush=True)
        if collapsed and step>collapsed+6:
            print(f"  >>> COLLAPSED at {collapsed}", flush=True); return collapsed
    print(f"  >>> no collapse, final {float(loss):.3f}", flush=True); return None
if __name__ == "__main__":
    dev="mps"
    run("BIGVOCAB_bf16", 50257, "bf16", 6e-4, 500, dev)
