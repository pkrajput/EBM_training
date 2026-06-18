import sys, os, math; sys.argv=["x"]
sys.path.insert(0, os.path.dirname(__file__))
import ebt_repro as R, torch
def run_long(name, cfg, lr, vocab, steps, dev, gc=0.5, batch=16, seq=16):
    torch.manual_seed(0)
    perm = torch.randperm(vocab, device=dev)
    model = R.EBT_NLP(cfg, vocab).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95), weight_decay=0.1)
    uni = math.log(vocab); print(f"=== {name} | scale_alpha={cfg['scale_alpha_with_energy']} lr={lr} | uniform={uni:.3f} ===", flush=True)
    collapsed_at = None
    for step in range(steps):
        x = R.make_batch(perm, batch, seq, vocab, dev)
        loss, aux = model.loss(x, learning=True)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), gc); opt.step()
        if abs(float(loss)-uni) < 0.05 and step > 30 and collapsed_at is None:
            collapsed_at = step
        if step % 40 == 0 or step==steps-1 or (collapsed_at and step<collapsed_at+5):
            print(f"  step {step:4d} | loss {float(loss):6.3f} | gap {aux['energy_gap']:7.2f} | e_absmax {aux['e_absmax']:8.2f} | e_absmean {aux['e_absmean']:7.2f} | gn {float(gn):7.1f}", flush=True)
        if collapsed_at and step > collapsed_at + 10:
            print(f"  >>> COLLAPSED at step {collapsed_at}", flush=True); return collapsed_at
    print(f"  >>> no collapse in {steps} steps (final loss {float(loss):.3f})", flush=True)
    return None
dev="mps"; vocab=4000
# deeper model, higher LR, LONG run to try to trigger the real collapse
cfgT = R.base_cfg(embedding_dim=512, num_transformer_blocks=12, multiheaded_attention_heads=8, max_seq_len=32, scale_alpha_with_energy=True)
run_long("SCALE_ALPHA_ON", cfgT, 6e-4, vocab, 1200, dev)
