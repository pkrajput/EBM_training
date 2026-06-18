import sys, os, math, statistics; sys.argv=["x"]
sys.path.insert(0, os.path.dirname(__file__))
import ebt_repro as R, torch
def run(name, cfg, lr, vocab, steps, dev, gc=0.5, skip_spike=False, spike_mult=4.0, spike_abs=50.0, batch=16, seq=16):
    torch.manual_seed(0)
    perm = torch.randperm(vocab, device=dev)
    model = R.EBT_NLP(cfg, vocab).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95), weight_decay=0.1)
    uni = math.log(vocab); ac = torch.autocast("mps", dtype=torch.bfloat16)
    print(f"=== {name} | bf16 skip_spike={skip_spike} | uniform={uni:.3f} ===", flush=True)
    recent=[]; skips=0; collapsed=None
    for step in range(steps):
        x = R.make_batch(perm, batch, seq, vocab, dev)
        with ac:
            loss, aux = model.loss(x, learning=True)
        opt.zero_grad(set_to_none=True); loss.backward()
        # pre-clip grad norm
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), gc)
        gnf = float(gn)
        do_skip = False
        if skip_spike and len(recent) >= 20:
            med = statistics.median(recent)
            if gnf > max(spike_abs, spike_mult*med):
                do_skip = True; skips += 1
        if not (do_skip or math.isnan(gnf) or math.isinf(gnf)):
            opt.step()
        recent.append(gnf if math.isfinite(gnf) else 0.0)
        if len(recent)>50: recent.pop(0)
        if abs(float(loss)-uni)<0.05 and step>30 and collapsed is None:
            collapsed=step
        if step%50==0 or step==steps-1 or (collapsed and step<collapsed+3):
            print(f"  step {step:4d} | loss {float(loss):6.3f} | gn {gnf:8.1f} | skips {skips}", flush=True)
        if collapsed and step>collapsed+8:
            print(f"  >>> COLLAPSED at {collapsed} (skips={skips})", flush=True); return collapsed
    print(f"  >>> NO collapse, final loss {float(loss):.3f} (skips={skips})", flush=True); return None
dev="mps"; vocab=4000
cfg = R.base_cfg(embedding_dim=512, num_transformer_blocks=12, multiheaded_attention_heads=8, max_seq_len=32, scale_alpha_with_energy=True)
import sys as _s
mode = _s.argv[1] if len(_s.argv)>1 else "both"
if mode in ("both","baseline"):
    run("BF16_BASELINE_noskip", cfg, 6e-4, vocab, 1200, dev, skip_spike=False)
if mode in ("both","skip"):
    run("BF16_SKIPSPIKE", cfg, 6e-4, vocab, 1200, dev, skip_spike=True)
