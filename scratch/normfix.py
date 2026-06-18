import sys, os, math
variant = sys.argv[1] if len(sys.argv) > 1 else "firststep"
sys.path.insert(0, os.path.dirname(__file__))
import ebt_repro as R, torch
def run(name, over, vocab=50257, lr=6e-4, steps=80, dev="mps", gc=0.5, batch=8, seq=32):
    torch.manual_seed(0)
    perm = torch.randperm(vocab, device=dev)
    cfg = R.base_cfg(embedding_dim=512, num_transformer_blocks=6, multiheaded_attention_heads=8, max_seq_len=seq*2+4, scale_alpha_with_energy=True, **over)
    model = R.EBT_NLP(cfg, vocab).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95), weight_decay=0.1)
    uni = math.log(vocab); ac = torch.autocast("mps", dtype=torch.bfloat16)
    print(f"=== {name} | uniform={uni:.3f} ===", flush=True)
    for step in range(steps):
        x = R.make_batch(perm, batch, seq, vocab, dev)
        with ac: loss, aux = model.loss(x, learning=True)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), gc); opt.step()
        if step%10==0 or step==steps-1:
            below = "  <-- BELOW UNIFORM (learning!)" if float(loss) < uni-0.3 else ""
            print(f"  step {step:3d} | loss {float(loss):7.3f} | gn {float(gn):7.1f}{below}", flush=True)
if variant=="firststep":
    run("norm_FIRST_step_only", dict(normalize_initial_condition=True, normalize_initial_condition_only_first_step=True))
elif variant=="off":
    run("norm_init_OFF", dict(normalize_initial_condition=False))
