import sys, os
which = sys.argv[1] if len(sys.argv) > 1 else "fp32"
sys.path.insert(0, os.path.dirname(__file__))
import bigvocab as B
dev="mps"
if which=="fp32":
    B.run("BIGVOCAB_fp32_lr6e-4", 50257, "fp32", 6e-4, 120, dev)
elif which=="bf16low":
    B.run("BIGVOCAB_bf16_lr1e-4", 50257, "bf16", 1e-4, 120, dev)
