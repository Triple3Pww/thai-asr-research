#!/usr/bin/env python3
"""Interpolate two checkpoints of the same model: w = (1 - a) * base + a * tuned.

    python adapter/avg_nemo.py --base E:/asr-spike/models/pathumma-...nemo \\
        --tuned checkpoints/rt_student_lot1.nemo --alpha 0.5 --out checkpoints/lot1_a50.nemo

WHY (2026-10-06). Fine-tuning a 10k-hour model on 20 h of one domain buys that domain and costs the
rest. Linear interpolation of a fine-tuned model with its own base cut forgetting to 3.4% in
Plantinga et al. (SLT 2024, "Parameter Averaging Is All You Need To Prevent Forgetting"); it costs
no training, so every fine-tune here is reported at several alphas, alpha = 1 being the raw tune.
Only floating-point tensors are mixed; integer buffers (step counters) are taken from `tuned`.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    # 2026-10-11 (loop iteration 9): a separate alpha for everything outside the encoder (prediction net +
    # joint). Higher alpha helps conversation and hurts read speech; this asks which half carries which.
    ap.add_argument("--alpha-dec", type=float, default=None)
    args = ap.parse_args()
    import torch
    import nemo.collections.asr as nemo_asr
    b = nemo_asr.models.ASRModel.restore_from(args.base, map_location="cpu").state_dict()
    t = nemo_asr.models.ASRModel.restore_from(args.tuned, map_location="cpu")
    sd = t.state_dict()
    assert sd.keys() == b.keys(), "not the same architecture"
    def a(k):
        return args.alpha if args.alpha_dec is None or k.startswith(("encoder.", "preprocessor.")) else args.alpha_dec
    mixed = {k: ((1 - a(k)) * b[k].float() + a(k) * v.float()).to(v.dtype)
             if torch.is_floating_point(v) else v for k, v in sd.items()}
    t.load_state_dict(mixed)
    t.save_to(args.out)
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
