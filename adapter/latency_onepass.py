#!/usr/bin/env python3
"""Per-turn latency of NECTEC + the CTC-head name splice in ONE encoder pass, vs NECTEC alone.

    PYTHONUTF8=1 python adapter/latency_onepass.py --head checkpoints/ctchead_h1.pt --condition bias@50

The accuracy arm (`arm_d6_splice.py --spotter head`) runs the frozen encoder twice for convenience;
its outputs are identical to one pass because the encoder is deterministic in eval mode. This script
times what a server would do: preprocess + encoder ONCE, then (a) RNN-T greedy decode and (b) the head
+ keyword spotting on the same encoder output. Batch 1, PCM16, first 5 clips warm-up, 682 real utts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arm_d6_splice import BENCH, HeadSpotter, load_lists, pcm16_roundtrip  # noqa: E402

NEC = "E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--condition", default="bias@50")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--fast-spot", action="store_true")
    args = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.restore_from(NEC, map_location="cuda")
    m.eval()
    sp = HeadSpotter(m, args.head)
    sp.fast = args.fast_spot
    lists = load_lists(args.condition)
    rows = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8") if l.strip()]
    rows = sorted((r for r in rows if r["split"] == "real"), key=lambda r: r["utt_id"])
    t_base, t_one, t_spot, aud = [], [], [], 0.0
    for i, r in enumerate(rows):
        y, _ = sf.read(BENCH / r["wav"], dtype="float32")
        y = pcm16_roundtrip(y.mean(axis=1) if y.ndim > 1 else y)
        # (1) NECTEC alone, the shipped path
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            m.transcribe([y], verbose=False, batch_size=1)
        torch.cuda.synchronize(); tb = time.perf_counter() - t0
        # (2) one encoder pass feeding both decoders. transcribe() above leaves the ENCODER in training
        # mode (NeMo 3.0.0); without this, dropout runs and BatchNorm stats drift (see arm_d6_splice.py).
        m.eval()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            x = torch.as_tensor(y, device="cuda")[None]
            ln = torch.tensor([x.shape[1]], device="cuda")
            f, fl = m.preprocessor(input_signal=x, length=ln)
            enc, el = m.encoder(audio_signal=f, length=fl)
            m.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=el)
            lp = torch.log_softmax(sp.head(enc[0, :, : int(el[0])].T.float()), -1).cpu().numpy()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        kws = lists.get(r["utt_id"], [])
        if kws:
            sp.spots(lp, kws, args.tau, 6)
        t2 = time.perf_counter()
        if i >= 5:
            t_base.append(tb); t_one.append(t2 - t0); t_spot.append(t2 - t1); aud += len(y) / 16000
    ms = lambda v: round(1000 * statistics.median(v), 1)
    p90 = lambda v: round(1000 * sorted(v)[int(0.9 * len(v))], 1)
    out = {"n": len(t_base), "mean_turn_s": round(aud / len(t_base), 2),
           "nectec_alone_ms_p50": ms(t_base), "nectec_alone_ms_p90": p90(t_base),
           "onepass_with_splice_ms_p50": ms(t_one), "onepass_with_splice_ms_p90": p90(t_one),
           "of_which_spotting_ms_p50": ms(t_spot), "condition": args.condition}
    tag = Path(args.head).stem + ("_fast" if args.fast_spot else "")
    out["fast_spot"] = args.fast_spot
    (HERE.parent / "results" / f"latency_onepass_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
