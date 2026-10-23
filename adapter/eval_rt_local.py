#!/usr/bin/env python3
"""Score an RNNT checkpoint on the 682 real utterances, in-process, with per-turn latency.

    python adapter/eval_rt_local.py --model typhoon-ai/typhoon-asr-realtime --tag rt_base_local
    python adapter/eval_rt_local.py --model checkpoints/rt_student_s1.nemo  --tag rt_student_s1

The baseline and every student go through THIS script, so the only thing that differs between
two rows is the weights. Audio is quantised to PCM16 first, because that is what the `:8007` wire
does and it changes 3.4% of transcripts (CLAUDE.md). Greedy batch-1 decoding, the loop's shipped
strategy; first 5 clips are warm-up and excluded from latency. Score with `eval_canon.py`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

BENCH = Path("E:/asr-bench/thai-contextasr-bench")
RES = Path(__file__).resolve().parent.parent / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    # 2026-10-11 (loop iteration 7): optional beam + n-gram shallow fusion, wired as asr/server.py does.
    ap.add_argument("--strategy", default="greedy_batch")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--lm", default=None, help="ARPA / .nemo n-gram over the model's tokens")
    ap.add_argument("--lm-alpha", type=float, default=0.0)
    # 2026-10-11 (loop iteration 13): prompt-conditioned models (typhoon-asr-streaming-nemotron-0.6b) need a
    # language prompt; cache-aware streaming encoders take a right-context choice ([left, right] frames)
    ap.add_argument("--prompt", default=None, help='e.g. "th-TH"')
    ap.add_argument("--att-context", default=None, help='e.g. "56,13"')
    args = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    if args.model.endswith(".nemo"):
        m = nemo_asr.models.ASRModel.restore_from(args.model, map_location="cuda")
    else:
        m = nemo_asr.models.ASRModel.from_pretrained(args.model, map_location="cuda")
    m.eval()
    if args.att_context:
        m.encoder.set_default_att_context_size([int(x) for x in args.att_context.split(",")])
    tkw = {}
    if args.prompt:
        m.set_inference_prompt(args.prompt)
        m.decoding.set_strip_lang_tags(True)
        tkw = {"target_lang": args.prompt}
    if args.strategy != "greedy_batch" or args.lm:
        import copy
        from omegaconf import open_dict
        cfg = copy.deepcopy(m.cfg.decoding)
        with open_dict(cfg):
            cfg.strategy = args.strategy
            sub = "greedy" if args.strategy in ("greedy", "greedy_batch") else "beam"
            if sub == "greedy":
                cfg.greedy.use_cuda_graph_decoder = not args.lm      # no Triton on Windows: graphs + LM die
            else:
                cfg.beam.allow_cuda_graphs = not args.lm
                cfg.beam.beam_size = args.beam
                cfg.beam.return_best_hypothesis = True              # the checkpoint ships false (server.py)
            if args.lm:
                cfg[sub].ngram_lm_model = args.lm
                cfg[sub].ngram_lm_alpha = args.lm_alpha
            # the saved checkpoint config itself carries enum NAMES ("LATE", "LM_WEIGHTED_FULL"); NeMo's
            # enum constructors take the lowercase values, and fail only once an LM is attached
            for k, v in list(cfg[sub].items()):
                if isinstance(v, str) and v.isupper() and "_" in v or (isinstance(v, str) and v in ("LATE", "EARLY")):
                    cfg[sub][k] = v.lower()
        m.change_decoding_strategy(cfg)
    rows = sorted((json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")),
                  key=lambda r: r["utt_id"])
    rows = [r for r in rows if r["split"] == "real"]
    lat, aud, out = [], 0.0, []
    for i, r in enumerate(rows):
        y, sr = sf.read(BENCH / r["wav"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = (np.clip(np.round(y * 32767), -32768, 32767) / 32768).astype(np.float32)   # PCM16 wire
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            h = m.transcribe([y], verbose=False, batch_size=1, **tkw)[0]
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        out.append({"utt_id": r["utt_id"], "condition": "none", "text": (h.text if hasattr(h, "text") else str(h)).strip()})
        if i >= 5:
            lat.append(dt); aud += len(y) / sr
    with (RES / f"{args.tag}_hyps.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    lat.sort()
    run = {"model": args.model, "strategy": args.strategy, "beam": args.beam, "lm": args.lm,
           "lm_alpha": args.lm_alpha, "prompt": args.prompt, "att_context": args.att_context, "n": len(rows), "latency_ms_p50": round(1000 * statistics.median(lat), 1),
           "latency_ms_p90": round(1000 * lat[int(0.9 * len(lat))], 1), "rtf": round(sum(lat) / aud, 4)}
    (RES / f"{args.tag}_run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(json.dumps(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
