#!/usr/bin/env python3
"""N-best lists from typhoon-asr-realtime (NeMo RNNT, `malsd_batch` beam) -- same format as nbest_fw.py.

    python adapter/nbest_rt.py --n 5 --tag nbest_rt5

Part of the N-best picker line (2026-10-04, his idea: the ASR returns its top possibilities and an
LLM picks with context). turbo's own top-5 has a ceiling of only 0.0717 -- the candidates are near-
copies of one hearing. A second, architecturally different model hears differently; its list is
what adds diversity. Audio is PCM16-quantised like the `:8007` wire. Batch 1, warm-up excluded.
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
from omegaconf import open_dict

BENCH = Path("E:/asr-bench/thai-contextasr-bench")
RES = Path(__file__).resolve().parent.parent / "results"


def load_rows(args, outp: Path):
    """The 682 real benchmark utterances, or any {utt_id, wav} manifest (e.g. Porjai, for picker
    training). Output is streamed per row, and a manifest run resumes from what is already written."""
    if args.manifest:
        rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    else:
        rows = sorted((json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")),
                      key=lambda r: r["utt_id"])
        rows = [dict(r, wav=str(BENCH / r["wav"])) for r in rows if r["split"] == "real"]
    done = set()
    if args.manifest and outp.exists():
        done = {json.loads(l)["utt_id"] for l in outp.open(encoding="utf-8") if l.strip()}
    return rows, done, outp.open("a" if done else "w", encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="typhoon-ai/typhoon-asr-realtime", help="any NeMo RNNT on the HF hub, e.g. nectec/Pathumma-stt-th-fastconformer-rnnt-large")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--manifest", default=None, help="any {utt_id, wav} jsonl (default: the 682 real utts)")
    args = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    if args.model.endswith(".nemo"):   # e.g. nectec: one .nemo whose name does not match the repo, so from_pretrained fails
        m = nemo_asr.models.ASRModel.restore_from(args.model, map_location="cuda").eval()
    else:
        m = nemo_asr.models.ASRModel.from_pretrained(args.model, map_location="cuda").eval()
    cfg = m.cfg.decoding
    with open_dict(cfg):
        cfg.strategy = "malsd_batch"
        cfg.beam.beam_size = args.n
        cfg.beam.return_best_hypothesis = False
    m.change_decoding_strategy(cfg)

    rows, done, fout = load_rows(args, RES / f"{args.tag}_nbest.jsonl")
    lat, out = [], []
    for i, r in enumerate(rows):
        if r["utt_id"] in done:
            continue
        y, sr = sf.read(r["wav"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = (np.clip(np.round(y * 32767), -32768, 32767) / 32768).astype(np.float32)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            res = m.transcribe([y], verbose=False, batch_size=1)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        nb = res[0]
        hyps = getattr(nb, "n_best_hypotheses", None) or (nb if isinstance(nb, list) else [nb])
        cands, seen = [], set()
        for h in sorted(hyps, key=lambda h: -float(h.score)):
            t = (h.text if hasattr(h, "text") else str(h)).strip()
            if t in seen:
                continue
            seen.add(t)
            cands.append({"text": t, "score": round(float(h.score), 4)})
        out.append({"utt_id": r["utt_id"], "cands": cands[: args.n]})
        fout.write(json.dumps(out[-1], ensure_ascii=False) + "\n"); fout.flush()
        if i >= 5:
            lat.append(dt)
    fout.close()
    lat.sort()
    run = {"model": args.model, "decoding": f"malsd_batch beam {args.n}, n-best",
           "utts": len(rows), "mean_cands": round(sum(len(o['cands']) for o in out) / max(len(out), 1), 2),
           "latency_ms_p50": round(1000 * statistics.median(lat), 1) if lat else None,
           "latency_ms_p90": round(1000 * lat[int(0.9 * len(lat))], 1) if lat else None}
    (RES / f"{args.tag}_run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(json.dumps(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
