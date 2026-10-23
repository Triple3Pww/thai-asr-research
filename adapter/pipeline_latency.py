#!/usr/bin/env python3
"""End-to-end wall time of the 3-ear system: NECTEC + Pathumma-whisper-large + typhoon-whisper-large, ROVER.

    PYTHONUTF8=1 python adapter/pipeline_latency.py --limit 682 --tag pipe3

Deployed shape, measured as one turn = audio in -> final text out:
  * NECTEC (main process) transcribes first (~43 ms) -- its text is the DRAFT for both Whispers
  * the two large Whispers run CONCURRENTLY in two worker processes, each speculative-decoding against
    NECTEC's draft (`spec_whisper.Spec.spec_cached`, lossless w.r.t. its own greedy output)
  * ROVER (`pilot_rover.rover`, uniform, NECTEC + the two Whispers) on the CPU
All three models stay resident on one 16 GB GPU. Batch 1, PCM16 audio, first 5 turns are warm-up.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BENCH = Path("E:/asr-bench/thai-contextasr-bench")


def worker(name: str, q_in, q_out):
    sys.path.insert(0, str(HERE))
    import torch
    from spec_whisper import Spec
    S = Spec(name)
    q_out.put(("ready", name))
    while True:
        job = q_in.get()
        if job is None:
            break
        y, draft = job
        with torch.no_grad():
            text, _ = S.spec_cached(S.encode(S.features(y)), draft)
        torch.cuda.synchronize()
        q_out.put((name, text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    import numpy as np
    import soundfile as sf
    import torch
    from ctc_head import load_nec, pcm16
    from pilot_rover import join, rover, toks

    ctx = mp.get_context("spawn")
    names = ["nectec/Pathumma-whisper-th-large-v3", "typhoon-ai/typhoon-whisper-large-v3"]
    qs = {n: ctx.Queue() for n in names}
    q_out = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(n, qs[n], q_out), daemon=True) for n in names]
    for p in procs:
        p.start()
    nec = load_nec()
    for _ in names:
        print(q_out.get(), flush=True)

    rows = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["split"] == "real"][: a.limit or None]
    out, lat = {}, []
    for i, r in enumerate(rows):
        y, _ = sf.read(BENCH / r["wav"], dtype="float32")
        y = pcm16(y.mean(axis=1) if y.ndim > 1 else y)
        t0 = time.perf_counter()
        nec.eval()
        h = nec.transcribe([y], verbose=False, batch_size=1)
        h = h[0] if h else ""
        h = h[0] if isinstance(h, (list, tuple)) else h
        draft = (h if isinstance(h, str) else str(getattr(h, "text", ""))).strip()
        for n in names:
            qs[n].put((y, draft))
        got = {"nec": draft}
        for _ in names:
            n, t = q_out.get()
            got[n] = t
        pool = ["nec"] + names
        merged, _ = rover({k: toks(got[k]) for k in pool}, {k: 1.0 for k in pool}, pool)
        out[r["utt_id"]] = join(merged)
        t1 = time.perf_counter()
        if i >= 5:
            lat.append(1000 * (t1 - t0))
        if i % 100 == 0:
            print(i, len(rows), f"p50 so far {np.median(lat) if lat else 0:.0f} ms", flush=True)
    for n in names:
        qs[n].put(None)
    rep = {"system": "NECTEC draft -> {Pathumma-L, typhoon-L} speculative, concurrent -> ROVER",
           "n": len(out), "turn_ms_p50": round(float(np.median(lat)), 1),
           "turn_ms_p90": round(float(np.percentile(lat, 90)), 1),
           "turn_ms_mean": round(float(np.mean(lat)), 1)}
    print(rep)
    with (ROOT / "results" / f"{a.tag}_hyps.jsonl").open("w", encoding="utf-8", newline="\n") as g:
        for u, t in out.items():
            g.write(json.dumps({"utt_id": u, "condition": "none", "text": t}, ensure_ascii=False) + "\n")
    (ROOT / "results" / f"{a.tag}_run.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
