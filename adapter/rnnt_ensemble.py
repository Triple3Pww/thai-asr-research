#!/usr/bin/env python3
"""Loop iteration 4: a STEP-SYNCHRONOUS ensemble of NECTEC-family transducers -- several ears, one decoder loop.

    PYTHONUTF8=1 python adapter/rnnt_ensemble.py --models base --tag ens_ctrl          # must equal NECTEC
    PYTHONUTF8=1 python adapter/rnnt_ensemble.py --models base lot2_a100 --tag ens_b_l2

WHY (`F39` + the 2026-10-08 oracle): the remaining error is hearing. NECTEC's own 5-best holds only -4.6%
(one model's beam is not diverse), but a per-utterance oracle over base NECTEC and its 1-epoch LOTUSDIS
tune (`lot2_a100`) holds -11.7%, and over four variants -20%. Different weights hear differently; a
weight-INTERPOLATION (A2, alpha .75) keeps little of that. So combine them at the OUTPUT, every step.

HOW. Every model encodes the same PCM16 audio (same preprocessor, same x8 frame grid -> same T). Greedy
transducer search as NeMo's `_greedy_decode`, except the joint distribution at (t, u) is the weighted
mean of each model's LOG-softmax (product of experts); the chosen token advances EVERY model's own
prediction network. max_symbols per frame = 10, as NeMo's default.

Models are all the same architecture and tokenizer (NECTEC's 2,048 pieces), so token ids agree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from ctc_head import NEC, load_nec, pcm16  # noqa: E402

BENCH = Path("E:/asr-bench/thai-contextasr-bench")
MODELS = {"base": NEC}
for p in (ROOT / "checkpoints").glob("lot*_a*.nemo"):
    MODELS[p.stem] = str(p)
for p in (ROOT / "checkpoints").glob("rt_student_lot*.nemo"):      # alpha = 1 is the raw fine-tune
    MODELS[p.stem.replace("rt_student_", "") + "_a100"] = str(p)


@torch.no_grad()
def encode(m, y, dev):
    m.preprocessor.eval(); m.encoder.eval()
    x = torch.from_numpy(y)[None].to(dev)
    f, fl = m.preprocessor(input_signal=x, length=torch.tensor([x.shape[1]], device=dev))
    enc, el = m.encoder(audio_signal=f, length=fl)
    return enc[0, :, : int(el[0])].T.unsqueeze(1)          # [T, 1, D]


@torch.no_grad()
def ens_greedy(ms, encs, w, max_symbols=10):
    T = min(e.shape[0] for e in encs)
    blank = ms[0].decoder.blank_idx
    states = [None] * len(ms)
    last = None
    gs = [None] * len(ms)
    ids = []
    for t in range(T):
        for _ in range(max_symbols):
            lps = []
            for i, m in enumerate(ms):
                if gs[i] is None:
                    lab = None if last is None else torch.tensor([[last]], device=encs[i].device)
                    g, h = m.decoder.predict(lab, states[i], add_sos=False, batch_size=1)
                    gs[i] = (g, h)
                logits = m.joint.joint(encs[i][t:t + 1].transpose(0, 1), gs[i][0])[0, 0, 0]
                lps.append(w[i] * torch.log_softmax(logits.float(), -1))
            k = int(torch.stack(lps).sum(0).argmax())
            if k == blank:
                break
            ids.append(k)
            last = k
            states = [gs[i][1] for i in range(len(ms))]
            gs = [None] * len(ms)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["base"])
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--split", default="real")
    ap.add_argument("--manifest", default="", help="jsonl with audio_filepath/text instead of the benchmark")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda"
    ms = [load_nec(dev, MODELS[k]) for k in a.models]
    w = a.weights or [1.0 / len(ms)] * len(ms)
    if a.manifest:
        rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
        items = [(r.get("utt_id") or r.get("id") or str(i), r["audio_filepath"]) for i, r in enumerate(rows)]
    else:
        rows = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")]
        items = [(r["utt_id"], BENCH / r["wav"]) for r in rows if r["split"] == a.split]
    if a.limit:
        items = items[: a.limit]
    out, t_enc, t_dec, aud = {}, 0.0, 0.0, 0.0
    for n, (uid, wav) in enumerate(items):
        y, sr = sf.read(wav, dtype="float32")
        y = pcm16(y.mean(axis=1) if y.ndim > 1 else y)
        aud += len(y) / 16000
        torch.cuda.synchronize(); t0 = time.perf_counter()
        encs = [encode(m, y, dev) for m in ms]
        torch.cuda.synchronize(); t1 = time.perf_counter()
        ids = ens_greedy(ms, encs, w)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        t_enc += t1 - t0; t_dec += t2 - t1
        out[uid] = ms[0].tokenizer.ids_to_text(ids).strip()
        if n % 100 == 0:
            print(n, len(items), flush=True)
    p = ROOT / "results" / f"{a.tag}_hyps.jsonl"
    with p.open("w", encoding="utf-8", newline="\n") as g:
        for u, t in out.items():
            g.write(json.dumps({"utt_id": u, "condition": "none", "text": t}, ensure_ascii=False) + "\n")
    run = {"tag": a.tag, "models": a.models, "weights": w, "n": len(out), "audio_s": round(aud, 1),
           "encode_ms_per_turn": round(1000 * t_enc / len(out), 1), "decode_ms_per_turn": round(1000 * t_dec / len(out), 1)}
    (ROOT / "results" / f"{a.tag}_run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
