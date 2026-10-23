#!/usr/bin/env python3
"""Loop iteration 3: a SESSION-ADAPTIVE ear -- NECTEC tunes its LayerNorms between turns, without labels.

    PYTHONUTF8=1 python adapter/tta_session.py --lr 0      --tag tta_ctrl     # control: must equal NECTEC
    PYTHONUTF8=1 python adapter/tta_session.py --lr 1e-4 --steps 3 --tag tta_a

WHY (`F39`): the remaining error is HEARING (tone, vowel length, names), not writing, and names are not
recoverable from lists the user never gave (loop iterations 1-2). Test-time adaptation changes hearing
itself: SUTA (Lin et al. 2022) minimises output entropy on the incoming audio; speaker codes report -20%
WER from one minute of a speaker (van Dalen et al. 2025). The benchmark's real split is 34 recording
sessions, so a voice loop's "earlier turns of this conversation" exist here too.

CAUSAL, so it is real-time: turn j is transcribed with the parameters adapted on turns < j of the SAME
session; adaptation on turn j happens AFTER its transcript is out (between turns, off the latency path).
Parameters reset at every session start. Only LayerNorm weight/bias in the encoder are trained; the
encoder stays in eval mode (BatchNorm statistics frozen, dropout off -- `F36`).

Objective: mean per-frame entropy of the frozen CTC head (`checkpoints/ctchead_h3.pt`, NECTEC's own
vocabulary) over frames whose argmax is not blank -- SUTA's entropy term on the speech frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from ctc_head import NEC, load_nec, pcm16  # noqa: E402

BENCH = Path("E:/asr-bench/thai-contextasr-bench")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--head", default=str(ROOT / "checkpoints/ctchead_h3.pt"))
    ap.add_argument("--split", default="real")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda"
    m = load_nec(dev)
    ck = torch.load(a.head, map_location=dev)
    D, V, Hd = ck["d"], ck["vocab"], ck.get("hidden", 0)
    head = (torch.nn.Sequential(torch.nn.Linear(D, Hd), torch.nn.GELU(), torch.nn.Linear(Hd, V + 1))
            if Hd else torch.nn.Linear(D, V + 1)).to(dev)
    head.load_state_dict(ck["state"]); head.eval()
    for p in list(m.parameters()) + list(head.parameters()):
        p.requires_grad_(False)
    ln = [mod for mod in m.encoder.modules() if isinstance(mod, torch.nn.LayerNorm)]
    params = [p for mod in ln for p in (mod.weight, mod.bias) if p is not None]
    init = [p.detach().clone() for p in params]
    print(f"{len(ln)} LayerNorms, {sum(p.numel() for p in params)} adapted params", flush=True)

    rows = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["split"] == a.split]
    sess = defaultdict(list)
    for r in rows:
        sess[r["utt_id"].rsplit("-", 1)[0]].append(r)
    out, t_adapt, n_adapt = {}, 0.0, 0
    for s, v in sess.items():
        v.sort(key=lambda r: int(r["utt_id"].rsplit("-", 1)[1]))
        with torch.no_grad():
            for p, p0 in zip(params, init):
                p.copy_(p0)                                         # reset per session
        opt = torch.optim.Adam(params, lr=a.lr) if a.lr > 0 else None
        for r in v:
            y, _ = sf.read(BENCH / r["wav"], dtype="float32")
            y = pcm16(y.mean(axis=1) if y.ndim > 1 else y)
            m.eval()
            hyp = m.transcribe([y], verbose=False, batch_size=1)
            h = hyp[0] if hyp else ""
            h = h[0] if isinstance(h, (list, tuple)) else h
            out[r["utt_id"]] = (h if isinstance(h, str) else str(getattr(h, "text", ""))).strip()
            if opt is None:
                continue
            t0 = time.perf_counter()
            m.preprocessor.eval(); m.encoder.eval()                 # F36: transcribe() leaves encoder.training True
            for p in params:
                p.requires_grad_(True)
            x = torch.from_numpy(y)[None].to(dev)
            lens = torch.tensor([x.shape[1]], device=dev)
            for _ in range(a.steps):
                f, fl = m.preprocessor(input_signal=x, length=lens)
                enc, el = m.encoder(audio_signal=f, length=fl)
                lp = torch.log_softmax(head(enc[0, :, : int(el[0])].T), -1)
                speech = lp.argmax(-1) != V
                if speech.sum() == 0:
                    break
                ent = -(lp.exp() * lp).sum(-1)[speech].mean()
                opt.zero_grad(); ent.backward(); opt.step()
            for p in params:
                p.requires_grad_(False)
            torch.cuda.synchronize()
            t_adapt += time.perf_counter() - t0; n_adapt += 1
        print(f"session {s}: {len(v)} turns", flush=True)
    p = ROOT / "results" / f"{a.tag}_hyps.jsonl"
    with p.open("w", encoding="utf-8", newline="\n") as g:
        for u, t in out.items():
            g.write(json.dumps({"utt_id": u, "condition": "none", "text": t}, ensure_ascii=False) + "\n")
    (ROOT / "results" / f"{a.tag}_run.json").write_text(json.dumps(
        {"tag": a.tag, "lr": a.lr, "steps": a.steps, "head": a.head, "n_utts": len(out),
         "adapt_ms_per_turn": round(1000 * t_adapt / max(n_adapt, 1), 1),
         "objective": "entropy of frozen CTC head on non-blank frames; encoder LayerNorm affine only; causal; reset per session"},
        indent=1), encoding="utf-8")
    print(f"-> {p}  adapt {1000 * t_adapt / max(n_adapt, 1):.0f} ms/turn (off the latency path)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
