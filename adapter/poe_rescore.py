#!/usr/bin/env python3
"""Toward CER 0.060: every ear scores every candidate transcript -- a product of experts at the SENTENCE level.

    PYTHONUTF8=1 python adapter/poe_rescore.py score --pool nec patL twL          # all 682, scores only
    PYTHONUTF8=1 python adapter/poe_rescore.py pick --half dev                     # grid on dev
    PYTHONUTF8=1 python adapter/poe_rescore.py pick --half test --final --w ...    # once

WHY. Per-utterance ORACLE over {NECTEC, Pathumma-L, typhoon-L} on the dev half is -24.5%; the vote reaches -3%.
The missing piece is a judge. A step-synchronous ensemble failed because transducer and attention models emit
on different frames (`F40` #4); scoring WHOLE candidate texts has no such problem.

Candidates per utterance: each ear's transcript + the ROVER output. Scores (teacher-forced, no search):
  * nec  : CTC log-likelihood under NECTEC's frozen encoder + the 2.6M CTC head (`ctchead_h3`, NECTEC's own
           2,048 pieces) -- `torch.nn.functional.ctc_loss`
  * patL / twL : Whisper decoder log-probability of <prompt> + tokens(candidate) + <|endoftext|>
Each score is stored raw and with its token count, so `pick` can try per-token normalisation.
`pick` chooses argmax_c sum_m w_m * s_m(c) / len_m(c)^alpha with (w, alpha) from a small grid on DEV only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from eval_canon import BENCH  # noqa: E402

SCORES = ROOT / "results/poe_scores.json"
WHISPER = {"patL": "nectec/Pathumma-whisper-th-large-v3", "twL": "typhoon-ai/typhoon-whisper-large-v3"}


def candidates(pool, H, u):
    from pilot_rover import join, rover, toks
    c = {k: H[k][u].strip() for k in pool}
    merged, _ = rover({k: toks(c[k]) for k in pool}, {k: 1.0 for k in pool}, pool)
    c["rover"] = join(merged)
    return c


@torch.no_grad()
def whisper_scores(name, items):
    """items: list of (wav_array, [texts]) -> list of [(logp, ntok)]"""
    from spec_whisper import Spec
    S = Spec(name)
    out = []
    for y, texts in items:
        enc = S.encode(S.features(y))
        seqs = [S.prompt + S.tok.encode(t, add_special_tokens=False)[:440] + [S.eot] for t in texts]
        L = max(map(len, seqs))
        x = torch.full((len(seqs), L), S.eot, dtype=torch.long, device="cuda")
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
        h = S.m.model.decoder(input_ids=x, encoder_hidden_states=enc.expand(len(seqs), -1, -1)).last_hidden_state
        lp = torch.log_softmax(S.m.proj_out(h).float(), -1)
        res = []
        for i, s in enumerate(seqs):
            P = len(S.prompt)
            tgt = torch.tensor(s[P:], device="cuda")
            res.append((float(lp[i, P - 1: len(s) - 1].gather(-1, tgt[:, None]).sum()), len(s) - P))
        out.append(res)
    del S
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def nec_scores(items):
    from ctc_head import encode, load_nec
    m = load_nec()
    ck = torch.load(ROOT / "checkpoints/ctchead_h3.pt", map_location="cuda")
    D, V, Hd = ck["d"], ck["vocab"], ck.get("hidden", 0)
    head = (torch.nn.Sequential(torch.nn.Linear(D, Hd), torch.nn.GELU(), torch.nn.Linear(Hd, V + 1))
            if Hd else torch.nn.Linear(D, V + 1)).cuda()
    head.load_state_dict(ck["state"]); head.eval()
    out = []
    for y, texts in items:
        f = encode(m, [y])[0].cuda().float()                       # [T, D]  (eval mode inside)
        lp = torch.log_softmax(head(f), -1)                           # [T, V+1]
        res = []
        for t in texts:
            ids = m.tokenizer.text_to_ids(t.replace(" ", "")) or [0]
            T = lp.shape[0]
            if len(ids) > T:
                res.append((-1e4, len(ids))); continue
            nll = torch.nn.functional.ctc_loss(lp[:, None], torch.tensor([ids], device="cuda"),
                                               torch.tensor([T]), torch.tensor([len(ids)]), blank=V,
                                               reduction="sum", zero_infinity=True)
            res.append((-float(nll), len(ids)))
        out.append(res)
    return out


def cmd_score(a):
    from ctc_head import pcm16
    from pool_search_dev import EARS
    H = {k: {json.loads(l)["utt_id"]: json.loads(l)["text"] for l in open(EARS[k], encoding="utf-8")
             if json.loads(l).get("condition", "none") == "none"} for k in a.pool}
    rows = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["split"] == "real"]
    cands, items = {}, []
    for r in rows:
        c = candidates(a.pool, H, r["utt_id"])
        cands[r["utt_id"]] = c
        y, _ = sf.read(BENCH / r["wav"], dtype="float32")
        items.append((pcm16(y.mean(axis=1) if y.ndim > 1 else y), list(c.values())))
    names = list(next(iter(cands.values())).keys())
    sc = {"nec": nec_scores(items)}
    print("nec scored", flush=True)
    for k, nm in WHISPER.items():
        sc[k] = whisper_scores(nm, items)
        print(k, "scored", flush=True)
    out = {}
    for i, r in enumerate(rows):
        out[r["utt_id"]] = {"cands": cands[r["utt_id"]],
                            "scores": {m: dict(zip(names, sc[m][i])) for m in sc}}
    SCORES.write_text(json.dumps({"pool": a.pool, "utts": out}, ensure_ascii=False), encoding="utf-8")
    print("->", SCORES)


def cmd_pick(a):
    from official_utt import Official
    O = Official()
    D = json.loads(SCORES.read_text(encoding="utf-8"))["utts"]
    split = json.loads((ROOT / "results/split_dev_test.json").read_text(encoding="utf-8"))
    if a.half == "test" and not (a.final and a.w):
        sys.exit("refused: test needs --final and weights fixed from dev")
    ids = split[a.half]
    E = lambda u, h: O.err(u, h)[0]                     # official per-utterance errors (2026-10-08)

    def choose(u, w, alpha):
        d = D[u]
        def s(c):
            return sum(w[m] * d["scores"][m][c][0] / max(d["scores"][m][c][1], 1) ** alpha for m in w)
        return d["cands"][max(d["cands"], key=s)]

    nec = sum(E(u, D[u]["cands"]["nec"]) for u in ids)
    if a.w:
        w = dict(zip(["nec", "patL", "twL"], a.w)); alpha = a.alpha
    else:
        best = None
        for wn, wp, wt in itertools.product([0, 0.5, 1, 2], repeat=3):
            if wn == wp == wt == 0:
                continue
            for alpha in (0.0, 0.5, 1.0):
                w = {"nec": wn, "patL": wp, "twL": wt}
                e = sum(E(u, choose(u, w, alpha)) for u in ids)
                if best is None or e < best[0]:
                    best = (e, w, alpha)
        e, w, alpha = best
        print(f"dev best: {e / nec - 1:+.3f} vs NECTEC with w={w} alpha={alpha}")
    with (ROOT / "results" / f"{a.tag}_hyps.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for u in ids:
            f.write(json.dumps({"utt_id": u, "condition": "none", "text": choose(u, w, alpha)}, ensure_ascii=False) + "\n")
    (ROOT / "results" / f"{a.tag}_run.json").write_text(json.dumps({"w": w, "alpha": alpha, "half": a.half}), encoding="utf-8")
    print("weights", w, "alpha", alpha)


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("score"); s.add_argument("--pool", nargs="+", default=["nec", "patL", "twL"])
    p = sp.add_parser("pick")
    p.add_argument("--half", choices=("dev", "test"), default="dev")
    p.add_argument("--final", action="store_true")
    p.add_argument("--w", nargs=3, type=float, default=None)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--tag", default="poe_dev")
    a = ap.parse_args()
    {"score": cmd_score, "pick": cmd_pick}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
