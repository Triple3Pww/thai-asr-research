#!/usr/bin/env python3
"""E2 of the Ear-Memory-Pen plan: a SOUND head on NECTEC's frozen encoder (onset / rhyme / tone CTC).

    PYTHONUTF8=1 python adapter/syllable_head.py train --tag s1          # minutes, on the ctc_head cache
    PYTHONUTF8=1 python adapter/syllable_head.py bench --tag s1          # encode 682 real utts, dump units

ASR venv (`E:/asr-spike/.venv`). Labels come from `adapter/syllable_labels.py labels` (thainlp venv);
features are the SAME cache `ctc_head.py` built (base NECTEC encoder, PCM16, eval mode).

THE QUESTION IT ANSWERS. The Ear only earns a place if it reads the SOUND better than NECTEC's own text
implies. `bench` writes, per real benchmark utterance, the head's greedy units; `syllable_labels.py
score` then compares unit error vs the reference's G2P units for (a) this head and (b) G2P of NECTEC's
transcript. If (a) is not below (b), the draft already carries everything the Ear would add -> stop.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from ctc_head import CACHE, NEC, encode, load_nec, pcm16  # noqa: E402

UNITS = CACHE / "syllable_units.json"
BENCH = Path("E:/asr-bench/thai-contextasr-bench")


def lev(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def make_head(D, V, hidden):
    if hidden:
        return torch.nn.Sequential(torch.nn.Linear(D, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, V + 1))
    return torch.nn.Linear(D, V + 1)


def greedy(head, f, blank, dev):
    ids = head(f.to(dev).float()).argmax(-1).tolist()
    out, prev = [], -1
    for i in ids:
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def group(us: list[str]) -> list[str]:
    """[O:kh, R:OON, T0, <oov>, ...] -> ['kh|OON|T0', '<oov>', ...]: ONE token per toned syllable.

    2026-10-08: the first head (s1) used the three units as three CTC tokens -- ~15 tokens/s against
    NECTEC's 12.5 encoder frames/s, so CTC could not fit them (24,322 of 32,634 clips kept, val unit
    error 0.38). Whole syllables are ~5/s; the top 4,000 cover 99.3% of syllable tokens."""
    out, i = [], 0
    while i < len(us):
        if us[i] == "<oov>":
            out.append("<oov>"); i += 1
        else:
            out.append("|".join(us[i:i + 3])); i += 3
    return out


def ungroup(ts: list[str]) -> list[str]:
    return [u for t in ts for u in (t.split("|") if "|" in t else [t])]


def cmd_train(a):
    from collections import Counter
    U = json.loads(UNITS.read_text(encoding="utf-8"))
    seqs = {n: [group(us) for us in U["units"][n]] for n in ["lot_train", "porjai_train", "lot_val"]}
    cnt = Counter(t for n in a.sets for s in seqs[n] for t in s)
    itos = [t for t, _ in cnt.most_common(a.vocab)] + ["<unk>"]
    stoi = {u: i for i, u in enumerate(itos)}
    V = len(itos)
    sets = {}
    for n in ["lot_train", "porjai_train", "lot_val"]:
        d = torch.load(CACHE / f"{n}.pt")
        sets[n] = [(f, torch.tensor([stoi.get(u, V - 1) for u in us], dtype=torch.int16))
                   for f, us in zip(d["feats"], seqs[n])]
        del d
    pairs = [(f, l) for n in a.sets for f, l in sets[n] if len(l) and f.shape[0] >= len(l)]
    val = sets["lot_val"]
    D = pairs[0][0].shape[1]
    print(f"train clips {len(pairs)}  unit vocab {V}+blank  d {D}", flush=True)
    dev = "cuda"
    head = make_head(D, V, a.hidden).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-2)
    ctc = torch.nn.CTCLoss(blank=V, zero_infinity=True)
    rng = random.Random(0)
    best, log = 9.0, []
    for ep in range(a.epochs):
        rng.shuffle(pairs)
        head.train()
        tot = 0.0
        for i in range(0, len(pairs), a.bs):
            b = pairs[i: i + a.bs]
            T = max(f.shape[0] for f, _ in b)
            x = torch.zeros(len(b), T, D, device=dev)
            for j, (f, _) in enumerate(b):
                x[j, : f.shape[0]] = f.to(dev).float()
            lp = torch.log_softmax(head(x), -1).transpose(0, 1)
            tgt = torch.cat([l.long() for _, l in b]).to(dev)
            loss = ctc(lp, tgt, torch.tensor([f.shape[0] for f, _ in b]), torch.tensor([len(l) for _, l in b]))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        head.eval()
        with torch.no_grad():
            e = n = 0
            for f, l in val:
                e += lev(l.tolist(), greedy(head, f, V, dev)); n += len(l)     # syllable error rate
        uer = e / max(n, 1)
        log.append({"epoch": ep, "loss": round(tot / len(pairs), 4), "val_unit_err": round(uer, 4)})
        print(log[-1], flush=True)
        if uer < best:
            best = uer
            torch.save({"state": head.state_dict(), "itos": itos, "d": D, "hidden": a.hidden, "log": log},
                       ROOT / "checkpoints" / f"sylhead_{a.tag}.pt")
    (ROOT / "results" / f"sylhead_{a.tag}_run.json").write_text(json.dumps(
        {"tag": a.tag, "sets": a.sets, "hidden": a.hidden, "lr": a.lr, "bs": a.bs, "epochs": a.epochs,
         "best_val_unit_err": best, "log": log}, indent=1), encoding="utf-8")
    print("best val unit error", best)


def cmd_bench(a):
    ck = torch.load(ROOT / "checkpoints" / f"sylhead_{a.tag}.pt")
    itos = ck["itos"]
    V = len(itos)
    dev = "cuda"
    head = make_head(ck["d"], V, ck["hidden"]).to(dev)
    head.load_state_dict(ck["state"]); head.eval()
    m = load_nec()
    m.encoder.eval()                                   # F36: NeMo transcribe() leaves training=True
    rows = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["split"] == "real"]
    out = {}
    with torch.no_grad():
        for r in rows:
            y, _ = sf.read(BENCH / r["wav"], dtype="float32")
            f = encode(m, [pcm16(y.mean(axis=1) if y.ndim > 1 else y)])[0]
            out[r["utt_id"]] = ungroup([itos[i] for i in greedy(head, f, V, dev)])
            if a.post:
                lp = torch.log_softmax(head(f.to(dev).float()), -1).half().cpu()
                torch.save(lp, Path(a.post) / f"{r['utt_id']}.pt")
    p = ROOT / "results" / f"sylhead_{a.tag}_bench_units.json"
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(len(out), "->", p)


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--tag", required=True)
    t.add_argument("--sets", nargs="+", default=["lot_train", "porjai_train"])
    t.add_argument("--hidden", type=int, default=0)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--vocab", type=int, default=4000, help="most frequent toned syllables; rest -> <unk>")
    b = sp.add_parser("bench")
    b.add_argument("--tag", required=True)
    b.add_argument("--post", default="", help="dir to save per-utt log-posteriors (for the Pen)")
    a = ap.parse_args()
    {"train": cmd_train, "bench": cmd_bench}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
