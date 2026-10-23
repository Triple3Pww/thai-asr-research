#!/usr/bin/env python3
"""Pilot: token-level ROVER over the public Thai ASRs. Label-free; reads only hypotheses.

    python adapter/pilot_rover.py --pool tw pat q17 isan bio qctx rt
    python adapter/pilot_rover.py --pool tw pat q17 --weights agree

WHAT THIS IS. `pilot_mbr_combine.py` can only *pick* one system's whole transcript per utterance.
ROVER votes position by position, so it can fix an error that every system makes in a different
place. Units are the benchmark's own SCORING TOKENS (`scorer/norm.py` normalize + `scorer/tok.py`
tokenize: one token per Thai character, one per Latin/digit word) -- the vote happens in exactly
the space the metric counts, so a convention the scorer already forgives never splits a vote.

Algorithm (classic ROVER, pivot form):
  1. pivot = the MBR medoid of the pool for this utterance (min summed token edit distance);
  2. every other hypothesis is aligned to the pivot (Levenshtein, deterministic backtrace);
  3. each pivot slot gets one vote per system (its aligned token, or EPS for a deletion), and each
     gap between pivot slots gets one vote per system (the run it inserted there, or EPS);
  4. the slot/gap emits the weighted plurality; ties go to the pivot.
Weights: `uniform` (1 per system) or `agree` -- each system's weight is its mean agreement with
the rest of the pool over the whole test set (1 - normalised distance). That is label-free too:
no reference is read to set it.

The reference is read ONLY by the scorer. The emitted text is tokens re-joined (Thai bare, Latin
words space-separated), which the scorer re-tokenizes to the identical token list.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = Path("E:/asr-bench/thai-contextasr-bench")
sys.path.insert(0, str(BENCH / "scorer"))
sys.path.insert(0, str(HERE))
from norm import normalize  # noqa: E402
from tok import tokenize  # noqa: E402
from pilot_mbr_combine import POOL, load, score  # noqa: E402

OUT = Path(os.environ.get("ADAPTER_OUT") or (HERE.parent / "results"))
EPS = ""


def toks(s: str) -> list[str]:
    return tokenize(normalize(s or ""))


def align(ref: list[str], hyp: list[str]) -> list[tuple[int | None, str | None]]:
    """Align hyp to ref. Returns ops: (ref_index, hyp_token) for match/sub, (i, None) for a
    deletion of ref[i], (None, tok) for an insertion. Tie order: match/sub > del > ins."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri, row, prev = ref[i - 1], d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j - 1] + (ri != hyp[j - 1]), prev[j] + 1, row[j - 1] + 1)
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append((i - 1, hyp[j - 1])); i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append((i - 1, None)); i -= 1
        else:
            ops.append((None, hyp[j - 1])); j -= 1
    return ops[::-1]


def edit(a: list[str], b: list[str]) -> int:
    return sum(1 for r, h in align(a, b) if r is None or h is None or a[r] != h)


def rover(seqs: dict[str, list[str]], w: dict[str, float], order: list[str]) -> tuple[list[str], str]:
    names = [k for k in order if k in seqs]
    dist = {(a, b): edit(seqs[a], seqs[b]) for a in names for b in names if a < b}
    dd = lambda a, b: 0 if a == b else dist[(min(a, b), max(a, b))]  # noqa: E731
    pivot = min(names, key=lambda k: (sum(dd(k, o) for o in names), names.index(k)))
    P = seqs[pivot]
    slot = [defaultdict(float) for _ in P]            # votes on each pivot token
    gap = [defaultdict(float) for _ in range(len(P) + 1)]  # votes on what goes BEFORE slot i
    for k in names:
        if k == pivot:
            for i, t in enumerate(P):
                slot[i][t] += w[k]
            for g in gap:
                g[EPS] += w[k]
            continue
        ins = defaultdict(list)
        cur = 0                                        # next pivot index
        for r, h in align(P, seqs[k]):
            if r is None:
                ins[cur].append(h)
            else:
                slot[r][EPS if h is None else h] += w[k]
                cur = r + 1
        for gi in range(len(P) + 1):
            gap[gi][" ".join(ins[gi]) if gi in ins else EPS] += w[k]

    def pick(votes, default):
        best = max(votes.values())
        winners = [t for t, v in votes.items() if v == best]
        return default if default in winners else sorted(winners)[0]

    out = []
    for i in range(len(P) + 1):
        g = pick(gap[i], EPS)
        if g:
            out.extend(g.split(" "))
        if i < len(P):
            t = pick(slot[i], P[i])
            if t:
                out.append(t)
    return out, pivot


def join(tokens: list[str]) -> str:
    s, prev_latin = [], False
    for t in tokens:
        latin = len(t) > 1 or t.isascii()
        if s and (latin or prev_latin):
            s.append(" ")
        s.append(t)
        prev_latin = latin
    return "".join(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="+", default=list(POOL))
    ap.add_argument("--weights", default="uniform", choices=("uniform", "agree"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--add", nargs="*", default=[], help="2026-10-08: extra systems as name=path (not in POOL)")
    args = ap.parse_args()
    for spec in args.add:
        k, v = spec.split("=", 1)
        POOL[k] = Path(v)

    ann = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")]
    keep = {a["utt_id"] for a in ann if a["split"] == "real"}
    hyps = {k: load(POOL[k], keep) for k in args.pool}
    ids = sorted(set.intersection(*(set(h) for h in hyps.values())))
    tk = {k: {u: toks(hyps[k][u]) for u in ids} for k in args.pool}

    w = {k: 1.0 for k in args.pool}
    if args.weights == "agree":                       # label-free: agreement with the pool
        for k in args.pool:
            num = den = 0
            for u in ids:
                for o in args.pool:
                    if o != k:
                        num += edit(tk[k][u], tk[o][u]); den += max(len(tk[o][u]), 1)
            w[k] = 1.0 - num / den
        print("weights:", {k: round(v, 4) for k, v in w.items()})

    rows, pivots = [], defaultdict(int)
    for u in ids:
        out, pv = rover({k: tk[k][u] for k in args.pool}, w, args.pool)
        pivots[pv] += 1
        rows.append({"utt_id": u, "condition": "none", "text": join(out)})

    tag = args.tag or f"rover_{args.weights}_" + "+".join(args.pool)
    with tempfile.TemporaryDirectory() as td:
        m = score(rows, Path(td), "rover")
    print(f"{tag}: n={len(ids)} CER {m['cer']:.4f} U {m['u_cer']:.4f} B {m['b_cer']:.4f} "
          f"recall {m['entity_recall']:.4f} | pivots {dict(pivots)}")
    rec = {"tag": tag, "pool": args.pool, "weights": w, "n_utts": len(ids), "metrics": m,
           "pivots": dict(pivots)}
    with (OUT / "pilot_rover_real.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    hp = OUT / f"{tag}_hyps.jsonl"
    with hp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
