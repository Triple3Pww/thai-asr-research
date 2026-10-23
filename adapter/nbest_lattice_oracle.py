#!/usr/bin/env python3
"""Ceilings of an N-best picker: whole-sentence pick vs per-spot (word-level) pick. Reads the reference.

    python adapter/nbest_lattice_oracle.py results/nbest_turbo5_nbest.jsonl results/nbest_rt5_nbest.jsonl

His two options (2026-10-04, "try both"):
  1. merge several models' N-best lists, let a picker choose ONE whole sentence;
  2. let the picker choose at EACH spot where the candidates disagree (Jev-style many small choices).

For (2) the candidates are aligned into a confusion network exactly as `pilot_rover.py` does
(pivot = the merged list's MBR medoid; every other candidate aligned to it; one slot per pivot token,
one gap before each), and the ORACLE path through it is found by dynamic programming: the cheapest
token sequence the network can spell, measured against the reference. That is the ceiling of a
perfect per-spot picker -- a real one gets part of it. Everything here reads the reference: these
are ceilings, never systems.

CER is computed on the scorer's tokens after `eval_canon.canon` (one normaliser for reference and
hypothesis); `top1` / sentence-oracle are reported on the same tokeniser so all rows compare.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_canon import canon  # noqa: E402
from pilot_rover import BENCH, align, edit, toks  # noqa: E402


def network(cands: list[list[str]]):
    """Confusion network: list of slots, each a set of options (tuples of tokens; () = skip)."""
    pivot = min(range(len(cands)), key=lambda i: (sum(edit(cands[i], c) for c in cands), i))
    P = cands[pivot]
    slot = [set() for _ in P]
    gap = [set() for _ in range(len(P) + 1)]
    for c in cands:
        ins, cur = defaultdict(list), 0
        for r, h in align(P, c):
            if r is None:
                ins[cur].append(h)
            else:
                slot[r].add(() if h is None else (h,))
                cur = r + 1
        for g in range(len(P) + 1):
            gap[g].add(tuple(ins[g]) if g in ins else ())
    net = []
    for i in range(len(P) + 1):
        net.append(gap[i])
        if i < len(P):
            net.append(slot[i])
    return net


def lattice_oracle(net, ref: list[str]) -> int:
    """Min edit distance between ref and any token sequence the network can spell."""
    INF = 10 ** 9
    D = list(range(len(ref) + 1))          # cost of consuming nothing yet vs ref prefix j (deletions)
    for options in net:
        best = [INF] * (len(ref) + 1)
        for opt in options:
            row = D[:]
            for t in opt:                   # extend Levenshtein by one hypothesis token
                new = [row[0] + 1] + [0] * len(ref)
                for j in range(1, len(ref) + 1):
                    new[j] = min(row[j] + 1, new[j - 1] + 1, row[j - 1] + (ref[j - 1] != t))
                row = new
            best = [min(a, b) for a, b in zip(best, row)]
        for j in range(1, len(best)):       # allow ref tokens skipped between slots
            best[j] = min(best[j], best[j - 1] + 1)
        D = best
    return D[-1]


def main() -> int:
    ann = {json.loads(l)["utt_id"]: json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")}
    lists = [{json.loads(l)["utt_id"]: json.loads(l)["cands"] for l in open(p, encoding="utf-8")} for p in sys.argv[1:]]
    names = [Path(p).stem.replace("_nbest", "") for p in sys.argv[1:]]
    ids = sorted(set.intersection(*(set(x) for x in lists)))
    tot = 0
    acc = defaultdict(int)
    for u in ids:
        ref = toks(canon(ann[u]["text"]))
        tot += len(ref)
        per = [[toks(canon(c["text"])) for c in L[u]] for L in lists]
        for n, cs in zip(names, per):
            acc[f"{n} top1"] += edit(ref, cs[0])
            acc[f"{n} best-of-list (sentence)"] += min(edit(ref, c) for c in cs)
        merged = [c for cs in per for c in cs]
        acc["MERGED best-of-list (sentence)"] += min(edit(ref, c) for c in merged)
        for n, cs in zip(names, per):
            acc[f"{n} per-spot (lattice)"] += lattice_oracle(network(cs), ref)
        acc["MERGED per-spot (lattice)"] += lattice_oracle(network(merged), ref)
    print(f"{len(ids)} utterances, {tot} reference tokens; own tokeniser, canon normaliser")
    for k, v in sorted(acc.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k:42s} CER {v / tot:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
