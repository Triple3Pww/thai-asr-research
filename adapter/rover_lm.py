#!/usr/bin/env python3
"""Toward CER 0.060: ROVER, then a language model settles the regions where the ears disagree.

    PYTHONUTF8=1 python adapter/rover_lm.py --pool nec patL twL --half dev --tag rlm_dev
    PYTHONUTF8=1 python adapter/rover_lm.py --pool nec patL twL --half test --gamma G --delta D --final --tag rlm_final

WHY. A character-level vote cannot tell which of two acoustically plausible stretches is real Thai; a language
model can, and unlike the Pen (`F39` E3) the candidates here are what the EARS heard, not dictionary homophones.

ALGORITHM (one batched LM forward per utterance -- real-time):
  base = ROVER over the pool (`pilot_rover.rover`, uniform)
  for every ear k, every contiguous stretch where ear k differs from base (difflib opcodes on scorer tokens)
      -> a candidate "swap": base with that stretch replaced by ear k's version; support(swap) = number of ears
         whose output contains exactly that replacement at that place
  score(swap) = [logP_LM(base with swap) - logP_LM(base)] + GAMMA * (support(swap) - support(base stretch))
  apply every swap with score > DELTA, non-overlapping, best first.
GAMMA and DELTA are chosen on the DEV half only (grid below); the test half is scored once with them fixed.
LM = Qwen3-0.6B base weights, sentence log-probability of the normalised text.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from eval_canon import BENCH  # noqa: E402
from paired_vs import lev, n  # noqa: E402
from pen_score import LM  # noqa: E402
from pilot_rover import join, rover, toks  # noqa: E402
from pool_search_dev import EARS  # noqa: E402

GAMMAS = [0.0, 0.5, 1.0, 2.0, 4.0]
DELTAS = [-2.0, 0.0, 1.0, 2.0, 4.0, 8.0]


def swaps_for(base: list[str], others: dict[str, list[str]]):
    """-> list of (i1, i2, replacement_tokens, support_alt, support_base)."""
    found = {}
    for k, seq in others.items():
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, base, seq, autojunk=False).get_opcodes():
            if tag != "equal":
                key = (i1, i2, tuple(seq[j1:j2]))
                found.setdefault(key, set()).add(k)
    out = []
    for (i1, i2, rep), ks in found.items():
        sup_base = sum(1 for k, seq in others.items() if k not in ks)   # ears that did not differ here
        out.append((i1, i2, list(rep), len(ks), sup_base))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="+", default=["nec", "patL", "twL"])
    ap.add_argument("--half", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.half == "test" and not (a.final and a.gamma is not None and a.delta is not None):
        sys.exit("refused: test half needs --final with --gamma/--delta fixed from dev")
    split = json.loads((ROOT / "results/split_dev_test.json").read_text(encoding="utf-8"))
    ids = split[a.half] if a.half != "all" else split["dev"] + split["test"]
    H = {k: {json.loads(l)["utt_id"]: json.loads(l)["text"] for l in open(EARS[k], encoding="utf-8")
             if json.loads(l).get("condition", "none") == "none"} for k in a.pool}
    lm = LM()
    data = {}
    for c, u in enumerate(ids):
        T = {k: toks(H[k].get(u, "")) for k in a.pool}
        base, _ = rover(T, {k: 1.0 for k in a.pool}, a.pool)
        sw = swaps_for(base, T)
        texts = [join(base)] + [join(base[:i1] + rep + base[i2:]) for i1, i2, rep, _, _ in sw]
        lps = []
        for k in range(0, len(texts), 16):
            lps += lm.logp([n(t) for t in texts[k:k + 16]])
        data[u] = (base, sw, [lp - lps[0] for lp in lps[1:]])
        if c % 100 == 0:
            print(c, len(ids), flush=True)

    def apply(u, g, d):
        base, sw, gains = data[u]
        cand = sorted(((gain + g * (sa - sb), i1, i2, rep) for (i1, i2, rep, sa, sb), gain in zip(sw, gains)
                       if gain + g * (sa - sb) > d), reverse=True)
        used, chosen = set(), []
        for s, i1, i2, rep in cand:
            span = set(range(i1, max(i2, i1 + 1)))
            if span & used:
                continue
            used |= span
            chosen.append((i1, i2, rep))
        out = list(base)
        for i1, i2, rep in sorted(chosen, reverse=True):
            out[i1:i2] = rep
        return join(out)

    refs = {}
    for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        if r["utt_id"] in data:
            refs[r["utt_id"]] = n(r["text"])
    nec = sum(lev(refs[u], n(H["nec"][u])) for u in ids) if "nec" in H else None
    if a.half == "dev" or a.gamma is None:
        grid = {}
        for g in GAMMAS:
            for d in DELTAS:
                grid[(g, d)] = sum(lev(refs[u], n(apply(u, g, d))) for u in ids)
        rv = sum(lev(refs[u], n(join(data[u][0]))) for u in ids)
        best = min(grid, key=grid.get)
        print(f"NECTEC {nec}  ROVER {rv} ({rv / nec - 1:+.3f})  best ROVER+LM {grid[best]} ({grid[best] / nec - 1:+.3f}) at gamma,delta={best}")
        for (g, d), e in sorted(grid.items(), key=lambda x: x[1])[:8]:
            print(f"   gamma {g} delta {d}: {e / nec - 1:+.3f}")
        g, d = best
    else:
        g, d = a.gamma, a.delta
    with (ROOT / "results" / f"{a.tag}_hyps.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for u in ids:
            f.write(json.dumps({"utt_id": u, "condition": "none", "text": apply(u, g, d)}, ensure_ascii=False) + "\n")
    (ROOT / "results" / f"{a.tag}_run.json").write_text(json.dumps(
        {"tag": a.tag, "pool": a.pool, "half": a.half, "gamma": g, "delta": d, "lm": "Qwen/Qwen3-0.6B"}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
