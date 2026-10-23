#!/usr/bin/env python3
"""Honest ROVER: choose the pool on two source groups, evaluate on the third; paired bootstrap.

    python adapter/pilot_rover_cv.py

THE PROBLEM IT FIXES. `pilot_rover.py` swept pools on all 682 real utterances and reported the
best -- a choice made on the test set. Here the 682 split into the benchmark's three SOURCE groups
(`c-validation` 240, `c-test` 186, `a-*` 256 conversational), the pool is chosen on two of them
and scored on the held-out third, three times. The concatenated held-out hypotheses are then a
system whose every utterance was decoded by a pool that never saw that utterance's group.

Metric: CER at the F17 `num` rung (the reference re-rendered into whichever number convention the
hypothesis used, identically for every system), computed on the scorer's own tokens. A paired
bootstrap over utterances (2,000 draws) gives P(combined CER >= Typhoon CER).
Final numbers for the record still go through `_score.py` via `pilot_rescore_num.py`.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _ortho import build_form_index, rewrite_reference  # noqa: E402
from pilot_mbr_combine import POOL, load  # noqa: E402
from pilot_rover import BENCH, align, edit, join, toks  # noqa: E402

EPS = ""


def group(u: str) -> str:
    return "c-validation" if u.startswith("c-validation") else "c-test" if u.startswith("c-test") else "a"


class Rover:
    """ROVER with every pairwise distance and alignment cached across pools."""

    def __init__(self, tk):
        self.tk, self._d, self._a = tk, {}, {}

    def d(self, a, b, u):
        if a == b:
            return 0
        k = (min(a, b), max(a, b), u)
        if k not in self._d:
            self._d[k] = edit(self.tk[a][u], self.tk[b][u])
        return self._d[k]

    def al(self, p, k, u):
        key = (p, k, u)
        if key not in self._a:
            self._a[key] = align(self.tk[p][u], self.tk[k][u])
        return self._a[key]

    def run(self, pool, u, order):
        pivot = min(pool, key=lambda k: (sum(self.d(k, o, u) for o in pool), order.index(k)))
        P = self.tk[pivot][u]
        slot = [{} for _ in P]
        gap = [{} for _ in range(len(P) + 1)]
        add = lambda dct, t: dct.__setitem__(t, dct.get(t, 0) + 1)  # noqa: E731
        for k in pool:
            if k == pivot:
                for i, t in enumerate(P):
                    add(slot[i], t)
                for g in gap:
                    add(g, EPS)
                continue
            ins, cur = {}, 0
            for r, h in self.al(pivot, k, u):
                if r is None:
                    ins.setdefault(cur, []).append(h)
                else:
                    add(slot[r], EPS if h is None else h)
                    cur = r + 1
            for gi in range(len(P) + 1):
                add(gap[gi], " ".join(ins[gi]) if gi in ins else EPS)

        def pick(v, default):
            b = max(v.values())
            w = [t for t, c in v.items() if c == b]
            return default if default in w else sorted(w)[0]

        out = []
        for i in range(len(P) + 1):
            g = pick(gap[i], EPS)
            if g:
                out.extend(g.split(" "))
            if i < len(P):
                t = pick(slot[i], P[i])
                if t:
                    out.append(t)
        return out


def main() -> int:
    ann = {a["utt_id"]: a for a in map(json.loads, (BENCH / "data/utterances.jsonl").open(encoding="utf-8"))}
    real = [u for u, a in ann.items() if a["split"] == "real"]
    fi = build_form_index(list(ann.values()))
    names = [k for k in POOL if POOL[k].exists()]
    hyps = {k: load(POOL[k], set(real)) for k in names}
    ids = sorted(set.intersection(*(set(h) for h in hyps.values())))
    tk = {k: {u: toks(hyps[k][u]) for u in ids} for k in names}
    R = Rover(tk)

    def err(text_by_u, us):
        """per-utterance (errors, ref_len) at the num rung"""
        out = {}
        for u in us:
            ref = rewrite_reference(ann[u]["text"], text_by_u[u], ann[u].get("entities") or [], "num", fi)
            r = toks(ref)
            out[u] = (edit(r, toks(text_by_u[u])), len(r))
        return out

    cer = lambda e, us: sum(e[u][0] for u in us) / sum(e[u][1] for u in us)  # noqa: E731
    pools = [p for n in range(3, len(names) + 1) for p in itertools.combinations(names, n)]
    print(f"{len(ids)} utts, {len(names)} systems, {len(pools)} candidate pools")

    groups = sorted({group(u) for u in ids})
    # every pool's per-utterance output and error, once
    pool_err = {}
    pool_txt = {}
    for i, p in enumerate(pools):
        txt = {u: join(R.run(p, u, names)) for u in ids}
        pool_txt[p], pool_err[p] = txt, err(txt, ids)
        if i % 20 == 0:
            print(f"  pools {i}/{len(pools)}", flush=True)
    single_err = {k: err(hyps[k], ids) for k in names}

    held_txt, chosen = {}, {}
    for g in groups:
        dev = [u for u in ids if group(u) != g]
        test = [u for u in ids if group(u) == g]
        best = min(pools, key=lambda p: cer(pool_err[p], dev))
        chosen[g] = best
        for u in test:
            held_txt[u] = pool_txt[best][u]
        print(f"held-out {g:13s} n={len(test):3d}: chose {'+'.join(best):30s} "
              f"dev {cer(pool_err[best], dev):.4f} | test {cer(pool_err[best], test):.4f} "
              f"vs tw {cer(single_err['tw'], test):.4f}")
    held_err = err(held_txt, ids)

    rng = random.Random(2026)
    def paired(a, b, draws=2000):
        worse = 0
        for _ in range(draws):
            s = [rng.choice(ids) for _ in ids]
            if sum(a[u][0] for u in s) / sum(a[u][1] for u in s) >= sum(b[u][0] for u in s) / sum(b[u][1] for u in s):
                worse += 1
        return worse / draws

    rows = {"CV-chosen ROVER (held-out)": held_err, "ROVER all-7 (no selection)": pool_err[tuple(names)]}
    rows.update({k: single_err[k] for k in names})
    print(f"\n{'system':32s} {'CER@num':>8s} {'P(>= tw)':>9s} {'P(>= pat)':>9s}")
    for k, e in sorted(rows.items(), key=lambda kv: cer(kv[1], ids)):
        print(f"{k:32s} {cer(e, ids):8.4f} {paired(e, single_err['tw']):9.4f} {paired(e, single_err['pat']):9.4f}")

    out = HERE.parent / "results/pilot_rover_cv.json"
    out.write_text(json.dumps({"chosen": {g: list(p) for g, p in chosen.items()},
                               "cer_num": {k: cer(e, ids) for k, e in rows.items()}}, indent=1),
                   encoding="utf-8")
    with (HERE.parent / "results/rover_cv_heldout_hyps.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for u in ids:
            f.write(json.dumps({"utt_id": u, "condition": "none", "text": held_txt[u]}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
