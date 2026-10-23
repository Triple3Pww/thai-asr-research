#!/usr/bin/env python3
"""Typhoon's keyword spotter (`E:/asr-bench/nemo-ctc/kws_spot.py: spot`), ported to numba -- same answers, fast.

    from kws_fast import spot_fast          # drop-in for kws_spot.spot(logp, edges, K, blank, max_gap)
    python adapter/kws_fast.py --check      # prove equality against the original on real posteriors

WHY (2026-10-07, `F36`). The one-pass name splice spends 154 of its 192 ms per turn in the original
pure-Python token-passing Viterbi, run once per list entry (50 per turn). The model side is already
one encoder pass; this is the last real-time blocker, and it is CPU code, not model.

FIDELITY, not approximation. Every quirk of the original is kept, in the same order: the start
state that restarts at 0 whenever negative, the gap kill before each frame, edges visited u-ascending
then v-ascending with strict `>`, the stay transition on blank OR a repeat of the piece that reached
the state, the reset of the final state after each completion. The `last` tuple of piece ids becomes
an edge index (-1 = blank), which is exact because a piece set's stay score is the max over its ids =
the edge's emission. Arithmetic matches too: emission minus filler in float32 (numpy scalars in the
original), accumulated in float64. `--check` asserts bit-identical (score, t_start, t_end).
"""

from __future__ import annotations

import numpy as np
from numba import njit

NEG = -1e9


def _edge_arrays(edges):
    """Original lattice (list over u of [(v, pids), ...]) -> flat arrays, in the original visiting order."""
    eu, ev, pids = [], [], []
    for u, out in enumerate(edges):
        for v, p in out:
            eu.append(u); ev.append(v); pids.append(p)
    return np.array(eu, np.int64), np.array(ev, np.int64), pids


@njit(cache=True)
def _viterbi(r, lpb, filler, eu, ev, K, max_gap):
    T = r.shape[0]
    E = eu.shape[0]
    cur = np.full(K + 1, NEG); cur[0] = 0.0
    start = np.full(K + 1, -1, dtype=np.int64)
    last = np.full(K + 1, -1, dtype=np.int64)         # -1 = (blank,)
    gap = np.zeros(K + 1, dtype=np.int64)
    best_s, best_a, best_b = NEG, -1, -1
    for t in range(T):
        if max_gap >= 0:
            for u in range(1, K + 1):
                if gap[u] > max_gap:
                    cur[u] = NEG
        nxt = cur.copy(); nstart = start.copy(); nlast = last.copy()
        ngap = gap + 1
        ngap[0] = 0
        if nxt[0] < 0.0:
            nxt[0] = 0.0; nstart[0] = t; nlast[0] = -1
        e = 0
        for u in range(K):
            # edges are stored u-ascending; skip this u's block when the state is dead
            if cur[u] <= NEG / 2 and u != 0:
                while e < E and eu[e] == u:
                    e += 1
                continue
            base = cur[u] if u > 0 else max(cur[0], 0.0)
            if u > 0:
                s0 = start[u]
            else:
                s0 = start[0] if cur[0] > 0 else t
            while e < E and eu[e] == u:
                v = ev[e]
                sc = base + (r[t, e] - filler[t])          # float32 diff, float64 sum -- as the original
                if sc > nxt[v]:
                    nxt[v] = sc; nstart[v] = s0; nlast[v] = e
                    ngap[v] = 0
                e += 1
        for u in range(1, K + 1):
            if cur[u] > NEG / 2:
                rl = lpb[t] if last[u] < 0 else r[t, last[u]]
                m = lpb[t] if lpb[t] >= rl else rl
                sc = cur[u] + (m - filler[t])
                if sc > nxt[u]:
                    nxt[u] = sc; nstart[u] = start[u]; nlast[u] = last[u]
        cur = nxt; start = nstart; last = nlast; gap = ngap
        if cur[K] > NEG / 2:
            norm = cur[K] / max(1, K)
            if norm > best_s:
                best_s, best_a, best_b = norm, start[K], t
            cur[K] = NEG
    return best_s, best_a, best_b


def spot_fast(logp: np.ndarray, edges, K: int, blank: int = 2048, max_gap: int | None = 6):
    """Drop-in for kws_spot.spot. logp: [T, V+1] float32."""
    logp = np.ascontiguousarray(logp, dtype=np.float32)
    eu, ev, pids = _edge_arrays(edges)
    r = np.empty((logp.shape[0], len(pids)), dtype=np.float32)
    for j, p in enumerate(pids):
        r[:, j] = logp[:, list(p)].max(axis=1)
    filler = logp.max(axis=1)
    s, a, b = _viterbi(r, np.ascontiguousarray(logp[:, blank]), filler, eu, ev, K,
                       -1 if max_gap is None else int(max_gap))
    return (s, int(a), int(b))


def _check(n_clips: int = 300) -> int:
    """Bit-identical vs the original on real posteriors: CTC head h3 on cached LOTUSDIS-val features,
    keywords = real benchmark bias-list entries (mostly absent) + substrings of the clip's own text."""
    import gzip, json, random, sys, time
    from pathlib import Path
    import torch
    sys.path.insert(0, "E:/asr-bench/nemo-ctc")
    from kws_spot import build_lattice, spot
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.restore_from(
        "E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo", map_location="cpu")
    sp = m.tokenizer.tokenizer
    piece_ids = {}
    for i in range(sp.get_piece_size()):
        p = sp.id_to_piece(i).lstrip("\u2581")
        piece_ids[p] = piece_ids.get(p, ()) + (i,)
    V = m.tokenizer.vocab_size
    ck = torch.load(Path(__file__).resolve().parent.parent / "checkpoints/ctchead_h3.pt", map_location="cpu")
    head = torch.nn.Sequential(torch.nn.Linear(ck["d"], ck["hidden"]), torch.nn.GELU(), torch.nn.Linear(ck["hidden"], V + 1))
    head.load_state_dict(ck["state"]); head.eval()
    va = torch.load("E:/Claude/datasets/ctchead_cache/lot_val.pt")
    ents = []
    with gzip.open("E:/asr-bench/thai-contextasr-bench/data/bias_lists.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("condition") == "bias@50":
                ents += d.get("entries") or []
    rng = random.Random(0)
    norm = lambda kw: "".join(sp.encode(kw, out_type=str)).replace("\u2581", "")
    n = mism = 0; t_old = t_new = 0.0
    for feats, text in list(zip(va["feats"], va["texts"]))[:n_clips]:
        with torch.no_grad():
            lp = torch.log_softmax(head(feats.float()), -1).numpy()
        t = text.replace(" ", "")
        kws = rng.sample(ents, 8) + ([t[i: i + rng.randint(3, 10)] for i in rng.sample(range(max(1, len(t) - 3)), min(4, max(1, len(t) - 3)))])
        for kw in kws:
            k = norm(kw.replace(" ", ""))
            if not k:
                continue
            edges, ok, _ = build_lattice(k, piece_ids)
            if not ok:
                continue
            a0 = time.perf_counter(); a = spot(lp, edges, len(k), V, max_gap=6); a1 = time.perf_counter()
            b = spot_fast(lp, edges, len(k), V, max_gap=6); a2 = time.perf_counter()
            t_old += a1 - a0; t_new += a2 - a1
            n += 1
            if (float(a[0]), int(a[1]), int(a[2])) != (float(b[0]), b[1], b[2]):
                mism += 1
                if mism <= 5:
                    print("MISMATCH", repr(kw), a, b)
    print(f"{n} (posterior, keyword) pairs; mismatches {mism}; original {1000*t_old/n:.2f} ms/kw, "
          f"fast {1000*t_new/n:.3f} ms/kw (incl. first-call JIT)")
    return 1 if mism else 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--clips", type=int, default=300)
    a = ap.parse_args()
    raise SystemExit(_check(a.clips) if a.check else 0)
