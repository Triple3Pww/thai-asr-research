#!/usr/bin/env python3
"""Pilot: an LLM picks at each spot where two ASRs' N-best candidates disagree (structured output).

    python adapter/pilot_picker.py --model qwen3.5:9b --limit 120 --tag picker_q35
    python adapter/pilot_picker.py --vote-only --tag picker_vote          # the control, no LLM

His idea (2026-10-04): the ASR returns its top possibilities and an LLM picks the right one in
context; then "try both" -- merge two models' lists, and pick per spot instead of per sentence.
The perfect-picker ceilings (`nbest_lattice_oracle.py`, own tokeniser): turbo top-1 0.0916 ->
merged per-spot 0.0636, bar 0.0838. This measures what a REAL picker gets.

Mechanics, Jev-style: the LLM never writes free text. The confusion network of the merged lists
(turbo top-5 + typhoon-asr-realtime top-5, `nbest_lattice_oracle.network`) is rendered as the
sentence with each disagreement shown as `[k: a | b | c]`, options ordered by how many of the 10
candidates voted for them (that count is shown -- it is the ASR's own confidence signal). The LLM
returns JSON {"k": option_index}. Anything missing or invalid falls back to the top-voted option, so
the picker can never produce text outside the candidates.

The CONTROL (--vote-only) takes the top-voted option at every spot: plain ROVER over the same 10.
If the LLM does not beat the vote, it is adding nothing but latency. Label-free: the reference is
read only by the scorer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot_rover import BENCH, align, edit, join, toks  # noqa: E402

RES = HERE.parent / "results"

PROMPT = """ด้านล่างคือประโยคภาษาไทยที่ได้จากการถอดเสียง ตำแหน่งที่ระบบไม่แน่ใจแสดงเป็น [หมายเลข: ตัวเลือก0 | ตัวเลือก1 | ...] พร้อมจำนวนเสียงโหวตในวงเล็บ
เลือกตัวเลือกที่ทำให้ประโยคถูกต้องและเป็นธรรมชาติที่สุดในแต่ละตำแหน่ง (ชื่อเฉพาะ คำสะกด ความหมายตามบริบท)
ตอบเป็น JSON เท่านั้น รูปแบบ {{"หมายเลขตำแหน่ง": เลขตัวเลือก}} เช่น {{"1": 0, "2": 1}}

ประโยค: {sent}"""


def regions(cands, glue: int = 2, weights=None):
    """Word-sized choices. Every candidate is laid out on the same slot grid (gap, token, gap, ...
    around the MBR pivot); slots where candidates disagree are grouped into REGIONS, absorbing any
    unanimous run of <= `glue` tokens between them (a Thai word straddles several characters, so
    a one-letter choice like [โ | ั | ว] is unjudgeable -- measured 2026-10-04: the LLM kept the
    vote on 14/14 such spots). Options are the strings the candidates actually wrote across the
    whole region, with how many of them wrote it."""
    w = weights or [1.0] * len(cands)
    pivot = min(range(len(cands)), key=lambda i: (sum(wj * edit(cands[i], c) for wj, c in zip(w, cands)), i))
    P = cands[pivot]
    nslot = 2 * len(P) + 1
    grid = []
    for c in cands:
        row = [() for _ in range(nslot)]
        ins, cur = defaultdict(list), 0
        for r, h in align(P, c):
            if r is None:
                ins[cur].append(h)
            else:
                row[2 * r + 1] = () if h is None else (h,)
                cur = r + 1
        for g, run in ins.items():
            row[2 * g] = tuple(run)
        grid.append(row)
    agree = [len({row[s] for row in grid}) == 1 for s in range(nslot)]
    # mark disagreeing slots, then glue across short unanimous runs
    dis = [not a for a in agree]
    idx = [s for s in range(nslot) if dis[s]]
    groups = []
    for s in idx:
        if groups and sum(len(grid[0][t]) for t in range(groups[-1][1] + 1, s)) <= glue:
            groups[-1][1] = s
        else:
            groups.append([s, s])
    # widen each region by >= `pad` unanimous tokens per side, so an option reads as a syllable /
    # word (พันธุ์ vs พันธ์) instead of a bare mark ([ุ | ]), then merge regions that now touch
    pad = 3
    wide = []
    for a, b in groups:
        n = 0
        while a > 0 and n < pad:
            a -= 1; n += len(grid[0][a])
        n = 0
        while b < nslot - 1 and n < pad:
            b += 1; n += len(grid[0][b])
        if wide and a <= wide[-1][1] + 1:
            wide[-1][1] = max(wide[-1][1], b)
        else:
            wide.append([a, b])
    groups = wide
    out, pos = [], 0
    for a, b in groups:
        fixed = [t for s in range(pos, a) for t in grid[0][s]]
        if fixed:
            out.append(("fixed", fixed))
        opts = Counter()
        for wj, row in zip(w, grid):
            opts[tuple(t for s in range(a, b + 1) for t in row[s])] += wj
        out.append(("spot", sorted(opts.items(), key=lambda kv: (-kv[1], kv[0]))))
        pos = b + 1
    tail = [t for s in range(pos, nslot) for t in grid[0][s]]
    if tail:
        out.append(("fixed", tail))
    return out


def network_votes(cands):
    """Confusion network with vote counts: list of (Counter{option_tuple: votes}) per slot."""
    pivot = min(range(len(cands)), key=lambda i: (sum(edit(cands[i], c) for c in cands), i))
    P = cands[pivot]
    slot = [Counter() for _ in P]
    gap = [Counter() for _ in range(len(P) + 1)]
    for c in cands:
        ins, cur = defaultdict(list), 0
        for r, h in align(P, c):
            if r is None:
                ins[cur].append(h)
            else:
                slot[r][() if h is None else (h,)] += 1
                cur = r + 1
        for g in range(len(P) + 1):
            gap[g][tuple(ins[g]) if g in ins else ()] += 1
    net = []
    for i in range(len(P) + 1):
        net.append(gap[i])
        if i < len(P):
            net.append(slot[i])
    return net


def spans(net):
    """Merge runs of unanimous slots into fixed text; return list of ('fixed', tokens) | ('spot', [(opt, votes)])."""
    out, run = [], []
    for c in net:
        if len(c) == 1:
            run.extend(next(iter(c)))
        else:
            if run:
                out.append(("fixed", run)); run = []
            out.append(("spot", sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))))
    if run:
        out.append(("fixed", run))
    # merge adjacent spots into one choice over their joint candidates is a refinement; keep slots as-is
    return out


def posterior(lists, u):
    """Per-model softmax over candidate scores. CTranslate2 reports a length-normalised log-prob,
    NeMo a total one, so the CT2 score is scaled by the candidate's length to make both totals."""
    import math
    out = []
    for L in lists:
        cs = L[u]
        tot = [c["score"] * max(len(toks(c["text"])), 1) if -1.0 < c["score"] <= 0 else c["score"] for c in cs]
        m = max(tot)
        e = [math.exp(t - m) for t in tot]
        z = sum(e)
        out.extend(x / z for x in e)
    return out


def ask(model, sent):
    body = json.dumps({"model": model, "prompt": PROMPT.format(sent=sent), "stream": False, "think": False,
                       "format": "json", "options": {"temperature": 0, "num_predict": 200, "seed": 2026}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(json.loads(r.read())["response"] or "{}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", nargs="+", default=[str(RES / "nbest_turbo5_nbest.jsonl"), str(RES / "nbest_rt5_nbest.jsonl")])
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--vote-only", action="store_true")
    ap.add_argument("--weights", default="count", choices=("count", "posterior"),
                    help="count: every candidate is one vote (2026-10-04 run: the vote LOST to turbo top-1, 0.0841 vs "
                         "0.0810 -- a model's 5th near-copy counted as much as its best). posterior: each model's "
                         "list is a softmax over its own scores, so each MODEL sums to 1 and its top answer dominates.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    lists = [{json.loads(l)["utt_id"]: json.loads(l)["cands"] for l in open(p, encoding="utf-8")} for p in args.lists]
    ids = sorted(set.intersection(*(set(x) for x in lists)))
    if args.limit:
        ids = ids[:: max(1, len(ids) // args.limit)][: args.limit]
    out_p = RES / f"{args.tag}_hyps.jsonl"
    stats = Counter()
    lat = []
    with out_p.open("w", encoding="utf-8", newline="\n") as f:
        for n, u in enumerate(ids):
            cands = [toks(c["text"]) for L in lists for c in L[u]]
            wts = posterior(lists, u) if args.weights == "posterior" else None
            sp = regions(cands, weights=wts)
            spots = [s for s in sp if s[0] == "spot"]
            choice = {}
            if spots and not args.vote_only:
                parts, k = [], 0
                for kind, x in sp:
                    if kind == "fixed":
                        parts.append(join(x))
                    else:
                        k += 1
                        parts.append(f"[{k}: " + " | ".join(f"{join(list(o)) or '(ไม่มี)'}({v:.2g})" for o, v in x) + "]")
                t0 = time.perf_counter()
                try:
                    got = ask(args.model, " ".join(parts))
                except Exception:  # noqa: BLE001
                    got = {}
                    stats["llm_error"] += 1
                lat.append(time.perf_counter() - t0)
                for key, val in (got or {}).items():
                    try:
                        choice[int(key)] = int(val)
                    except (TypeError, ValueError):
                        stats["bad_value"] += 1
            words, k = [], 0
            for kind, x in sp:
                if kind == "fixed":
                    words.extend(x)
                else:
                    k += 1
                    i = choice.get(k, 0)
                    if not 0 <= i < len(x):
                        i = 0; stats["out_of_range"] += 1
                    stats["spots"] += 1
                    stats["llm_overrode_vote"] += i != 0
                    words.extend(x[i][0])
            f.write(json.dumps({"utt_id": u, "condition": "none", "text": join(words)}, ensure_ascii=False) + "\n")
            if n % 50 == 0:
                print(f"  {n}/{len(ids)}  {dict(stats)}", flush=True)
    run = {"tag": args.tag, "model": None if args.vote_only else args.model, "n": len(ids), "stats": dict(stats),
           "llm_latency_ms_p50": round(1000 * sorted(lat)[len(lat) // 2], 1) if lat else None}
    (RES / f"{args.tag}_run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(run, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
