#!/usr/bin/env python3
"""Shared pieces of the trained N-best picker: regions with source tags, prompts, oracle labels.

His design (2026-10-04): the ASRs return their top possibilities and a fast model PICKS (Jev-style:
structured choice, no free text). The zero-shot pilot (`pilot_picker.py`) showed the hard part is a
learned judgement -- WHEN the weaker model's minority answer is right -- so this module turns
human-labelled audio (Porjai, never the benchmark) into supervised examples.

A region is a word-sized span where the merged candidates (turbo top-5 + typhoon-asr-realtime top-5)
disagree (`pilot_picker.regions` grid: MBR pivot, glue <= 2 tokens, widened by 3 tokens per side).
Each option is tagged with how many of each model's candidates wrote it and whether it was that
model's top-1 -- the evidence a picker needs. Options are ordered by posterior-weighted support, so
option 0 is the vote's answer and a picker that always says 0 IS the vote.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_canon import canon  # noqa: E402
from pilot_rover import align, edit, join, toks  # noqa: E402

PROMPT_HEAD = ("เลือกคำที่ถูกต้องในแต่ละตำแหน่ง [หมายเลข: ตัวเลือก...] "
               "T=ระบบ turbo R=ระบบ realtime N=ระบบ nectec ตัวเลขคือจำนวนผลลัพธ์ที่เขียนแบบนั้น ★=คำตอบอันดับหนึ่งของระบบนั้น\n")


def softmax_scores(cands):
    """CTranslate2 scores are length-normalised log-probs, NeMo's are totals; make both totals."""
    tot = [c["score"] * max(len(toks(c["text"])), 1) if -1.0 < c["score"] <= 0 else c["score"] for c in cands]
    m = max(tot)
    e = [math.exp(t - m) for t in tot]
    z = sum(e)
    return [x / z for x in e]


LETTERS = "TRN"   # T = typhoon-whisper-turbo, R = typhoon-asr-realtime, N = NECTEC Pathumma FastConformer


def build(*lists, glue=2, pad=3, canonical=False):
    """Return (parts, regions) over any number of N-best lists, tagged T, R, N in order.
    parts: list of ('fixed', tokens) | ('spot', idx). regions: list of option lists; each option =
    dict(tokens, w, and per source letter X: X = count, Xtop = was X's top-1).

    canonical=True runs `eval_canon.canon` first (digit runs -> Thai number words). Needed once
    NECTEC is a source: it writes digits (77/682 benchmark clips) where Typhoon writes words, and
    without this a convention difference would show up as a disagreement region. The 2-source
    picker p1 was built with canonical=False; keep it that way to reproduce p1."""
    norm = (lambda s: toks(canon(s))) if canonical else toks
    srcs, cands, w = [], [], []
    for L, cl in zip(LETTERS, lists):
        srcs += [(L, i) for i in range(len(cl))]
        cands += [norm(c["text"]) for c in cl]
        w += softmax_scores(cl)
    letters = LETTERS[: len(lists)]
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
    dis = [len({row[s] for row in grid}) > 1 for s in range(nslot)]
    groups = []
    for s in (s for s in range(nslot) if dis[s]):
        if groups and sum(len(grid[pivot][t]) for t in range(groups[-1][1] + 1, s)) <= glue:
            groups[-1][1] = s
        else:
            groups.append([s, s])
    wide = []
    for a, b in groups:
        n = 0
        while a > 0 and n < pad:
            a -= 1; n += len(grid[pivot][a])
        n = 0
        while b < nslot - 1 and n < pad:
            b += 1; n += len(grid[pivot][b])
        if wide and a <= wide[-1][1] + 1:
            wide[-1][1] = max(wide[-1][1], b)
        else:
            wide.append([a, b])
    parts, regs, pos = [], [], 0
    for a, b in wide:
        fixed = [t for s in range(pos, a) for t in grid[pivot][s]]
        if fixed:
            parts.append(("fixed", fixed))
        opts = {}
        for (m, rank), wj, row in zip(srcs, w, grid):
            key = tuple(t for s in range(a, b + 1) for t in row[s])
            o = opts.get(key)
            if o is None:
                o = opts[key] = {"tokens": list(key), "w": 0.0}
                for L in letters:
                    o[L], o[L + "top"] = 0, False
            o[m] += 1
            o["w"] += wj
            if rank == 0:
                o[m + "top"] = True
        parts.append(("spot", len(regs)))
        regs.append(sorted(opts.values(), key=lambda o: (-o["w"], o["tokens"])))
        pos = b + 1
    tail = [t for s in range(pos, nslot) for t in grid[pivot][s]]
    if tail:
        parts.append(("fixed", tail))
    return parts, regs


def render(parts, regs) -> str:
    out = []
    for kind, x in parts:
        if kind == "fixed":
            out.append(join(x))
        else:
            ops = []
            for j, o in enumerate(regs[x]):
                tag = " ".join(f"{L}{o[L]}{'★' if o[L + 'top'] else ''}" for L in LETTERS if L in o)
                ops.append(f"{j}) {join(o['tokens']) or '(ว่าง)'} <{tag}>")
            out.append(f"[{x + 1}: " + " | ".join(ops) + "]")
    return PROMPT_HEAD + " ".join(out)


def assemble(parts, regs, choice) -> str:
    words = []
    for kind, x in parts:
        if kind == "fixed":
            words.extend(x)
        else:
            i = choice[x] if x < len(choice) and 0 <= choice[x] < len(regs[x]) else 0
            words.extend(regs[x][i]["tokens"])
    return join(words)


def oracle_choice(parts, regs, ref_text):
    """Coordinate-wise label: for each region, the option that minimises sentence edit distance to
    the reference with every other region at its current choice (two passes). Ties keep the vote."""
    ref = toks(canon(ref_text))
    ch = [0] * len(regs)
    for _ in range(2):
        for k in range(len(regs)):
            best, best_d = ch[k], None
            for j in range(len(regs[k])):
                trial = ch[:]; trial[k] = j
                d = edit(ref, toks(canon(assemble(parts, regs, trial))))
                if best_d is None or d < best_d:
                    best, best_d = j, d
            ch[k] = best
    return ch
