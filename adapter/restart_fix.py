#!/usr/bin/env python3
"""Drop read-speech FALSE STARTS from a hypothesis: "...A X | A Y..." -> "...A Y..." when a run A of >= MIN chars
recurs within GAP chars of its own end.

    PYTHONUTF8=1 python adapter/restart_fix.py results/X_bench_hyps.jsonl results/X_rs_bench_hyps.jsonl

WHY (2026-10-11, loop iteration 8). The best model's worst dev utterances are FLEURS readings where the reader
stumbled and re-read ("...และความเชื่อว่าประเทศของและความเชื่อว่าประเทศเป็นของ..."): the model writes what was
said, the reference is the clean prompt, and every false start costs 10-50 insertions. The thresholds are
fixed a priori (MIN 8 chars ~ 2-3 Thai words, GAP 30 chars), NOT tuned on the dev half; the rule keeps the
LAST reading, the one the reader finished. Spaces are ignored for matching (NECTEC writes few).
"""

from __future__ import annotations

import json
import sys

MIN, GAP = 18, 30


def fix(t: str) -> str:
    s = t.replace(" ", "")
    changed = True
    while changed:
        changed = False
        n = len(s)
        best = None
        # earliest i, longest A starting at i, whose copy starts at j with i + len(A) <= j <= i + len(A) + GAP
        for i in range(n - 2 * MIN + 1):
            for L in range(min((n - i) // 2, 60), MIN - 1, -1):
                a = s[i:i + L]
                lo, hi = i + L, min(i + L + GAP, n - L)
                j = s.find(a, lo, hi + L)
                if j != -1 and j <= hi:
                    best = (i, j)
                    break
            if best:
                break
        if best:
            i, j = best
            s = s[:i] + s[j:]
            changed = True
    return s if s != t.replace(" ", "") else t


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    n = 0
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for l in open(src, encoding="utf-8"):
            r = json.loads(l)
            new = fix(r["text"])
            n += new != r["text"]
            r["text"] = new
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{n} hypotheses changed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
