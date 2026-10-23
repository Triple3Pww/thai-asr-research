#!/usr/bin/env python3
"""Rewrite LOTUSDIS labels into the convention the BASE model already writes -> *_conv.jsonl manifests.

    PYTHONUTF8=1 python adapter/lotusdis_convention.py

WHY (2026-10-06, run `deepseek-asr`). Fine-tuning NECTEC on raw LOTUSDIS (`lot1`) cut its in-domain
CER 25% (test 0.203 -> 0.153) and RAISED benchmark CER 0.0695 -> 0.0913. The diff showed a convention
shift, not forgetting -- the `F2`/`F17` trap arriving through training labels. LOTUSDIS transcribes
verbatim and phonetically; NECTEC (and the benchmark) do not. Counts, LOTUSDIS train vs NECTEC's own
output on the 682 benchmark utts:

  | pattern                  | LOTUSDIS | NECTEC |  rule here                               |
  |--------------------------|----------|--------|------------------------------------------|
  | Latin letters            |  27,585  |   678  | DROP the row (19%): loanwords are Thai-   |
  |                          |          |        | script in NECTEC; no safe transliterator  |
  | เค้า / เขา                |  1490/52 |  5/88  | เค้า -> เขา                               |
  | มั้ย / ไหม                |  1601/13 |  3/13  | มั้ย -> ไหม                               |
  | นึง / หนึ่ง                |  666/233 |  4/60  | นึง -> หนึ่ง                               |
  | ๆ                        |      0   |    50  | whitelisted reduplication XX -> "X ๆ"     |
  | เอ่อ                      |    732   |     3  | dropped (filler NECTEC does not write)    |

The target is the BASE MODEL's output, not the benchmark references: the convention is read from
what NECTEC already emits, so the rule set was not tuned on the test labels. Disfluent repeats
(ใช่ใช่, ก็ก็) are left alone -- only words whose doubling is a reduplication are folded, from a
fixed list, because an unsegmented-Thai regex cannot tell the two apart.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

D = Path("E:/Claude/datasets/lotusdis_lav")
LATIN = re.compile(r"[A-Za-z]")
SUBS = [("เค้า", "เขา"), ("มั้ย", "ไหม"), ("นึง", "หนึ่ง")]
REDUP = ["จริง", "ต่าง", "เด็ก", "มาก", "เร็ว", "ช้า", "ดี", "นาน", "ใกล้", "ไกล", "เบา", "ค่อย",
         "บ่อย", "แรก", "หลาย", "ใหม่", "เก่า", "ชัด", "สวย", "ง่าย", "ยาก", "เล็ก", "ใหญ่", "เยอะ", "อื่น"]
REDUP_RE = re.compile("(" + "|".join(sorted(REDUP, key=len, reverse=True)) + r")\1")
FILLER = re.compile(r"\s*เอ่อ\s*")


def conv(t: str) -> str:
    for a, b in SUBS:
        t = t.replace(a, b)
    t = REDUP_RE.sub(r"\1 ๆ ", t)
    t = FILLER.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def main() -> int:
    stats = {}
    for name in ("train", "val", "test"):
        kept = dropped = changed = 0
        with (D / f"manifest_{name}_conv.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for l in (D / f"manifest_{name}.jsonl").open(encoding="utf-8"):
                r = json.loads(l)
                if LATIN.search(r["text"]):
                    dropped += 1
                    continue
                t = conv(r["text"])
                if not t:
                    dropped += 1
                    continue
                changed += t != r["text"]
                r["text"] = t
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
        stats[name] = {"kept": kept, "dropped": dropped, "relabelled": changed}
        print(name, stats[name])
    (D / "stats_conv.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
