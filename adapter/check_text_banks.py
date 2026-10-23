#!/usr/bin/env python3
"""Audit a generated text bank. Read this output before spending GPU hours speaking it.

    python adapter/check_text_banks.py adapter/texts/phase1_known.json
    python adapter/check_text_banks.py adapter/texts/phase2_rare.json --samples 15

`make_text_banks.py` validates every line as it arrives, so a bank on disk has already passed
the per-line rules. This checks the properties that are only visible ACROSS the bank -- coverage,
repetition, and whether one batch quietly dominates -- plus it prints samples, because the
judgement that matters ("is this natural Thai?") is not automatable and nothing here pretends
otherwise.

🔴 What this CANNOT check, and what therefore stays unverified:
  - whether a place name is a REAL Thai place. The postal prefix check is a range test only.
  - whether the spoken form is how a Thai person would actually say it.
Both need a Thai speaker's eye. That is what `--samples` is for.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

THAI = re.compile(r"[฀-๿]")
DIGIT = re.compile(r"[0-9๐-๙]")
LATIN = re.compile(r"[A-Za-z]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bank", type=Path)
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()

    rows = json.loads(args.bank.read_text(encoding="utf-8"))
    print(f"{args.bank.name}: {len(rows)} entries\n")

    texts = [r["text"] for r in rows]
    lens = [len(t) for t in texts]
    print("TEXT (what the TTS speaks)")
    print(f"  chars      min {min(lens)}  median {int(statistics.median(lens))}  max {max(lens)}")
    print(f"  unique     {len(set(texts))} / {len(texts)}")
    print(f"  with digits {sum(bool(DIGIT.search(t)) for t in texts)}   "
          f"with Latin {sum(bool(LATIN.search(t)) for t in texts)}   "
          f"no Thai {sum(not THAI.search(t) for t in texts)}")

    # Repetition that per-line validation cannot see: a bank can be 400 unique strings and still
    # be eight templates. A shared 5-gram prefix is the cheap tell.
    heads = collections.Counter(t[:18] for t in texts)
    dom = [(h, c) for h, c in heads.most_common(5) if c > 1]
    if dom:
        print("  repeated openings (first 18 chars):")
        for h, c in dom:
            print(f"    {c:3d}x  {h}")
    else:
        print("  repeated openings: none")

    if "theme" in rows[0]:
        print("\nBY THEME")
        for theme, c in collections.Counter(r["theme"] for r in rows).most_common():
            print(f"  {c:4d}  {theme[:66]}")

    if "entity" in rows[0]:
        ents = [r["entity"] for r in rows]
        print("\nENTITIES (what phase 2 is teaching)")
        print(f"  distinct   {len(set(ents))} / {len(ents)}")
        print(f"  by kind    {dict(collections.Counter(r['entity_kind'] for r in rows))}")
        # every entity must be recoverable from the label, or Sec 8 cannot score it
        missing = [r for r in rows if r["entity"] not in r["label_text"]]
        print(f"  entity absent from label_text: {len(missing)}")
        # the spoken form must be speakable: no digits anywhere in it
        spoken_digits = [r for r in rows if DIGIT.search(r["text"])]
        print(f"  digits in the SPOKEN field:    {len(spoken_digits)}  (must be 0)")
        # and the label must differ from the spoken form exactly where a number exists
        codes = [r for r in rows if re.search(r"\d", r["label_text"])]
        print(f"  rows whose label carries digits: {len(codes)} "
              f"(these are the normalisation lesson)")
        pre = collections.Counter(r["entity"][:2] for r in rows
                                 if re.fullmatch(r"\d{5}", r["entity"]))
        if pre:
            print(f"  postal province prefixes seen: {len(pre)} distinct -> "
                  f"{sorted(pre)[:12]}{' ...' if len(pre) > 12 else ''}")

    print(f"\nSAMPLES (judge these yourself -- no automated check covers naturalness)")
    step = max(1, len(rows) // args.samples)
    for r in rows[::step][: args.samples]:
        if "entity" in r:
            print(f"\n  entity : {r['entity']}  [{r['entity_kind']}]")
            print(f"  spoken : {r['text']}")
            print(f"  label  : {r['label_text']}")
        else:
            print(f"  {r['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
