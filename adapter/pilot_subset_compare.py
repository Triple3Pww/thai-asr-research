#!/usr/bin/env python3
"""Compare several hypothesis files on EXACTLY the utterances the first one covers, at none + num.

    python adapter/pilot_subset_compare.py results/ger_x_hyps.jsonl results/rover_..._hyps.jsonl tw pat

A partial pilot (e.g. 90 GER utterances) must be compared against the baselines on the same 90,
never against their 682-utterance numbers. Short names resolve through `pilot_mbr_combine.POOL`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _ortho import build_form_index  # noqa: E402
from ortho_rescore import BENCH, score_one  # noqa: E402
from pilot_mbr_combine import POOL  # noqa: E402


def main() -> int:
    paths = [Path(a) if a not in POOL else POOL[a] for a in sys.argv[1:]]
    first = {json.loads(l)["utt_id"] for l in paths[0].open(encoding="utf-8") if l.strip()}
    ann = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8") if l.strip()]
    keep = {a["utt_id"] for a in ann if a["split"] == "real"} & first
    fi = build_form_index(ann)
    tmp = Path(tempfile.mkdtemp())
    print(f"n = {len(keep)} utterances (the first file's coverage)")
    for name, p in zip(sys.argv[1:], paths):
        r = {g: score_one(name, p, ann, keep, g, fi, tmp, "dp") for g in ("none", "num")}
        print(f"{Path(name).stem[:46]:46s} none {r['none']['cer']:.4f}  num {r['num']['cer']:.4f}  "
              f"U {r['num']['u_cer']:.4f}  B {r['num']['b_cer']:.4f}  recall {r['num']['entity_recall']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
