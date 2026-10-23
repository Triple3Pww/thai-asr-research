#!/usr/bin/env python3
"""The ceiling of an N-best picker: top-1 vs the best candidate in the list (reads the reference).

    python adapter/nbest_oracle.py results/nbest_turbo5_nbest.jsonl

Writes three hypothesis files beside the input and scores them with `eval_canon.py`:
  _top1     -- the ASR's own first choice (what greedy/beam already gives you)
  _oracleN  -- the candidate closest to the reference, at N = 2, 3, ..., full list
The oracle files READ THE REFERENCE. They are a ceiling for a picker, never a system.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_canon import canon  # noqa: E402
from pilot_rover import BENCH, edit, toks  # noqa: E402


def main() -> int:
    src = Path(sys.argv[1])
    ann = {json.loads(l)["utt_id"]: json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")}
    rows = [json.loads(l) for l in src.open(encoding="utf-8")]
    nmax = max(len(r["cands"]) for r in rows)
    outs = []
    for name, k in [("top1", 1)] + [(f"oracle{k}", k) for k in range(2, nmax + 1)]:
        p = src.with_name(src.stem.replace("_nbest", "") + f"_{name}_hyps.jsonl")
        with p.open("w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                ref = toks(canon(ann[r["utt_id"]]["text"]))
                cands = r["cands"][:k]
                best = min(cands, key=lambda c: edit(ref, toks(canon(c["text"])))) if k > 1 else cands[0]
                f.write(json.dumps({"utt_id": r["utt_id"], "condition": "none", "text": best["text"]},
                                   ensure_ascii=False) + "\n")
        outs.append(str(p))
    in_list = sum(1 for r in rows
                  if any(toks(canon(c["text"])) == toks(canon(ann[r["utt_id"]]["text"])) for c in r["cands"]))
    print(f"utterances where SOME candidate is exactly right: {in_list}/{len(rows)}")
    subprocess.run([sys.executable, str(HERE / "eval_canon.py"), *outs])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
