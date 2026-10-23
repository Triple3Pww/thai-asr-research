#!/usr/bin/env python3
"""Build picker examples: prompt (rendered regions) -> target (choices), from N-best lists.

    python adapter/picker_data.py --turbo results/porjai_tr_turbo5_nbest.jsonl --rt results/porjai_tr_rt5_nbest.jsonl \
        --manifest E:/Claude/datasets/porjai/manifest_train.jsonl --out E:/Claude/datasets/porjai/picker_train.jsonl
    python adapter/picker_data.py --turbo results/nbest_turbo5_nbest.jsonl --rt results/nbest_rt5_nbest.jsonl \
        --out results/picker_bench_prompts.jsonl            # benchmark: prompts only, NO labels

With --manifest the human transcript gives an oracle label per region (`picker_lib.oracle_choice`).
Without it (the benchmark) only prompts are written -- the benchmark is never a training source
(CC-BY-NC, and it would be the test set). Utterances with no disagreement produce no example.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from picker_lib import build, oracle_choice, render  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turbo", required=True)
    ap.add_argument("--rt", required=True)
    ap.add_argument("--nectec", default=None, help="third source (N); turns on canonical numbers")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    T = {json.loads(l)["utt_id"]: json.loads(l)["cands"] for l in open(args.turbo, encoding="utf-8")}
    R = {json.loads(l)["utt_id"]: json.loads(l)["cands"] for l in open(args.rt, encoding="utf-8")}
    N = {json.loads(l)["utt_id"]: json.loads(l)["cands"] for l in open(args.nectec, encoding="utf-8")} if args.nectec else None
    refs = {json.loads(l)["utt_id"]: json.loads(l)["text"] for l in open(args.manifest, encoding="utf-8")} if args.manifest else {}
    ids = sorted(set(T) & set(R) & (set(N) if N else set(T)))
    n_ex = n_reg = n_nonvote = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for i, u in enumerate(ids):
            parts, regs = build(T[u], R[u], N[u], canonical=True) if N else build(T[u], R[u])
            row = {"utt_id": u, "n_regions": len(regs), "prompt": render(parts, regs) if regs else None,
                   "parts": [[k, x] for k, x in parts], "regs": regs}
            if refs and regs:
                ch = oracle_choice(parts, regs, refs[u])
                row["target"] = ",".join(map(str, ch))
                n_nonvote += sum(c != 0 for c in ch)
            n_reg += len(regs)
            n_ex += bool(regs)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 2000 == 0:
                print(f"  {i}/{len(ids)}", flush=True)
    print(json.dumps({"utts": len(ids), "with_regions": n_ex, "regions": n_reg,
                      "labels_not_vote": n_nonvote if refs else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
