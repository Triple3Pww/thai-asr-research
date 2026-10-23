#!/usr/bin/env python3
"""Toward CER 0.060: search ROVER pools on the DEV half only (results/split_dev_test.json).

    python adapter/pool_search_dev.py

Every pool contains NECTEC; size 3-6 from the candidate ears below. Score = per-utterance Levenshtein under
`paired_vs.n` (canon + SARA AM fold) on the dev utterances, relative to NECTEC alone. Also prints the
per-utterance ORACLE of each top pool (the ceiling a perfect selector would reach). The test half is never
read here. Candidate files are whatever transcripts exist; the final system must re-run its ears itself.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paired_vs import lev, n  # noqa: E402
from pilot_rover import join, rover, toks  # noqa: E402
from eval_canon import BENCH  # noqa: E402

R = HERE.parent / "results"
B = BENCH / "baselines"
EARS = {
    "nec": R / "nectec_fc_greedy_hyps.jsonl", "a2": R / "lot2_a75_bench_hyps.jsonl", "l3": R / "lot3_a75_bench_hyps.jsonl",
    "patL": R / "spec_patL_hyps.jsonl", "twL": R / "spec_twL_from_patL_hyps.jsonl", "turbo": R / "fw_turbo_fp16_hyps.jsonl",
    "bdm": R / "fw_biodistmed_fp16_hyps.jsonl", "patM": R / "fw_patmed_fp16_hyps.jsonl",
    "bio": B / "whisper_biodatlab_whisper-th-large-v3-combined/hyps.jsonl", "q17": B / "qwen3_asr_1p7b/hyps.jsonl",
    "q06": B / "qwen3_asr_0p6b/hyps.jsonl", "isan": B / "whisper_typhoon-ai_typhoon-isan-asr-whisper/hyps.jsonl",
}


def main() -> int:
    dev = json.loads((R / "split_dev_test.json").read_text(encoding="utf-8"))["dev"]
    refs = {}
    for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        if r["utt_id"] in set(dev):
            refs[r["utt_id"]] = n(r["text"])
    H = {}
    for k, p in EARS.items():
        H[k] = {json.loads(l)["utt_id"]: json.loads(l)["text"] for l in open(p, encoding="utf-8")
                if json.loads(l).get("condition", "none") == "none"}
    T = {k: {u: toks(H[k].get(u, "")) for u in dev} for k in H}
    E = {k: {u: lev(refs[u], n(H[k].get(u, ""))) for u in dev} for k in H}
    tot = sum(len(refs[u]) for u in dev)
    base = sum(E["nec"].values())
    print("single ears, dev rel CER vs NECTEC:", {k: round(sum(E[k].values()) / base - 1, 3) for k in H})
    others = [k for k in EARS if k != "nec"]
    res = []
    for size in range(2, 6):
        for combo in itertools.combinations(others, size):
            pool = ["nec", *combo]
            err = 0
            for u in dev:
                merged, _ = rover({k: T[k][u] for k in pool}, {k: 1.0 for k in pool}, pool)
                err += lev(refs[u], n(join(merged)))
            orc = sum(min(E[k][u] for k in pool) for u in dev)
            res.append((err / base - 1, orc / base - 1, pool))
        print("size", size + 1, "done", flush=True)
    res.sort()
    for rel, orc, pool in res[:25]:
        print(f"{rel:+.3f}  oracle {orc:+.3f}  {pool}")
    (R / "pool_search_dev.json").write_text(json.dumps([{"rel": a, "oracle": b, "pool": c} for a, b, c in res],
                                                       indent=0), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
