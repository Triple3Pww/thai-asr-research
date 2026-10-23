#!/usr/bin/env python3
"""Official per-utterance CER on the benchmark DEV half, split by source prefix (a = conversation, c = FLEURS
sentences). DEV ONLY -- 2026-10-11 a breakdown script that looped over both halves leaked test numbers
(logged in results/test_evaluations.log); this one cannot.

    python adapter/dev_by_source.py results/A_hyps.jsonl [results/B_hyps.jsonl ...]
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from official_utt import Official  # noqa: E402

O = Official()
dev = json.loads((HERE.parent / "results/split_dev_test.json").read_text(encoding="utf-8"))["dev"]
for p in sys.argv[1:]:
    h = {json.loads(l)["utt_id"]: json.loads(l)["text"] for l in open(p, encoding="utf-8")}
    agg = collections.defaultdict(lambda: [0, 0])
    for u in dev:
        e, r = O.err(u, h.get(u, ""))
        for k in (u.split("-")[0], "all"):
            agg[k][0] += e; agg[k][1] += r
    print(Path(p).name, " ".join(f"{k} {e / r:.4f}" for k, (e, r) in sorted(agg.items())))
