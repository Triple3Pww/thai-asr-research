#!/usr/bin/env python3
"""Paired comparison of two hypothesis files on the real split: official CER (canon) + paired bootstrap.

    python adapter/paired_vs.py results/nectec_fc_greedy_hyps.jsonl results/<arm>_hyps.jsonl

The loop's success test (2026-10-08): arm beats the base with paired-bootstrap P(arm not better) < 0.05.
Official numbers come from `eval_canon.py`'s scorer (benchmark `_score.py`, canon normaliser, condition
`none`); the bootstrap runs on per-utterance Levenshtein under the same canon + SARA AM fold, because the
official scorer reports only totals. Resampling unit = the utterance, 10,000 resamples, seed 0.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _thai import SCORING_PUNCT, fold_sara_am  # noqa: E402
from eval_canon import BENCH, canon, score  # noqa: E402


def lev(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def n(s):
    return SCORING_PUNCT.sub("", fold_sara_am(canon(s))).lower()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("arm")
    ap.add_argument("--half", choices=("all", "dev", "test"), default="all",
                    help="2026-10-08: results/split_dev_test.json -- tune on dev; test is scored ONCE")
    ap.add_argument("--final", action="store_true", help="required for --half test; every use is logged")
    args = ap.parse_args()
    base, arm = Path(args.base), Path(args.arm)
    ann = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")]
    ann = [a_ for a_ in ann if a_["split"] == "real"]
    if args.half != "all":
        if args.half == "test" and not args.final:
            sys.exit("refused: the test half is scored once, with --final (see results/split_dev_test.json)")
        ids = set(json.loads((HERE.parent / "results/split_dev_test.json").read_text(encoding="utf-8"))[args.half])
        ann = [a_ for a_ in ann if a_["utt_id"] in ids]
        if args.half == "test":
            import datetime
            with (HERE.parent / "results/test_evaluations.log").open("a", encoding="utf-8") as lg:
                lg.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}  {arm}  vs  {base}\n")
    keep = {a["utt_id"] for a in ann}
    tmp = Path(tempfile.mkdtemp())
    ap = tmp / "ann.jsonl"
    with ap.open("w", encoding="utf-8", newline="\n") as f:
        for a in ann:
            a2 = dict(a, text=canon(a["text"]),
                      entities=[dict(e, span_text=canon(e["span_text"])) for e in a.get("entities") or []])
            f.write(json.dumps(a2, ensure_ascii=False) + "\n")
    off = {k: score(p, ap, keep, tmp, k) for k, p in (("base", base), ("arm", arm))}
    refs = {a["utt_id"]: n(a["text"]) for a in ann}
    H = {}
    for k, p in (("base", base), ("arm", arm)):
        H[k] = {json.loads(l)["utt_id"]: json.loads(l)["text"] for l in p.open(encoding="utf-8")
                if json.loads(l).get("condition", "none") == "none"}
    us = sorted(refs)
    from official_utt import Official             # 2026-10-08: bootstrap on OFFICIAL per-utt errors
    O = Official()
    e = {k: {u: O.err(u, H[k].get(u, ""))[0] for u in us} for k in H}
    rng = random.Random(0)
    worse = 0
    for _ in range(10000):
        s = [rng.choice(us) for _ in us]
        worse += sum(e["arm"][u] for u in s) >= sum(e["base"][u] for u in s)
    tot = sum(len(refs[u]) for u in us)
    res = {"base": str(base), "arm": str(arm),
           "official_cer": {k: round(off[k]["cer"], 5) for k in off},
           "official_entity_recall": {k: round(off[k]["entity_recall"], 4) for k in off},
           "own_cer": {k: round(sum(e[k].values()) / tot, 5) for k in e},
           "rel_change": round(off["arm"]["cer"] / off["base"]["cer"] - 1, 4),
           "p_arm_not_better": worse / 10000,
           "utts_changed": sum(H["arm"].get(u) != H["base"].get(u) for u in us)}
    for k, v in res.items():
        print(f"{k:24s} {v}")
    res["half"] = args.half
    out = HERE.parent / "results" / f"paired_{arm.stem.replace('_hyps', '')}_vs_{base.stem.replace('_hyps', '')}_{args.half}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
