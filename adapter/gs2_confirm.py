#!/usr/bin/env python3
"""GigaSpeech2-th TRAIN shard -> the clips NECTEC gets wrong AND an independent ear confirms the label.

    PYTHONUTF8=1 python adapter/gs2_confirm.py --shard 1 --stage hard      # after gs2train_prep.py --shard 1
    PYTHONUTF8=1 python adapter/gs2_confirm.py --shard 1 --stage confirm   # after fw_batch.py on the hard clips

WHY (2026-10-11, loop iterations 1b/2). On machine labels "NECTEC is wrong" mostly means "the label is wrong"
(`F48`): training on all of shard 0's hard clips was +25% at full strength. Pathumma-whisper-large confirms only
18% of them; those (4.2 h) tied the best model, and moved the conversational "a" half of the benchmark the most.
This scales the same filter to more shards:

  hard    : rows of gs2train_prep's NECTEC pass with any NECTEC error (labels already in NECTEC's convention,
            leak-guarded, Latin-NECTEC clips dropped) -> manifest_shard{N}_hardonly.jsonl
  confirm : keep CER(label, Pathumma-L) <= .02 AND Pathumma-L closer to the label than NECTEC, no Latin in
            Pathumma-L -> manifest_shard{N}_conf.jsonl (the shard-0 rule, unchanged)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gs2_convention import LATIN, convert  # noqa: E402
from gs2train_prep import lev  # noqa: E402

D = Path("E:/Claude/datasets/gs2_train")


def n(s: str) -> str:
    return s.replace(" ", "").replace("\u0e4d\u0e32", "\u0e33")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--stage", choices=("hard", "confirm"), required=True)
    args = ap.parse_args()
    if not sys.flags.utf8_mode:
        raise SystemExit("run with PYTHONUTF8=1")
    s = args.shard
    nec = {json.loads(l)["id"]: json.loads(l)["hyp"] for l in open(D / f"nectec_shard{s}.jsonl", encoding="utf-8")}
    if args.stage == "hard":
        rows = [json.loads(l) for l in open(D / f"manifest_shard{s}_hard.jsonl", encoding="utf-8")]
        hard = [r for r in rows if not LATIN.search(nec[r["id"]]) and lev(r["text"], n(nec[r["id"]])) > 0]
        with open(D / f"manifest_shard{s}_hardonly.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in hard:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"shard {s}: hard {len(hard)} clips, {sum(r['duration'] for r in hard) / 3600:.1f} h")
        return 0
    man = [json.loads(l) for l in open(D / f"manifest_shard{s}_hardonly.jsonl", encoding="utf-8")]
    pat = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(D / f"patL_shard{s}_hard.jsonl", encoding="utf-8")}
    keep = []
    for r in man:
        p = pat.get(r["id"])
        if p is None or LATIN.search(p):
            continue
        t = r["text"]
        ep, en = lev(t, n(convert(p))), lev(t, n(nec[r["id"]]))
        if ep / max(len(t), 1) <= 0.02 and ep < en:
            keep.append(r)
    with open(D / f"manifest_shard{s}_conf.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"shard {s}: confirmed {len(keep)} / {len(man)} hard clips, {sum(r['duration'] for r in keep) / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
