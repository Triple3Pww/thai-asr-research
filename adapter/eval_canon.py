#!/usr/bin/env python3
"""Score hypothesis files under ONE fixed normaliser applied to reference and hypothesis alike.

    python adapter/eval_canon.py results/rover_all8_hyps.jsonl ...      # + every baseline

WHY THIS EXISTS (2026-10-03). The F17 `num` rung re-renders the reference toward EACH hypothesis'
number convention. A reviewer called that a moving target, and they are right that it is not a
single frozen reference: two systems are scored against two different references. This scorer
removes the objection. One deterministic function, `canon()`, is applied identically to the
reference and to every hypothesis, and it never looks at the other side:

  * thousands separators inside digit runs are dropped  (1,400 -> 1400)
  * every remaining ASCII/Thai digit run becomes Thai number WORDS (`_thai.thai_cardinal`)

Then the benchmark's frozen scorer runs unchanged on the result (`_score.py --align dp`).
Words, not digits, because that is Typhoon's convention (`F2`) -- choosing digits would hand the
bar a penalty, choosing words hands it nothing. Phone-style digit-wise readings and decimals are
mis-normalised the same way for every system; that is noise, not bias.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _thai import thai_cardinal  # noqa: E402
from pilot_mbr_combine import POOL  # noqa: E402

BENCH = Path("E:/asr-bench/thai-contextasr-bench")
_TH_DIG = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_SEP = re.compile(r"(?<=\d),(?=\d{3})")
_RUN = re.compile(r"\d+")


def canon(s: str) -> str:
    s = _SEP.sub("", (s or "").translate(_TH_DIG))
    return _RUN.sub(lambda m: thai_cardinal(int(m.group())) if len(m.group()) <= 9 else m.group(), s)


def score(hyps: Path, ann_path: Path, keep: set[str], tmp: Path, tag: str) -> dict:
    hp, mp = tmp / f"{tag}.h.jsonl", tmp / f"{tag}.m.json"
    # 2026-10-07: a single-condition file (our bias@N / dis@N arms) is scored on ITS condition -- they
    # were silently scored as zero rows while this hardcoded "none". The benchmark's own baseline
    # files hold every condition; those are still scored on "none", as before.
    rows = [json.loads(l) for l in hyps.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["utt_id"] in keep]
    conds = {r["condition"] for r in rows}
    if len(conds) > 1:
        rows, conds = [r for r in rows if r["condition"] == "none"], {"none"}
    with hp.open("w", encoding="utf-8", newline="\n") as g:
        for r in rows:
            g.write(json.dumps({"utt_id": r["utt_id"], "condition": r["condition"],
                                "text": canon(r["text"])}, ensure_ascii=False) + "\n")
    subprocess.run([sys.executable, str(HERE / "_score.py"), "--align", "dp", "--hyps", str(hp),
                    "--out", str(mp), "--annotations", str(ann_path)],
                   capture_output=True, text=True, encoding="utf-8")
    return json.loads(mp.read_text(encoding="utf-8"))[conds.pop()]


def main() -> int:
    ann = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8") if l.strip()]
    keep = {a["utt_id"] for a in ann if a["split"] == "real"}
    tmp = Path(tempfile.mkdtemp())
    ap = tmp / "ann_canon.jsonl"
    moved_ent = 0
    with ap.open("w", encoding="utf-8", newline="\n") as f:
        for a in ann:
            a2 = dict(a, text=canon(a["text"]))
            ents = []
            for e in a.get("entities") or []:
                e2 = dict(e, span_text=canon(e["span_text"]))
                moved_ent += e2["span_text"] != e["span_text"]
                ents.append(e2)
            a2["entities"] = ents
            f.write(json.dumps(a2, ensure_ascii=False) + "\n")
    print(f"{len(keep)} real utts; entity spans touched by canon(): {moved_ent}")
    systems = {k: p for k, p in POOL.items() if p.exists()}
    for a in sys.argv[1:]:
        systems[Path(a).stem.replace("_hyps", "")] = Path(a)
    res = {k: score(p, ap, keep, tmp, f"s{i}") for i, (k, p) in enumerate(systems.items())}
    print(f"{'system':40s} {'CER':>7s} {'U-CER':>7s} {'B-CER':>7s} {'recall':>7s}  CER ci95")
    for k, m in sorted(res.items(), key=lambda kv: kv[1]["cer"]):
        print(f"{k[:40]:40s} {m['cer']:.4f} {m['u_cer']:.4f} {m['b_cer']:.4f} {m['entity_recall']:.4f}  {m['ci95']['cer']}")
    (HERE.parent / "results/eval_canon.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
