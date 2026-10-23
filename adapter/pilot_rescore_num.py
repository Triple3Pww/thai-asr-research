#!/usr/bin/env python3
"""Re-score any hypothesis files at the F17 `none` and `num` rungs, next to the single systems.

    python adapter/pilot_rescore_num.py results/rover_uniform_tw+pat+q17+isan+bio_hyps.jsonl ...

WHY. A combination that votes in digits beats Typhoon on RAW CER partly because the corpus
typist wrote digits and Typhoon writes number words (`F2`, `F17`). The `num` rung re-renders the
REFERENCE into whichever convention each hypothesis used, identically for every system, so only
recognition is compared. A combination claim must survive this rung or it is a convention artefact.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _ortho import build_form_index  # noqa: E402
from ortho_rescore import BENCH, score_one, systems  # noqa: E402


def main() -> int:
    ann = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8") if l.strip()]
    keep = {r["utt_id"] for r in ann if r["split"] == "real"}
    fi = build_form_index(ann)
    sysmap = systems(HERE.parent / "results/arm1_rareword_hyps.jsonl")
    sysmap["qwenctx-0.6b"] = HERE.parent / "results/arm1_rareword_qwenctx_hyps.jsonl"
    q17 = HERE.parent / "results/bar_qwen17ctx_hyps.jsonl"
    if q17.exists():
        sysmap["qwenctx-1.7b"] = q17
    for p in sys.argv[1:]:
        sysmap[Path(p).stem.replace("_hyps", "")] = Path(p)
    tmp = Path(tempfile.mkdtemp())
    res = {}
    for name, p in sysmap.items():
        res[name] = {r: score_one(name, p, ann, keep, r, fi, tmp, "dp") for r in ("none", "num")}
    print(f"{'system':52s} {'CER none':>9s} {'CER num':>8s} {'U num':>7s} {'B num':>7s} {'recall':>7s}")
    for name, m in sorted(res.items(), key=lambda kv: kv[1]["num"]["cer"]):
        n, u = m["none"], m["num"]
        print(f"{name[:52]:52s} {n['cer']:9.4f} {u['cer']:8.4f} {u['u_cer']:7.4f} {u['b_cer']:7.4f} "
              f"{u['entity_recall']:7.4f}  ci{u['ci95']['cer']}")
    out = HERE.parent / "results/pilot_rescore_num.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
