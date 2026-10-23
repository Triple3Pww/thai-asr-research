#!/usr/bin/env python3
"""ROVER over hypothesis files of ANY manifest (TVSpeech etc.) + a paired bootstrap against one base system.

    python adapter/rover_manifest.py --manifest E:/asr-bench/typhoon-eval/manifest_tvspeech.jsonl \
        --base nec --hyps nec=results/deepseek_tvspeech_base_hyps.jsonl patL=... twL=... --tag tvs_rover_nec_patL_twL

The vote is `pilot_rover.rover` unchanged (MBR-medoid pivot, uniform weights, the benchmark scorer's tokens);
CER and the per-row errors use `eval_manifest.py`'s normaliser (`gate_lotusdis.norm` / `lev`), so numbers are
comparable with `F33`'s TVSpeech table. Paired bootstrap: 10,000 resamples over rows, seed 0.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gate_lotusdis import lev, norm  # noqa: E402
from pilot_rover import join, rover, toks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hyps", nargs="+", required=True, help="name=path ({id,text} jsonl)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    H = {}
    for spec in a.hyps:
        k, p = spec.split("=", 1)
        H[k] = {(h := json.loads(l))["id"]: h["text"] for l in open(p, encoding="utf-8") if l.strip()}
    pool = list(H)
    w = {k: 1.0 for k in pool}
    out = {}
    for r in rows:
        merged, _ = rover({k: toks(H[k][r["id"]]) for k in pool}, w, pool)
        out[r["id"]] = join(merged)
    refs = {r["id"]: norm(r["text"]) for r in rows}
    ids = [r["id"] for r in rows]
    e = {k: {i: lev(refs[i], norm(H[k][i].strip())) for i in ids} for k in pool}
    e["rover"] = {i: lev(refs[i], norm(out[i].strip())) for i in ids}
    tot = sum(len(refs[i]) for i in ids)
    cer = {k: round(sum(v.values()) / tot, 5) for k, v in e.items()}
    rng = random.Random(0)
    worse = 0
    for _ in range(10000):
        s = [rng.choice(ids) for _ in ids]
        worse += sum(e["rover"][i] for i in s) >= sum(e[a.base][i] for i in s)
    res = {"manifest": a.manifest, "pool": pool, "cer": cer,
           "rel_vs_base": round(cer["rover"] / cer[a.base] - 1, 4), "p_rover_not_better": worse / 10000}
    print(json.dumps(res, indent=1))
    res_dir = HERE.parent / "results"
    (res_dir / f"{a.tag}_metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    with (res_dir / f"{a.tag}_hyps.jsonl").open("w", encoding="utf-8", newline="\n") as g:
        for i in ids:
            g.write(json.dumps({"id": i, "text": out[i]}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
