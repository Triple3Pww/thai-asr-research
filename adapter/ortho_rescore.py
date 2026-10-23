#!/usr/bin/env python3
"""Re-score all seven systems under permissible orthographic variation. Does the ordering hold?

    python adapter/ortho_rescore.py                       # all four rungs, real split
    python adapter/ortho_rescore.py --rungs none num      # just the two

THE QUESTION THIS ANSWERS, AND WHY IT IS WORTH ASKING.

`RESEARCH-FINDINGS.md` F3 -- "the rare-word gap is field-wide" -- rests on one table: seven Thai
ASR systems scored on the 682 real `thai-contextasr-bench` utterances, every one of them 2-3x
worse inside entity spans than outside (B-CER/U-CER 2.01 to 3.04). That table is computed on RAW
Thai CER.

A40 (Bhogale et al. 2026, 2603.00941) is the reason to distrust a raw-CER ranking: accounting for
permissible orthographic variation narrowed a Gemini-vs-Canary gap from 18.1 points to 11.5. If a
comparable share of this table is convention rather than recognition, the ordering could be an
artefact -- and F3 would be resting on it. Thai has exactly the variation those papers normalise,
and this repo has hit it three separate times already (F1 SARA AM, F2 digits, the script trap).

The test costs nothing: all seven hypothesis files are already on disk. No GPU, no new decoding,
no model. Only the scoring convention changes, and it changes identically for every system.

WHAT MOVES AND WHAT CANNOT. Entity recall is computed from the ORIGINAL reference and the
untouched hypothesis, so it is invariant across rungs by construction -- if it moves, this script
is broken, and it says so. CER, U-CER and B-CER are the metrics under test.

🔴 Read `_ortho.py` before believing any number here. The rewrite only ever re-renders the
REFERENCE into a form the hypothesis contains verbatim; it cannot make a wrong transcription
right, and it under-corrects rather than over-corrects.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _ortho import RUNGS, build_form_index, rewrite_reference, strip_fillers  # noqa: E402

BENCH = Path("E:/asr-bench/thai-contextasr-bench")
OUT = Path(os.environ.get("ADAPTER_OUT") or (HERE.parent / "results"))
CONDITION = "none"
OURS = "typhoon-asr-realtime (OURS, ~115M RNNT, GPU)"


def systems(ours_path: Path) -> dict[str, Path]:
    out = {}
    if ours_path.exists():
        out[OURS] = ours_path
    else:
        print(f"  (our hyps not found at {ours_path} -- run arm1_rareword.py first)")
    for f in sorted(glob.glob(str(BENCH / "baselines/*/hyps.jsonl"))):
        out[os.path.basename(os.path.dirname(f))] = Path(f)
    return out


def score_one(name: str, hyps_path: Path, ann: list[dict], keep: set[str], rung: str,
              form_index: dict[str, list[str]], tmp: Path, align: str) -> dict | None:
    """Rewrite this system's references under `rung`, then score with the benchmark's scorer."""
    rows = []
    with hyps_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["condition"] == CONDITION and r["utt_id"] in keep:
                rows.append(r)
    if not rows:
        print(f"  {name}: no rows for condition {CONDITION}, skipped")
        return None

    by_id = {r["utt_id"]: r for r in rows}
    n_moved = 0
    new_ann = []
    for a in ann:
        h = by_id.get(a["utt_id"])
        if h is None:
            new_ann.append(a)
            continue
        a2 = dict(a)
        a2["text"] = rewrite_reference(a["text"], h["text"], a.get("entities") or [],
                                       rung, form_index)
        n_moved += a2["text"] != a["text"]
        new_ann.append(a2)

    slug = "".join(c if c.isalnum() else "_" for c in name)[:40]
    ap = tmp / f"{slug}.{rung}.ann.jsonl"
    with ap.open("w", encoding="utf-8", newline="\n") as f:
        for a in new_ann:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    hp = tmp / f"{slug}.{rung}.hyps.jsonl"
    with hp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            if rung == "filler":                 # the one rung that touches the hypothesis, and
                r = dict(r)                      # it must, or a correct เอ่อ would be charged
                r["text"] = strip_fillers(r["text"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mp = tmp / f"{slug}.{rung}.metrics.json"
    p = subprocess.run([sys.executable, str(HERE / "_score.py"), "--align", align,
                        "--annotations", str(ap), "--lists", str(BENCH / "data/bias_lists.jsonl.gz"),
                        "--hyps", str(hp), "--out", str(mp)],
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        print(f"  {name}: scorer failed -- {(p.stderr or '')[:300]}")
        return None
    m = json.loads(mp.read_text(encoding="utf-8"))[CONDITION]
    m["_n_rows"] = len(rows)
    m["_n_refs_rewritten"] = n_moved
    return m


def kendall_tau(a: list[str], b: list[str]) -> tuple[float, int]:
    """Rank correlation between two orderings of the same items, plus the raw swap count."""
    pos_a = {k: i for i, k in enumerate(a)}
    pos_b = {k: i for i, k in enumerate(b)}
    items = list(pos_a)
    con = dis = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            x, y = items[i], items[j]
            s = (pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y])
            if s > 0:
                con += 1
            elif s < 0:
                dis += 1
    n = con + dis
    return ((con - dis) / n if n else 1.0), dis


def table(rung: str, res: dict[str, dict]) -> None:
    hdr = (f"{'model':46s} {'n':>5s} {'refs+':>5s} {'CER':>8s} {'U-CER':>8s} {'B-CER':>8s} "
           f"{'recall':>7s} {'B/U':>5s}")
    print(f"\n--- rung `{rung}` " + "-" * (len(hdr) - 12 - len(rung)))
    print(hdr)
    print("-" * len(hdr))
    for k, m in sorted(res.items(), key=lambda kv: kv[1]["cer"]):
        ratio = m["b_cer"] / m["u_cer"] if m["u_cer"] else float("nan")
        print(f"{k:46s} {m['_n_rows']:5d} {m['_n_refs_rewritten']:5d} {m['cer']:8.5f} "
              f"{m['u_cer']:8.5f} {m['b_cer']:8.5f} {m['entity_recall']:7.3f} {ratio:5.2f}")
    ratios = [m["b_cer"] / m["u_cer"] for m in res.values() if m["u_cer"]]
    print(f"{'':46s} {'':5s} {'':5s} {'':8s} {'':8s} B/U range {min(ratios):.2f}-{max(ratios):.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="real", choices=("real", "synth", "all"))
    ap.add_argument("--align", default="dp", choices=("dp", "kaldialign"))
    ap.add_argument("--rungs", nargs="+", default=[r for r in RUNGS if r != "filler"],
                    choices=list(RUNGS),
                    help="`filler` is excluded by default -- it is 1%% of the error mass and it "
                         "deletes sounds the speaker made. See _ortho.FILLERS.")
    ap.add_argument("--ours", default=str(OUT / "arm1_rareword_hyps.jsonl"))
    args = ap.parse_args()
    if "none" not in args.rungs:
        args.rungs = ["none"] + list(args.rungs)          # the baseline is the comparison

    ann = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")
           if l.strip()]
    keep = {r["utt_id"] for r in ann if args.split == "all" or r["split"] == args.split}
    form_index = build_form_index(ann)
    print(f"{len(keep)} utts (split={args.split}, condition={CONDITION}), "
          f"{len(form_index)} indexed entity forms, align={args.align}")

    sysmap = systems(Path(args.ours))
    tmp = Path(tempfile.mkdtemp())
    all_res: dict[str, dict[str, dict]] = {}
    for rung in args.rungs:
        res = {}
        for name, p in sysmap.items():
            m = score_one(name, p, ann, keep, rung, form_index, tmp, args.align)
            if m:
                res[name] = m
        all_res[rung] = res
        table(rung, res)

    base = all_res["none"]
    by_cer_base = [k for k, _ in sorted(base.items(), key=lambda kv: kv[1]["cer"])]
    print("\n\nDOES THE ORDERING SURVIVE?  (ranking by CER, best first)")
    print(f"{'rung':8s} {'tau':>6s} {'swaps':>6s}  {'dCER(mean)':>11s} {'dCER(spread)':>13s}  ranking")
    for rung in args.rungs:
        res = all_res[rung]
        order = [k for k, _ in sorted(res.items(), key=lambda kv: kv[1]["cer"])]
        tau, swaps = kendall_tau(by_cer_base, order)
        d = [base[k]["cer"] - res[k]["cer"] for k in res]
        spread_b = max(m["cer"] for m in base.values()) - min(m["cer"] for m in base.values())
        spread_r = max(m["cer"] for m in res.values()) - min(m["cer"] for m in res.values())
        short = " > ".join(k.split("_")[-1][:12] if k != OURS else "OURS" for k in order)
        print(f"{rung:8s} {tau:6.3f} {swaps:6d}  {sum(d)/len(d):11.5f} "
              f"{spread_b - spread_r:+13.5f}  {short}")
        # entity recall is computed off the untouched hyp: it must not move at all.
        moved = [k for k in res if abs(res[k]["entity_recall"] - base[k]["entity_recall"]) > 1e-9]
        if moved:
            print(f"         🔴 entity recall MOVED for {len(moved)} systems -- the rewrite leaked "
                  f"into a metric it cannot touch, do not trust this rung")

    path = OUT / f"ortho_rescore_{args.split}.json"
    path.write_text(json.dumps({"split": args.split, "condition": CONDITION,
                                "align": args.align, "n_utts": len(keep),
                                "rungs": all_res}, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
