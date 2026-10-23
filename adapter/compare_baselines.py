#!/usr/bin/env python3
"""Place our ASR against the benchmark's published baselines on IDENTICAL audio.

    python adapter/compare_baselines.py          # needs arm1_rareword.py to have run first

WHY THIS RECOMPUTES INSTEAD OF QUOTING THEIR `metrics.json`.

`wayu-ai/thai-contextasr-bench` ships, per baseline, both the raw hypotheses (`hyps.jsonl`) and a
scored `metrics.json`. Quoting the published metrics would be wrong, because they are computed over
all 2,062 utterances while we score only the 682 real-split ones (`arm1_rareword.py` explains why:
evaluating a synthetic-data method on the 1,380 TTS utterances is circular).

Measured how much that matters, on `typhoon-whisper-large-v3`, condition `none`:

    metric          theirs (2,062)   ours (682 real)
    CER                     0.0483            0.0924
    U-CER                   0.0391            0.0878
    B-CER                   0.1166            0.1829
    entity recall            0.459             0.408

🔴 **Including the synthetic half makes every model look about twice as good on CER.** So comparing
our 682-real number against their published full-benchmark number would have manufactured a gap of
0.13 in entity recall where the real gap is 0.08 -- and blamed our model for what is mostly corpus
composition. Same family of error as this repo's GPU-drift rule: compare things measured in the same
window, on the same material, with the same instrument.

⭐ It is also quiet support for the repo's own thesis -- synthetic speech is measurably EASIER than
real speech, even as evaluation material.

So: take their hypotheses (which are real model outputs and perfectly good), filter to the same 682
utt_ids and the same condition, and run them through the same scorer we ran ourselves.

⚠️ NOT SIZE-MATCHED, and the table must always say so. `typhoon-asr-realtime` is a ~115M-class RNNT
at RTF 0.007 on CPU; the Whisper baselines are ~1.5B on a GPU. Ranking 6th of 7 while being the
smallest and ~100x cheapest is a different claim from being worst.
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
BENCH = Path("E:/asr-bench/thai-contextasr-bench")
# Results dir -- `results/` at the repo root, beside this `adapter/` package.
# ADAPTER_OUT still overrides it, for a scratch run that must not touch the record.
OUT = Path(os.environ.get("ADAPTER_OUT") or (Path(__file__).resolve().parent.parent / "results"))
CONDITION = "none"
OURS = "typhoon-asr-realtime (OURS, ~115M RNNT, GPU)"   # F8c: it was never on the CPU


def score(hyps_path: Path, name: str, keep_ids: set[str], tmp: Path,
          align: str = "dp") -> dict | None:
    rows = []
    with open(hyps_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["condition"] == CONDITION and r["utt_id"] in keep_ids:
                rows.append(r)
    if not rows:
        print(f"  {name}: no rows for condition {CONDITION}, skipped")
        return None
    hp = tmp / f"{name}.jsonl"
    with hp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mp = tmp / f"{name}.metrics.json"
    # 🔴 THROUGH `_score.py`, NEVER `scorer/score.py` DIRECTLY. Until 2026-09-17 this called the
    # benchmark's scorer straight, which is the one hole `_score.py` exists to close: its span
    # metrics silently depend on whether `kaldialign` is importable in the interpreter running it.
    # CER and recall need no span attribution and do not move; B-CER and U-CER do.
    #
    # That is not hypothetical here. The recorded table in RESEARCH-PROGRESS.md was produced from
    # the VoxCPM venv, which has no kaldialign, so it is on the DP path. Re-run from the documented
    # ASR venv, which HAS kaldialign, this script silently returned B-CER 0.2505 where the record
    # says 0.26178 -- for all seven systems at once, with CER and recall byte-identical. A whole
    # comparison table moved and nothing errored.
    p = subprocess.run([sys.executable, str(HERE / "_score.py"),
                        "--align", align,
                        "--annotations", str(BENCH / "data/utterances.jsonl"),
                        "--lists", str(BENCH / "data/bias_lists.jsonl.gz"),
                        "--hyps", str(hp), "--out", str(mp)],
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        print(f"  {name}: scorer failed -- {(p.stderr or '')[:200]}")
        return None
    m = json.loads(mp.read_text(encoding="utf-8"))[CONDITION]
    m["_n_rows"] = len(rows)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="real", choices=("real", "synth", "all"))
    ap.add_argument("--align", default="dp", choices=("dp", "kaldialign"),
                    help="alignment backend for B-CER/U-CER attribution. `dp` is what every "
                         "number already in this repo used; kaldialign moves B-CER 0.011 on "
                         "byte-identical hypotheses. See adapter/_score.py")
    ap.add_argument("--ours", default=str(OUT / "arm1_rareword_hyps.jsonl"))
    args = ap.parse_args()

    ann = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")
           if l.strip()]
    keep = {r["utt_id"] for r in ann
            if args.split == "all" or r["split"] == args.split}
    print(f"scoring every system on the same {len(keep)} utts "
          f"(split={args.split}, condition={CONDITION})\n")

    tmp = Path(tempfile.mkdtemp())
    results: dict[str, dict] = {}

    ours = Path(args.ours)
    if ours.exists():
        m = score(ours, "ours", keep, tmp, args.align)
        if m:
            results[OURS] = m
    else:
        print(f"  (our hyps not found at {ours} -- run arm1_rareword.py first)")

    for f in sorted(glob.glob(str(BENCH / "baselines/*/hyps.jsonl"))):
        name = os.path.basename(os.path.dirname(f))
        m = score(Path(f), name, keep, tmp, args.align)
        if m:
            results[name] = m

    hdr = (f"{'model':46s} {'n':>5s} {'CER':>8s} {'U-CER':>8s} {'B-CER':>8s} "
           f"{'recall':>7s} {'B/U':>5s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for k, m in sorted(results.items(), key=lambda kv: kv[1]["entity_recall"] or -1,
                       reverse=True):
        ratio = m["b_cer"] / m["u_cer"] if m["u_cer"] else float("nan")
        print(f"{k:46s} {m['_n_rows']:5d} {m['cer']:8.4f} {m['u_cer']:8.4f} "
              f"{m['b_cer']:8.4f} {m['entity_recall']:7.3f} {ratio:5.2f}")

    ratios = [m["b_cer"] / m["u_cer"] for m in results.values() if m["u_cer"]]
    print(f"\nB-CER / U-CER ranges {min(ratios):.2f}-{max(ratios):.2f} across "
          f"{len(ratios)} systems -- entity spans are universally 2-3x harder, across RNNT, CTC")
    print("and encoder-decoder alike. That is the rare-word gap, and it is field-wide.")
    print("\n\U0001f534 THE B/U RATIO IS THE CLAIM. THE CER ORDERING ABOVE IS NOT (F17, 2026-09-17).")
    print("Neutralising ONE convention -- digit runs vs Thai number words -- reorders this table:")
    print("Kendall tau 0.429, 6 of 21 pairs swapped, ours 7th of 7 -> 5th of 7 at -15.6% CER with")
    print("no change to the model. The ratio survives (2.01-3.04 -> 2.07-3.08) and recall cannot")
    print("move at all. Run `adapter/ortho_rescore.py` before reading the CER column as a ranking.")
    print("\nNOT size-matched: ours is ~115M, the Whispers are ~1.5B -- BOTH ON GPU (F8c).")
    print(f"align backend = {args.align}  (every recorded number in this repo is on dp)")

    path = OUT / f"arm1_rareword_comparison{'' if args.split == 'real' else '_' + args.split}.json"
    path.write_text(json.dumps({"split": args.split, "condition": CONDITION,
                                "n_utts": len(keep), "systems": results},
                               ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
