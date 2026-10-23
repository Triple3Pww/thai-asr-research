#!/usr/bin/env python3
"""Teacher pass 2: vote the 8 systems' YODAS hypotheses into one training label per segment.

    python adapter/teacher_combine.py                       # all-8 ROVER labels -> train manifest
    python adapter/teacher_combine.py --max-disagree 0.25   # stricter confidence filter

Same ROVER as the benchmark pilot (`pilot_rover.rover`, uniform votes, MBR-medoid pivot, scorer
tokens), over ALL eight systems -- the no-selection configuration, the only one that beat the bar
honestly (`idea-stage/IDEA_REPORT.md`).

Three filters, each with its reason:
  1. LEAK GUARD. Benchmark `a-` clips are GigaSpeech2 TEST, cut from YouTube; YODAS is YouTube too.
     Any segment whose label shares a >= `--leak-chars` character run with ANY benchmark reference
     (all 2,062 utterances, canonical form) is dropped and counted. Without this a student could be
     trained on the test sentences.
  2. CONFIDENCE. Pool disagreement = mean token edit distance of the 7 non-pivot systems to the
     pivot, over pivot length. High disagreement = nobody knows; such labels teach noise
     (`F12`/`H36`: filter, don't use everything). Dropped above `--max-disagree`.
  3. SHAPE. Empty labels, and labels whose characters-per-second is implausible for Thai speech.

Label convention matches what typhoon-asr-realtime already emits (`F2`, CLAUDE.md): digits ->
unspaced Thai number WORDS, SARA AM decomposed, no spaces between Thai characters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _thai import as_typhoon_writes  # noqa: E402
from eval_canon import canon  # noqa: E402
from pilot_rover import BENCH, edit, join, rover, toks  # noqa: E402

DATA = Path("E:/Claude/datasets/yodas2_th")
SYSTEMS = ["tw", "pat", "q17", "isan", "bio", "q06", "qctx", "rt"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-disagree", type=float, default=0.30)
    ap.add_argument("--leak-chars", type=int, default=20)
    ap.add_argument("--out", default=str(DATA / "train_rover8.jsonl"))
    ap.add_argument("--student", default=None,
                    help="TEACH MODE (round 2). Confidence is measured on the OTHER systems only, "
                         "and the training set is every segment the student gets wrong (student "
                         "CER vs label > --teach-cer) plus an equal number it already gets right. "
                         "Round 1 measured confidence on all 8 INCLUDING the student, so 'confident' "
                         "meant 'the student already agrees': base rt was 3.4% from its own labels "
                         "and fine-tuning moved CER 0.0945 -> 0.0943.")
    ap.add_argument("--teach-cer", type=float, default=0.05)
    args = ap.parse_args()

    man = {json.loads(l)["id"]: json.loads(l) for l in (DATA / "manifest.jsonl").open(encoding="utf-8")}
    hyps = {}
    for s in SYSTEMS:
        p = DATA / "teacher" / f"{s}.jsonl"
        hyps[s] = {json.loads(l)["id"]: json.loads(l)["text"] for l in p.open(encoding="utf-8") if l.strip()}
    ids = sorted(set(man).intersection(*(set(h) for h in hyps.values())))
    print(f"{len(ids)} segments have all {len(SYSTEMS)} systems (manifest {len(man)})")

    # leak guard: every benchmark reference, canonical, unspaced
    refs = ["".join(toks(canon(json.loads(l)["text"])))
            for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8") if l.strip()]
    K = args.leak_chars
    grams = {r[i: i + K] for r in refs for i in range(len(r) - K + 1)}

    w = {s: 1.0 for s in SYSTEMS}
    judges = [s for s in SYSTEMS if s != args.student]
    stats = {"leak": 0, "disagree": 0, "shape": 0, "kept": 0, "teach": 0}
    kept_s = 0.0
    rows_out = []
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for u in ids:
            seqs = {s: toks(hyps[s][u]) for s in SYSTEMS}
            # teach mode: the label is voted by the judges only, so the student's own vote cannot
            # pull the label toward the student's mistake
            out, pivot = rover({s: seqs[s] for s in judges}, w, judges)
            jp = pivot
            P = seqs[jp]
            dis = sum(edit(P, seqs[s]) for s in judges if s != jp) / (len(judges) - 1) / max(len(P), 1)
            flat = "".join(out)
            dur = man[u]["dur"]
            if any(flat[i: i + K] in grams for i in range(len(flat) - K + 1)):
                stats["leak"] += 1
                continue
            if dis > args.max_disagree:
                stats["disagree"] += 1
                continue
            label = as_typhoon_writes(canon(join(out)))
            cps = len(flat) / dur
            if not label.strip() or not (2.0 <= cps <= 30.0):
                stats["shape"] += 1
                continue
            row = {"audio_filepath": man[u]["wav"], "duration": dur, "text": label,
                   "id": u, "disagree": round(dis, 3), "pivot": pivot}
            if args.student:
                st = toks(canon(hyps[args.student][u]))
                row["student_cer"] = round(edit(out, st) / max(len(out), 1), 3)
                row["teach"] = row["student_cer"] > args.teach_cer
            rows_out.append(row)
        if args.student:                          # every teach row + as many easy rows, by id order
            teach = [r for r in rows_out if r["teach"]]
            easy = [r for r in rows_out if not r["teach"]][: len(teach)]
            rows_out = sorted(teach + easy, key=lambda r: r["id"])
            stats["teach"] = len(teach)
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            kept_s += row["duration"]
    print(stats, f"kept {kept_s/3600:.2f} h -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
