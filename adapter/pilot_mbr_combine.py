#!/usr/bin/env python3
"""Pilot: does a label-free system combination beat the best single Typhoon model?

    python adapter/pilot_mbr_combine.py                     # every subset of the pool, real split
    python adapter/pilot_mbr_combine.py --pool tw pat q17   # one subset

WHAT THIS IS. Minimum-Bayes-risk *selection* (the "medoid" hypothesis): for each utterance, out of
N systems' transcripts, emit the one whose summed character edit distance to the other N-1 is
smallest. The reference is never read -- selection uses only agreement between systems -- so this
is a legitimate decode-time system, not an oracle. All inputs are hypothesis files already on
disk; no GPU, no decoding.

WHY FIRST. The direction is "beat the best Typhoon model". Typhoon trained on ~10,800 h; we hold
10.57 h of synthetic audio. Before spending GPU on training, measure how much the field's
already-public models leave on the table when they disagree. The oracle row (best hypothesis per
utterance BY REFERENCE) is printed only as a ceiling and must never be quoted as a system.

Distances are computed on a canonical form (SARA AM folded, whitespace and punctuation stripped)
so a convention difference does not count as disagreement; the EMITTED text is the system's own,
untouched, and scored by the benchmark's frozen scorer (`_score.py`, --align dp).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _thai import fold_sara_am  # noqa: E402

BENCH = Path("E:/asr-bench/thai-contextasr-bench")
OUT = Path(os.environ.get("ADAPTER_OUT") or (HERE.parent / "results"))
RES = HERE.parent / "results"

POOL = {
    "tw":   BENCH / "baselines/whisper_typhoon-ai_typhoon-whisper-large-v3/hyps.jsonl",
    "pat":  BENCH / "baselines/whisper_nectec_Pathumma-whisper-th-large-v3/hyps.jsonl",
    "q17":  BENCH / "baselines/qwen3_asr_1p7b/hyps.jsonl",
    "isan": BENCH / "baselines/whisper_typhoon-ai_typhoon-isan-asr-whisper/hyps.jsonl",
    "bio":  BENCH / "baselines/whisper_biodatlab_whisper-th-large-v3-combined/hyps.jsonl",
    "q06":  BENCH / "baselines/qwen3_asr_0p6b/hyps.jsonl",
    "qctx": RES / "arm1_rareword_qwenctx_hyps.jsonl",
    "rt":   RES / "arm1_rareword_hyps.jsonl",
}
_STRIP = re.compile(r"[\s\.,!?\"'()\-:;]+")


def canon(s: str) -> str:
    return _STRIP.sub("", fold_sara_am(s or "")).lower()


def lev(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load(path: Path, keep: set[str]) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["condition"] == "none" and r["utt_id"] in keep:
                    out[r["utt_id"]] = r["text"]
    return out


def score(rows: list[dict], tmp: Path, tag: str) -> dict:
    hp, mp = tmp / f"{tag}.hyps.jsonl", tmp / f"{tag}.metrics.json"
    with hp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    p = subprocess.run([sys.executable, str(HERE / "_score.py"), "--align", "dp",
                        "--hyps", str(hp), "--out", str(mp)],
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0 and not mp.exists():
        raise SystemExit(f"scorer failed: {p.stderr[:400]}")
    return json.loads(mp.read_text(encoding="utf-8"))["none"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="*", default=None)
    ap.add_argument("--min", type=int, default=3)
    args = ap.parse_args()

    ann = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")]
    ann = {a["utt_id"]: a for a in ann if a["split"] == "real"}
    keep = set(ann)
    hyps = {k: load(p, keep) for k, p in POOL.items() if p.exists()}
    common = set.intersection(*(set(h) for h in hyps.values()))
    print(f"systems {list(hyps)}; utterances with all systems: {len(common)} / {len(keep)}")
    ids = sorted(common)
    canons = {k: {u: canon(h[u]) for u in ids} for k, h in hyps.items()}

    names = list(hyps)
    pair = {}                                     # (a, b, utt) -> distance, computed once
    for a, b in itertools.combinations(names, 2):
        for u in ids:
            d = lev(canons[a][u], canons[b][u])
            pair[(a, b, u)] = pair[(b, a, u)] = d

    subsets = [tuple(args.pool)] if args.pool else [
        s for n in range(args.min, len(names) + 1) for s in itertools.combinations(names, n)]

    results = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for k in names:                            # singles on the SAME utterance set
            results[k] = score([{"utt_id": u, "condition": "none", "text": hyps[k][u]} for u in ids],
                               tmp, f"single_{k}")
        for s in subsets:
            rows, picks = [], {}
            for u in ids:
                best = min(s, key=lambda k: (sum(pair[(k, o, u)] for o in s if o != k),
                                             names.index(k)))
                picks[best] = picks.get(best, 0) + 1
                rows.append({"utt_id": u, "condition": "none", "text": hyps[best][u]})
            tag = "mbr_" + "+".join(s)
            results[tag] = score(rows, tmp, tag)
            results[tag]["_picks"] = picks
        # oracle ceiling (reads the reference -- NOT a system)
        orows = []
        for u in ids:
            ref = canon(ann[u]["text"])
            best = min(names, key=lambda k: lev(canons[k][u], ref))
            orows.append({"utt_id": u, "condition": "none", "text": hyps[best][u]})
        results["ORACLE_all (ceiling, not a system)"] = score(orows, tmp, "oracle")

    order = sorted(results.items(), key=lambda kv: kv[1]["cer"])
    print(f"\n{'system':48s} {'CER':>7s} {'U-CER':>7s} {'B-CER':>7s} {'recall':>7s}")
    for k, m in order[:25]:
        print(f"{k:48s} {m['cer']:.4f} {m['u_cer']:.4f} {m['b_cer']:.4f} {m['entity_recall']:.4f}")
    print("...\nsingles:")
    for k in names:
        m = results[k]
        print(f"{k:48s} {m['cer']:.4f} {m['u_cer']:.4f} {m['b_cer']:.4f} {m['entity_recall']:.4f}")

    path = OUT / "pilot_mbr_combine_real.json"
    path.write_text(json.dumps({"n_utts": len(ids), "systems": {k: str(p) for k, p in POOL.items()},
                                "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
