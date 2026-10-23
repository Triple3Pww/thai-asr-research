#!/usr/bin/env python3
"""Loop iteration 5: one model's EAR, another's PEN -- merge two transcripts by whether they SOUND the same.

    E:/thainlp-spike/.venv/Scripts/python.exe adapter/ear_pen_merge.py --pen results/nectec_fc_greedy_hyps.jsonl \
        --ear results/lot2_a75_bench_hyps.jsonl --tag loop5_merge_nec_a2

WHY (`F39` E0). A2 (NECTEC + 1 epoch convention-matched LOTUSDIS, alpha .75) reads the SOUND 11% better than
base NECTEC (SylER-sb .1149 vs .1293) but ties it on CER, because what it writes drifts toward LOTUSDIS's
spelling convention. Base NECTEC writes in the convention the benchmark uses. So:

  word-align the two transcripts (pythainlp newmm); for every span where they differ --
    * same toned sound (tltk G2P)  -> keep the PEN's spelling   (a convention difference, not hearing)
    * different sound              -> take the EAR's words      (the better listener)

Label-free: the rule reads only the two hypotheses. Both models are 115M FastConformer-RNNT, so the pair
costs two encoder passes (~25 ms each, batchable) plus a lookup -- real-time.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_soundblind import BENCH, norm  # noqa: E402
from syllable_labels import g2p_syls  # noqa: E402

_THAI = re.compile(r"^[\u0e00-\u0e7f]+$")


def sound(words: list[str]) -> str:
    t = "".join(words)
    if not t:
        return ""
    return "~".join(g2p_syls(t)) if _THAI.match(t) else t


def merge(pen: str, ear: str, stats: dict) -> str:
    from pythainlp.tokenize import word_tokenize
    a = word_tokenize(norm(pen), engine="newmm", keep_whitespace=False)
    b = word_tokenize(norm(ear), engine="newmm", keep_whitespace=False)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            out += a[i1:i2]
        elif sound(a[i1:i2]) == sound(b[j1:j2]):
            out += a[i1:i2]; stats["same_sound_keep_pen"] += 1
        else:
            out += b[j1:j2]; stats["diff_sound_take_ear"] += 1
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pen", required=True)
    ap.add_argument("--ear", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    real = {json.loads(l)["utt_id"] for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")
            if json.loads(l)["split"] == "real"}
    L = lambda p: {json.loads(l)["utt_id"]: json.loads(l)["text"] for l in open(p, encoding="utf-8")
                   if json.loads(l).get("condition", "none") == "none" and json.loads(l)["utt_id"] in real}
    P, E = L(a.pen), L(a.ear)
    stats = {"same_sound_keep_pen": 0, "diff_sound_take_ear": 0}
    out = {u: merge(P[u], E[u], stats) for u in sorted(real)}
    p = HERE.parent / "results" / f"{a.tag}_hyps.jsonl"
    with p.open("w", encoding="utf-8", newline="\n") as g:
        for u, t in out.items():
            g.write(json.dumps({"utt_id": u, "condition": "none", "text": t}, ensure_ascii=False) + "\n")
    (HERE.parent / "results" / f"{a.tag}_run.json").write_text(json.dumps(
        {"tag": a.tag, "pen": a.pen, "ear": a.ear, **stats}, indent=1), encoding="utf-8")
    print(stats, "->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
