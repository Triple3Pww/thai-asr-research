#!/usr/bin/env python3
"""Loop iteration 1: a name list the user never gave -- retrieved from a public catalog by the DRAFT.

    E:/thainlp-spike/.venv/Scripts/python.exe adapter/catalog_retrieve.py results/nectec_fc_greedy_hyps.jsonl --k 50

WHY (`F39` E4). NECTEC misses 284 of 439 entity mentions; Thai Wikipedia's titles hold 30% of them, and the
one-pass CTC-head splice (`F36`/`F37`) already turns a 50-name list into recall at NECTEC's own speed.
The missing piece is the list. This builds one per utterance from NECTEC's own transcript:

  catalog  = Thai Wikipedia article titles (thwiki-latest-all-titles-in-ns0, downloaded 2026-10-08),
             disambiguators "(...)" dropped, normalised like every scorer here, 3-30 chars, and NOT an
             ordinary dictionary word (pythainlp `thai_words`) -- names, not vocabulary
  retrieve = character-trigram index; candidates sharing >= half their trigrams with the draft; rerank by
             semi-global edit similarity (whole entry vs best-matching stretch of the draft) >= --min-sim;
             top --k by similarity, longer first on ties

The acoustic spotter then accepts or rejects each candidate frame by frame (tau, unchanged from `F36`),
so a retrieved name only enters the transcript if the audio supports it. Nothing here reads the
benchmark's references, entity annotations or bias lists.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_soundblind import BENCH, norm  # noqa: E402

TITLES = Path("E:/Claude/datasets/pen/thwiki/titles.gz")
_THAI_ONLY = re.compile(r"^[\u0e00-\u0e7f]+$")


def load_catalog() -> list[str]:
    from pythainlp.corpus import thai_words
    common = {norm(w) for w in thai_words()}
    cat = set()
    for l in gzip.open(TITLES, "rt", encoding="utf-8"):
        t = re.sub(r"\s*\(.*\)$", "", l.strip().replace("_", " "))
        t = norm(t)
        if _THAI_ONLY.match(t) and 3 <= len(t) <= 30 and t not in common:
            cat.add(t)
    return sorted(cat)


def grams(s: str) -> set[str]:
    return {s[i:i + 3] for i in range(len(s) - 2)}


def semiglobal_sim(entry: str, text: str) -> float:
    """1 - (min edits to turn `entry` into ANY substring of `text`) / len(entry)."""
    prev = [0] * (len(text) + 1)                     # free start anywhere in text
    for i, a in enumerate(entry, 1):
        cur = [i]
        for j, b in enumerate(text, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return 1 - min(prev) / len(entry)                # free end anywhere


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hyps")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--min-sim", type=float, default=0.6)
    ap.add_argument("--split", default="real")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cat = load_catalog()
    print("catalog entries", len(cat), flush=True)
    index = defaultdict(list)
    for i, e in enumerate(cat):
        for g in grams(e):
            index[g].append(i)
    index = {g: v for g, v in index.items() if len(v) <= 5000}     # very common trigrams carry no signal
    want = {json.loads(l)["utt_id"] for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")
            if json.loads(l)["split"] == a.split}
    lists, sizes = {}, []
    for l in open(a.hyps, encoding="utf-8"):
        r = json.loads(l)
        if r["utt_id"] not in want or r.get("condition", "none") != "none":
            continue
        d = norm(r["text"])
        hits = Counter(i for g in grams(d) for i in index.get(g, ()))
        cands = [i for i, c in hits.most_common(3000) if c >= max(1, len(grams(cat[i])) / 3) and cat[i] not in d]
        # an entry already spelled exactly in the draft cannot change the transcript -- it only takes a slot
        scored = [(semiglobal_sim(cat[i], d), len(cat[i]), cat[i]) for i in cands]
        scored = sorted((s for s in scored if s[0] >= a.min_sim), reverse=True)[: a.k]
        lists[r["utt_id"]] = [e for _, _, e in scored]
        sizes.append(len(lists[r["utt_id"]]))
    out = Path(a.out or f"E:/Claude/datasets/pen/catlists_{Path(a.hyps).stem}_k{a.k}.json")
    out.write_text(json.dumps(lists, ensure_ascii=False), encoding="utf-8")
    print(f"{len(lists)} utts, list size mean {sum(sizes) / len(sizes):.1f}, empty {sizes.count(0)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
