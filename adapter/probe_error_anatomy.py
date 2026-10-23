#!/usr/bin/env python3
"""Where are the remaining errors? A character-class anatomy of one system's hypotheses.

    python adapter/probe_error_anatomy.py results/nectec_fc_greedy_hyps.jsonl [--nbest results/nbest_nectec5_nbest.jsonl]

WHY (2026-10-08, "invent a new ASR architecture from zero"). Every new architecture is a bet on
WHICH errors it removes. Before betting, measure what the best real-time model still gets wrong:

  * edit ops by Thai character class (tone mark / vowel sign / consonant / Latin / other)
  * ablation CERs: drop tone marks on BOTH sides, drop every vowel sign, map to a crude SOUND key
    (consonants folded into sound classes, silent letters + tone marks removed) -- the CER that is
    left under the sound key is the share an acoustic model genuinely mis-HEARD; the rest is
    spelling the right sound the wrong way (a lexical / LM problem, not an acoustic one)
  * share of reference-side errors that land inside annotated entity spans
  * with --nbest: the same anatomy for the 5-best oracle (what a perfect re-ranker would reach)

Own Levenshtein, after `canon()` + SARA AM fold + punctuation strip on both sides. Absolute CERs
differ slightly from the benchmark's `_score.py`; only the SHARES are the point here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _thai import SCORING_PUNCT, fold_sara_am  # noqa: E402
from eval_canon import canon  # noqa: E402

BENCH = Path("E:/asr-bench/thai-contextasr-bench/data/utterances.jsonl")

TONE = set("\u0e48\u0e49\u0e4a\u0e4b")                       # mai ek / tho / tri / chattawa
ABOVE_BELOW = set("\u0e31\u0e34\u0e35\u0e36\u0e37\u0e38\u0e39\u0e3a\u0e47\u0e4c\u0e4d\u0e4e")
LEAD_FOLLOW = set("\u0e40\u0e41\u0e42\u0e43\u0e44\u0e30\u0e32\u0e33\u0e45\u0e46")
CONS = {chr(c) for c in range(0x0E01, 0x0E2F)}
VOWEL = ABOVE_BELOW | LEAD_FOLLOW
THANTHAKHAT = "\u0e4c"

# Consonants that SOUND alike as initials (Thai has 44 letters for 21 initial sounds).
SOUND = {}
for grp, key in [("ขฃคฅฆ", "K"), ("จฉชฌ", "C"), ("ซศษส", "S"), ("ฎด", "D"), ("ฏต", "T"),
                 ("ฐฑฒถทธ", "H"), ("ผพภ", "P"), ("ฝฟ", "F"), ("ญย", "Y"), ("ณน", "N"),
                 ("ลฬ", "L"), ("หฮ", "h")]:
    for ch in grp:
        SOUND[ch] = key


def norm(s: str) -> str:
    return SCORING_PUNCT.sub("", fold_sara_am(canon(s or ""))).lower()


def drop(s: str, chars: set[str]) -> str:
    return "".join(c for c in s if c not in chars)


def sound_key(s: str) -> str:
    out = []
    for i, c in enumerate(s):
        if c in TONE or c == THANTHAKHAT:
            continue
        if i + 1 < len(s) and s[i + 1] == THANTHAKHAT:         # silent letter (karan)
            continue
        out.append(SOUND.get(c, c))
    return "".join(out)


def cls(c: str) -> str:
    if c in TONE:
        return "tone"
    if c in VOWEL:
        return "vowel"
    if c in CONS:
        return "consonant"
    if c.isascii():
        return "latin/digit"
    return "other"


def align(r: str, h: str):
    """Levenshtein with backtrace -> list of (op, ref_char, hyp_char, ref_index)."""
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri, row, prev = r[i - 1], d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ri != h[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            if r[i - 1] != h[j - 1]:
                ops.append(("S", r[i - 1], h[j - 1], i - 1))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", r[i - 1], "", i - 1))
            i -= 1
        else:
            ops.append(("I", "", h[j - 1], max(i - 1, 0)))
            j -= 1
    return d[n][m], ops


def edits(r: str, h: str) -> int:
    return align(r, h)[0]


def entity_mask(ref: str, ents: list[dict]) -> list[bool]:
    mask = [False] * len(ref)
    for e in ents:
        span = norm(e.get("span_text", ""))
        if not span:
            continue
        k = ref.find(span)
        while k >= 0:
            for x in range(k, k + len(span)):
                mask[x] = True
            k = ref.find(span, k + 1)
    return mask


def thai_ref(row) -> str:
    """Latin-script entity spans -> their Thai form, which is what a Thai-script ASR emits.
    The benchmark scorer accepts either form; a plain Levenshtein would charge the convention."""
    t = row["text"]
    row["entities"] = [dict(e) for e in row.get("entities") or []]   # never mutate the shared row
    for e in row["entities"]:
        sp, th = e.get("span_text") or "", e.get("form_th") or ""
        if sp and th and re.search(r"[A-Za-z0-9]", sp):
            t = t.replace(sp, th)
            e["span_text"] = th
    return t


def bursts(ops) -> list[int]:
    """Lengths of runs of edits at adjacent reference positions."""
    pos = sorted(o[3] for o in ops)
    out, run = [], 0
    for k, p in enumerate(pos):
        run = run + 1 if k and p - pos[k - 1] <= 1 else 1
        if k + 1 == len(pos) or pos[k + 1] - p > 1:
            out.append(run)
    return out


def anatomy(pairs, refs):
    tot = {"burst1": 0, "burst2_3": 0, "burst4p": 0, "chars": 0, "edits": 0, "no_tone": 0, "no_vowel": 0, "sound": 0, "ent_edits": 0,
           "ent_chars": 0, "perfect": 0, "n": 0}
    by = {}
    for uid, hyp in pairs:
        row = refs[uid]
        r, h = norm(thai_ref(row)), norm(hyp)
        e, ops = align(r, h)
        mask = entity_mask(r, row.get("entities") or [])
        tot["chars"] += len(r)
        tot["edits"] += e
        tot["perfect"] += e == 0
        tot["n"] += 1
        tot["ent_chars"] += sum(mask)
        tot["no_tone"] += edits(drop(r, TONE), drop(h, TONE))
        tot["no_vowel"] += edits(drop(r, TONE | VOWEL), drop(h, TONE | VOWEL))
        tot["sound"] += edits(sound_key(r), sound_key(h))
        for b in bursts(ops):
            tot["burst1" if b == 1 else "burst2_3" if b <= 3 else "burst4p"] += b
        for op, rc, hc, ri in ops:
            key = f"{op}:{cls(rc or hc) if op != 'S' else cls(rc) + '->' + cls(hc)}"
            by[key] = by.get(key, 0) + 1
            if ri < len(mask) and mask[ri]:
                tot["ent_edits"] += 1
    return tot, by


def report(name, tot, by):
    c, e = tot["chars"], tot["edits"]
    print(f"\n== {name}  ({tot['n']} utts, {c} ref chars)")
    print(f"CER (own scorer)             {e / c:.4f}   perfect utts {tot['perfect']}/{tot['n']}")
    print(f"  tone marks dropped         {tot['no_tone'] / c:.4f}   -> tone-only share {1 - tot['no_tone'] / e:.1%}")
    print(f"  + all vowel signs dropped  {tot['no_vowel'] / c:.4f}")
    print(f"  SOUND key (homophone-fold) {tot['sound'] / c:.4f}   -> spelling-not-hearing share {1 - tot['sound'] / e:.1%}")
    print(f"  edits inside entity spans  {tot['ent_edits']}/{e} = {tot['ent_edits'] / e:.1%}"
          f"   (entity chars are {tot['ent_chars'] / c:.1%} of the text)")
    print(f"  edits in bursts of 1 / 2-3 / 4+ chars: {tot['burst1'] / e:.1%} / {tot['burst2_3'] / e:.1%} / {tot['burst4p'] / e:.1%}")
    print("  top edit classes:")
    for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:12]:
        print(f"    {k:28s} {v:5d}  {v / e:.1%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hyps")
    ap.add_argument("--nbest")
    ap.add_argument("--out")
    a = ap.parse_args()
    refs = {json.loads(l)["utt_id"]: json.loads(l) for l in BENCH.open(encoding="utf-8")}
    refs = {k: v for k, v in refs.items() if v["split"] == "real"}
    hyp = [json.loads(l) for l in open(a.hyps, encoding="utf-8")]
    hyp = [(r["utt_id"], r["text"]) for r in hyp if r["utt_id"] in refs and r.get("condition", "none") == "none"]
    t, b = anatomy(hyp, refs)
    report(Path(a.hyps).name, t, b)
    out = {"top1": {"tot": t, "by": b}}
    if a.nbest:
        nb = [json.loads(l) for l in open(a.nbest, encoding="utf-8")]
        orc = []
        for r in nb:
            if r["utt_id"] not in refs:
                continue
            ref = norm(thai_ref(refs[r["utt_id"]]))
            orc.append((r["utt_id"], min((c["text"] for c in r["cands"]), key=lambda x: edits(ref, norm(x)))))
        t2, b2 = anatomy(orc, refs)
        report("5-best ORACLE", t2, b2)
        out["oracle5"] = {"tot": t2, "by": b2}
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
