#!/usr/bin/env python3
"""Permissible orthographic variation: rewrite the REFERENCE, never the hypothesis.

    python adapter/_ortho.py --selftest

WHY THIS EXISTS, AND WHY IT IS NOT A WAY OF MAKING OUR NUMBERS LOOK BETTER.

This repo spent two days finding its "convention traps" one at a time -- SARA AM (F1), number
words vs digits (F2), script convention, colloquial spelling. On 2026-09-17 a literature pass
found they are one named problem with published methods:

    A39  Guha et al. 2023 (2303.05046)   unsupervised WER standardization, -13.28% WER avg
    A40  Bhogale et al. 2026 (2603.00941) OIWER; accounting for permissible variation narrowed
                                          a Gemini-vs-Canary gap from 18.1 to 11.5 points

A40's shape is the one used here: a reference is not one string, it is a SET of renderings that
are all correct transcriptions of the same speech, and the score should be taken against the
best-matching member of that set. That is not leniency -- charging a model for writing
`หนึ่งพันเก้าร้อยเจ็ดสิบเก้า` where the corpus typed `1979` measures the corpus's typing habits.

🔴 THE RULE THAT KEEPS THIS HONEST. Only the reference moves, and only between renderings a Thai
reader would call equally correct. The hypothesis is never touched, no threshold ever forgives a
near-miss, and every rung is applied IDENTICALLY to all seven systems. If a rung can turn a wrong
transcription into a right one, it does not belong here.

WHAT THE BENCHMARK ALREADY DOES, so this does not redo it. `scorer/metrics.py:_ref_variants`
already scores span metrics against the best per-entity dual-form substitution -- so a gold
entity written in either script is already free. Probed 2026-09-17 on the real split: 439 of 439
gold spans located, and the `cap=32` identity-only fallback fires on only 9 of 298 entity-bearing
utterances. That defence is healthy; what leaks past it is everything NOT annotated as a gold
entity.

MEASURED ERROR MASS, all 7 systems, real split, condition `none` (56,232 aligned error chars):

    20.4%   script: Latin <-> Thai, outside gold spans   (13.2% ref-Latin, 7.2% ref-Thai)
     9.1%   number: digits <-> Thai words                 (7.6% ref-digits, 1.5% ref-words)
     3.0%   catastrophic runs >= 60 chars                 (decoder collapse -- a REAL failure)
     1.0%   filler dropped by the model
    66.5%   everything else

So roughly **three tenths of the error mass in this benchmark is convention**, before a single
recognition error is counted.

THE RUNGS (`--rung`), each strictly containing the last:

    none    the recorded baseline -- the benchmark's scorer untouched
    num     digit runs may be read as digits, as a cardinal, or digit-by-digit
    ent     + a known entity's other listed forms, at un-annotated occurrences
    filler  + standalone hesitation markers dropped from both sides   (NOT on by default)

`filler` is the one rung that is NOT purely orthographic -- deleting เอ่อ is deleting a sound the
speaker made. It is last, it is 1% of the mass, and it is off by default; see `FILLERS` for the
real word its first implementation ate.

🔴 THE SCRIPT FAMILY IS THE BIG ONE AND IS **NOT** NORMALISED HERE. Latin<->Thai disagreement is
20.4% of the error mass, twice the number family, and the `ent` rung barely dents it: it moved 4-5
extra references per system and bought 0.0002 CER. The reason is that what leaks past the
benchmark's own dual-form defence is un-annotated loanwords (`prawat`/`ประวัติ`,
`japanesecoaster`/`แจแปนิสโคสเตอร์`) whose pairing no inventory in this corpus declares.
Normalising it would need Thai<->Latin transliteration, which is a guess, and a guess here can
forgive a real error. Left measured and unnormalised on purpose.

SELECTION, and why it is EXACT PRESENCE and not nearest-match. The first draft picked the
candidate with the smallest infix edit distance to the hypothesis. Its own selftest killed it:
reference `Khlong Toei`, hypothesis `ตลาดบางกะปิเลย` -- a flatly wrong place -- and the rewriter
swapped the reference to `คลองเตย`, because a Thai candidate is nearer to Thai output than a Latin
one is. That is forgiving four characters of a real error for being in the right alphabet.

So a candidate is used ONLY if it appears verbatim in the hypothesis (casefolded, despaced).
No threshold, no nearest-match, no way for the rewrite to turn a wrong transcription into a
right one. The cost is that a half-right number keeps its whole convention penalty -- `1979`
heard as `หนึ่งพันเก้าร้อยแปดสิบเก้า` is still charged all 26 characters rather than the one digit
the model actually got wrong. That under-corrects, which biases against the effect being
measured, and is the direction to be wrong in.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _thai import DIGIT_RUN, thai_cardinal, thai_digitwise  # noqa: E402

RUNGS = ("none", "num", "ent", "filler")

#: Thai digits are folded to Arabic by the benchmark's own `normalize`, but the raw reference
#: text has not been through it yet when we rewrite, so both must be recognised here.
_THAI_DIGITS = {ord("๐") + i: str(i) for i in range(10)}

#: Hesitation markers. Deliberately short and closed: every entry is a sound, not a word that
#: carries meaning. `เนี่ย` and `น่ะ` are NOT here -- they are particles with a function.
#:
#: 🔴 REMOVED BY SUBSTRING, THESE EAT REAL WORDS. Measured 2026-09-17 over all 7 systems' real-split
#: hypotheses: 139 substring hits, and the single most frequent is **อ่าน, "to read"** -- `ผู้อ่าน`
#: (reader), `ผมอ่าน`, `ไปอ่าน`, 35+ occurrences that a naive `.replace` turns into `ผู้น`. It also
#: showed up as a 0.003 drop in typhoon-whisper's entity recall, which is the invariant this
#: script asserts, so the harness caught it rather than the eye.
#:
#: The removal is therefore BOUNDED: a filler goes only when whitespace or the string edge sits on
#: both sides of it, which is a property no syllable inside อ่าน can have. Thai writes no word
#: spaces, so that leaves most genuine unspaced fillers (`เอ่อทฤษ`) in place -- see `strip_fillers`
#: for what the rung is worth once it is safe.
FILLERS = ("เอ่อ", "อ่า", "เอิ่ม", "อืม", "เอ้อ", "เออ")

_BOUNDED_FILLER = re.compile(r"(?:(?<=\s)|(?<=^))(?:" + "|".join(FILLERS) + r")(?=\s|$)")

_WS = re.compile(r"\s+")


def _despace(s: str) -> str:
    return _WS.sub("", s)


_LATIN = re.compile(r"[A-Za-z]")
_LATIN_CHAR = re.compile(r"[a-z0-9]")


def _key(s: str) -> str:
    """Comparison form: despaced and casefolded, matching what the scorer compares on."""
    return _despace(s).casefold()


def _flanked_by_latin(hay: str, i: int, j: int) -> bool:
    return bool((i > 0 and _LATIN_CHAR.match(hay[i - 1]))
                or (j < len(hay) and _LATIN_CHAR.match(hay[j])))


def present(candidate: str, hyp_key: str) -> bool:
    """Is this rendering verbatim in the hypothesis? The only question the rewriter asks.

    🔴 A LATIN CANDIDATE MUST NOT BE GLUED INSIDE A LONGER LATIN WORD. Caught 2026-09-17 by
    eye, on a real spot-check, after the selftest passed: the corpus has an entity `ลิน`/`Lin`,
    and `Lin` sits inside the hypothesis `Clint Clint`, so the reference `คลินตัน` was rewritten
    to `คLinตัน`. Same for `นิว`/`New` inside `New Selland`, turning `นิวซีแลนด์` into
    `Newซีแลนด์`. Both are the benchmark's own documented collision (`metrics.py` carries a
    substring guard for exactly this, citing `ไลน์` inside `เดทไลน์`), arrived at independently.

    Thai gets no such flank test, because Thai writes no word spaces -- `บีทีเอส` is legitimately
    glued to its neighbours. Short Thai forms are excluded at index time instead.
    """
    if not candidate:
        return False
    key = _key(candidate)
    if not _LATIN.search(candidate):
        return key in hyp_key
    start = 0
    while True:
        i = hyp_key.find(key, start)
        if i < 0:
            return False
        if not _flanked_by_latin(hyp_key, i, i + len(key)):
            return True
        start = i + 1


def _pick(candidates: list[str], hyp_key: str) -> str:
    """The reference's own wording unless the hypothesis demonstrably used another rendering.

    Candidate 0 is always what the corpus wrote, and it wins whenever it is present -- so a
    hypothesis that already matches the reference can never move it. Among the alternatives the
    longest present one wins, because a longer form is the more specific claim.
    """
    if present(candidates[0], hyp_key):
        return candidates[0]
    for c in sorted(candidates[1:], key=len, reverse=True):
        if present(c, hyp_key):
            return c
    return candidates[0]


# ------------------------------------------------------------------ numbers

def _number_sites(ref: str) -> list[tuple[int, int, str]]:
    """(start, end, digits) for every digit run in `ref`, Thai or Arabic numerals."""
    arabic = ref.translate(_THAI_DIGITS)
    return [(m.start(), m.end(), m.group()) for m in DIGIT_RUN.finditer(arabic)]


def number_variants(digits: str) -> list[str]:
    """Every reading of a digit run a Thai speaker might have produced.

    The run itself stays first (see `_pick`). A cardinal reading is offered only up to 7 digits:
    past that the number is a phone number or an ID, and `หนึ่งพันล้าน...` is not what was said.
    """
    out = [digits, thai_digitwise(digits)]
    if len(digits) <= 7:
        out.append(thai_cardinal(int(digits)))
    if len(digits) == 4 and digits[0] != "0":       # years are often read as two pairs
        out.append(thai_cardinal(int(digits[:2])) + thai_cardinal(int(digits[2:])))
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------ entities

def build_form_index(annotations: list[dict], min_thai: int = 5,
                     min_latin: int = 3) -> dict[str, list[str]]:
    """Every surface form the corpus declares, mapped to its siblings.

    Built from the WHOLE annotation file, not one utterance: the point of this rung is the
    occurrences that were never annotated as gold.

    🔴 THE TWO LENGTH FLOORS ARE THE COLLISION GUARD, and `min_thai` is the load-bearing one.
    A Thai form has no word boundary to test against, so a short one is simply a syllable that
    will turn up inside unrelated words: the corpus contains `ลิน`, `นิว`, `ไลน์`, `พงศ์`, `ต้น`
    -- 447 forms of 3-4 characters, each a landmine inside ordinary Thai. Five characters keeps
    `บีทีเอส`, `ติ๊กต็อก` and `คลองเตย`, which are the cases this rung exists for. Latin forms
    get a real word-boundary test in `present` instead, so 3 is enough there (and `B2` is still
    out, which is the point).
    """
    index: dict[str, set[str]] = {}
    for row in annotations:
        for e in row.get("entities") or []:
            forms = [f for f in ([e.get("form_th"), e.get("form_latin")]
                                 + list(e.get("form_th_alt") or [])
                                 + list(e.get("latin_alts") or [])) if f]
            forms = [f for f in forms
                     if len(f) >= (min_latin if _LATIN.search(f) else min_thai)]
            for f in forms:
                index.setdefault(f, set()).update(x for x in forms if x != f)
    return {k: sorted(v) for k, v in index.items() if v}


def _entity_sites(ref: str, index: dict[str, list[str]], protect: list[tuple[int, int]]):
    """(start, end, forms) for occurrences of known forms, longest-first, no overlaps.

    `protect` holds the spans the benchmark's own dual-form machinery already covers, which are
    left alone -- rewriting them would move the span the scorer is about to locate.
    """
    sites = []
    taken = [False] * len(ref)
    for a, b in protect:
        for i in range(max(0, a), min(len(ref), b)):
            taken[i] = True
    for form in sorted(index, key=len, reverse=True):
        start = 0
        while True:
            i = ref.find(form, start)
            if i < 0:
                break
            # same flank guard on the reference side: `Lin` must not be found inside `Clinton`
            if (not any(taken[i:i + len(form)])
                    and not (_LATIN.search(form)
                             and _flanked_by_latin(ref.casefold(), i, i + len(form)))):
                for k in range(i, i + len(form)):
                    taken[k] = True
                sites.append((i, i + len(form), [form] + index[form]))
            start = i + 1
    return sorted(sites)


def _protected_spans(ref: str, entities: list[dict]) -> list[tuple[int, int]]:
    """Where each annotated gold entity sits -- first occurrence of its span_text."""
    out = []
    for e in entities or []:
        st = e.get("span_text")
        if not st:
            continue
        i = ref.find(st)
        if i >= 0:
            out.append((i, i + len(st)))
    return out


# ------------------------------------------------------------------ the rewriter

def rewrite_reference(ref: str, hyp: str, entities: list[dict], rung: str,
                      form_index: dict[str, list[str]] | None = None) -> str:
    """The reference, re-rendered into whichever permissible form this hypothesis used.

    Returns `ref` unchanged at rung `none`, and unchanged anywhere no candidate beats what the
    corpus already wrote.
    """
    if rung == "none":
        return ref
    hyp_key = _key(hyp)
    protect = _protected_spans(ref, entities)

    sites: list[tuple[int, int, list[str]]] = []
    for a, b, digits in _number_sites(ref):
        if any(a < pb and pa < b for pa, pb in protect):   # inside a gold span: not ours
            continue
        sites.append((a, b, number_variants(digits)))
    if rung in ("ent", "filler") and form_index:
        num_spans = [(a, b) for a, b, _ in sites]
        sites += _entity_sites(ref, form_index, protect + num_spans)

    out, prev = [], 0
    for a, b, cands in sorted(sites):
        if a < prev:                                        # overlap: first writer wins
            continue
        out.append(ref[prev:a])
        out.append(_pick(cands, hyp_key))
        prev = b
    out.append(ref[prev:])
    new = "".join(out)

    if rung == "filler":
        new = strip_fillers(new)
    return new


def strip_fillers(text: str) -> str:
    """Drop standalone hesitation markers. The only place a hypothesis is ever touched.

    Symmetry is the point: dropping เอ่อ from the reference alone would charge a model that
    correctly transcribed it.

    Bounded by whitespace or the string edge -- see the `FILLERS` comment for the word this cost
    before it was. That safety is most of the rung's value: Thai writes no word spaces, so an
    unspaced `เอ่อทฤษฎี` keeps its filler, and the rung ends up worth well under its 1.0% share of
    the error mass. It is kept, off by default, as the measured answer to "should we strip
    fillers" rather than an assumption either way.
    """
    return _BOUNDED_FILLER.sub("", text)


# ------------------------------------------------------------------ selftest

def _selftest() -> int:
    idx = build_form_index([{"entities": [
        {"span_text": "Khlong Toei", "form_th": "คลองเตย", "form_latin": "Khlong Toei"},
        {"span_text": "บีทู", "form_th": "บีทู", "form_latin": "B2"},
    ]}])
    ok = True

    def chk(label, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {label:46s} got={got!r} want={want!r}")

    chk("present: verbatim substring", present("เจ็ดสิบ", _key("มีเจ็ดสิบคน")), True)
    chk("present: one character off is absent", present("เจ็ดสิบ", _key("มีเจ็ดสับคน")), False)
    chk("present: ignores case and spaces", present("Khlong Toei", _key("ตลาด khlongtoei")), True)

    chk("num: hyp said the cardinal",
        rewrite_reference("ปี 1979 นะ", "ปีหนึ่งพันเก้าร้อยเจ็ดสิบเก้านะ", [], "num"),
        "ปี หนึ่งพันเก้าร้อยเจ็ดสิบเก้า นะ")
    chk("num: hyp said the digits, ref untouched",
        rewrite_reference("ปี 1979 นะ", "ปี 1979 นะ", [], "num"), "ปี 1979 นะ")
    chk("num: phone read digit-wise",
        rewrite_reference("โทร 0812", "โทรศูนย์แปดหนึ่งสอง", [], "num"),
        "โทร ศูนย์แปดหนึ่งสอง")
    # 🔴 the rung must not rescue a WRONG number: 1979 misheard as 1989 stays wrong.
    chk("num: a misheard number stays wrong",
        rewrite_reference("ปี 1979 นะ", "ปีหนึ่งพันเก้าร้อยแปดสิบเก้านะ", [], "num"),
        "ปี 1979 นะ")
    chk("num: rung `none` changes nothing",
        rewrite_reference("ปี 1979 นะ", "ปีหนึ่งพันเก้าร้อยเจ็ดสิบเก้านะ", [], "none"),
        "ปี 1979 นะ")

    chk("ent: ref Latin, hyp Thai",
        rewrite_reference("ตลาด Khlong Toei เลย", "ตลาดคลองเตยเลย", [], "ent", idx),
        "ตลาด คลองเตย เลย")
    chk("ent: ref Thai, hyp Latin",
        rewrite_reference("ตลาดคลองเตยเลย", "ตลาด Khlong Toei เลย", [], "ent", idx),
        "ตลาดKhlong Toeiเลย")
    # 🔴 an annotated gold span is the benchmark's job, not ours -- leave it exactly alone.
    chk("ent: annotated gold span is protected",
        rewrite_reference("ตลาด Khlong Toei เลย", "ตลาดคลองเตยเลย",
                          [{"span_text": "Khlong Toei"}], "ent", idx),
        "ตลาด Khlong Toei เลย")
    # 🔴 a different entity is not a permissible variant of this one.
    chk("ent: a wrong entity stays wrong",
        rewrite_reference("ตลาด Khlong Toei เลย", "ตลาดบางกะปิเลย", [], "ent", idx),
        "ตลาด Khlong Toei เลย")
    chk("ent: rung `num` does not do entities",
        rewrite_reference("ตลาด Khlong Toei เลย", "ตลาดคลองเตยเลย", [], "num", idx),
        "ตลาด Khlong Toei เลย")
    chk("ent: short forms like B2 are not indexed", "B2" in idx, False)
    # 🔴 the two collisions a spot-check found after the first selftest passed.
    collide = build_form_index([{"entities": [
        {"span_text": "ลิน", "form_th": "ลิน", "form_latin": "Lin"},
        {"span_text": "นิว", "form_th": "นิว", "form_latin": "New"},
        {"span_text": "บีทีเอส", "form_th": "บีทีเอส", "form_latin": "BTS"},
    ]}])
    chk("ent: 3-char Thai form is not indexed", "ลิน" in collide, False)
    chk("ent: its Latin twin loses its partner too", "Lin" in collide, False)
    chk("ent: a 7-char Thai form is kept", collide.get("บีทีเอส"), ["BTS"])
    chk("ent: `คลินตัน` is not rewritten via ลิน/Lin",
        rewrite_reference("ไบเด้น Hillary คลินตัน", "ไบเดน Clint Clint", [], "ent", collide),
        "ไบเด้น Hillary คลินตัน")
    chk("ent: glued-in Latin is not a match",
        present("Lin", _key("ไบเดน Clint Clint")), False)
    chk("ent: unglued Latin still matches", present("BTS", _key("ลง BTS ที่หมอชิต")), True)
    chk("ent: BTS through the rewriter",
        rewrite_reference("มาด้วยบีทีเอสก็ได้", "มาด้วย BTS ก็ได้", [], "ent", collide),
        "มาด้วยBTSก็ได้")

    chk("filler: standalone marker is dropped",
        rewrite_reference("เอ่อ ไปไหน", "ไปไหน", [], "filler", idx), " ไปไหน")
    chk("filler: dropped from the hypothesis too", strip_fillers("เอ่อ ไปไหน"), " ไปไหน")
    chk("filler: a particle is not a filler", strip_fillers("เนี่ย ไปไหน"), "เนี่ย ไปไหน")
    # 🔴 the one that cost a rung: อ่าน ("to read") contains อ่า, 35+ times in these hypotheses.
    chk("filler: อ่าน survives -- it is a word", strip_fillers("ผู้อ่านที่ดี"), "ผู้อ่านที่ดี")
    chk("filler: unspaced filler is left alone (the price of that)",
        strip_fillers("เอ่อทฤษฎี"), "เอ่อทฤษฎี")

    print("\n  " + ("SELFTEST OK" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show-index", type=int, default=0,
                    help="print N entries of the form index built from the benchmark")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if args.show_index:
        bench = Path("E:/asr-bench/thai-contextasr-bench/data/utterances.jsonl")
        ann = [json.loads(l) for l in bench.open(encoding="utf-8") if l.strip()]
        idx = build_form_index(ann)
        print(f"{len(idx)} indexed forms")
        for k, v in list(idx.items())[:args.show_index]:
            print(f"  {k!r} -> {v}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
