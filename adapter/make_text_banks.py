#!/usr/bin/env python3
"""Build the two Thai text banks the adapter experiment needs, via the Antigravity CLI (`agy`).

    python adapter/make_text_banks.py --phase 1
    python adapter/make_text_banks.py --phase 2
    python adapter/make_text_banks.py --phase 1 --batch 40 --batches 10

WHY TWO BANKS AND NOT ONE. the adapter design (`git -C E:/Claude/asr-research show 4734fc5:docs/specs/2026-09-11-synthetic-adapter-asr-design.md`) Sec 5.3 trains in
two phases against two different datasets, and the split is the whole mechanism:

  phase 1  adapter trains, decoder FROZEN, on words the decoder ALREADY KNOWS.
           If rare words appeared here the loss would be high because the decoder cannot spell
           them yet, and the adapter would be blamed for a vocabulary problem it cannot fix.
           Restrict phase 1 to known words and the only remaining source of loss is TTS-ness.
  phase 2  decoder trains, adapter FROZEN, on the rare words -- the audio no Thai corpus
           contains, which is the entire reason synthetic data is worth the risk.

So bank 1 must contain NO rare tokens and NO digits, and bank 2 must contain one rare token per
sentence, recorded separately so Sec 8's rare-word accuracy metric has something to score.

THE SPOKEN/WRITTEN SPLIT, which is the non-obvious part.

A postal code is *written* 50200 and *spoken* "ha sun song sun sun" -- digit by digit. The TTS
must be fed the spoken form (it is a speech model; it cannot be relied on to read 50200 the way
a Thai person does), while the ASR label must be the written form. So bank 2 carries both:

    text        what VoxCPM speaks      ... no digits, ever
    label_text  what the ASR must emit  ... digits where digits belong
    entity      the rare token alone    ... what Sec 8 scores

This is exactly the job Typhoon's own synthetic data does -- in both their general corpus (0.03%
synthetic) and their Isan stage (20.6%), the TTS exists for numeric/date normalisation and
nothing else. It teaches what to WRITE, never how speech SOUNDS, which is the finding this whole
design is arranged around.

`gen_colab.py` carries every key except `id`/`text` straight into the manifest, so `label_text`
and `entity` arrive beside the audio with no extra plumbing.

WHY `agy` AND NOT CLAUDE WRITING THE THAI. House rule (global CLAUDE.md): gemini writes the
words. The output is still reviewed here -- see `validate_*`, which rejected nothing on the
probe batches but is the reason a bad batch cannot reach the bank silently.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

AGY = Path(r"C:\Users\MARU\AppData\Local\agy\bin\agy.exe")
OUT_DIR = Path(__file__).resolve().parent / "texts"

THAI = re.compile(r"[\u0E00-\u0E7F]")
DIGIT = re.compile(r"[0-9\u0E50-\u0E59]")     # ASCII and Thai numerals both
LATIN = re.compile(r"[A-Za-z]")

#: Thai postal codes start with a two-digit province code in the range 10-96. This is a RANGE
#: SANITY CHECK ONLY, not a province table -- there are real gaps in that range and this does not
#: know them, so it catches "00123" and "99999" and nothing subtler. Province-correctness was
#: checked by hand on the probe batch (Phayao 56, Khon Kaen 40, Chai Nat 17, Chumphon 86,
#: Trat 23 -- all correct), which is evidence about the model, not a guarantee about every row.
POST_PREFIX = set(range(10, 97))

# ---------------------------------------------------------------------------------------------
# Phase 1: everyday Thai, no rare tokens, no digits. Themes are call-centre shaped because that
# is the deployment domain -- a voice agent on a phone call, not read news.
# ---------------------------------------------------------------------------------------------
PHASE1_THEMES = [
    "a customer phoning a shop to ask about opening hours and how to find it",
    "a customer asking why something they ordered has not arrived yet",
    "a customer asking what payment methods are accepted and how to pay",
    "a customer describing something wrong with an item they bought",
    "someone making, moving or confirming an appointment",
    "someone asking a call centre to explain how a service works",
    "greetings, thanks, apologies and polite closings on a phone call",
    "someone cancelling or postponing something they had arranged",
    "someone asking whether a discount or a cheaper option is available",
    "someone on the phone describing where they are standing right now",
    "a customer asking to speak to a different person or department",
    "someone repeating or correcting what they just said because it was misheard",
]

PHASE1_PROMPT = """You are building a Thai speech corpus. Write Thai sentences that a \
text-to-speech system will read aloud.

Write exactly {n} Thai sentences on this theme: {theme}.

HARD RULES:
- Use ONLY common everyday Thai vocabulary that any Thai speaker knows.
- NO numbers, NO digits, NO dates, NO times, NO quantities.
- NO brand names, NO product names, NO names of places, NO names of people.
- NO English words, NO abbreviations, NO acronyms.
- Each sentence is ONE complete spoken utterance, 8 to 20 words, natural spoken Thai (not \
formal written Thai) - the kind of thing a person actually says out loud on a phone call.
- Vary the speaker: some male (ครับ), some female (ค่ะ/คะ).
- Every sentence must be different in wording, not a template with one word swapped.

OUTPUT FORMAT: plain text, one sentence per line, nothing else. No numbering, no bullets, no \
markdown, no quotes, no blank lines, no preamble, no explanation."""

# ---------------------------------------------------------------------------------------------
# Phase 2: one rare token per sentence. Three kinds, because they fail differently -- a place
# name is an unseen syllable sequence, a postal code is a normalisation problem, a product name
# is a foreign word forced through Thai phonotactics.
# ---------------------------------------------------------------------------------------------
PHASE2_SPECS = [
    ("place", "REAL but UNCOMMON Thai sub-district (ตำบล) or district (อำเภอ) names from the "
              "NORTH of Thailand (Chiang Rai, Nan, Phayao, Mae Hong Son, Lampang)"),
    ("place", "REAL but UNCOMMON Thai sub-district (ตำบล) or district (อำเภอ) names from the "
              "NORTHEAST of Thailand (Ubon Ratchathani, Sakon Nakhon, Roi Et, Buriram, Loei)"),
    ("place", "REAL but UNCOMMON Thai sub-district (ตำบล) or district (อำเภอ) names from the "
              "SOUTH of Thailand (Phatthalung, Satun, Trang, Chumphon, Narathiwat)"),
    ("place", "REAL but UNCOMMON Thai sub-district (ตำบล) or district (อำเภอ) names from the "
              "CENTRAL and EASTERN provinces (Suphanburi, Chai Nat, Sa Kaeo, Trat, Prachinburi)"),
    ("postcode", "Thai postal codes for small districts across different provinces"),
    ("postcode", "Thai postal codes, where the caller is reading the code back to confirm it"),
    ("postcode", "Thai postal codes, where the caller corrects a code that was written wrongly"),
    ("product", "product and brand names written in THAI SCRIPT as Thai people transliterate "
                "them - household goods, electronics, medicines, snack brands"),
    ("product", "specific product model or variant names written in THAI SCRIPT - the kind a "
                "customer reads off a box or a receipt"),
    ("product", "names of Thai food and drink products, regional specialities and local brands, "
                "written in THAI SCRIPT"),
]

PHASE2_PROMPT = """You are building a Thai speech corpus for training a Thai speech-recognition \
model to handle RARE WORDS that general corpora do not contain.

Write exactly {n} lines. Each line has THREE fields separated by a vertical bar:

SPOKEN|WRITTEN|ENTITY

- SPOKEN = the sentence exactly as a Thai person SAYS it out loud. Any number or code is spoken \
digit by digit in Thai words (example: 50200 is spoken as "ห้า ศูนย์ สอง ศูนย์ ศูนย์"). There \
must be NO digits at all in this field.
- WRITTEN = the same sentence as it should be TRANSCRIBED, with numbers and codes written as \
digits.
- ENTITY = the rare word or code being taught in that sentence, on its own, exactly as it \
appears in WRITTEN.

Content rules:
- The rare word in each sentence must be: {kind}.
- Each sentence is someone on a real phone call - confirming a delivery address, asking about a \
product, or checking an order.
- Natural spoken Thai, 10 to 25 words. Vary male (ครับ) and female (ค่ะ/คะ) speakers.
- NO English letters anywhere, in any field.
- Every sentence must teach a DIFFERENT rare word.

OUTPUT: plain text, one line per item, exactly three bar-separated fields. No numbering, no \
markdown, no quotes, no preamble."""


#: Per-call ceiling. 180 s lost one batch of 40 outright on the first phase-1 run; a phase-2 call
#: emits three fields per line and is longer still, so this is generous on purpose -- a timeout
#: costs the whole batch, and `agy` gives no partial output.
ASK_TIMEOUT = 420


def ask(prompt: str, timeout: int = ASK_TIMEOUT) -> str:
    """One `agy` call. The prompt goes in on stdin, which is both the prompt and the EOF.

    `agy` is a TUI-style CLI: left with an open stdin and no EOF it blocks forever. Piping is
    the documented way to avoid that and it also sidesteps every quoting problem with Thai.
    """
    p = subprocess.run([str(AGY)], input=prompt.encode("utf-8"),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return p.stdout.decode("utf-8", errors="replace")


def validate_phase1(line: str) -> str | None:
    """Returns a reason to reject, or None to keep."""
    if not THAI.search(line):
        return "no Thai script"
    if DIGIT.search(line):
        return "contains a digit"
    if LATIN.search(line):
        return "contains Latin letters"
    if not (30 <= len(line) <= 160):
        return f"length {len(line)}"
    if line.count("|"):
        return "stray delimiter"
    return None


def validate_phase2(fields: list[str]) -> str | None:
    if len(fields) != 3:
        return f"{len(fields)} fields, expected 3"
    spoken, written, entity = fields
    if not all(THAI.search(x) or DIGIT.search(x) for x in (spoken, written, entity)):
        return "a field has neither Thai nor digits"
    if DIGIT.search(spoken):
        return "digit in SPOKEN (it must be speakable)"
    if any(LATIN.search(x) for x in fields):
        return "contains Latin letters"
    if entity not in written:
        return "entity does not appear in WRITTEN"
    if not (30 <= len(spoken) <= 220):
        return f"spoken length {len(spoken)}"
    # A bare 5-digit entity is a postal code: check the province prefix is real. This is the one
    # automated authenticity check available -- a hallucinated code usually gets it wrong.
    if re.fullmatch(r"\d{5}", entity) and int(entity[:2]) not in POST_PREFIX:
        return f"postal prefix {entity[:2]} is not a Thai province"
    return None


def build_phase1(n: int, batches: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    rejected: dict[str, int] = {}
    for i in range(batches):
        theme = PHASE1_THEMES[i % len(PHASE1_THEMES)]
        print(f"[{i+1}/{batches}] {theme[:60]}...", flush=True)
        try:
            raw = ask(PHASE1_PROMPT.format(n=n, theme=theme))
        except subprocess.TimeoutExpired:
            print("    timed out, skipping batch", flush=True)
            continue
        kept = 0
        for line in (l.strip() for l in raw.splitlines()):
            if not line:
                continue
            line = re.sub(r"^\s*[-*\d]+[.)]?\s*", "", line)      # strip stray numbering
            why = validate_phase1(line)
            if why:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            if line in seen:
                rejected["duplicate"] = rejected.get("duplicate", 0) + 1
                continue
            seen.add(line)
            rows.append({"text": line, "theme": theme, "phase": 1})
            kept += 1
        print(f"    kept {kept}", flush=True)
    if rejected:
        print("rejected:", rejected)
    return rows


def build_phase2(n: int, batches: int) -> list[dict]:
    rows: list[dict] = []
    seen_ent: set[str] = set()
    seen_txt: set[str] = set()
    rejected: dict[str, int] = {}
    for i in range(batches):
        kind_name, kind = PHASE2_SPECS[i % len(PHASE2_SPECS)]
        print(f"[{i+1}/{batches}] {kind_name}: {kind[:55]}...", flush=True)
        try:
            raw = ask(PHASE2_PROMPT.format(n=n, kind=kind))
        except subprocess.TimeoutExpired:
            print("    timed out, skipping batch", flush=True)
            continue
        kept = 0
        for line in (l.strip() for l in raw.splitlines()):
            if not line or "|" not in line:
                continue
            line = re.sub(r"^\s*[-*\d]+[.)]?\s*", "", line)
            fields = [x.strip() for x in line.split("|")]
            why = validate_phase2(fields)
            if why:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            spoken, written, entity = fields
            if spoken in seen_txt:
                rejected["duplicate sentence"] = rejected.get("duplicate sentence", 0) + 1
                continue
            # One clip per rare word. A repeated entity is a repeated lesson, and the bank's
            # value is coverage -- 300 sentences teaching 120 words is a worse bank than 300
            # teaching 300.
            if entity in seen_ent:
                rejected["duplicate entity"] = rejected.get("duplicate entity", 0) + 1
                continue
            seen_txt.add(spoken)
            seen_ent.add(entity)
            rows.append({"text": spoken, "label_text": written, "entity": entity,
                         "entity_kind": kind_name, "phase": 2})
            kept += 1
        print(f"    kept {kept}", flush=True)
    if rejected:
        print("rejected:", rejected)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=(1, 2), required=True)
    ap.add_argument("--batch", type=int, default=40, help="sentences per agy call")
    ap.add_argument("--batches", type=int, default=0, help="default: one per theme/spec")
    args = ap.parse_args()

    if not AGY.exists():
        print(f"agy not found at {AGY}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.phase == 1:
        batches = args.batches or len(PHASE1_THEMES)
        rows = build_phase1(args.batch, batches)
        prefix, out = "p1", OUT_DIR / "phase1_known.json"
    else:
        batches = args.batches or len(PHASE2_SPECS)
        rows = build_phase2(args.batch, batches)
        prefix, out = "p2", OUT_DIR / "phase2_rare.json"

    # Ids are assigned last and are stable for a given file, because gen_colab seeds each
    # generation from the id -- renumbering the bank would regenerate the whole corpus.
    for i, r in enumerate(rows, 1):
        r["id"] = f"{prefix}_{i:04d}"
    rows = [{"id": r.pop("id"), **r} for r in rows]

    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    el = time.perf_counter() - t0
    print(f"\n{len(rows)} sentences -> {out}  ({el:.0f}s)")
    if args.phase == 2:
        import collections
        print("by kind:", dict(collections.Counter(r["entity_kind"] for r in rows)))
        print("distinct entities:", len({r["entity"] for r in rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
