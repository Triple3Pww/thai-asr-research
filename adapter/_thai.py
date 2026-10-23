"""Thai text normalisation shared by the adapter scripts. One definition, because a CER metric
and a training label that disagree about Thai spelling produce a fake win.

THE ONE RULE HERE, AND WHY IT IS NOT OBVIOUS.

Thai SARA AM -- the vowel in `บำรุง` -- has two encodings that RENDER IDENTICALLY:

    U+0E33                composed    ำ
    U+0E4D U+0E32         decomposed  ํ + า   (NIKHAHIT + SARA AA)

🔴 Unicode gives U+0E33 **no canonical decomposition**, so `unicodedata.normalize('NFC'|'NFD')`
does NOT convert between them. Measured: NFC changed 0 of 150 clips. Every naive pipeline
therefore treats two identical-looking strings as different, and silently.

Measured 2026-09-12 on the 150 Common Voice refs through `typhoon-asr-realtime` (:8007):

    Typhoon's hypotheses   57 decomposed,  0 composed   -- it ALWAYS decomposes
    Common Voice refs       0 decomposed, 57 composed   -- they ALWAYS compose
    clips affected         47 of 150 (31%)

Folding them costs nothing and buys a lot:

    CER unfolded   mean 0.0706   median 0.0405   perfect 57/150
    CER folded     mean 0.0518   median 0.0000   perfect 85/150

26.6% of the mean and ALL of the median was an encoding difference, not a recognition error.

🔴 WHY THIS IS A TRAP AND NOT JUST AN INACCURACY. The generated text banks use the composed form
exclusively (987 occurrences, 0 decomposed). Train the decoder on those labels with an unfolded
metric and it learns to emit composed SARA AM, scoring ~26% better on general CER **purely from
the convention change** -- and the design's Sec 8 gate would adopt the arm for a reason that has
nothing to do with synthetic audio. It is the same class of error as labelling numbers as digits
when Typhoon writes words, found the same way, in the same hour.

So: fold when SCORING (both sides), and match Typhoon when writing a LABEL. `text` -- what VoxCPM
speaks -- keeps the composed form, because that is ordinary written Thai and the TTS should see
ordinary Thai.
"""

from __future__ import annotations

import re

SARA_AM = "ำ"
NIKHAHIT_SARA_AA = "ํา"

#: Stripped before scoring. Thai has no word spaces, so whitespace is never contrastive here, and
#: Typhoon emits almost no punctuation (1 space in 53 real transcripts) while reference corpora do.
SCORING_PUNCT = re.compile(r"[\s\.,!\?;:\"'()\[\]{}…\-–—‚„“”‘’]+")


def fold_sara_am(s: str) -> str:
    """Decomposed -> composed. Use on BOTH sides of a comparison, never on one."""
    return s.replace(NIKHAHIT_SARA_AA, SARA_AM)


def as_typhoon_writes(s: str) -> str:
    """Composed -> decomposed: the form `typhoon-asr-realtime` actually emits.

    For a training label, so fine-tuning does not quietly change the model's output spelling.
    """
    return s.replace(SARA_AM, NIKHAHIT_SARA_AA)


def for_scoring(s: str) -> str:
    """Fold SARA AM, drop whitespace and punctuation. What both CER operands go through."""
    return SCORING_PUNCT.sub("", fold_sara_am(s))


# ---------------------------------------------------------------- Thai numerals
#
# THE SECOND CONVENTION, and it is not an encoding one. Typhoon writes number WORDS and never
# digits (0 ASCII digits in 53 real turns); reference corpora write digits (81 of the 682 real
# `thai-contextasr-bench` utterances contain one). Both transcribe the same speech correctly, and
# a character-level CER charges the full distance between them -- `1979` against
# `หนึ่งพันเก้าร้อยเจ็ดสิบเก้า` is 26 characters of "error" for a perfect transcription.
#
# Measured 2026-09-17 over all 7 systems on the real split: digit-vs-word disagreement is
# **9.1% of the total error mass**. See `_ortho.py`, which spends these functions.
#
# Lived in verify_synth.py until 2026-09-17; moved here when `_ortho.py` became a second caller,
# because two copies of a number speller is exactly how a convention rule drifts apart.

_THAI_DIGIT = ["ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า"]
_SCALE = [(1000000, "ล้าน"), (100000, "แสน"), (10000, "หมื่น"), (1000, "พัน"), (100, "ร้อย")]

DIGIT_RUN = re.compile(r"[0-9]+")


def thai_cardinal(n: int) -> str:
    """Thai number words for a non-negative integer, with the two irregulars Thai actually has.

    ยี่สิบ, not สองสิบ, for 20-29; and the final 1 of any number above 10 is เอ็ด, not หนึ่ง.
    Those two are why this cannot be a digit-by-digit lookup.
    """
    if n < 10:
        return _THAI_DIGIT[n]
    out = []
    for value, name in _SCALE:
        if n >= value:
            out.append(thai_cardinal(n // value) + name)
            n %= value
    if n >= 10:
        tens = n // 10
        out.append(("" if tens == 1 else "ยี่" if tens == 2 else _THAI_DIGIT[tens]) + "สิบ")
        n %= 10
    if n:
        out.append("เอ็ด" if (out and n == 1) else _THAI_DIGIT[n])
    return "".join(out)


def thai_digitwise(s: str) -> str:
    """Read a digit run out one digit at a time -- the reading Typhoon uses for phone numbers."""
    return "".join(_THAI_DIGIT[int(c)] for c in s)


def infix_cer(needle: str, hay: str) -> float:
    """Edit distance from `needle` to its best-matching substring of `hay`, over len(needle).

    Free start and free end in `hay`: the question is "does this appear somewhere in there", not
    "is the whole string this". Used to ask whether ONE word -- a rare entity -- survived, which a
    whole-utterance CER cannot answer: an 8-character name inside an 85-character sentence can be
    destroyed while the sentence still scores 0.09.

    Both sides should already be through `for_scoring`.
    """
    if not needle:
        return 0.0
    if not hay:
        return 1.0
    prev = [0] * (len(hay) + 1)                    # row 0: starting anywhere in hay is free
    for i, nc in enumerate(needle, 1):
        cur = [i] + [0] * len(hay)
        for j, hc in enumerate(hay, 1):
            cur[j] = min(prev[j - 1] + (nc != hc), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return min(prev) / len(needle)                  # ending anywhere in hay is free


# ------------------------------------------------------------- lexical tone, as a measurable thing
#
# Thai writes tone with combining marks, so a tone error is a CHARACTER error and can be isolated
# from every other kind. That makes "did this transformation damage lexical tone?" answerable with
# string arithmetic instead of a pitch model -- the instrument `docs/.../design.md` Phase 0b calls
# the ruler, in its cheapest form.
#
# ⚠️ IT IS A FLOOR, NOT THE TRUTH. Thai tone is determined by mark AND consonant class AND syllable
# shape, so an unmarked syllable still carries a tone this cannot see, and a recogniser can get the
# tone right while writing a different word. A rise in TONE-CER is strong evidence of tone damage;
# no rise is weak evidence of safety.

TONE_MARKS = "\u0e48\u0e49\u0e4a\u0e4b"        # MAI EK, MAI THO, MAI TRI, MAI CHATTAWA


def tone_sequence(s: str) -> str:
    """Just the tone marks, in order. `ข้าวเหนียวม่วง` -> `้่`."""
    return "".join(c for c in s if c in TONE_MARKS)
