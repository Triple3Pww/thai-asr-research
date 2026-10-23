#!/usr/bin/env python3
"""Rewrite both text banks into Typhoon's OWN transcription convention.

    python adapter/fix_label_convention.py --apply

WHY. Measured 2026-09-12 against 53 real Thai turns in `live/_out/turns-*.jsonl` (the live panel
drives `asr/server.py` on :8007, so `asked` is `typhoon-asr-realtime`'s own output):

    ASCII digits in Typhoon's transcripts   0 / 53
    Thai number words                       7
    transcripts containing any space        1 / 53   (mean 0.02 per transcript)

The decisive row is a phone number read aloud -- exactly where a digit-writing ASR would emit
digits: `ชื่อพอร์ตศูนย์แปดสามศูนย์หนึ่งแปดเก้าห้าสามเจ็ดครับ`. It wrote words. Dates too:
`วันที่ยี่สิบเอ็ด`.

The generated banks had it the other way round -- 279/279 phase-2 labels carried ASCII digits and
averaged 4.64 spaces. Phase 2 trains the DECODER, so training on those labels teaches it to change
convention, which raises general CER -- the harm term in the design's Sec 8 gate -- and would read
as a leak when it is really a relabelling.

WHAT THIS DOES.

1. `label_text` = `text` with all whitespace removed, for BOTH banks. `text` keeps its spaces:
   it is fed to VoxCPM, where spacing helps it phrase a digit sequence as separate digits instead
   of one word. The label is what the ASR must emit; the text is what the speaker says. They were
   always different fields and now they differ on spacing rather than on digits.

   🔴 Phase 1 needs this too, and it is easy to miss because that bank has no numbers. Phase 1
   trains the adapter through a FROZEN decoder, so if the target text disagrees with the decoder's
   spacing convention the loss is dominated by an error the adapter cannot fix -- the exact
   failure the two-phase split exists to avoid.

2. Entities are reduced to their Thai core (digits and spaces dropped), because that is the part
   that is actually out-of-vocabulary. `ควอนตัมชาร์จ 160` scores as `ควอนตัมชาร์จ`; the model
   number is a normalisation artifact, not a rare word.

3. A postcode row's entity reduces to nothing, so the row cannot be scored by Sec 8 at all. Those
   rows are SALVAGED where the sentence also names a district or sub-district -- the entity is
   re-pointed at that place name -- and dropped otherwise. This is the honest consequence of the
   finding: with Typhoon already writing number words there is no digit-normalisation gap to
   teach, so a row whose only lesson was normalisation has no lesson left.

IDS ARE NEVER RENUMBERED. `colab/gen_colab.py` seeds every generation from the clip id, so
renumbering would regenerate the corpus instead of extending it. Surviving rows keep their id and
dropped ids simply go missing -- that is intended.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _thai import as_typhoon_writes                                # noqa: E402

TEXTS = Path(__file__).resolve().parent / "texts"

#: A place name runs from a marker to the next space. Verified on this bank: the generator always
#: puts a space after the name (before จังหวัด or รหัสไปรษณีย์), so a greedy Thai-only run
#: terminates on its own -- spot-checked against สะเมิง, เมืองปอน, เวียงแก่น, บ่อเกลือ,
#: ภูกามยาว, วังชิ้น, all clean.
PLACE_AFTER_MARKER = re.compile(r"(?:ตำบล|อำเภอ|แขวง|เขต)\s*([฀-๿]{2,20})")
THAI_ONLY = re.compile(r"[^฀-๿]+")


def to_label(s: str) -> str:
    """Spoken text -> the transcript Typhoon would write.

    Two transformations, both measured against its real output (see `_thai.py` and the
    2026-09-12 UPDATES entry): whitespace removed (1 space in 53 real transcripts), and Thai
    SARA AM written in the DECOMPOSED form it always emits. The second one is invisible -- the
    two encodings render identically -- and is worth 26.6% of mean CER if a scorer does not fold
    them. Training on the composed form would hand a later arm a ~26% CER 'win' that is purely a
    spelling change.
    """
    return as_typhoon_writes(re.sub(r"\s+", "", s))


def entity_core(entity: str) -> str:
    """The out-of-vocabulary part: Thai characters only, no digits, no spaces."""
    return THAI_ONLY.sub("", entity)


def fix_phase1(rows: list[dict]) -> tuple[list[dict], dict]:
    stats = {"rows": len(rows), "label_added": 0, "spaces_removed": 0, "sara_am_decomposed": 0}
    for r in rows:
        label = to_label(r["text"])
        stats["spaces_removed"] += len(r["text"]) - len(re.sub(r"\s+", "", r["text"]))
        r["label_text"] = label
        stats["label_added"] += 1
        stats["sara_am_decomposed"] += r["text"].count("ำ")
    return rows, stats


def fix_phase2(rows: list[dict]) -> tuple[list[dict], dict]:
    stats = {"rows_in": len(rows), "relabelled": 0, "entity_reduced": 0,
             "postcode_salvaged": 0, "postcode_dropped": 0, "dropped_other": 0}
    kept: list[dict] = []
    seen_ent: set[str] = set()

    # Non-postcode rows first, so a salvaged place name can never displace a real place entity.
    order = sorted(rows, key=lambda r: r["entity_kind"] == "postcode")

    for r in order:
        old_label = r["label_text"]
        label = to_label(r["text"])
        core = as_typhoon_writes(entity_core(r["entity"]))

        if core and core != r["entity"]:
            stats["entity_reduced"] += 1

        if not core:
            # pure-numeric entity: the lesson was normalisation and there is none left
            m = PLACE_AFTER_MARKER.search(old_label)
            cand = as_typhoon_writes(m.group(1)) if m else ""
            if cand and cand not in seen_ent and cand in label:
                r["entity"] = cand
                r["entity_kind"] = "place"
                r["salvaged_from"] = "postcode"
                stats["postcode_salvaged"] += 1
                core = cand
            else:
                stats["postcode_dropped"] += 1
                continue
        else:
            r["entity"] = core

        if core in seen_ent or core not in label:
            stats["dropped_other"] += 1
            continue

        seen_ent.add(core)
        r["label_text"] = label
        stats["relabelled"] += 1
        kept.append(r)

    # back to id order, because a bank that reads in id order is easier to diff
    kept.sort(key=lambda r: r["id"])
    stats["rows_out"] = len(kept)
    return kept, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the files (default: dry run)")
    args = ap.parse_args()

    for name, fixer in (("phase1_known.json", fix_phase1), ("phase2_rare.json", fix_phase2)):
        path = TEXTS / name
        if not path.exists():
            print(f"{name}: missing, skipped")
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        out, stats = fixer([dict(r) for r in rows])
        print(f"\n{name}")
        for k, v in stats.items():
            print(f"  {k:20s} {v}")
        # every surviving row must satisfy the contract the bank exists to provide
        assert all(not re.search(r"\s", r["label_text"]) for r in out), "space left in a label"
        assert all(not re.search(r"[0-9]", r["label_text"]) for r in out), "digit left in a label"
        if out and "entity" in out[0]:
            assert all(r["entity"] in r["label_text"] for r in out), "entity not in its label"
        if args.apply:
            raw = path.with_suffix(".raw.json")
            if not raw.exists():
                shutil.copy2(path, raw)          # keep the untouched agy output, once
                print(f"  original preserved -> {raw.name}")
            path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                            encoding="utf-8")
            print(f"  written ({len(out)} rows)")
        else:
            print("  dry run -- pass --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
