#!/usr/bin/env python3
"""Per-utterance errors under the benchmark's OFFICIAL scorer, exactly as `eval_canon.py` feeds it.

    from official_utt import Official
    O = Official(); err, ref_len = O.err(utt_id, hyp_text)

WHY (2026-10-08). Tuning toward CER 0.060 on the dev half with a plain Levenshtein misled: it ignores the
scorer's acceptance of an entity in EITHER Thai or Latin form (`entity_forms`), so e.g. a pick that looked
-4.4% plainly was -0.7% officially. This calls the scorer's own `metrics.score_utterance` with the same
`canon()` on reference text, entity span_text and hypothesis that `eval_canon.py` applies, condition `none`
(no list): sums over utterances equal the official CER.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_canon import BENCH, canon  # noqa: E402

sys.path.insert(0, str(BENCH / "scorer"))
from metrics import Entity, score_utterance  # noqa: E402
from score import entity_forms  # noqa: E402


class Official:
    def __init__(self):
        self.ann = {}
        for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8"):
            a = json.loads(l)
            if a["split"] == "real":
                ents = [dict(e, span_text=canon(e["span_text"])) for e in a.get("entities") or []]
                self.ann[a["utt_id"]] = (canon(a["text"]), [Entity(forms=entity_forms(e), listed_form=None) for e in ents])
        self._cache = {}

    def err(self, uid: str, hyp: str) -> tuple[int, int]:
        key = (uid, hyp)
        if key not in self._cache:
            ref, gold = self.ann[uid]
            s = score_utterance(ref, canon(hyp or ""), gold, [])
            self._cache[key] = (s.overall.err(), s.overall.ref)
        return self._cache[key]
