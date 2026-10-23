#!/usr/bin/env python3
"""GigaSpeech2-th dev+test -> NECTEC fine-tuning manifests in NECTEC's OWN writing convention (`F32` rule).

    PYTHONUTF8=1 python adapter/gs2_convention.py

WHY (2026-10-09, toward CER 0.060). GigaSpeech2-th's human-refined dev+test (20 h) is fresh to NECTEC (CER
.122/.148 on 300+300 clips, `F44`). `F32`: a corpus teaches its TYPIST -- so the labels are rewritten toward what
NECTEC already writes, read from NECTEC's own output on these clips (judge_gs2/ears.json), never from the
benchmark references:

  | per 1k chars      | GS2 refs | NECTEC on GS2 | rule                                                     |
  |-------------------|----------|---------------|----------------------------------------------------------|
  | Latin letters     | 0        | 3.65          | DROP clips of turns where NECTEC writes Latin: GS2 spells |
  |                   |          |               | loanwords in Thai, NECTEC in Latin; no safe mapping       |
  | ๆ                 | 0        | 0.14          | whitelisted reduplication XX -> Xๆ (`lotusdis_convention`)|
  | SARA AM           | decomposed | composed    | compose (NECTEC writes ำ)                                 |
  | digits / spaces   | 0        | small         | none (GS2 already writes number words, unspaced)          |

Leak guard: any clip sharing a 20-character run with any of the 2,062 benchmark references is dropped (same
guard as every other training set here). Validation = 5% of VIDEOS, held out whole.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lotusdis_convention import REDUP_RE  # noqa: E402

GS2 = Path("E:/Claude/datasets/gigaspeech2_th")
J = Path("E:/Claude/datasets/judge_gs2")
BENCH = Path("E:/asr-bench/thai-contextasr-bench/data/utterances.jsonl")
LATIN = re.compile(r"[A-Za-z]")


def convert(t: str) -> str:
    t = t.replace(" ", "").replace("\u0e4d\u0e32", "\u0e33")          # compose SARA AM, as NECTEC writes
    return REDUP_RE.sub(lambda m: m.group(1) + "ๆ", t)


def main() -> int:
    ears = json.loads((J / "ears.json").read_text(encoding="utf-8"))["nec"]
    latin_seg = set()                                                   # (split, video, seg) to drop
    for tid, h in ears.items():
        if LATIN.search(h):
            split_v, a, b = tid.rsplit("-", 2)
            split = "dev" if split_v.startswith("devv") else "test"      # ids look like "devv12-3-7"
            video = split_v[len(split) + 1:]
            for s in range(int(a), int(b) + 1):
                latin_seg.add((split, video, s))
    grams = set()
    for l in BENCH.open(encoding="utf-8"):
        t = re.sub(r"\s", "", json.loads(l)["text"])
        grams.update(t[i:i + 20] for i in range(len(t) - 19))
    out = {"train": [], "val": []}
    drop = {"latin": 0, "leak": 0}
    for split in ("dev", "test"):
        for l in open(GS2 / f"manifest_{split}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            video, seg = r["id"].split("-")[0], int(r["id"].split("-")[1])
            if (split, video, seg) in latin_seg:
                drop["latin"] += 1; continue
            text = convert(r["text"])
            if any(text[i:i + 20] in grams for i in range(len(text) - 19)):
                drop["leak"] += 1; continue
            fold = "val" if int(hashlib.md5(f"{split}{video}".encode()).hexdigest(), 16) % 20 == 0 else "train"
            out[fold].append({"audio_filepath": r["audio_filepath"], "duration": r["duration"], "text": text,
                              "id": f"{split}-{r['id']}"})
    for fold, rows in out.items():
        with open(GS2 / f"manifest_{fold}_conv.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(fold, len(rows), f"{sum(r['duration'] for r in rows) / 3600:.1f} h")
    print("dropped", drop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
