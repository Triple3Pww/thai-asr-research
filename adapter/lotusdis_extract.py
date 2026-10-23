#!/usr/bin/env python3
"""LOTUSDIS (nectec/LOTUSDIS, CC-BY-SA-4.0) -> 16 kHz PCM16 wavs + NeMo manifests, near-mic only.

    PYTHONUTF8=1 python adapter/lotusdis_extract.py                    # train + val + test slices

WHY THIS CORPUS (2026-10-06, the "DeepSeek for Thai ASR" run). `F28`: training on data the model
already knows teaches nothing. LOTUSDIS passed the freshness gate (`results/gate_lotusdis.json`):
NECTEC reads its lavalier test clips at CER 0.205 and train at 0.124 -- seen in part, not memorised.
It is also the one large *conversational* Thai set on disk, and conversation is where NECTEC is
weakest (talk 0.1379 vs the picker's 0.1287, `F29`).

Choices, each with its why:
  * mic `lav123` only. Every utterance exists 5 times (lav, con, bt3m, bt10m, jbl); the benchmark is
    near-mic, and 5 copies of one sentence would be 5x the steps for ~1x the text.
  * validation = held-out SPEAKERS of the train split (a clip split would leak voices), 600 clips.
  * test = a seeded 1,000-clip sample of the official test split, the in-domain number.
  * 20-char leak guard against all 2,062 benchmark references, as for YODAS and Porjai.
  * labels kept as LOTUSDIS writes them -- digits are 2 of 24,049 rows, and NECTEC's own output is
    composed SARA AM, same as these labels; nothing to convert.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_canon import canon  # noqa: E402
from pilot_rover import BENCH, toks  # noqa: E402

SRC = Path("E:/Claude/datasets/LOTUSDIS/data")
OUT = Path("E:/Claude/datasets/lotusdis_lav")


def rows_of(split: str) -> list[dict]:
    rows = []
    for p in sorted(SRC.glob(f"{split}-*.parquet")):
        if not pq.read_table(p, columns=["mic"]).to_pandas().mic.eq("lav123").any():
            continue
        t = pq.read_table(p).to_pandas()
        rows += t[(t.mic == "lav123") & (t.duration >= 1.0) & (t.duration <= 20.0)].to_dict("records")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=int, default=600)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--leak-chars", type=int, default=20)
    args = ap.parse_args()

    refs = ["".join(toks(canon(json.loads(l)["text"]))) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")]
    K = args.leak_chars
    grams = {r[i: i + K] for r in refs for i in range(len(r) - K + 1)}

    def leaks(s: str) -> bool:
        flat = "".join(toks(canon(s)))
        return any(flat[i: i + K] in grams for i in range(len(flat) - K + 1))

    train = rows_of("train")
    test = rows_of("test")
    leak = sum(leaks(r["sentence"]) for r in train)
    train = [r for r in train if r["sentence"].strip() and not leaks(r["sentence"])]
    spk = sorted({r["speaker_id"] for r in train})
    random.Random(2026).shuffle(spk)
    held, val = set(), []
    for s in spk:                                   # whole speakers until the val quota is met
        held.add(s)
        val = [r for r in train if r["speaker_id"] in held]
        if len(val) >= args.val:
            break
    train = [r for r in train if r["speaker_id"] not in held]
    test = [r for r in test if r["sentence"].strip()]
    random.Random(0).shuffle(test)
    test = test[: args.test]

    stats = {"train_leak_dropped": leak, "val_speakers": sorted(held)}
    for name, part in (("train", train), ("val", val), ("test", test)):
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        tot = 0.0
        with (OUT / f"manifest_{name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for i, r in enumerate(part):
                y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                assert sr == 16000, sr
                wav = d / f"{name}-{i:06d}.wav"
                sf.write(wav, np.clip(y, -1, 1), 16000, subtype="PCM_16")
                dur = len(y) / 16000
                tot += dur
                # id ends in "-a-b-c" so finetune_rt.split_by_video's rsplit("-", 3) groups by speaker
                f.write(json.dumps({"audio_filepath": str(wav).replace("\\", "/"), "duration": round(dur, 3),
                                    "text": r["sentence"].strip(), "id": f"{r['speaker_id']}-x-x-{i}"},
                                   ensure_ascii=False) + "\n")
        stats[name] = {"clips": len(part), "hours": round(tot / 3600, 2)}
        print(name, stats[name], flush=True)
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
