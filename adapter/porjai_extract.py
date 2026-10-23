#!/usr/bin/env python3
"""Extract a Porjai Thai-central subset to wav + manifests, for the trained N-best picker.

    python adapter/porjai_extract.py --shards 00000 00001 --train 16000 --heldout 1000

Source: `CMKL/Porjai-Thai-voice-dataset-central` (Suwanbandit et al., Interspeech 2023), ~700 h of
human-transcribed READ Thai, CC-BY-SA 4.0, ungated; downloaded parquet shards in
E:/Claude/datasets/porjai/. Transcripts are word-segmented with spaces and write numbers as words
(0 ASCII digits in 20,980 rows of shard 0) -- the scorer ignores spaces, so they are kept as is.

Why a picker needs this: its training label is "which candidate region is closest to the HUMAN
transcript", so it needs human-labelled audio that is not the benchmark (CC-BY-NC, evaluation only).

Splits are by SENTENCE, not by clip: Porjai has several speakers reading the same text (15,036
distinct sentences in 20,980 rows), so a clip-level split would put the same sentence on both sides.
A 20-char leak guard against all 2,062 benchmark references runs first (the YODAS run found 27 hits).
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

SRC = Path("E:/Claude/datasets/porjai")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--train", type=int, default=16000)
    ap.add_argument("--heldout", type=int, default=1000)
    ap.add_argument("--leak-chars", type=int, default=20)
    args = ap.parse_args()

    refs = ["".join(toks(canon(json.loads(l)["text"]))) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")]
    K = args.leak_chars
    grams = {r[i: i + K] for r in refs for i in range(len(r) - K + 1)}

    rows = []
    for sh in args.shards:
        t = pq.read_table(SRC / f"train-{sh}-of-00016.parquet")
        rows.extend(t.to_pylist())
    leak = 0
    by_sent = {}
    for r in rows:
        flat = "".join(toks(canon(r["sentence"])))
        if any(flat[i: i + K] in grams for i in range(len(flat) - K + 1)):
            leak += 1
            continue
        by_sent.setdefault(r["sentence"], []).append(r)
    sents = sorted(by_sent)
    random.Random(2026).shuffle(sents)

    def take(sent_list, n):
        out = []
        for s in sent_list:
            for r in by_sent[s]:
                if len(out) < n:
                    out.append(r)
            if len(out) >= n:
                break
        return out

    # held-out sentences first, then train from the remaining sentences only
    held, used = [], 0
    for s in sents:
        held.extend(by_sent[s]); used += 1
        if len(held) >= args.heldout:
            break
    held = held[: args.heldout]
    train = take(sents[used:], args.train)

    out_dir = SRC / "wav"
    out_dir.mkdir(exist_ok=True)
    stats = {"rows": len(rows), "leak_dropped": leak, "distinct_sentences": len(sents)}
    for name, part in (("heldout", held), ("train", train)):
        tot = 0.0
        with (SRC / f"manifest_{name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in part:
                y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if sr != 16000:
                    import librosa
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                wav = out_dir / f"{r['utterance']}.wav"
                sf.write(wav, np.clip(y, -1, 1), 16000, subtype="PCM_16")
                d = len(y) / 16000
                tot += d
                f.write(json.dumps({"utt_id": r["utterance"], "wav": str(wav), "dur": round(d, 2),
                                    "text": r["sentence"]}, ensure_ascii=False) + "\n")
        stats[name] = {"utts": len(part), "hours": round(tot / 3600, 2)}
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
