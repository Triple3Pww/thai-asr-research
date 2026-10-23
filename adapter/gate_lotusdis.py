#!/usr/bin/env python3
"""Freshness gate (CLAUDE.md): has NECTEC's FastConformer already seen LOTUSDIS?

    PYTHONUTF8=1 python adapter/gate_lotusdis.py --n 300

LOTUSDIS (nectec/LOTUSDIS, CC-BY-SA-4.0) is NECTEC's own far-field meeting corpus, and their
FastConformer was released a year after it, so it may have been trained on it. The test: score
NECTEC and typhoon-asr-realtime on the SAME seeded sample of lavalier (near-mic) clips from the
train split and the test split. On the benchmark NECTEC/rt CER = 0.0676/0.0945 = 0.72; on Porjai,
which NECTEC memorised, NECTEC read 0.015. A ratio far below 0.72, or a train<<test gap, means seen.
Greedy, batch 1, PCM16 like the wire. Writes results/gate_lotusdis.json.
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _thai import for_scoring  # noqa: E402
from eval_canon import canon  # noqa: E402

D = Path("E:/Claude/datasets/LOTUSDIS/data")
RES = Path(__file__).resolve().parent.parent / "results"
MODELS = {"nectec": "E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo",
          "rt": "typhoon-ai/typhoon-asr-realtime"}


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(s: str) -> str:
    return for_scoring(canon(s)).replace(" ", "")


def sample(split: str, n: int, seed: int) -> list[dict]:
    rows = []
    for p in sorted(D.glob(f"{split}-*.parquet")):     # shards are grouped by mic: scan for lav ones
        if not pq.read_table(p, columns=["mic"]).to_pandas().mic.str.startswith("lav").any():
            continue
        t = pq.read_table(p, columns=["audio", "sentence", "mic", "duration"]).to_pandas()
        t = t[(t.mic.str.startswith("lav")) & (t.duration >= 1.0)]
        rows += t.to_dict("records")
        if len(rows) >= 4 * n:
            break
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    sets = {s: sample(s, args.n, 0) for s in ("train", "test")}
    audio = {}
    for s, rows in sets.items():
        audio[s] = []
        for r in rows:
            y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            audio[s].append((np.clip(np.round(y * 32767), -32768, 32767) / 32768).astype(np.float32))
    out = {"n": args.n, "mic": "lav*", "results": {}, "examples": []}
    for name, path in MODELS.items():
        m = (nemo_asr.models.ASRModel.restore_from(path, map_location="cuda") if path.endswith(".nemo")
             else nemo_asr.models.ASRModel.from_pretrained(path, map_location="cuda"))
        m.eval()
        for s, rows in sets.items():
            errs = chars = 0
            for k, (r, y) in enumerate(zip(rows, audio[s])):
                with torch.inference_mode():
                    h = m.transcribe([y], verbose=False, batch_size=1)[0]
                hyp = (h.text if hasattr(h, "text") else str(h)).strip()
                ref = norm(r["sentence"])
                errs += lev(ref, norm(hyp)); chars += len(ref)
                if k < 5 and s == "test":
                    out["examples"].append({"model": name, "ref": r["sentence"], "hyp": hyp})
            out["results"][f"{name}_{s}"] = round(errs / max(chars, 1), 4)
            print(name, s, out["results"][f"{name}_{s}"], flush=True)
        del m
        torch.cuda.empty_cache()
    r = out["results"]
    out["ratio_test"] = round(r["nectec_test"] / r["rt_test"], 3)
    out["ratio_train"] = round(r["nectec_train"] / r["rt_train"], 3)
    (RES / "gate_lotusdis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
