#!/usr/bin/env python3
"""CER of one or more RNNT checkpoints on a NeMo manifest -- the in-domain / held-out number.

    PYTHONUTF8=1 python adapter/eval_manifest.py --manifest E:/Claude/datasets/lotusdis_lav/manifest_test.jsonl \\
        --models base=E:/asr-spike/models/pathumma-...nemo lot1=checkpoints/rt_student_lot1.nemo --tag lotus_test

The benchmark number comes from `eval_rt_local.py` + `eval_canon.py`; this one answers "did the tune
learn its own domain". Same normaliser as the freshness gate (`gate_lotusdis.norm`: canon digits ->
words, SARA AM folded, spaces dropped) on both sides; greedy, PCM16-quantised like the wire, batched
(batch 1 is only needed for latency, which this script does not report).
Writes results/<tag>_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gate_lotusdis import lev, norm  # noqa: E402

RES = HERE.parent / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--models", nargs="*", default=[], help="name=path ... (.nemo or HF id)")
    ap.add_argument("--hyps", nargs="*", default=[], help="name=hyps.jsonl ({id, text}) from another runtime")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=0, help="first N rows only (0 = all)")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    if args.n:
        rows = rows[: args.n]
    audio = []
    for r in (rows if args.models else []):
        y, _ = sf.read(r["audio_filepath"], dtype="float32")
        audio.append((np.clip(np.round(y * 32767), -32768, 32767) / 32768).astype(np.float32))
    out = {"manifest": args.manifest, "n": len(rows), "cer": {}}

    def score(name: str, texts: list[str]) -> None:
        errs = chars = 0
        for r, t in zip(rows, texts):
            ref = norm(r["text"])
            errs += lev(ref, norm(t.strip()))
            chars += len(ref)
        out["cer"][name] = round(errs / max(chars, 1), 4)
        print(name, out["cer"][name], flush=True)

    for spec in args.hyps:                          # same normaliser, same rows, other runtime
        name, path = spec.split("=", 1)
        got = {(h := json.loads(l))["id"]: h["text"] for l in open(path, encoding="utf-8") if l.strip()}
        score(name, [got[r["id"]] for r in rows])
    if args.models:
        import torch
        import nemo.collections.asr as nemo_asr
    for spec in args.models:
        name, path = spec.split("=", 1)
        m = (nemo_asr.models.ASRModel.restore_from(path, map_location="cuda") if path.endswith(".nemo")
             else nemo_asr.models.ASRModel.from_pretrained(path, map_location="cuda"))
        m.eval()
        with torch.inference_mode():
            hs = m.transcribe(audio, verbose=False, batch_size=16)
        texts = [(h.text if hasattr(h, "text") else str(h)) for h in hs]
        score(name, texts)
        with (RES / f"{args.tag}_{name}_hyps.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r, t in zip(rows, texts):
                f.write(json.dumps({"id": r["id"], "text": t.strip()}, ensure_ascii=False) + "\n")
        del m
        torch.cuda.empty_cache()
    (RES / f"{args.tag}_metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
