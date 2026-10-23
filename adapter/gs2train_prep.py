#!/usr/bin/env python3
"""GigaSpeech2-th TRAIN shards -> NECTEC fine-tuning manifests, keeping the clips NECTEC gets WRONG.

    PYTHONUTF8=1 python adapter/gs2train_prep.py --shard 0            # NECTEC pass + manifest
    PYTHONUTF8=1 python adapter/gs2train_prep.py --shard 0 --easy-frac 0.15

WHY (2026-10-10 loop, iteration 1). `F45`: 18 h of fresh, convention-matched human GS2 (dev+test) moved NECTEC
-6.8% on the benchmark dev half. The train split is ~10k h (193 x ~51 h shards) with machine-refined labels.
Freshness gate on 300 shard-0 clips: NECTEC .0333, Pathumma-L .0344, typhoon-L .0397 -- typhoon is KNOWN to train
on this split and is not lower, so the low CER is the labels' filtering (segments ASR already agrees on), not
memorisation. Most clips therefore teach nothing (`F28`); the clips where NECTEC errs carry the signal. So:

  * run NECTEC (greedy, eval) over every labelled clip in the shard, PCM16 like the wire
  * labels in NECTEC's convention, exactly `gs2_convention.convert` (unspaced, SARA AM composed, reduplication)
  * DROP clips where NECTEC writes Latin (GS2 has none; same rule as `F45`), and the 20-char benchmark leak guard
  * keep every clip with NECTEC CER > 0, plus `--easy-frac` of the perfect ones (so the model still sees
    what it already does right), durations 1-20 s
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gs2_convention import BENCH, LATIN, convert  # noqa: E402

D = Path("E:/Claude/datasets/gs2_train")
NEC = "E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo"


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--easy-frac", type=float, default=0.15)
    args = ap.parse_args()
    if not sys.flags.utf8_mode:
        raise SystemExit("run with PYTHONUTF8=1")
    import soundfile as sf
    pre = f"{args.shard}-"
    lab = {}
    for l in open(D / "data/th/train_refined.tsv", encoding="utf-8"):
        if l.startswith(pre):
            k, t = l.rstrip("\n").split("\t", 1)
            lab[k] = convert(t)
    wavs = {p.stem: p for p in (D / "audio" / str(args.shard)).rglob("*.wav")}
    keys = sorted(k for k in wavs if k in lab)
    grams = set()
    for l in BENCH.open(encoding="utf-8"):
        t = re.sub(r"\s", "", json.loads(l)["text"])
        grams.update(t[i:i + 20] for i in range(len(t) - 19))
    rows, drop = [], {"leak": 0, "dur": 0}
    for k in keys:
        t = lab[k]
        if not t or any(t[i:i + 20] in grams for i in range(len(t) - 19)):
            drop["leak"] += 1; continue
        d = sf.info(str(wavs[k])).duration
        if not 1.0 <= d <= 20.0:
            drop["dur"] += 1; continue
        rows.append({"audio_filepath": str(wavs[k]).replace("\\", "/"), "duration": round(d, 2), "text": t, "id": f"tr{k}"})
    print(f"shard {args.shard}: labelled {len(keys)}, kept for NECTEC pass {len(rows)}, dropped {drop}", flush=True)

    hyp_path = D / f"nectec_shard{args.shard}.jsonl"
    done = {}
    if hyp_path.exists():
        done = {json.loads(l)["id"]: json.loads(l)["hyp"] for l in hyp_path.open(encoding="utf-8")}
    todo = [r for r in rows if r["id"] not in done]
    if todo:
        import nemo.collections.asr as nemo_asr
        m = nemo_asr.models.ASRModel.restore_from(NEC, map_location="cuda")
        m.eval()
        with hyp_path.open("a", encoding="utf-8", newline="\n") as f:
            for i in range(0, len(todo), 256):
                chunk = todo[i: i + 256]
                import torch
                with torch.inference_mode():
                    out = m.transcribe([r["audio_filepath"] for r in chunk], batch_size=32, verbose=False)
                for r, h in zip(chunk, out):
                    t = (h.text if hasattr(h, "text") else str(h)).strip()
                    done[r["id"]] = t
                    f.write(json.dumps({"id": r["id"], "hyp": t}, ensure_ascii=False) + "\n")
                if i % 5120 == 0:
                    print(f"  NECTEC {i}/{len(todo)}", flush=True)
    hard, easy, latin = [], [], 0
    E = C = 0
    for r in rows:
        h = done[r["id"]]
        if LATIN.search(h):
            latin += 1; continue
        e = lev(r["text"], h.replace(" ", "").replace("\u0e4d\u0e32", "\u0e33"))
        E += e; C += len(r["text"])
        (hard if e > 0 else easy).append(r)
    random.Random(args.shard).shuffle(easy)
    keep = hard + easy[: int(len(easy) * args.easy_frac)]
    keep.sort(key=lambda r: r["id"])
    out = D / f"manifest_shard{args.shard}_hard.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {"shard": args.shard, "rows": len(rows), "latin_dropped": latin, "nectec_cer": round(E / C, 4),
            "hard": len(hard), "easy": len(easy), "kept": len(keep),
            "kept_hours": round(sum(r["duration"] for r in keep) / 3600, 1), "easy_frac": args.easy_frac}
    (HERE.parent / "results" / f"gs2train_shard{args.shard}_prep.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(summ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
