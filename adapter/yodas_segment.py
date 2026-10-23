#!/usr/bin/env python3
"""Cut YODAS2 Thai videos into caption-aligned segments for teacher labelling.

    python adapter/yodas_segment.py --shards 00000000 00000001 00000002

Source: `espnet/yodas2`, `data/th000` (CC-BY-3.0, ungated), downloaded to
E:/Claude/datasets/yodas2_th/. Segment ids encode `<video>-<n>-<start_cs>-<end_cs>`.

Kept: 2-20 s segments (Whisper's window is 30 s; under 2 s is mostly a word), 16 kHz mono.
The caption is stored as `caption` -- it is NOT the label. YODAS captions are uploader text,
including spam ("เว็บตรง ... คลิกลิงค์"); the label comes from the teacher. The caption is kept only
for a weak agreement check and for the benchmark-overlap guard (`teacher_label.py`).
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf

SRC = Path("E:/Claude/datasets/yodas2_th")
OUT = SRC / "segments"
SR = 16000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--min", type=float, default=2.0)
    ap.add_argument("--max", type=float, default=20.0)
    ap.add_argument("--manifest", default="manifest.jsonl", help="file name under yodas2_th/")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man = (SRC / args.manifest).open("a", encoding="utf-8", newline="\n")
    total = 0.0
    for sh in args.shards:
        caps = {}
        for v in json.loads((SRC / f"text_{sh}.json").read_text(encoding="utf-8")):
            caps[v["audio_id"]] = v["text"]
        n = 0
        with tarfile.open(SRC / f"audio_{sh}.tar.gz", "r:gz") as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(".wav"):
                    continue
                vid = Path(m.name).stem
                y, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if sr != SR:
                    import librosa
                    y = librosa.resample(y, orig_sr=sr, target_sr=SR)
                for sid, cap in caps.get(vid, {}).items():
                    p = sid.split("-")
                    s, e = int(p[-2]) / 100, int(p[-1]) / 100
                    d = e - s
                    if not (args.min <= d <= args.max):
                        continue
                    seg = y[int(s * SR): int(e * SR)]
                    if len(seg) < args.min * SR or np.abs(seg).max() < 1e-3:
                        continue
                    wav = OUT / f"{sid}.wav"
                    sf.write(wav, seg, SR, subtype="PCM_16")
                    man.write(json.dumps({"id": sid, "video": vid, "shard": sh, "wav": str(wav),
                                          "dur": round(d, 2), "caption": cap},
                                         ensure_ascii=False) + "\n")
                    n += 1
                    total += d
        print(f"shard {sh}: {n} segments, running total {total/3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
