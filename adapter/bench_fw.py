#!/usr/bin/env python3
"""Per-turn latency + hypotheses of a Whisper model under CTranslate2 (faster-whisper).

    E:/fw-spike/.venv/Scripts/python.exe adapter/bench_fw.py --model E:/fw-spike/models/typhoon-whisper-turbo-ct2 --tag fw_turbo_fp16

WHY. `bench_student.py` measured typhoon-whisper-turbo at 1,229 ms p50 per turn under HF
`generate` -- too slow for the loop. That number is the runtime, not the model: HF generate is a
Python loop with no fused kernels. CTranslate2 is the usual production runtime for Whisper. If it
brings turbo to a few hundred ms, turbo (CER 0.0789, 7% from the bar) is a far more closable
student than the 115M RNNT (0.0945, 22% away, and two distillation rounds moved it 0.5%).

Runs in its OWN venv (E:/fw-spike/.venv) -- CLAUDE.md forbids adding packages to E:/asr-spike.
Same protocol as `bench_student.py`: batch 1, greedy, language th, one finished turn per call,
first 5 clips warm-up, PCM16-quantised audio. No VAD, no timestamps, no conditioning on prior text.
"""

from __future__ import annotations

import argparse
import json
import os
import site
import statistics
import time
from pathlib import Path

for sp in site.getsitepackages():                       # CUDA 12 DLLs shipped as pip wheels
    for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
        d = Path(sp) / sub
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

BENCH = Path("E:/asr-bench/thai-contextasr-bench")
RES = Path(__file__).resolve().parent.parent / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--compute", default="float16")
    args = ap.parse_args()
    m = WhisperModel(args.model, device="cuda", compute_type=args.compute)
    rows = sorted((json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")),
                  key=lambda r: r["utt_id"])
    rows = [r for r in rows if r["split"] == "real"]
    lat, aud, out = [], 0.0, []
    for i, r in enumerate(rows):
        y, sr = sf.read(BENCH / r["wav"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = (np.clip(np.round(y * 32767), -32768, 32767) / 32768).astype(np.float32)
        t0 = time.perf_counter()
        segs, _ = m.transcribe(y, language="th", task="transcribe", beam_size=1, best_of=1,
                               temperature=0.0, vad_filter=False, without_timestamps=True,
                               condition_on_previous_text=False)
        text = "".join(s.text for s in segs).strip()          # generator: decoding happens here
        dt = time.perf_counter() - t0
        out.append({"utt_id": r["utt_id"], "condition": "none", "text": text})
        if i >= 5:
            lat.append(dt); aud += len(y) / sr
    with (RES / f"{args.tag}_hyps.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    lat.sort()
    run = {"model": args.model, "runtime": "ctranslate2/faster-whisper", "compute": args.compute,
           "n": len(rows), "latency_ms_p50": round(1000 * statistics.median(lat), 1),
           "latency_ms_p90": round(1000 * lat[int(0.9 * len(lat))], 1), "rtf": round(sum(lat) / aud, 4)}
    (RES / f"{args.tag}_run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(json.dumps(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
