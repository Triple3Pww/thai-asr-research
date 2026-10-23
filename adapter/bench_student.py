#!/usr/bin/env python3
"""Accuracy AND per-turn latency of a candidate real-time student, on the 682 real utterances.

    python adapter/bench_student.py --model typhoon-ai/typhoon-whisper-turbo
    python adapter/bench_student.py --model typhoon-ai/typhoon-whisper-large-v3 --tag bar_tw_latency

WHY. The goal changed on 2026-10-03 to *"a model that beat typhoon best score that can be use in
realtime"*. The 8-model ROVER + LLM teacher beats the bar but is not one model and not real-time.
Before distilling it into a student, measure what each candidate student costs per turn on THIS
GPU and where it starts on accuracy -- the student has to close the gap from there.

Latency is measured the way the voice loop uses ASR: batch 1, one finished turn at a time,
fp16 on CUDA, `torch.cuda.synchronize()` around each call, first 5 clips discarded as warm-up
(CLAUDE.md: discard the first sweep after any start). Reported: p50 / p90 ms per turn and RTF.
Hypotheses go to `results/<tag>_hyps.jsonl` in the benchmark's format; score with `eval_canon.py`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

BENCH = Path("E:/asr-bench/thai-contextasr-bench")
RES = Path(__file__).resolve().parent.parent / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--beams", type=int, default=1)
    args = ap.parse_args()
    tag = args.tag or "student_" + args.model.split("/")[-1].replace("-", "_")

    rows = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")]
    rows = sorted((r for r in rows if r["split"] == "real"), key=lambda r: r["utt_id"])
    if args.limit:
        rows = rows[: args.limit]

    dev = "cuda"
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, torch_dtype=torch.float16).to(dev).eval()
    nparams = sum(p.numel() for p in model.parameters()) / 1e9
    gen = dict(language="th", task="transcribe", num_beams=args.beams, max_new_tokens=440)

    lat, audio_s, out = [], 0.0, []
    for i, r in enumerate(rows):
        y, sr = sf.read(BENCH / r["wav"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        feats = proc(y, sampling_rate=sr, return_tensors="pt").input_features.to(dev, torch.float16)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(feats, **gen)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        out.append({"utt_id": r["utt_id"], "condition": "none", "text": text})
        if i >= 5:
            lat.append(dt); audio_s += len(y) / sr
        if i % 100 == 0:
            print(f"  {i}/{len(rows)}  last {dt*1000:.0f} ms", flush=True)

    with (RES / f"{tag}_hyps.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    lat.sort()
    run = {"model": args.model, "params_B": round(nparams, 3), "n": len(rows), "beams": args.beams,
           "device": torch.cuda.get_device_name(0), "dtype": "fp16", "batch": 1,
           "latency_ms_p50": round(1000 * statistics.median(lat), 1),
           "latency_ms_p90": round(1000 * lat[int(0.9 * len(lat))], 1),
           "rtf": round(sum(lat) / audio_s, 4), "mean_clip_s": round(audio_s / len(lat), 2)}
    (RES / f"{tag}_run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(json.dumps(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
