#!/usr/bin/env python3
"""Teacher pass 1: run ONE pool system over the YODAS segments, batched, resumable.

    python adapter/teacher_label.py --system tw          # one of: tw pat bio isan q17 q06 qctx rt
    python adapter/teacher_label.py --system tw --limit 64   # smoke

The teacher is the no-selection all-8 ROVER (+ GER) that beat typhoon-whisper-large-v3 on the
benchmark (`idea-stage/IDEA_REPORT.md`, I1/I2). This script produces each member's hypotheses on
unlabelled real Thai audio; `teacher_combine.py` votes them. One system per process so a crash or
an OOM costs one system, never the lot, and memory holds one model at a time (2026-10-03: three
concurrent jobs got the box reaped).

Output: E:/Claude/datasets/yodas2_th/teacher/<system>.jsonl  {"id", "text"} -- resumable: rows
already present are skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch

DATA = Path("E:/Claude/datasets/yodas2_th")
WHISPER = {
    "tw": "typhoon-ai/typhoon-whisper-large-v3",
    "pat": "nectec/Pathumma-whisper-th-large-v3",
    "bio": "biodatlab/whisper-th-large-v3-combined",
    "isan": "typhoon-ai/typhoon-isan-asr-whisper",
}
QWEN = {
    "q17": "Qwen/Qwen3-ASR-1.7B-hf",
    "q06": "Qwen/Qwen3-ASR-0.6B-hf",
    "qctx": "typhoon-ai/typhoon-asr-qwen-0.6b-ctx",
}
NEMO = {"rt": "typhoon-ai/typhoon-asr-realtime"}


def batches(rows, bs):
    for i in range(0, len(rows), bs):
        yield rows[i: i + bs]


def load_audio(r):
    y, _ = sf.read(r["wav"], dtype="float32")
    return y if y.ndim == 1 else y.mean(axis=1)


def run_whisper(name, rows, write, bs):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    proc = AutoProcessor.from_pretrained(name)
    m = AutoModelForSpeechSeq2Seq.from_pretrained(name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda").eval()
    for b in batches(rows, bs):
        feats = proc([load_audio(r) for r in b], sampling_rate=16000, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = m.generate(feats.to("cuda", torch.float16), language="th", task="transcribe",
                             num_beams=1, max_new_tokens=220)
        for r, t in zip(b, proc.batch_decode(ids, skip_special_tokens=True)):
            write(r["id"], t.strip())


def run_qwen(name, rows, write, bs):
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    proc = AutoProcessor.from_pretrained(name)
    m = AutoModelForMultimodalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    try:
        proc.tokenizer.padding_side = "left"
    except AttributeError:
        pass
    for b in batches(rows, bs):
        msgs = [[{"role": "system", "content": [{"type": "text", "text": ""}]},
                 {"role": "user", "content": [{"type": "audio", "audio": load_audio(r)}]}] for r in b]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, padding=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
        inp = {k: (v.to(torch.bfloat16) if hasattr(v, "is_floating_point") and v.is_floating_point() else v)
               for k, v in inp.items()}
        with torch.inference_mode():
            g = m.generate(**inp, max_new_tokens=220)
        g = g[:, inp["input_ids"].shape[1]:]
        for r, t in zip(b, proc.decode(g, return_format="transcription_only")):
            write(r["id"], t.strip())


def run_nemo(name, rows, write, bs):
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained(name, map_location="cuda").eval()
    for b in batches(rows, bs):
        out = m.transcribe([load_audio(r) for r in b], verbose=False, batch_size=bs)
        for r, h in zip(b, out):
            write(r["id"], (h.text if hasattr(h, "text") else str(h)).strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=[*WHISPER, *QWEN, *NEMO])
    ap.add_argument("--manifest", default=str(DATA / "manifest.jsonl"))
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    rows.sort(key=lambda r: r["dur"])                     # length-sorted: less padding per batch
    if args.limit:
        rows = rows[: args.limit]
    out = DATA / "teacher" / f"{args.system}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.open(encoding="utf-8") if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{args.system}: {len(done)} done, {len(todo)} to go", flush=True)
    f = out.open("a", encoding="utf-8", newline="\n")
    n, t0, aud = [0], time.time(), sum(r["dur"] for r in todo)

    def write(i, t):
        f.write(json.dumps({"id": i, "text": t}, ensure_ascii=False) + "\n")
        n[0] += 1
        if n[0] % 500 == 0:
            f.flush()
            print(f"  {n[0]}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)

    if args.system in WHISPER:
        run_whisper(WHISPER[args.system], todo, write, args.bs)
    elif args.system in QWEN:
        run_qwen(QWEN[args.system], todo, write, args.bs)
    else:
        run_nemo(NEMO[args.system], todo, write, args.bs)
    f.close()
    el = time.time() - t0
    print(f"{args.system}: {n[0]} segs, {aud/3600:.2f} h audio, {el:.0f}s, RTF {el/max(aud,1):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
