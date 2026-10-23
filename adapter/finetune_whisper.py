#!/usr/bin/env python3
"""Fine-tune typhoon-whisper-turbo's DECODER on teacher labels -> the ~340 ms real-time student.

    python adapter/finetune_whisper.py --train E:/Claude/datasets/yodas2_th/train_rover8.jsonl --tag w1

Why turbo (2026-10-03): under CTranslate2 it runs 339 ms p50 per turn and sits at CER 0.0803,
8.5% from the bar (typhoon-whisper-large-v3, 0.0735). The 115M RNNT is 22% away and three
distillation rounds moved it 0.0945 -> 0.0940 / 0.0943 / 0.0978. He chose "try both", turbo first.

Why decoder-only: turbo's 32-layer encoder is the acoustic model Typhoon trained on ~11k h; 10.9 h
of labels cannot improve it and could only erase it. The 4-layer decoder writes the text -- where
the teacher's wins live (spelling, entity forms, the words several systems agree on). It also fits
16 GB: 809M frozen + ~170M trainable.

Why these labels: `train_rover8.jsonl`, the all-8 ROVER with confidence over all 8. Turbo is NOT
one of the 8, so "the 8 agree" does not mean "the student already agrees" -- round 1's defect does
not apply. Validation = held-out VIDEOS of the teacher set, never the benchmark.
Writes an HF checkpoint to checkpoints/whisper_student_<tag>/ (convert with ct2 for latency).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parent.parent
BASE = "typhoon-ai/typhoon-whisper-turbo"


def load_rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def video(r):
    return r["id"].rsplit("-", 3)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.03)
    args = ap.parse_args()
    torch.manual_seed(2026)

    rows = load_rows(args.train)
    vids = sorted({video(r) for r in rows})
    random.Random(2026).shuffle(vids)
    held = set(vids[: max(1, int(len(vids) * args.val_frac))])
    tr = [r for r in rows if video(r) not in held]
    va = [r for r in rows if video(r) in held]
    print(f"train {len(tr)}  val {len(va)}  (held-out videos {len(held)})", flush=True)

    proc = WhisperProcessor.from_pretrained(BASE)
    model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float32).cuda()
    model.config.forced_decoder_ids = None
    for p in model.model.encoder.parameters():
        p.requires_grad = False
    model.model.encoder.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable {sum(p.numel() for p in trainable)/1e6:.0f}M of "
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M", flush=True)
    # same prefix the model sees at inference: <|startoftranscript|><|th|><|transcribe|><|notimestamps|>
    proc.tokenizer.set_prefix_tokens(language="th", task="transcribe", predict_timestamps=False)

    def batchify(b):
        auds = []
        for r in b:
            y, _ = sf.read(r["audio_filepath"], dtype="float32")
            auds.append(y if y.ndim == 1 else y.mean(axis=1))
        feats = proc.feature_extractor(auds, sampling_rate=16000, return_tensors="pt").input_features
        lab = proc.tokenizer([r["text"] for r in b], return_tensors="pt", padding=True)
        ids = lab.input_ids.masked_fill(lab.attention_mask == 0, -100)
        # decoder input = labels shifted right; HF does that when `labels` is passed. The first
        # token (<|startoftranscript|>) is the decoder start token, so drop it from the labels.
        if (ids[:, 0] == model.config.decoder_start_token_id).all():
            ids = ids[:, 1:]
        return feats.cuda(), ids.cuda()

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(tr) / args.bs / args.accum) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * max(0.1, 1 - s / max(steps, 1)))

    def val_loss():
        model.eval(); tot = n = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(va), args.bs):
                f, ids = batchify(va[i: i + args.bs])
                tot += model(input_features=f, labels=ids).loss.item() * len(ids); n += len(ids)
        model.train(); model.model.encoder.eval()
        return tot / max(n, 1)

    print(f"val loss before {val_loss():.4f}", flush=True)
    model.train(); model.model.encoder.eval()
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        random.Random(ep).shuffle(tr)
        for i in range(0, len(tr), args.bs):
            f, ids = batchify(tr[i: i + args.bs])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_features=f, labels=ids).loss / args.accum
            loss.backward()
            if (i // args.bs + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 50 == 0:
                    print(f"  ep {ep} step {step}/{steps} loss {loss.item()*args.accum:.4f} "
                          f"{time.time()-t0:.0f}s", flush=True)
        print(f"epoch {ep}: val loss {val_loss():.4f}", flush=True)

    out = ROOT / "checkpoints" / f"whisper_student_{args.tag}"
    model.save_pretrained(out); proc.save_pretrained(out)
    (ROOT / "results" / f"finetune_whisper_{args.tag}_run.json").write_text(json.dumps(
        {"base": BASE, "train": args.train, "n_train": len(tr), "n_val": len(va), "lr": args.lr,
         "epochs": args.epochs, "bs": args.bs, "accum": args.accum, "steps": step,
         "frozen": "encoder", "checkpoint": str(out)}, indent=1), encoding="utf-8")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
