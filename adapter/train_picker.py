#!/usr/bin/env python3
"""Fine-tune a small LLM to be the N-best picker; evaluate on held-out Porjai; optionally predict.

    python adapter/train_picker.py --train E:/Claude/datasets/porjai/picker_train.jsonl \
        --heldout E:/Claude/datasets/porjai/picker_heldout.jsonl --tag p1
    python adapter/train_picker.py --predict results/picker_bench_prompts.jsonl --ckpt checkpoints/picker_p1 --tag p1

Model: `Qwen/Qwen3-0.6B`, full fine-tune, bf16. Input = the rendered regions (`picker_lib.render`),
then "\\nคำตอบ: ", target = comma-separated option numbers, then EOS. Loss on the target only. At
inference the model writes a handful of digits -- never free text -- and anything unparsable falls back
to option 0 (the vote), so the output is always a sentence the ASRs actually proposed.

Held-out evaluation (sentences never seen in training) reports CER of: turbo top-1, the vote
(always 0), the picker, and the oracle -- all assembled the same way, canon normaliser.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from eval_canon import canon  # noqa: E402
from picker_lib import assemble  # noqa: E402
from pilot_rover import edit, toks  # noqa: E402

SEP = "\nคำตอบ: "


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def parts_of(r):
    return [(k, x) for k, x in r["parts"]]


def parse(text, n):
    out = []
    for t in text.split(",")[:n]:
        try:
            out.append(int(t.strip()))
        except ValueError:
            out.append(0)
    return out + [0] * (n - len(out))


@torch.inference_mode()
def predict(model, tok, rows, bs=16):
    model.eval()
    tok.padding_side = "left"
    preds, lat = {}, []
    todo = [r for r in rows if r.get("prompt")]
    for i in range(0, len(todo), bs):
        b = todo[i: i + bs]
        enc = tok([r["prompt"] + SEP for r in b], return_tensors="pt", padding=True).to("cuda")
        t0 = time.perf_counter()
        g = model.generate(**enc, max_new_tokens=4 * max(r["n_regions"] for r in b) + 2, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        lat.append((time.perf_counter() - t0) / len(b))
        for r, ids in zip(b, g[:, enc["input_ids"].shape[1]:]):
            preds[r["utt_id"]] = parse(tok.decode(ids, skip_special_tokens=True), r["n_regions"])
    return preds, lat


@torch.inference_mode()
def latency_bs1(model, tok, rows, n=100):
    """What one voice-loop turn pays: batch 1, greedy, warm-up excluded."""
    tok.padding_side = "left"
    ts = []
    for r in [r for r in rows if r.get("prompt")][: n + 5]:
        enc = tok([r["prompt"] + SEP], return_tensors="pt").to("cuda")
        torch.cuda.synchronize(); t0 = time.perf_counter()
        model.generate(**enc, max_new_tokens=4 * r["n_regions"] + 2, do_sample=False,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    ts = sorted(ts[5:])
    return round(1000 * ts[len(ts) // 2], 1), round(1000 * ts[int(0.9 * len(ts))], 1)


def score_heldout(rows, preds, refs):
    acc = {"vote": 0, "picker": 0, "oracle": 0}
    tot = 0
    for r in rows:
        ref = toks(canon(refs[r["utt_id"]]))
        tot += len(ref)
        p, regs = parts_of(r), r["regs"]
        n = len(regs)
        acc["vote"] += edit(ref, toks(canon(assemble(p, regs, [0] * n))))
        acc["picker"] += edit(ref, toks(canon(assemble(p, regs, preds.get(r["utt_id"], [0] * n)))))
        acc["oracle"] += edit(ref, toks(canon(assemble(p, regs, [int(x) for x in r["target"].split(",")] if r.get("target") else [0] * n))))
    return {k: round(v / tot, 4) for k, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train")
    ap.add_argument("--heldout")
    ap.add_argument("--heldout-manifest", default="E:/Claude/datasets/porjai/manifest_heldout.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--predict", default=None)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    torch.manual_seed(2026)

    src = args.ckpt or args.base
    tok = AutoTokenizer.from_pretrained(src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).cuda()

    if args.predict:
        rows = load(args.predict)
        preds, _ = predict(model, tok, rows)
        p50, p90 = latency_bs1(model, tok, rows)
        out = ROOT / "results" / f"picker_{args.tag}_bench_hyps.jsonl"
        with out.open("w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                n = r["n_regions"]
                text = assemble(parts_of(r), r["regs"], preds.get(r["utt_id"], [0] * n))
                f.write(json.dumps({"utt_id": r["utt_id"], "condition": "none", "text": text}, ensure_ascii=False) + "\n")
        print(json.dumps({"hyps": str(out), "picker_latency_ms_bs1_p50": p50, "p90": p90}))
        return 0

    train = [r for r in load(args.train) if r.get("prompt") and r.get("target")]
    held = [r for r in load(args.heldout) if r.get("prompt") and r.get("target")]
    refs = {json.loads(l)["utt_id"]: json.loads(l)["text"] for l in open(args.heldout_manifest, encoding="utf-8")}
    print(f"train {len(train)}  heldout {len(held)}", flush=True)

    def batch(rs):
        tok.padding_side = "right"
        ids, labels = [], []
        for r in rs:
            p = tok(r["prompt"] + SEP, add_special_tokens=False)["input_ids"]
            t = tok(r["target"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids.append(p + t); labels.append([-100] * len(p) + t)
        L = max(map(len, ids))
        x = torch.full((len(ids), L), tok.pad_token_id); y = torch.full((len(ids), L), -100); m = torch.zeros((len(ids), L), dtype=torch.long)
        for i, (a, b) in enumerate(zip(ids, labels)):
            x[i, :len(a)] = torch.tensor(a); y[i, :len(b)] = torch.tensor(b); m[i, :len(a)] = 1
        return x.cuda(), y.cuda(), m.cuda()

    base_preds, _ = predict(model, tok, held)
    print("before training:", score_heldout(held, base_preds, refs), flush=True)

    # Memory (2026-10-04, first run reaped for low RAM before step 100): Qwen3's vocabulary is 151k,
    # so full logits for 8 x ~300 tokens are ~1.5 GB fp32 before gradients; on Windows a VRAM overflow
    # spills into system RAM. Only the ~6 answer tokens carry loss, so the LM head runs on those
    # positions alone (~50x less), with gradient checkpointing and bs x accum instead of one big batch.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    steps = (len(train) // (args.bs * args.accum)) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / steps))
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        model.config.use_cache = False
        random.Random(ep).shuffle(train)
        for k, i in enumerate(range(0, len(train) - args.bs + 1, args.bs)):
            x, y, m = batch(train[i: i + args.bs])
            h = model.model(input_ids=x, attention_mask=m).last_hidden_state[:, :-1]
            tgt = y[:, 1:]
            sel = tgt != -100
            logits = model.lm_head(h[sel]).float()
            loss = torch.nn.functional.cross_entropy(logits, tgt[sel]) / args.accum
            loss.backward()
            if (k + 1) % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 100 == 0:
                print(f"  ep {ep} step {step}/{steps} loss {loss.item() * args.accum:.4f} {time.time() - t0:.0f}s "
                      f"vram {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        model.config.use_cache = True
        preds, _ = predict(model, tok, held)
        print(f"epoch {ep}:", score_heldout(held, preds, refs), flush=True)

    out = ROOT / "checkpoints" / f"picker_{args.tag}"
    model.save_pretrained(out); tok.save_pretrained(out)
    p50, p90 = latency_bs1(model, tok, held)
    res = {"base": args.base, "n_train": len(train), "n_heldout": len(held), "epochs": args.epochs, "lr": args.lr,
           "heldout_before": score_heldout(held, base_preds, refs), "heldout_after": score_heldout(held, preds, refs),
           "latency_ms_bs1": {"p50": p50, "p90": p90}, "checkpoint": str(out)}
    (ROOT / "results" / f"picker_{args.tag}_run.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
