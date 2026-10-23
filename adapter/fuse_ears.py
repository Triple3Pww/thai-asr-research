#!/usr/bin/env python3
"""Two ears, one decoder: Pathumma-whisper-large's ENCODER fused into NECTEC's RNN-T, frame by frame.

    PYTHONUTF8=1 python adapter/fuse_ears.py cache --set bench          # 682 real utts, batch 1
    PYTHONUTF8=1 python adapter/fuse_ears.py cache --set gs2_train      # + gs2_val
    PYTHONUTF8=1 python adapter/fuse_ears.py check                      # identity fusion == NECTEC?
    PYTHONUTF8=1 python adapter/fuse_ears.py train --tag f1             # fusion (Whisper on)
    PYTHONUTF8=1 python adapter/fuse_ears.py train --tag c1 --no-whisper   # control: same block, Whisper zeroed
    PYTHONUTF8=1 python adapter/fuse_ears.py eval --tag f1              # -> results/fuse_f1_bench_hyps.jsonl

WHY (2026-10-10, his ask: CER 0.05 by "more architecture"). `F42`: the per-utterance oracle over
{NECTEC, Pathumma-L, typhoon-L} is -24.5% on the dev half, but every TEXT-level combiner (ROVER, LM,
product of experts, three trained judges, `F42`-`F44`) lands within +-3%: once each ear has written its
transcript, the evidence of WHICH one heard right is gone. Fuse earlier -- at the encoder output, where
the acoustics still are -- and let a trained decoder choose per frame. Multi-encoder / multi-stream
E2E ASR reports -18..32% (MEM-Res, Li et al. 2019) and a two-stage "train only the fusion on frozen
encoders" recipe -8..32% (Li et al. 2020).

Choices, each with its why:
  * Whisper ENCODER only (72 ms per 30-s window on this GPU): the decoder is what made large Whisper
    ~1 s/turn. Fused system ~ NECTEC 43 ms + 72 ms, inside the 500 ms budget with room.
  * Pathumma-whisper-th-large-v3: best sound-level ear on the benchmark (`F39` SylER), and the
    NECTEC+Pathumma-L oracle is the larger one (`F40`: -20%).
  * Both encoders FROZEN and cached; GigaSpeech2-th dev+test (`F45` manifests) is fresh to BOTH
    (`F44` gate) -- a fusion trained on data one ear has heard learns to trust that ear (`F43`).
  * Fusion output is ZERO-INITIALISED and added to NECTEC's encoder output: at step 0 the system is
    NECTEC exactly, so anything the block learns is measured against the right base.
  * Control = the identical block with the Whisper input zeroed (`--no-whisper`), same data, same
    trainable decoder/joint: isolates what the second ear adds from what GS2 adaptation adds.
  * Whisper 50 fps -> mean-pool x4 -> 12.5 fps = NECTEC's rate (both from a 10 ms hop); a depthwise
    conv in the block absorbs a frame or two of offset between the ears.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from ctc_head import NEC, encode, load_nec, pcm16  # noqa: E402

WHISPER = "nectec/Pathumma-whisper-th-large-v3"
CACHE = Path("E:/Claude/datasets/fuse_cache")
BENCH = Path("E:/asr-bench/thai-contextasr-bench")
G = "E:/Claude/datasets/gigaspeech2_th"
SETS = {"gs2_train": f"{G}/manifest_train_conv.jsonl", "gs2_val": f"{G}/manifest_val_conv.jsonl"}
DN, DW = 512, 1280


# ----------------------------------------------------------------------------------------------- data
def rows_of(name):
    if name == "bench":
        rows = sorted((json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")),
                      key=lambda r: r["utt_id"])
        return [{"id": r["utt_id"], "wav": str(BENCH / r["wav"]), "text": ""} for r in rows if r["split"] == "real"]
    return [{"id": r["id"], "wav": r["audio_filepath"], "text": r["text"], "dur": r["duration"]}
            for r in (json.loads(l) for l in open(SETS[name], encoding="utf-8") if l.strip())]


def read(p):
    y, _ = sf.read(p, dtype="float32")
    return pcm16(y.mean(axis=1) if y.ndim > 1 else y)


class WhisperEar:
    def __init__(self, name=WHISPER, dev="cuda"):
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.fe = WhisperFeatureExtractor.from_pretrained(name)
        self.enc = WhisperModel.from_pretrained(name, dtype=torch.float16).encoder.to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, ys: list[np.ndarray]) -> list[torch.Tensor]:
        """waveforms -> list of [ceil(frames50/4), 1280] fp16 (pooled to 12.5 fps), any length (30-s windows)."""
        wins, owner = [], []
        for i, y in enumerate(ys):
            for s in range(0, max(len(y), 1), 480000):
                wins.append(y[s: s + 480000]); owner.append(i)
        outs = []
        for k in range(0, len(wins), 8):
            chunk = wins[k: k + 8]
            x = self.fe(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(self.dev).half()
            h = self.enc(x).last_hidden_state                                   # [b, 1500, 1280]
            for j, w in enumerate(chunk):
                outs.append(h[j, : math.ceil(len(w) / 320)])
        per = [[] for _ in ys]
        for o, i in zip(outs, owner):
            per[i].append(o)
        res = []
        for p in per:
            h = torch.cat(p)                                                    # [T50, 1280]
            T = math.ceil(h.shape[0] / 4)
            h = torch.cat([h, h[-1:].expand(T * 4 - h.shape[0], -1)])
            res.append(h.view(T, 4, DW).float().mean(1).half().cpu())
        return res


def align(w: torch.Tensor, T: int) -> torch.Tensor:
    if w.shape[0] >= T:
        return w[:T]
    return torch.cat([w, w[-1:].expand(T - w.shape[0], -1)])


def cmd_cache(args):
    names = [args.set] + (["gs2_val"] if args.set == "gs2_train" else [])
    CACHE.mkdir(parents=True, exist_ok=True)
    m = load_nec()
    tok = m.tokenizer
    ear = WhisperEar()
    for name in names:
        if (CACHE / f"{name}.json").exists():
            print(name, "cached"); continue
        rows = rows_of(name)
        bs = 1 if name == "bench" else 16              # bench: batch 1, exactly as eval_rt_local decodes it
        if name != "bench":
            rows.sort(key=lambda r: r["dur"])
        fn, fw = open(CACHE / f"{name}.nec.bin", "wb"), open(CACHE / f"{name}.wsp.bin", "wb")
        idx, off, mism = [], 0, []
        t0 = time.time()
        for i in range(0, len(rows), bs):
            b = rows[i: i + bs]
            ys = [read(r["wav"]) for r in b]
            en = encode(m, ys)
            ew = ear(ys)
            for r, a, w in zip(b, en, ew):
                mism.append(w.shape[0] - a.shape[0])
                w = align(w, a.shape[0])
                fn.write(a.numpy().tobytes()); fw.write(w.numpy().tobytes())
                idx.append({"id": r["id"], "off": off, "T": a.shape[0], "text": r["text"],
                            "labels": tok.text_to_ids(r["text"]) if r["text"] else []})
                off += a.shape[0]
            if i % (bs * 100) == 0:
                print(name, i, len(rows), f"{time.time() - t0:.0f}s", flush=True)
        fn.close(); fw.close()
        mm = np.array(mism)
        meta = {"rows": idx, "frames": off, "len_mismatch_w_minus_n": {str(k): int((mm == k).sum()) for k in sorted(set(mm.tolist()))}}
        (CACHE / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        print(name, "->", off, "frames; whisper-minus-nectec frame counts", meta["len_mismatch_w_minus_n"], flush=True)


class Cache:
    def __init__(self, name):
        self.meta = json.loads((CACHE / f"{name}.json").read_text(encoding="utf-8"))
        self.rows = self.meta["rows"]
        F = self.meta["frames"]
        self.n = np.memmap(CACHE / f"{name}.nec.bin", dtype=np.float16, mode="r", shape=(F, DN))
        self.w = np.memmap(CACHE / f"{name}.wsp.bin", dtype=np.float16, mode="r", shape=(F, DW))

    def get(self, r):
        s = slice(r["off"], r["off"] + r["T"])
        return torch.from_numpy(np.array(self.n[s])), torch.from_numpy(np.array(self.w[s]))


# --------------------------------------------------------------------------------------------- model
class Fuse(nn.Module):
    """out = nec + Zero( Block( [nec ; P(whisper)] ) ). Block = in-proj, depthwise conv, n transformer layers."""

    def __init__(self, layers=2, d=DN, dw=DW, k=9, drop=0.1, use_whisper=True):
        super().__init__()
        self.use_whisper = use_whisper
        self.wp = nn.Sequential(nn.LayerNorm(dw), nn.Linear(dw, d))
        self.inp = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU())
        self.conv = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, drop, batch_first=True,
                                                                    norm_first=True, activation="gelu"), layers)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.drop = nn.Dropout(drop)

    def forward(self, n, w, mask):                  # n [B,T,512], w [B,T,1280], mask [B,T] True = pad
        wv = self.wp(w) if self.use_whisper else torch.zeros_like(n)
        h = self.inp(torch.cat([n, self.drop(wv)], -1))
        # review 2026-10-10: zero pad frames before the conv, or a clip's last frames see LN/bias junk in a
        # padded training batch but zeros at batch-1 eval -- a train/eval mismatch at every word ending
        h = h.masked_fill(mask[..., None], 0.0)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.tf(h, src_key_padding_mask=mask)
        return n + self.out(h)


def batchify(cache, rows, dev):
    T = max(r["T"] for r in rows)
    n = torch.zeros(len(rows), T, DN); w = torch.zeros(len(rows), T, DW)
    for i, r in enumerate(rows):
        a, b = cache.get(r)
        n[i, : r["T"]] = a.float(); w[i, : r["T"]] = b.float()
    L = torch.tensor([r["T"] for r in rows])
    mask = torch.arange(T)[None] >= L[:, None]
    return n.to(dev), w.to(dev), L.to(dev), mask.to(dev)


def greedy(m, enc, L):
    """enc [B,T,512] -> list of texts, through NECTEC's own greedy RNN-T decoding."""
    hyps = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc.transpose(1, 2).contiguous(), encoded_lengths=L)
    hyps = hyps[0] if isinstance(hyps, tuple) else hyps
    return [(h.text if hasattr(h, "text") else str(h)).strip() for h in hyps]


def setup_decoding(m):
    from omegaconf import open_dict
    with open_dict(m.cfg.decoding):
        m.cfg.decoding.strategy = "greedy_batch"
        m.cfg.decoding.greedy.use_cuda_graph_decoder = False
    m.change_decoding_strategy(m.cfg.decoding, verbose=False)


def val_cer(m, fuse, cache, dev, bs=16):
    from gate_lotusdis import lev, norm
    errs = chars = 0
    fuse.eval(); m.eval()
    rows = sorted(cache.rows, key=lambda r: r["T"])
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            b = rows[i: i + bs]
            n, w, L, mask = batchify(cache, b, dev)
            for r, h in zip(b, greedy(m, fuse(n, w, mask), L)):
                ref = norm(r["text"]); errs += lev(ref, norm(h)); chars += len(ref)
    return errs / chars


def cmd_check(args):
    """Identity fusion (zero-init block) on the bench cache must reproduce NECTEC's transcripts."""
    dev = "cuda"
    m = load_nec(); setup_decoding(m)
    fuse = Fuse().to(dev).eval()
    c = Cache("bench")
    ref = {json.loads(l)["utt_id"]: json.loads(l)["text"] for l in open(ROOT / "results/nectec_fc_greedy_hyps.jsonl", encoding="utf-8")}
    same = 0
    with torch.no_grad():
        for r in c.rows:
            n, w, L, mask = batchify(c, [r], dev)
            same += greedy(m, fuse(n, w, mask), L)[0] == ref[r["id"]].strip()
    print(f"identity fusion reproduces NECTEC on {same}/{len(c.rows)} benchmark utts")


def cmd_train(args):
    dev = "cuda"
    torch.manual_seed(args.seed); rng = random.Random(args.seed)
    m = load_nec(dev); setup_decoding(m)
    from _rnnt_fast import rnnt_nll
    blank = m.joint.num_classes_with_blank - 1
    fuse = Fuse(layers=args.layers, use_whisper=not args.no_whisper).to(dev)
    for p in m.encoder.parameters():
        p.requires_grad_(False)
    groups = []
    if args.lr > 0:
        groups.append({"params": fuse.parameters(), "lr": args.lr})
    else:                       # control c0 (review): no block at all -- zero-init stays identity, decoder/joint only
        for p in fuse.parameters():
            p.requires_grad_(False)
    if args.dec_lr > 0:
        groups.append({"params": list(m.decoder.parameters()) + list(m.joint.parameters()), "lr": args.dec_lr})
    else:
        for p in list(m.decoder.parameters()) + list(m.joint.parameters()):
            p.requires_grad_(False)
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    tr, va = Cache("gs2_train"), Cache("gs2_val")
    rows = [r for r in tr.rows if r["labels"] and r["T"] >= len(r["labels"]) // 4]
    # length-bucketed batches, shuffled order
    rows.sort(key=lambda r: r["T"])
    batches = [rows[i: i + args.bs] for i in range(0, len(rows), args.bs)]
    steps = args.epochs * len(batches) if args.max_steps <= 0 else args.max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    base = val_cer(m, fuse, va, dev)
    log = [{"step": 0, "val_cer": round(base, 4)}]
    print(f"train clips {len(rows)}  batches/epoch {len(batches)}  whisper={'off' if args.no_whisper else 'on'}  step-0 val CER {base:.4f}", flush=True)
    best, step = base, 0
    ck = ROOT / "checkpoints" / f"fuse_{args.tag}.pt"
    for ep in range(args.epochs):
        rng.shuffle(batches)
        tot, k = 0.0, 0
        for b in batches:
            if step >= steps:
                break
            fuse.train(); m.decoder.train(); m.joint.train()
            n, w, L, mask = batchify(tr, b, dev)
            U = max(len(r["labels"]) for r in b)
            y = torch.zeros(len(b), U, dtype=torch.long, device=dev)
            for i, r in enumerate(b):
                y[i, : len(r["labels"])] = torch.tensor(r["labels"])
            yl = torch.tensor([len(r["labels"]) for r in b], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = fuse(n, w, mask)
                g, _ = m.decoder(targets=y, target_length=yl)[:2]             # [B, H, U+1]
                acts = m.joint.joint(f, g.transpose(1, 2))                   # [B, T, U+1, V+1]
            loss = (rnnt_nll(acts, y, L, yl, blank) / yl.clamp(min=1)).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in list(fuse.parameters()) + list(m.decoder.parameters())
                                            + list(m.joint.parameters()) if p.requires_grad], 1.0)
            opt.step(); sched.step(); step += 1
            tot += loss.item(); k += 1
            if step % 200 == 0:
                print(f"ep {ep} step {step}/{steps} loss/token {tot / k:.4f}", flush=True); tot, k = 0.0, 0
            if step % args.eval_every == 0 or step == steps:
                cer = val_cer(m, fuse, va, dev)
                log.append({"step": step, "val_cer": round(cer, 4)}); print(log[-1], flush=True)
                if cer < best:
                    best = cer
                    torch.save({"fuse": fuse.state_dict(), "decoder": m.decoder.state_dict(), "joint": m.joint.state_dict(),
                                "layers": args.layers, "use_whisper": not args.no_whisper, "args": vars(args), "log": log}, ck)
    (ROOT / "results" / f"fuse_{args.tag}_run.json").write_text(json.dumps(
        {"args": vars(args), "whisper": WHISPER, "train_clips": len(rows), "step0_val_cer": base,
         "best_val_cer": best, "log": log, "checkpoint": str(ck) if best < base else None}, indent=1), encoding="utf-8")
    print("best val CER", best, "(step 0:", base, ")")


def cmd_eval(args):
    dev = "cuda"
    m = load_nec(); setup_decoding(m)
    ck = torch.load(ROOT / "checkpoints" / f"fuse_{args.tag}.pt", map_location=dev)
    fuse = Fuse(layers=ck["layers"], use_whisper=ck["use_whisper"]).to(dev).eval()
    fuse.load_state_dict(ck["fuse"]); m.decoder.load_state_dict(ck["decoder"]); m.joint.load_state_dict(ck["joint"])
    with torch.no_grad():       # 2026-10-10: residual scale, like F45's base interpolation (f2 drifted to GS2's no-Latin style)
        fuse.out.weight.mul_(args.alpha); fuse.out.bias.mul_(args.alpha)
    m.eval()
    c = Cache("bench")
    out = []
    with torch.no_grad():
        for r in c.rows:
            n, w, L, mask = batchify(c, [r], dev)
            out.append({"utt_id": r["id"], "condition": "none", "text": greedy(m, fuse(n, w, mask), L)[0]})
    sfx = "" if args.alpha == 1.0 else f"_a{int(round(args.alpha * 100))}"
    p = ROOT / "results" / f"fuse_{args.tag}{sfx}_bench_hyps.jsonl"
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("->", p)


def main() -> int:
    if not sys.flags.utf8_mode:
        raise SystemExit("run with PYTHONUTF8=1")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache"); c.add_argument("--set", required=True, choices=("bench", "gs2_train"))
    sub.add_parser("check")
    t = sub.add_parser("train")
    t.add_argument("--tag", required=True)
    t.add_argument("--no-whisper", action="store_true", help="control: identical block, Whisper input zeroed")
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--dec-lr", type=float, default=5e-5, help="0 = decoder and joint frozen")
    t.add_argument("--bs", type=int, default=8)
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--eval-every", type=int, default=1000)
    t.add_argument("--seed", type=int, default=2026)
    t.add_argument("--max-steps", type=int, default=0, help="smoke test")
    e = sub.add_parser("eval"); e.add_argument("--tag", required=True)
    e.add_argument("--alpha", type=float, default=1.0, help="scale on the fusion residual (1 = as trained)")
    args = ap.parse_args()
    {"cache": cmd_cache, "check": cmd_check, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
