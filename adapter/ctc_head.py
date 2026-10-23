#!/usr/bin/env python3
"""A CTC head on NECTEC's FROZEN encoder -> the name splice (`F34`) in ONE encoder pass.

    PYTHONUTF8=1 python adapter/ctc_head.py cache                 # encoder features, once (~GPU 15 min)
    PYTHONUTF8=1 python adapter/ctc_head.py train --tag h1        # linear head on the cache (minutes)

WHY (2026-10-07). `F34`: NECTEC's transcript + Typhoon's 115M CTC keyword spotter, spliced by frame,
beats the vendor's 0.78B list model at bias@50 -- but runs TWO encoders (~350 ms/turn vs 113). The
spotter only needs a per-frame [T, V] posterior; NECTEC is a pure transducer with none. A linear CTC
layer on NECTEC's own encoder output gives one, so the transcript and the spots come from ONE pass.

Choices, each with its why:
  * encoder FROZEN, eval mode, no SpecAugment: the RNN-T path is byte-for-byte NECTEC, so nothing the
    transcript does can get worse; and frozen features can be cached, so training is minutes.
  * NECTEC's own tokenizer (2,048 unigram pieces) + blank = its vocab size, as NeMo's hybrid models do:
    spots are scored in the same pieces the transcript is written in.
  * data: convention-matched LOTUSDIS train (`F32`) + Porjai train 16k (spaces stripped -- NECTEC
    writes unspaced Thai). Porjai is memorised by NECTEC (`F30`); harmless for a head on frozen
    features, it only has to learn to READ them.
  * PCM16-quantised audio, like the `:8007` wire. Val = LOTUSDIS val (speaker-held-out), greedy CER.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NEC = "E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo"
CACHE = Path("E:/Claude/datasets/ctchead_cache")
SETS = {
    "lot_train": ("E:/Claude/datasets/lotusdis_lav/manifest_train_conv.jsonl", "audio_filepath", "text"),
    "porjai_train": ("E:/Claude/datasets/porjai/manifest_train.jsonl", "wav", "text"),
    "lot_val": ("E:/Claude/datasets/lotusdis_lav/manifest_val_conv.jsonl", "audio_filepath", "text"),
}


def load_nec(device="cuda", path=None):
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.restore_from(path or NEC, map_location=device)
    m.eval()
    return m


@torch.no_grad()
def encode(m, ys: list[np.ndarray], device="cuda"):
    """Batch of float32 waveforms -> list of [T, D] fp16 encoder outputs (eval mode, no augmentation)."""
    lens = torch.tensor([len(y) for y in ys], device=device)
    x = torch.zeros(len(ys), int(lens.max()), device=device)
    for i, y in enumerate(ys):
        x[i, : len(y)] = torch.from_numpy(y)
    feats, fl = m.preprocessor(input_signal=x, length=lens)
    enc, el = m.encoder(audio_signal=feats, length=fl)          # [B, D, T]
    return [enc[i, :, : int(el[i])].T.half().cpu() for i in range(len(ys))]


def pcm16(y):
    return (np.clip(np.round(y * 32767), -32768, 32767) / 32768).astype(np.float32)


def cache_dir(base):
    # 2026-10-07: one cache per encoder -- the A2 (conv-LOTUSDIS) encoder's features differ from base NECTEC's
    return CACHE if base == NEC else CACHE.parent / f"ctchead_cache_{Path(base).stem}"


def cmd_cache(args):
    m = load_nec(path=args.base)
    tok = m.tokenizer
    CACHE_ = cache_dir(args.base)
    CACHE_.mkdir(parents=True, exist_ok=True)
    for name, (man, wk, tk) in SETS.items():
        out = CACHE_ / f"{name}.pt"
        if out.exists():
            print(name, "cached");
            continue
        rows = [json.loads(l) for l in open(man, encoding="utf-8") if l.strip()]
        rows.sort(key=lambda r: r.get("duration", r.get("dur", 0)))   # similar lengths per batch
        feats, labels, texts = [], [], []
        for i in range(0, len(rows), 16):
            batch = rows[i: i + 16]
            ys = []
            for r in batch:
                y, _ = sf.read(r[wk], dtype="float32")
                ys.append(pcm16(y.mean(axis=1) if y.ndim > 1 else y))
            feats += encode(m, ys)
            for r in batch:
                t = r[tk].replace(" ", "") if name.startswith("porjai") else r[tk]
                texts.append(t)
                labels.append(torch.tensor(tok.text_to_ids(t), dtype=torch.int16))
            if i % 1600 == 0:
                print(name, i, len(rows), flush=True)
        torch.save({"feats": feats, "labels": labels, "texts": texts}, out)
        print(name, "->", out, sum(f.shape[0] for f in feats), "frames", flush=True)


def greedy_cer(head, feats, texts, tok, blank, device):
    sys.path.insert(0, str(HERE))
    from gate_lotusdis import lev, norm
    errs = chars = 0
    with torch.no_grad():
        for f, t in zip(feats, texts):
            ids = head(f.to(device).float()).argmax(-1).tolist()
            out, prev = [], -1
            for i in ids:
                if i != prev and i != blank:
                    out.append(i)
                prev = i
            ref = norm(t)
            errs += lev(ref, norm(tok.ids_to_text(out)))
            chars += len(ref)
    return errs / max(chars, 1)


def cmd_train(args):
    import nemo.collections.asr as nemo_asr  # noqa: F401  (tokenizer class)
    m = load_nec("cpu", args.base)
    tok, V = m.tokenizer, m.tokenizer.vocab_size
    D = m.cfg.encoder.d_model
    del m
    C = cache_dir(args.base)
    tr = [torch.load(C / f"{n}.pt") for n in args.sets]
    va = torch.load(C / "lot_val.pt")
    pairs = [(f, l) for d in tr for f, l in zip(d["feats"], d["labels"]) if len(l) and f.shape[0] >= len(l)]
    print(f"train clips {len(pairs)}  frames {sum(p[0].shape[0] for p in pairs)}  vocab {V}+blank  d {D}")
    dev = "cuda"
    if args.hidden:
        head = torch.nn.Sequential(torch.nn.Linear(D, args.hidden), torch.nn.GELU(),
                                   torch.nn.Linear(args.hidden, V + 1)).to(dev)
    else:
        head = torch.nn.Linear(D, V + 1).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-2)
    ctc = torch.nn.CTCLoss(blank=V, zero_infinity=True)
    rng = random.Random(0)
    best, log = 9.0, []
    for ep in range(args.epochs):
        rng.shuffle(pairs)
        tot = 0.0
        for i in range(0, len(pairs), args.bs):
            b = pairs[i: i + args.bs]
            T = max(f.shape[0] for f, _ in b)
            x = torch.zeros(len(b), T, D, device=dev)
            for j, (f, _) in enumerate(b):
                x[j, : f.shape[0]] = f.to(dev).float()
            lp = torch.log_softmax(head(x), -1).transpose(0, 1)       # [T, B, V+1]
            tgt = torch.cat([l.long() for _, l in b]).to(dev)
            loss = ctc(lp, tgt, torch.tensor([f.shape[0] for f, _ in b]), torch.tensor([len(l) for _, l in b]))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        cer = greedy_cer(head, va["feats"], va["texts"], tok, V, dev)
        log.append({"epoch": ep, "loss": round(tot / len(pairs), 4), "val_greedy_cer": round(cer, 4)})
        print(log[-1], flush=True)
        if cer < best:
            best = cer
            torch.save({"state": head.state_dict(), "vocab": V, "d": D, "hidden": args.hidden,
                        "base": args.base, "sets": args.sets, "log": log}, ROOT / "checkpoints" / f"ctchead_{args.tag}.pt")
    (ROOT / "results" / f"ctchead_{args.tag}_run.json").write_text(json.dumps(
        {"tag": args.tag, "sets": args.sets, "hidden": args.hidden, "lr": args.lr, "bs": args.bs,
         "epochs": args.epochs, "best_val_greedy_cer": best, "log": log}, indent=1), encoding="utf-8")
    print("best val greedy CER", best)


def main() -> int:
    if not sys.flags.utf8_mode:
        raise SystemExit("run with PYTHONUTF8=1")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--base", default=NEC)
    t = sub.add_parser("train")
    t.add_argument("--base", default=NEC, help="the encoder the head sits on (must match the cache)")
    t.add_argument("--tag", required=True)
    t.add_argument("--sets", nargs="+", default=["lot_train", "porjai_train"])
    t.add_argument("--hidden", type=int, default=0, help="0 = one linear layer (NeMo's CTC decoder shape)")
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    {"cache": cmd_cache, "train": cmd_train}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
