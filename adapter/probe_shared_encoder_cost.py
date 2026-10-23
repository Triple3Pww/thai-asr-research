"""probe_shared_encoder_cost.py -- what does SHARING the ASR encoder cost the emotion head?

His question, 2026-09-22: *"make emotion in the same head like the same or we need specific head
for train to train emotion"*. Half of it is already settled and this arm does not re-open it:

\U0001f534 **THE SHARED CTC HEAD IS NOT AN OPTION.** Measured on the same 410 clips: SenseVoice's own
shipped head emits SAD at **0.019** recall while a single affine layer on **that same model's own
frozen encoder** reads SAD at **0.830**, with 40% `EMO_UNKNOWN` abstention. The information is in
the encoder; the shared head refuses to emit it. One emotion token competing in a 25,055-token
softmax, contributing one position of gradient out of hundreds, drifts to the safe default.

\u2b50 **THE OPEN QUESTION IS THE ENCODER, NOT THE HEAD.** Given a separate emotion head, does it sit
on the ASR encoder (one pass, one model -- the spec's goal) or on its own emotion encoder (two
passes, more VRAM, better features)? The 2026-09-19 probe put a frozen ASR encoder ~10 points
below emotion2vec -- **but that probe was deliberately ONE affine layer, an explicit lower bound.**
If the gap is an artefact of head capacity, a real head closes it and one shared encoder wins. If
the gap survives a real head, the information is genuinely not in the ASR encoder and two encoders
is the honest answer.

THE INSTRUMENT. Identical clips, identical actor-disjoint split, identical pooling (mean+std, as
cached by `probe_encoder_emotion.py`). The ONLY things that vary are the encoder and the head
capacity, which is swept: linear, MLP-128, MLP-512. Capacity and weight decay are chosen on the
DEV actors; the test actors are scored ONCE by the dev-selected winner (`A38`).

\u26a0 **SCOPE, stated so this is not over-read.** This measures the FROZEN shared-encoder case only.
It does NOT measure multi-task fine-tuning, where the shared encoder is trained on ASR and emotion
together -- that needs Thai ASR data in the same loop and a forgetting control (`S10`), and it is a
different and much larger experiment. What this bounds is: *if you freeze the ASR encoder and bolt
on the best head you can, how close do you get?*

Usage:
    $py = "D:/sensevoice-gpu/.venv/Scripts/python.exe"
    & $py adapter/probe_shared_encoder_cost.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_emotion import SV_CLASSES  # noqa: E402
from probe_encoder_emotion import macro_recall, split_by_actor  # noqa: E402

FEAT_DIR = Path("D:/sensevoice-spike/feats")
ENCODERS = ["mel", "typhoon", "sensevoice", "emotion2vec"]
# ASR-trained encoders, i.e. the ones a SHARED-encoder architecture would actually use.
ASR_ENCODERS = {"typhoon", "sensevoice"}


def build_head(dim: int, n_cls: int, hidden: int, seed: int):
    import torch
    torch.manual_seed(seed)
    if hidden == 0:
        return torch.nn.Linear(dim, n_cls)
    return torch.nn.Sequential(
        torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.3),
        torch.nn.Linear(hidden, n_cls))


def fit(Z, y, tr, ev, dim, n_cls, hidden, wd, seed=0, epochs=400, lr=1e-3):
    import torch
    import torch.nn.functional as F
    h = build_head(dim, n_cls, hidden, seed)
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=wd)
    xt, yt = torch.from_numpy(Z[tr]), torch.from_numpy(y[tr])
    best = None
    for ep in range(epochs):
        h.train(); opt.zero_grad()
        F.cross_entropy(h(xt), yt).backward(); opt.step()
        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            h.eval()
            with torch.no_grad():
                p = h(torch.from_numpy(Z[ev])).argmax(1).numpy()
            acc = float((p == y[ev]).mean())
            if best is None or acc > best[0]:
                best = (acc, ep + 1, {k: v.detach().clone() for k, v in h.state_dict().items()})
    h.load_state_dict(best[2]); h.eval()
    return h, best[0], best[1]


def mcnemar(a_ok, b_ok):
    nb = int(((a_ok == 1) & (b_ok == 0)).sum())
    nc = int(((a_ok == 0) & (b_ok == 1)).sum())
    n = nb + nc
    if n == 0:
        return nb, nc, 1.0
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(nb, nc) + 1)) / 2 ** n)
    return nb, nc, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("ADAPTER_OUT", "results"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch

    cls = {c: i for i, c in enumerate(SV_CLASSES)}
    results, preds_by_enc, t0 = {}, {}, time.time()

    for name in ENCODERS:
        fp = FEAT_DIR / f"feats_{name}.npz"
        if not fp.exists():
            print(f"[skip] {fp} missing"); continue
        d = np.load(fp, allow_pickle=True)
        X, ok = d["X"].astype(np.float32), d["ok"]
        meta = json.loads(str(d["meta"]))
        tr, dev, te, actors, test_actors = split_by_actor(meta)
        tr = [i for i in tr if ok[i]]; dev = [i for i in dev if ok[i]]
        te = [i for i in te if ok[i]]
        y = np.array([cls[m["gold"]] for m in meta], dtype=np.int64)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6     # standardise on TRAIN only
        Z = ((X - mu) / sd).astype(np.float32)

        # Capacity AND regularisation chosen together on the dev actors. Sweeping capacity is the
        # entire point: a fixed linear head is what made the 2026-09-19 number a lower bound.
        best = None
        grid = []
        for hidden, wd in itertools.product((0, 128, 512), (1e-3, 1e-2, 1e-1, 1.0)):
            _, dev_acc, ep = fit(Z, y, tr, dev, Z.shape[1], len(cls), hidden, wd, a.seed)
            grid.append({"hidden": hidden, "weight_decay": wd, "dev_accuracy": dev_acc,
                         "epoch": ep})
            if best is None or dev_acc > best[0]:
                best = (dev_acc, hidden, wd)
        dev_acc, hidden, wd = best
        head, _, ep = fit(Z, y, tr, dev, Z.shape[1], len(cls), hidden, wd, a.seed)
        with torch.no_grad():
            pt = head(torch.from_numpy(Z[te])).argmax(1).numpy()
        acc = float((pt == y[te]).mean())
        n = len(te)
        se = (acc * (1 - acc) / n) ** 0.5
        preds_by_enc[name] = (pt == y[te]).astype(int)

        results[name] = {
            "dim": int(X.shape[1]), "n_test": n,
            "head_chosen_on_dev": ("linear" if hidden == 0 else f"mlp{hidden}"),
            "weight_decay": wd, "dev_accuracy": dev_acc,
            "test_accuracy": acc,
            "ci95": [max(0., acc - 1.96 * se), min(1., acc + 1.96 * se)],
            "test_macro_recall": macro_recall(y[te], pt, len(cls)),
            "per_class_recall": {c: (float((pt[y[te] == i] == i).mean())
                                     if (y[te] == i).sum() else None) for c, i in cls.items()},
            "capacity_sweep": grid,
        }
        r = results[name]
        print(f"[{name:12s}] dim {r['dim']:5d} | head {r['head_chosen_on_dev']:7s} | "
              f"dev {dev_acc:.4f} | TEST {acc:.4f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] | "
              f"macro {r['test_macro_recall']:.4f}", flush=True)

    # The decision: best ASR encoder vs the emotion encoder, paired on the same clips.
    verdict, pair = None, None
    asr_best = max((k for k in results if k in ASR_ENCODERS),
                   key=lambda k: results[k]["test_accuracy"], default=None)
    if asr_best and "emotion2vec" in results:
        nb, nc, p = mcnemar(preds_by_enc[asr_best], preds_by_enc["emotion2vec"])
        gap = results["emotion2vec"]["test_accuracy"] - results[asr_best]["test_accuracy"]
        pair = {"asr_encoder": asr_best, "gap_to_emotion2vec": gap,
                "mcnemar_b": nb, "mcnemar_c": nc, "mcnemar_p": p}
        if p < 0.05:
            verdict = (f"\U0001f534 SHARING COSTS REAL ACCURACY. Best ASR encoder ({asr_best}) reaches "
                       f"{results[asr_best]['test_accuracy']:.4f} against emotion2vec's "
                       f"{results['emotion2vec']['test_accuracy']:.4f} -- a {gap:.4f} gap that a "
                       f"swept head did NOT close (paired p={p:.4f}). The emotion information is "
                       f"genuinely thinner in an ASR encoder. Budget two encoders, or accept the "
                       f"gap knowingly.")
        else:
            verdict = (f"\u2705 SHARING IS VIABLE. Best ASR encoder ({asr_best}) reaches "
                       f"{results[asr_best]['test_accuracy']:.4f} against emotion2vec's "
                       f"{results['emotion2vec']['test_accuracy']:.4f}; the difference is not "
                       f"separable on these 410 clips (paired p={p:.4f}). The 2026-09-19 gap was "
                       f"an artefact of head capacity -- one encoder, two heads.")

    out = {
        "arm": "shared_encoder_cost",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": "given a SEPARATE emotion head, must it sit on its own encoder?",
        "settled_separately": {
            "shared_ctc_head_sad_recall": 0.019,
            "linear_probe_on_same_encoder_sad_recall": 0.830,
            "note": "the shared CTC head is not a candidate; this arm is only about the encoder",
        },
        "scope": "FROZEN encoders only -- multi-task fine-tuning is a different, larger experiment",
        "encoders": results, "head_to_head": pair, "verdict": verdict,
        "seconds": round(time.time() - t0, 1),
    }
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    (d / "shared_encoder_cost_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nVERDICT: {verdict}\n[done] -> {d / 'shared_encoder_cost_metrics.json'}")


if __name__ == "__main__":
    main()
