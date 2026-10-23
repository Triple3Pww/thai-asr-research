"""probe_emotion_duration.py -- how many seconds does emotion2vec need before it can decide?

His question, 2026-09-22: *"how long audio need for emotion2vec for an prediction"*. It is a
latency question for the voice loop -- deciding emotion from 1.5 s is a different product from
needing 4 s -- and the repo already has an OBSERVATIONAL answer that cannot be trusted:

    zero-shot accuracy by clip length, 1,256 THAI-SER clips (2026-09-19)
    <1 s 0.656 | 1-2 s 0.697 | 2-4 s 0.770 | >4 s 0.737

\U0001f534 **That table confounds duration with content.** Short THAI-SER clips are not long clips cut
short -- they are different utterances, by different actors, carrying a different mix of emotions.
A 2-4 s peak could mean "2-4 s is the sweet spot" or simply "the clips that happen to be 2-4 s
long are easier". Nothing in an observational split can separate those.

THE FIX: TRUNCATE THE SAME CLIPS. Take only clips at least as long as the longest window tested,
then cut each one to every window length. The clip set, the speakers, the emotions and the label
are then IDENTICAL at every point on the curve, so the only thing that varies is how much audio
the model was given. Anything that moves is duration.

\u26a0 **Truncated from the START**, not the centre: a live loop has the leading N seconds, not a
centred excerpt. That makes this the operationally honest curve, and it is also the harder one --
the emotional peak of an utterance is often not at its beginning.

Reported for both heads, because they can differ: the shipped 9-class `proj` head (what you get
out of the box) and the dev-selected 4-class head from `F18` (what we would actually deploy).
Per-class recall is reported too -- there is no reason to assume ANGRY and SAD need the same
amount of evidence.

Usage:
    $py = "D:/sensevoice-gpu/.venv/Scripts/python.exe"
    & $py adapter/probe_emotion_duration.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_emotion2vec as FT  # noqa: E402
from finetune_emotion2vec import Clips, load_core, load_split  # noqa: E402
from probe_emotion import E2V_TO_SV, SV_CLASSES  # noqa: E402
from probe_encoder_emotion import macro_recall  # noqa: E402
from probe_pseudolabel_yield import fit_head, softmax  # noqa: E402

WINDOWS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, None]      # None = the whole clip
SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--feats-device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--min-seconds", type=float, default=4.0,
                    help="only clips at least this long, so the set is fixed across windows")
    ap.add_argument("--out", default=os.environ.get("ADAPTER_OUT", "results"))
    a = ap.parse_args()

    import torch
    FT.DEVICE = a.device
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda but cuda is unavailable; use D:/sensevoice-gpu")

    clips = Clips()
    tr, dev, te, _, test_actors = load_split(clips)
    gold_all = [r["gold"] for r in clips.meta]
    cls = {c: i for i, c in enumerate(SV_CLASSES)}
    y_all = np.array([cls[g] for g in gold_all], dtype=np.int64)

    # The fixed evaluation set: held-out actors AND long enough for every window.
    keep = [i for i in te if clips.meta[i]["seconds"] >= a.min_seconds]
    if not keep:
        sys.exit(f"no test clips >= {a.min_seconds}s")
    gold = [gold_all[i] for i in keep]
    y = np.array([cls[g] for g in gold], dtype=np.int64)
    print(f"[set] {len(keep)} test clips >= {a.min_seconds}s, from {len(test_actors)} held-out "
          f"actors | class mix " +
          " ".join(f"{c}:{sum(1 for g in gold if g == c)}" for c in SV_CLASSES), flush=True)

    # The F18 head, refit deterministically on the FULL-LENGTH training features -- the deployed
    # head is trained once on whole clips, so truncation must be a test-time change only.
    feats = np.load(FT.CACHE_DIR / f"feats_e2v_mean_{a.feats_device}.npy")
    mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-6
    Z = ((feats - mu) / sd).astype(np.float32)
    logits_ts, head_info = fit_head(Z, y_all, tr, dev)
    A = np.concatenate([Z, np.ones((Z.shape[0], 1), np.float32)], 1)
    W, *_ = np.linalg.lstsq(A, logits_ts, rcond=None)      # exact affine recovery, verified

    core, labels = load_core()
    core.eval()
    keep_cols = [k for k, l in enumerate(labels) if not str(l).startswith("unuse")]
    names9 = [E2V_TO_SV.get(str(labels[k]).split("/")[-1].strip().lower(),
                            str(labels[k]).split("/")[-1].strip().upper()) for k in keep_cols]

    rows, t0 = {}, time.time()
    with torch.no_grad():
        Wp = core.proj.weight.detach().cpu().numpy()
        bp = core.proj.bias.detach().cpu().numpy()
        for w in WINDOWS:
            n_samp = None if w is None else int(w * SR)
            p4, p9, frames = [], [], []
            for i in keep:
                wav = clips.wav(i)
                if n_samp is not None:
                    wav = wav[:n_samp]
                x = FT.encode(core, wav)                   # [1, T, 1024]
                frames.append(int(x.shape[1]))
                f = x.mean(dim=1).squeeze(0).cpu().numpy()
                z = (f - mu) / sd
                p4.append(softmax((np.concatenate([z, [1.0]]) @ W)[None, :])[0])
                p9.append(softmax((f @ Wp.T + bp)[None, keep_cols])[0])
            p4 = np.stack(p4); p9 = np.stack(p9)
            pred4 = np.array([cls[SV_CLASSES[k]] for k in p4.argmax(1)])
            pred9lab = [names9[k] for k in p9.argmax(1)]
            acc9 = float(np.mean([pred9lab[j] == gold[j] for j in range(len(gold))]))
            acc4 = float((pred4 == y).mean())
            key = "full" if w is None else f"{w:g}s"
            rows[key] = {
                "window_seconds": w, "n": len(keep),
                "mean_encoder_frames": float(np.mean(frames)),
                "finetuned_head": {
                    "accuracy": acc4,
                    "macro_recall": macro_recall(y, pred4, len(cls)),
                    "mean_confidence": float(p4.max(1).mean()),
                    "per_class_recall": {c: (float((pred4[y == k] == k).mean())
                                             if (y == k).sum() else None)
                                         for c, k in cls.items()},
                },
                "shipped_9class_head": {"accuracy": acc9},
            }
            r = rows[key]
            print(f"  [{key:>5s}] frames {r['mean_encoder_frames']:6.1f} | finetuned acc "
                  f"{acc4:.4f} macro {r['finetuned_head']['macro_recall']:.4f} "
                  f"conf {r['finetuned_head']['mean_confidence']:.3f} | shipped acc {acc9:.4f}",
                  flush=True)

    full = rows["full"]["finetuned_head"]["accuracy"]
    reach = None
    for w in WINDOWS:
        if w is None:
            continue
        k = f"{w:g}s"
        if rows[k]["finetuned_head"]["accuracy"] >= full - 0.02:
            reach = w
            break

    out = {
        "arm": "emotion_duration",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": "how much audio does emotion2vec need before its prediction stops improving?",
        "method": ("same clips truncated from the start to each window, so speakers, emotions and "
                   "labels are identical at every point -- only the amount of audio varies"),
        "eval_set": {"n": len(keep), "min_seconds": a.min_seconds,
                     "test_actors": test_actors,
                     "class_mix": {c: sum(1 for g in gold if g == c) for c in SV_CLASSES}},
        "head": head_info,
        "by_window": rows,
        "shortest_window_within_2pts_of_full": reach,
        "seconds": round(time.time() - t0, 1),
    }
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    (d / "emotion_duration_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nshortest window within 2 points of full-clip accuracy: "
          f"{reach if reach else 'none of the tested windows'}")
    print(f"[done] -> {d / 'emotion_duration_metrics.json'}")


if __name__ == "__main__":
    main()
