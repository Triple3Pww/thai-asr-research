"""bench_public_thai_ser.py -- benchmark the ONLY public Thai SER model. Nobody ever has.

`A59` established that the Thai speech-emotion model landscape is one checkpoint:
`awghuku/wav2vec2-base-thai-ser` (wav2vec2-base, 5-class, last touched 2023-04-28, 19 downloads).
It ships **no model card, no metrics, no dataset declaration** -- only `config.json`,
`pytorch_model.bin` and eight tensorboard run files. So its accuracy is unknown to everyone,
including its author, in the sense that nothing was ever published.

\U0001f534 IT ALSO SHIPS NO LABEL NAMES. `id2label` is `{'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}`.
Five class indices and no statement of which is angry. As published, the model is **unusable
without guessing** -- which is itself a finding about the state of public Thai SER.

HOW THE MAPPING IS RECOVERED, and why it is fair rather than generous. The permutation from
class index to emotion is fitted by **majority vote on the TRAINING folds only**, then applied
unchanged to the test folds. Fitting it on test would be scoring the model on data used to
choose its own decoding -- the same leak `S8` exists to prevent. Two mappings are reported:
  - `fitted`  : the empirical best, derived on train folds
  - `canonical`: [neutral, angry, happy, sad, frustrated], the column order the corpus authors
                 use in `thaiser-experiments` (`score_cols`). If the fitted mapping agrees, that
                 is strong evidence the model really was trained on THAI-SER in that convention.

\U0001f534\U0001f534 THE CAVEAT THAT BOUNDS EVERY NUMBER THIS PRODUCES: **its train/test split is unknown.**
The model is named `-thai-ser` and was plainly trained on this corpus, but nothing states which
utterances it trained on. Our test folds may be its training data. **Every score here is
therefore an UPPER bound for it** and must never be quoted as its honest generalisation. It is
still worth measuring: an upper bound that loses is decisive, and one that wins is a warning.

FAIRNESS CHOICES, all made in the model's favour so a loss cannot be blamed on the harness:
  - scored on the SAME 8 folds, mic, agreement threshold and 4 classes as `F23`
  - its 5th logit (frustrated) is DROPPED rather than counted wrong, so a frustrated prediction
    never costs it -- argmax is taken over the four mapped classes only
  - reported both uncapped and at the 3 s cap (`F22`/`F23`), and the better one is highlighted

Usage:
    $py = "D:/sensevoice-gpu/.venv/Scripts/python.exe"
    & $py adapter/bench_public_thai_ser.py --device cuda
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_emotion2vec as FT  # noqa: E402
from finetune_emotion2vec import Clips  # noqa: E402
from probe_emotion import SV_CLASSES  # noqa: E402
from probe_thaiser_protocol import (  # noqa: E402
    AGREEMENT,
    FOLD_MAPPING,
    TRAIN_MIC,
    is_zoom,
    session_num,
    ua,
)

MODEL_ID = "awghuku/wav2vec2-base-thai-ser"
# The corpus authors' own column order, from thaiser-experiments `score_cols`.
CANONICAL = ["NEUTRAL", "ANGRY", "HAPPY", "SAD", "FRUSTRATED"]


def predict_all(device: str, cap_s: float | None, limit: int | None = None):
    """Run the model over every cached studio clip once; return [N, 5] logits."""
    import json as _json
    import urllib.request as _url

    import torch
    from transformers import (AutoFeatureExtractor, Wav2Vec2Config,
                              Wav2Vec2ForSequenceClassification)

    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    # 🔴 THE MODEL CANNOT BE LOADED AS PUBLISHED. Its config has
    #   "id2label": {"0": 0, ...}   -- integer VALUES where strings are required
    #   "label2id": {"0": "0", ...} -- string VALUES where ints are required
    # Current transformers validates the config strictly and refuses it:
    #   StrictDataclassFieldValidationError: Field 'id2label' ... doesn't match any type in
    #   (dict[int, str], dict[str, str], NoneType)
    # So the only public Thai SER model is not merely undocumented, it is BROKEN on a modern
    # stack. Repaired here in memory -- the hub copy is untouched -- so it can still be measured.
    with _url.urlopen(f"https://huggingface.co/{MODEL_ID}/raw/main/config.json", timeout=30) as r:
        raw = _json.load(r)
    raw["id2label"] = {str(k): str(v) for k, v in (raw.get("id2label") or {}).items()}
    raw["label2id"] = {str(k): int(v) for k, v in (raw.get("label2id") or {}).items()}
    cfg = Wav2Vec2Config(**raw)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID, config=cfg).to(device).eval()
    n_lab = model.config.num_labels
    print(f"[model] {MODEL_ID} | {sum(p.numel() for p in model.parameters())/1e6:.0f} M params "
          f"| {n_lab} classes | cap {cap_s or 'none'}", flush=True)

    clips = Clips(TRAIN_MIC)
    out = np.full((len(clips), n_lab), np.nan, dtype=np.float32)
    n_samp = int(cap_s * 16000) if cap_s else None
    t0, done = time.time(), 0
    with torch.no_grad():
        for i in range(len(clips)):
            if not clips.has(i):
                continue
            wav = clips.wav(i)
            if n_samp:
                wav = wav[:n_samp]
            if len(wav) < 640:
                wav = np.pad(wav, (0, 640 - len(wav)))
            iv = fe(wav, sampling_rate=16000, return_tensors="pt")
            logits = model(iv.input_values.to(device)).logits
            out[i] = logits[0].float().cpu().numpy()
            done += 1
            if done % 1000 == 0:
                print(f"  ... {done} clips, {time.time()-t0:.0f}s", flush=True)
            if limit and done >= limit:
                break
    print(f"[done] {done} clips scored in {time.time()-t0:.0f}s", flush=True)
    return out, clips


def fit_mapping(logits, y4, idx, n_lab):
    """Choose class-index -> emotion assignment that maximises accuracy on `idx` (TRAIN only).

    Brute force over the 5P4 = 120 injective assignments of four emotions to five indices. The
    fifth index is left unassigned, which is exactly the `frustrated` slot the 4-class protocol
    drops.
    """
    best = None
    for perm in itertools.permutations(range(n_lab), len(SV_CLASSES)):
        sub = logits[np.ix_(idx, list(perm))]
        pred = np.argmax(sub, axis=1)
        acc = float((pred == y4[idx]).mean())
        if best is None or acc > best[0]:
            best = (acc, perm)
    return best[1], best[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--caps", default="none,3",
                    help="comma list of window caps in seconds; 'none' = full clip")
    ap.add_argument("--out", default=os.environ.get("ADAPTER_OUT", "results"))
    a = ap.parse_args()

    cls = {c: i for i, c in enumerate(SV_CLASSES)}
    results = {}

    for cap_tok in [c.strip() for c in a.caps.split(",") if c.strip()]:
        cap = None if cap_tok == "none" else float(cap_tok)
        logits, clips = predict_all(a.device, cap)
        meta = clips.meta
        ok = ~np.isnan(logits[:, 0])
        keep = np.array([(r.get("agreement") or 0.0) >= AGREEMENT and r["gold"] in cls
                         and not is_zoom(r) for r in meta]) & ok
        snum = np.array([session_num(r["session_id"]) or -1 for r in meta])
        y4 = np.array([cls.get(r["gold"], -1) for r in meta], dtype=np.int64)
        n_lab = logits.shape[1]

        per_fold, maps = {}, []
        for f in range(8):
            tr_s, _, te_s = FOLD_MAPPING[f]
            tr = np.where(keep & np.isin(snum, tr_s))[0]
            te = np.where(keep & np.isin(snum, te_s))[0]
            if len(tr) < 50 or len(te) == 0:
                continue
            perm, tr_acc = fit_mapping(logits, y4, tr, n_lab)
            maps.append(perm)
            pred = np.argmax(logits[np.ix_(te, list(perm))], axis=1)
            per_fold[f] = {"n_test": int(len(te)), "mapping_fitted_on_train": list(perm),
                           "train_accuracy_of_mapping": tr_acc,
                           "WA": float((pred == y4[te]).mean()), "UA": ua(y4[te], pred)}
            print(f"  fold {f}: map {perm} (train acc {tr_acc:.4f}) | "
                  f"n={len(te)} WA {per_fold[f]['WA']:.4f} UA {per_fold[f]['UA']:.4f}", flush=True)

        # canonical-order control
        can = [CANONICAL.index(c) for c in SV_CLASSES]
        can_scores = []
        for f, r in per_fold.items():
            _, _, te_s = FOLD_MAPPING[f]
            te = np.where(keep & np.isin(snum, te_s))[0]
            pred = np.argmax(logits[np.ix_(te, can)], axis=1)
            can_scores.append(float((pred == y4[te]).mean()))

        wa = np.array([r["WA"] for r in per_fold.values()])
        uarr = np.array([r["UA"] for r in per_fold.values()])
        results[cap_tok] = {
            "cap_seconds": cap, "n_folds": len(per_fold),
            "WA_mean": float(wa.mean()), "WA_std": float(wa.std(ddof=1)),
            "UA_mean": float(uarr.mean()), "UA_std": float(uarr.std(ddof=1)),
            "canonical_order_WA_mean": float(np.mean(can_scores)) if can_scores else None,
            "mapping_stable_across_folds": len({tuple(m) for m in maps}) == 1,
            "mappings": [list(m) for m in maps],
            "per_fold": per_fold,
        }
        r = results[cap_tok]
        print(f"\n  [cap={cap_tok}] WA {r['WA_mean']*100:.2f} +/- {r['WA_std']*100:.2f}  "
              f"UA {r['UA_mean']*100:.2f} +/- {r['UA_std']*100:.2f} | "
              f"canonical-order WA {(r['canonical_order_WA_mean'] or 0)*100:.2f} | "
              f"mapping stable: {r['mapping_stable_across_folds']}\n", flush=True)

    best_cap = max(results, key=lambda k: results[k]["WA_mean"])
    out = {
        "arm": "bench_public_thai_ser", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_ID,
        "question": "how good is the only public Thai SER model? nobody had ever measured it",
        "protocol": "identical to F23 -- THAI-SER 8 folds, agreement>=0.71, 4 classes, mic_con",
        "label_recovery": ("model ships id2label {0..4} with NO emotion names; the index->emotion "
                           "permutation is fitted on TRAIN folds and applied to test"),
        "upper_bound_warning": ("its train/test split is UNKNOWN and our test folds may be its "
                                "training data -- every score here is an UPPER bound for it"),
        "fairness": ("its 5th logit is dropped rather than scored wrong, so a frustrated "
                     "prediction never costs it"),
        "by_cap": results, "best_cap": best_cap,
        "comparison": {
            "ours_F23_frozen_head_3s": {"WA": 77.18, "UA": 73.18},
            "ours_F23_frozen_head_uncapped": {"WA": 82.22, "UA": 79.82},
            "thaiser_published_baseline": {"WA": 67.34, "UA": 62.61},
        },
    }
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    (d / "bench_public_thai_ser_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 74)
    print(f"  {MODEL_ID}")
    for cap, r in results.items():
        print(f"    cap={cap:5s}  WA {r['WA_mean']*100:6.2f} +/- {r['WA_std']*100:4.2f}   "
              f"UA {r['UA_mean']*100:6.2f} +/- {r['UA_std']*100:4.2f}   ({r['n_folds']} folds)")
    print(f"\n  ours (frozen head, 3 s cap)      WA 77.18   UA 73.18")
    print(f"  ours (frozen head, uncapped)     WA 82.22   UA 79.82")
    print(f"  THAI-SER published baseline      WA 67.34   UA 62.61")
    print(f"\n  [!] every number for the public model is an UPPER bound -- its training split is unknown")
    print(f"  -> {d / 'bench_public_thai_ser_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
