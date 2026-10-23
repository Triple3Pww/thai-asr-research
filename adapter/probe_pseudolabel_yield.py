"""probe_pseudolabel_yield.py -- can the teacher's CONFIDENCE rescue pseudo-labelling?

His question, 2026-09-22: *"what if we use this emotion model to label all the dataset which
might have most sound be the natural, will that be a problem"*. The answer is yes, and the
arithmetic is in `RESEARCH-PROGRESS.md`: the teacher mislabels a neutral clip as HAPPY 13.8% of
the time, which is harmless when NEUTRAL is a third of the corpus and fatal when it is 85% --
the leaked neutrals outnumber the true happy clips 3:1 and HAPPY pseudo-label precision falls to
0.232. Accuracy barely moves; the labels become garbage. That is CLASS PRIOR SHIFT: nothing
about the teacher changed, only the mix of what it is asked to label.

🔴 THE QUESTION THIS SCRIPT SETTLES, and it decides the whole S13 step-2 plan:
    does the teacher KNOW when it is wrong?
If confidence separates its right answers from its wrong ones, then a per-class threshold buys
back precision and pseudo-labelling is viable at reduced yield. If it does not, no threshold
will, and the only honest routes are soft distillation or abandoning the minority classes.

THE INSTRUMENT. Two things are reported and they answer different questions:
  1. SELECTIVE PREDICTION on the 410 actor-held-out clips -- AUROC of confidence as a detector
     of its own errors, plus precision at each coverage. This is measured, not projected.
  2. PROJECTED YIELD under a natural-speech prior -- take P(pred | true) measured at each
     threshold, re-weight by a corpus that is mostly neutral, and report what survives.
     \u26a0 This half is a PROJECTION and it is OPTIMISTIC: P(pred|true) is measured on ACTED
     studio speech where the teacher is at its best (impro 0.699 vs script 0.786, and a raw-mel
     floor of 0.5756 says much of the apparent skill is reading exaggerated actor prosody). On
     natural speech the teacher is worse, so real precision sits BELOW these numbers.

TWO TEACHERS, because the model we would actually label with is not the one on the shelf:
  zeroshot   the shipped 9-class `proj` head. Emits FEARFUL/SURPRISED/<UNK> on real clips --
             those become unusable labels, so they are carried as an explicit OTHER column.
  finetuned  the dev-selected 4-class head from `finetune_emotion2vec.py --condition head`
             (test 0.7805). Refit here deterministically from the same cached features, same
             seed and same dev-chosen weight decay, so it is the same head, not a lookalike.

NO FORWARD PASSES. Both heads are affine layers over the mean-pooled features that
`--zeroshot` already cached, so this runs in seconds and cannot contend with a GPU training job.

Usage:
    $py = "D:/sensevoice-gpu/.venv/Scripts/python.exe"
    & $py adapter/probe_pseudolabel_yield.py                     # uses the cuda feature cache
    & $py adapter/probe_pseudolabel_yield.py --feats-device cpu
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_emotion2vec import (  # noqa: E402
    CACHE_DIR,
    EMBED_DIM,
    Clips,
    load_core,
    load_split,
)
from probe_emotion import E2V_TO_SV, SV_CLASSES  # noqa: E402

# The mixes a pseudo-labelling run would actually meet. THAI-SER's own is included as the
# control: it is the only one where the measured numbers apply unprojected.
PRIORS = {
    "thaiser_acted_measured": {"NEUTRAL": .354, "HAPPY": .268, "ANGRY": .249, "SAD": .129},
    "mildly_neutral_70": {"NEUTRAL": .70, "HAPPY": .10, "ANGRY": .10, "SAD": .10},
    "spontaneous_85": {"NEUTRAL": .85, "HAPPY": .06, "ANGRY": .05, "SAD": .04},
    "read_corpus_95": {"NEUTRAL": .95, "HAPPY": .02, "ANGRY": .015, "SAD": .015},
}
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via rank statistic. Here: does higher confidence mean more often correct?

    0.5 means confidence carries NO information about correctness -- the case in which no
    threshold can help and the whole selective-prediction route is closed.
    """
    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties, or a head that emits identical confidences reads as informative
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1, n0 = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def conditional(gold, pred, keep, classes):
    """P(pred | true) restricted to the kept (confident) clips, as a dict of dicts.

    🔴 Denominator is ALL clips of that true class, not only the kept ones. A threshold that
    drops 90% of sad clips has not improved the teacher on SAD -- it has stopped answering -- and
    normalising by the survivors would hide exactly that.
    """
    out = {}
    for g in classes:
        idx = [i for i in range(len(gold)) if gold[i] == g]
        tot = len(idx)
        row = collections.Counter(pred[i] for i in idx if keep[i])
        out[g] = {p: (row.get(p, 0) / tot if tot else 0.0)
                  for p in classes + ["OTHER"]}
        out[g]["_abstained"] = (sum(1 for i in idx if not keep[i]) / tot) if tot else 0.0
    return out


def project(cond, prior, classes):
    """Precision and yield of each pseudo-label class under a given corpus prior."""
    out = {}
    for p in classes:
        num = sum(prior[g] * cond[g].get(p, 0.0) for g in classes)      # P(pred = p)
        tp = prior[p] * cond[p].get(p, 0.0)                              # P(true = p, pred = p)
        out[p] = {"precision": (tp / num if num else None),
                  "share_of_corpus_labelled_this": num,
                  "correct_per_100k": round(tp * 100000)}
    out["OTHER_unusable"] = sum(prior[g] * cond[g].get("OTHER", 0.0) for g in classes)
    out["kept_fraction"] = sum(
        prior[g] * sum(cond[g].get(p, 0.0) for p in classes + ["OTHER"]) for g in classes)
    return out


def per_class_thresholds(probs, pred, gold, classes, prior, bar):
    """One threshold PER CLASS, each chosen on dev.

    🔴 Why not a single global threshold: the classes do not fail together. A global rule
    is pinned by the worst class -- HAPPY needs t=0.99 before its projected precision clears the
    bar -- and at that threshold ANGRY, which was already clean at t=0.95, has stopped being
    emitted at all (49 correct per 100k against 1,569). One number is made to serve four very
    different decision boundaries, and three of them pay for the fourth.

    Each class gets the LOWEST threshold whose projected precision under `prior` clears `bar`,
    so no yield is discarded for nothing. Returns {class: threshold or None}; None means that
    class cannot be pseudo-labelled at this bar at any confidence, which is a finding.
    """
    conf = probs.max(axis=1)
    out = {}
    for c in classes:
        out[c] = None
        for t in THRESHOLDS:
            keep = [conf[i] >= t for i in range(len(pred))]
            cond = conditional(gold, pred, keep, classes)
            num = sum(prior[g] * cond[g].get(c, 0.0) for g in classes)
            tp = prior[c] * cond[c].get(c, 0.0)
            if num and (tp / num) >= bar:
                out[c] = t
                break
    return out


def apply_per_class(probs, pred, gold, classes, thresholds, prior):
    """Project the yield when each class uses its own dev-chosen threshold."""
    conf = probs.max(axis=1)
    keep = [thresholds.get(pred[i]) is not None and conf[i] >= thresholds[pred[i]]
            for i in range(len(pred))]
    cond = conditional(gold, pred, keep, classes)
    proj = project(cond, prior, classes)
    proj["coverage_measured"] = sum(keep) / len(keep)
    return proj


def fit_head(Z, y, tr, dev, seed=0, epochs=400, lr=1e-3):
    """The dev-selected 4-class head, refit exactly as the `head` rung fits it."""
    import torch
    import torch.nn.functional as F
    xt, yt = torch.from_numpy(Z[tr]), torch.from_numpy(y[tr])
    best = None
    for wd in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        torch.manual_seed(seed)
        h = torch.nn.Linear(Z.shape[1], len(SV_CLASSES))
        opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=wd)
        for ep in range(epochs):
            h.train(); opt.zero_grad()
            F.cross_entropy(h(xt), yt).backward(); opt.step()
            if (ep + 1) % max(1, epochs // 20) == 0 or ep == epochs - 1:
                with torch.no_grad():
                    acc = float((h(torch.from_numpy(Z[dev])).argmax(1).numpy() == y[dev]).mean())
                if best is None or acc > best[0]:
                    best = (acc, wd, ep + 1,
                            {k: v.detach().clone() for k, v in h.state_dict().items()})
    h = torch.nn.Linear(Z.shape[1], len(SV_CLASSES))
    h.load_state_dict(best[3])
    with torch.no_grad():
        return h(torch.from_numpy(Z)).numpy(), {"dev_accuracy": best[0], "weight_decay": best[1],
                                                "epoch": best[2]}


def analyse(name, probs, pred, gold_te, te, classes):
    """Selective prediction on the held-out actors, then the projection onto natural corpora."""
    conf = probs.max(axis=1)
    correct = np.array([int(pred[i] == gold_te[i]) for i in range(len(te))])
    res = {
        "n": len(te),
        "accuracy_at_full_coverage": float(correct.mean()),
        "confidence_auroc_for_correctness": auroc(conf, correct),
        "mean_confidence_when_right": float(conf[correct == 1].mean()),
        "mean_confidence_when_wrong": float(conf[correct == 0].mean()),
        "by_threshold": {},
        "projected": {},
    }
    for t in THRESHOLDS:
        keep = conf >= t
        n = int(keep.sum())
        acc = float(correct[keep].mean()) if n else None
        cond = conditional(gold_te, pred, keep, classes)
        res["by_threshold"][f"{t:g}"] = {
            "coverage": n / len(te), "n_kept": n,
            "precision_on_kept": acc,
            "per_class_kept_recall": {g: sum(cond[g].get(p, 0) for p in classes + ["OTHER"])
                                      for g in classes},
            "per_class_precision_on_kept": {
                p: (sum(1 for i in range(len(te)) if keep[i] and pred[i] == p and gold_te[i] == p)
                    / max(1, sum(1 for i in range(len(te)) if keep[i] and pred[i] == p)))
                if any(keep[i] and pred[i] == p for i in range(len(te))) else None
                for p in classes},
        }
        res["projected"][f"{t:g}"] = {pn: project(cond, pv, classes)
                                      for pn, pv in PRIORS.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats-device", default="cuda", choices=["cpu", "cuda"],
                    help="which cached feature file to read; they are equivalent (verified)")
    ap.add_argument("--out", default=os.environ.get("ADAPTER_OUT", "results"))
    a = ap.parse_args()

    import torch

    fp = CACHE_DIR / f"feats_e2v_mean_{a.feats_device}.npy"
    if not fp.exists():
        sys.exit(f"{fp} missing -- run finetune_emotion2vec.py --zeroshot first")
    feats = np.load(fp)

    clips = Clips()
    tr, dev, te, actors, test_actors = load_split(clips)
    gold = [r["gold"] for r in clips.meta]
    gold_te = [gold[i] for i in te]
    cls = {c: i for i, c in enumerate(SV_CLASSES)}
    y = np.array([cls[g] for g in gold], dtype=np.int64)
    t0 = time.time()

    # --- teacher A: the shipped 9-class head, applied to the same cached features ---------
    core, labels = load_core()
    with torch.no_grad():
        W = core.proj.weight.detach().cpu().numpy()
        b = core.proj.bias.detach().cpu().numpy()
    logits9 = feats @ W.T + b
    keep_cols = [k for k, l in enumerate(labels) if not str(l).startswith("unuse")]
    p9 = softmax(logits9[:, keep_cols])
    names9 = [E2V_TO_SV.get(str(labels[k]).split("/")[-1].strip().lower(),
                            str(labels[k]).split("/")[-1].strip().upper()) for k in keep_cols]
    arg = p9.argmax(axis=1)
    pred_zs = [names9[k] if names9[k] in SV_CLASSES else "OTHER" for k in arg]
    zs = analyse("zeroshot", p9[te], [pred_zs[i] for i in te], gold_te, te, SV_CLASSES)
    zs_dev = analyse("zeroshot", p9[dev], [pred_zs[i] for i in dev],
                     [gold[i] for i in dev], dev, SV_CLASSES)

    # --- teacher B: the dev-selected 4-class head -----------------------------------------
    mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-6
    Z = ((feats - mu) / sd).astype(np.float32)
    logits4, head_info = fit_head(Z, y, tr, dev)
    p4 = softmax(logits4)
    pred_ft = [SV_CLASSES[k] for k in p4.argmax(axis=1)]
    ft = analyse("finetuned", p4[te], [pred_ft[i] for i in te], gold_te, te, SV_CLASSES)
    ft_dev = analyse("finetuned", p4[dev], [pred_ft[i] for i in dev],
                     [gold[i] for i in dev], dev, SV_CLASSES)

    # \U0001f534 THE THRESHOLD IS CHOSEN ON DEV, then reported once on test -- the `A38`
    # discipline. Reading a threshold off the test curve and then quoting that curve's precision
    # is selection on the test set, and it would inflate exactly the number this arm exists to
    # produce. The rule: the lowest threshold whose DEV projection clears 0.75 precision on every
    # class under the 85%-neutral prior, so yield is not thrown away for nothing.
    def pick(dev_res):
        for t in THRESHOLDS:
            pj = dev_res["projected"][f"{t:g}"]["spontaneous_85"]
            vals = [pj[c]["precision"] for c in SV_CLASSES]
            if all(v is not None and v >= 0.75 for v in vals):
                return f"{t:g}"
        return None

    selection = {}
    BAR, PRI = 0.75, PRIORS["spontaneous_85"]
    for nm, dres, tres, probs, preds in (
            ("zeroshot", zs_dev, zs, p9, pred_zs),
            ("finetuned", ft_dev, ft, p4, pred_ft)):
        t = pick(dres)
        pct = per_class_thresholds(probs[dev], [preds[i] for i in dev],
                                   [gold[i] for i in dev], SV_CLASSES, PRI, BAR)
        selection[nm] = {
            "global": {
                "threshold_chosen_on_dev": t,
                "rule": "lowest t whose DEV projection >= 0.75 on ALL 4 classes @85% neutral",
                "dev_projection": dres["projected"][t]["spontaneous_85"] if t else None,
                "test_projection": tres["projected"][t]["spontaneous_85"] if t else None,
                "test_coverage": tres["by_threshold"][t]["coverage"] if t else None,
            },
            "per_class": {
                "thresholds_chosen_on_dev": pct,
                "rule": f"per class, lowest t whose DEV projection >= {BAR} @85% neutral",
                "dev_projection": apply_per_class(probs[dev], [preds[i] for i in dev],
                                                  [gold[i] for i in dev], SV_CLASSES, pct, PRI),
                "test_projection": apply_per_class(probs[te], [preds[i] for i in te],
                                                   gold_te, SV_CLASSES, pct, PRI),
            },
        }

    out = {
        "arm": "pseudolabel_yield",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": "does the teacher's confidence separate its right answers from its wrong "
                    "ones well enough to rescue pseudo-labelling on mostly-neutral speech?",
        "measured_on": "410 actor-held-out THAI-SER clips (acted); projections are OPTIMISTIC",
        "test_actors": test_actors,
        "feature_cache": str(fp),
        "finetuned_head": head_info,
        "priors": PRIORS,
        "threshold_selection": selection,
        "teachers": {"zeroshot": {"test": zs, "dev": zs_dev},
                     "finetuned": {"test": ft, "dev": ft_dev}},
        "seconds": round(time.time() - t0, 1),
    }
    dest = Path(a.out); dest.mkdir(parents=True, exist_ok=True)
    (dest / "pseudolabel_yield_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    for nm, r in (("zeroshot", zs), ("finetuned", ft)):
        print(f"\n=== {nm} === acc {r['accuracy_at_full_coverage']:.4f} | "
              f"confidence-AUROC-for-correctness {r['confidence_auroc_for_correctness']:.4f} "
              f"(0.5 = confidence tells you nothing) | conf right {r['mean_confidence_when_right']:.3f} "
              f"vs wrong {r['mean_confidence_when_wrong']:.3f}")
        print(f"{'thresh':>7s}{'coverage':>10s}{'prec@kept':>11s}"
              + "".join(f"{('P(' + c[:4] + ')'):>10s}" for c in SV_CLASSES))
        for t in THRESHOLDS:
            d = r["by_threshold"][f"{t:g}"]
            pr = d["per_class_precision_on_kept"]
            print(f"{t:7g}{d['coverage']:10.3f}{(d['precision_on_kept'] or 0):11.3f}"
                  + "".join(f"{(pr[c] if pr[c] is not None else float('nan')):10.3f}"
                            for c in SV_CLASSES))
        print(f"  projected onto an 85%-neutral corpus:")
        for t in THRESHOLDS:
            pj = r["projected"][f"{t:g}"]["spontaneous_85"]
            print(f"    t={t:<5g} kept {pj['kept_fraction']:.3f} | "
                  + " ".join(f"{c[:4]} {(pj[c]['precision'] or float('nan')):.3f}"
                             f"({pj[c]['correct_per_100k']:>5d})" for c in SV_CLASSES))
    print("\n=== thresholds chosen on DEV, confirmed once on TEST (85%-neutral corpus) ===")
    for nm, sel in selection.items():
        g = sel["global"]
        if g["threshold_chosen_on_dev"]:
            tt = g["test_projection"]
            print(f"  {nm:10s} GLOBAL t={g['threshold_chosen_on_dev']} "
                  f"coverage {g['test_coverage']:.3f} | test "
                  + " ".join(f"{c[:4]} {(tt[c]['precision'] or 0):.3f}"
                             f"({tt[c]['correct_per_100k']:>5d})" for c in SV_CLASSES))
        else:
            print(f"  {nm:10s} GLOBAL   -- no single threshold clears 0.75 on all four classes")
        pc = sel["per_class"]
        print(f"{'':13s}PER-CLASS t=" + " ".join(
            f"{c[:4]}:{(pc['thresholds_chosen_on_dev'][c] if pc['thresholds_chosen_on_dev'][c] is not None else 'never')}"
            for c in SV_CLASSES))
        d, t2 = pc["dev_projection"], pc["test_projection"]
        print(f"{'':13s}  dev  " + " ".join(f"{c[:4]} {(d[c]['precision'] or 0):.3f}"
                                            for c in SV_CLASSES))
        print(f"{'':13s}  test " + " ".join(f"{c[:4]} {(t2[c]['precision'] or 0):.3f}"
                                            f"({t2[c]['correct_per_100k']:>5d})"
                                            for c in SV_CLASSES)
              + f"  | kept {t2['coverage_measured']:.3f}")
    print(f"\n[done] -> {dest / 'pseudolabel_yield_metrics.json'}")


if __name__ == "__main__":
    main()
