"""probe_natural_prior.py -- what IS the emotion prior of natural Thai speech?

`F19` computed what pseudo-labelling buys on a mostly-neutral corpus, and every number in it
rests on an **assumed** 85%-neutral prior. Nobody has measured the emotion prior of any natural
Thai corpus, and measuring it with the teacher alone is circular -- the teacher's own errors
move the answer. This arm does the non-circular half and says exactly where the circle remains.

THE CORPUS. Common Voice 17 Thai, `test` split, the only natural Thai audio on this box
(`D:/thai-ser` is acted; LOTUSDIS audio is NOT downloaded -- 9 KB of metadata only, checked
2026-09-22). 11,042 clips, read speech from volunteers.
\u26a0 **Read speech is the EXTREME neutral end, not a typical natural corpus.** Someone reading a
prompt aloud is less expressive than someone talking. So this measures a LOWER bound on
emotional content and an UPPER bound on neutral share. Conversational Thai will sit between
this and THAI-SER. Quote it as "read speech", never as "natural speech" unqualified.

THE TWO ESTIMATES, and they answer different questions.
  1. **What the teacher SAYS** -- the raw predicted label distribution. This is what a naive
     pseudo-labelling run would actually write to disk, so it is the operationally real number
     even though it is biased.
  2. **What the prior probably IS** -- the predicted distribution corrected by the teacher's
     measured confusion matrix (black-box shift estimation: solve `C^T p_true = p_pred`, where
     `C[g][p] = P(pred=p | true=g)` measured on the 410 held-out THAI-SER clips).
     \U0001f534 **This correction assumes `C` transfers from ACTED studio speech to READ speech, and
     `F18` already measured that it does not fully** (impro 0.804 vs script 0.903, and the gap
     widened under fine-tuning). So estimate 2 is indicative, not authoritative. The only way to
     close it is human labels on a sample -- which is the point of `--sample-for-listening`.

THE THIRD OUTPUT, which is the one that decides the plan: **actual yield at the per-class
confidence thresholds `F19` chose on dev.** Not projected onto a hypothetical corpus -- counted
on real clips. "How many usable HAPPY labels does Common Voice Thai actually contain?"

Usage:
    $py = "D:/sensevoice-gpu/.venv/Scripts/python.exe"
    & $py adapter/probe_natural_prior.py --limit 2000 --device cuda
    & $py adapter/probe_natural_prior.py --limit 2000 --device cuda --sample-for-listening 40
"""

from __future__ import annotations

import argparse
import collections
import io as _io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_emotion2vec as FT  # noqa: E402
from finetune_emotion2vec import Clips, load_core, load_split  # noqa: E402
from probe_emotion import E2V_TO_SV, SV_CLASSES  # noqa: E402
from probe_pseudolabel_yield import fit_head, softmax  # noqa: E402

CV_GLOB = ("C:/Users/MARU/.cache/huggingface/hub/"
           "datasets--fsicoli--common_voice_17_0/snapshots/*/")
# The per-class thresholds `F19` selected on the THAI-SER dev actors. Reused verbatim: choosing
# new ones here would be selection on the very corpus whose yield is being measured.
F19_THRESHOLDS = {"NEUTRAL": 0.0, "HAPPY": 0.99, "ANGRY": 0.6, "SAD": 0.9}


def cv_root() -> Path:
    import glob
    hits = sorted(glob.glob(CV_GLOB))
    if not hits:
        sys.exit(f"Common Voice cache not found at {CV_GLOB}")
    return Path(hits[0])


def iter_cv_clips(limit: int):
    """Stream (clip_id, 16 kHz mono float32) out of the Common Voice test tar.

    Decoded with soundfile (libsndfile >= 1.1 reads mp3) and resampled with torchaudio, the same
    pair `to_wav16k` uses, so the front end matches every other arm in this repo.
    """
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    tar_path = cv_root() / "audio/th/test/th_test_0.tar"
    if not tar_path.exists():
        sys.exit(f"{tar_path} missing")
    with tarfile.open(tar_path) as t:
        n = 0
        for mem in t:
            if not mem.isfile() or not mem.name.endswith(".mp3"):
                continue
            raw = t.extractfile(mem).read()
            try:
                data, sr = sf.read(_io.BytesIO(raw), dtype="float32", always_2d=True)
            except Exception as e:
                print(f"  !! decode failed {mem.name}: {e}", flush=True)
                continue
            wav = data.mean(axis=1)
            if sr != 16000:
                wav = AF.resample(torch.from_numpy(wav), sr, 16000).numpy()
            yield Path(mem.name).name, np.ascontiguousarray(wav, dtype=np.float32)
            n += 1
            if limit and n >= limit:
                return


def measured_confusion():
    """C[g][p] = P(pred = p | true = g) for the 4-class head, on the 410 held-out clips.

    Read from the recorded run rather than recomputed, so this arm cannot silently disagree with
    the confusion matrix `F19` was built on.
    """
    m = json.loads(Path("results/finetune_e2v_head_metrics.json").read_text(encoding="utf-8"))
    rows = collections.defaultdict(lambda: collections.Counter())
    for k, v in m["test"]["confusion"].items():
        g, p = k.split("->")
        rows[g][p] += v
    C = np.zeros((len(SV_CLASSES), len(SV_CLASSES)))
    for i, g in enumerate(SV_CLASSES):
        tot = sum(rows[g].values())
        for j, p in enumerate(SV_CLASSES):
            C[i, j] = rows[g][p] / tot if tot else 0.0
    return C


def correct_prior(C: np.ndarray, q: np.ndarray):
    """Black-box shift estimation: solve C^T p_true = p_pred.

    Returns (estimate, note). Negative entries mean the linear system has no valid probability
    solution -- which is itself informative: it says the observed prediction mix is not
    reachable from ANY true prior under this confusion matrix, i.e. the matrix does not transfer.
    """
    try:
        raw = np.linalg.solve(C.T, q)
    except np.linalg.LinAlgError:
        return None, "confusion matrix is singular -- no correction possible"
    note = None
    if (raw < -1e-6).any():
        note = ("\U0001f534 negative components before clipping -- the observed prediction mix is "
                "NOT reachable from any valid prior under the acted-speech confusion matrix, so "
                "the matrix does not transfer to this corpus. Treat as a warning, not a prior.")
    clipped = np.clip(raw, 0, None)
    s = clipped.sum()
    return (clipped / s if s > 0 else None), note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--feats-device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--sample-for-listening", type=int, default=0,
                    help="write N clip ids per predicted class for a human to check")
    ap.add_argument("--out", default=os.environ.get("ADAPTER_OUT", "results"))
    a = ap.parse_args()

    import torch
    FT.DEVICE = a.device
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda but torch.cuda.is_available() is False; use D:/sensevoice-gpu")

    # The 4-class head, refit exactly as `F18`'s head rung and `F19` fit it.
    feats_ts = np.load(FT.CACHE_DIR / f"feats_e2v_mean_{a.feats_device}.npy")
    clips_ts = Clips()
    tr, dev, te, _, _ = load_split(clips_ts)
    cls = {c: i for i, c in enumerate(SV_CLASSES)}
    y = np.array([cls[r["gold"]] for r in clips_ts.meta], dtype=np.int64)
    mu, sd = feats_ts[tr].mean(0), feats_ts[tr].std(0) + 1e-6
    Z = ((feats_ts - mu) / sd).astype(np.float32)
    _, head_info = fit_head(Z, y, tr, dev)
    import torch.nn as nn
    head = nn.Linear(FT.EMBED_DIM, len(SV_CLASSES))
    # refit once more and keep the module (fit_head returns logits, not the layer)
    from probe_pseudolabel_yield import fit_head as _fh  # noqa: F811
    logits_ts, _ = _fh(Z, y, tr, dev)
    # recover the affine map by least squares on the standardised features -- exact, since the
    # head IS affine: [Z 1] @ W = logits
    A = np.concatenate([Z, np.ones((Z.shape[0], 1), np.float32)], 1)
    W, *_ = np.linalg.lstsq(A, logits_ts, rcond=None)

    core, labels = load_core()
    core.eval()
    keep_cols = [k for k, l in enumerate(labels) if not str(l).startswith("unuse")]
    names9 = [E2V_TO_SV.get(str(labels[k]).split("/")[-1].strip().lower(),
                            str(labels[k]).split("/")[-1].strip().upper()) for k in keep_cols]

    ids, p4_all, p9_all, secs = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        Wp = core.proj.weight.detach().cpu().numpy()
        bp = core.proj.bias.detach().cpu().numpy()
        for cid, wav in iter_cv_clips(a.limit):
            f = FT.encode(core, wav).mean(dim=1).squeeze(0).cpu().numpy()
            z = (f - mu) / sd
            p4_all.append(softmax((np.concatenate([z, [1.0]]) @ W)[None, :])[0])
            p9_all.append(softmax((f @ Wp.T + bp)[None, keep_cols])[0])
            ids.append(cid)
            secs.append(len(wav) / 16000.0)
            if len(ids) % 250 == 0:
                print(f"  ... {len(ids)} clips, {time.time() - t0:.0f}s", flush=True)

    p4 = np.stack(p4_all)
    p9 = np.stack(p9_all)
    pred4 = [SV_CLASSES[k] for k in p4.argmax(1)]
    pred9 = [names9[k] for k in p9.argmax(1)]
    conf4 = p4.max(1)

    n = len(ids)
    dist4 = {c: sum(1 for p in pred4 if p == c) / n for c in SV_CLASSES}
    dist9 = {c: v / n for c, v in collections.Counter(pred9).items()}

    C = measured_confusion()
    q = np.array([dist4[c] for c in SV_CLASSES])
    est, note = correct_prior(C, q)

    # The number that decides the plan: real clips surviving F19's dev-chosen thresholds.
    yield_rows = {}
    for c in SV_CLASSES:
        t = F19_THRESHOLDS[c]
        kept = [i for i in range(n) if pred4[i] == c and conf4[i] >= t]
        yield_rows[c] = {"threshold": t, "n_predicted": sum(1 for p in pred4 if p == c),
                         "n_kept": len(kept), "share_of_corpus": len(kept) / n}

    out = {
        "arm": "natural_prior",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": "what is the emotion prior of natural (read) Thai speech, and what does "
                    "pseudo-labelling it actually yield?",
        "corpus": {"name": "Common Voice 17 Thai, test split", "clips_scored": n,
                   "audio_hours": round(sum(secs) / 3600, 3),
                   "caveat": "READ speech -- the extreme neutral end, not conversational Thai"},
        "teacher": {"head": "4-class, refit from the THAI-SER split", **head_info},
        "what_the_teacher_says": {"finetuned_4class": dist4, "shipped_9class": dist9},
        "mean_confidence": float(conf4.mean()),
        "corrected_prior_estimate": (None if est is None else
                                     {c: float(v) for c, v in zip(SV_CLASSES, est)}),
        "corrected_prior_note": note,
        "f19_assumed_prior": {"NEUTRAL": .85, "HAPPY": .06, "ANGRY": .05, "SAD": .04},
        "actual_yield_at_f19_thresholds": yield_rows,
        "seconds": round(time.time() - t0, 1),
    }

    if a.sample_for_listening:
        per = max(1, a.sample_for_listening // len(SV_CLASSES))
        rng = np.random.default_rng(0)
        samp = {}
        for c in SV_CLASSES:
            idx = [i for i in range(n) if pred4[i] == c and conf4[i] >= F19_THRESHOLDS[c]]
            pick = rng.choice(idx, size=min(per, len(idx)), replace=False) if idx else []
            samp[c] = [{"clip": ids[i], "confidence": float(conf4[i])} for i in pick]
        out["sample_for_human_check"] = samp
        out["sample_note"] = ("\U0001f534 THE ONLY NON-CIRCULAR VALIDATION AVAILABLE. Listen to these "
                              "and mark the emotion by ear. The corrected prior above is an "
                              "estimate from an acted-speech confusion matrix; these clips are "
                              "ground truth.")

    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    (d / "natural_prior_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    # Per-clip hypotheses, the repo's flat convention. Needed by anything that joins these
    # predictions back to the transcripts -- e.g. the audio-vs-text agreement test, which asks
    # whether an audio-only SAD call lands on text that is independently sad.
    with (d / "natural_prior_hyps.jsonl").open("w", encoding="utf-8") as f:
        for i, cid in enumerate(ids):
            f.write(json.dumps({
                "clip": cid, "pred": pred4[i], "confidence": float(conf4[i]),
                "probs": {c: float(p4[i][k]) for k, c in enumerate(SV_CLASSES)},
                "pred_shipped9": pred9[i], "seconds": round(secs[i], 3),
                "kept_at_f19_threshold": bool(conf4[i] >= F19_THRESHOLDS[pred4[i]]),
            }, ensure_ascii=False) + "\n")

    print(f"\n=== Common Voice 17 Thai (READ speech), {n} clips, "
          f"{out['corpus']['audio_hours']:.2f} h ===")
    print("what the FINE-TUNED teacher says:  "
          + "  ".join(f"{c} {dist4[c]:.3f}" for c in SV_CLASSES))
    print("what the SHIPPED 9-class head says: "
          + "  ".join(f"{c} {v:.3f}" for c, v in sorted(dist9.items(), key=lambda x: -x[1])))
    print(f"\nF19 ASSUMED prior:    " + "  ".join(f"{c} {v:.3f}"
          for c, v in out["f19_assumed_prior"].items()))
    if est is not None:
        print("confusion-CORRECTED:  " + "  ".join(f"{c} {v:.3f}"
              for c, v in zip(SV_CLASSES, est)))
    if note:
        print("  " + note)
    print(f"\nACTUAL yield at F19's dev-chosen thresholds, on {n} real clips:")
    for c in SV_CLASSES:
        r = yield_rows[c]
        print(f"  {c:8s} t={r['threshold']:<5g} predicted {r['n_predicted']:5d} -> "
              f"kept {r['n_kept']:5d}  ({r['share_of_corpus']*100:5.2f}% of corpus)")
    print(f"\n[done] -> {d / 'natural_prior_metrics.json'}")


if __name__ == "__main__":
    main()
