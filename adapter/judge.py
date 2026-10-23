#!/usr/bin/env python3
"""Toward CER 0.060 (`F42`): a TRAINED judge that picks among the three ears' transcripts + their vote.

    PYTHONUTF8=1 python adapter/judge.py build      # LOTUSDIS train+test clips joined into ~10 s turns
    PYTHONUTF8=1 python adapter/judge.py ears       # NECTEC, then Pathumma-L / typhoon-L speculative (NECTEC draft)
    PYTHONUTF8=1 python adapter/judge.py score      # every ear scores every candidate (poe_rescore scorers)
    PYTHONUTF8=1 python adapter/judge.py train      # small MLP, expected-error loss, LOTUSDIS-held-out early stop
    PYTHONUTF8=1 python adapter/judge.py apply --half dev | --half test --final

WHY. On the benchmark dev half the per-utterance oracle over {NECTEC, Pathumma-L, typhoon-L} is -24.5%, but
untrained judges (vote, LM, product of experts) reach only -2 to -3%: each ear prefers its own text. A judge
that has SEEN many (ears' outputs, truth) pairs can learn when each ear is wrong. It must learn that on
audio that is NOT the benchmark (CC-BY-NC, evaluation only): LOTUSDIS (CC-BY-SA, leak-guarded against the
benchmark), joined into benchmark-like ~10 s turns because its clips are 2 s. ⚠️ NECTEC has likely seen part of
LOTUSDIS train (`F30` follow-up: CER 0.124 there vs 0.205 on test) -- so the judge may over-trust NECTEC.

Candidates per turn: nec, patL, twL, rover. Features per candidate (relative within the turn): per-token and
total log-likelihood under each of the 3 models, minus the turn max; length vs the turn median; mean
normalised edit distance to the other candidates; which candidate it is. Loss: expected errors
sum_c softmax(score)_c * err_c / ref_len -- the quantity CER counts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import os
# 2026-10-09: JUDGE_DATA=gs2 -> GigaSpeech2-th dev+test (fresh to ALL three ears, `F43` follow-up);
# default = the LOTUSDIS build that taught the judge NECTEC's prior (`F43`).
DATA = os.environ.get("JUDGE_DATA", "lotusdis")
D = Path("E:/Claude/datasets/judge" if DATA == "lotusdis" else f"E:/Claude/datasets/judge_{DATA}")
GS2 = Path("E:/Claude/datasets/gigaspeech2_th")
LOT = Path("E:/Claude/datasets/lotusdis_lav")
CANDS = ["nec", "patL", "twL", "rover"]
MODELS = ["nec", "patL", "twL"]


def cmd_build(a):
    (D / "wav").mkdir(parents=True, exist_ok=True)
    rows = []
    if DATA == "gs2":
        for split in ("dev", "test"):
            rr = [json.loads(l) for l in open(GS2 / f"manifest_{split}.jsonl", encoding="utf-8")]
            # id "<video>-<segment>": same video + consecutive segment = the next stretch of the same talk
            rr.sort(key=lambda r: (int(r["id"].split("-")[0]), int(r["id"].split("-")[1])))
            rows += [dict(r, split=f"{split}v{r['id'].split('-')[0]}") for r in rr]
    else:
        for split in ("train", "test"):
            rows += [dict(json.loads(l), split=split) for l in open(LOT / f"manifest_{split}_conv.jsonl", encoding="utf-8")]
    num = lambda r: int(r["id"].rsplit("-", 1)[1])
    out, cur, dur = [], [], 0.0
    sil = np.zeros(int(0.15 * 16000), dtype=np.float32)

    def flush():
        nonlocal cur, dur
        if cur and dur >= 4.0:
            ys = []
            for r in cur:
                y, sr = sf.read(r["audio_filepath"], dtype="float32")
                ys += [y.mean(axis=1) if y.ndim > 1 else y, sil]
            uid = f"{cur[0]['split']}-{num(cur[0])}-{num(cur[-1])}"
            p = D / "wav" / f"{uid}.wav"
            sf.write(p, np.concatenate(ys[:-1]), 16000)
            out.append({"id": uid, "audio_filepath": str(p), "duration": round(dur, 2),
                        "text": "".join(r["text"].replace(" ", "") for r in cur), "split": cur[0]["split"]})
        cur, dur = [], 0.0

    prev = None
    for r in rows:
        if prev is not None and (r["split"] != prev["split"] or num(r) - num(prev) > 2 or dur + r["duration"] > 14.0):
            flush()
        cur.append(r); dur += r["duration"]
        if dur >= 8.0 + random.Random(num(r)).random() * 4:
            flush()
        prev = r
    flush()
    random.Random(0).shuffle(out)
    for i, r in enumerate(out):
        r["fold"] = "val" if i % 10 == 0 else "train"
    with open(D / "manifest.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    d = [r["duration"] for r in out]
    print(f"{len(out)} turns, {sum(d) / 3600:.1f} h, median {np.median(d):.1f} s")


def load_manifest():
    return [json.loads(l) for l in open(D / "manifest.jsonl", encoding="utf-8")]


def audio(r):
    from ctc_head import pcm16
    y, _ = sf.read(r["audio_filepath"], dtype="float32")
    return pcm16(y.mean(axis=1) if y.ndim > 1 else y)


def cmd_ears(a):
    rows = load_manifest()
    out = D / "ears.json"
    E = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if "nec" not in E:
        from ctc_head import load_nec
        m = load_nec()
        E["nec"] = {}
        for i in range(0, len(rows), 16):
            b = rows[i:i + 16]
            m.eval()
            hs = m.transcribe([audio(r) for r in b], verbose=False, batch_size=16)
            for r, h in zip(b, hs):
                E["nec"][r["id"]] = (h if isinstance(h, str) else str(getattr(h, "text", ""))).strip()
        del m; torch.cuda.empty_cache()
        out.write_text(json.dumps(E, ensure_ascii=False), encoding="utf-8")
        print("nec done", flush=True)
    from spec_whisper import Spec
    for k, name in (("patL", "nectec/Pathumma-whisper-th-large-v3"), ("twL", "typhoon-ai/typhoon-whisper-large-v3")):
        if len(E.get(k, {})) == len(rows):
            continue
        S = Spec(name)
        E.setdefault(k, {})                       # 2026-10-08: resume a partial model (the run was reaped twice)
        for i, r in enumerate(rows):
            if r["id"] in E[k]:
                continue
            with torch.no_grad():
                E[k][r["id"]], _ = S.spec_cached(S.encode(S.features(audio(r))), E["nec"][r["id"]])
            if i % 200 == 0:
                out.write_text(json.dumps(E, ensure_ascii=False), encoding="utf-8")   # checkpoint
                print(k, i, len(rows), flush=True)
        del S; torch.cuda.empty_cache()
        out.write_text(json.dumps(E, ensure_ascii=False), encoding="utf-8")
        print(k, "done", flush=True)
        # 2026-10-09: one large model per PROCESS -- loading the second while the first's RAM was not yet
        # returned got the run reaped for low memory at the handover. The caller runs `ears` again.
        return


def cmd_score(a):
    from pilot_rover import join, rover, toks
    from poe_rescore import nec_scores, whisper_scores, WHISPER
    rows = load_manifest()
    E = json.loads((D / "ears.json").read_text(encoding="utf-8"))
    cands = {}
    for r in rows:
        c = {k: E[k][r["id"]] for k in MODELS}
        merged, _ = rover({k: toks(c[k]) for k in MODELS}, {k: 1.0 for k in MODELS}, MODELS)
        c["rover"] = join(merged)
        cands[r["id"]] = c

    class Lazy:
        """2026-10-08: the first version held all 4,314 turns' audio in RAM (~3 GB) next to a large
        model and the run was reaped for low memory. Read each turn's audio only when it is scored."""
        def __init__(self, sub):
            self.sub = sub

        def __iter__(self):
            for r in self.sub:
                yield audio(r), [cands[r["id"]][k] for k in CANDS]

    # 2026-10-09: a scoring PROCESS still grew until the box ran short (reaped mid-Pathumma on 7,099 turns),
    # so each process now scores at most SLICE turns of one model, checkpoints, and exits; the caller loops
    # `score` until it prints "ALL SCORED".
    SLICE = 1500
    part = D / "scores_partial.json"
    sc = json.loads(part.read_text(encoding="utf-8")) if part.exists() else {}
    for k in MODELS:
        done = len(sc.get(k, []))
        if done >= len(rows):
            continue
        sub = rows[done: done + SLICE]
        new = nec_scores(Lazy(sub)) if k == "nec" else whisper_scores(WHISPER[k], Lazy(sub))
        sc[k] = sc.get(k, []) + new
        part.write_text(json.dumps(sc), encoding="utf-8")
        print(k, "scored", len(sc[k]), "of", len(rows), flush=True)
        return
    print("ALL SCORED", flush=True)
    res = {r["id"]: {"cands": cands[r["id"]], "scores": {m: dict(zip(CANDS, sc[m][i])) for m in sc}}
           for i, r in enumerate(rows)}
    (D / "scores.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print("->", D / "scores.json")


# ---------------------------------------------------------------------------- features + model
def _ed(a, b):
    from paired_vs import lev
    return lev(a, b) / max(len(a), len(b), 1)


def features(d) -> np.ndarray:
    """[4 candidates, F] for one turn."""
    from paired_vs import n
    texts = {c: n(d["cands"][c]) for c in CANDS}
    F = []
    for m in MODELS:
        tot = np.array([d["scores"][m][c][0] for c in CANDS], dtype=np.float64)
        per = np.array([d["scores"][m][c][0] / max(d["scores"][m][c][1], 1) for c in CANDS])
        F += [np.clip(tot - tot.max(), -50, 0) / 10, np.clip(per - per.max(), -5, 0)]
    L = np.array([len(texts[c]) for c in CANDS], dtype=np.float64)
    F.append((L - np.median(L)) / max(np.median(L), 1))
    F.append(np.array([np.mean([_ed(texts[c], texts[o]) for o in CANDS if o != c]) for c in CANDS]))
    X = np.stack(F, 1)
    return np.concatenate([X, np.eye(len(CANDS))], 1).astype(np.float32)


class Judge(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(f, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))

    def forward(self, x):                      # [B, 4, F] -> [B, 4]
        return self.net(x).squeeze(-1)


def cmd_train(a):
    from paired_vs import lev, n
    rows = {r["id"]: r for r in load_manifest()}
    S = json.loads((D / "scores.json").read_text(encoding="utf-8"))
    X, Y, fold = [], [], []
    for u, d in S.items():
        ref = n(rows[u]["text"])
        X.append(features(d))
        Y.append([lev(ref, n(d["cands"][c])) / max(len(ref), 1) for c in CANDS])
        fold.append(rows[u]["fold"])
    X, Y = torch.tensor(np.stack(X)), torch.tensor(Y, dtype=torch.float32)
    tr = torch.tensor([f == "train" for f in fold]); va = ~tr
    for c, k in enumerate(CANDS):
        print(f"  always {k:6s}: val mean CER {Y[va, c].mean():.4f}")
    print(f"  oracle      : val mean CER {Y[va].min(1).values.mean():.4f}")
    torch.manual_seed(0)
    J = Judge(X.shape[2])
    opt = torch.optim.AdamW(J.parameters(), lr=3e-3, weight_decay=1e-3)
    best, state = 9.0, None
    for ep in range(300):
        J.train()
        loss = (torch.softmax(J(X[tr]), -1) * Y[tr]).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        J.eval()
        with torch.no_grad():
            pick = J(X[va]).argmax(-1)
            v = Y[va].gather(1, pick[:, None]).mean().item()
        if v < best:
            best, state = v, {k: t.clone() for k, t in J.state_dict().items()}
        if ep % 50 == 0:
            print(f"ep {ep} train exp-err {loss.item():.4f}  val argmax CER {v:.4f}", flush=True)
    ck = ROOT / ("checkpoints/judge_v1.pt" if DATA == "lotusdis" else f"checkpoints/judge_{DATA}.pt")
    torch.save({"state": state, "f": X.shape[2], "cands": CANDS, "val_cer": best, "data": DATA}, ck)
    print("best val argmax CER", round(best, 4), "->", ck)


def cmd_apply(a):
    split = json.loads((ROOT / "results/split_dev_test.json").read_text(encoding="utf-8"))
    if a.half == "test" and not a.final:
        sys.exit("refused: the test half is scored once, with --final")
    P = json.loads((ROOT / "results/poe_scores.json").read_text(encoding="utf-8"))["utts"]
    ck = torch.load(ROOT / ("checkpoints/judge_v1.pt" if DATA == "lotusdis" else f"checkpoints/judge_{DATA}.pt"))
    J = Judge(ck["f"]); J.load_state_dict(ck["state"]); J.eval()
    picks = {}
    with (ROOT / "results" / f"judge_{'v1' if DATA == 'lotusdis' else DATA}_{a.half}_hyps.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for u in split[a.half]:
            with torch.no_grad():
                c = CANDS[int(J(torch.tensor(features(P[u]))[None]).argmax())]
            picks[c] = picks.get(c, 0) + 1
            f.write(json.dumps({"utt_id": u, "condition": "none", "text": P[u]["cands"][c]}, ensure_ascii=False) + "\n")
    print("picks", picks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "ears", "score", "train", "apply"))
    ap.add_argument("--half", choices=("dev", "test"), default="dev")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    {"build": cmd_build, "ears": cmd_ears, "score": cmd_score, "train": cmd_train, "apply": cmd_apply}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
