"""probe_encoder_emotion.py -- WHICH frozen encoder actually carries Thai emotion?

SPIKE, 2026-09-19. Decides an architecture with a number instead of an argument.

THE DISAGREEMENT THIS SETTLES. To get Thai text + emotion + events from ONE encoder pass, the
three heads must share one encoder. Whose?
  - Typhoon's encoder knows Thai (11k h) but was never taught emotion. ASR training actively
    SUPPRESSES the variation emotion lives in -- speaker, register, affect are nuisance
    variables for transcription.
  - emotion2vec's encoder knows emotion (0.7381 zero-shot on Thai, measured 2026-09-19) but
    knows no Thai text.

🔴 **The Thai-specific sharpening, and it is his argument, not mine.** Thai tone is LEXICAL, so
Typhoon's encoder MUST preserve pitch CONTOUR or it cannot separate ขาว/ข่าว/ข้าว. But speaker
REGISTER -- where that contour sits and how wide it is -- is pure nuisance for transcription and
the encoder is rewarded for discarding it. Register is exactly the emotional signal. That is the
same split the design spec makes in decision `D5`, which means **the thing that makes Typhoon
good at Thai may be the thing that destroys its emotion information.**

THE INSTRUMENT. A LINEAR probe: freeze the encoder, pool its features, fit ONE affine layer to
predict the emotion. Deliberately too weak to be clever -- if a single layer can read emotion off
the features then the emotion is genuinely PRESENT in the representation, not reconstructed by
the probe. This is the standard representation-probing setup and it measures availability, not
achievable accuracy: a deeper head can only do better, so a LOW linear score is the informative
one.

FOUR ENCODERS, identical clips, identical labels, identical actor-disjoint split, identical
pooling. The encoder is the ONLY thing that varies.
  mel          -- 80-dim log-mel, no encoder at all. THE FLOOR. How much emotion is trivially in
                  the audio before any model touches it. Without this the other three are
                  uninterpretable.
  typhoon      -- the decisive one. ConformerEncoder, 17 layers, d_model 512, frozen.
  emotion2vec  -- the reference ceiling. Trained for exactly this.
  sensevoice   -- free third point. 70 SAN-M layers, 5 languages, no Thai.

POOLING. mean-and-std over time (statistics pooling), concatenated -> 2*D. Std matters: emotion
lives in how much a signal MOVES, and mean-only pooling throws that away.

PROTOCOL. Split by ACTOR, never by clip -- the same actor saying the same three scripted
sentences appears many times, so a clip-level split leaks the speaker and every number becomes
meaningless. Regularisation is chosen on a slice of the TRAIN actors and reported ONCE on the
held-out actors (the `A38` discipline the repo already uses for alpha).

TWO VENVS, because no single interpreter has both stacks:
    # NeMo side -- Typhoon + the mel floor
    & "E:/asr-spike/.venv/Scripts/python.exe" adapter/probe_encoder_emotion.py --extract typhoon,mel
    # funasr side -- emotion2vec + SenseVoice
    & "D:/sensevoice-spike/.venv/Scripts/python.exe" adapter/probe_encoder_emotion.py --extract emotion2vec,sensevoice
    # either side -- fits the probes on whatever .npz files exist
    & "E:/asr-spike/.venv/Scripts/python.exe" adapter/probe_encoder_emotion.py --fit
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
from probe_emotion import iter_kept, to_wav16k, SV_CLASSES  # noqa: E402  same 1,256-clip subset

FEAT_DIR = Path("D:/sensevoice-spike/feats")


# --- pooling ------------------------------------------------------------------------------

def pool(frames: np.ndarray) -> np.ndarray:
    """[T, D] -> [2D]. Mean AND std: emotion is in how much the signal moves, not only where
    it sits, and a mean-only pool discards precisely that."""
    if frames.ndim != 2 or frames.shape[0] == 0:
        return np.zeros(2 * (frames.shape[-1] if frames.ndim == 2 else 1), dtype=np.float32)
    return np.concatenate([frames.mean(0), frames.std(0)]).astype(np.float32)


# --- extractors ---------------------------------------------------------------------------

def make_mel():
    """THE FLOOR. 80-dim log-mel, 25 ms / 10 ms -- the same front end Typhoon uses, with no
    encoder behind it. Any encoder that cannot beat this has added nothing."""
    # librosa, not torchaudio: this extractor runs in E:/asr-spike (NeMo) where torchaudio is
    # absent and librosa 0.11.0 is present. Same 25 ms / 10 ms / 80-bin geometry as Typhoon's
    # own preprocessor, so the floor is measured on the front end the encoder actually sees.
    import librosa

    def fn(wav):
        m = librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=512,
                                           win_length=400, hop_length=160, n_mels=80)
        return np.log(m + 1e-6).T.astype(np.float32)   # [T, 80]
    return fn


def make_typhoon():
    """Frozen Typhoon ConformerEncoder. NeMo emits [B, D, T]; transposed to [T, D] here."""
    import torch
    from nemo.collections.asr.models import ASRModel
    # Same id and loader the repo's own server uses (asr/server.py:49,142) so this probe cannot
    # quietly measure a different checkpoint from the one every other arm scores -- the exact
    # mistake that burned two full runs on 2026-09-17.
    name = os.environ.get("ASR_MODEL", "typhoon-ai/typhoon-asr-realtime")
    m = ASRModel.from_pretrained(name, map_location="cpu")
    m.eval()
    m.freeze()
    print(f"[typhoon] {name} | encoder layers={len(m.encoder.layers)} "
          f"d_model={m.encoder.d_model}", flush=True)

    @torch.no_grad()
    def fn(wav):
        sig = torch.from_numpy(wav).float().unsqueeze(0)
        ln = torch.tensor([sig.shape[-1]])
        feats, flen = m.preprocessor(input_signal=sig, length=ln)
        enc, _ = m.encoder(audio_signal=feats, length=flen)
        return enc.squeeze(0).transpose(0, 1).cpu().numpy()   # [T, 512]
    return fn


def make_funasr(model_id: str, frame_granularity: bool):
    """emotion2vec exposes frame embeddings directly; SenseVoice needs its encoder called."""
    import torch
    os.environ.setdefault("HF_HOME", "D:/sensevoice-spike/models")
    from funasr import AutoModel
    am = AutoModel(model=model_id, hub="hf", device="cpu", disable_update=True)

    if frame_granularity:                                   # emotion2vec
        @torch.no_grad()
        def fn(wav):
            out = am.generate(wav, granularity="frame", extract_embedding=True,
                              disable_pbar=True)
            f = out[0].get("feats") if out else None
            f = np.asarray(f, dtype=np.float32) if f is not None else np.zeros((1, 1024), np.float32)
            return f if f.ndim == 2 else f.reshape(1, -1)
        return fn

    core = am.model                                          # SenseVoice: call the encoder

    @torch.no_grad()
    def fn(wav):
        sig = torch.from_numpy(wav).float().unsqueeze(0)
        ln = torch.tensor([sig.shape[-1]])
        feats, flen = am.kwargs["frontend"](sig, ln)
        enc, _ = core.encoder(feats.float(), flen)
        return enc.squeeze(0).cpu().numpy()                  # [T, 512]
    return fn


EXTRACTORS = {
    "mel": make_mel,
    "typhoon": make_typhoon,
    "emotion2vec": lambda: make_funasr("emotion2vec/emotion2vec_plus_large", True),
    "sensevoice": lambda: make_funasr("FunAudioLLM/SenseVoiceSmall", False),
}


def extract(names, shard_dir, mic, agreement_min, limit):
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    fns = {}
    for n in names:
        print(f"[load] {n} ...", flush=True)
        fns[n] = EXTRACTORS[n]()

    acc = {n: [] for n in names}
    meta, t0, i = [], time.time(), 0
    for rec, is_held in iter_kept(Path(shard_dir), mic, agreement_min, limit):
        if is_held is None or is_held:
            continue                                        # scored subset only
        wav, _ = to_wav16k(rec["audio"])
        for n, fn in fns.items():
            try:
                acc[n].append(pool(fn(wav)))
            except Exception as e:                          # never lose the whole run to one clip
                print(f"  !! {n} failed on {rec['audio_id']}: {e}", flush=True)
                acc[n].append(None)
        meta.append({"audio_id": rec["audio_id"], "gold": rec["gold"],
                     "actor_id": rec["actor_id"], "turn_type": rec["turn_type"],
                     "seconds": len(wav) / 16000.0})
        i += 1
        if i % 100 == 0:
            print(f"  ... {i} clips, {time.time() - t0:.0f}s", flush=True)

    for n in names:
        rows = acc[n]
        dim = next((len(r) for r in rows if r is not None), 0)
        X = np.stack([r if r is not None else np.zeros(dim, np.float32) for r in rows])
        ok = np.array([r is not None for r in rows])
        np.savez_compressed(FEAT_DIR / f"feats_{n}.npz", X=X, ok=ok,
                            meta=json.dumps(meta, ensure_ascii=False))
        print(f"[saved] {n}: X{X.shape}, {int(ok.sum())}/{len(ok)} ok "
              f"-> {FEAT_DIR / f'feats_{n}.npz'}", flush=True)


# --- the linear probe ----------------------------------------------------------------------

def split_by_actor(meta, holdout_every=3):
    """ACTOR-disjoint, deterministic. A clip-level split would leak: the same actor repeats the
    same three scripted sentences many times, so the probe would learn the speaker."""
    actors = sorted({m["actor_id"] for m in meta})
    test = {a for k, a in enumerate(actors) if k % holdout_every == 0}
    dev = {a for k, a in enumerate(actors) if k % holdout_every == 1}   # for choosing lambda
    return ([i for i, m in enumerate(meta) if m["actor_id"] not in test | dev],
            [i for i, m in enumerate(meta) if m["actor_id"] in dev],
            [i for i, m in enumerate(meta) if m["actor_id"] in test],
            actors, sorted(test))


def fit_linear(Xtr, ytr, Xev, n_cls, wd, epochs=300, seed=0):
    import torch
    torch.manual_seed(seed)
    lin = torch.nn.Linear(Xtr.shape[1], n_cls)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=wd)
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return lin(torch.from_numpy(Xev)).argmax(1).numpy()


def macro_recall(y, p, n_cls):
    rs = [float((p[y == c] == c).mean()) for c in range(n_cls) if (y == c).sum()]
    return sum(rs) / len(rs) if rs else 0.0


def fit_all(out_path: Path):
    files = sorted(FEAT_DIR.glob("feats_*.npz"))
    if not files:
        sys.exit(f"no feature files in {FEAT_DIR} -- run --extract first")

    cls = {c: i for i, c in enumerate(SV_CLASSES)}
    results, header = {}, None
    for f in files:
        name = f.stem.replace("feats_", "")
        d = np.load(f, allow_pickle=True)
        X, ok = d["X"].astype(np.float32), d["ok"]
        meta = json.loads(str(d["meta"]))
        tr, dev, te, actors, test_actors = split_by_actor(meta)
        tr = [i for i in tr if ok[i]]; dev = [i for i in dev if ok[i]]; te = [i for i in te if ok[i]]
        y = np.array([cls[m["gold"]] for m in meta], dtype=np.int64)

        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6         # standardise on TRAIN only
        Z = (X - mu) / sd

        best = None                                         # choose lambda on dev actors
        for wd in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            p = fit_linear(Z[tr], y[tr], Z[dev], len(cls), wd)
            a = float((p == y[dev]).mean())
            if best is None or a > best[1]:
                best = (wd, a)
        wd = best[0]
        pt = fit_linear(Z[tr], y[tr], Z[te], len(cls), wd)   # report ONCE on test actors
        acc = float((pt == y[te]).mean())
        n = len(te)
        se = (acc * (1 - acc) / n) ** 0.5
        maj = float(max((y[te] == c).sum() for c in range(len(cls))) / n)

        results[name] = {
            "dim": int(X.shape[1]), "n_train": len(tr), "n_dev": len(dev), "n_test": n,
            "weight_decay_chosen_on_dev": wd, "dev_accuracy": best[1],
            "test_accuracy": acc, "ci95": [max(0., acc - 1.96 * se), min(1., acc + 1.96 * se)],
            "test_macro_recall": macro_recall(y[te], pt, len(cls)),
            "test_majority_class": maj,
            "beats_majority_class": acc > maj,
            "per_class_recall": {c: (float((pt[y[te] == i] == i).mean())
                                     if (y[te] == i).sum() else None)
                                 for c, i in cls.items()},
            "feature_failures": int((~ok).sum()),
        }
        header = {"actors_total": len(actors), "test_actors": test_actors}
        r = results[name]
        print(f"[{name:12s}] dim {r['dim']:5d} | test acc {acc:.4f} "
              f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] | macro-recall "
              f"{r['test_macro_recall']:.4f} | maj-class {maj:.4f} | wd {wd}", flush=True)

    floor = results.get("mel", {}).get("test_accuracy")
    ref = results.get("emotion2vec", {}).get("test_accuracy")
    typ = results.get("typhoon", {}).get("test_accuracy")
    verdict = None
    if floor is None or typ is None:
        verdict = "INCOMPLETE -- need both the mel floor and typhoon before any verdict"
    elif typ <= floor + 0.02:
        verdict = ("typhoon encoder is AT THE MEL FLOOR -- ASR training discarded the emotion, "
                   "and a frozen-Typhoon head would be reading raw acoustics. TWO ENCODERS.")
    elif ref is None:
        verdict = (f"typhoon ({typ:.4f}) is above the mel floor ({floor:.4f}) -- its encoder does "
                   "carry emotion. NO CEILING YET: emotion2vec not measured, so this cannot say "
                   "how much is missing.")
    elif typ >= ref - 0.05:
        verdict = ("typhoon carries emotion comparably to emotion2vec -- ONE ENCODER, one pass, "
                   "small head.")
    else:
        verdict = (f"typhoon ({typ:.4f}) sits between the mel floor ({floor:.4f}) and emotion2vec "
                   f"({ref:.4f}). One pass remains viable but the head must be non-linear, and "
                   "the gap to emotion2vec is the price of the single encoder.")

    out = {"probe": "which-frozen-encoder-carries-thai-emotion",
           "date": time.strftime("%Y-%m-%d %H:%M:%S"),
           "instrument": "linear probe (single affine layer) on mean+std pooled frozen features",
           "split": "actor-disjoint; lambda chosen on dev actors, reported once on test actors",
           **(header or {}), "encoders": results, "verdict": verdict}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nVERDICT: {verdict}\n[done] -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", default=None,
                    help="comma list: mel,typhoon,emotion2vec,sensevoice")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--shards", default="D:/thai-ser")
    ap.add_argument("--mic", default="mic_clip")
    ap.add_argument("--agreement-min", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.environ.get("ADAPTER_OUT", "results"))
    a = ap.parse_args()

    if a.extract:
        extract([s.strip() for s in a.extract.split(",") if s.strip()],
                a.shards, a.mic, a.agreement_min, a.limit)
    if a.fit:
        fit_all(Path(a.out) / "probe_encoder_emotion.json")
    if not a.extract and not a.fit:
        ap.error("pass --extract and/or --fit")


if __name__ == "__main__":
    main()
