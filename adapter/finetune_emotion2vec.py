"""finetune_emotion2vec.py -- does fine-tuning emotion2vec on THAI-SER beat its zero-shot score?

Phase 2 step 1 of `docs/superpowers/specs/2026-09-19-sensevoice-thai-design.md` §0. It is the
TEACHER in `S13`: everything downstream -- pseudo-labelling Thai audio, distilling into
SenseVoice's emotion slot -- is bounded by how good this one model is. So it is measured first.

🔴 THE NUMBER TO BEAT IS 0.7317, on the 410 clips of the eight held-out actors. Anything scored
on a different denominator is not a comparison. The full ladder this script fills in:

    majority class (always guess NEUTRAL)          0.3537
    emotion2vec_plus_large ZERO-SHOT               0.7317   <- the bar
    linear probe, mean+std pooled frozen feats     0.7707   (2026-09-19, probe_encoder_emotion)
    THAI-SER's own trained CNN+LSTM                WA 65-70 (their paper, 8-fold speaker-indep CV)
    human ceiling on this subset                   0.9286

THE RUNGS, and why there is more than one. "Fine-tune" is not one thing, and the interesting
question is WHERE the gain comes from. Each rung changes exactly one thing from the rung below:

  zeroshot  the shipped 9-class `proj` head, argmax mapped to our 4 classes. The bar.
  head      backbone FROZEN, a fresh 4-class head on the same mean-pooled frames. Isolates "the
            shipped head is answering the wrong question" from "the features need to change".
            🔴 This is the rung that decides whether a backbone fine-tune is even worth running:
            if the head alone closes the gap, the features were never the problem.
  top2/top4 the last N transformer blocks unfrozen with the head. Where representation
            fine-tuning normally pays, and cheap enough to run honestly.
  full      all 164 M parameters. The most capacity and the most overfitting risk -- there are
            only 426 training clips from SEVEN actors, so this rung is as likely to be a warning
            as a win, and a loss here is a finding, not a failure (`R4`).

PROTOCOL, fixed by earlier sessions and NOT re-drawn here:
  1. ACTOR-disjoint split, imported from `probe_encoder_emotion.split_by_actor` so it cannot
     drift: 22 actors -> test `001 004 108 111 114 182 185 188`. The script ASSERTS that set and
     dies if it ever changes, because every number in RESEARCH-PROGRESS.md is on that one split.
  2. `agreement >= 0.70`, matching every probe so far and the corpus authors' own instruction
     (`A50`: they say 0.71; 0.70 is what is already in the record).
  3. Four classes. **Frustrated stays held out**, never folded onto ANGRY -- the paper confirms
     it is confused with angry, sad AND neutral.
  4. Hyperparameters and the stopping epoch are chosen on the DEV actors and the test actors are
     scored ONCE, at the end, from the dev-selected checkpoint (the `A38` discipline).
  5. macro-recall reported beside accuracy: the classes are imbalanced and accuracy alone hides
     SAD, which is where everything interesting happens.
  6. The improvised half is reported separately -- both models already score lower there and it
     is the closest thing to real speech in an acted corpus.

ENVIRONMENT -- NOT the repo's usual interpreter. TWO venvs, and which one matters:
    D:/sensevoice-spike/.venv/Scripts/python.exe   funasr 1.4.16, numpy 1.26.4, torch 2.14.0+CPU
    D:/sensevoice-gpu/.venv/Scripts/python.exe     the same stack on torch +cu130  (--device cuda)
🔴 Do NOT install funasr into E:/asr-spike/.venv: SenseVoice pins numpy<=1.26.4 and that venv
runs numpy 2.4.6 under NeMo 3.0.0. The downgrade would break every other arm in this repo.
The GPU venv is a SEPARATE tree rather than an upgrade of the CPU one, because the CPU one
produced the zero-shot bar and every encoder-probe number already on record; upgrading it in
place would make those unreproducible to save 8 GB of a 24 GB disk.

⚠ CPU IS VIABLE, GPU IS 19x. Measured, not assumed: 164 M params, fwd 0.41 s and fwd+bwd
1.31 s per 3-second clip on 20 CPU threads -- a 9-minute full-fine-tune epoch over 426 clips.
On the RTX 5060 Ti the same shape is 58 ms (18.9x), so an epoch is ~30 s and an honest lr sweep
becomes affordable instead of an overnight job.
🔴 A device change is a dependency change, so it is BRACKETED like one: re-run --zeroshot on
the new device and check it still lands on 0.7317 before trusting any rung measured there. This
repo already pays 4.3% B-CER for an unpinned kaldialign; a different BLAS is the same class of
bug, and the bar is the one number every rung is compared against.

Usage:
    $py = "D:/sensevoice-gpu/.venv/Scripts/python.exe"     # or sensevoice-spike for CPU
    & $py adapter/finetune_emotion2vec.py --cache                       # once: decode audio
    & $py adapter/finetune_emotion2vec.py --zeroshot --device cuda      # the bar + mean feats
    & $py adapter/finetune_emotion2vec.py --train --condition head   --device cuda
    & $py adapter/finetune_emotion2vec.py --train --condition top2   --device cuda
    & $py adapter/finetune_emotion2vec.py --train --condition full    --device cuda
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_emotion import (  # noqa: E402
    E2V_TO_SV,
    HELD_OUT,
    META_COLS,
    SV_CLASSES,
    THAISER_TO_SV,
    to_wav16k,
)
from probe_encoder_emotion import macro_recall, split_by_actor  # noqa: E402

CACHE_DIR = Path(os.environ.get("E2V_CACHE", "D:/sensevoice-spike/ft_cache"))
MODEL_CACHE = Path("D:/sensevoice-spike/models")
MODEL_ID = "emotion2vec/emotion2vec_plus_large"
MICS = ["mic_clip", "mic_con", "mic_middle", "mic_zoom"]
SCORE_MIC = "mic_clip"          # the channel every earlier number was measured on
EMBED_DIM = 1024

# The eight actors held out by probe_encoder_emotion.py on 2026-09-19. Asserted, not assumed:
# every number in RESEARCH-PROGRESS.md is on this split and a silent re-draw would make the
# comparison to 0.7317 meaningless while still printing a plausible number.
EXPECTED_TEST_ACTORS = ["001", "004", "108", "111", "114", "182", "185", "188"]

# Set once in main(). Measured on this box with a matched stand-in workload (12 blocks, d=1024,
# 16 heads, seq 150, batch 1): CPU 1103 ms/clip vs CUDA 58 ms/clip, an 18.9x speedup, which
# turns a 9-minute full-fine-tune epoch into ~30 seconds and makes an honest lr sweep affordable.
# 🔴 Batch stays 1 even on the GPU (bs=8 would be 55x). Batching needs padding plus a mask in the
# mean-pool, and a pooled vector that silently averages over pad frames is exactly the kind of
# quiet defect this repo keeps finding UNDER its numbers. Speed is not worth that here.
DEVICE = "cpu"


# --- stage 1: decode the audio ONCE --------------------------------------------------------

def build_cache(shard_dir: Path, agreement_min: float, mics: list[str]) -> None:
    """Decode the filtered subset out of the parquet shards into flat float32 .npy files.

    🔴 Why cache at all: a fine-tune reads every clip once per epoch, and the parquet path costs
    a FLAC decode plus a 44.1k->16k resample per clip. Caching turns ~10 minutes per epoch of
    pure I/O into a memmap read. It is also the only way two conditions can be shown to have
    seen byte-identical audio.

    float32, not int16. The repo already measured that a PCM16 wire changes 3.4% of ASR
    transcripts; whether it moves an emotion score is unknown, and a cache is the wrong place to
    introduce an unmeasured variable.

    All four mic channels are stored. Only `mic_clip` is ever scored -- the other three exist so
    a later condition can use them as training-time channel augmentation without a second pass
    over 1.4 GB of parquet.
    """
    import pyarrow.parquet as pq

    shards = sorted(shard_dir.glob("train-*.parquet"))
    if not shards:
        sys.exit(f"no parquet shards in {shard_dir} -- download them first")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    buffers = {m: [] for m in mics}
    meta, seen, skipped = [], 0, collections.Counter()
    t0 = time.time()

    for shard in shards:
        pf = pq.ParquetFile(shard)
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg, columns=META_COLS + mics)
            cols = {c: table.column(c).to_pylist() for c in table.column_names}
            for i in range(table.num_rows):
                seen += 1
                rec = {c: cols[c][i] for c in META_COLS}
                maj = rec["majority_emo"]
                # Identical filter to probe_emotion.iter_kept -- held-out Frustrated, unmapped
                # majorities ('None'/'other'), then the agreement gate.
                if maj in HELD_OUT:
                    skipped["held_out_frustrated"] += 1
                    continue
                if maj not in THAISER_TO_SV:
                    skipped["no_majority_or_unmapped"] += 1
                    continue
                if (rec["agreement"] or 0.0) < agreement_min:
                    skipped["below_agreement"] += 1
                    continue

                # 🔴 mic_zoom is NULL on the studio sessions -- it only exists for the 20 Zoom
                # sessions the corpus authors hold out as an out-of-domain set. A missing
                # channel is stored as a zero-length segment and excluded by `Clips.has`,
                # rather than dropping the clip: every other channel of it is still usable.
                lengths = {}
                for m in mics:
                    raw = cols[m][i]
                    blob = raw.get("bytes") if isinstance(raw, dict) else raw
                    if not blob:
                        lengths[m] = 0
                        continue
                    wav, _ = to_wav16k(raw)
                    buffers[m].append(wav)
                    lengths[m] = len(wav)
                rec.pop("annotated", None)          # votes are not needed downstream; they are
                rec["gold"] = THAISER_TO_SV[maj]    # large and make the index file unreadable
                rec["lengths"] = lengths
                rec["seconds"] = lengths[SCORE_MIC] / 16000.0
                meta.append(rec)
            del table, cols
        print(f"  [{shard.name}] {len(meta)} kept / {seen} seen, "
              f"{time.time() - t0:.0f}s", flush=True)

    available = {}
    for m in mics:
        have = sum(1 for r in meta if r["lengths"][m])
        available[m] = have
        if not have:
            print(f"  [skip] {m}: absent on all {len(meta)} kept clips", flush=True)
            continue
        flat = np.concatenate(buffers[m]).astype(np.float32)
        np.save(CACHE_DIR / f"wav_{m}.npy", flat)
        print(f"  [saved] {m}: {flat.nbytes / 1e6:.0f} MB, {have}/{len(meta)} clips", flush=True)
        buffers[m] = None

    index = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shards": [s.name for s in shards],
        "agreement_min": agreement_min,
        "mics": mics,
        "mic_clips_available": available,
        "rows_seen": seen,
        "rows_kept": len(meta),
        "skipped": dict(skipped),
        "actors": sorted({r["actor_id"] for r in meta}),
        "gold_distribution": dict(collections.Counter(r["gold"] for r in meta)),
        "audio_seconds": round(sum(r["seconds"] for r in meta), 1),
        "meta": meta,
    }
    (CACHE_DIR / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"[cache] {len(meta)} clips, {len(index['actors'])} actors, "
          f"{index['audio_seconds'] / 3600:.2f} h -> {CACHE_DIR}", flush=True)


class Clips:
    """Memmapped access to the cached audio. One object per mic channel."""

    def __init__(self, mic: str = SCORE_MIC):
        idx = json.loads((CACHE_DIR / "index.json").read_text(encoding="utf-8"))
        self.meta = idx["meta"]
        self.index = idx
        self.mic = mic
        self.flat = np.load(CACHE_DIR / f"wav_{mic}.npy", mmap_mode="r")
        offs, o = [], 0
        for r in self.meta:
            n = r["lengths"][mic]
            offs.append((o, n))
            o += n
        self.offs = offs
        assert o == self.flat.shape[0], f"cache length mismatch for {mic}: {o} vs {self.flat.shape[0]}"

    def __len__(self):
        return len(self.meta)

    def has(self, i: int) -> bool:
        """False when this clip was not recorded on this channel (see mic_zoom in build_cache)."""
        return self.offs[i][1] > 0

    def wav(self, i: int) -> np.ndarray:
        o, n = self.offs[i]
        return np.asarray(self.flat[o:o + n], dtype=np.float32)


def load_split(clips: Clips):
    """The actor-disjoint split, imported and then ASSERTED against the recorded test actors."""
    tr, dev, te, actors, test_actors = split_by_actor(clips.meta)
    if test_actors != EXPECTED_TEST_ACTORS:
        sys.exit(f"🔴 SPLIT DRIFT: test actors {test_actors} != recorded {EXPECTED_TEST_ACTORS}. "
                 "Every number in RESEARCH-PROGRESS.md is on the recorded split; refusing to "
                 "score a different one under the same name.")
    return tr, dev, te, actors, test_actors


# --- the model ------------------------------------------------------------------------------

def load_core():
    """The raw funasr Emotion2vec module, plus its 9-token label list.

    Loaded through AutoModel rather than by hand so this script cannot quietly end up on a
    different checkpoint from `probe_emotion.py` -- the 2026-09-17 failure mode that cost two
    full runs. `am.model` is the nn.Module; `am.kwargs['tokenizer'].token_list` is the 9-class
    label vocabulary the shipped `proj` head emits over.
    """
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
    os.environ.setdefault("MODELSCOPE_CACHE", str(MODEL_CACHE))
    from funasr import AutoModel
    am = AutoModel(model=MODEL_ID, hub="hf", device="cpu", disable_update=True)
    tok = am.kwargs.get("tokenizer")
    labels = list(tok.token_list) if tok is not None else []
    return am.model.to(DEVICE), labels


def encode(core, wav: np.ndarray, train: bool = False):
    """[N] float32 -> [1, T, 1024] frame features.

    The layer_norm and the [1, -1] view reproduce funasr's own inference path exactly
    (`funasr/models/emotion2vec/model.py`, `inference`). If this drifts from that path the
    zero-shot arm stops reproducing 0.7317 and every rung above it is measured against a
    baseline that no longer exists.
    """
    import torch
    import torch.nn.functional as F
    src = torch.from_numpy(wav).to(DEVICE)
    if core.cfg.normalize:
        src = F.layer_norm(src, src.shape)
    src = src.view(1, -1)
    # mask=False on both paths: the SSL masking is a pre-training objective, and leaving it on
    # would make the same clip score differently on every pass.
    return core.extract_features(src, padding_mask=None, mask=False)["x"]


# --- stage 2: the zero-shot bar ---------------------------------------------------------------

def run_zeroshot(out_dir: Path) -> dict:
    """Score the SHIPPED head on all three split parts, and cache the mean-pooled features.

    Two jobs in one forward pass because the forward is the expensive part:
      1. reproduce the 0.7317 bar on the identical 410 test clips -- the like-for-like re-score
         the 2026-09-19 entry says must happen before any of this is quoted;
      2. dump mean-pooled 1024-d features so the `head` rung trains in seconds instead of
         re-running the frozen backbone once per epoch.
    """
    import torch

    clips = Clips(SCORE_MIC)
    tr, dev, te, actors, test_actors = load_split(clips)
    core, labels = load_core()
    core.eval()
    print(f"[zeroshot] {len(clips)} clips | {len(actors)} actors | "
          f"train {len(tr)} dev {len(dev)} test {len(te)} | head labels {len(labels)}", flush=True)

    feats = np.zeros((len(clips), EMBED_DIM), dtype=np.float32)
    preds, t0 = [], time.time()
    with torch.no_grad():
        for i in range(len(clips)):
            x = encode(core, clips.wav(i))              # [1, T, 1024]
            pooled = x.mean(dim=1)                      # the model's own pooling
            feats[i] = pooled.squeeze(0).cpu().numpy()
            logits = core.proj(pooled)
            # Reproduce probe_emotion.predict_emotion2vec: argmax over the non-"unuse" labels,
            # then map the Chinese/English pair to our 4-class vocabulary. Labels outside the
            # four (FEARFUL, SURPRISED, OTHER, <UNK>) stay as themselves and score as wrong --
            # that is the honest zero-shot behaviour, not a bug to paper over.
            keep = [(k, l) for k, l in enumerate(labels) if not l.startswith("unuse")]
            best = max(keep, key=lambda kl: float(logits[0, kl[0]]))[1]
            key = str(best).split("/")[-1].strip().lower()
            preds.append(E2V_TO_SV.get(key, key.upper()))
            if (i + 1) % 200 == 0:
                print(f"  ... {i + 1}/{len(clips)}, {time.time() - t0:.0f}s", flush=True)

    np.save(CACHE_DIR / f"feats_e2v_mean_{DEVICE}.npy", feats)
    gold = [r["gold"] for r in clips.meta]
    parts = {}
    for name, idx in (("train", tr), ("dev", dev), ("test", te)):
        parts[name] = score_subset([gold[i] for i in idx], [preds[i] for i in idx],
                                   [clips.meta[i] for i in idx])
    res = {
        "arm": "finetune_emotion2vec", "condition": "zeroshot",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "what": "the SHIPPED 9-class emotion2vec_plus_large head, argmax mapped to 4 classes",
        "test_actors": test_actors, "splits": parts,
        "headline_test_accuracy": parts["test"]["accuracy"],
        "seconds": round(time.time() - t0, 1),
    }
    write_run(out_dir, "zeroshot", res, clips, te, gold, preds)
    print(f"\n[zeroshot] TEST accuracy {parts['test']['accuracy']:.4f} "
          f"| macro-recall {parts['test']['macro_recall']:.4f} | n={parts['test']['n']}", flush=True)
    return res


# --- the SenseVoice re-score, which closes a provenance question ---------------------------

def run_sensevoice(out_dir: Path) -> dict:
    """Score SenseVoiceSmall's SHIPPED emotion token on the identical 410 test clips.

    WHY THIS LIVES HERE. It is not a fine-tune rung -- it is the other half of the file whose
    provenance was open. `results/probe_emotion_thai_testactors.json` appeared on 2026-09-19
    from a script that could not have written it, and it carries TWO arms. `--zeroshot`
    reproduced its emotion2vec half exactly (410/410 per-clip predictions identical); this
    reproduces its SenseVoice half, so the whole file is either confirmed or caught.

    WHAT IT GATES. The headline comparison in `RESEARCH-PROGRESS.md` -- SenseVoice's own head
    emits SAD at 0.019 recall while a linear probe on SenseVoice's own frozen encoder reads it
    at 0.830 -- rests on the SenseVoice number being real and being measured on this exact
    denominator. `HISTORY.md` (2026-09-19 night) says in as many words that this re-score
    "should be run before this is quoted anywhere".

    🔴 The raw token is read, NOT rich_transcription_postprocess: that maps tokens to emoji and
    collapses NEUTRAL into "no emotion emitted", which is the distinction being measured.
    """
    import torch
    from probe_emotion import SV_EMO, SV_EVENT, SV_TOKEN_RE, _load

    clips = Clips(SCORE_MIC)
    tr, dev, te, actors, test_actors = load_split(clips)
    m = _load("FunAudioLLM/SenseVoiceSmall", MODEL_CACHE)
    if DEVICE == "cuda":                     # funasr AutoModel was built with device="cpu"
        m.model = m.model.to(DEVICE)
        m.kwargs["device"] = DEVICE
    gold = [r["gold"] for r in clips.meta]

    preds, events, raw, t0 = [], [], [], time.time()
    with torch.no_grad():
        for pos, i in enumerate(te):
            out = m.generate(input=clips.wav(i), cache={}, language="auto", use_itn=False,
                             ban_emo_unk=False, disable_pbar=True)
            text = out[0]["text"] if out else ""
            if len(raw) < 8:
                raw.append(text)
            toks = SV_TOKEN_RE.findall(text)
            preds.append(next((t for t in toks if t in SV_EMO), "NONE"))
            events.append(next((t for t in toks if t in SV_EVENT), "NONE"))
            if (pos + 1) % 100 == 0:
                print(f"  ... {pos + 1}/{len(te)}, {time.time() - t0:.0f}s", flush=True)

    metrics = score_subset([gold[i] for i in te], preds, [clips.meta[i] for i in te])
    # EMO_UNKNOWN is an ABSTENTION, not a wrong answer, and the distinction is the whole point:
    # the encoder has the evidence (0.830 SAD under a linear probe) and the head declines to
    # emit it. Precision-when-committing is therefore reported beside accuracy.
    committed = [(g, p) for g, p in zip((gold[i] for i in te), preds) if p != "EMO_UNKNOWN"]
    res = {
        "arm": "rescore_sensevoice_head", "condition": "testactors410",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "what": "SenseVoiceSmall's shipped CTC emotion token, raw, on the 410 held-out clips",
        "test_actors": test_actors, "test": metrics,
        "abstention_rate": sum(1 for p in preds if p == "EMO_UNKNOWN") / len(preds),
        "precision_when_committing": (sum(g == p for g, p in committed) / len(committed)
                                      if committed else None),
        "n_committed": len(committed),
        "event_distribution": dict(collections.Counter(events)),
        "raw_samples": raw,
        "reproduces": "results/probe_emotion_thai_testactors.json -> sensevoice",
        "seconds": round(time.time() - t0, 1),
    }
    write_run(out_dir, "sensevoice_head_410", res, clips, te, gold, _scatter(preds, te, len(clips)))
    print(f"\n[sensevoice] TEST accuracy {metrics['accuracy']:.4f} | macro-recall "
          f"{metrics['macro_recall']:.4f} | abstained {res['abstention_rate']:.1%} | "
          f"SAD recall {metrics['per_class']['SAD']['recall']:.4f}", flush=True)
    return res


def _scatter(preds, idx, n):
    """Per-clip predictions back into full-corpus positions, which write_run indexes by."""
    full = [None] * n
    for pos, i in enumerate(idx):
        full[i] = preds[pos]
    return full


# --- scoring ----------------------------------------------------------------------------------

def score_subset(gold, pred, rows) -> dict:
    """Accuracy, macro-recall, per class, confusion, and the improvised/scripted split."""
    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n if n else 0.0
    se = (acc * (1 - acc) / n) ** 0.5 if n else 0.0
    conf = collections.Counter(zip(gold, pred))
    per = {}
    for c in SV_CLASSES:
        sup = sum(1 for g in gold if g == c)
        pp = sum(1 for p in pred if p == c)
        per[c] = {"support": sup,
                  "recall": conf[(c, c)] / sup if sup else None,
                  "precision": conf[(c, c)] / pp if pp else None}
    rec = [v["recall"] for v in per.values() if v["recall"] is not None]
    maj = max((sum(1 for g in gold if g == c) for c in SV_CLASSES), default=0) / n if n else 0.0
    by_turn = {}
    for val in sorted({str(r.get("turn_type")) for r in rows}):
        idx = [i for i, r in enumerate(rows) if str(r.get("turn_type")) == val]
        if len(idx) >= 20:
            g2 = [gold[i] for i in idx]
            p2 = [pred[i] for i in idx]
            by_turn[val] = {"n": len(idx),
                            "accuracy": sum(a == b for a, b in zip(g2, p2)) / len(idx),
                            "macro_recall": _macro(g2, p2)}
    return {
        "n": n, "accuracy": acc,
        "ci95": [max(0.0, acc - 1.96 * se), min(1.0, acc + 1.96 * se)],
        "macro_recall": sum(rec) / len(rec) if rec else 0.0,
        "chance_majority_class": maj, "beats_majority_class": acc > maj,
        "per_class": per,
        "confusion": {f"{g}->{p}": v for (g, p), v in sorted(conf.items())},
        "pred_distribution": dict(collections.Counter(pred)),
        "by_turn_type": by_turn,
    }


def _macro(gold, pred) -> float:
    rs = []
    for c in SV_CLASSES:
        sup = sum(1 for g in gold if g == c)
        if sup:
            rs.append(sum(1 for g, p in zip(gold, pred) if g == c and p == c) / sup)
    return sum(rs) / len(rs) if rs else 0.0


def write_run(out_dir: Path, condition: str, res: dict, clips: Clips, te, gold, preds) -> None:
    """results/<arm>_<condition>_{metrics,hyps,run}.json -- the repo's flat convention."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"finetune_e2v_{condition}"
    (out_dir / f"{stem}_metrics.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / f"{stem}_hyps.jsonl").open("w", encoding="utf-8") as f:
        for i in te:
            r = clips.meta[i]
            f.write(json.dumps({"audio_id": r["audio_id"], "actor_id": r["actor_id"],
                                "gold": gold[i], "pred": preds[i],
                                "turn_type": r["turn_type"],
                                "script_intensity": r.get("script_intensity"),
                                "agreement": r["agreement"],
                                "seconds": round(r["seconds"], 3),
                                "correct": gold[i] == preds[i]}, ensure_ascii=False) + "\n")
    import torch
    (out_dir / f"{stem}_run.json").write_text(json.dumps({
        "arm": "finetune_emotion2vec", "condition": condition,
        "date": res.get("date"), "model": MODEL_ID,
        "cache": str(CACHE_DIR), "score_mic": SCORE_MIC, "device": DEVICE,
        "cache_index": {k: v for k, v in clips.index.items() if k != "meta"},
        "hyperparameters": res.get("hyperparameters"),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "device": DEVICE,
                        "gpu": (torch.cuda.get_device_name(0)
                                if DEVICE == "cuda" else None),
                        "threads": torch.get_num_threads(),
                        "venv": "D:/sensevoice-spike/.venv", "platform": platform.platform()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")


# --- stage 3: the fine-tune ---------------------------------------------------------------------

def set_trainable(core, condition: str) -> tuple[list, dict]:
    """Freeze everything, then unfreeze what the condition names. Returns (params, tally)."""
    for p in core.parameters():
        p.requires_grad_(False)
    groups = []
    if condition == "head":
        pass                                       # backbone stays frozen entirely
    elif condition.startswith("top"):
        n = int(condition[3:])
        for b in core.blocks[-n:]:
            for p in b.parameters():
                p.requires_grad_(True)
        groups = list(core.blocks[-n:])
    elif condition == "full":
        for p in core.parameters():
            p.requires_grad_(True)
        groups = [core]
    else:
        sys.exit(f"unknown condition {condition!r} -- use head, topN or full")
    if core.proj is not None:                      # the shipped 9-class head is replaced, never
        for p in core.proj.parameters():           # trained: it answers a different question
            p.requires_grad_(False)
    trainable = [p for p in core.parameters() if p.requires_grad]
    tally = {"trainable_params": int(sum(p.numel() for p in trainable)),
             "total_params": int(sum(p.numel() for p in core.parameters())),
             "unfrozen_groups": len(groups)}
    return trainable, tally


def evaluate(core, head, clips: Clips, idx, gold):
    import torch
    core.eval(); head.eval()
    preds = []
    with torch.no_grad():
        for i in idx:
            logits = head(encode(core, clips.wav(i)).mean(dim=1))
            preds.append(SV_CLASSES[int(logits.argmax(1))])
    return preds, score_subset([gold[i] for i in idx], preds, [clips.meta[i] for i in idx])


def evaluate_cached(head, feats, idx, clips, gold):
    import torch
    head.eval()
    with torch.no_grad():
        p = head(torch.from_numpy(feats[idx]).to(DEVICE)).argmax(1).cpu().numpy()
    preds = [SV_CLASSES[k] for k in p]
    return preds, score_subset([gold[i] for i in idx], preds, [clips.meta[i] for i in idx])


def train(condition: str, epochs: int, lr_head: float, lr_backbones: list[float], accum: int,
          seed: int, out_dir: Path, mics: list[str], smoke: int = 0) -> dict:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    clips = Clips(SCORE_MIC)
    tr, dev, te, actors, test_actors = load_split(clips)
    if smoke:
        # Walk the WHOLE code path on a handful of clips first. An 8-epoch full fine-tune is
        # ~90 minutes; discovering a shape error at the end of it is the expensive way to find
        # out. The result is thrown away -- `--smoke` also redirects the output so a toy run
        # can never land in results/ under a real condition name.
        tr, dev, te = tr[:smoke], dev[:smoke], te[:smoke]
        print(f"[SMOKE] {smoke} clips per split -- numbers are meaningless by construction",
              flush=True)
    gold = [r["gold"] for r in clips.meta]
    cls = {c: i for i, c in enumerate(SV_CLASSES)}
    y = np.array([cls[g] for g in gold], dtype=np.int64)

    head = torch.nn.Linear(EMBED_DIM, len(SV_CLASSES)).to(DEVICE)
    t0 = time.time()
    hist = []

    if condition == "head":
        # The frozen-backbone rung runs on the mean-pooled features the zero-shot pass cached,
        # so it costs seconds instead of an hour. Identical pooling to every other rung.
        # Device-matched by construction: a head rung on cuda reads the features cuda wrote.
        fp = CACHE_DIR / f"feats_e2v_mean_{DEVICE}.npy"
        if not fp.exists():
            sys.exit(f"{fp} missing -- run --zeroshot --device {DEVICE} first "
                     "(it caches the pooled features for this device)")
        feats = np.load(fp)
        mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-6   # standardise on TRAIN only
        Z = ((feats - mu) / sd).astype(np.float32)
        xt = torch.from_numpy(Z[tr]).to(DEVICE); yt = torch.from_numpy(y[tr]).to(DEVICE)
        # Weight decay swept on the DEV actors, exactly as probe_encoder_emotion.py does for its
        # linear probe -- otherwise this rung is not comparable to the 0.7707 already on record.
        best, chosen_wd = None, None
        for wd in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            torch.manual_seed(seed)
            h = torch.nn.Linear(EMBED_DIM, len(SV_CLASSES)).to(DEVICE)
            opt = torch.optim.AdamW(h.parameters(), lr=lr_head, weight_decay=wd)
            for ep in range(epochs):
                h.train()
                opt.zero_grad()
                loss = F.cross_entropy(h(xt), yt)
                loss.backward()
                opt.step()
                if (ep + 1) % max(1, epochs // 20) == 0 or ep == epochs - 1:
                    _, dm = evaluate_cached(h, Z, dev, clips, gold)
                    hist.append({"weight_decay": wd, "epoch": ep + 1, "train_loss": float(loss),
                                 "dev_accuracy": dm["accuracy"],
                                 "dev_macro_recall": dm["macro_recall"]})
                    if best is None or dm["accuracy"] > best[1]:
                        best = (ep + 1, dm["accuracy"], copy.deepcopy(h.state_dict()))
                        chosen_wd = wd
        head.load_state_dict(best[2])
        lr_backbone = 0.0                       # recorded honestly: the backbone never moved
        epochs = f"{epochs} x 5 wd values; best wd={chosen_wd}"
        preds_te, te_metrics = evaluate_cached(head, Z, te, clips, gold)
        _, dev_metrics = evaluate_cached(head, Z, dev, clips, gold)
        tally = {"trainable_params": int(sum(p.numel() for p in head.parameters())),
                 "total_params": 164048921 + 4100, "unfrozen_groups": 0}
        core = None
    else:
        # Training-time channel augmentation, off by default. The other three mics are the SAME
        # performance through a different microphone, so they multiply the 426 training clips
        # without inventing anything -- but only on TRAIN, and test stays mic_clip.
        extra = [Clips(m) for m in mics
                 if m != SCORE_MIC and (CACHE_DIR / f"wav_{m}.npy").exists()]

        # 🔴 THE BACKBONE LEARNING RATE IS SWEPT, NOT GUESSED. One arbitrary lr cannot answer
        # "does unfreezing help" -- too low and the backbone never moves, too high and it forgets
        # on 426 clips, and BOTH failures look identical from the outside: "fine-tuning did not
        # work". Every config is scored on DEV; the test actors are touched ONCE, at the end, by
        # the dev-selected winner alone (the `A38` discipline). The losing configs stay in
        # `history` so the sweep is auditable, but they are NOT selection criteria.
        best = None
        for lr_b in lr_backbones:
            torch.manual_seed(seed)
            core, _ = load_core()
            head = torch.nn.Linear(EMBED_DIM, len(SV_CLASSES)).to(DEVICE)
            trainable, tally = set_trainable(core, condition)
            tally["trainable_params"] += int(sum(p.numel() for p in head.parameters()))
            opt = torch.optim.AdamW(
                [{"params": head.parameters(), "lr": lr_head},
                 {"params": trainable, "lr": lr_b}], weight_decay=1e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
            print(f"[{condition}] lr_backbone {lr_b:g} | trainable "
                  f"{tally['trainable_params'] / 1e6:.1f} M / {tally['total_params'] / 1e6:.1f} M"
                  f" | train {len(tr)} (+{len(extra)} extra mics) dev {len(dev)} test {len(te)}",
                  flush=True)

            rng = np.random.default_rng(seed)
            for ep in range(epochs):
                core.train(); head.train()
                order = rng.permutation(len(tr))
                opt.zero_grad()
                run_loss, nstep = 0.0, 0
                for k, j in enumerate(order):
                    i = tr[j]
                    src = clips
                    if extra and k % (len(extra) + 1) != 0:
                        cand = extra[(k - 1) % len(extra)]
                        src = cand if cand.has(i) else clips
                    logits = head(encode(core, src.wav(i), train=True).mean(dim=1))
                    loss = F.cross_entropy(logits, torch.tensor([y[i]], device=DEVICE)) / accum
                    loss.backward()
                    run_loss += float(loss) * accum
                    if (k + 1) % accum == 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in head.parameters()] + trainable, 5.0)
                        opt.step(); opt.zero_grad(); nstep += 1
                opt.zero_grad()
                sched.step()
                _, dm = evaluate(core, head, clips, dev, gold)
                hist.append({"lr_backbone": lr_b, "epoch": ep + 1,
                             "train_loss": run_loss / max(1, len(order)),
                             "dev_accuracy": dm["accuracy"],
                             "dev_macro_recall": dm["macro_recall"],
                             "optimizer_steps": nstep,
                             "elapsed_s": round(time.time() - t0, 1)})
                print(f"  [lr {lr_b:g} ep {ep + 1}/{epochs}] loss "
                      f"{hist[-1]['train_loss']:.4f} | dev acc {dm['accuracy']:.4f} | "
                      f"dev macro {dm['macro_recall']:.4f} | {hist[-1]['elapsed_s']:.0f}s",
                      flush=True)
                if best is None or dm["accuracy"] > best[1]:
                    # Winner parked on the CPU: a three-lr sweep would otherwise hold three
                    # 656 MB checkpoints of configs that have already lost.
                    best = (ep + 1, dm["accuracy"],
                            {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                            {k: v.detach().cpu().clone() for k, v in core.state_dict().items()},
                            lr_b)
            del core, head, opt, trainable
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        lr_backbone = best[4]
        core, _ = load_core()
        head = torch.nn.Linear(EMBED_DIM, len(SV_CLASSES)).to(DEVICE)
        head.load_state_dict({k: v.to(DEVICE) for k, v in best[2].items()})
        core.load_state_dict({k: v.to(DEVICE) for k, v in best[3].items()})
        _, tally = set_trainable(core, condition)
        tally["trainable_params"] += int(sum(p.numel() for p in head.parameters()))
        tally["swept_lr_backbone"] = list(lr_backbones)
        preds_te, te_metrics = evaluate(core, head, clips, te, gold)
        _, dev_metrics = evaluate(core, head, clips, dev, gold)

    preds_full = [None] * len(clips)
    for pos, i in enumerate(te):
        preds_full[i] = preds_te[pos]

    res = {
        "arm": "finetune_emotion2vec", "condition": condition,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "what": {"head": "backbone FROZEN, fresh 4-class head on mean-pooled frames",
                 "full": "all 164 M parameters unfrozen, plus a fresh 4-class head"}.get(
                     condition, f"last {condition[3:]} transformer blocks unfrozen, plus a head"),
        "test_actors": test_actors,
        "baseline_to_beat": {"zeroshot_test_accuracy": 0.7317,
                             "linear_probe_meanstd_2026_09_19": 0.7707,
                             "majority_class": 0.3537},
        "hyperparameters": {"epochs": epochs, "lr_head": lr_head, "lr_backbone": lr_backbone,
                            "grad_accum": accum, "seed": seed, "weight_decay": 1e-2,
                            "train_mics": mics, "selection": "dev accuracy, reported once on test",
                            **tally},
        "epoch_selected_on_dev": best[0], "dev_accuracy_at_selection": best[1],
        "dev": dev_metrics, "test": te_metrics, "history": hist,
        "headline_test_accuracy": te_metrics["accuracy"],
        "beats_zeroshot": te_metrics["accuracy"] > 0.7317,
        "seconds": round(time.time() - t0, 1),
    }
    write_run(out_dir, condition, res, clips, te, gold, preds_full)
    print(f"\n[{condition}] epoch {best[0]} selected on dev ({best[1]:.4f}) | "
          f"TEST accuracy {te_metrics['accuracy']:.4f} "
          f"[{te_metrics['ci95'][0]:.3f},{te_metrics['ci95'][1]:.3f}] | "
          f"macro-recall {te_metrics['macro_recall']:.4f} | "
          f"vs zero-shot 0.7317 -> {'BEATS' if res['beats_zeroshot'] else 'DOES NOT BEAT'}",
          flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true", help="decode the filtered subset once")
    ap.add_argument("--zeroshot", action="store_true", help="score the shipped head; cache feats")
    ap.add_argument("--sensevoice", action="store_true",
                    help="re-score SenseVoice's shipped emotion token on the same 410")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--condition", default="head", help="head | top2 | top4 | full")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", default="1e-5",
                    help="comma list swept on dev; test is scored once by the winner")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0,
                    help="walk the code path on N clips per split; writes to results/_smoke/")
    ap.add_argument("--mics", default=SCORE_MIC,
                    help=f"comma list for TRAIN-time augmentation; test is always {SCORE_MIC}")
    ap.add_argument("--cache-mics", default=",".join(MICS),
                    help="which mic channels to decode; fewer = less disk")
    ap.add_argument("--shards", default="D:/thai-ser")
    ap.add_argument("--agreement-min", type=float, default=0.7)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default=os.environ.get("ADAPTER_OUT", "results"))
    a = ap.parse_args()

    global DEVICE
    DEVICE = a.device
    if DEVICE == "cuda":
        import torch
        if not torch.cuda.is_available():
            sys.exit("--device cuda but torch.cuda.is_available() is False. The CPU venv "
                     "(D:/sensevoice-spike) has a +cpu torch build; use D:/sensevoice-gpu.")
        print(f"[device] cuda | {torch.cuda.get_device_name(0)} | torch {torch.__version__}",
              flush=True)

    if a.cache:
        build_cache(Path(a.shards), a.agreement_min,
                    [m.strip() for m in a.cache_mics.split(",") if m.strip()])
    if a.zeroshot:
        run_zeroshot(Path(a.out))
    if a.sensevoice:
        run_sensevoice(Path(a.out))
    if a.train:
        epochs = a.epochs if a.epochs is not None else (400 if a.condition == "head" else 8)
        out = Path(a.out) / "_smoke" if a.smoke else Path(a.out)
        lrs = [float(v) for v in str(a.lr_backbone).split(",") if v.strip()]
        train(a.condition, epochs, a.lr_head, lrs, a.accum, a.seed,
              out, [m.strip() for m in a.mics.split(",") if m.strip()], a.smoke)
    if not (a.cache or a.zeroshot or a.sensevoice or a.train):
        ap.error("pass --cache, --zeroshot, --sensevoice and/or --train")


if __name__ == "__main__":
    main()
