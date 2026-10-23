"""probe_emotion.py -- does prosody-based emotion recognition transfer to Thai, zero-shot?

SPIKE, 2026-09-19. Throwaway by intent: the output is an ANSWER, not a pipeline.

THE QUESTION. SenseVoiceSmall emits an emotion token at output position 2 of its CTC head,
learned from 30M pseudo-labels over zh/yue/en/ja/ko. It has never heard Thai and carries no
Thai in its vocabulary. Emotion is largely prosodic, and two of its five training languages
(zh, yue) are tonal -- so the model has already had to separate lexical pitch from affective
pitch. Does that survive contact with an unseen tonal language?

THE TWO ARMS, and the second is the one that bounds the answer:
  1. SenseVoiceSmall          -- the student. Its transcript will be garbage; we read position 2.
  2. emotion2vec_plus_large   -- the TEACHER that produced SenseVoice's 30M SER pseudo-labels.
     Its card claims to "overcome the effects of language ... to achieve universal" recognition
     but NAMES NO LANGUAGES. If the teacher cannot do Thai, the student never could, and that
     is a cheaper and more general finding than arm 1 alone.

THE CEILING, which is why this script computes one. THAI-SER's annotators agree at mean 0.707,
and on 673 clips they agreed on nothing at all ("None"). A model cannot beat the humans whose
majority vote is the ground truth. So we report accuracy against a LEAVE-ONE-ANNOTATOR-OUT
human score on the identical filtered subset. An accuracy quoted without that ceiling is
meaningless, in the same way a Thai CER quoted without SARA AM folding is (F1).

ENVIRONMENT -- deliberately NOT the repo's usual interpreter.
  D:/sensevoice-spike/.venv/Scripts/python.exe
  SenseVoice pins numpy<=1.26.4; E:/asr-spike/.venv runs numpy 2.4.6 and NeMo on top of it.
  Installing funasr there would downgrade numpy under NeMo. Separate venv, nothing shared.
  CPU torch on purpose: ~2k clips is ~20 min on CPU, and it keeps the GPU free for the loop.

DATA. airesearch/thai-ser (THAI-SER), CC-BY-SA-4.0, ungated. 27,854 utts / 41h36m / 200 actors,
5 emotions, 4 mic channels per clip. Shards 0/10/20 of 29 are pulled (spread, not the first
three, so actors and rooms vary). Local: D:/thai-ser/.

Usage:
    $py = "D:/sensevoice-spike/.venv/Scripts/python.exe"
    & $py adapter/probe_emotion.py --limit 200          # smoke test
    & $py adapter/probe_emotion.py                      # full filtered subset
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

# --- THAI-SER -> SenseVoice label mapping ------------------------------------------------
# Frustration has NO SenseVoice class. It is deliberately NOT mapped: it is held out as a
# diagnostic -- where does an unmapped emotion land? Forcing it onto ANGRY would manufacture
# agreement and is exactly the kind of convention choice F17 is about.
THAISER_TO_SV = {
    "Neutral": "NEUTRAL",
    "Happy": "HAPPY",
    "Angry": "ANGRY",
    "Sad": "SAD",
}
HELD_OUT = {"Frustrated"}
SV_CLASSES = ["NEUTRAL", "HAPPY", "ANGRY", "SAD"]

# emotion2vec emits labels like "生气/angry" or "angry". Normalise to the SV vocabulary.
E2V_TO_SV = {
    "angry": "ANGRY", "happy": "HAPPY", "sad": "SAD", "neutral": "NEUTRAL",
    "fearful": "FEARFUL", "disgusted": "DISGUSTED", "surprised": "SURPRISED",
    "other": "OTHER", "unknown": "UNKNOWN",
}

# 🔴 [A-Za-z_], NOT [A-Z_]. The emotion tokens are all-caps (<|HAPPY|>, <|EMO_UNKNOWN|>) but the
# EVENT tokens are mixed case (<|Speech|>, <|Laughter|>, <|BGM|>) and an uppercase-only class
# silently read every one of them as "no event emitted". Caught in the 20-clip smoke test by a
# distribution that was 100% NONE -- the kind of result that is wrong rather than interesting.
SV_TOKEN_RE = re.compile(r"<\|([A-Za-z_]+)\|>")

META_COLS = [
    "audio_id", "assigned_emo", "majority_emo", "annotated", "agreement",
    "session_id", "room_type", "actor_id", "actor_gender", "actor_age",
    "turn_type", "script_intensity",
]


# --- data ---------------------------------------------------------------------------------

def iter_kept(shard_dir: Path, mic: str, agreement_min: float, limit: int | None):
    """Stream the shards ROW GROUP AT A TIME, yielding kept records with their audio.

    🔴 Why streaming and not pq.read_table: a whole shard is ~430 MB of FLAC bytes, and
    to_pylist() turns that into Python bytes objects -- measured at 1.5 GB resident for ONE
    shard. Three shards would be ~4 GB on a box whose commit limit already killed a job on
    2026-09-19. A row group is 100 rows, so peak stays around 40 MB and the run cannot die
    the silent exit-0 death the repo keeps warning about.

    Yields (record, is_held_out). `seen` is reported by the caller via the counter dict.
    """
    shards = sorted(shard_dir.glob("train-*.parquet"))
    if not shards:
        sys.exit(f"no parquet shards in {shard_dir} -- download them first")

    counts = {"seen": 0, "kept": 0, "held": 0}
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg, columns=META_COLS + [mic])
            cols = {c: table.column(c).to_pylist() for c in table.column_names}
            for i in range(table.num_rows):
                counts["seen"] += 1
                rec = {c: cols[c][i] for c in META_COLS}
                maj = rec["majority_emo"]
                if maj in HELD_OUT:
                    if not limit or counts["held"] < max(1, limit // 4):
                        counts["held"] += 1
                        yield {**rec, "audio": cols[mic][i]}, True
                    continue
                if maj not in THAISER_TO_SV:
                    continue                  # 'None' (no annotator majority) and 'other'
                if (rec["agreement"] or 0.0) < agreement_min:
                    continue
                if limit and counts["kept"] >= limit:
                    continue
                counts["kept"] += 1
                rec["audio"] = cols[mic][i]
                rec["gold"] = THAISER_TO_SV[maj]
                yield rec, False
            del table, cols
        if limit and counts["kept"] >= limit and counts["held"] >= max(1, limit // 4):
            break
    yield counts, None                        # sentinel: final tallies


def to_wav16k(audio_struct) -> tuple[np.ndarray, int]:
    """HF audio struct -> mono float32 at 16 kHz, which is what both models expect.

    THAI-SER ships FLAC at 44.1 kHz STEREO; both models want 16 kHz mono, so neither the
    downmix nor the resample is optional. Measured on shard 0: mean 2.72 s, min 0.53, max 9.80.
    """
    raw = audio_struct["bytes"] if isinstance(audio_struct, dict) else audio_struct
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != 16000:
        # Two venvs, two available resamplers: D:/sensevoice-spike has torchaudio, and
        # E:/asr-spike (NeMo) has librosa but NOT torchaudio. Try both rather than making the
        # caller care which interpreter it is running under.
        try:
            import torch
            import torchaudio.functional as AF
            data = AF.resample(torch.from_numpy(data), sr, 16000).numpy()
        except ModuleNotFoundError:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000
    return np.ascontiguousarray(data, dtype=np.float32), sr


def duration_band(seconds: float) -> str:
    """Clips run 0.53-9.80 s. A half-second clip carries almost no prosodic contour, so a
    flat accuracy over all of them hides whether the model is failing or merely starved."""
    if seconds < 1.0:
        return "a_under_1s"
    if seconds < 2.0:
        return "b_1_2s"
    if seconds < 4.0:
        return "c_2_4s"
    return "d_over_4s"


# --- the human ceiling --------------------------------------------------------------------

def human_ceiling(rows) -> dict:
    """Leave-one-annotator-out: how often does a randomly chosen annotator match the majority?

    This is the number a model is actually competing with. The `annotated` column holds each
    annotator's vote; its exact nesting is not documented, so flatten defensively and report
    how many rows were usable rather than silently scoring a subset.
    """
    hits = total = usable = 0
    for r in rows:
        votes = []
        for a in (r.get("annotated") or []):
            for v in (a if isinstance(a, (list, tuple)) else [a]):
                if isinstance(v, str) and v in THAISER_TO_SV:
                    votes.append(THAISER_TO_SV[v])
        if len(votes) < 2:
            continue
        usable += 1
        for k in range(len(votes)):
            rest = votes[:k] + votes[k + 1:]
            maj = collections.Counter(rest).most_common(1)[0][0]
            hits += int(votes[k] == maj)
            total += 1
    # A SECOND, HARDER CEILING. The one above asks "can a model match the annotators", on clips
    # already selected for annotator agreement -- circular by construction, so it reads high.
    # This one asks whether the annotators recovered what the ACTOR WAS TOLD to perform. That is
    # the quantity the THAI-SER paper reports as "human recognition accuracy ... up to 0.772
    # post-filtering" (arXiv 2507.09618), so it is the comparable number. Ours is 4-way on a
    # filtered subset and theirs is 5-way over the corpus, so the two are NOT interchangeable --
    # both are recorded rather than picking whichever flatters the model.
    ass_hits = ass_total = 0
    for r in rows:
        a = THAISER_TO_SV.get(r.get("assigned_emo"))
        if a is None:
            continue
        ass_total += 1
        ass_hits += int(a == r["gold"])
    return {
        "leave_one_out_accuracy": (hits / total) if total else None,
        "rows_usable": usable, "rows_total": len(rows), "votes_scored": total,
        "majority_vs_assigned_accuracy": (ass_hits / ass_total) if ass_total else None,
        "majority_vs_assigned_n": ass_total,
        "published_human_accuracy_5way": 0.772,
        "published_krippendorff_alpha": 0.692,
        "published_source": "THAI-SER, arXiv 2507.09618 -- 5-way over the whole corpus, "
                            "NOT directly comparable to the 4-way filtered numbers here",
    }


# --- the two arms -------------------------------------------------------------------------

def _load(model_id: str, cache: Path):
    """funasr's AutoModel has changed its cache kwarg across versions; steer it with the
    environment instead, which every version honours, and fall back if cache_dir is rejected."""
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("MODELSCOPE_CACHE", str(cache))
    from funasr import AutoModel
    try:
        return AutoModel(model=model_id, hub="hf", device="cpu",
                         disable_update=True, cache_dir=str(cache))
    except TypeError:
        return AutoModel(model=model_id, hub="hf", device="cpu", disable_update=True)


SV_EMO = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL",
          "DISGUSTED", "SURPRISED", "EMO_UNKNOWN"}
SV_EVENT = {"BGM", "Speech", "Applause", "Laughter", "Cry",
            "Sneeze", "Breath", "Cough", "Sing", "Speech_Noise"}


def predict_sensevoice(m, wav, keep_raw: list | None = None) -> tuple[str, str]:
    """Read the emotion and event tokens straight off SenseVoice's raw output.

    The raw text is `<|lang|><|event|><|emotion|><|itn|>transcript`. We deliberately do NOT
    call rich_transcription_postprocess: it maps the tokens to emoji and drops the distinction
    between NEUTRAL and 'no emotion emitted', which is exactly what this probe measures.
    The transcript itself is expected to be garbage -- there is no Thai in the vocabulary.
    """
    out = m.generate(input=wav, cache={}, language="auto", use_itn=False,
                     ban_emo_unk=False, disable_pbar=True)
    text = out[0]["text"] if out else ""
    if keep_raw is not None and len(keep_raw) < 8:
        keep_raw.append(text)                 # evidence that the transcript really is garbage
    toks = SV_TOKEN_RE.findall(text)
    return (next((t for t in toks if t in SV_EMO), "NONE"),
            next((t for t in toks if t in SV_EVENT), "NONE"))


def predict_emotion2vec(m, wav) -> str:
    out = m.generate(wav, granularity="utterance", extract_embedding=False, disable_pbar=True)
    lab = "NONE"
    if out:
        o = out[0]
        if "labels" in o and "scores" in o:
            lab = o["labels"][int(np.argmax(o["scores"]))]
        elif "labels" in o:
            lab = o["labels"][0]
    key = str(lab).split("/")[-1].strip().lower()
    return E2V_TO_SV.get(key, key.upper())


def run_arms(shard_dir, mic, agreement_min, limit, cache: Path, skip_e2v: bool):
    """ONE pass over the corpus, both models per clip.

    Single pass because the parquet read is the expensive part and because holding every
    clip's audio to run two separate passes is what the streaming rewrite exists to avoid.
    Audio is decoded once, handed to both models, then dropped.
    """
    sv_model = _load("FunAudioLLM/SenseVoiceSmall", cache)
    e2v_model = None if skip_e2v else _load("emotion2vec/emotion2vec_plus_large", cache)

    rows, held_preds, raw_samples = [], [], []
    sv_pred, sv_event, e2v_pred = [], [], []
    counts, t0, n = {}, time.time(), 0

    for rec, is_held in iter_kept(shard_dir, mic, agreement_min, limit):
        if is_held is None:                   # sentinel
            counts = rec
            break
        wav, _ = to_wav16k(rec["audio"])
        emo, ev = predict_sensevoice(sv_model, wav, raw_samples)
        if is_held:
            held_preds.append(emo)
            continue
        rec["seconds"] = len(wav) / 16000.0
        rec["band"] = duration_band(rec["seconds"])
        rec.pop("audio", None)                # drop the bytes; nothing downstream needs them
        rows.append(rec)
        sv_pred.append(emo)
        sv_event.append(ev)
        if e2v_model is not None:
            e2v_pred.append(predict_emotion2vec(e2v_model, wav))
        n += 1
        if n % 100 == 0:
            print(f"  ... {n} clips, {time.time() - t0:.0f}s elapsed", flush=True)

    return (rows, sv_pred, sv_event, e2v_pred, held_preds, counts,
            raw_samples, time.time() - t0)


# --- scoring ------------------------------------------------------------------------------

def score(gold, pred, classes):
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, pred))
    acc = correct / n if n else 0.0
    se = (acc * (1 - acc) / n) ** 0.5 if n else 0.0
    conf = collections.Counter(zip(gold, pred))
    per = {}
    for c in classes:
        tp = conf[(c, c)]
        sup = sum(v for (g, _), v in conf.items() if g == c)
        pp = sum(v for (_, p), v in conf.items() if p == c)
        per[c] = {"support": sup, "recall": tp / sup if sup else None,
                  "precision": tp / pp if pp else None}
    # The 4 classes are NOT balanced (NEUTRAL 415 / HAPPY 352 / ANGRY 289 / SAD 200 on the
    # 1,256-clip subset), so uniform 1/4 is the wrong floor to beat: always guessing the
    # largest class already scores 33%. Report both, and treat the larger as the real floor.
    maj_share = max(sum(1 for g in gold if g == c) for c in classes) / n if n else 0.0
    return {
        "n": n, "accuracy": acc, "ci95": [max(0.0, acc - 1.96 * se), min(1.0, acc + 1.96 * se)],
        "chance_uniform": 1.0 / len(classes),
        "chance_majority_class": maj_share,
        "beats_majority_class": acc > maj_share,
        "macro_recall": (sum(v["recall"] for v in per.values() if v["recall"] is not None)
                         / max(1, sum(1 for v in per.values() if v["recall"] is not None))),
        "per_class": per,
        "confusion": {f"{g}->{p}": v for (g, p), v in sorted(conf.items())},
        "pred_distribution": dict(collections.Counter(pred)),
    }


def by_condition(rows, gold, pred, field):
    out = {}
    for val in sorted({str(r.get(field)) for r in rows}):
        idx = [i for i, r in enumerate(rows) if str(r.get(field)) == val]
        if len(idx) >= 20:
            out[val] = {"n": len(idx),
                        "accuracy": sum(gold[i] == pred[i] for i in idx) / len(idx)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="D:/thai-ser")
    ap.add_argument("--mic", default="mic_clip")
    ap.add_argument("--agreement-min", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache", default="D:/sensevoice-spike/models")
    ap.add_argument("--skip-e2v", action="store_true")
    ap.add_argument("--out", default=os.environ.get("ADAPTER_OUT", "results"))
    a = ap.parse_args()

    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    print("[run] one streaming pass, both models per clip ...", flush=True)
    (rows, sv_pred, sv_event, e2v_pred, sv_held,
     counts, raw_samples, secs) = run_arms(Path(a.shards), a.mic, a.agreement_min,
                                           a.limit, cache, a.skip_e2v)
    seen = counts.get("seen", 0)
    print(f"[data] {seen} rows scanned -> {len(rows)} kept "
          f"(majority_emo in 4 mapped classes, agreement >= {a.agreement_min}), "
          f"{len(sv_held)} held-out Frustrated, {secs:.0f}s", flush=True)
    if not rows:
        sys.exit("nothing survived the filter")

    gold = [r["gold"] for r in rows]
    print(f"[data] gold distribution: {dict(collections.Counter(gold))}", flush=True)

    ceiling = human_ceiling(rows)
    print(f"[ceiling] leave-one-annotator-out = {ceiling['leave_one_out_accuracy']:.4f} | "
          f"annotators vs assigned = {ceiling['majority_vs_assigned_accuracy']:.4f}", flush=True)

    sv = score(gold, sv_pred, SV_CLASSES)
    print(f"[arm 1] SenseVoiceSmall  accuracy {sv['accuracy']:.4f} "
          f"vs majority-class {sv['chance_majority_class']:.4f} | "
          f"macro-recall {sv['macro_recall']:.4f}", flush=True)
    print(f"[arm 1] prediction distribution: {sv['pred_distribution']}", flush=True)

    e2v = None
    if not a.skip_e2v:
        e2v = score(gold, e2v_pred, SV_CLASSES)
        e2v["seconds"] = secs
        e2v["by_turn_type"] = by_condition(rows, gold, e2v_pred, "turn_type")
        e2v["by_duration_band"] = by_condition(rows, gold, e2v_pred, "band")
        e2v["by_room_type"] = by_condition(rows, gold, e2v_pred, "room_type")
        print(f"[arm 2] emotion2vec_plus_large  accuracy {e2v['accuracy']:.4f} | "
              f"macro-recall {e2v['macro_recall']:.4f}", flush=True)
        print(f"[arm 2] prediction distribution: {e2v['pred_distribution']}", flush=True)

    sv_secs = secs
    result = {
        "probe": "emotion-transfer-to-thai",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": "does prosody-based emotion recognition transfer zero-shot to Thai?",
        "corpus": {"name": "airesearch/thai-ser", "licence": "CC-BY-SA-4.0",
                   "mic": a.mic, "agreement_min": a.agreement_min,
                   "rows_scanned": seen, "rows_kept": len(rows),
                   "held_out_frustrated_n": len(sv_held),
                   "actors": len({r["actor_id"] for r in rows}),
                   "audio_seconds": round(sum(r["seconds"] for r in rows), 1),
                   "gold_distribution": dict(collections.Counter(gold))},
        "human_ceiling": ceiling,
        "sensevoice_raw_samples": raw_samples,
        "sensevoice": {**sv, "seconds": sv_secs,
                       "event_distribution": dict(collections.Counter(sv_event)),
                       "by_turn_type": by_condition(rows, gold, sv_pred, "turn_type"),
                       "by_room_type": by_condition(rows, gold, sv_pred, "room_type"),
                       "by_actor_gender": by_condition(rows, gold, sv_pred, "actor_gender"),
                       "by_duration_band": by_condition(rows, gold, sv_pred, "band"),
                       "held_out_frustrated": dict(collections.Counter(sv_held))},
        "emotion2vec": e2v,
        "environment": {"python": sys.version.split()[0], "device": "cpu",
                        "venv": "D:/sensevoice-spike/.venv"},
    }

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dest = out / "probe_emotion_thai.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] -> {dest}", flush=True)


if __name__ == "__main__":
    main()
