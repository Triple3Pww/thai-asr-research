#!/usr/bin/env python3
"""What does Typhoon cost on SPONTANEOUS Thai, and on a far microphone?

    # asr/run.ps1 must be up on :8007
    py adapter/arm_spontaneous.py                     # 150 utts x 5 mics
    py adapter/arm_spontaneous.py --n 300 --min-dur 3

WHY THIS EXISTS. He reported it plainly: Typhoon "even miss the normal conversation context". The
instrument already hinted at it and nobody had followed up -- `arm1_baseline` reads CER 0.0518 on
150 Common Voice clips, and `arm1_rareword` reads 0.1125 on 682 real benchmark utterances, 2.2x
worse. But those two differ in corpus, speaker set, channel AND register at once, so the gap cannot
be attributed. `A19` says only that a read-vs-spontaneous gap exists for Thai; no one publishes what
it costs.

⭐ LOTUSDIS MAKES THIS A CONTROLLED EXPERIMENT RATHER THAN A COMPARISON. Its test split holds
**5,516 rows per microphone across five devices, row-aligned**: verified 5516/5516 identical on
sentence, speaker_id and duration for lav123 vs each of con123, bt3m, bt10m, jbl. The SAME utterance
was recorded simultaneously on all five. So the mic rows differ in exactly one factor -- the
channel -- and nothing else. That is the rarest thing in this repo: a comparison with one variable.

TWO NUMBERS COME OUT.

  1. READ vs SPONTANEOUS. lav123 (a lavalier, the closest and cleanest channel) against the
     Common Voice 0.0518 from arm1_baseline. Both scored by the same folding, on this box.
  2. WHAT THE CAPTURE DEVICE COSTS, on identical speech. Not 'distance': see MIC_NOTE.

⚠️ THREE THINGS THAT WOULD HAVE MADE THIS MEANINGLESS, AND WHAT IS DONE ABOUT THEM.

  * THE SPLIT OPENS WITH SCRIPTED GREETINGS. The first rows are "สวัสดีครับ M ศูนย์สามครับ" -- a
    calibration preamble, not dialogue. Sampling from the top would have measured read speech and
    called it spontaneous. Sampling is random over the whole split with a fixed seed.
  * DURATION IS NOT COMPARABLE BY DEFAULT. LOTUSDIS utterances run to a 1.69 s median where Common
    Voice averages ~6 s, and CER on a 1.7 s utterance is dominated by edge effects. `--min-dur`
    defaults to 3.0 s, which leaves 488 candidates per mic. The number is reported WITH the mean
    duration so a reader can see the residual mismatch rather than being told it is absent.
  * THE SAME UTTERANCE IDS ARE USED FOR EVERY MIC. Sampling each mic independently would put a
    different 150 utterances behind each row and turn a controlled comparison back into a noisy one.

SCORING IS arm1_baseline's, EXACTLY: `_thai.py`'s fold on BOTH sides (F1 -- SARA AM is 26.6% of mean
CER and `unicodedata.normalize` does nothing about it), the same three normalisation modes reported,
and the FOLD column is the one to quote.

⚠️ NOT A CLAIM ABOUT LOTUSDIS AS A BENCHMARK. It is 3-party meeting speech, not dyadic telephone
(`A21`), so it is *a* spontaneous register rather than *the* call-centre one. What it measures
honestly is the direction and rough size of the penalty.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _thai import fold_sara_am, SCORING_PUNCT  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("ADAPTER_OUT") or (HERE.parent / "results"))
LOTUSDIS = Path("E:/Claude/datasets/LOTUSDIS/data")
URL = "http://127.0.0.1:8007/asr?sr=16000"
SR = 16000
# Ordered by measured CER, not by presumed distance -- the measurement says they do not agree.
MIC_ORDER = ["lav123", "con123", "jbl", "bt3m", "bt10m"]
# 🔴 THE LABEL -> DEVICE MAPPING IS NOT DOCUMENTED AND IS NOT ASSERTED HERE.
# The LOTUSDIS paper (arXiv 2509.18722) says only "nine independent single-channel devices spanning
# six microphone types at distances from 0.12 m to 10 m". Neither the dataset card nor the abstract
# maps `bt3m` or `jbl` to a device. The readings below are what the STRINGS suggest, flagged as
# guesses -- and the measurement contradicts the obvious one anyway: bt3m scores WORSE than bt10m,
# so whatever separates them is not distance.
MIC_NOTE = {"lav123": "(label suggests lavalier)", "con123": "(label suggests conference mic)",
            "jbl": "(label suggests a JBL device)", "bt3m": "(label suggests bluetooth, 3 m?)",
            "bt10m": "(label suggests bluetooth, 10 m?)"}


def norm(s: str, mode: str) -> str:
    s = "".join(s.split())
    if mode == "raw":
        return s
    s = SCORING_PUNCT.sub("", s)
    return fold_sara_am(s) if mode == "fold" else s


def cer(ref: str, hyp: str, mode: str) -> float:
    r, h = norm(ref, mode), norm(hyp, mode)
    if not r:
        return 1.0
    prev = list(range(len(h) + 1))
    for i, cr in enumerate(r, 1):
        cur = [i]
        for j, ch in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cr != ch)))
        prev = cur
    return prev[-1] / len(r)


def transcribe(y: np.ndarray, timeout: float = 120.0) -> dict:
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    req = urllib.request.Request(URL, data=pcm, method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="utterances per mic (150 matches arm1_baseline)")
    ap.add_argument("--min-dur", type=float, default=3.0,
                    help="skip utterances shorter than this -- CER on a ~1.7 s clip is edge effects")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--mics", default=",".join(MIC_ORDER))
    args = ap.parse_args()

    try:
        with urllib.request.urlopen("http://127.0.0.1:8007/health", timeout=10) as r:
            health = json.loads(r.read().decode())
        print(f"asr    : {health}")
    except Exception as e:
        print(f"🔴 :8007 not reachable ({e}). Start asr/run.ps1 first.", file=sys.stderr)
        return 1

    import pyarrow.parquet as pq
    import soundfile as sf

    mics = [m.strip() for m in args.mics.split(",") if m.strip()]
    # One pass over the split, keeping every mic's rows for the chosen utterances. `key` is what
    # makes this controlled: the SAME utterances are scored on every channel.
    print(f"reading LOTUSDIS test split (min_dur {args.min_dur}s) ...", flush=True)
    per_mic: dict[str, list[dict]] = {m: [] for m in mics}
    for p in sorted(glob.glob(str(LOTUSDIS / "test-*.parquet"))):
        f = pq.ParquetFile(p)
        for rg in range(f.num_row_groups):
            for r in f.read_row_group(rg).to_pylist():
                if r["mic"] in per_mic and r["duration"] >= args.min_dur:
                    per_mic[r["mic"]].append(r)

    ref_mic = mics[0]
    n_avail = len(per_mic[ref_mic])
    keys = [(r["speaker_id"], r["sentence"], round(r["duration"], 2)) for r in per_mic[ref_mic]]
    idx = list(range(n_avail))
    random.Random(args.seed).shuffle(idx)
    idx = sorted(idx[: args.n])
    print(f"  {n_avail} utterances >= {args.min_dur}s per mic; sampling {len(idx)} "
          f"(seed {args.seed}, same utterances on every mic)")

    # Verify alignment on the sampled rows rather than trusting the row order.
    for m in mics[1:]:
        k2 = [(r["speaker_id"], r["sentence"], round(r["duration"], 2)) for r in per_mic[m]]
        bad = sum(1 for i in idx if keys[i] != k2[i])
        if bad:
            print(f"🔴 {m} is NOT row-aligned with {ref_mic} on {bad} of {len(idx)} sampled rows. "
                  f"The comparison would not be controlled. Aborting.", file=sys.stderr)
            return 2
    print(f"  alignment verified across {len(mics)} mics on all {len(idx)} sampled rows")

    results: dict[str, dict] = {}
    for m in mics:
        rows, t0 = [], time.perf_counter()
        audio_s = 0.0
        for n, i in enumerate(idx, 1):
            r = per_mic[m][i]
            y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != SR:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            out = transcribe(y)
            hyp = (out.get("text") or "").strip()
            audio_s += len(y) / SR
            rows.append({"speaker": r["speaker_id"], "dur": r["duration"], "ref": r["sentence"],
                         "hyp": hyp,
                         "cer_raw": cer(r["sentence"], hyp, "raw"),
                         "cer_punct": cer(r["sentence"], hyp, "punct"),
                         "cer": cer(r["sentence"], hyp, "fold")})
            if n % 50 == 0:
                med = statistics.median(x["cer"] for x in rows)
                print(f"  [{m}] {n}/{len(idx)}  {time.perf_counter()-t0:.0f}s  running median {med:.4f}",
                      flush=True)
        el = time.perf_counter() - t0
        c = [x["cer"] for x in rows]
        results[m] = {
            "n": len(rows), "audio_s": round(audio_s, 1), "rtf": round(el / audio_s, 4),
            "mean_dur": round(audio_s / len(rows), 2),
            "cer_raw": round(statistics.mean(x["cer_raw"] for x in rows), 4),
            "cer_punct": round(statistics.mean(x["cer_punct"] for x in rows), 4),
            "cer": round(statistics.mean(c), 4), "median": round(statistics.median(c), 4),
            "p90": round(sorted(c)[int(len(c) * .9)], 4), "max": round(max(c), 4),
            "perfect": sum(1 for x in c if x == 0.0),
            "rows": rows,
        }

    print("\n" + "=" * 78)
    print("SPONTANEOUS THAI, and what the microphone costs")
    print("=" * 78)
    print(f"  {'mic':8s} {'':28s} {'CER':>8s} {'median':>8s} {'p90':>7s} {'perfect':>8s} {'RTF':>7s}")
    for m in mics:
        r = results[m]
        print(f"  {m:8s} {MIC_NOTE.get(m, ''):28s} {r['cer']:8.4f} {r['median']:8.4f} "
              f"{r['p90']:7.4f} {r['perfect']:4d}/{r['n']:<3d} {r['rtf']:7.3f}")

    base = results[mics[0]]
    print(f"\n  read speech, same scorer, this box: arm1_baseline CER 0.0518 "
          f"(150 Common Voice clips, ~6.0 s mean)")
    print(f"  spontaneous, closest mic ({mics[0]}): CER {base['cer']:.4f} "
          f"({base['n']} utts, {base['mean_dur']:.1f} s mean)")
    if base["cer"] > 0:
        print(f"  --> the read-vs-spontaneous penalty is {base['cer']/0.0518:.2f}x on this box")
    print(f"\n  ⚠️ duration is still not matched ({base['mean_dur']:.1f}s vs ~6.0s) and the speaker "
          f"sets differ.\n     LOTUSDIS is 3-party meeting speech, not dyadic telephone (A21).")
    print("  ⚠️ FOLD column quoted throughout. Raw and punct are in the json.")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "spontaneous_lotusdis_metrics.json"
    path.write_text(json.dumps({
        "arm": "spontaneous / mic ladder", "corpus": "nectec/LOTUSDIS test split (CC-BY-SA-4.0)",
        "n_per_mic": len(idx), "min_dur": args.min_dur, "seed": args.seed,
        "aligned": True, "asr_health": health,
        "read_baseline": {"source": "arm1_baseline.py", "cer_fold": 0.0518, "n": 150},
        "mics": {m: {k: v for k, v in results[m].items() if k != "rows"} for m in mics},
        "rows": {m: results[m]["rows"] for m in mics},
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
