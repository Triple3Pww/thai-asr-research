#!/usr/bin/env python3
"""ARM 1 -- Typhoon as shipped, no fine-tune. The control the whole design is measured against.

    # asr/server.py must be up on :8007
    python adapter/arm1_baseline.py

WHY THIS EXISTS AND WHY IT IS NON-NEGOTIABLE (design S5).

The 2026-08-27 endpointer work produced four fine-tunes and an EER table that could not
distinguish "the method failed" from "this setup cannot learn", because nobody had measured the
untrained model first. Design Sec 3 S5 makes arm 1 mandatory and first for exactly that reason.
This is that measurement.

WHAT SEC 8 ASKS FOR, AND WHAT THIS CAN ACTUALLY DELIVER TODAY.

The gate needs TWO numbers reported together:

    general CER        on real Thai speech  -- the HARM term. An arm that improves rare words
                                              while this rises is a leak, not a win.
    rare-word accuracy on real Thai speech  -- the thing the experiment is trying to buy.

🔴 Only the first is measurable here. The rare-word half needs a real Thai corpus that CONTAINS
rare words, and the candidate (`wayu-ai/thai-contextasr-bench`, 2,062 utts) is not pulled. This
script therefore establishes the harm-term baseline and says so loudly rather than implying arm 1
is complete.

THE CORPUS, AND THREE HONEST LIMITS ON IT.

`data/refs/` -- 150 Common Voice 17.0 Thai clips, CC0, each with its ground-truth sentence.
(Was `smartturn/refs/` until 2026-09-17. The endpointer line still holds its own copy and uses
the same clips as VoxCPM cloning references; this is IMMUTABLE INPUT, never edited, so the two
copies cannot drift. Moving rather than copying would have broken that line.)
Real human speech with real labels, which is what S7 requires. But:

  1. They are DENOISED (`refs.json._denoised: true`). This is clean-speech CER, not telephone CER.
     A deployment number would be worse.
  2. They are read speech from Common Voice, not call-centre speech.
  3. They contain NO rare words, which is why they cannot serve the other metric.

All three mean this number is a valid BASELINE to compare later arms against -- same corpus, same
harness -- and is not a claim about production accuracy.

NORMALISATION. Two CERs are reported because the choice changes the number and hiding it would be
dishonest: `cer` strips whitespace only (the definition already used in `loop/bench/_spike_asr.py`,
so numbers stay comparable with the 0.032 this repo quotes), and `cer_clean` additionally strips
punctuation, which matters because Common Voice sentences carry it and Typhoon emits almost none.
"""

from __future__ import annotations

import os
import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _thai import SCORING_PUNCT, fold_sara_am                      # noqa: E402

REPO = Path(__file__).resolve().parent.parent
REFS = REPO / "data" / "refs"
# Results dir -- `results/` at the repo root, beside this `adapter/` package.
# ADAPTER_OUT still overrides it, for a scratch run that must not touch the record.
OUT = Path(os.environ.get("ADAPTER_OUT") or (Path(__file__).resolve().parent.parent / "results"))
URL = "http://127.0.0.1:8007/asr?sr=16000"
SR = 16000


def cer(ref: str, hyp: str, mode: str = "fold") -> float:
    """Character error rate. Thai has no word spaces, so character level is the only honest
    unit -- a word metric would be measuring a tokeniser, not the ASR.

    Three modes, all reported, because the choice moves the number a lot and picking one
    silently would be the whole problem:
      raw   whitespace stripped only -- matches `loop/bench/_spike_asr.py`, so the 0.032 this
            repo quotes stays comparable
      punct also strips punctuation -- Common Voice carries it, Typhoon emits almost none
      fold  punct, PLUS folding Thai SARA AM. THE HONEST ONE. See `_thai.py`: Typhoon always
            writes the decomposed form and reference corpora always write the composed one, they
            render identically, NFC does not bridge them, and 47 of 150 clips were affected.
    """
    if mode == "raw":
        ref, hyp = ref.replace(" ", ""), hyp.replace(" ", "")
    elif mode == "punct":
        ref, hyp = SCORING_PUNCT.sub("", ref), SCORING_PUNCT.sub("", hyp)
    elif mode == "fold":
        ref, hyp = (SCORING_PUNCT.sub("", fold_sara_am(ref)),
                    SCORING_PUNCT.sub("", fold_sara_am(hyp)))
    else:
        raise ValueError(mode)
    if not ref:
        return float("nan")
    d = np.arange(len(hyp) + 1)
    for i, rc in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(hyp, 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
            prev = cur
    return float(d[len(hyp)]) / len(ref)


def transcribe(y16: np.ndarray, timeout: float = 120.0) -> dict:
    pcm = (np.clip(y16, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    req = urllib.request.Request(URL, data=pcm,
                                headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    # Thai and the report's own red flags on a cp1252 console: without this the whole
    # run dies at the LAST print, after the measurement and before the result file.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="arm1_typhoon_shipped")
    args = ap.parse_args()

    import librosa
    import soundfile as sf

    meta = json.loads((REFS / "refs.json").read_text(encoding="utf-8"))
    items = meta["items"][: args.limit] if args.limit else meta["items"]
    print(f"corpus : {len(items)} Common Voice Thai clips  ({meta.get('_source')})")
    print(f"        denoised={meta.get('_denoised')}  native sr={meta.get('_sr')}")

    # The server's own account of what it is running. Recorded into the result file, not just
    # printed: once there is more than one arm, a result whose decoding config is only in a
    # console scrollback is a result nobody can place.
    try:
        with urllib.request.urlopen("http://127.0.0.1:8007/health", timeout=10) as r:
            health = json.loads(r.read().decode())
        print(f"asr    : {health}")
    except Exception as e:
        print(f"🔴 :8007 not reachable ({e}). Start asr/server.py first.", file=sys.stderr)
        return 1

    rows = []
    t_start = time.perf_counter()
    for i, it in enumerate(items, 1):
        y, sr = sf.read(REFS / it["wav"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        y16 = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
        res = transcribe(y16)
        hyp, ref = res["text"], it["text"]
        rows.append({"id": it["id"], "ref": ref, "hyp": hyp,
                     "cer_raw": cer(ref, hyp, "raw"),
                     "cer_punct": cer(ref, hyp, "punct"),
                     "cer": cer(ref, hyp, "fold"),
                     "audio_s": res["audio_s"], "infer_ms": res["infer_ms"]})
        if i % 25 == 0 or i == len(items):
            el = time.perf_counter() - t_start
            print(f"  [{i}/{len(items)}] {el:.0f}s  running median CER "
                  f"{statistics.median(r['cer'] for r in rows):.4f}", flush=True)

    raw = sorted(r["cer_raw"] for r in rows)
    pun = sorted(r["cer_punct"] for r in rows)
    c = sorted(r["cer"] for r in rows)
    audio_s = sum(r["audio_s"] for r in rows)

    def pct(v, p):
        return v[min(len(v) - 1, int(p / 100 * len(v)))]

    print("\n" + "=" * 74)
    # 🔴 THIS LINE SAID "CPU" UNTIL 2026-09-17, AND IT WAS WRONG THE WHOLE TIME (F8c).
    # `from_pretrained` without `map_location` lands on cuda:0, so every "RTF on CPU" figure in
    # the older docs is a GPU figure. Print what /health actually reports rather than a constant
    # -- a label that cannot go stale is worth more than one that happens to be right today.
    dev = ((health.get("decoding") or {}).get("device") or "?") if isinstance(health, dict) else "?"
    print(f"ARM 1 -- Typhoon as shipped (typhoon-asr-realtime, :8007, {dev})")
    print("=" * 74)
    print(f"  clips {len(rows)}   audio {audio_s/60:.1f} min   "
          f"RTF {sum(r['infer_ms'] for r in rows)/1000/audio_s:.3f}")
    for lbl, v in (("raw   (whitespace only)", raw), ("punct (+ punctuation)", pun),
                   ("FOLD  (+ SARA AM)  <--", c)):
        print(f"  CER {lbl:24s} mean {statistics.mean(v):.4f}  median "
              f"{statistics.median(v):.4f}  p90 {pct(v, 90):.4f}  max {v[-1]:.4f}")
    print(f"  perfect transcripts          raw {sum(1 for x in raw if x == 0)}  "
          f"punct {sum(1 for x in pun if x == 0)}  FOLD {sum(1 for x in c if x == 0)}"
          f"  of {len(rows)}")
    print("\n  FOLD is the number to quote. Typhoon always writes Thai SARA AM decomposed and")
    print("  reference corpora always write it composed; the two render identically and Unicode")
    print("  NFC does not bridge them. 47 of 150 clips here. Details in `_thai.py`.")
    print("\n  🔴 THIS IS THE HARM-TERM BASELINE ONLY. Sec 8's other metric -- rare-word")
    print("     accuracy -- needs a real Thai corpus containing rare words, which is not")
    print("     pulled. Arm 1 is NOT complete until that number exists beside this one.")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.tag}.json"
    path.write_text(json.dumps({
        "arm": 1, "label": "typhoon as shipped, no fine-tune",
        "model": "typhoon-asr-realtime", "endpoint": URL, "asr_health": health,
        "rtf": round(sum(r["infer_ms"] for r in rows) / 1000 / audio_s, 4),
        "corpus": {"name": "data/refs (Common Voice 17.0 th test, CC0; was smartturn/refs before 2026-09-17)",
                   "n": len(rows), "audio_s": round(audio_s, 1),
                   "denoised": meta.get("_denoised"),
                   "caveats": ["denoised -> clean-speech CER, not telephone",
                               "read speech, not call-centre speech",
                               "contains no rare words -> cannot measure rare-word accuracy"]},
        "cer": {"mean": statistics.mean(c), "median": statistics.median(c),
                "p90": pct(c, 90), "max": c[-1],
                "_mode": "punctuation stripped + Thai SARA AM folded; quote this one"},
        "cer_punct": {"mean": statistics.mean(pun), "median": statistics.median(pun)},
        "cer_raw": {"mean": statistics.mean(raw), "median": statistics.median(raw),
                    "_mode": "whitespace only; comparable with loop/bench/_spike_asr.py"},
        "rare_word_accuracy": None,
        "rare_word_blocked_by": "wayu-ai/thai-contextasr-bench not pulled",
        "rows": rows,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
