#!/usr/bin/env python3
"""ARM 1, second half -- rare-word accuracy on the Thai contextual-biasing benchmark.

    # asr/server.py must be up on :8007
    python adapter/arm1_rareword.py

WHY. Design Sec 8 gates every arm on TWO numbers reported together: rare-word accuracy (the thing
the experiment buys) and general CER (the harm term -- an arm that improves rare words while this
rises is a leak, not a win). `arm1_baseline.py` measured the harm term on Common Voice, which has
no rare words. This measures both on a corpus built for the purpose.

THE BENCHMARK. `wayu-ai/thai-contextasr-bench` (Wayu AI, July 2026, CC-BY-NC-4.0): 2,062 Thai
utterances, each annotated with the named entities actually spoken in it -- brands, person names,
places -- plus a frozen bias list per condition, and **its own scorer**. That scorer already
reports exactly the split Sec 8 wants:

    U-CER           errors OUTSIDE gold-entity spans   -> the harm term
    B-CER           errors INSIDE gold-entity spans    -> rare-word damage
    entity_recall   gold entities transcribed at all   -> the rare-word metric
                    (dual-form crediting: Thai OR Latin spelling both count)

plus bootstrap 95% CIs. Using their metric rather than inventing one also makes our number
directly comparable to the six baselines they publish -- and `typhoon-asr-realtime` is NOT among
those six, so this fills a real gap.

🔴 FOUR THINGS THAT CONSTRAIN WHAT THIS NUMBER MEANS. Read before quoting it.

1. **REAL SPLIT ONLY -- 682 of 2,062 utterances.** The other 1,380 are OmniVoice TTS with cloned
   voices. This whole experiment is about whether synthetic speech can teach an ASR; evaluating it
   on synthetic audio would be circular, and design S7 says real speech. `split == "real"` is
   therefore not a convenience, it is a correctness requirement. 682 utts, 298 entity-bearing,
   439 gold mentions, 379 distinct entities.

2. **Condition `none` only.** `typhoon-asr-realtime` takes audio and returns text; it has no bias
   list input, so the nine biasing conditions are not applicable. `none` is also the right
   condition for our purpose: we want the UNBIASED rare-word accuracy that the adapter is supposed
   to improve.

3. 🔴 **The benchmark's own authors declare split A may be CONTAMINATED** -- it is old YouTube
   audio, so a model may have trained on it. Their stated mitigation is to quote
   context-vs-no-context DELTAS on identical audio. We cannot form that delta, because our model
   has only one condition. So this absolute number carries a contamination risk we cannot remove.
   It remains valid for what Sec 8 actually needs -- arm-vs-arm on identical audio and harness,
   where contamination is a constant -- and it is NOT a claim about Typhoon's true ability.

4. ⚠️ **CC-BY-NC-4.0: NonCommercial.** Fine for evaluation and for a thesis. NOT fine as training
   data, and not fine inside a commercial product. This repo is a business repo; keep this corpus
   on the evaluation side of that line.
"""

from __future__ import annotations

import os
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

BENCH = Path("E:/asr-bench/thai-contextasr-bench")
# Results dir -- `results/` at the repo root, beside this `adapter/` package.
# ADAPTER_OUT still overrides it, for a scratch run that must not touch the record.
OUT = Path(os.environ.get("ADAPTER_OUT") or (Path(__file__).resolve().parent.parent / "results"))
URL = "http://127.0.0.1:8007/asr?sr=16000"
SR = 16000
CONDITION = "none"


def transcribe(y16: np.ndarray, timeout: float = 180.0) -> dict:
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
    ap.add_argument("--split", default="real", choices=("real", "synth", "all"))
    ap.add_argument("--tag", default="arm1_rareword")
    ap.add_argument("--align", default="dp", choices=("dp", "kaldialign"),
                    help="alignment backend for B-CER/U-CER attribution; see adapter/_score.py")
    args = ap.parse_args()

    import soundfile as sf

    rows = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")
            if l.strip()]
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]
    rows = [r for r in rows if (BENCH / r["wav"]).exists()]
    if args.limit:
        rows = rows[: args.limit]

    ents = [e for r in rows for e in r["entities"]]
    print(f"corpus : {len(rows)} utts (split={args.split}) | "
          f"{sum(1 for r in rows if not r['entity_free'])} entity-bearing | "
          f"{len(ents)} gold mentions | {len({e['span_text'] for e in ents})} distinct")

    try:
        with urllib.request.urlopen("http://127.0.0.1:8007/health", timeout=10) as r:
            health = json.loads(r.read().decode())
        print(f"asr    : {health}")
    except Exception as e:
        print(f":8007 not reachable ({e}). Start asr/server.py first.", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    hyps_path = OUT / f"{args.tag}_hyps.jsonl"
    audio_s = 0.0
    t0 = time.perf_counter()
    with hyps_path.open("w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(rows, 1):
            y, sr = sf.read(BENCH / r["wav"], dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != SR:                     # the benchmark ships 16 kHz; guard, do not assume
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            res = transcribe(y)
            audio_s += res["audio_s"]
            f.write(json.dumps({"utt_id": r["utt_id"], "condition": CONDITION,
                                "text": res["text"]}, ensure_ascii=False) + "\n")
            if i % 100 == 0 or i == len(rows):
                el = time.perf_counter() - t0
                print(f"  [{i}/{len(rows)}] {el:.0f}s", flush=True)
    rtf = (time.perf_counter() - t0) / audio_s
    print(f"\nhyps -> {hyps_path}  ({audio_s/3600:.2f} h of audio, RTF {rtf:.3f})")
    # Sidecar: the scorer writes only its own metrics, so the run's provenance -- which
    # decoding config produced these hypotheses, and what it cost -- has nowhere else to live.
    (OUT / f"{args.tag}_run.json").write_text(json.dumps(
        {"tag": args.tag, "split": args.split, "condition": CONDITION, "n_utts": len(rows),
         "audio_h": round(audio_s / 3600, 3), "rtf": round(rtf, 4), "asr_health": health,
         "align_backend": args.align},
        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # Their validator first: the scorer does not error on duplicate or missing rows, it silently
    # under-reports. Their README says to check before scoring, so check before scoring.
    py = sys.executable
    print("\n--- check_hyps ---", flush=True)
    chk = subprocess.run([py, str(BENCH / "scripts/check_hyps.py"),
                          "--hyps", str(hyps_path),
                          "--lists", str(BENCH / "data/bias_lists.jsonl.gz"),
                          "--annotations", str(BENCH / "data/utterances.jsonl"),
                          "--conditions", CONDITION],
                         capture_output=True, text=True, encoding="utf-8")
    print((chk.stdout or "").strip() or "(no output)")
    if chk.returncode != 0:
        print((chk.stderr or "").strip()[:1500])
        print(f"🔴 check_hyps exited {chk.returncode} -- scoring anyway, but read the above.")

    print("\n--- score ---", flush=True)
    metrics_path = OUT / f"{args.tag}_metrics.json"
    # Through `_score.py`, never score.py directly: the benchmark's span metrics silently
    # depend on whether `kaldialign` is importable in the interpreter that runs it (B-CER moves
    # 0.0113 on byte-identical hypotheses). `dp` is the path every number already in this repo
    # used. See `adapter/_score.py`.
    sc = subprocess.run([py, str(Path(__file__).resolve().parent / "_score.py"),
                         "--align", args.align,
                         "--annotations", str(BENCH / "data/utterances.jsonl"),
                         "--lists", str(BENCH / "data/bias_lists.jsonl.gz"),
                         "--hyps", str(hyps_path),
                         "--out", str(metrics_path)],
                        capture_output=True, text=True, encoding="utf-8")
    print((sc.stdout or "").strip() or "(no output)")
    if sc.returncode != 0:
        print((sc.stderr or "").strip()[:2000], file=sys.stderr)
        return sc.returncode

    m = json.loads(metrics_path.read_text(encoding="utf-8")).get(CONDITION, {})
    print("\n" + "=" * 74)
    print("ARM 1 -- rare-word half (typhoon-asr-realtime as shipped, condition 'none')")
    print("=" * 74)
    for k in ("n_utts", "cer", "u_cer", "b_cer", "entity_recall", "entity_precision",
              "entity_f1", "listed_form_compliance"):
        if k in m and m[k] is not None:
            v = m[k]
            print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")
    for k, ci in (m.get("ci95") or {}).items():
        if ci:
            print(f"  ci95[{k}]{'':14s} {ci}")
    print("\n  U-CER is the harm term; B-CER and entity_recall are the rare-word metric.")
    print("  Later arms must raise recall WITHOUT U-CER rising above this line.")
    print(f"\n  -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
