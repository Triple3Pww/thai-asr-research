#!/usr/bin/env python3
"""ARM D4 -- context biasing on `typhoon-asr-realtime` via NeMo's GPU boosting tree.

    # dev sweep on the SYNTH split (hyperparameters), then one held-out run on REAL:
    E:/asr-spike/.venv/Scripts/python.exe adapter/arm_d4_boosting.py --split synth --condition bias@50 --sweep
    E:/asr-spike/.venv/Scripts/python.exe adapter/arm_d4_boosting.py --split real  --condition bias@50 --alpha 1.0 --score 3.0

WHY THIS IS NOT A SERVER ARM. `asr/server.py` holds ONE decoding config for its lifetime, and this
experiment needs a DIFFERENT bias list per utterance -- that is what the benchmark's conditions are.
NeMo's per-stream biasing (`enable_per_stream_biasing`) wants trees pre-compiled and addressed by
`multi_model_id`, which fits a server with a few fixed profiles, not 682 unique lists. So this
loads the model in-process and rebuilds the tree per utterance (measured: **3-8 ms** for 50-500
phrases, negligible beside inference).

🔴 THE CONTROL THAT MAKES THAT SAFE. Running in-process is a different harness from the one that
produced every other arm (HTTP to :8007). `--condition none` therefore reproduces the D0 baseline
through THIS code path, and it must land on the published 0.11248 / 0.10571 / 0.26178 / 0.32802
before any biased number here is comparable. Run it first; it is not optional.

🔴 TWO NeMo DEFECTS THIS WORKS AROUND, both found 2026-09-12.

1. **The boosting tree's CUDA path is Triton-only, with no availability check.**
   `GPUBoostingTreeModel.advance()` dispatches to `_advance_triton` whenever
   `use_triton and device == cuda`, and Triton has no Windows wheel -- so it raises
   `ModuleNotFoundError: Module triton required for the function _advance_triton`, mid-decode,
   on a box where everything else works. Its sibling `NGramGPULanguageModel` *does* check and
   falls back with a warning. We force `use_triton = False`, which selects the shipped
   `_advance_pytorch` path. ⚠️ That is the slow path; report RTF, do not assume it.
2. **Hot-swapping `computer.fusion_models[i]` does NOT take effect** -- the computer caches its
   fusion models at construction, so replacing the list element silently keeps decoding with the
   old tree (verified: byte-identical output after swapping in a gold-entity tree). The only
   reliable path is `change_decoding_strategy` per utterance. Measured cost is in the report.

DEV vs HELD-OUT. `alpha` and `context_score` are selected on the **synth** split (1,380 utts, TTS
voices) and reported once on the **real** split (682 utts), which is the split every other number
in this repo uses and which no hyperparameter here has seen. Tuning on real and reporting on real
would be the same error `A38` warns about, and the synth split exists precisely so that the real
one does not have to be spent. ⚠️ Synth is a distribution shift for *hyperparameters*; say so.

⚠️ CC-BY-NC-4.0 -- evaluation only, never training data, never in the product.
"""

from __future__ import annotations

import os
import argparse
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

BENCH = Path("E:/asr-bench/thai-contextasr-bench")
HERE = Path(__file__).resolve().parent
# Results dir -- `results/` at the repo root, beside this `adapter/` package.
# ADAPTER_OUT still overrides it, for a scratch run that must not touch the record.
OUT = Path(os.environ.get("ADAPTER_OUT") or (HERE.parent / "results"))
MODEL = "typhoon-ai/typhoon-asr-realtime"
SR = 16000


def pcm16_roundtrip(y):
    """float32 -> int16 -> float32, exactly as the HTTP path does.

    🔴 NOT cosmetic. Every other arm reaches the model through `:8007`, whose wire format is raw
    PCM16, so the model sees 16-bit-quantised audio. Feeding `soundfile`'s float32 straight in
    changes **23 of 682 transcripts (3.4%)** -- verified 23/23 by replaying exactly those clips
    both ways. Aggregate CER does not move (0.11248 vs 0.11249) and recall is identical, so the
    quantisation is harmless *on average* and would have been invisible in a metrics table; it is
    the per-utterance comparison that exposes it. Doing it here keeps ONE harness rather than two
    that agree in the mean and disagree in detail.
    """
    import numpy as np

    pcm = (np.clip(y, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def load_bias_lists(condition: str) -> dict[str, list[str]]:
    """utt_id -> the frozen bias list for this condition. `none` means no list at all."""
    out: dict[str, list[str]] = {}
    if condition == "none":
        return out
    with gzip.open(BENCH / "data/bias_lists.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("condition") == condition:
                out[d["utt_id"]] = list(d.get("entries") or [])
    return out


class Biaser:
    """Owns the model and swaps the boosting tree per utterance."""

    def __init__(self, device: str = "cuda", model: str = MODEL):
        import nemo.collections.asr as nemo_asr

        print(f"[d4] loading {model} on {device} ...", flush=True)
        # 2026-10-06: --model also takes a local .nemo (NECTEC's FastConformer has no loadable HF id)
        self.model = (nemo_asr.models.ASRModel.restore_from(model, map_location=device) if model.endswith(".nemo")
                      else nemo_asr.models.ASRModel.from_pretrained(model, map_location=device))
        self.model.eval()
        self.device = device
        self._key = None                       # (phrases tuple, alpha, score) currently compiled
        self.rebuilds = 0
        self.rebuild_s = 0.0

    def set_bias(self, phrases: list[str], alpha: float, score: float) -> None:
        # A placeholder phrase keeps the fusion machinery wired while boosting nothing real: an
        # empty list is rejected by `from_config` ("No key phrases file or list specified"), and
        # dropping the tree entirely would change the decoder object between utterances.
        key = (tuple(phrases), alpha, score)
        if key == self._key:
            return
        from omegaconf import OmegaConf, open_dict

        t0 = time.perf_counter()
        cfg = OmegaConf.structured(self.model.cfg.decoding)
        with open_dict(cfg):
            cfg.strategy = "greedy_batch"
            # CUDA graphs cannot capture the fusion path on this box (no Triton); see module doc.
            cfg.greedy.use_cuda_graph_decoder = False
            cfg.greedy.boosting_tree = {"key_phrases_list": phrases or ["zzzzplaceholderzzzz"],
                                        "context_score": score}
            cfg.greedy.boosting_tree_alpha = alpha if phrases else 0.0
        self.model.change_decoding_strategy(cfg)
        for fm in (self.model.decoding.decoding.decoding_computer.fusion_models or []):
            if hasattr(fm, "use_triton"):
                fm.use_triton = False           # defect 1: no availability check upstream
        self._key = key
        self.rebuilds += 1
        self.rebuild_s += time.perf_counter() - t0

    def transcribe(self, y) -> str:
        out = self.model.transcribe([y], verbose=False, batch_size=1)
        if not out:
            return ""
        first = out[0]
        if isinstance(first, (list, tuple)):
            first = first[0] if first else ""
        if isinstance(first, str):
            return first.strip()
        return str(getattr(first, "text", "")).strip()


def run_condition(b: Biaser, rows: list[dict], lists: dict[str, list[str]],
                  alpha: float, score: float, tag: str, condition: str) -> Path:
    import librosa
    import soundfile as sf

    OUT.mkdir(parents=True, exist_ok=True)
    hyps = OUT / f"{tag}_hyps.jsonl"
    audio_s = 0.0
    t0 = time.perf_counter()
    with hyps.open("w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(rows, 1):
            y, sr = sf.read(BENCH / r["wav"], dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            audio_s += len(y) / SR
            y = pcm16_roundtrip(y)
            b.set_bias(lists.get(r["utt_id"], []), alpha, score)
            f.write(json.dumps({"utt_id": r["utt_id"], "condition": condition,
                                "text": b.transcribe(y)}, ensure_ascii=False) + "\n")
            if i % 100 == 0 or i == len(rows):
                print(f"  [{i}/{len(rows)}] {time.perf_counter()-t0:.0f}s", flush=True)
    el = time.perf_counter() - t0
    (OUT / f"{tag}_run.json").write_text(json.dumps(
        {"tag": tag, "condition": condition, "n_utts": len(rows), "alpha": alpha,
         "context_score": score, "audio_h": round(audio_s / 3600, 3),
         "rtf": round(el / audio_s, 4), "tree_rebuilds": b.rebuilds,
         "tree_rebuild_s": round(b.rebuild_s, 1), "device": b.device,
         "align_backend": "dp"}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"  -> {hyps}  ({audio_s/3600:.2f} h, RTF {el/audio_s:.3f}, "
          f"{b.rebuilds} tree rebuilds costing {b.rebuild_s:.0f}s)")
    return hyps


def score(hyps: Path, tag: str) -> dict:
    metrics = OUT / f"{tag}_metrics.json"
    p = subprocess.run([sys.executable, str(HERE / "_score.py"), "--align", "dp",
                        "--hyps", str(hyps), "--out", str(metrics)],
                       capture_output=True, text=True, encoding="utf-8")
    print((p.stdout or "").strip()[-400:] or "(no output)")
    if p.returncode != 0:
        print((p.stderr or "")[-1500:], file=sys.stderr)
        return {}
    return json.loads(metrics.read_text(encoding="utf-8"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="real", choices=("real", "synth"))
    ap.add_argument("--condition", default="none")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--score", type=float, default=3.0)
    ap.add_argument("--sweep", default="", help="comma list of alpha:score pairs, e.g. 1:3,2:3")
    ap.add_argument("--limit", type=int, default=0, help="first N rows -- ordered, so it can skew")
    ap.add_argument("--sample", type=int, default=0,
                    help="seeded random subsample; prefer this to --limit, the corpus is ordered "
                         "by domain and the first N are not representative")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default=MODEL, help="HF id or local .nemo (default: typhoon-asr-realtime)")
    args = ap.parse_args()

    utts = [json.loads(l) for l in
            (BENCH / "data/utterances.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [u for u in utts if u.get("split") == args.split]
    missing = [r for r in rows if not (BENCH / r["wav"]).exists()]
    if missing:
        print(f"🔴 {len(missing)} of {len(rows)} {args.split} clips have no audio.", file=sys.stderr)
        return 1
    if args.sample and args.sample < len(rows):
        import random

        random.Random(20260912).shuffle(rows)
        rows = sorted(rows[: args.sample], key=lambda r: r["utt_id"])
    elif args.limit:
        rows = rows[: args.limit]
    lists = load_bias_lists(args.condition)
    sizes = [len(lists[r["utt_id"]]) for r in rows if r["utt_id"] in lists]
    print(f"split {args.split}: {len(rows)} utts | condition {args.condition} | "
          f"lists for {len(sizes)} utts"
          + (f", entries min/med/max {min(sizes)}/{sorted(sizes)[len(sizes)//2]}/{max(sizes)}"
             if sizes else ""))
    if args.split == "real" and args.sweep:
        print("🔴 REFUSING to sweep on the real split -- that is the held-out report. "
              "Sweep on --split synth.", file=sys.stderr)
        return 2

    b = Biaser(device=args.device, model=args.model)
    combos = ([tuple(float(x) for x in p.split(":")) for p in args.sweep.split(",")]
              if args.sweep else [(args.alpha, args.score)])
    results = []
    for alpha, sc in combos:
        # a non-default model gets its own prefix so it can never overwrite typhoon-rt's d4_* records
        pre = "" if args.model == MODEL else Path(args.model).stem.split("-")[0][:8] + "_"
        tag = args.tag or f"{pre}d4_{args.split}_{args.condition.replace('@','')}_a{alpha:g}_s{sc:g}"
        print(f"\n=== {tag}  alpha={alpha} context_score={sc} ===", flush=True)
        hyps = run_condition(b, rows, lists, alpha, sc, tag, args.condition)
        m = score(hyps, tag).get(args.condition, {})
        if m:
            g = m["counts"]["gold"]
            print(f"  CER {m['cer']:.5f}  U-CER {m['u_cer']:.5f}  B-CER {m['b_cer']:.5f}  "
                  f"recall {m['entity_recall']:.5f} ({g[0]}/{g[1]})")
        results.append({"tag": tag, "alpha": alpha, "context_score": sc, **{
            k: m.get(k) for k in ("cer", "u_cer", "b_cer", "entity_recall")}})

    if len(results) > 1:
        print("\n" + "=" * 78)
        print(f"{'alpha':>6} {'score':>6} {'CER':>9} {'U-CER':>9} {'B-CER':>9} {'recall':>9}")
        for r in results:
            print(f"{r['alpha']:6g} {r['context_score']:6g} " + " ".join(
                f"{(r[k] if r[k] is not None else float('nan')):9.5f}"
                for k in ("cer", "u_cer", "b_cer", "entity_recall")))
        print("\n  U-CER is the harm term. An arm that lifts recall while it rises is a leak.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
