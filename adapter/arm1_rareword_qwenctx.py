#!/usr/bin/env python3
"""ARM 1 rare-word half, re-run on `typhoon-asr-qwen-0.6b-ctx` -- the contextual-biasing model.

    & "E:/asr-spike/.venv/Scripts/python.exe" adapter/arm1_rareword_qwenctx.py

WHY THIS ARM EXISTS. `F8b` is this repo's flattest failure: *"No decode-time arm moved entity
recall. 144 -> 143 -> 143."* Five decode-time arms were built and none of them moved the number.
Typhoon ships a model whose entire purpose is that problem -- `contextual-biasing`, `hotwords`,
a LoRA fine-tune of `Qwen/Qwen3-ASR-0.6B-hf` that takes the term list in its system prompt --
and **this repo has never run it.** The spec even quotes this model's card for Typhoon's CER
table; it took the number off the card and walked past what the model does.

\U0001f534 THIS IS A VERIFICATION, NOT AN EXPLORATION. The card publishes results on **this exact
benchmark with this exact frozen scorer**, which makes every claim directly checkable here:

    typhoon-asr-qwen-0.6b-ctx   none 5.71 CER / .397 recall
                                bias@50   5.25 / .530
                                bias@0    4.01 / .668
    our measured baseline (typhoon-asr-realtime, RNN-T, condition none):
                                11.25 CER / **.328 recall**

A card is not evidence in this repo -- "the third time the failure mode was 'I read the paper'".
So the numbers get reproduced on our harness, our audio, our scorer, or they do not count.

THREE CONDITIONS, and the third is not optional:
  none      no list at all. The baseline, directly comparable to `arm1_rareword`.
  bias@50   51 entries including the gold entity, the benchmark's frozen list.
  dis@50    \U0001f534 **DISTRACTORS ONLY, same list length.** `F8d`/the non-negotiable rule: the
            benchmark's scorer charges insertions to B-CER on a >=2-char substring match against
            the LISTED forms, so a longer list deflates U-CER for free. `bias@50` compared to
            `none` mixes "biasing works" with "a 50-item list changes the scoring". Only
            `bias@50` vs `dis@50` isolates the first. Without this row the arm cannot be quoted.

PROMPT FORMAT IS STRICT (from the card): the system message is the **space-joined** bias list,
then `Thai` on its own line -- or just `Thai` with no list. Getting this wrong degrades silently
into a worse plain-ASR model, which is exactly the kind of failure this repo keeps finding under
its numbers, so the exact string used is recorded in `_run.json`.

ENVIRONMENT: `E:/asr-spike/.venv` (transformers 5.15, torch 2.13.0+cu130, CUDA). The model is
`qwen3_asr`, NOT NeMo, so `asr/server.py` on :8007 cannot serve it -- this arm loads in-process.
\u26a0 That means it does NOT go over the PCM16 wire, which changes 3.4% of transcripts on the NeMo
path. Audio here is float32 straight from the benchmark's wavs, as the card's usage expects.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BENCH = Path("E:/asr-bench/thai-contextasr-bench")
OUT = Path(os.environ.get("ADAPTER_OUT") or (Path(__file__).resolve().parent.parent / "results"))
MODEL_ID = "typhoon-ai/typhoon-asr-qwen-0.6b-ctx"
SR = 16000


def load_bias_lists(conditions: set[str]) -> dict[tuple[str, str], list[str]]:
    """(utt_id, condition) -> the frozen entry list the benchmark ships for it."""
    out = {}
    with gzip.open(BENCH / "data/bias_lists.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("condition") in conditions:
                out[(r["utt_id"], r["condition"])] = r["entries"]
    return out


def build_prompt(entries: list[str] | None) -> str:
    """The card's exact format. Space-joined list, newline, `Thai`; or bare `Thai`."""
    if not entries:
        return "Thai"
    return " ".join(entries) + "\nThai"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="real", choices=("real", "synth", "all"))
    ap.add_argument("--tag", default="arm1_rareword_qwenctx")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--conditions", default="none,bias@50,dis@50")
    ap.add_argument("--align", default="dp", choices=("dp", "kaldialign"))
    ap.add_argument("--max-new-tokens", type=int, default=440)
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    import soundfile as sf
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

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
          f"{len(ents)} gold mentions | {len({e['span_text'] for e in ents})} distinct",
          flush=True)

    listed = load_bias_lists({c for c in conds if c != "none"})
    print(f"lists  : {len(listed)} (utt, condition) pairs loaded for {conds}", flush=True)

    print(f"model  : loading {args.model} ...", flush=True)
    t_load = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    # .to("cuda") rather than the card's `device_map="cuda"`: device_map pulls in `accelerate`,
    # and E:/asr-spike is the venv CLAUDE.md warns against perturbing (NeMo + a torch build the
    # live loop depends on). Same placement, one fewer dependency.
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to("cuda").eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model  : {n_par/1e9:.2f} B params, bf16, cuda | "
          f"{time.perf_counter()-t_load:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    hyps_path = OUT / f"{args.tag}_hyps.jsonl"
    audio_s, n_done, t0 = 0.0, 0, time.perf_counter()
    prompt_samples = {}

    with hyps_path.open("w", encoding="utf-8", newline="\n") as f:
        for cond in conds:
            t_c = time.perf_counter()
            for i, r in enumerate(rows, 1):
                y, sr = sf.read(BENCH / r["wav"], dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if sr != SR:
                    import librosa
                    y = librosa.resample(y, orig_sr=sr, target_sr=SR)
                if cond == conds[0]:
                    audio_s += len(y) / SR

                entries = None if cond == "none" else listed.get((r["utt_id"], cond))
                system_text = build_prompt(entries)
                if cond not in prompt_samples:
                    prompt_samples[cond] = system_text[:300]

                msgs = [{"role": "system", "content": [{"type": "text", "text": system_text}]},
                        {"role": "user", "content": [{"type": "audio", "audio": y}]}]
                inputs = processor.apply_chat_template(
                    [msgs], tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt").to(model.device)
                # The processor emits float32 mel features; the model is bf16. The card's
                # `device_map=` path casts these for you and ours does not, so cast the float
                # tensors explicitly -- ids and masks must stay integer.
                inputs = {k: (v.to(torch.bfloat16)
                              if hasattr(v, "is_floating_point") and v.is_floating_point() else v)
                          for k, v in inputs.items()}
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                gen = gen[:, inputs["input_ids"].shape[1]:]
                text = processor.decode(gen, return_format="transcription_only")[0].strip()

                f.write(json.dumps({"utt_id": r["utt_id"], "condition": cond,
                                    "text": text}, ensure_ascii=False) + "\n")
                n_done += 1
                if i % 100 == 0 or i == len(rows):
                    print(f"  [{cond}] {i}/{len(rows)}  "
                          f"{time.perf_counter()-t_c:.0f}s", flush=True)
            f.flush()

    el = time.perf_counter() - t0
    print(f"\nhyps -> {hyps_path}  ({n_done} rows, {audio_s/3600:.2f} h of audio, "
          f"{el:.0f}s total, RTF {el/max(audio_s*len(conds),1):.3f})", flush=True)

    (OUT / f"{args.tag}_run.json").write_text(json.dumps({
        "tag": args.tag, "model": args.model, "split": args.split,
        "conditions": conds, "n_utts": len(rows), "n_rows": n_done,
        "audio_h": round(audio_s / 3600, 3), "seconds": round(el, 1),
        "dtype": "bfloat16", "device": "cuda", "max_new_tokens": args.max_new_tokens,
        "prompt_format": "space-joined entries + newline + 'Thai'; bare 'Thai' for none",
        "prompt_samples": prompt_samples,
        "params_b": round(n_par / 1e9, 3),
        "align_backend": args.align,
        "in_process": True,
        "note": ("loads in-process (qwen3_asr is not NeMo, :8007 cannot serve it), so audio does "
                 "NOT cross the PCM16 wire that changes 3.4% of transcripts on the NeMo path"),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__},
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    py = sys.executable
    print("\n--- check_hyps ---", flush=True)
    # One condition per invocation: `--conditions` is single-valued, and passing several makes it
    # exit 2 while the scorer then runs anyway -- a validator that silently did not validate.
    for cond in conds:
        chk = subprocess.run([py, str(BENCH / "scripts/check_hyps.py"),
                              "--hyps", str(hyps_path),
                              "--lists", str(BENCH / "data/bias_lists.jsonl.gz"),
                              "--annotations", str(BENCH / "data/utterances.jsonl"),
                              "--conditions", cond],
                             capture_output=True, text=True, encoding="utf-8")
        print(f"[{cond}] " + ((chk.stdout or "").strip() or "(no output)"))
        if chk.returncode != 0:
            print((chk.stderr or "").strip()[:1200])
            print(f"\U0001f534 check_hyps({cond}) exited {chk.returncode} -- read the above.")

    print("\n--- score ---", flush=True)
    metrics_path = OUT / f"{args.tag}_metrics.json"
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

    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    print("\n" + "=" * 78)
    print(f"ARM 1 rare-word -- {args.model}")
    print("=" * 78)
    print(f"  {'condition':12s} {'cer':>8s} {'u_cer':>8s} {'b_cer':>8s} {'recall':>8s} "
          f"{'prec':>8s}")
    for c in conds:
        d = m.get(c, {})
        if not d:
            continue
        def g(k):
            v = d.get(k)
            return f"{v:8.4f}" if isinstance(v, float) else f"{str(v):>8s}"
        print(f"  {c:12s} {g('cer')} {g('u_cer')} {g('b_cer')} {g('entity_recall')} "
              f"{g('entity_precision')}")
    print("\n  baseline for reference (typhoon-asr-realtime, arm1_rareword, none):")
    print("    cer 0.1125   u_cer 0.1057   b_cer 0.2618   entity_recall 0.3280")
    print("\n  \U0001f534 bias@50 is only meaningful against dis@50 at the SAME list length --")
    print("     a longer list deflates U-CER for free (F8d).")
    print(f"\n  -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
