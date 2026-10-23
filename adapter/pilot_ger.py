#!/usr/bin/env python3
"""Pilot: generative error correction -- a local Thai LLM reads N systems' hypotheses and writes one.

    python adapter/pilot_ger.py --model scb10x/typhoon2.1-gemma3-12b --limit 60
    python adapter/pilot_ger.py --pool tw pat q17 isan bio --rover results/rover_uniform_..._hyps.jsonl

WHAT THIS TESTS. ROVER can only choose among tokens the systems emitted at an aligned position. A
language model can also repair a word that every system split differently, and knows Thai
spelling. HyPoradise (NeurIPS 2023) reports fine-tuned GER can beat the N-best oracle; zero-shot is
weaker. This is ZERO-SHOT, via Ollama, label-free: the prompt contains only hypotheses.

Guard: an LLM can paraphrase. If the output's token edit distance to the ROVER consensus exceeds
`--max-drift` x its length, the ROVER consensus is emitted instead (counted and reported).
Hypotheses are ordered by the pool order, never by any quality signal derived from a reference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot_mbr_combine import POOL, load  # noqa: E402
from pilot_rover import BENCH, edit, toks  # noqa: E402

PROMPT = """คุณคือผู้ตรวจแก้ข้อความถอดเสียงภาษาไทย ด้านล่างคือผลถอดเสียงจากระบบรู้จำเสียงหลายระบบ ของคลิปเสียงเดียวกัน แต่ละระบบผิดคนละจุด
งานของคุณ: เขียนข้อความที่ถูกต้องที่สุดเพียงหนึ่งบรรทัด โดยรวมส่วนที่ระบบส่วนใหญ่เห็นตรงกัน และแก้คำสะกดผิดหรือคำที่ไม่มีความหมาย
กฎ: ห้ามแปล ห้ามสรุป ห้ามเพิ่มเนื้อหาที่ไม่มีในผลถอดเสียง ห้ามอธิบาย ตอบเฉพาะข้อความที่แก้แล้วเท่านั้น
ตัวเลขให้เขียนแบบเดียวกับที่ระบบส่วนใหญ่ใช้

ผลรวมจากการโหวต: {rover}
{cands}

ข้อความที่ถูกต้อง:"""


def ask(model: str, prompt: str) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "think": False,
                       "options": {"temperature": 0, "num_predict": 512, "seed": 2026}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())["response"].strip()
    return out.splitlines()[0].strip() if out else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="scb10x/typhoon2.1-gemma3-12b")
    ap.add_argument("--pool", nargs="+", default=["tw", "pat", "q17", "isan", "bio"])
    ap.add_argument("--rover", default="results/rover_uniform_tw+pat+q17+isan+bio_hyps.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-drift", type=float, default=0.3)
    ap.add_argument("--tag", default="ger_pilot")
    args = ap.parse_args()

    ann = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")]
    real = sorted(a["utt_id"] for a in ann if a["split"] == "real")
    if args.limit:                       # deterministic spread over all three source groups
        real = real[:: max(1, len(real) // args.limit)][: args.limit]
    keep = set(real)
    hyps = {k: load(POOL[k], keep) for k in args.pool}
    rover = {json.loads(l)["utt_id"]: json.loads(l)["text"] for l in Path(args.rover).open(encoding="utf-8")}

    out_p = HERE.parent / f"results/{args.tag}_hyps.jsonl"
    done = {}
    if out_p.exists():
        done = {json.loads(l)["utt_id"]: json.loads(l) for l in out_p.open(encoding="utf-8")}
    t0, fell = time.time(), 0
    with out_p.open("a", encoding="utf-8", newline="\n") as f:
        for i, u in enumerate(real):
            if u in done:
                continue
            cands = "\n".join(f"ระบบที่ {j + 1}: {hyps[k][u]}" for j, k in enumerate(args.pool))
            try:
                txt = ask(args.model, PROMPT.format(rover=rover[u], cands=cands))
            except Exception as e:  # noqa: BLE001
                print(f"  {u}: ollama error {e}", flush=True)
                txt = ""
            rt, gt = toks(rover[u]), toks(txt)
            drift = edit(rt, gt) / max(len(rt), 1)
            used = "llm"
            if not gt or drift > args.max_drift:
                txt, used = rover[u], "rover_fallback"
                fell += 1
            f.write(json.dumps({"utt_id": u, "condition": "none", "text": txt, "used": used,
                                "drift": round(drift, 3)}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(real)}  {time.time() - t0:.0f}s  fallbacks {fell}", flush=True)
    print(f"done {len(real)} utts, fallbacks {fell}, {time.time() - t0:.0f}s -> {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
