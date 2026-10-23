#!/usr/bin/env python3
"""Has the correcting LLM memorised the benchmark's sentences? A prefix-completion probe.

    python adapter/probe_contamination.py --model qwen3.5:9b

WHY. `pilot_ger.py` gains most where an LLM could know the answer without listening: the
`c-validation` / `c-test` utterances read like FLEURS / Common Voice sentences, whose text is on
the public web (arXiv 2605.19711 makes exactly this point for West Frisian). If the LLM can
continue a reference sentence from its first half, a GER gain on that sentence is recall from
pre-training, not correction.

Per utterance: show the first half of the REFERENCE (by character), ask for the continuation,
score the continuation against the true second half by CER. A memorised sentence comes back near
CER 0; an unseen one near 1. Reported per source group, with the conversational `a` group -- not
public text -- as the in-house control. This probe READS the reference by design; it is a
diagnostic, never a system.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot_ger import ask  # noqa: E402
from pilot_rover import BENCH, edit, toks  # noqa: E402

PROMPT = "ต่อประโยคภาษาไทยต่อไปนี้ให้จบ ตอบเฉพาะส่วนที่ต่อ ไม่ต้องทวนส่วนแรก:\n{prefix}"


def group(u):
    return "c-validation" if u.startswith("c-validation") else "c-test" if u.startswith("c-test") else "a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--per-group", type=int, default=40)
    args = ap.parse_args()
    ann = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")]
    real = sorted((a for a in ann if a["split"] == "real"), key=lambda a: a["utt_id"])
    out = {}
    rows = []
    for g in ("a", "c-test", "c-validation"):
        pick = [a for a in real if group(a["utt_id"]) == g and len(a["text"]) >= 40]
        pick = pick[:: max(1, len(pick) // args.per_group)][: args.per_group]
        cers, exact = [], 0
        for a in pick:
            t = a["text"]
            cut = len(t) // 2
            prefix, rest = t[:cut], t[cut:]
            cont = ask(args.model, PROMPT.format(prefix=prefix))
            r, h = toks(rest), toks(cont)[: len(toks(rest)) + 5]
            c = edit(r, h) / max(len(r), 1)
            cers.append(min(c, 1.0))
            exact += c < 0.1
            rows.append({"utt_id": a["utt_id"], "group": g, "cer": round(c, 3), "rest": rest, "cont": cont})
        out[g] = {"n": len(pick), "mean_cont_cer": sum(cers) / len(cers), "near_verbatim(<0.1)": exact}
        print(f"{g:13s} n={len(pick)}  mean continuation CER {out[g]['mean_cont_cer']:.3f}  "
              f"near-verbatim {exact}", flush=True)
    p = HERE.parent / f"results/probe_contamination_{args.model.replace(':', '_').replace('/', '_')}.json"
    p.write_text(json.dumps({"summary": out, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
