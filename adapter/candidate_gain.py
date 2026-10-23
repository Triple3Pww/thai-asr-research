#!/usr/bin/env python3
"""Which third option source helps the picker most? Measured on held-out Porjai, NOT the benchmark.

    python adapter/candidate_gain.py porjai_ho_biodatlab_distill-whisper-th-medium porjai_ho_nectec_fc ...

Literature (Consensus, 2026-10-04): combined error ~ average error MINUS diversity (Audhkhasi 2014),
and N-best lists from systems trained on DIFFERENT DATA are the most diverse (Audhkhasi 2013). On the
benchmark's read speech, the strongest systems were other labs' (biodatlab 0.0562, Pathumma 0.0607 vs
the bar 0.0618) -- but those are large. This compares their FAST relatives on Porjai held-out (sentences
never in the picker's training), so the choice is not made on the test set.

Per candidate tag (results/<tag>_nbest.jsonl + _run.json): top-1 CER, per-turn latency, and the
perfect-picker ceilings (sentence / per-spot) of turbo top-5 + realtime top-5 + this model's top-5.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
sys.path.insert(0, str(HERE))
from eval_canon import canon  # noqa: E402
from nbest_lattice_oracle import lattice_oracle, network  # noqa: E402
from pilot_rover import edit, toks  # noqa: E402

MAN = "E:/Claude/datasets/porjai/manifest_heldout.jsonl"


def load(tag):
    return {json.loads(l)["utt_id"]: [toks(canon(c["text"])) for c in json.loads(l)["cands"]]
            for l in (RES / f"{tag}_nbest.jsonl").open(encoding="utf-8")}


def main() -> int:
    ref = {json.loads(l)["utt_id"]: toks(canon(json.loads(l)["text"])) for l in open(MAN, encoding="utf-8")}
    T, R = load("porjai_ho_turbo5"), load("porjai_ho_rt5")
    tot = sum(len(r) for r in ref.values())
    base_s = sum(min(edit(ref[u], c) for c in T[u] + R[u]) for u in ref) / tot
    base_l = sum(lattice_oracle(network(T[u] + R[u]), ref[u]) for u in ref) / tot
    print(f"{'source':44s} {'top1':>7s} {'p50 ms':>7s} {'ceil sent':>9s} {'ceil spot':>9s}")
    print(f"{'turbo5 + rt5 (current)':44s} {'':7s} {'':7s} {base_s:9.4f} {base_l:9.4f}")
    out = {"baseline": {"sentence": base_s, "spot": base_l}}
    for tag in sys.argv[1:]:
        X = load(tag)
        run = json.loads((RES / f"{tag}_run.json").read_text(encoding="utf-8"))
        top1 = sum(edit(ref[u], X[u][0]) for u in ref) / tot
        s = sum(min(edit(ref[u], c) for c in T[u] + R[u] + X[u]) for u in ref) / tot
        l = sum(lattice_oracle(network(T[u] + R[u] + X[u]), ref[u]) for u in ref) / tot
        out[tag] = {"top1": top1, "latency_ms_p50": run.get("latency_ms_p50"), "ceil_sentence": s, "ceil_spot": l}
        print(f"{'+ ' + tag.replace('porjai_ho_', ''):44s} {top1:7.4f} {run.get('latency_ms_p50')!s:>7s} {s:9.4f} {l:9.4f}", flush=True)
    (RES / "candidate_gain_porjai.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
