#!/usr/bin/env python3
"""E3 Pen v0, step 2 (ASR venv): a language model picks one spelling per slot of a same-sound lattice.

    PYTHONUTF8=1 python adapter/pen_score.py E:/Claude/datasets/pen/lattice_nectec_fc_greedy.json

LM = Qwen3-0.6B (base weights, local HF cache), sentence log-probability, no prompt. Left to right, each
slot with alternatives: score the sentence with every alternative swapped in (others fixed at their
current choice); switch only if the best beats the current by more than DELTA nats.

DELTA is CROSS-FITTED, never tuned on the half it is scored on: utterances split in two by a hash of
their id; DELTA picked on fold A from a fixed grid is applied to fold B and vice versa. Reported CER is
the pooled cross-fit result, own Levenshtein against the same normalised reference as
`eval_soundblind.py` (Latin entity spans -> Thai form). Paired bootstrap vs the draft.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _thai import SCORING_PUNCT, fold_sara_am, thai_cardinal  # noqa: E402

BENCH = Path("E:/asr-bench/thai-contextasr-bench")
GRID = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 1e9]       # 1e9 = never switch (the draft itself)
_TH_DIG = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def norm(s: str) -> str:
    s = re.sub(r"(?<=\d),(?=\d{3})", "", (s or "").translate(_TH_DIG))
    s = re.sub(r"\d+", lambda m: thai_cardinal(int(m.group())) if len(m.group()) <= 9 else m.group(), s)
    return SCORING_PUNCT.sub("", fold_sara_am(s)).lower()


def thai_ref(row):
    t = row["text"]
    for e in row.get("entities") or []:
        sp, th = e.get("span_text") or "", e.get("form_th") or ""
        if sp and th and re.search(r"[A-Za-z0-9]", sp):
            t = t.replace(sp, th)
    return t


def lev(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class LM:
    def __init__(self, name="Qwen/Qwen3-0.6B"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval()

    @torch.no_grad()
    def logp(self, texts: list[str]) -> list[float]:
        bos = self.tok.eos_token_id                     # Qwen has no BOS; condition on <|endoftext|>
        ids = [[bos] + self.tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        L = max(map(len, ids))
        x = torch.full((len(ids), L), bos, dtype=torch.long)
        mask = torch.zeros((len(ids), L), dtype=torch.long)
        for i, s in enumerate(ids):
            x[i, : len(s)] = torch.tensor(s); mask[i, : len(s)] = 1
        x, mask = x.cuda(), mask.cuda()
        lp = torch.log_softmax(self.m(input_ids=x, attention_mask=mask).logits.float(), -1)
        tgt = x[:, 1:]
        tok_lp = lp[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask[:, 1:]
        return tok_lp.sum(-1).tolist()


def decode(lm: LM, slots: list[list[str]], delta: float, cache: dict) -> str:
    choice = [s[0] for s in slots]
    if delta >= 1e9:
        return "".join(choice)
    for i, s in enumerate(slots):
        if len(s) < 2:
            continue
        variants = []
        for w in s:
            c = list(choice); c[i] = w
            variants.append("".join(c))
        need = [v for v in variants if v not in cache]
        for k in range(0, len(need), 16):
            for v, p in zip(need[k:k + 16], lm.logp(need[k:k + 16])):
                cache[v] = p
        cur = cache[variants[s.index(choice[i])]]
        best = max(range(len(s)), key=lambda j: cache[variants[j]])
        if cache[variants[best]] - cur > delta:
            choice[i] = s[best]
    return "".join(choice)


def main() -> int:
    lat = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    refs = {}
    for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        if r["utt_id"] in lat:
            refs[r["utt_id"]] = norm(thai_ref(r))
    fold = {u: int(hashlib.md5(u.encode()).hexdigest(), 16) % 2 for u in lat}
    lm = LM()
    out = {d: {} for d in GRID}
    for n, (u, slots) in enumerate(lat.items()):
        cache: dict = {}
        for d in GRID:
            out[d][u] = decode(lm, slots, d, cache)
        if n % 100 == 0:
            print(n, len(lat), flush=True)
    err = {d: {u: lev(refs[u], h) for u, h in out[d].items()} for d in GRID}
    chars = {u: len(refs[u]) for u in lat}
    def cer(d, f):
        us = [u for u in lat if fold[u] == f]
        return sum(err[d][u] for u in us) / sum(chars[u] for u in us)
    pick = {f: min(GRID, key=lambda d: cer(d, 1 - f)) for f in (0, 1)}   # tuned on the OTHER fold
    final = {u: out[pick[fold[u]]][u] for u in lat}
    e_pen = {u: lev(refs[u], final[u]) for u in lat}
    e_draft = err[1e9]
    tot = sum(chars.values())
    rng = random.Random(0)
    us = list(lat)
    wins = 0
    for _ in range(2000):
        s = [rng.choice(us) for _ in us]
        wins += sum(e_pen[u] for u in s) < sum(e_draft[u] for u in s)
    res = {"lattice": sys.argv[1], "draft_cer": sum(e_draft.values()) / tot,
           "pen_cer_crossfit": sum(e_pen.values()) / tot, "delta_by_fold": pick,
           "grid_cer_all": {str(d): sum(err[d].values()) / tot for d in GRID},
           "p_pen_better_bootstrap": wins / 2000,
           "changed_utts": sum(final[u] != out[1e9][u] for u in lat)}
    for k, v in res.items():
        print(f"{k:26s} {v}")
    tag = Path(sys.argv[1]).stem.replace("lattice_", "")
    (HERE.parent / f"results/pen_v0_{tag}_metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    with open(HERE.parent / f"results/pen_v0_{tag}_hyps.jsonl", "w", encoding="utf-8", newline="\n") as g:
        for u, h in final.items():
            g.write(json.dumps({"utt_id": u, "condition": "none", "text": h}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
