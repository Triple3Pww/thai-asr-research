#!/usr/bin/env python3
"""Token-level n-gram LM over NECTEC's own tokenizer, in the ARPA dialect NeMo's GPU LM reads.

    PYTHONUTF8=1 python adapter/build_lm_arpa.py --n-utts 2000000 --order 4 --out lm/nec_gs2_4g.arpa

WHY (2026-10-11, loop iteration 7). Every training-side try in the loop saturates at ~-10% vs NECTEC; the
decoder is still greedy with no language model. GigaSpeech2-th ships `train_refined.tsv`: 9.9 M machine
transcripts of Thai YouTube speech -- a text corpus ~500x larger than every label NECTEC was tuned on
here, usable WITHOUT downloading the audio. Shallow fusion of an n-gram over it into beam search is
the decoder-side lever that `F8` measured only on Typhoon (beam -8.9%, + its shipped 4-gram best
CER of any arm) and never on NECTEC.

  text   : a deterministic 1-in-k sample of train_refined.tsv -> `gs2_convention.convert` (the training
           labels' convention), + GS2 dev/test training labels, + FLEURS-train labels (Latin names).
  LEAK   : drop any line sharing a 20-character run with ANY of the 2,062 benchmark references
           (lowercased, spaces removed) -- the benchmark is CC-BY-NC, evaluation only.
  tokens : NECTEC's tokenizer (`m.tokenizer.text_to_ids`); written as chr(id + 100), NeMo's
           DEFAULT_TOKEN_OFFSET, one symbol per token, space-separated, with <s> </s>.
  model  : interpolated Kneser-Ney, one discount D = .75 per order, continuation counts for lower
           orders (raw counts where an n-gram has no left context, i.e. starts with <s>), stored as
           ARPA prob + backoff (the interpolation weight). Orders >= 3 pruned to raw count >= 2;
           raw counts make the pruned set prefix- and suffix-closed, which the suffix tree needs.
  n-grams are packed into int64 keys (vocab < 2^15, order <= 4) and counted with numpy, not dicts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gs2_convention import BENCH, LATIN, convert  # noqa: E402

TSV = Path("E:/Claude/datasets/gs2_train/data/th/train_refined.tsv")
EXTRA = [Path("E:/Claude/datasets/gigaspeech2_th/manifest_train_conv.jsonl"),
         Path("E:/Claude/datasets/fleurs_th/manifest_train_conv.jsonl")]
NEC = "E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo"
OFFSET = 100
D = 0.75
BITS = 15


def leak_grams() -> set[str]:
    g = set()
    for l in BENCH.open(encoding="utf-8"):
        t = re.sub(r"\s", "", json.loads(l)["text"]).lower()
        g.update(t[i:i + 20] for i in range(len(t) - 19))
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-utts", type=int, default=2_000_000)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--prune-from", type=int, default=3, help="orders >= this keep raw count >= 2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not sys.flags.utf8_mode:
        raise SystemExit("run with PYTHONUTF8=1")
    t0 = time.time()
    grams = leak_grams()
    total = 9_904_271
    step = max(1, total // args.n_utts)
    texts, drop = [], {"leak": 0, "latin": 0, "empty": 0}
    for i, l in enumerate(TSV.open(encoding="utf-8")):
        if i % step:
            continue
        parts = l.rstrip("\n").split("\t", 1)
        if len(parts) < 2:
            drop["empty"] += 1; continue
        t = convert(parts[1])
        if not t:
            drop["empty"] += 1; continue
        if LATIN.search(t):
            drop["latin"] += 1; continue
        k = t.lower()
        if any(k[j:j + 20] in grams for j in range(len(k) - 19)):
            drop["leak"] += 1; continue
        texts.append(t)
    n_gs2 = len(texts)
    for p in EXTRA:
        for l in p.open(encoding="utf-8"):
            t = json.loads(l)["text"]
            k = re.sub(r"\s", "", t).lower()
            if any(k[j:j + 20] in grams for j in range(len(k) - 19)):
                drop["leak"] += 1; continue
            texts.append(t)
    print(f"text: {n_gs2} GS2-train lines (1 in {step}) + {len(texts) - n_gs2} labels; dropped {drop} "
          f"[{time.time() - t0:.0f}s]", flush=True)

    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.restore_from(NEC, map_location="cpu")
    V = m.tokenizer.vocab_size
    assert V + 2 < (1 << BITS), V
    BOS, EOS = V, V + 1                                    # internal ids for <s>, </s>
    ids = []
    for t in texts:
        ids.append(BOS); ids.extend(m.tokenizer.text_to_ids(t)); ids.append(EOS)
    del m
    seq = np.asarray(ids, dtype=np.int64)
    del ids, texts
    print(f"tokens {len(seq):,} (vocab {V}) [{time.time() - t0:.0f}s]", flush=True)

    # n-gram windows must not cross a sentence boundary: a window is valid if no </s> sits before its
    # last position and no <s> after its first.
    raw = {}
    for n in range(1, args.order + 1):
        L = len(seq) - n + 1
        key = np.zeros(L, dtype=np.int64)
        ok = np.ones(L, dtype=bool)
        for j in range(n):
            w = seq[j:j + L]
            key = (key << BITS) | w
            if j < n - 1:
                ok &= w != EOS
            if j > 0:
                ok &= w != BOS
        if n == 1:
            ok &= seq[:L] != BOS                          # <s> is context only, never predicted
        k, c = np.unique(key[ok], return_counts=True)
        raw[n] = (k, c)
        print(f"order {n}: {len(k):,} unique [{time.time() - t0:.0f}s]", flush=True)
    del seq

    mask = (1 << BITS) - 1

    def ctx(k, n):        # key of the first n-1 symbols
        return k >> BITS

    def suffix(k, n):     # key of the last n-1 symbols
        return k & ((1 << (BITS * (n - 1))) - 1)

    def first(k, n):
        return (k >> (BITS * (n - 1))) & mask

    # counts used for estimation: highest order raw; lower orders continuation counts
    est = {args.order: raw[args.order][1].astype(np.float64)}
    for n in range(1, args.order):
        k, c = raw[n]
        hk = raw[n + 1][0]
        sfx = suffix(hk, n + 1)
        su, sc = np.unique(sfx, return_counts=True)
        pos = np.searchsorted(su, k)
        pos = np.clip(pos, 0, len(su) - 1)
        cont = np.where(su[pos] == k, sc[pos], 0).astype(np.float64)
        starts_bos = first(k, n) == BOS
        est[n] = np.where((cont == 0) | starts_bos, c.astype(np.float64), cont)

    logp, bow = {}, {}
    # unigram
    k1, _ = raw[1]
    c1 = est[1]
    tot = c1.sum()
    gamma0 = D * len(k1) / tot
    p_uni_floor = 1.0 / (V + 1)                            # uniform over real tokens + </s>
    p1 = np.maximum(c1 - D, 0) / tot + gamma0 * p_uni_floor
    probs = {1: p1}
    for n in range(2, args.order + 1):
        k, _ = raw[n]
        c = est[n]
        h = ctx(k, n)
        hu, inv = np.unique(h, return_inverse=True)
        hsum = np.bincount(inv, weights=c)
        htypes = np.bincount(inv)
        gamma = D * htypes / hsum
        # lower-order prob of (suffix) n-1-gram
        lk = raw[n - 1][0]
        sfx = suffix(k, n)
        pos = np.searchsorted(lk, sfx)
        assert np.all(lk[np.clip(pos, 0, len(lk) - 1)] == sfx), f"order {n}: suffix missing"
        plow = probs[n - 1][pos]
        probs[n] = np.maximum(c - D, 0) / hsum[inv] + gamma[inv] * plow
        bow[n - 1] = (hu, gamma)                          # backoff weight lives on the context n-1-gram
        print(f"order {n}: estimated [{time.time() - t0:.0f}s]", flush=True)

    keep = {}
    for n in range(1, args.order + 1):
        keep[n] = raw[n][1] >= 2 if n >= args.prune_from else np.ones(len(raw[n][0]), dtype=bool)
    # all real tokens appear as unigrams (unseen ones get the floor)
    seen = set(raw[1][0].tolist())
    missing = [t for t in range(V) if t not in seen]

    def sym(t):
        return "<s>" if t == BOS else "</s>" if t == EOS else chr(t + OFFSET)

    def syms(k, n):
        return " ".join(sym(int((k >> (BITS * (n - 1 - j))) & mask)) for j in range(n))

    counts = {n: int(keep[n].sum()) for n in range(1, args.order + 1)}
    counts[1] += len(missing) + 1                         # + <s>
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\\data\\\n")
        for n in range(1, args.order + 1):
            f.write(f"ngram {n}={counts[n]}\n")
        for n in range(1, args.order + 1):
            f.write(f"\n\\{n}-grams:\n")
            k = raw[n][0][keep[n]]
            p = probs[n][keep[n]]
            if n < args.order and n in bow:
                hu, g = bow[n]
                pos = np.clip(np.searchsorted(hu, k), 0, len(hu) - 1)
                b = np.where(hu[pos] == k, g[pos], 1.0)
            else:
                b = None
            if n == 1:
                hu, g = bow[1]
                pos = np.searchsorted(hu, BOS)
                gb = g[pos] if pos < len(hu) and hu[pos] == BOS else 1.0
                f.write(f"-99\t<s>\t{math.log10(gb):.6f}\n")
                for t in missing:
                    f.write(f"{math.log10(gamma0 * p_uni_floor):.6f}\t{sym(t)}\n")
            lp = np.log10(p)
            lb = np.log10(b) if b is not None else None
            for i in range(len(k)):
                line = f"{lp[i]:.6f}\t{syms(int(k[i]), n)}"
                if lb is not None:
                    line += f"\t{lb[i]:.6f}"
                f.write(line + "\n")
            print(f"wrote order {n}: {counts[n]:,} [{time.time() - t0:.0f}s]", flush=True)
        f.write("\n\\end\\\n")
    meta = {"n_gs2_lines": n_gs2, "step": step, "dropped": drop, "order": args.order, "counts": counts,
            "vocab": V, "discount": D, "prune_from": args.prune_from}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
