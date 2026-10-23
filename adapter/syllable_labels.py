#!/usr/bin/env python3
"""E1 + E2 labels of the Ear-Memory-Pen plan: Thai text -> sound units, and how ambiguous sound -> spelling is.

    E:/thainlp-spike/.venv/Scripts/python.exe adapter/syllable_labels.py labels     # cache texts -> unit ids
    E:/thainlp-spike/.venv/Scripts/python.exe adapter/syllable_labels.py ambiguity  # E1: P2G ambiguity

🔴 Runs in `E:/thainlp-spike/.venv` (tltk needs numpy<2). Reads only the `texts` lists the CTC-head
cache already holds (`E:/Claude/datasets/ctchead_cache/*.pt` via a JSON sidecar written by the ASR
venv -- torch is not installed here), writes `syllable_units.json` beside the cache.

UNITS. Every Thai syllable from tltk's G2P (`khlOON0`) is split into THREE CTC tokens:
onset (`kh` `l` -> the cluster `khl`), rhyme (`OON`), tone (`T0`..`T4`). Factored rather than one token
per whole syllable because Thai has thousands of distinct toned syllables and the corpus is ~100 h --
three small inventories (dozens of onsets, a few hundred rhymes, 5 tones) are all well covered.
Non-Thai runs (Latin, digits) map to one `<oov>` unit, so the head is never trained to spell English.

E1 AMBIGUITY. For every word (pythainlp newmm segmentation) record its sound (toned syllables); count
how many DIFFERENT spellings share one sound, token-weighted. That is the share of words where a
sound-only Ear leaves the Pen a real choice -- the Pen's job size, from text alone.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

CACHE = Path("E:/Claude/datasets/ctchead_cache")
SIDE = CACHE / "texts.json"                       # written by the ASR venv: {set: [text, ...]}
OUT = CACHE / "syllable_units.json"
CONS = set("pbtdkcfshmnNjwrl?")
_THAI = re.compile(r"[\u0e00-\u0e7f]+")


def split_syl(s: str) -> list[str]:
    m = re.match(r"^(.*?)(\d)$", s)
    body, tone = (m.group(1), m.group(2)) if m else (s, "0")
    k = 0
    while k < len(body) and body[k] in CONS:
        k += 1
    if k == len(body) and k > 1:                  # no vowel letter: keep last consonant as rhyme
        k -= 1
    onset, rhyme = body[:k] or "?", body[k:] or "-"
    return ["O:" + onset, "R:" + rhyme, "T" + tone]


def g2p_syls(text: str) -> list[str]:
    from tltk.nlp import g2p
    out = []
    for word in g2p(text).split("<s/>"):
        if "<tr/>" in word:
            ph = word.split("<tr/>")[1].strip().strip("|")
            out += [p for p in ph.replace("|", "~").split("~") if p]
    return out


def units(text: str) -> list[str]:
    out = []
    for piece in re.findall(r"[\u0e00-\u0e7f]+|[^\u0e00-\u0e7f\s]+", text):
        if _THAI.fullmatch(piece):
            for s in g2p_syls(piece):
                out += split_syl(s)
        elif out[-1:] != ["<oov>"]:
            out.append("<oov>")
    return out


def word_sounds(text: str) -> list[tuple[str, str]]:
    from pythainlp.tokenize import word_tokenize
    out = []
    for w in word_tokenize(text, engine="newmm", keep_whitespace=False):
        if _THAI.fullmatch(w):
            out.append((w, "~".join(g2p_syls(w))))
    return out


def cmd_labels() -> None:
    texts = json.loads(SIDE.read_text(encoding="utf-8"))
    res = {}
    with Pool(10) as p:
        for name, ts in texts.items():
            res[name] = p.map(units, ts, chunksize=64)
            print(name, len(ts), flush=True)
    vocab = Counter(u for seqs in res.values() for s in seqs for u in s)
    itos = sorted(vocab)
    OUT.write_text(json.dumps({"itos": itos, "counts": vocab, "units": res}, ensure_ascii=False), encoding="utf-8")
    kinds = Counter(u[:2] if u[1] == ":" else u[:1] for u in itos)
    print(f"vocab {len(itos)}  by kind {dict(kinds)}  -> {OUT}")


def cmd_ambiguity() -> None:
    texts = json.loads(SIDE.read_text(encoding="utf-8"))
    allt = [t for name, ts in texts.items() if name != "lot_val" for t in ts]
    with Pool(10) as p:
        pairs = [x for ws in p.map(word_sounds, allt, chunksize=64) for x in ws]
    by_sound = defaultdict(Counter)
    for w, snd in pairs:
        by_sound[snd][w] += 1
    tok = len(pairs)
    amb_tok = sum(sum(c.values()) for c in by_sound.values() if len(c) > 1)
    # tokens a sound-only Pen gets wrong if it always writes the most frequent spelling (no context)
    unigram_miss = sum(sum(c.values()) - max(c.values()) for c in by_sound.values())
    toneless = defaultdict(Counter)
    for w, snd in pairs:
        toneless[re.sub(r"\d", "", snd)][w] += 1
    amb_tl = sum(sum(c.values()) for c in toneless.values() if len(c) > 1)
    top = sorted(((sum(c.values()), s, c.most_common(4)) for s, c in by_sound.items() if len(c) > 1), reverse=True)[:25]
    rep = {"word_tokens": tok, "distinct_sounds": len(by_sound),
           "tokens_whose_sound_has_2plus_spellings": amb_tok / tok,
           "context_free_pen_word_error": unigram_miss / tok,
           "same_without_tone_2plus_spellings": amb_tl / tok,
           "top_ambiguous": [(n, s, c) for n, s, c in top]}
    (Path(__file__).resolve().parent.parent / "results/p2g_ambiguity_metrics.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in rep.items():
        if k != "top_ambiguous":
            print(f"{k:42s} {v if isinstance(v, int) else f'{v:.4f}'}")
    for n, s, c in top:
        print(f"  {n:6d}  {s:20s} {c}")


def _lev(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cmd_score() -> None:
    """E2 verdict: sound-unit error of the Ear head vs G2P(each system's transcript), same reference.

        ... syllable_labels.py score results/sylhead_s1_bench_units.json [hyps.jsonl ...]
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_soundblind import BENCH, SYSTEMS, norm, thai_ref
    head = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    refs = {}
    for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        if r["split"] == "real":
            refs[r["utt_id"]] = units(norm(thai_ref(r)))
    systems = {"EAR head (" + Path(sys.argv[2]).stem + ")": head}
    files = {k: SYSTEMS[k] for k in ["NECTEC-fastconformer", "typhoon-whisper-large-v3", "NECTEC+LOTUSDIS-conv (A2)"]}
    files.update({Path(x).stem: Path(x) for x in sys.argv[3:]})
    for k, p in files.items():
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        systems[k] = {r["utt_id"]: units(norm(r["text"])) for r in rows
                      if r["utt_id"] in refs and r.get("condition", "none") == "none"}
    def strip(us, keep):
        return [u for u in us if u[0] in keep or u == "<oov>"]
    res = {}
    for k, hy in systems.items():
        n = e = et = eo = nt = no = 0
        for uid, ru in refs.items():
            hu = hy.get(uid, [])
            n += len(ru); e += _lev(ru, hu)
            nt += len(strip(ru, "T")); et += _lev(strip(ru, "T"), strip(hu, "T"))
            no += len(strip(ru, "OR")); eo += _lev(strip(ru, "OR"), strip(hu, "OR"))
        res[k] = {"unit_err": e / n, "tone_err": et / nt, "onset_rhyme_err": eo / no}
        print(f"{k[:34]:34s} unit err {e / n:.4f}   onset+rhyme {eo / no:.4f}   tone {et / nt:.4f}")
    (Path(__file__).resolve().parent.parent / f"results/{Path(sys.argv[2]).stem}_score_metrics.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"labels": cmd_labels, "ambiguity": cmd_ambiguity, "score": cmd_score}[sys.argv[1]]()
