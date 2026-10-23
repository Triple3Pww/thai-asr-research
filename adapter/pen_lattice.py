#!/usr/bin/env python3
"""E3 Pen v0, step 1 (thainlp venv): same-sound alternative spellings for every word of a draft.

    E:/thainlp-spike/.venv/Scripts/python.exe adapter/pen_lattice.py dict
    E:/thainlp-spike/.venv/Scripts/python.exe adapter/pen_lattice.py lattice results/nectec_fc_greedy_hyps.jsonl [--ear results/sylhead_s2_bench_units.json]

TEXT-ONLY KNOWLEDGE. The homophone dictionary is built from pythainlp's Thai word list plus the words
of the cached TRAINING transcripts (LOTUSDIS train, Porjai train) -- never from the benchmark. Key =
toned tltk syllables (`ทำ` and `ธรรม` are both `tham0`).

`lattice` writes, per utterance, a list of slots: [[draft_word, alt1, alt2, ...], ...]. Step 2
(`pen_score.py`, ASR venv) lets a language model pick one spelling per slot.

--ear (the Ear's contribution, used only in the Ear arm): for each draft word whose sound DISAGREES with
the Ear's syllables at the aligned position, also offer the dictionary spellings of the Ear's sound.
Without --ear this is the draft-only control the self-review demanded.
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DIR = Path("E:/Claude/datasets/pen")
DICT = DIR / "homophones.json"
_THAI = re.compile(r"^[\u0e00-\u0e7f]+$")


def sound(w: str) -> str:
    from syllable_labels import g2p_syls
    return "~".join(g2p_syls(w))


def _pair(w):
    try:
        return w, sound(w)
    except Exception:
        return w, ""


def cmd_dict() -> None:
    from pythainlp.corpus import thai_words
    from pythainlp.tokenize import word_tokenize
    texts = json.loads(Path("E:/Claude/datasets/ctchead_cache/texts.json").read_text(encoding="utf-8"))
    freq = Counter()
    for n in ["lot_train", "porjai_train"]:
        for t in texts[n]:
            freq.update(w for w in word_tokenize(t, engine="newmm", keep_whitespace=False) if _THAI.match(w))
    words = sorted({w for w in thai_words() if _THAI.match(w)} | set(freq))
    print("words", len(words), flush=True)
    with Pool(10) as p:
        pairs = p.map(_pair, words, chunksize=256)
    by = defaultdict(list)
    for w, s in pairs:
        if s:
            by[s].append(w)
    by = {s: ws for s, ws in by.items() if len(ws) > 1}
    DIR.mkdir(parents=True, exist_ok=True)
    DICT.write_text(json.dumps({"by_sound": by, "freq": freq}, ensure_ascii=False), encoding="utf-8")
    print(f"sounds with 2+ spellings: {len(by)}  words in them: {sum(len(v) for v in by.values())}")


def cmd_lattice(path: str, ear: str | None) -> None:
    from pythainlp.tokenize import word_tokenize
    from eval_soundblind import BENCH, norm
    from syllable_labels import g2p_syls
    D = json.loads(DICT.read_text(encoding="utf-8"))["by_sound"]
    real = {json.loads(l)["utt_id"] for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")
            if json.loads(l)["split"] == "real"}
    E = json.loads(Path(ear).read_text(encoding="utf-8")) if ear else {}
    out, n_alt = {}, 0
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if r["utt_id"] not in real or r.get("condition", "none") != "none":
            continue
        words = word_tokenize(norm(r["text"]), engine="newmm", keep_whitespace=False)
        snd = [g2p_syls(w) if _THAI.match(w) else [] for w in words]
        ear_alts = [[] for _ in words]
        if E.get(r["utt_id"]):
            es = [x for x in _ear_syls(E[r["utt_id"]])]
            flat, owner = [], []
            for i, s in enumerate(snd):
                flat += s; owner += [i] * len(s)
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, flat, es, autojunk=False).get_opcodes():
                if tag == "replace" and i2 - i1 == j2 - j1:          # same number of syllables, different sound
                    ws = sorted(set(owner[i1:i2]))
                    if len(ws) == 1 and len(snd[ws[0]]) == i2 - i1:   # exactly one whole draft word
                        ear_alts[ws[0]] = D.get("~".join(es[j1:j2]), []) or []
        slots = []
        for w, s, ea in zip(words, snd, ear_alts):
            alts = [x for x in D.get("~".join(s), []) if x != w] if s else []
            alts += [x for x in ea if x != w and x not in alts]
            n_alt += bool(alts)
            slots.append([w] + alts[:8])
        out[r["utt_id"]] = slots
    tag = Path(path).stem.replace("_hyps", "") + ("_ear" if ear else "")
    p = DIR / f"lattice_{tag}.json"
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} utts, {n_alt} slots with alternatives -> {p}")


def _ear_syls(units: list[str]) -> list[str]:
    """Ear units [O:kh, R:OON, T0, ...] -> tltk-style syllables 'khOON0' (the dictionary's key format)."""
    out, i = [], 0
    while i < len(units):
        if units[i] == "<oov>" or units[i] == "<unk>":
            i += 1
            continue
        o, r_, t = units[i:i + 3]
        out.append(o[2:].replace("?", "?") + r_[2:].replace("-", "") + t[1:])
        i += 3
    return out


if __name__ == "__main__":
    if sys.argv[1] == "dict":
        cmd_dict()
    else:
        ear = sys.argv[sys.argv.index("--ear") + 1] if "--ear" in sys.argv else None
        cmd_lattice(sys.argv[2], ear)
