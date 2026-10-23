#!/usr/bin/env python3
"""E0 of the Ear-Memory-Pen plan: score every system blind to SCRIPT and, separately, to SPELLING.

    E:/thainlp-spike/.venv/Scripts/python.exe adapter/eval_soundblind.py [extra_hyps.jsonl ...]

🔴 Runs in `E:/thainlp-spike/.venv` (pythainlp + tltk), NOT the ASR venv -- tltk needs numpy<2 and
pandas<3, the ASR venv runs numpy 2.4.6 under NeMo. Pure text, no GPU, no server.

WHY (`F38`, 2026-10-08). 23% of NECTEC's character edits are a loanword written in Latin on one side
and Thai on the other (`wifi` / `ไวไฟ`), and ~20% of its pure-Thai edits vanish under a sound key.
The benchmark's references themselves mix scripts for loanwords, so part of the "error" every system
is ranked on may be convention. Same class of finding as `F1` (SARA AM) and `F17` (spelling).
Four numbers per system, one normaliser for all of them:

  CER         plain char Levenshtein after canon() + SARA AM fold + punct strip, Latin entity
              spans in the ref replaced by their annotated Thai form (the official scorer accepts
              either form for annotated entities; Levenshtein would not)
  CER-sb      script-blind: a diff segment that is pure Latin on one side and pure Thai on the
              other costs 0 when the two are the same word by sound (consonant-skeleton match,
              `same_word`); every other segment costs its Levenshtein
  SylER       syllable error rate over tltk G2P syllables WITH tone (`ทำ` = `ธรรม` = tham0), so
              spelling variants of the same sound cost nothing; Latin words are one token each
  SylER-sb    SylER with the same script-blind rule

⚠️ The sound key is tltk's G2P (~10% word error published for Thai G2P generally), applied
identically to both sides of every system -- its errors are noise, not bias, but they are not zero.
⚠️ `same_word` is a heuristic. `--show` prints every pair it accepted so it can be audited.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _thai import SCORING_PUNCT, fold_sara_am, thai_cardinal  # noqa: E402

from tltk.nlp import g2p  # noqa: E402

BENCH = Path("E:/asr-bench/thai-contextasr-bench")
RES = HERE.parent / "results"
SYSTEMS = {
    "typhoon-whisper-large-v3": BENCH / "baselines/whisper_typhoon-ai_typhoon-whisper-large-v3/hyps.jsonl",
    "pathumma-whisper-large-v3": BENCH / "baselines/whisper_nectec_Pathumma-whisper-th-large-v3/hyps.jsonl",
    "biodatlab-whisper-large-v3": BENCH / "baselines/whisper_biodatlab_whisper-th-large-v3-combined/hyps.jsonl",
    "qwen3-asr-1.7b": BENCH / "baselines/qwen3_asr_1p7b/hyps.jsonl",
    "qwen3-asr-0.6b": BENCH / "baselines/qwen3_asr_0p6b/hyps.jsonl",
    "typhoon-isan-whisper": BENCH / "baselines/whisper_typhoon-ai_typhoon-isan-asr-whisper/hyps.jsonl",
    "typhoon-asr-realtime": RES / "arm1_rareword_hyps.jsonl",
    "NECTEC-fastconformer": RES / "nectec_fc_greedy_hyps.jsonl",
    "NECTEC+LOTUSDIS-conv (A2)": RES / "lot2_a75_bench_hyps.jsonl",
    "typhoon-turbo (fw)": RES / "fw_turbo_fp16_hyps.jsonl",
}
_TH_DIG = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_SEP = re.compile(r"(?<=\d),(?=\d{3})")
_RUN = re.compile(r"\d+")
_TOK = re.compile(r"[a-z]+|[\u0e00-\u0e7f]+|[^a-z\u0e00-\u0e7f]+")
_THAI = re.compile(r"^[\u0e00-\u0e7f]+$")
_LATIN = re.compile(r"^[a-z]+$")


def norm(s: str) -> str:
    s = _SEP.sub("", (s or "").translate(_TH_DIG))
    s = _RUN.sub(lambda m: thai_cardinal(int(m.group())) if len(m.group()) <= 9 else m.group(), s)
    return SCORING_PUNCT.sub("", fold_sara_am(s)).lower()


def thai_ref(row: dict) -> str:
    t = row["text"]
    for e in row.get("entities") or []:
        sp, th = e.get("span_text") or "", e.get("form_th") or ""
        if sp and th and re.search(r"[A-Za-z0-9]", sp):
            t = t.replace(sp, th)
    return t


@lru_cache(maxsize=None)
def syllables(thai: str) -> tuple[str, ...]:
    """tltk G2P -> syllables with tone digit, e.g. ไวไฟ -> ('waj0', 'faj0')."""
    out = []
    for word in g2p(thai).split("<s/>"):
        if "<tr/>" not in word:
            continue
        ph = word.split("<tr/>")[1].strip().strip("|")
        out += [p for p in ph.replace("|", "~").split("~") if p]
    return tuple(out) or (thai,)


def tokens(s: str) -> list[str]:
    """Thai runs -> G2P syllables; Latin runs -> one token each, marked."""
    out = []
    for t in _TOK.findall(s):
        if _THAI.match(t):
            out += syllables(t)
        elif _LATIN.match(t):
            out.append("L:" + t)
    return out


# ---- "same word, other script" ------------------------------------------------------------------
_ROM = [("ph", "P"), ("th", "T"), ("kh", "K"), ("ch", "C"), ("sh", "C"), ("ng", "N"), ("ck", "K"),
        ("qu", "K"), ("x", "KS")]
_CLS = {"p": "P", "b": "P", "f": "F", "v": "F", "t": "T", "d": "T", "k": "K", "g": "K", "q": "K",
        "s": "S", "z": "S", "j": "C", "l": "L", "r": "L", "m": "M", "n": "N", "h": ""}


def skel_latin(s: str) -> str:
    s = s.lower()
    s = re.sub(r"c(?=[eiy])", "s", s)
    s = s.replace("c", "k")
    for a, b in _ROM:
        s = s.replace(a, b)
    return "".join(_CLS.get(c, c if c.isupper() else "") for c in s)


def skel_thai(syls: tuple[str, ...]) -> str:
    """tltk romanisation -> the same consonant skeleton. tltk: c=จ/ช, N=ง, j/w glides, @/O vowels."""
    out = []
    for sy in syls:
        sy = re.sub(r"\d", "", sy)
        for a, b in [("ph", "P"), ("th", "T"), ("kh", "K"), ("ch", "C"), ("N", "N")]:
            sy = sy.replace(a, b)
        out.append("".join(_CLS.get(c, c if c in "PTKCN" else "") for c in sy.replace("c", "C")))
    return "".join(out)


def same_word(latin: str, thai: str) -> bool:
    if len(latin) < 3:
        return False
    a, b = skel_latin(latin), skel_thai(syllables(thai))
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.6


ACCEPTED: list[tuple[str, str]] = []


def lev(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def blind_cost(r, h, as_tokens: bool) -> int:
    """Sum of segment costs; a pure-Latin vs pure-Thai segment that is the same word costs 0."""
    cost = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, r, h, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        a, b = r[i1:i2], h[j1:j2]
        if as_tokens:
            la = [t[2:] for t in a if t.startswith("L:")]
            lb = [t[2:] for t in b if t.startswith("L:")]
            if len(la) == len(a) and not lb and b:      # ref Latin, hyp Thai syllables
                pair = ("".join(la), b)
            elif len(lb) == len(b) and not la and a:
                pair = ("".join(lb), a)
            else:
                pair = None
            if pair and len(pair[0]) >= 3:
                sk = skel_thai(tuple(pair[1]))
                sl = skel_latin(pair[0])
                if sk and sl and difflib.SequenceMatcher(None, sl, sk, autojunk=False).ratio() >= 0.6:
                    continue
        else:
            sa, sb = "".join(a), "".join(b)
            if _LATIN.match(sa or "-") and _THAI.match(sb or "-") and same_word(sa, sb):
                ACCEPTED.append((sa, sb))
                continue
            if _THAI.match(sa or "-") and _LATIN.match(sb or "-") and same_word(sb, sa):
                ACCEPTED.append((sb, sa))
                continue
        cost += lev(a, b)
    return cost


def score(path: Path, refs: dict) -> dict:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    hyp = {r["utt_id"]: r["text"] for r in rows if r["utt_id"] in refs and r.get("condition", "none") == "none"}
    if len(hyp) != len(refs):
        return {"n": len(hyp)}
    t = dict(n=len(hyp), chars=0, cer=0, cer_sb=0, syls=0, syl=0, syl_sb=0)
    per = {}
    for uid, ref in refs.items():
        r, h = norm(thai_ref(ref)), norm(hyp[uid])
        rt, ht = tokens(r), tokens(h)
        e, esb, s, ssb = lev(r, h), blind_cost(r, h, False), lev(rt, ht), blind_cost(rt, ht, True)
        esb, ssb = min(esb, e), min(ssb, s)                 # opcode sums can exceed Levenshtein
        t["chars"] += len(r); t["cer"] += e; t["cer_sb"] += esb
        t["syls"] += len(rt); t["syl"] += s; t["syl_sb"] += ssb
        per[uid] = (len(r), e, esb, len(rt), s, ssb)
    return {"n": t["n"], "CER": t["cer"] / t["chars"], "CER-sb": t["cer_sb"] / t["chars"],
            "SylER": t["syl"] / t["syls"], "SylER-sb": t["syl_sb"] / t["syls"], "per_utt": per}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("extra", nargs="*")
    ap.add_argument("--show", action="store_true", help="print accepted Latin~Thai pairs")
    a = ap.parse_args()
    refs = {}
    for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        if r["split"] == "real":
            refs[r["utt_id"]] = r
    systems = {k: p for k, p in SYSTEMS.items() if p.exists()}
    for x in a.extra:
        systems[Path(x).stem.replace("_hyps", "")] = Path(x)
    res = {}
    for k, p in systems.items():
        res[k] = score(p, refs)
        m = res[k]
        if "CER" in m:
            print(f"{k[:32]:32s} CER {m['CER']:.4f}  CER-sb {m['CER-sb']:.4f}  SylER {m['SylER']:.4f}  SylER-sb {m['SylER-sb']:.4f}", flush=True)
        else:
            print(f"{k[:32]:32s} skipped: {m['n']} of {len(refs)} utts", flush=True)
    print(f"\nrank by CER    : {[k for k, _ in sorted(((k, m['CER']) for k, m in res.items() if 'CER' in m), key=lambda x: x[1])]}")
    print(f"rank by SylER-sb: {[k for k, _ in sorted(((k, m['SylER-sb']) for k, m in res.items() if 'CER' in m), key=lambda x: x[1])]}")
    if a.show:
        seen = sorted(set(ACCEPTED))
        print(f"\n{len(seen)} distinct accepted Latin~Thai pairs:")
        for x, y in seen:
            print(f"  {x} ~ {y}")
    (RES / "eval_soundblind_metrics.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
