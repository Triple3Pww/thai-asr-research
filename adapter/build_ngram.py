#!/usr/bin/env python3
"""Convert Typhoon's 4-gram ARPA to NeMo's binary LM once, so the server never parses 2.7 GB.

    E:/asr-spike/.venv/Scripts/python.exe adapter/build_ngram.py

WHY. `typhoon-ai/typhoon-asr-streaming-115m-ngram` ships `ngram_4gram_bpe2048.arpa` -- 2.7 GB of
text: 2,045 unigrams / 1.96 M bigrams / 30.9 M trigrams / 71.9 M 4-grams. `NGramGPULanguageModel`
accepts the ARPA directly, but re-parsing it on every server boot costs minutes of startup and a
large transient allocation on a box whose commit limit is already this repo's documented failure
mode (`CLAUDE.md`: commit 39.5 GB, VoxCPM reserves ~22.7 GB). `.nemo` is a state dict -- build it
once here, load the binary thereafter.

TOKENIZER LOCK -- re-checked here, not trusted from a note. The LM is over token IDs, so it means
nothing with a different tokenizer; its card says so outright ("Tokenizer-locked: it works only
with the matching model's tokenizer"). This script hashes the tokenizer inside the checkpoint
:8007 actually serves and refuses to build if it is not
md5 88e04338795f42f0492c953e461ac785 -- the hash shared by `typhoon-asr-streaming-115m` (the LM's
paired model), `typhoon-asr-realtime` (ours) and `typhoon-asr-realtime-nemo-ctc`.

VOCAB SIZE -- the one argument that fails silently. `rnnt_decoding.py` calls
`NGramGPULanguageModel.from_file(lm_path=..., vocab_size=self.blank_id)`, and for this model
`blank_id` is 2048 (ids 0..2047 are real tokens). Pass a different number and nothing errors:
every token id shifts and the LM scores the wrong words. The ARPA's header `ngram 1=2045` is
consistent with that -- 2048 real tokens, three of the LM's own slots spent on
`<unk>` / `<s>` / `</s>`.

MEMORY. Reported, not assumed: the build streams the ARPA but holds a suffix tree sized
`2 + vocab_size + sum(counts)` ~= 105 M states. This script prints peak RSS and the commit
headroom before and after, because "it died with no message" on this box has meant the page file
three times and VRAM once.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import tarfile
import time
from pathlib import Path

ARPA = Path("E:/asr-bench/typhoon-ngram/ngram_4gram_bpe2048.arpa")
OUT = Path("E:/asr-bench/typhoon-ngram/ngram_4gram_bpe2048.nemo")
TOKENIZER_MD5 = "88e04338795f42f0492c953e461ac785"


def served_tokenizer_md5() -> tuple[str, str]:
    """md5 of the tokenizer inside the checkpoint :8007 serves, read from the .nemo tar."""
    pat = os.path.expanduser(
        "~/.cache/huggingface/hub/models--typhoon-ai--typhoon-asr-realtime/"
        "snapshots/*/typhoon-asr-realtime.nemo")
    hits = glob.glob(pat)
    if not hits:
        raise SystemExit(f"served checkpoint not found under {pat}")
    with tarfile.open(hits[0]) as tf:
        for m in tf.getmembers():
            if m.name.endswith("tokenizer.model"):
                return hashlib.md5(tf.extractfile(m).read()).hexdigest(), hits[0]
    raise SystemExit("no tokenizer.model inside the served checkpoint")


def mem() -> str:
    try:
        import ctypes
        import ctypes.wintypes as wt

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        phys, commit = st.ullAvailPhys / 2**30, st.ullAvailPageFile / 2**30
        return f"RAM free {phys:.1f} GB | commit free {commit:.1f} GB"
    except Exception as e:                                  # never let the probe kill the build
        return f"(memory probe failed: {e})"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arpa", type=Path, default=ARPA)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--vocab-size", type=int, default=2048,
                    help="tokens WITHOUT blank; must equal the model's blank_id (2048 here)")
    ap.add_argument("--skip-tokenizer-check", action="store_true")
    args = ap.parse_args()

    if not args.arpa.exists():
        print(f"missing {args.arpa}", file=sys.stderr)
        return 1
    print(f"arpa   : {args.arpa}  ({args.arpa.stat().st_size/2**30:.2f} GiB)")

    if not args.skip_tokenizer_check:
        md5, src = served_tokenizer_md5()
        ok = md5 == TOKENIZER_MD5
        print(f"tokenz : {md5}  {'MATCH' if ok else 'MISMATCH'}  <- {src}")
        if not ok:
            print("🔴 The LM is tokenizer-locked and this checkpoint's tokenizer is not the one "
                  "it was built on. Refusing to build.", file=sys.stderr)
            return 2

    print(f"before : {mem()}")
    from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel

    t0 = time.perf_counter()
    lm = NGramGPULanguageModel.from_file(lm_path=args.arpa, vocab_size=args.vocab_size)
    print(f"built  : {time.perf_counter()-t0:.0f}s   {mem()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lm.save_to(str(args.out))
    print(f"saved  : {args.out}  ({args.out.stat().st_size/2**30:.2f} GiB)")
    print(f"after  : {mem()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
