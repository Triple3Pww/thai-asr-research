#!/usr/bin/env python3
"""Does pitch perturbation damage Thai? The cheapest test that can kill the thesis.

    python adapter/probe_pitch.py --make --n 40      # transform real clips, write a manifest
    python adapter/probe_pitch.py --analyse          # after verify_synth has read them back

WHY THIS RUNS BEFORE THE TONE-PRESERVING TRANSFORM IS BUILT.

The spec's Phase 2 rows 4-5 are "the paper": vary pitch register while holding lexical tone, and
show that varying it *naively* damages Thai where it was merely ineffective in English. Building
the careful transform first assumes the careless one hurts. **Nobody has measured that here.** If
uniform pitch perturbation turns out to be harmless in Thai, there is nothing to preserve, and
the contribution needs reshaping -- which is worth knowing in an afternoon rather than in week six.

It also does not need the thing this box does not have. `pyworld`, `parselmouth` and
`pyrubberband` are all absent from both venvs; only `librosa` is present. Every transform here is
implementable with it, and the one that would need a vocoder -- rescaling the contour's excursion
-- is deliberately deferred until this probe says whether it is worth the dependency.

🔴 THE ARTIFACT CONTROL IS NOT OPTIONAL. `librosa.effects.pitch_shift` is a phase vocoder: it
degrades audio whether or not it moves the pitch. `roundtrip` runs the identical code path at
**zero semitones**. Any CER it costs is the ALGORITHM, and every shifted row must be read against
it, not against `none`. Without this row a vocoder artifact reads as a tone finding.

WHAT IS MEASURED, AND THE ONE THING THAT MAKES IT THAI.

    CER        the usual readback error
    TONE-CER   edit distance over the TONE MARKS ALONE (`_thai.tone_sequence`)

Thai writes tone with combining marks, so a tone error is a character error and separable from
every other kind. If pitch manipulation hurts *tone specifically*, TONE-CER rises faster than CER.
If both rise together, the damage is general -- artifacts, not tone. That contrast is the whole
instrument. ⚠️ It is a floor: an unmarked syllable still carries a tone this cannot see, so a rise
in TONE-CER is strong evidence of damage and no rise is only weak evidence of safety.

🔴 REAL AUDIO, NOT SYNTHETIC. The clips are the 150 Common Voice Thai references in `data/refs/`,
which carry ground-truth sentences. Running this on VoxCPM output would confound "pitch hurt it"
with "the TTS was already wobbly there". Real speech has no such excuse.

⚠️ THIS IS A RECOGNITION TEST, NOT A TRAINING TEST. It asks whether a pitch-shifted Thai clip is
still readable, which is the cheap proxy. Whether shifted clips TRAIN a better recogniser is
Phase 2 proper and costs weeks. A transform that destroys readability cannot help training; one
that preserves it still has to prove it helps.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "results"
REFS = REPO / "data" / "refs"
SR = 16000

sys.path.insert(0, str(HERE))
from _thai import for_scoring, tone_sequence  # noqa: E402

#: (name, kind, amount). `shift` = constant semitones, phase vocoder, duration preserved.
#: `speed` = resample, which moves pitch AND tempo together -- this is Ko et al. 2015 speed
#: perturbation, the augmentation that is actually in everyone's ASR recipe, and therefore the
#: row that says whether the field is already damaging Thai without noticing.
CONDITIONS = [
    ("none",        None,    0.0),
    ("roundtrip",   "shift", 0.0),    # 🔴 the artifact control -- same vocoder, no pitch change
    ("shift_m4",    "shift", -4.0),
    ("shift_m2",    "shift", -2.0),
    ("shift_p2",    "shift", 2.0),
    ("shift_p4",    "shift", 4.0),
    ("speed_090",   "speed", 0.90),
    ("speed_110",   "speed", 1.10),
]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r, h = for_scoring(ref), for_scoring(hyp)
    return 1.0 if not r else levenshtein(r, h) / len(r)


def tone_cer(ref: str, hyp: str) -> float | None:
    """Edit distance over tone marks alone. None when the reference carries none to damage."""
    r = tone_sequence(for_scoring(ref))
    h = tone_sequence(for_scoring(hyp))
    return None if not r else levenshtein(r, h) / len(r)


def transform(y, sr: int, kind: str | None, amount: float):
    import librosa
    if kind is None:
        return y
    if kind == "shift":
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=amount)
    if kind == "speed":
        # Resample-in-place: the clip plays at `amount` speed and its pitch moves by the same
        # factor. Duration changes, which is the point -- this is the real recipe, not a
        # duration-preserving idealisation of it.
        # Resample to sr/amount and then PLAY at sr: N samples become N/amount, so the clip lasts
        # 1/amount as long and its pitch moves by amount. 0.90 = slower and lower, 1.10 = faster
        # and higher, which is the convention Ko et al. use.
        return librosa.resample(y, orig_sr=sr, target_sr=int(sr / amount))
    raise ValueError(kind)


def make(n: int, out_dir: Path) -> int:
    import numpy as np
    import soundfile as sf
    import librosa

    items = json.loads((REFS / "refs.json").read_text(encoding="utf-8"))["items"]
    # Longest first: a longer sentence carries more tone marks, so TONE-CER has more to measure.
    items = sorted(items, key=lambda r: -len(for_scoring(r["text"])))[:n]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in items:
        y, sr = sf.read(str(REFS / r["wav"]), dtype="float32", always_2d=True)
        y = y[:, 0]
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        for name, kind, amt in CONDITIONS:
            z = transform(y, SR, kind, amt)
            gen_id = f"{r['id']}__{name}"
            sf.write(str(out_dir / f"{gen_id}.wav"), z, SR)
            rows.append({"clip": f"{gen_id}.wav", "gen_id": gen_id, "text_id": r["id"],
                         "condition": name, "kind": kind, "amount": amt,
                         "text": r["text"], "label_text": r["text"],
                         "dur_s": round(len(z) / SR, 3),
                         "n_chars": len(for_scoring(r["text"])),
                         "n_tone": len(tone_sequence(for_scoring(r["text"])))})
        print(f"  {r['id']}  {len(for_scoring(r['text'])):3d} chars, "
              f"{len(tone_sequence(for_scoring(r['text']))):2d} tone marks", flush=True)

    mp = out_dir / "manifest.jsonl"
    mp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                  encoding="utf-8")
    tot_tone = sum(r["n_tone"] for r in rows if r["condition"] == "none")
    print(f"\n{len(rows)} clips ({len(items)} sentences x {len(CONDITIONS)} conditions) -> {mp}")
    print(f"{tot_tone} tone marks per condition to get wrong")
    print("\nnow read them back:\n"
          f"  E:/asr-spike/.venv/Scripts/python.exe adapter/verify_synth.py "
          f"--manifest {mp} --clips {out_dir} --asr biodatlab --tag probe_pitch --cer-max 1.0")
    return 0


def analyse(out_dir: Path) -> int:
    rows = {}
    for f in ("probe_pitch_verify_keep.jsonl", "probe_pitch_verify_reject.jsonl"):
        p = OUT / f
        if p.exists():
            for l in p.open(encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    if r.get("hyps"):
                        rows[r["gen_id"]] = r
    if not rows:
        raise SystemExit("no readback found -- run verify_synth.py with --tag probe_pitch first")

    by = {}
    for r in rows.values():
        hyp = next(iter(r["hyps"].values()))
        ref = r["label_text"]
        by.setdefault(r["condition"], []).append((cer(ref, hyp), tone_cer(ref, hyp)))

    print(f"{'condition':12s} {'n':>3s} {'CER':>7s} {'vs rt':>7s} | {'TONE-CER':>9s} {'vs rt':>7s} "
          f"| {'tone/CER':>8s}")
    print("-" * 66)
    base = {}
    for name, *_ in CONDITIONS:
        xs = by.get(name, [])
        if not xs:
            continue
        c = statistics.mean(a for a, _ in xs)
        ts = [t for _, t in xs if t is not None]
        t = statistics.mean(ts) if ts else float("nan")
        base[name] = (c, t)
    rt_c, rt_t = base.get("roundtrip", base.get("none", (0, 0)))
    for name, *_ in CONDITIONS:
        if name not in base:
            continue
        c, t = base[name]
        print(f"{name:12s} {len(by[name]):3d} {c:7.4f} {c - rt_c:+7.4f} | {t:9.4f} {t - rt_t:+7.4f} "
              f"| {t / c if c else float('nan'):8.2f}")

    n_c, n_t = base.get("none", (float("nan"),) * 2)
    print(f"\nARTIFACT FLOOR: the vocoder at zero semitones costs {rt_c - n_c:+.4f} CER and "
          f"{rt_t - n_t:+.4f} TONE-CER on its own.")
    print("🔴 Every shifted row above is measured against `roundtrip`, never against `none`.\n")

    shifts = [n for n, k, _ in CONDITIONS if k == "shift" and n != "roundtrip" and n in base]
    if shifts:
        worst = max(shifts, key=lambda n: base[n][1] - rt_t)
        dt, dc = base[worst][1] - rt_t, base[worst][0] - rt_c
        print(f"worst uniform shift = {worst}: TONE-CER {dt:+.4f}, CER {dc:+.4f} over the artifact "
              f"floor")
        if dt <= 0.01:
            print("  -> 🔴 UNIFORM PITCH SHIFT DOES NOT DAMAGE THAI TONE at this size. The "
                  "'naive pitch hurts tonal languages' premise does NOT hold here, and Phase 2 "
                  "rows 4-5 need reshaping before any of it is built.")
        elif dt > dc:
            print("  -> tone is damaged FASTER than the text as a whole: the damage is "
                  "tone-specific, which is the premise rows 4-5 rest on.")
        else:
            print("  -> tone and text degrade together, so this looks like general degradation "
                  "rather than tone damage. Not yet evidence for rows 4-5.")
    print("\n⚠️ A recognition test, not a training test. And TONE-CER is a floor -- an unmarked")
    print("   syllable still carries a tone it cannot see.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=str(OUT / "probe_pitch"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out_dir = Path(args.out)
    if args.analyse:
        return analyse(out_dir)
    if args.make:
        return make(args.n, out_dir)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
