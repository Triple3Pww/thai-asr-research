#!/usr/bin/env python3
"""Which control mechanism does VoxCPM actually obey? Two days, before eight weeks.

    E:/voxcpm-spike/.venv/Scripts/python.exe adapter/probe_control.py --texts 8
    E:/asr-spike/.venv/Scripts/python.exe   adapter/probe_control.py --analyse

WHY. `R2` fired: Phase 2 was designed to vary one attribute at a time and measure which variation
helps the ASR, and VoxCPM exposes **no numeric per-attribute knob**. Confirmed by reading the
package, not the docs -- `VoxCPM._generate` takes `text, prompt_wav_path, prompt_text,
reference_wav_path, cfg_value, inference_timesteps, min_len, max_len, normalize, denoise,
retry_badcase...` and nothing else. There is no speed, no style, no emotion.

What does exist is `voxcpm/cli.py:66`:

    def build_final_text(text, control):
        return f"({control}){text}" if control else text

-- the control is a parenthesised prefix **inside the text string**. And `cli.py:123`:

    if args.control and prompt_text:
        parser.error("--control cannot be used together with --prompt-text or --prompt-file.")

🔴 **THE CLI FORBIDS COMBINING CONTROL WITH VOICE CLONING.** Our whole corpus clones the 150
Common Voice reference voices, so if that restriction is real at the model level, mechanism B
(reference selection) and mechanism C (control text) are mutually exclusive -- not complementary,
which is what the spec assumed. Whether it is a model restriction or only a CLI guideline is an
empirical question, and it is the single most valuable thing this probe answers.

THE THREE MECHANISMS, and what each would mean for the thesis:

    A  post-process the audio (DSP)   precise and free, but plain data augmentation -- closed
                                      ground, and the nearest thing to the "easy config" the
                                      advisor already rejected. Measured here as the CEILING:
                                      what perfect control looks like.
    B  reference-voice selection      control the attribute by choosing WHAT to condition on.
                                      The reference bank spans 2.15x in speaking rate at
                                      p10-p90 (measured 2026-09-17), so there is a range to
                                      select along.
    C  natural-language control       `(พูดเร็วๆ)text`. Native to VoxCPM2, and the only one that
                                      generalises past attributes we can measure.

THE ATTRIBUTE IS SPEAKING RATE, because it is the one that can be measured without a model:
label characters over clip duration. An attribute probe that needed an ASR to score it would be
measuring two things at once.

🔴 THE NOISE FLOOR IS A CONDITION, NOT AN AFTERTHOUGHT. `base_a` and `base_b` are the same text
and the same reference at two different seeds. VoxCPM samples at every step, so rate varies run to
run for free. **A mechanism that moves rate less than the seed does is not a control**, and
without that row every other number here would look like an effect.

THE ANSWER, MEASURED 2026-09-17 (8 texts x 9 conditions, VoxCPM2, readback by biodatlab):

    noise floor -- same text, same voice, new seed            0.96 ch/s

    B  reference selection      11.83 -> 16.82 ch/s   +4.99   5.2x noise   USABLE
    C  control + cloning        unusable: 0 of 8 clips clean, readback CER median 0.157
    C  control, no cloning      15.77 -> 15.82 ch/s   +0.05   0.0x noise   NO EFFECT

**C fails in two different ways, and the difference matters.** With a cloned voice the model
SPEAKS THE INSTRUCTION: the readback of `(พูดช้าๆ)สวัสดีค่ะ...` begins `พูดช้า ๆ สวัสดีค่ะ`. So
`cli.py`'s refusal to combine `--control` with `--prompt-text` is a real property of the model,
not a CLI guideline -- the prefix is only stripped in voice-design mode. Without cloning the
prefix IS consumed (readback CER 0.027, clean) and changes nothing about the rate.

⚠️ **B's usable range is far narrower than the reference bank's.** The references span 3.95-13.10
ch/s (3.3x); the clips generated from them span 11.83-16.82 (1.42x). VoxCPM2 borrows the
reference's rate but compresses it heavily toward its own default, so Phase 2 gets ~1.4x of
controllable variation, not 3.3x. Plan the ablation around the achieved range, never the
reference range.

✅ And VoxCPM2 is stable on this material: **1 of 72 clips catastrophic**, readback CER median
0.051 on the baseline conditions.

WHAT THIS DOES NOT ANSWER. Whether the variation helps the ASR -- that is Phase 2 itself. This
only answers whether the variation can be produced and set on purpose, which is the question that
decides which Phase 2 is worth eight weeks.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "results"

# 🔴 THE MODEL AND ITS SETTINGS COME FROM `colab/gen_colab.py`, NOT FROM HERE. A probe that
# generates with different settings than the corpus is measuring a different system.
#
# This is not hypothetical: the first two runs of this probe hardcoded `openbmb/VoxCPM-0.5B`,
# the small original, while the corpus is generated with `openbmb/VoxCPM2` -- the 2B, 30-language
# checkpoint that is the only one with Thai (`H29`). The 0.5B produced a **26% catastrophic rate**
# (clips of 0.46 s and of 41 s for the same 75-character sentence) and, of course, ignored the
# `(control)` prefix entirely, since `cli.py:376` calls it "Control instruction for VoxCPM2".
# Two full 16-minute runs were spent measuring a model this repo does not use.
sys.path.insert(0, str(REPO / "colab"))
from gen_colab import CFG_VALUE, TIMESTEPS, VOXCPM_MODEL  # noqa: E402
REFS = REPO / "data" / "refs"
SR = 16000

sys.path.insert(0, str(HERE))
from _thai import for_scoring  # noqa: E402

#: Control strings, in Thai, because the text they prefix is Thai and VoxCPM prepends them into
#: the very string it is asked to speak. An English instruction in front of a Thai sentence is a
#: second variable.
CTL_SLOW = "พูดช้าๆ"
CTL_FAST = "พูดเร็วๆ"

#: (name, which reference, control string, seed slot). `None` reference = no cloning prompt at
#: all, which is the mode `cli.py` says control is for.
CONDITIONS = [
    ("base_a",       "mid",  None,      0),
    ("base_b",       "mid",  None,      1),   # the noise floor: same everything, different seed
    ("ref_slow",     "slow", None,      0),
    ("ref_fast",     "fast", None,      0),
    ("ctl_slow",     "mid",  CTL_SLOW,  0),   # control WITH cloning -- what the CLI forbids
    ("ctl_fast",     "mid",  CTL_FAST,  0),
    ("noclone_none", None,   None,      0),   # control WITHOUT cloning -- what the CLI intends
    ("noclone_slow", None,   CTL_SLOW,  0),
    ("noclone_fast", None,   CTL_FAST,  0),
]


def ref_rates(min_chars: int = 30) -> list[tuple[float, dict]]:
    """Reference voices by speaking rate, shortest ones excluded.

    ⚠️ A `dur_s` includes leading and trailing silence, so a 11-character clip reads 2.2 ch/s and
    says more about its silence than its speaker. Below `min_chars` the rate is mostly padding.
    """
    d = json.loads((REFS / "refs.json").read_text(encoding="utf-8"))
    out = []
    for r in d["items"]:
        n = len(for_scoring(r["text"]))
        if n >= min_chars and r.get("dur_s"):
            out.append((n / r["dur_s"], r))
    out.sort(key=lambda x: x[0])
    return out


def pick_refs() -> dict[str, dict]:
    rr = ref_rates()
    if len(rr) < 3:
        raise SystemExit(f"only {len(rr)} usable reference voices -- cannot pick slow/mid/fast")
    return {"slow": rr[0][1], "mid": rr[len(rr) // 2][1], "fast": rr[-1][1]}


# ------------------------------------------------------------------ generate (VoxCPM venv)

def _patch_torchaudio_load() -> str:
    """Route `torchaudio.load` through soundfile. Without this, generation cannot start.

    🔴 FOUND 2026-09-17, THE FIRST TIME A CLIP WAS CLONED FROM A REFERENCE ON THIS BOX SINCE
    torchaudio 2.11. `VoxCPMModel.build_prompt_cache` calls `torchaudio.load(prompt_wav_path)`;
    torchaudio >= 2.9 routes `load` through torchcodec, and torchcodec needs FFmpeg shared
    libraries this machine does not have:

        OSError: Could not load this library: ...torchcodec\\libtorchcodec_core5.dll
        FileNotFoundError: Could not find module ...libtorchcodec_core4.dll

    The model loads fine and dies on the first generate() with a prompt -- 40 s of warm-up before
    anything fails, which is why this is caught here and not at import.

    soundfile 0.14 is already in the venv and reads these 48 kHz wavs directly. `torchaudio.load`
    returns (channels, samples); soundfile returns (samples, channels), hence the transpose.

    ⚠️ `colab/gen_colab.py` calls the same path and will hit the same wall on the full corpus run.
    """
    import torch
    import torchaudio
    import soundfile as sf

    def _load(path, *a, **k):
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(y.T.copy()), sr

    torchaudio.load = _load
    return torchaudio.__version__


def generate(n_texts: int, seed_base: int, out_dir: Path) -> int:
    import numpy as np
    import librosa
    import soundfile as sf
    import torch

    print(f"[shim] torchaudio.load -> soundfile (torchaudio {_patch_torchaudio_load()})")
    print(f"[voxcpm] {VOXCPM_MODEL}  cfg={CFG_VALUE}  timesteps={TIMESTEPS}"
          "   (imported from colab/gen_colab.py -- never hardcoded here)")
    from voxcpm import VoxCPM

    texts = json.loads((HERE / "texts" / "phase1_known.json").read_text(encoding="utf-8"))
    texts = texts[:n_texts]
    refs = pick_refs()
    print("reference voices chosen by measured speaking rate:")
    for k, r in refs.items():
        rate = len(for_scoring(r["text"])) / r["dur_s"]
        print(f"  {k:5s} {rate:5.2f} ch/s  {r['id']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # 🔴 EXACTLY colab/gen_colab.py's load and settings, or this probe measures a different
    # system than the corpus will be generated with. The first run used library defaults
    # (denoiser on, timesteps 10) and produced a 25% catastrophic rate against 0 of 12 on the
    # real pipeline -- the failures drowned the attribute being measured.
    model = VoxCPM.from_pretrained(VOXCPM_MODEL, optimize=True, load_denoiser=False)
    rows, t_start = [], time.perf_counter()
    total = len(texts) * len(CONDITIONS)
    i = 0
    for t in texts:
        for name, which, ctl, slot in CONDITIONS:
            i += 1
            gen_id = f"{t['id']}__{name}"
            # zlib.crc32, not hash(): Python randomises string hashing per process, so the
            # first run of this probe could not be reproduced even from its own manifest.
            seed = seed_base + slot * 1000 + (zlib.crc32(t["id"].encode()) & 0xFFFF)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # The control is a prefix INSIDE the text, exactly as cli.build_final_text does it.
            spoken = f"({ctl}){t['text']}" if ctl else t["text"]
            kw = {}
            if which is not None:
                r = refs[which]
                kw = {"prompt_wav_path": str(REFS / r["wav"]), "prompt_text": r["text"]}
            t0 = time.perf_counter()
            y = np.asarray(model.generate(text=spoken, cfg_value=CFG_VALUE,
                                          inference_timesteps=TIMESTEPS, retry_badcase=True, **kw),
                           dtype=np.float32).reshape(-1)
            wall = time.perf_counter() - t0
            y = librosa.resample(y, orig_sr=model.tts_model.sample_rate, target_sr=SR)
            sf.write(str(out_dir / f"{gen_id}.wav"), y, SR)
            dur = len(y) / SR
            n_chars = len(for_scoring(t["label_text"]))
            rows.append({"clip": f"{gen_id}.wav", "gen_id": gen_id, "text_id": t["id"],
                         "condition": name, "ref": which, "control": ctl, "seed": seed,
                         # `text` is what the ASR must match: the LABEL, never the control prefix.
                         "text": t["text"], "label_text": t["label_text"],
                         "spoken_text": spoken,
                         "dur_s": round(dur, 3), "n_chars": n_chars,
                         "chars_per_s": round(n_chars / dur, 4) if dur else None,
                         "rtf": round(wall / dur, 3) if dur else None})
            print(f"  [{i}/{total}] {gen_id:34s} {dur:6.2f}s  "
                  f"{rows[-1]['chars_per_s']:5.2f} ch/s", flush=True)

    mpath = out_dir / "manifest.jsonl"
    mpath.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                     encoding="utf-8")
    print(f"\n{len(rows)} clips in {time.perf_counter() - t_start:.0f}s -> {mpath}")
    print("next: run the panel over it, because a rate that moved by dropping words is not "
          "a rate change --\n"
          f"  E:/asr-spike/.venv/Scripts/python.exe adapter/verify_synth.py "
          f"--manifest {mpath} --clips {out_dir} --asr panel --tag probe_control")
    return 0


# ------------------------------------------------------------------ analyse (either venv)

#: The same three constants `colab/gen_colab.py` uses to find the end of speech. Duplicated rather
#: than imported because that module imports torch at call time and this half runs in either venv.
WIN_S = 0.01
LEAD_FLOOR_RMS = 0.003


def voiced_span_s(y, sr: int) -> float:
    """First to last audible 10 ms window. THE denominator, and not the clip length.

    🔴 THE FIRST RUN OF THIS PROBE MEASURED CLIP LENGTH AND WAS MEANINGLESS. `gen_colab.py:113`
    already says why, in its own words: *"VoxCPM samples its own trailing pause (measured 0-2380
    ms per reply), so without this the clip length carries information about the draw rather than
    about the speech."* Raw duration put `base_a` at 21.3 ch/s and `base_b` -- same text, same
    reference, different seed -- at 3.6 ch/s, a 6x "effect" that was mostly silence.
    """
    import numpy as np
    y = np.asarray(y, dtype="float32").reshape(-1)
    win = int(sr * WIN_S)
    n = len(y) // win
    if win <= 0 or n == 0:
        return 0.0
    env = np.sqrt(np.mean(y[: n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
    loud = np.where(env > LEAD_FLOOR_RMS)[0]
    return float((loud[-1] - loud[0] + 1) * WIN_S) if len(loud) else 0.0


def remeasure(rows: list[dict], out_dir: Path) -> list[dict]:
    """Recompute each clip's rate over its VOICED span, reading the wav back off disk."""
    import soundfile as sf
    for r in rows:
        p = out_dir / r["clip"]
        if not p.exists():
            continue
        y, sr = sf.read(str(p), dtype="float32", always_2d=True)
        span = voiced_span_s(y[:, 0], sr)
        r["voiced_s"] = round(span, 3)
        r["silence_s"] = round(r["dur_s"] - span, 3)
        r["chars_per_voiced_s"] = round(r["n_chars"] / span, 4) if span > 0.2 else None
    return rows


def analyse(out_dir: Path) -> int:
    mpath = out_dir / "manifest.jsonl"
    if not mpath.exists():
        raise SystemExit(f"no manifest at {mpath} -- run the generate half first")
    rows = remeasure([json.loads(l) for l in mpath.open(encoding="utf-8") if l.strip()], out_dir)
    by = {}
    for r in rows:
        by.setdefault(r["condition"], []).append(r)

    sil = [r["silence_s"] for r in rows if "silence_s" in r]
    if sil:
        sil.sort()
        print(f"trailing+leading silence across {len(sil)} clips: "
              f"min {sil[0]:.2f}s  median {statistics.median(sil):.2f}s  max {sil[-1]:.2f}s")
        print("that is why the denominator is the voiced span, not the clip length "
              "(gen_colab.py:113)\n")

    def rate(c):
        return [x["chars_per_voiced_s"] for x in by.get(c, []) if x.get("chars_per_voiced_s")]

    base_a, base_b = rate("base_a"), rate("base_b")
    # Paired, because the same texts run through both: the seed's effect per text, not the
    # difference of two means, which would hide it.
    pairs = {x["text_id"]: x["chars_per_voiced_s"] for x in by.get("base_b", [])
             if x.get("chars_per_voiced_s")}
    noise = [abs(x["chars_per_voiced_s"] - pairs[x["text_id"]])
             for x in by.get("base_a", [])
             if x.get("chars_per_voiced_s") and x["text_id"] in pairs]
    floor = statistics.mean(noise) if noise else float("nan")

    print(f"{'condition':15s} {'n':>3s} {'ch/s mean':>10s} {'sd':>6s} {'vs base_a':>10s} "
          f"{'x noise':>8s}")
    print("-" * 60)
    m_base = statistics.mean(base_a) if base_a else float("nan")
    for name, *_ in CONDITIONS:
        xs = rate(name)
        if not xs:
            continue
        m = statistics.mean(xs)
        sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        d = m - m_base
        print(f"{name:15s} {len(xs):3d} {m:10.3f} {sd:6.3f} {d:+10.3f} "
              f"{abs(d) / floor if floor else float('nan'):8.2f}")

    print(f"\nNOISE FLOOR: same text, same reference, different seed -> mean |delta| "
          f"{floor:.3f} ch/s over {len(noise)} texts.")
    print("🔴 A mechanism whose effect is under ~2x this is not a control, it is sampling.\n")

    for label, lo, hi in (("B  reference selection", "ref_slow", "ref_fast"),
                          ("C  control + cloning  ", "ctl_slow", "ctl_fast"),
                          ("C  control, no clone  ", "noclone_slow", "noclone_fast")):
        a, b = rate(lo), rate(hi)
        if not a or not b:
            print(f"{label}: not generated")
            continue
        span = statistics.mean(b) - statistics.mean(a)
        verdict = ("USABLE" if abs(span) >= 2 * floor else
                   "marginal" if abs(span) >= floor else "NO EFFECT")
        print(f"{label}: slow {statistics.mean(a):5.2f} -> fast {statistics.mean(b):5.2f} ch/s, "
              f"span {span:+.2f} ({abs(span) / floor:.1f}x noise)  -> {verdict}")

    print("\n⚠️ A rate that moved because the model DROPPED WORDS is not a rate change. Nothing")
    print("   above can tell those apart -- run verify_synth.py over the manifest before")
    print("   believing any row, and discard conditions whose readback CER blew up.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", type=int, default=8,
                    help="how many phase1_known texts, each spoken under every condition")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", default=str(OUT / "probe_control"))
    ap.add_argument("--analyse", action="store_true",
                    help="read the manifest and report, without generating (ASR venv is fine)")
    ap.add_argument("--refs-only", action="store_true",
                    help="print the reference speaking-rate range and exit -- no model needed")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out_dir = Path(args.out)

    if args.refs_only:
        rr = ref_rates()
        xs = [r for r, _ in rr]
        print(f"{len(rr)} reference voices of >=30 chars")
        print(f"  min {xs[0]:.2f}  median {statistics.median(xs):.2f}  max {xs[-1]:.2f}  "
              f"({xs[-1] / xs[0]:.2f}x)")
        for k, r in pick_refs().items():
            print(f"  {k:5s} {len(for_scoring(r['text'])) / r['dur_s']:5.2f} ch/s  {r['id']}")
        return 0
    if args.analyse:
        return analyse(out_dir)
    return generate(args.texts, args.seed, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
