#!/usr/bin/env python3
"""Can VoxCPM2 DESIGN a voice from a text description? The question the rate probe mis-asked.

    E:/voxcpm-spike/.venv/Scripts/python.exe adapter/probe_voice.py --texts 4
    E:/asr-spike/.venv/Scripts/python.exe   adapter/probe_voice.py --analyse

WHY THIS EXISTS, AND WHY `probe_control.py`'S VERDICT WAS TOO BROAD.

`probe_control.py` concluded "control text does not work". It measured one attribute -- speaking
rate -- by prefixing `(พูดช้าๆ)` / "speak slowly". That is a DELIVERY instruction, and it is not
what the knob is documented for. `voxcpm/cli.py` has exactly one example of it:

    voxcpm design --text "Hello world" --control "warm female voice" --output out.wav

-- an instruction about **who is speaking**: timbre, gender, age. Asking a voice-design control to
change tempo and reporting "control does not work" is testing the wrong hypothesis. The honest
statement from that probe is narrower: *it does not control speaking rate.*

🔴 AND THIS IS THE MORE VALUABLE QUESTION FOR THE THESIS. Reference cloning is capped at the 150
Common Voice reference clips this repo owns -- 150 speakers, and they are real people whose licence
and provenance we inherit. A voice designed from a sentence of English is **an unbounded supply of
speakers**. For ASR training data, speaker diversity is worth more than tempo diversity: it is the
axis a recogniser most needs to generalise over, and it is the one `phase1_known`/`phase2_rare`
currently get only 150 of.

WHAT IS MEASURED, AND WHY NOT BY EAR OR BY EMBEDDING. Two cheap acoustic descriptors, both from
librosa, neither needing a model that would have to be justified:

    f0_median        voiced pitch. Separates adult male / adult female / child at a glance.
    centroid_median  spectral centre of mass -- a coarse brightness/timbre proxy.

⚠️ THESE ARE PROXIES, NOT SPEAKER IDENTITY. Two genuinely different speakers can share a pitch.
So this probe can prove that voices DID separate; it cannot prove that two clips with similar
pitch are the same voice. A negative on pitch is weak evidence; a positive is strong. Read it in
that direction only, and confirm by ear before anything is built on it.

THE ANSWER, MEASURED 2026-09-18 (4 texts x 8 conditions, VoxCPM2, readback median CER 0.000):

    seed-only floor, design mode, no instruction        37.8 Hz of pitch, for free

    deep male voice  MINUS  warm female voice           -4.4 Hz     paired, same seed
    "young child"                                      +50.4 Hz     but +153 / +29 / +17 / +2
    "elderly man"                                      -21.5 Hz
    thai instruction vs english                          2.3 Hz     neither works, so neither wins

🔴 **THE GENDER INSTRUCTION DOES NOTHING.** A real male/female pitch gap is 80-100 Hz. Asking for
a deep male voice instead of a warm female one moves the pitch by **4 Hz**, against 38 Hz that the
random seed moves it for free -- an order of magnitude the wrong way. Only "young child" separates
at all, and it is unreliable: +153 Hz on one sentence and +1.8 Hz on the next. **The seed picks
the voice; the instruction is decoration.**

✅ **BUT THE FREE VARIETY IS THE FINDING.** Design mode invents a speaker per draw, and across
these four sentences the pitch ran 105-251 Hz with the words still correct (readback CER median
0.000, 29 of 32 kept). So VoxCPM2 IS an unbounded speaker supply -- it is simply not a STEERABLE
one. For a corpus that currently has 150 reference speakers, "unlimited but unsteerable" is worth
more than it sounds, and it costs nothing but a different seed.

⚠️ So Phase 2 cannot hold speaker identity fixed while varying something else, and cannot request
a demographic. What it can do is: set rate by reference selection (1.42x, `probe_control.py`), and
sample speakers freely by seed.

🔴 THE NOISE FLOOR IS A CONDITION AGAIN. `plain_a` and `plain_b` are the same text with no control
at two seeds. VoxCPM2 in design mode invents a voice per draw, so it produces speaker variety for
free -- and if that free variety is as large as what the control strings produce, the control is
decoration. This is the row that made the rate probe honest and it is the row that matters most
here, because "the voices are all different" is exactly what an uncontrolled sampler also gives.
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
SR = 16000

sys.path.insert(0, str(REPO / "colab"))
from gen_colab import CFG_VALUE, TIMESTEPS, VOXCPM_MODEL  # noqa: E402
sys.path.insert(0, str(HERE))
from _thai import for_scoring  # noqa: E402
from probe_control import _patch_torchaudio_load, voiced_span_s  # noqa: E402

#: (name, control string). `None` = design mode with no instruction, the noise floor.
#:
#: The strings are ENGLISH because `cli.py`'s only documented example is English, and VoxCPM2 is a
#: 30-language model whose instruction-following was presumably trained that way. `th_female` is
#: the one Thai string, present to answer a question worth one clip: does the instruction have to
#: be in the language of the text? It is a separate variable and is reported separately.
CONDITIONS = [
    ("plain_a",    None,                             0),
    ("plain_b",    None,                             1),   # noise floor: no control, new seed
    ("warm_female", "warm female voice",             0),
    ("deep_male",   "deep male voice",               0),
    ("elderly_man", "elderly man, slow and gravelly", 0),
    ("young_woman", "young woman, bright and clear", 0),
    ("child",       "young child",                   0),
    ("th_female",   "เสียงผู้หญิงนุ่มนวล",                    0),
]


def describe(y, sr: int) -> dict:
    """Pitch and brightness of one clip. The two numbers that separate obvious voice types."""
    import numpy as np
    import librosa
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr,
                                 frame_length=1024, hop_length=256)
    f0v = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    cen = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return {"f0_median": round(float(np.median(f0v)), 1) if len(f0v) else None,
            "f0_iqr": round(float(np.percentile(f0v, 75) - np.percentile(f0v, 25)), 1)
                      if len(f0v) else None,
            "voiced_frac": round(float(np.mean(voiced)), 3) if voiced is not None else None,
            "centroid_median": round(float(np.median(cen)), 1)}


def generate(n_texts: int, seed_base: int, out_dir: Path) -> int:
    import numpy as np
    import librosa
    import soundfile as sf
    import torch

    print(f"[shim] torchaudio.load -> soundfile (torchaudio {_patch_torchaudio_load()})")
    print(f"[voxcpm] {VOXCPM_MODEL}  cfg={CFG_VALUE}  timesteps={TIMESTEPS}"
          "   (imported from colab/gen_colab.py -- never hardcoded here)")
    from voxcpm import VoxCPM

    texts = json.loads((HERE / "texts" / "phase1_known.json").read_text(encoding="utf-8"))[:n_texts]
    out_dir.mkdir(parents=True, exist_ok=True)
    model = VoxCPM.from_pretrained(VOXCPM_MODEL, optimize=True, load_denoiser=False)

    rows, t0 = [], time.perf_counter()
    total = len(texts) * len(CONDITIONS)
    i = 0
    for t in texts:
        for name, ctl, slot in CONDITIONS:
            i += 1
            gen_id = f"{t['id']}__{name}"
            seed = seed_base + slot * 1000 + (zlib.crc32(t["id"].encode()) & 0xFFFF)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            spoken = f"({ctl}){t['text']}" if ctl else t["text"]
            # 🔴 NO prompt_wav_path AND NO prompt_text. That is the whole point: this is design
            # mode. `cli.py:142` -- "`design` does not accept prompt/reference audio" -- and
            # `probe_control.py` proved the restriction is real, by hearing the model read a
            # control prefix aloud when a cloning prompt was also supplied.
            y = np.asarray(model.generate(text=spoken, cfg_value=CFG_VALUE,
                                          inference_timesteps=TIMESTEPS, retry_badcase=True),
                           dtype=np.float32).reshape(-1)
            y = librosa.resample(y, orig_sr=model.tts_model.sample_rate, target_sr=SR)
            sf.write(str(out_dir / f"{gen_id}.wav"), y, SR)
            d = describe(y, SR)
            span = voiced_span_s(y, SR)
            n_chars = len(for_scoring(t["label_text"]))
            rows.append({"clip": f"{gen_id}.wav", "gen_id": gen_id, "text_id": t["id"],
                         "condition": name, "control": ctl, "seed": seed,
                         "text": t["text"], "label_text": t["label_text"],
                         "spoken_text": spoken,
                         "dur_s": round(len(y) / SR, 3), "voiced_s": round(span, 3),
                         "n_chars": n_chars,
                         "chars_per_s": round(n_chars / span, 3) if span > 0.2 else None, **d})
            print(f"  [{i}/{total}] {gen_id:26s} f0={str(d['f0_median']):>6s}Hz "
                  f"cen={d['centroid_median']:7.1f}  {span:5.2f}s", flush=True)

    mp = out_dir / "manifest.jsonl"
    mp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                  encoding="utf-8")
    print(f"\n{len(rows)} clips in {time.perf_counter() - t0:.0f}s -> {mp}")
    print("then read the words back, because a designed voice that drops words is not a voice:\n"
          f"  E:/asr-spike/.venv/Scripts/python.exe adapter/verify_synth.py "
          f"--manifest {mp} --clips {out_dir} --asr biodatlab --tag probe_voice")
    return 0


def analyse(out_dir: Path) -> int:
    mp = out_dir / "manifest.jsonl"
    if not mp.exists():
        raise SystemExit(f"no manifest at {mp} -- run the generate half first")
    rows = [json.loads(l) for l in mp.open(encoding="utf-8") if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["condition"], []).append(r)

    def col(c, k):
        return [r[k] for r in by.get(c, []) if r.get(k) is not None]

    # Paired floor: the same text with no control at two seeds. What design mode varies for free.
    pa = {r["text_id"]: r for r in by.get("plain_a", [])}
    pb = {r["text_id"]: r for r in by.get("plain_b", [])}
    shared = [t for t in pa if t in pb
              and pa[t].get("f0_median") and pb[t].get("f0_median")]
    f0_floor = statistics.mean(abs(pa[t]["f0_median"] - pb[t]["f0_median"]) for t in shared)
    cen_floor = statistics.mean(abs(pa[t]["centroid_median"] - pb[t]["centroid_median"])
                                for t in shared)

    print(f"{'condition':13s} {'n':>2s} {'f0 med':>7s} {'f0 sd':>6s} {'centroid':>9s} "
          f"{'ch/s':>6s} | {'dF0 vs plain':>12s} {'xfloor':>7s}")
    print("-" * 78)
    base_f0 = statistics.mean(col("plain_a", "f0_median")) if col("plain_a", "f0_median") else 0
    for name, ctl, _ in CONDITIONS:
        f0s, cens = col(name, "f0_median"), col(name, "centroid_median")
        if not f0s:
            print(f"{name:13s}  -- no voiced frames measured")
            continue
        m = statistics.mean(f0s)
        sd = statistics.pstdev(f0s) if len(f0s) > 1 else 0.0
        d = m - base_f0
        rates = col(name, "chars_per_s")
        print(f"{name:13s} {len(f0s):2d} {m:7.1f} {sd:6.1f} {statistics.mean(cens):9.1f} "
              f"{statistics.mean(rates) if rates else float('nan'):6.2f} | "
              f"{d:+12.1f} {abs(d) / f0_floor if f0_floor else float('nan'):7.2f}")

    print(f"\nNOISE FLOOR, design mode with NO instruction, same text, new seed:")
    print(f"  pitch    {f0_floor:6.1f} Hz     brightness {cen_floor:7.1f} Hz")
    print("🔴 Design mode invents a voice per draw, so it makes speaker variety for free. A")
    print("   control string that moves pitch less than ~2x this floor is decoration.\n")

    named = [c for c, ctl, _ in CONDITIONS if ctl]
    ms = {c: statistics.mean(col(c, "f0_median")) for c in named if col(c, "f0_median")}
    if len(ms) >= 2:
        spread = max(ms.values()) - min(ms.values())
        hi, lo = max(ms, key=ms.get), min(ms, key=ms.get)
        verdict = ("USABLE" if spread >= 2 * f0_floor else
                   "marginal" if spread >= f0_floor else "NO EFFECT")
        print(f"spread across the named voices: {spread:.1f} Hz "
              f"({lo} {ms[lo]:.0f} -> {hi} {ms[hi]:.0f}), {spread / f0_floor:.1f}x floor"
              f"  -> {verdict}")
        # The one question the Thai string exists to answer.
        if "th_female" in ms and "warm_female" in ms:
            print(f"instruction language: english 'warm female voice' {ms['warm_female']:.0f} Hz "
                  f"vs thai 'เสียงผู้หญิงนุ่มนวล' {ms['th_female']:.0f} Hz "
                  f"(diff {abs(ms['th_female'] - ms['warm_female']):.1f} Hz, "
                  f"floor {f0_floor:.1f})")
    print("\n⚠️ Pitch is a PROXY. It proves voices separated; it cannot prove two clips with the")
    print("   same pitch are the same speaker. Confirm by ear before building on this.")
    print("⚠️ And run verify_synth.py first -- probe_control.py caught the model READING a control")
    print("   prefix aloud, which would show up here as a voice that simply talks for longer.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", default=str(OUT / "probe_voice"))
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out_dir = Path(args.out)
    return analyse(out_dir) if args.analyse else generate(args.texts, args.seed, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
