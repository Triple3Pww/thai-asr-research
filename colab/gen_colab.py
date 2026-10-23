#!/usr/bin/env python3
"""Generate Thai synthetic speech with VoxCPM on Colab -- sharded to Drive, resumable.

    python gen_colab.py --texts texts.json --refs refs/ --out /content/drive/MyDrive/thai-synth
    python gen_colab.py ... --smoke 3        # 3 clips, then stop. RUN THIS FIRST.

WHY THIS FILE EXISTS AND IS NOT `smartturn/gen.py`.

`smartturn/gen.py` is the local generator and it is correct; it just assumes a machine this is
not. It imports `voxcpm/app.py` (FastAPI, `_squeeze`, sys.path games), writes one wav per file
into a directory that lives forever, and trusts the process to survive ~7 hours. On Colab all
three assumptions break: there is no app.py, the VM's disk is wiped on disconnect, and the
session is capped. So this is a port, not a wrapper -- but it preserves the four properties that
make the local corpus trustworthy, and those are the reason to read this docstring:

1. NO LoRA. EVER. `app.py` defaults `VOXCPM_LORA` to the mint_v4 adapter, fine-tuned on ONE
   person's voice -- and that voice was cloned from YouTube without consent. Cloning 150 Common
   Voice references *through* it would pull every output toward mint, quietly destroying the
   speaker diversity the reference bank exists to provide while still appearing to work. This
   file never loads an adapter at all: there is no flag for it. It also means nothing here
   touches the unconsented voice, which is why this corpus is safe to generate on someone
   else's computer.

2. EVERY GENERATION IS SEEDED FROM ITS OWN `gen_id`, NEVER FROM ITS POSITION. That is what
   makes a resumed run produce exactly what an uninterrupted one would have. On Colab this stops
   being a nicety: you WILL be disconnected mid-run, and a position-seeded corpus would silently
   become a different corpus every time it resumed.

3. PLAN ORDER IS A HASH OF `gen_id`. Any prefix of the run is therefore a representative
   sample of the whole bank. A run killed at 40% gives you 40% of a corpus, not the first 40%
   of the text file -- which, if the bank is sorted by class or by source, is a corpus of one
   thing.

4. 48 kHz in, 16 kHz out, via librosa. VoxCPM2 is natively 48 kHz. The repo's never-resample
   rule guards audio a *caller hears*; no caller hears this, and 16 kHz is what Typhoon and
   every ASR here consume. librosa specifically (not torchaudio) so clips stay comparable with
   the corpora already on the local box.

WHAT IS NEW HERE, AND WHY EACH PIECE IS LOAD-BEARING ON COLAB.

- Shards, not files. Clips are generated to local scratch, then packed into one tar per
  `--shard-clips` and copied to Drive. Drive is a network filesystem: writing 12,000 small wavs
  to it directly is slow and it handles huge flat directories badly. Few big files, always.
- Resume reads Drive, not scratch. Scratch is gone after a disconnect; the manifests on Drive
  are the record. Resume granularity is therefore one shard -- size `--shard-clips` so a shard
  is ~15-20 min of generation, not hours, because a shard in progress when the VM dies is the
  work you lose.
- The manifest is appended to Drive only at a shard boundary. Per-clip flushing to Drive (what
  gen.py does locally) would make the network round-trip the bottleneck. The cost is that an
  interrupted shard leaves clips on a dead VM's disk with no record -- which is exactly why they
  are regenerated bit-identically on resume (property 2). The two design choices are a pair; do
  not keep one and drop the other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tarfile
import time
import wave
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------------------------
# Constants. These are COPIES of the local pipeline's values, inlined so this file has no repo
# imports and runs on a bare VM. If the originals change, these are wrong -- the provenance is
# recorded here so a future reader can check.
#   SR / TTS_SR        smartturn/gen.py
#   TAIL_MS / WIN_S    smartturn/cut.py
#   LEAD_FLOOR_RMS     voxcpm/_trim.py
#   CFG / TIMESTEPS    voxcpm/app.py  (VOXCPM_CFG=2.0, VOXCPM_TIMESTEPS=8)
# ---------------------------------------------------------------------------------------------
SR = 16000
TTS_SR = 48000
TAIL_MS = 200.0
WIN_S = 0.01
LEAD_FLOOR_RMS = 0.003
CFG_VALUE = 2.0
TIMESTEPS = 8
VOXCPM_MODEL = "openbmb/VoxCPM2"


class Plan(NamedTuple):
    gen_id: str
    text_id: str
    text: str
    client_id: str
    seed: int
    meta: dict


def _voice_end_s(y: np.ndarray, sr: int) -> float:
    """End of the last audible 10 ms window. 0.0 when nothing clears the floor."""
    win = int(sr * WIN_S)
    n = len(y) // win
    if win <= 0 or n == 0:
        return 0.0
    env = np.sqrt(np.mean(y[: n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
    loud = np.where(env > LEAD_FLOOR_RMS)[0]
    return float((loud[-1] + 1) * WIN_S) if len(loud) else 0.0


def normalise_tail(y: np.ndarray, sr: int, tail_ms: float = TAIL_MS) -> np.ndarray:
    """Cut or pad so the clip ends exactly `tail_ms` after the last audible window.

    VoxCPM samples its own trailing pause (measured 0-2380 ms per reply), so without this the
    clip length carries information about the draw rather than about the speech.
    """
    y = np.asarray(y, dtype="float32").reshape(-1)
    end = _voice_end_s(y, sr)
    if end <= 0.0:
        return np.zeros(0, dtype="float32")
    keep = int((end + tail_ms / 1000.0) * sr)
    if keep <= len(y):
        return y[:keep]
    return np.concatenate([y, np.zeros(keep - len(y), dtype="float32")]).astype("float32")


def write_wav(path: Path, y: np.ndarray, sr: int = SR) -> None:
    data = (np.clip(y, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data)


def load_texts(path: Path) -> list[dict]:
    """A text bank is a JSON list of objects with at least `id` and `text`.

    Every other key is carried through to the manifest untouched, so a bank that already carries
    labels (`cls`, `label`, `particle`, ...) keeps them without this file knowing what they mean.
    """
    bank = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(bank, dict):                       # tolerate {"items": [...]}
        bank = bank.get("items", [])
    out = []
    for i, t in enumerate(bank):
        if "text" not in t:
            raise SystemExit(f"text bank entry {i} has no 'text' key")
        out.append({**t, "id": str(t.get("id", i))})
    return out


def load_refs(refs_dir: Path) -> dict[str, dict]:
    """The speaker bank: Common Voice 17.0 Thai test split, CC0 -- 150 distinct speakers.

    CC0 is why this is safe to upload to Google: it is already public-domain data. The test
    split specifically because it has 4,416 distinct speakers against train's 205, and only
    timbre is being harvested.
    """
    meta = json.loads((refs_dir / "refs.json").read_text(encoding="utf-8"))
    return {r["client_id"]: r for r in meta["items"]}


def plan_generations(texts: list[dict], client_ids: list[str], per_text: int,
                     seed: int) -> list[Plan]:
    """One plan row per (text, voice). The seed derives from `gen_id`; the voice does not.

    TWO THINGS HERE ARE NOT FREE CHOICES, and a hash-mod assignment gets both wrong.

    1. A MATCHED PAIR MUST BE SPOKEN BY THE SAME VOICE. Banks like `texts_v3.json` carry a
       `pair_id`: two texts that differ only in how the utterance ends, meant to be contrasted.
       If voice varies within the pair, voice becomes a confound and the contrast is no longer
       the only difference. So the voice is chosen per GROUP -- `pair_id` when the bank has one,
       otherwise the text id -- and every member of a group inherits it.

    2. THE SPREAD OVER THE BANK MUST BE EVEN. `client_id % len(ids)` is an independent draw per
       group, which is exactly the lumpy thing a speaker bank exists to avoid: measured on
       `texts_v3`, hash-mod gave one voice 9 texts and another 36. Round-robin over a
       hash-ordered group list is even by construction (counts differ by at most one) while
       staying deterministic.

    The cost of (2) is that a voice depends on a group's POSITION in the hashed group order, so
    adding texts to the bank reshuffles voices. That is acceptable because a changed bank is a
    different corpus anyway -- but the SEED still comes from `gen_id` alone, so resuming an
    unchanged bank reproduces it exactly (docstring property 2).
    """
    ids = sorted(client_ids)
    groups: list[str] = []
    seen: set[str] = set()
    for t in texts:
        g = str(t.get("pair_id", t["id"]))
        if g not in seen:
            seen.add(g)
            groups.append(g)

    # 🔴 THE VOICE DEPENDS ON BOTH THE GROUP AND k. Keyed on the group alone, `--per-text 8`
    # gives one text eight clips in the SAME voice differing only by seed -- which multiplies
    # sampling noise and adds no speaker diversity at all. Speaker diversity is the strongest
    # lever in the TTS-for-ASR literature and `--per-text` is the knob that buys it (also the
    # knob the design's Sec 7 ratio sweep turns), so that would quietly defeat both.
    #
    # Construction: order the groups by hash, then voice index = (base(group) + k*STRIDE) % n.
    # STRIDE is coprime to len(ids), so for one group the k values hit n DISTINCT voices -- the
    # 8 clips of a text are 8 different speakers by construction, not by luck. Hashing (g, k)
    # instead would leave it to chance: measured on this bank it gave 8 distinct voices for only
    # 365 of 440 texts, 7 for 74 and 6 for one. Pairs still match, because the same group has
    # the same base and k is the same within a pair.
    groups.sort(key=lambda g: hashlib.sha256(f"voice:{seed}:{g}".encode()).hexdigest())
    n = len(ids)
    stride = next(s for s in (7, 11, 13, 17, 19, 23, 1) if math.gcd(s, n) == 1)
    voice_of = {(g, k): ids[(i + k * stride) % n]
                for i, g in enumerate(groups) for k in range(per_text)}

    plans: list[Plan] = []
    for t in texts:
        g = str(t.get("pair_id", t["id"]))
        for k in range(per_text):
            gen_id = f"{t['id']}_{k}"
            h = hashlib.sha256(f"{seed}:{gen_id}".encode()).hexdigest()
            plans.append(Plan(
                gen_id=gen_id,
                text_id=t["id"],
                text=t["text"],
                client_id=voice_of[(g, k)],
                seed=int(h[8:16], 16),
                meta={k2: v for k2, v in t.items() if k2 not in ("id", "text")},
            ))
    # See docstring property 3: any prefix must be representative.
    plans.sort(key=lambda p: hashlib.sha256(f"order:{seed}:{p.gen_id}".encode()).hexdigest())
    return plans


def done_gen_ids(out_dir: Path) -> set[str]:
    """What Drive says is already generated. The VM's own disk is not evidence -- it is wiped."""
    done: set[str] = set()
    mdir = out_dir / "manifests"
    if not mdir.is_dir():
        return done
    for m in sorted(mdir.glob("*.jsonl")):
        with m.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line)["gen_id"])
                    except Exception:
                        pass                         # a torn last line from a killed VM
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", required=True, help="JSON list of {id, text, ...}")
    ap.add_argument("--refs", required=True, help="dir holding refs.json + the ref wavs")
    ap.add_argument("--out", required=True, help="Drive dir: shards/ + manifests/ land here")
    ap.add_argument("--scratch", default="/content/scratch", help="VM-local working dir")
    ap.add_argument("--shard-clips", type=int, default=500,
                    help="clips per tar. Size it so a shard is ~15-20 min of generation: a "
                         "shard in progress when the VM dies is the work you lose")
    ap.add_argument("--per-text", type=int, default=1, help="how many voices speak each text")
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--retry-badcase", dest="retry_badcase", action="store_true", default=True,
                    help="VoxCPM regenerates when audio_frames >= text_tokens * 6.0 -- its only "
                         "guard against runaway/babble, up to 3 times. ON by default since "
                         "2026-09-16. See the note above the generate() call")
    ap.add_argument("--no-retry-badcase", dest="retry_badcase", action="store_false",
                    help="restore the pre-2026-09-16 behaviour: every seed reproduces exactly, "
                         "and a runaway is silently truncated at max_len instead of retried")
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips this session")
    ap.add_argument("--smoke", type=int, default=0,
                    help="generate N clips straight into Drive and stop. RUN THIS FIRST: it is "
                         "the only check that VoxCPM works on the GPU Colab gave you")
    args = ap.parse_args()

    refs_dir, out_dir = Path(args.refs), Path(args.out)
    scratch = Path(args.scratch)
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    texts = load_texts(Path(args.texts))
    refs = load_refs(refs_dir)
    plans = plan_generations(texts, list(refs), args.per_text, args.seed)
    print(f"{len(texts)} texts x {args.per_text} voices from {len(refs)} refs "
          f"-> {len(plans)} generations")

    done = done_gen_ids(out_dir)
    if done:
        plans = [p for p in plans if p.gen_id not in done]
        print(f"resuming: {len(done)} already on Drive, {len(plans)} to go")
    # --limit and --smoke apply AFTER the resume filter, so repeated calls each advance.
    if args.smoke:
        plans = plans[: args.smoke]
    elif args.limit:
        plans = plans[: args.limit]
    if not plans:
        print("nothing to do")
        return 0

    # -----------------------------------------------------------------------------------------
    # Model. No LoRA (docstring property 1), denoiser off -- it was being loaded on every boot
    # and never used, because the reference clips are denoised offline. `optimize=True` is what
    # the local server uses.
    # -----------------------------------------------------------------------------------------
    import librosa
    import torch

    # 🔴 torchaudio >= 2.9 ROUTES `load` THROUGH torchcodec, WHICH NEEDS FFmpeg DLLs THIS BOX
    # DOES NOT HAVE. `VoxCPMModel.build_prompt_cache` calls `torchaudio.load(prompt_wav_path)`,
    # so every reference-cloned clip -- which is all of them -- dies with
    #     OSError: Could not load this library: ...torchcodec\libtorchcodec_core5.dll
    # and it dies AFTER a 40 s model load, on the first generate(), not at import. Found
    # 2026-09-17 by adapter/probe_control.py, the first cloning run since torchaudio 2.11.
    # soundfile is already a dependency and reads these 48 kHz wavs directly; torchaudio returns
    # (channels, samples) where soundfile returns (samples, channels), hence the transpose.
    # Harmless on Colab, where torchcodec usually does load.
    try:
        import torchaudio
        import soundfile as _sf

        def _load_via_soundfile(path, *a, **k):
            y, sr = _sf.read(str(path), dtype="float32", always_2d=True)
            return torch.from_numpy(y.T.copy()), sr

        torchaudio.load = _load_via_soundfile
        print(f"[shim] torchaudio.load -> soundfile (torchaudio {torchaudio.__version__})")
    except ImportError:
        pass

    from voxcpm import VoxCPM

    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[gpu] {dev}")
    if not torch.cuda.is_available():
        print("NO GPU -- Runtime > Change runtime type. On CPU this is ~50x slower.")
    print(f"[voxcpm] loading {VOXCPM_MODEL} (ref-only, no LoRA, timesteps={TIMESTEPS}) ...",
          flush=True)
    model = VoxCPM.from_pretrained(VOXCPM_MODEL, optimize=True, load_denoiser=False)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    mpath = out_dir / "manifests" / f"manifest-{run_id}.jsonl"

    state = {"shard_n": 0, "rows": [], "dir": None}
    n_done = n_silent = 0
    audio_s = 0.0
    t_start = time.perf_counter()

    def open_shard() -> None:
        state["dir"] = scratch / f"shard-{run_id}-{state['shard_n']:04d}"
        (state["dir"] / "clips").mkdir(parents=True, exist_ok=True)

    def close_shard() -> None:
        """Pack to Drive, append the manifest, then delete local. In that order.

        The manifest is appended AFTER the tar is on Drive. Reversed, a VM death between the two
        would leave the manifest claiming clips that no shard contains -- and resume trusts the
        manifest, so those clips would never be regenerated. A duplicate tar is recoverable; a
        phantom manifest row is a silent hole in the corpus.
        """
        if state["dir"] is None or not state["rows"]:
            return
        tar_path = out_dir / "shards" / f"{state['dir'].name}.tar"
        tmp = tar_path.with_suffix(".tar.part")
        with tarfile.open(tmp, "w") as tf:
            tf.add(state["dir"] / "clips", arcname="clips")
        os.replace(tmp, tar_path)                    # atomic: a .part is never read as done
        with mpath.open("a", encoding="utf-8", newline="\n") as f:
            for row in state["rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        mb = tar_path.stat().st_size / 2**20
        print(f"  -> {tar_path.name}  {len(state['rows'])} clips  {mb:.0f} MB", flush=True)
        shutil.rmtree(state["dir"], ignore_errors=True)
        state["shard_n"] += 1
        state["rows"] = []
        state["dir"] = None

    # Smoke clips go straight to Drive as loose wavs -- the point is to LISTEN to them, and
    # untarring to do that defeats it. The repo's standing rule: waveform metrics do not tell
    # you how something sounds.
    smoke_dir = out_dir / "smoke" if args.smoke else None
    if smoke_dir:
        smoke_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(plans, 1):
        if not args.smoke and state["dir"] is None:
            open_shard()
        ref = refs[p.client_id]
        torch.manual_seed(p.seed)
        torch.cuda.manual_seed_all(p.seed)

        t0 = time.perf_counter()
        y48 = np.asarray(model.generate(
            text=p.text,
            prompt_wav_path=str(refs_dir / ref["wav"]),
            prompt_text=ref["text"],
            cfg_value=CFG_VALUE,
            inference_timesteps=TIMESTEPS,
            # 🔴 WAS False, UNCOMMENTED, UNTIL 2026-09-16. VoxCPM's only built-in guard:
            # if audio_frames >= len(text_tokenizer(text)) * 6.0 it calls the clip a badcase and
            # regenerates, up to 3 times (voxcpm/model/voxcpm.py:479). It catches runaway and
            # babble and NOTHING else -- not skipped words (the test is one-directional), not
            # right-length-wrong-words, not wrong tone, not language drift. Those are
            # adapter/verify_synth.py's job.
            #
            # ⚠️ THE COST, AND IT IS WHY THIS MAY HAVE BEEN OFF: a retry consumes more randomness,
            # so a clip that retried is NOT reproducible from its seed alone. The manifest records
            # `retried` per clip so a corpus can still be audited; --no-retry-badcase restores
            # exact seed reproducibility if an experiment needs it.
            #
            # ⚠️ 6.0 is a default that was never calibrated on Thai, whose tokenisation differs.
            # `sec_per_text_tok` is logged below so it can be set from the Thai distribution.
            retry_badcase=args.retry_badcase,
        ), dtype=np.float32).reshape(-1)
        wall = time.perf_counter() - t0

        audio = normalise_tail(librosa.resample(y48, orig_sr=TTS_SR, target_sr=SR), SR)
        if len(audio) == 0:
            n_silent += 1
            print(f"  [{i}] {p.gen_id}: silent generation, skipped", flush=True)
            continue

        dur = len(audio) / SR
        dst = (smoke_dir if args.smoke else state["dir"] / "clips") / f"{p.gen_id}.wav"
        write_wav(dst, audio)
        # VoxCPM's OWN ratio, with its OWN tokenizer -- the quantity its 6.0 threshold is applied
        # to. Logged rather than acted on, so the Thai threshold can be chosen from the measured
        # distribution instead of inherited. Guarded because the tokenizer is an internal.
        try:
            n_text_tok = len(model.tts_model.text_tokenizer(p.text))
            # SECONDS per text token, not frames per text token. VoxCPM's own test is in audio
            # FEATURE FRAMES, whose duration depends on its audio VAE and patch_size and is not
            # exposed by generate(). Converting with a guessed frame rate would make this a
            # fabricated number; seconds is measured, and it is monotone in the same quantity, so
            # the Thai threshold can be read straight off this distribution.
            sec_per_text_tok = round(dur / max(1, n_text_tok), 4)
        except Exception:
            n_text_tok, sec_per_text_tok = None, None

        row = {"clip": f"{p.gen_id}.wav", "gen_id": p.gen_id, "text_id": p.text_id,
               "client_id": p.client_id, "seed": p.seed, "text": p.text,
               "dur_s": round(dur, 3),
               "n_text_tok": n_text_tok, "sec_per_text_tok": sec_per_text_tok,
               "retry_badcase": args.retry_badcase,
               "shard": None if args.smoke else f"shard-{run_id}-{state['shard_n']:04d}.tar",
               **p.meta}
        n_done += 1
        audio_s += dur

        if args.smoke:
            with mpath.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  [{i}] {p.gen_id}  {dur:.2f}s audio in {wall:.2f}s "
                  f"(RTF {wall/dur:.2f})  voice={p.client_id[:8]}", flush=True)
        else:
            state["rows"].append(row)
            if len(state["rows"]) >= args.shard_clips:
                close_shard()
            if i % 25 == 0:
                el = time.perf_counter() - t_start
                print(f"  [{i}/{len(plans)}] {el/60:.1f} min, {el/i:.2f} s/clip, "
                      f"eta {(len(plans)-i)*el/i/60:.0f} min", flush=True)

    close_shard()

    el = time.perf_counter() - t_start
    print(f"\n{n_done} clips / {audio_s/3600:.2f} h of audio in {el/60:.1f} min "
          f"({n_silent} silent)")
    if audio_s > 0:
        # Quote THIS, not the local box's rate -- a different card gives a different number.
        print(f"generation RTF {el/audio_s:.3f}  ->  {audio_s/el:.1f} h of audio per wall hour")
    if args.smoke:
        print(f"\nNOW LISTEN. The wavs are in {smoke_dir} -- open them from Drive.")
        print("A waveform check cannot tell you the voice is right. Nothing in this repo trusts")
        print("an automated verdict on how audio sounds.")
    else:
        print(f"shards on Drive: {out_dir / 'shards'}")
        print(f"manifest:        {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
