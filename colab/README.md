# Generating Thai synthetic speech on Colab

Why this exists: the box is out of both kinds of memory and the GPU is full at one concurrent
call (VRAM 14.2 of 16.3 GB with the loop up). Generation is GPU-hours, which is the thing we are
actually short of — so it moves to Colab and this box keeps serving the demo.

🔴 **Not because of storage.** Measured from the corpora already here, Thai speech at 16 kHz mono
costs **~115 MB per hour** (`smartturn/data`: 11,971 clips, 2.7 GB, ~24 h). 100 hours is 11.5 GB.
A 5 TB Drive would hold ~44,000 hours — more than Typhoon's entire 11,000-hour training corpus.
If a plan here starts being shaped by storage, the plan is wrong.

| file | what |
|---|---|
| `gen_colab.py` | the generator. Self-contained: no repo imports, runs on a bare VM |
| `Generate_Thai_Synthetic.ipynb` | the notebook — GPU check, install, mount, smoke test, run |

---

## 🔴 The open decision: there is no text bank for this yet

The notebook speaks whatever text bank you point it at, and **the one the adapter experiment
needs does not exist.** Both banks in the repo are for the *other* thesis direction:

| bank | entries | what it is |
|---|---|---|
| `smartturn/texts_v3.json` | 3,188 | turn-taking. `cls` ∈ complete_plain / complete_midfiller / incomplete_endfiller / incomplete_endconnective, 1,594 POSITIVE / 1,594 NEGATIVE, with `pair_id` matched pairs |
| `smartturn/texts_caller.json` | 3,188 | the same bank before `pair_id` existed |

Use `texts_v3.json` for the **smoke test** — it proves the pipeline works. Do not spend a long
run on it: it is already generated locally as `smartturn/data_v3` (3,188 clips, 707 MB), and
regenerating it on Colab buys nothing.

What the adapter design (the adapter design (`git -C E:/Claude/asr-research show 4734fc5:docs/specs/2026-09-11-synthetic-adapter-asr-design.md`) §5.3) actually needs
is **two** banks, and they are different on purpose:

- **phase 1 — words the decoder already knows.** If rare words appeared here the loss would be
  high because the decoder cannot spell them yet, and the adapter would be blamed for a
  vocabulary problem it cannot fix. Restrict phase 1 to known words and the only remaining
  source of loss is TTS-ness.
- **phase 2 — the rare words themselves.** Product names, place names, postal codes: the audio
  no Thai corpus contains, which is the entire reason synthetic data is worth the risk.

Neither is written. That is the next piece of work, and it is text, not audio — it does not need
Colab at all.

⚠️ Also still open from the design's §10: the **rare-word evaluation corpus**
(`wayu-ai/thai-contextasr-bench`, 2,062 utts, not yet pulled or sized) and **Thai
telephone-channel room impulse responses**, which gate arm 3 — the arm that sets the bar the
whole architecture has to beat. Generating audio does not unblock either.

---

## What the generator preserves from the local pipeline

These are the reasons the local corpus is trustworthy, and a careless port loses them. Full
reasoning is in `gen_colab.py`'s docstring; the short version:

1. **No LoRA, ever.** `voxcpm/app.py` defaults `VOXCPM_LORA` to the mint_v4 adapter — one
   person's voice, cloned from YouTube without consent. Cloning 150 Common Voice references
   *through* it would pull every output toward mint while still appearing to work. This file has
   no flag for an adapter. It is also why this corpus is safe to generate on someone else's
   computer: **nothing here touches the unconsented voice.** The speaker bank is Common Voice
   17.0 Thai test split, **CC0** — public-domain data already on the internet.
2. **Seeds come from each clip's own id, never its position in the run.** This is what makes a
   resumed run produce what an uninterrupted one would have. On Colab it stops being a nicety.
3. **Plan order is a hash**, so any prefix is a representative sample. A run killed at 40% gives
   40% of a corpus, not the first 40% of the file.
4. **48 kHz in, 16 kHz out, via librosa.** VoxCPM2 is natively 48 kHz; the never-resample rule
   guards audio a *caller hears*, and no caller hears this. librosa specifically so clips stay
   comparable with the corpora already on the box.

Two properties are corrected relative to a naive port, both caught by testing rather than
reading:

- **Matched pairs share a voice.** `texts_v3` pairs two texts that differ only in how the
  utterance ends. If voice varied within a pair, voice would become a confound and the contrast
  would no longer be the only difference. Voice is assigned per `pair_id`.
- **The speaker spread is even by construction.** `hash % len(voices)` is an independent draw
  per group — measured on `texts_v3` it gave one voice 9 texts and another 36. Round-robin over
  a hash-ordered group list gives **20–22 each** across all 150.

---

## The Colab gotchas, in the order they bite

1. **Colab's local disk is not your 5 TB.** The VM has ~100–250 GB of *ephemeral* disk, wiped on
   disconnect; the 5 TB is Drive, over a network filesystem. Writing 12,000 small wavs straight
   to Drive is slow and Drive handles huge flat folders badly. Hence shards: generate to local
   scratch, pack one tar per `--shard-clips`, copy that up. **Few big files, always.**
2. **Session death.** Pro disconnects on idle and caps the session, and burns compute units
   (~100/month; L4 ≈ 2/hr, A100 ≈ 11–13/hr). Resume is per shard, so size `--shard-clips` for
   ~15–20 min of generation — a shard in flight when the VM dies is the work you lose.
3. **Smoke-test before the long run.** The local box is Blackwell with `torch.compile` and CUDA
   graphs; Colab gives T4 (sm_75, no bf16) / L4 / A100. A model card is not evidence. `--smoke 3`
   writes loose wavs to `thai-synth/smoke/` — **listen to them.** Waveform metrics produced ~5
   confident wrong conclusions here in one session; transcript containment is the only automated
   check that earned trust, and nothing automated is trusted on how audio *sounds*.
4. **Pick L4, not A100.** VoxCPM's DiT runs on a `(2,64,4)` tensor and is launch-bound — which is
   why TensorRT only *tied* `torch.compile` on it. A bigger card buys little and costs ~5× the
   units.

### One ordering detail that is not cosmetic

`close_shard()` writes the tar to Drive **before** appending the manifest. Reversed, a VM death
between the two leaves the manifest claiming clips no shard contains — and resume trusts the
manifest, so those clips would never be regenerated. A duplicate tar is recoverable; a phantom
manifest row is a silent hole in the corpus.

---

## Building the upload bundle

This repo has no git remote, so the inputs go up by hand — once. From the repo root:

```powershell
$dst = "$env:TEMP\colab-bundle.zip"
Remove-Item $dst -ErrorAction SilentlyContinue
Compress-Archive -Path colab\gen_colab.py, smartturn\refs, smartturn\texts_v3.json -DestinationPath $dst
"{0:N1} MB -> {1}" -f ((Get-Item $dst).Length/1MB), $dst
```

Drop the zip in `MyDrive/thai-synth/` and the notebook's cell 4 unpacks it. Measured **52.0 MB**,
153 entries — `gen_colab.py`, `texts_v3.json`, and `refs/` (150 wavs + `refs.json`) at the top
level, which is the layout the notebook expects.

## How long, and how much

From the local box's observed rate (11,971 clips / ~24 h of audio in ~7 h wall, RTF ~0.29):
roughly **1 hour of audio per 17 minutes of generation**, at ~115 MB/hour on disk. The script
prints its own measured RTF at the end — quote that, not this estimate, and expect Colab's
number to differ from the local one.

**Do not plan a big run.** The design's discipline is cheap arms first, each gated, and arm 1 is
the untrained baseline. 20–30 hours (~3 GB) is enough to get arm 2 and arm 5 signal and to start
the ratio sweep that yields the Thai synthetic-to-real exchange rate nobody has published
(`H7` — EN 7–8×, ES 9×, FR 12×, no Thai figure exists). Generating 1,000 hours before there is a
baseline to compare against is how the `smartturn` EER table happened: four fine-tunes, every arm
at or below the untrained model, and no control to tell "the method failed" from "this setup
cannot learn".
