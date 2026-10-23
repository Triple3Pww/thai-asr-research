# Design — making synthetic Thai speech *trainable* for a transducer

**Date:** 2026-09-16
**Status:** approved direction, nothing implemented
**Scope:** the ASR accuracy research line. Self-contained in this repo since 2026-09-16 --
evidence in `results/`, code in `adapter/`, `asr/` and `colab/`. Nothing here reads the voice-loop repo.
**Absorbs** `git show 4734fc5:docs/specs/2026-09-11-synthetic-adapter-asr-design.md` (deleted
2026-09-15 when this repo was briefly reduced to evidence). That design's arms 2, 3, 4 and 6 and its
cut of the adversary all survive here; its arm 5/7 -- the adapter itself -- survives **demoted to a
comparison arm**, which is where its own 2026-09-12 revision had already put it. Its teaching
companion `lessons/adapter.html` still covers the mechanisms.
**Redirects** the same line: the target is no longer "generate synthetic data and fine-tune", which
the academic advisor rejected twice.

---

## 1. The question, in one paragraph

Typhoon ASR's rare-word and spontaneous-speech failures are real and measured here. Synthetic
speech is the only data source we control. But the literature says synthetic speech *"has no
relevant effect"* on transducer ASR (Rossenbach 2021) — and Typhoon is a FastConformer-Transducer.
A 2025 paper refutes that for a Conformer-Transducer, **but only with high-diversity generation**,
and it finds **pitch augmentation ineffective** — in English and French. **Thai is tonal: pitch is
lexical.** The recipe therefore cannot transfer, and nobody has worked out what replaces it. This
design measures which generation attributes make synthetic Thai trainable for a transducer, and
changes the model so it can absorb them.

> **Objective, stated precisely.** Not *"make VoxCPM sound more human"* — naturalness is measured
> to be the wrong target. **Make VoxCPM produce Thai a transducer can learn from: maximise
> variation along attributes that carry no meaning, while holding exactly fixed the one that does.**

---

## 2. Decisions taken (do not silently reverse these)

| # | Decision | Why |
|---|---|---|
| **D1** | **The contribution is the method and the architecture, never the data volume** | The advisor rejected "we have more data than Typhoon" twice. Her words, relayed: *"typhoon can do this in a few days"* |
| **D2** | **Synthetic data stays, as the subject of the research** | She did not reject synthetic data. She rejected it as the whole claim |
| **D3** | **Trainability is measured by training on synthetic and testing on REAL** | Alharthi 2023. Naturalness, MOS and intelligibility are measured **not** to predict ASR gains (Rossenbach 2024) |
| **D4** | **Every arm runs on the transducer.** No Whisper results count | Rossenbach 2021: synthetic data helps attention encoder-decoder and not transducers. A Whisper number does not transfer to our stack |
| **D5** | **Lexical tone is held fixed; pitch register is varied** | In Thai ขาว/ข่าว/ข้าว differ only in pitch contour. An off-the-shelf pitch shift moves contour and register together and corrupts the label |
| **D6** | **Encoder frozen, features cached to disk; only decoder+joint (6.65 M) trains** | Makes an experiment minutes instead of days, so dozens of ablations fit. Also makes general-CER regression structurally impossible |
| **D7** | **`thai-contextasr-bench` is never training data** | CC-BY-NC-4.0. Needs a hard guard in the training script, not a comment |
| **D8** | **The vocabulary/Latin-script hypothesis is dead.** Do not revive it | Tested 2026-09-15, falsified. See §8 |
| **D9** | **The bottleneck adapter is a comparison arm, never the headline** | `H3`'s **6% is the smallest number in a 40-paper family** (`H22`: 19-60%), and `H15` published this design's own reframe in **2023**. It earns its place as one of the ways to separate the paths, not as the claim |
| **D10** | **The Thai synthetic-to-real exchange rate gets measured and reported on its own** | It is publishable **independently of whether any architecture wins**, and it falls out of `--per-text` on a bank that already exists. EN 7-8x, ES 9x, FR 12x; **Thai absent** (`H7`) |

---

## 3. What this rests on

### 3.1 Measured on this box

| fact | number | source |
|---|---|---|
| general CER, read speech, SARA AM folded | **0.0518** | `results/arm1_typhoon_shipped.json` |
| CER on real spontaneous speech | **0.1125** | `results/arm1_rareword_metrics.json` — **2.2× worse than read** |
| entity recall, unbiased | **0.328** (144/439) | same |
| entity recall, two-model splice (D6) | **0.554** (243/439) | `results/d6_splice_summary.json` |
| re-rank the model's own candidates (D1/D2/D3) | **−1 mention** | `results/decoding_arms_summary.json` |
| bias the model's own scores (D4) | **+8 mentions** | `results/d4_boosting_summary.json` |
| a second acoustic opinion (D6) | **+99 mentions** | `results/d6_splice_summary.json` |
| Typhoon's own CTC sibling adds, over the RNN-T | **+5** (and loses 80 of its own) | measured 2026-09-15 |
| biodatlab whisper adds / qwen3-1.7b adds | **+72 / +89** | measured 2026-09-15 |
| oracle union of all six systems | **0.708** (311/439); **128 recovered by nobody** | measured 2026-09-15 |
| four synthetic endpointer fine-tunes | **0.425 / 0.435 / 0.435 / 0.450 EER vs 0.4175 untrained** | `F4` — every arm at or below baseline, **and the real-data control was never run** |

### 3.2 The checkpoint, read rather than guessed

`typhoon-asr-realtime.nemo`: `EncDecRNNTBPEModel`, ConformerEncoder 17 layers, `d_model 512`,
`dw_striding ×8` (0.08 s per frame), `conv_norm_type: batch_norm` with **51 running-stat buffers**,
BPE **2048**, `pred_rnn_layers: 1`, `pred_hidden: 640`, `strategy: greedy_batch`,
`train_ds.augmentor: None`, `variational_noise.std: 0.0`, `fastemit_lambda: 0.0`.

Encoder **108.78 M** (94%) · decoder+joint **6.65 M** (6%) · total **115.45 M**.

🔴 **Those 51 buffers were computed on real speech only** (Typhoon's corpus is 0.03% TTS). They are
the mechanism §5 Phase 1 Arm D attacks.

### 3.3 The contradiction the thesis sits in

| | claim | setup |
|---|---|---|
| Rossenbach et al., ASRU 2021 (13 cites) `10.1109/asru51503.2021.9688255` | synthetic data has **"no relevant effect"** on CTC and transducer models | LibriSpeech-100h, **English**, monophone CTC + **monotonic** transducer, pre-cloning TTS |
| Ogun et al., 2025 (6 cites) `arXiv:2503.08954` | flow-based TTS/VC attribute augmentation cuts **Conformer-Transducer** WER **11% rel. on Common Voice, up to 35% on LibriSpeech** — and **pitch augmentation is ineffective** | **English and French.** Non-tonal |

⚠️ Rossenbach 2021 is a real result, not a law — 13 citations, one corpus, one language, an
architecture generation older than ours. Treat it as a hypothesis to test, not a wall.

### 3.4 Why naturalness is the wrong target, with the receipt

Minixhofer et al.: ASR trained on TTS output **that humans judge as natural** still performs badly
on real speech. Best reported real-to-synthetic ratio is **1.46** — still 46% more errors.
Rossenbach 2024 (`arXiv:2407.21476`) compared five TTS decoders against NISQA MOS and
intelligibility and found **"no clear relations to the ASR performance."**

The mechanism, from the 2026-09-07 survey: **MSE-trained TTS regresses toward the mean of its
training distribution, which is exactly the variance an ASR model needs.** The generator is
optimised to remove what the recogniser must see.

---

## 4. Why this clears the advisor's bar

Her test: *if Typhoon reads the paper, can they copy it in a few days?*

| what | copyable in days? |
|---|---|
| "we generated synthetic Thai data" | **yes** — rejected, twice |
| "we trained a small head on their frozen encoder" | **yes** — it is their own house move (`A28`: a 26 M spotter head and a 29 M aligner head, both on a frozen Thai encoder, both shipped 2026) |
| "we transliterated the bias list" | **yes** — thirty lines of string handling |
| **"pitch augmentation is harmful in tonal languages and here is the tone-preserving variant, plus the normalisation change that lets a transducer absorb it"** | **no** — they must find the limitation, explain it, and fix it. 10,999 hours does not tell them which attributes matter |

And the frame matches what the advisor said makes low-resource work publishable: reviewers expect a
**method** claim ("a technique that gets more from limited data"), not a **data** claim.

---

## 5. The plan

### Phase 0a — does VoxCPM say what it was told? (≈1 week, before everything)

He reports it directly: *"sometimes it make some random voice that not what it should speak."*
That is a named, published failure of autoregressive TTS, not a local misconfiguration — Peinl et
al. 2022: *"due to their stochastic nature, the output may vary from very good to completely
unusable from one inference to another."* The model samples at every step, which is what stops the
voice sounding flat; one unlucky sample puts it in a state its training never covered, and from
there it loops, skips, or drifts off the text. **Nothing errors.** It produces confident audio of
the wrong thing.

🔴 **Settle this before the tone check, because a tone check run on misgenerated audio measures
nothing** — and before Phase 1, because every later arm trains on these clips.

⚠️ **The trap in the objective.** Phase 2 wants *maximum variation*; variation and misgeneration
come out of the same sampling knob. This cannot be tuned to zero without destroying what the plan
needs. The answer is **generate freely and filter hard**, never generate conservatively.

**What is already there and switched off.** `colab/gen_colab.py:367` passes `retry_badcase=False`,
uncommented. From `voxcpm/model/voxcpm.py:479` that flag divides audio frames by text tokens and
regenerates (up to 3 times) when the ratio reaches **6.0**. It catches runaway and babble, and only
that: it is blind to skipped words (the test is one-directional), to right-length-wrong-words, to
wrong tone, and to language drift. Worse, `max_len` is `target_text_length * 6.0 + 10` — the same
ratio — so with retry **off** a runaway is not retried, it is **silently truncated at the cap**.
⚠️ `6.0` is a default not calibrated on Thai, whose tokenisation differs.

⚠️ **MEASURED 2026-09-17, AND IT CHANGES THIS PHASE'S GATE.** The first 12 clips all passed
the utterance gate at `--cer-max 0.10`. Scored on the **rare entity alone**, two of them were
wrecked -- `พลับพลาไชย` came back as `พัฒนาชัย` / `พรรคพราชัย`, entity-CER 0.40-0.60 for three of
four jurors, inside a sentence that still scored under 0.10 because the name is 10 characters of 85.

🔴 **For `phase2_rare` the entity IS the clip.** A sentence built around a rare name that does
not contain a sayable version of that name teaches the model a wrong spelling for a sound -- worse
than not training on it, and exactly the shape of `H37`'s ceiling surviving a verifier. So the gate
now scores the entity separately: `--entity-max 0.25 --entity-votes 2` in `verify_synth.py`.
Reject rate on the pilot **2 of 12 (17%)**, and the verdict is identical anywhere in 0.15-0.25, so
the threshold sits on a plateau rather than a knife edge. ⚠️ Still needs an ear (`H38`: 21-34% of
TTS-derived ASR failures are false alarms).

✅ **And the ceiling is NOT firing hard here, which is the good news.** An exact-substring test
called 4 of 12 unsayable; graded, every one of those four was **one tone mark or one letter** out
(`อุไดเจริญ` heard as `อุได้เจริญ`, 0.11) and the four jurors agreed with each other on the
near-miss, so VoxCPM is producing the name, not refusing it. **Zero of 12 were unsayable.** That
exact-match error nearly went into the record as a finding -- it is the same mistake `F17` is about.

**The layers, cheapest first.**

| # | check | cost | catches |
|---|---|---|---|
| 1 | `retry_badcase=True`, threshold recalibrated from the Thai ratio distribution | free, at generation | runaway, babble |
| 2 | silent output | free (`n_silent` already counts it) | total failure |
| 3 | duration sanity **in both directions**, thresholds from the corpus's own chars-per-second distribution rather than a constant | free | skips as well as run-ons |
| 4 | audio length sitting at the `max_len` cap | free | silent truncation |
| 5 | repetition — self-similarity between non-adjacent windows | cheap, no model | loops the ratio missed |
| 6 | **ASR readback**, plus CER bands, a tail check and an alien-character check | one forward pass per clip | wrong words, wrong tone, language drift |
| 7 | forced alignment (`A27`, 29 M char-CTC head, jitter **2.0 ms**) | one more pass | *where* a skip happened, which CER averages away |
| 8 | listen to a random sample every batch | his time | what no threshold can judge |

🔴🔴 **Layer 6 must not use Typhoon.** Filtering synthetic clips by "Typhoon transcribed it
correctly" keeps exactly the clips Typhoon **already understands** and discards the hard ones —
a filter that removes the training signal. Same shape as `F5`, where synthetic evaluation audio made
every model look ~2x better. Verify with a different architecture:
`biodatlab/whisper-th-large-v3-combined` — best entity recall of the seven systems scored here
(0.435 vs Typhoon's 0.328) and **the only one on disk**. `nectec/Pathumma-whisper-th-large-v3` has
the best overall CER (0.0814) but is **not cached** — selecting it downloads ~3 GB.

🔴 **The verifier's own error is the floor, and its bias is worse than its floor.** biodatlab scores
CER **0.1068** on the 682 real utterances; on synthetic, `F5`'s ~2x gap puts it near 0.05, which is
why the reject band sits at 0.10 rather than 0.02. More importantly: **when the verifier mishears a
rare entity, a good clip is rejected** — so the filter culls hardest exactly where the corpus is
most valuable, which is the opposite of what the plan needs. **The answer is two independent
verifiers that must agree**, with disagreements going to the ear. ⚠️ Not yet implemented; until it
is, the reject pile must be read for rare-entity clips before the keep-rate is trusted.

⚠️ `typhoon-whisper-large-v3` is cached and is **refused** — it shares an org and very likely a
corpus with `typhoon-asr-realtime`, so it shares blind spots. Same circularity in a thinner coat.

**Layer 8 is not optional**, and the repo already says so — `colab/gen_colab.py:412`: *"A waveform
check cannot tell you the voice is right. Nothing in this repo trusts an automated verdict on how
audio sounds."* Its job is to **calibrate layer 6's threshold**: mark 30 clips good/bad by ear and
set the CER cut where the filter agrees with him. Otherwise the threshold is a guess with a decimal
point.

**Best-of-N, and the literature sets N.** ⭐ `H37` (Asaria et al. 2026) names this failure exactly
-- *"silence, terminate early, or collapse into repetitive or hallucinated content"* -- and shows
best-of-N ASR self-verification drives it to **near-zero by N=2**, N=4 on hard prompts, replicated
across four TTS systems and three codecs. 🔴 **So the design changes from FILTER to REGENERATE:**
discarding rejects throws away the keep-rate, where N=2 keeps nearly all of it. ⚠️ Their ceiling
applies directly to `phase2_rare`: *"a rare-word capability ceiling that no self-distillation method
overcomes"* -- round-tripping fixes catastrophic failures, it does not teach a TTS a name it cannot
say. ⚠️ And `H38`: **21-34% of TTS-derived failures are false alarms**, so the reject pile is
suspect by default.

Generate N clips per text (the planner already derives a seed per `gen_id`), run
layers 1-6 on all N, and **keep every clip that passes** — not the single best, which would bias the
corpus toward what the verifier likes and narrow the diversity Phase 2 exists to maximise. Texts
where nothing passes are logged as **hard cases**, never silently dropped. The corpus is 5,480 clips
~ 3 h of generation; N=2 is six hours on Colab, trivial beside a training run.

⭐ **Two numbers fall out, and neither exists for Thai.** VoxCPM's **misgeneration rate by failure
type** — which answers the question behind `H29` from behaviour rather than a model card — and the
**rejection rate as a function of text properties** (rare entities? Latin script? long sentences?
unusual tone sequences?), which feeds Phase 2 directly. It also turns `H36` from a footnote into a
measured component: a similarity scorer held full accuracy on **under 30%** of the TTS data, and
this filter's keep-rate is the Thai version of that number.

**Tooling:** `adapter/verify_synth.py`. **Gate:** the keep-rate and the per-failure breakdown are
reported before Phase 0b runs.

### Phase 0b — build the ruler (≈2 weeks)

You cannot improve trainability without measuring it. Alharthi et al. 2023 (`arXiv:2310.00706`):
**train an ASR on synthetic only, test on real; the real-speech error rate is the trainability
score**, because it measures distribution overlap rather than audio quality.

**Instrument.** Typhoon encoder frozen and **cached** (`[T, 512]` fp16 ≈ 46 MB per hour of audio,
so 10 h ≈ 460 MB); decoder + joint (6.65 M) trainable. Train on a fixed 10 h budget. Test on
held-out **real** Thai: LOTUSDIS (spontaneous) and the Common Voice 150 (read). Score with
`adapter/_score.py`, SARA AM folded on both sides, `--align dp` pinned.

**Gate.** Reproduce a known ordering: 10 h of real LOTUSDIS under the identical recipe must score
better than 10 h of synthetic. If it does not, the harness is broken, not the data.

🔴 **Next, and only after Phase 0a passes: does VoxCPM get Thai TONES right?**
A Typhoon readback is the check and this repo owns the server for it (`asr/run.ps1` on `:8007`).
Write a small **tone check** over a few hundred clips built from minimal tone pairs (ขาว / ข่าว / ข้าว, เสือ / เสื้อ, …).
If the readback disagrees with the label on tone, **the audio does not match the label** and every
number downstream is meaningless. That is a harder failure than the domain gap, and it is `H29`
(how much Thai did VoxCPM see?) arriving as something concrete.

⭐ **This phase also closes `F4`.** Four synthetic fine-tunes sat at or below baseline and the
real-data control was never run, so "synthetic cannot train it" and "this model cannot learn it"
were never separable. Phase 0b's gate *is* that control, at no extra cost.

### Phase 1 — four published ways to separate real from synthetic, compared (≈4 weeks)

**The question underneath.** Rossenbach 2021 tested synthetic data the way everyone uses it —
**naively mixed with real**. The 2026-09-07 survey (§4 item 4) lists **four** published mechanisms
that each beat naive mixing, and ⭐ **none of them has ever been compared head-to-head, in any
language.** Doing that comparison on a Thai transducer is itself an architecture result, and it is
the arm that answers "does the 2021 limitation hold, or was it an artefact of how the data was fed?"

All arms at equal hours, **bracketed in one window with a control at each end**, first sweep after
any server start discarded.

| arm | mechanism | published | new weights? |
|---|---|---|---|
| **A** | real only | — | control |
| **B** | synthetic only | — | control |
| **C** | **naively mixed** — what everyone does, and what Rossenbach tested | — | no |
| **D** | **separate batch-norm statistics** for real vs synthetic; shared weights learn from both, only the normalisation splits, and inference uses the real set | SYNT++, **13% / 18% rel.** | **no** |
| **E** | **synthetic-only bottleneck adapter** — 512 → 32 → 512 after each encoder layer, **33,792 params each**, and **only synthetic audio passes through it**, so the acoustic lesson lands somewhere deletable while the vocabulary lesson lands in the shared decoder | Rolland et al. 2024, **6% rel.** | ~0.57 M, **deleted before anything ships** |
| **F** | **encoder freeze + elastic weight consolidation** — a penalty pulling weights back toward their original values, so learning something new costs something | Zheng 2020 (`H19`), **−57% on OOV**, on an RNN-T | no |

🔴 **Arm D acts on something this box has verified.** Typhoon's encoder carries **51 running-stat
buffers computed on real speech only** (its corpus is 0.03% TTS). Synthetic audio is therefore
normalised by the wrong reference on the way in, and training drifts those statistics toward the
synthetic distribution — invisibly, because nothing errors and the loss looks fine.

⚠️ **Arm E carries a known deviation and must be reported as one.** `H3` (Rolland) **unfreezes the
whole model** in their step 2; we keep the encoder frozen throughout (the `S1` decision of the
2026-09-11 design). That is a hypothesis, not a free choice. Run both if the arm survives.
The two-phase variant — phase 1 freezes the decoder too, so the decoder never eats features from
an adapter that is still random — is `§5.3` of that design.

**Every outcome is publishable.** If C fails while D/E/F work, the 2021 limitation is an artefact of
naive mixing and we have named the fix and ranked the alternatives. If all fail, it is the first
Thai measurement of a four-year-old untested claim and the thesis moves to *why*.

### Phase 1b — the Thai exchange rate (≈1 week, runs alongside)

⭐⭐ **Publishable on its own, whatever the architecture arms do.** The synthetic-to-real exchange
rate — how many hours of synthetic buy one hour of real — is published for **English 7–8×,
Spanish 9×, French 12×**, and **Thai is absent** (`H7`). The survey notes the ratio worsens as the
language gets further from English, so the Thai number is not interpolable.

It falls out of `--per-text` on a text bank that already exists: hold text fixed, vary synthetic
hours, train with the Phase 0 ruler, and find where the synthetic curve crosses a real-data
baseline. No new tooling, no new data, and it is the cheapest genuinely new number in this plan.

### Phase 2 — the attribute ablation, Thai-specific (≈6–8 weeks) — **the contribution**

Generate corpora of identical size and identical text, each varying **exactly one** attribute.
Train with the Phase 0 ruler. Record trainability.

| # | attribute varied | the Thai question |
|---|---|---|
| 1 | speaker identity (many cloned references) | does VoxCPM's natural-language voice design give real diversity, or variations on one voice? (`T16`) |
| 2 | speaking rate / duration | Rossenbach 2023 (`10.1109/asru57964.2023.10389782`) shifted synthetic duration distributions toward real and measurably improved ASR. Untested for Thai |
| 3 | channel and noise, **via the cloning reference** | clone from noisy, spontaneous, telephone Thai and the clone inherits the channel. Labrak et al. 2026 closed the gap with room-impulse-response convolution *after* generation; conditioning *at* generation is untested. 🔴 **Thai telephone-channel RIRs are the blocker** the 2026-09-11 design already hit; **codec and narrowband augmentation is the substitute**, and unlike an RIR replication it is *our* intervention |
| 4 | **pitch register, lexical tone held fixed** | 🔴 **the one nobody has done** |
| 5 | **pitch varied naively** (control) | if this *damages* Thai where it was merely ineffective in English, that is the finding |

🔄 **PROBED 2026-09-17/18, AND THIS TABLE NOW HAS ANSWERS AND ONE CASUALTY.**
`adapter/probe_control.py` and `adapter/probe_voice.py`, both against VoxCPM2, both with a
seed-only noise floor as a condition, both cross-checked by ASR readback.

| row | probe result | verdict |
|---|---|---|
| 1 speaker identity | the natural-language design control does **nothing**: "deep male voice" minus "warm female voice" = **-4.4 Hz** paired, against **37.8 Hz** the seed moves for free. But design mode invents a speaker per draw, pitch 105-251 Hz across 4 sentences, readback CER median 0.000 | ✅ **`T16` ANSWERED: real diversity, zero steering.** An unbounded speaker supply that cannot be asked for anything |
| 2 speaking rate | reference selection moves rate 11.83 -> 16.82 ch/s, **5.2x** the 0.96 ch/s floor, readback clean | ✅ works, but the usable range is **1.42x**, not the reference bank's 3.3x -- VoxCPM2 compresses a reference's rate toward its own default |
| 3 channel and noise via the reference | **untested.** Rate transfers through the cloning reference, so channel plausibly does too -- this is now the most promising untested row | ❓ probe it the same way |
| 4-5 pitch register | 🔴 **no pitch lever exists at generation.** Not by control string (4 Hz on a knob whose own noise is 38), and reference selection changes the whole speaker, not the register | 🔴 **must be post-processing** |

🔴 **SO THE CONTRIBUTION MOVES TO POST-PROCESSING, AND THAT IS NOT THE RETREAT IT LOOKS LIKE.**
`R2`'s route (a) was listed as the weakest because plain augmentation is closed ground. But rows
4-5 were never asking VoxCPM for a knob it might have -- they ask for a transformation that scales
and shifts an F0 track **while preserving its contour shape**. That is a signal-processing
operation by nature, it needs no TTS cooperation, and doing it after generation gives *exact*
control over the transformation, which is precisely what a "tone preserved" claim has to
demonstrate. The novelty was never "we post-processed"; it is "we post-processed in a way that
holds lexical tone while moving speaker register", and nothing in the probes touches that.

⚠️ **What the probes DO cost rows 4-5:** the clean factorial is gone. Phase 2 cannot hold speaker
identity fixed while varying another attribute at generation, because identity is drawn with the
seed. The ablation has to be built the other way round -- generate once, then transform the SAME
clip several ways, so the speaker is held fixed by construction rather than by control.

🔴 **HIS CHALLENGE, 2026-09-18: "why we need to control pitch at all, why not just gen with
difference prompt or ref" -- and on the part that is measurable, he is right.**

Measured that hour: median pitch of 60 real Common Voice Thai clips against VoxCPM2's free
design-mode draws.

| | real Thai | synthetic, free draws |
|---|---|---|
| spread p10-p90 | 139 Hz | **125 Hz** |
| below 130 Hz | 28% | 38% |
| above 220 Hz | 23% | 12% |

**There is no coverage hole.** Changing the seed already spans roughly the pitch range of real
Thai speakers -- thin at the top, heavy at the bottom, nothing that justifies building a transform
to fix it. ⚠️ n=16 synthetic draws, so the tails are soft. **The "VoxCPM cannot make deep or high
voices" argument for rows 4-5 is dead, and it should not be revived without new evidence.**

**What survives is the claim, not the corpus.** Two reasons, and they are the only two:

1. **A seed moves everything at once** -- pitch, timbre, rate, style. If training on many seeds
   helps, nothing says *which axis* helped. That is "we made more data and it got better", which
   is precisely the contribution the advisor rejected. Isolating one axis needs a transform on a
   FIXED clip; a seed cannot do it.
2. **The finding is negative and Thai-specific.** Pitch and speed perturbation are standard ASR
   augmentation, in everyone's recipe. In a tonal language they may be quietly damaging. Row 5
   already says this -- *"if this damages Thai where it was merely ineffective in English, that is
   the finding"* -- and it is the half of rows 4-5 that his challenge leaves untouched.

➡️ **So the corpus is built his way** -- seeds and references, which are free and cover the range.
The pitch work earns its place as the *claim*, and it shrinks to a three-way comparison on
identical clips: no shift / naive shift / tone-preserving shift.

🔴 **AND THAT COMPARISON HAS A PREREQUISITE NOBODY HAS RUN.** Building a tone-preserving transform
assumes the careless one hurts. `adapter/probe_pitch.py` tests that first, on REAL audio, with a
phase-vocoder artifact control and a **TONE-CER** -- edit distance over the tone marks alone
(`_thai.tone_sequence`), which is separable because Thai writes tone as characters. If uniform
pitch perturbation turns out not to damage Thai tone, **rows 4-5 need reshaping before anything is
built**, and that is an afternoon's answer rather than week six's.

**Rows 4 and 5 are the paper.** A Thai tone is a **contour** — a shape over time relative to the
speaker's own range. The **shape** makes the word; **where it sits and how wide it is** makes the
speaker. The transformation therefore scales and shifts the pitch track while preserving its shape,
which a flat pitch shift does not.

⚠️ **Engineering unknown, resolve in week one of this phase:** does VoxCPM expose enough control to
vary one attribute at a time at generation, or must this be post-processing on generated audio?
Ogun's flow-based models had explicit attribute control; VoxCPM may not. If not, the ablation is
weaker and the design must be revisited before committing the eight weeks.

**Why it generalises:** a tone-preserving diversity recipe applies to Vietnamese, Mandarin, Lao and
Burmese. That is a method contribution, not a Thai curiosity.

### Phase 3 — rebuild the generator around what Phase 2 ranks first (≈4 weeks)

Two fixes the literature already prescribes and this repo has not made:

- **`H16` / Yuen et al. 2023** (`10.1109/apsipaasc58517.2023.10317116`): **multiple TTS systems beat
  one TTS with many voices — 9.8% lower WER at identical data size**, because different systems
  occupy different regions of speaker-embedding space even conditioned on the same speaker.
  `colab/gen_colab.py` is built the weaker way.
- **`H36`:** a similarity scorer held full accuracy on **under 30%** of the TTS data.
  the readback idea exists (the endpointer line reads every clip back through Typhoon) but that
  script stayed in the voice-loop repo. **Write the readback here, and make it a filter, not a
  report** — this repo owns `asr/server.py`, which is all it needs.

- **`H28` / Lim et al. 2025: pretrain on synthetic first, then transfer to real.** The reverse
  order is measurably worse, and the dominant error is **prosody mismatch**. This is an ordering
  decision, so it costs nothing to get right and cannot be retrofitted. ⚠️ The prosody control that
  would exploit it lives in the voice-loop repo's TTS server, so the lever here is the generation
  request, not that code.

Optionally, **Hard-Synth** (`arXiv:2411.13159`): generate the samples the model finds *hard* rather
than more of everything — 6.5%/4.4% relative, data-efficient. This is boosting, and it is only
possible because generation is on-demand.

---

## 5b. What this sounds like to the advisor

> *"The literature says synthetic speech helps attention-based ASR and has no relevant effect on
> transducers — but that was measured with the data **naively mixed in**. Four published mechanisms
> each beat naive mixing and **none has been compared head-to-head, in any language**; we do that
> comparison on a Thai transducer. A 2025 paper shows high-diversity generation does reach a
> Conformer-Transducer, but it finds pitch augmentation useless — in English and French. **Thai is
> tonal: pitch is lexical**, so the recipe cannot transfer. We measure which generation attributes
> make synthetic Thai trainable, show that naive pitch augmentation is harmful where it was merely
> useless in English, and propose tone-preserving pitch augmentation. We also report the first Thai
> synthetic-to-real exchange rate, which exists for English, Spanish and French and not for Thai."*

Data is the subject. The contributions are a measurement instrument, a head-to-head comparison
nobody has run, a language-specific correction to the current best recipe, and a number that exists
for three European languages and not for this one. **None of it is a setting**, and 10,999 hours of
real speech does not tell them which attributes matter.

---

## 6. Protocol rules that must hold in every arm

Each of these cost a session to learn. Violating one invalidates the run.

1. **Fold Thai SARA AM (U+0E33 vs U+0E4D U+0E32) on BOTH sides of every CER**, and write training
   labels **decomposed**. `unicodedata.normalize` does nothing — measured: NFC changed 0 of 150
   clips. Worth **26.6% of mean CER**. One definition: `adapter/_thai.py`.
2. **Unspaced Thai number WORDS, never digits.** 0 of 53 real turns contained an ASCII digit, and
   the tokenizer contains **zero digit tokens** — it is structural, not stylistic.
3. **Bracket everything compared back-to-back in one window**; discard the first sweep after any
   server start. The box drifts 25–45% across an evening and not equally between engines.
4. **Run the control even when nothing changed**, and **pin any dependency that moves a number**
   (B-CER/U-CER shift 4.3% on whether `kaldialign` is installed — hence `--align`).
5. **Compare `bias@N` only against `dis@N` at the same N.** Growing a bias list deflates U-CER for
   free; the `none` comparison is partly artefact.
6. **`$env:ADAPTER_OUT = "E:/Claude/asr-research/results"` before every arm**, or results land in an
   empty `adapter/_out/` and a delta against nothing is not a delta. Wired 2026-09-15, **not yet run
   once** — expect to fix something the first time.
7. **Capture stderr first, conclude second.** On this box a silent exit-0 during model load is as
   often the page file as it is VRAM.

---

## 7. Environment and budget

- ASR Python: `E:/asr-spike/.venv/Scripts/python.exe` — NeMo 3.0.0, torch 2.13.0+cu130, CUDA available.
- GPU: RTX 5060 Ti, **16311 MiB**, ~14.4 GiB free with the loop down. Commit limit 39.5 GB, ~15.5 GB free.
- Frozen-encoder training (D6): **< 4 GB**, minutes per run on cached features.
- Data on disk: LOTUSDIS **13 GB / 160,915 utterances**, CC-BY-SA-4.0 (`A21`);
  `thai-contextasr-bench` 247 MB, **CC-BY-NC-4.0, eval only** (D7); `nemo-ctc` 516 MB, CC-BY-4.0.
- Text banks: `phase1_known` 440 rows, `phase2_rare` 245 rows. At `--per-text 8`:
  **5,480 clips ≈ 10.2 h ≈ 1.09 GB ≈ ~3 h of generation.**
- Phases 1 and 2 need the GPU **with the voice loop down**, repeatedly. Budget it, or rule 3 breaks.

---

## 8. Out of scope — hypotheses tested and killed (do not revive)

Kept deliberately. A rejected row is worth more than a quiet deletion.

- ❌ **"Typhoon cannot spell Latin-script entities, and that is the rare-word hole."**
  The tokenizer facts are true — **61 of 2048 tokens carry a Latin letter, only two are
  multi-character, `Khlong Toei` costs 9.63 tokens at 0.90 chars/token against 6.25 at 1.48 for
  Thai**. The *inference* is false. `scorer/metrics.py:278` credits an entity as recalled if **any**
  of its forms appears, and `score.py:49` builds those from `form_th`, `form_latin` and both alt
  lists — so writing คลองเตย already scores a hit. The control: Latin-to-Thai span recall ratio is
  **0.91 for ours** against **0.89 for a Whisper with full Latin subwords**, and **1.06 for
  qwen3-asr-1.7b**, where Latin is *easier*. Script is not what separates these models.
- ❌ **"The script mismatch is a novel metric finding."** SN-WER (2026,
  `10.18653/v1/2026.mellm-1.24`) published script-normalised WER for five Indic languages. Cite it;
  do not claim it.
- ❌ **Contextual biasing in any form.** Typhoon shipped it four ways on 2026-09-08 with its own
  benchmark (`A26`, `F15`).
- ❌ **"First Thai code-switched ASR."** Taken: Mixture-of-Experts decoder, IEEE Access 2025
  (`10.1109/access.2025.3562146`); NECTEC did the acoustic side in 2020.
- ❌ **The adversary arm** (`F14`): no published number, premise contradicted by its own source.
- ⏸️ **The bottleneck adapter is NOT cut — it is Phase 1 Arm E.** Its own 2026-09-12 revision
  demoted it from headline to comparison arm (`H3`'s 6% is the smallest number in a 40-paper
  family; `H15` published its reframe in 2023). Demoted is not dead: as one of four ways to
  separate the paths it earns a row, and the comparison is the contribution.
- ⏸️ **A second trained encoder fused via acoustic-model fusion** (Lei et al. 2023,
  `10.1109/asru57964.2023.10389720`; Li et al. 2019, `10.1109/taslp.2019.2959721`). Strong, and
  supported by the +5 / +72 / +89 complementarity measurement — **parked**, because it needs a
  second encoder trained from scratch on a card shared with a live loop. Revisit after Phase 2.

⚠️ **Still live but not yet chased:** the D6 splice's `tau` is **−1.5 from Typhoon's README and has
never been swept on this box.** One afternoon. Do it before claiming anything about the fusion rule.

---

## 9. Risks and kill criteria

| # | risk | how it shows | what we do |
|---|---|---|---|
| R1 | 🔴 **VoxCPM misgenerates** -- observed by him: *"sometimes it make some random voice that not what it should speak"* | the Phase 0a keep-rate is low, or failures concentrate on the rare-entity texts the plan depends on | **stop and fix generation before anything trains on it.** `retry_badcase=True` first, then the filter. A low keep-rate on rare entities specifically would undercut the whole vocabulary premise |
| R1b | **Thai tone accuracy is unverified** and everything rests on it | readback disagrees with the label on minimal tone pairs | **stop.** Fix or replace the TTS before Phase 1. ⚠️ Only testable once R1 is settled -- a tone check on misgenerated audio measures nothing. ⬇️ **DOWNGRADED 2026-09-17:** VoxCPM2 reports Thai **CER 0.94%** on its own 30-language benchmark and **WER 2.961** on Minimax-MLS. Thai tone marks are characters, so systematic tone errors would surface in a CER that low. Their test and their text -- still worth running, but no longer the likeliest way this dies |
| R2 | ~~VoxCPM may not expose per-attribute controls~~ **CONFIRMED 2026-09-17: it does not** | `generate()` exposes no speech attribute; the only attribute lever is a natural-language string prepended as `(control)text` | 🔴 **Phase 2 cannot run as designed.** Three routes remain and one must be chosen BEFORE the eight weeks: **(a) post-process** the generated audio — what Labrak did with RIRs, and the only route with direct precedent; **(b) reference selection** — clone from references differing in one factor, which is his own instinct, costs nothing, but needs a reference set built for it; **(c) natural-language control** — untested, and it guarantees nothing about what else moved, so every corpus would have to be measured rather than assumed. ⚠️ (c) alone cannot deliver tone-preserving pitch variation: there is no pitch parameter to hold tone against |
| R3 | Rossenbach 2021 may simply not hold for Thai | Phase 1 Arm C already works fine | the "limitation" framing weakens; Phase 2 becomes the whole thesis. Still sufficient, but know it is the likely shape |
| R4 | Box contention | the loop and the training want the card at once | schedule; never break the bracketing rule to save time |
| R5 | Phase 0's ruler is too small to be sensitive | real and synthetic score the same | increase the trainable surface (unfreeze top encoder layers) — costs the D6 guarantee, so only with an explicit general-CER control |

---

## 10. Open questions to answer, in order

1. **How often does VoxCPM say something other than the text, and in which failure mode?** (R1 -- blocks everything, and is a publishable Thai number)
2. **Does VoxCPM produce correct Thai tones?** (R1b — answerable only after 1)
3. ~~How much Thai was VoxCPM trained on?~~ **Asked 2026-09-17: NOT PUBLISHED.** 2M+ hours over
   30 languages, no per-language split, and the card itself concedes *"performance varies across
   languages depending on training data availability."* Thai intelligibility is strong (their
   CER 0.94%) but `H29` is about **coverage**, which no document addresses — so it is now
   answerable only by measurement, and their 0.94% is the control Phase 0a measures against.
4. ~~What per-attribute control does VoxCPM expose?~~ **ANSWERED 2026-09-17: none that is
   numeric, and it constrains Phase 2.** `generate()` takes `cfg_value`, `inference_timesteps`,
   `min_len`/`max_len`, `normalize`, `denoise`, `retry_badcase*`, `streaming` — **not one of which
   is a speech attribute.** The levers that exist: the **cloning reference** (inherits whatever the
   reference has), **`cfg_value`** (faithfulness vs variety, blunt), the **seed**, LoRA, and **voice
   design — a NATURAL-LANGUAGE string injected as literally `(control)text`** (`cli.py:66`,
   `build_final_text`). 🔴 So "vary exactly one attribute and hold the rest fixed" is **not
   implementable at generation time**, and there is **no pitch parameter at all** — which is the
   core of the proposed tone-preserving augmentation. See the revised R2.
5. **How large is the read-vs-spontaneous penalty on this box?** He reports Typhoon missing ordinary
   conversation; the instrument already shows **0.0518 read vs 0.1125 spontaneous**. LOTUSDIS makes
   a proper measurement cheap, and no such Thai figure is published (`A19` reports only that a gap
   exists).
6. ~~Does the `tau` sweep leave anything for a learned fusion rule?~~ **SWEPT 2026-09-17, and the
   answer argues FOR one.** A better *constant* does not exist: `-1.5` is the largest |tau| at which
   pure decoys stay inert (144/439, three times, bit-identical) and the last that passes the
   over-biasing gate (U-CER 0.10351, *below* the 0.10571 unbiased baseline). Past it a list with no
   gold in it starts **destroying** correct mentions -- 135/439 at `-2.0`, 126/439 at `-2.5`. But
   **~45 more gold mentions sit between `-1.5` and `-2.5`** (243 -> 288) and the paired delta grows
   the whole way, so they are gold rather than noise -- **unreachable by any constant**, because the
   loosening that admits them admits the fabrications. A learned, input-dependent threshold is
   exactly the shape of thing that could take them selectively. Closing control reproduced
   byte-exactly, so the window is certified. `HISTORY.md` 2026-09-17 (later).

---

## 11. References

Rossenbach et al. ASRU 2021 `10.1109/asru51503.2021.9688255` · Ogun et al. 2025 `arXiv:2503.08954` ·
Alharthi et al. 2023 `arXiv:2310.00706` · Hu et al. ICASSP 2022 `10.1109/icassp43922.2022.9746217` ·
Fazel et al. Interspeech 2021 (`H25`/`F13`) · Labrak et al. 2026 `arXiv:2606.29031` ·
Minixhofer et al. Interspeech 2025 `10.21437/interspeech.2025-2750` ·
Rossenbach et al. ASRU 2023 `10.1109/asru57964.2023.10389782` ·
Rossenbach et al. 2024 `arXiv:2407.21476` · Yuen et al. APSIPA 2023 `10.1109/apsipaasc58517.2023.10317116` ·
Yu et al. 2024 `arXiv:2411.13159` · Unni et al. ICASSP 2022 `arXiv:2203.02317` ·
Lei et al. ASRU 2023 `10.1109/asru57964.2023.10389720` · Li et al. TASLP 2019 `10.1109/taslp.2019.2959721`.

Full per-claim ledger: `E:\Claude\asr-research\LITERATURE.md`. Synthesis:
`RESEARCH-FINDINGS.md`. TTS-for-ASR survey: `docs/notes/2026-09-07-tts-for-asr-survey.md`.
