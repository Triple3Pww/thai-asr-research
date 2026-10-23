# Training speech models on synthetic (TTS) audio — the state of the art

**Date:** 2026-09-07 · **Method:** 6 Consensus queries, ~120 unique records, no web search
(he asked for Consensus specifically). **Rows live in `LITERATURE.md` §2.9 as `G1`–`G24`.**

**Why this was searched.** Typhoon ASR's own paper says **3.2 of 10,999 hours (0.03%)** of its
training corpus is TTS-generated, and that those 3.2 hours exist for one narrow job: forcing
correct number/date normalisation. Meanwhile `smartturn/data_v3/` in this repo is **~100%
VoxCPM-generated**. The question was whether Typhoon's 0.03% is a *finding* or an artefact of
what data they happened to have.

> **It is a finding.** The literature has converged on it since 2019, and the convergence is
> quantitative, not vibes.

---

## 1. The headline, in one paragraph

Synthetic speech works for ASR, reliably, and has for six years — **but as a targeted supplement,
never as a replacement.** Where it wins is narrow and specific: teaching a model *words it has
never heard* (rare words, out-of-vocabulary terms, named entities, numbers). Where it loses is
broad: teaching a model *how human speech actually sounds*. Typhoon's 3.2 hours sit exactly on
the winning side of that line, and the size of the losing side has been measured repeatedly.

## 2. The exchange rate — how much synthetic buys one hour of real

This is the number worth carrying around. Three independent measurements, three setups:

| study | setup | synthetic needed for parity with real |
|---|---|---|
| Dhamyal et al. 2024 (Voicebox) | clean conditions | **7×** |
| Dhamyal et al. 2024 (Voicebox) | noisy conditions | **10×** |
| Prakash et al. 2026 (XTTS) | English | **8×** |
| Prakash et al. 2026 (XTTS) | Spanish | **9×** |
| Prakash et al. 2026 (XTTS) | French | **12×** |

**So: roughly 8–12 hours of synthetic audio buys you one hour of real audio**, and the ratio gets
worse as conditions get noisier and as the language gets further from English.

The ceiling is measured too. Minixhofer et al. hold **the best reported real-to-synthetic WER
ratio to date at 1.46** — i.e. a model trained on the best available fully-synthetic corpus still
makes ~46% more errors than the same model on real data — and they say plainly that "a large gap
remains."

## 3. Why the gap exists — and it is NOT naturalness

The most useful thing in this literature is that it kills the intuitive explanation. Minixhofer
et al. state the paradox directly: ASR systems trained on TTS output **that humans judge as
natural** still perform badly on real speech. Naturalness and trainability are different axes.

Four mechanisms, each measured:

**(a) Diversity, not quality — and this is the dominant one.** Czyznikiewicz et al. 2024 ran the
decisive comparison: a voice-cloned corpus of *lower* audio quality but *more distinct voices*
beat a conventional-TTS corpus of higher quality with few voices. More importantly, **low-variability
synthetic speech saturates** — adding more of it stops helping — while high-variability speech
keeps paying even at +30% total data.

**(b) Prosody and duration mismatch.** Lim et al. 2025 decomposed the errors: **substitution
errors dominate** in synthetic-trained ASR, and they track the *prosody* mismatch specifically.
Rossenbach et al. 2023 isolated one component of that — **phoneme duration modelling in
non-autoregressive TTS** — and showed that shifting synthetic duration distributions toward real
ones measurably improves the resulting ASR.

**(c) Oversmoothing.** MSE-trained TTS regresses toward the mean of its training distribution,
which is exactly the variance an ASR model needs. DDPM-based TTS beats MSE-based *once* it has
enough data to approximate the real distribution — below that, the oversmoothed MSE model
actually trains better ASR.

**(d) Missing acoustic irregularity — the best-evidenced fix.** Labrak et al. 2026 probed a
SLAM-ASR model to find *where* it distinguishes real from synthetic, and localised the signal to
the **early-to-middle layers**, disrupted most by temporal and prosodic perturbation. Their fix
is the quotable one:

> convolving synthetic audio with room impulse responses (RIRs) narrows the gap **not by making
> synthetic speech sound cleaner or more natural, but by reproducing the acoustic irregularities
> of real recordings.**

With RIR augmentation plus layer selection they matched a fully-real baseline using **25% of the
real speech (13.6 h)**, and beat it above that.

## 4. What actually works — the recipe, ranked by evidence

1. **Maximise speaker diversity before anything else.** VAE-sampled speaker augmentation gives
   **30% relative WER reduction** on 50 h Switchboard (Du et al. 2020). Counter-intuitively,
   running *multiple different TTS systems on one speaker* beats *one TTS system on many
   speakers* — **9.8% lower WER at identical data size** — because different TTS systems occupy
   different regions of speaker-embedding space even when conditioned identically (Yuen et al. 2023).
2. **Add real-world acoustics.** RIR convolution (§3d), multi-style training, GAN+MTR — the last
   giving **35% relative WER reduction** on a voice-search task.
3. **Text diversity is worth as much as acoustic diversity, and more when you have little data.**
   Prakash et al.: lexical diversity dominates below ~500 h; acoustic variability becomes the
   binding constraint beyond 1000 h. Liu et al. 2023 found that selecting synthetic samples by
   *dissimilarity* to real speech lets you keep **under 30% of the TTS data** at no accuracy cost.
4. **Do not naively mix real and synthetic — separate the paths.** Four approaches that each beat
   naive mixing: adapters where **only synthetic passes through the adapter** (Rolland et al.
   2024, 6% relative); **domain-adversarial training** (JODAL, Kim et al. 2026 — which *surpassed
   models trained on 100% real data*); **task arithmetic** (SYN2REAL, Su et al. 2024, 10.03%
   average WER improvement); and encoder freezing + elastic weight consolidation (SynthASR).
5. **Order matters.** Pretrain on synthetic, then transfer to real. The reverse is worse
   (Lim et al. 2025).
6. **Filter.** Confidence-based filtering of TTS samples beat naive augmentation by **6.63
   percentage points, p < 0.001**, retaining 59% of generated samples (Mrayyen et al. 2025).

## 5. 🔴 The architecture trap — this one is about our stack

Rossenbach et al. 2021 compared synthetic-data gains across ASR architectures and found the
benefit is **not** uniform:

> synthetic data substantially helps **attention encoder-decoder** models, but has
> **"no relevant effect"** on **CTC** and **transducer** models.

**Typhoon ASR is a FastConformer-*Transducer*.** The streaming model is a Nemotron transducer.
So the architecture family in our loop is the one the literature says benefits *least* from TTS
augmentation. Whisper is attention encoder-decoder, which is the family that benefits — which
means **a TTS-augmentation result obtained on Whisper does not transfer to our ASR**, and any
experiment here must be run on the transducer to count.

`UNTESTED` here. But it reframes the Typhoon 0.03% figure: they may have used almost no synthetic
audio *because their architecture could not spend it*.

## 6. Where synthetic wins big — and it is exactly Typhoon's use

Every large win in this literature is a **vocabulary** win, not an acoustic one:

- **57% relative WER reduction on OOV-containing utterances**, no degradation elsewhere, using
  TTS audio for out-of-vocabulary words with elastic weight consolidation on the encoder
  (Zheng et al. 2020).
- **>65% relative improvement** recognising medication names, no degradation on existing
  applications (SynthASR, Fazel et al. 2021).
- **57.6% → 16.5% WER** on a rare-word set (road names and addresses) **using no real audio at
  all** (Yuen et al. 2023).
- **43–44% B-WER reduction** via TTS-synthesised multi-pronunciation contextual biasing, zero-shot
  on Whisper (Liu et al. 2025).

Typhoon's 3.2 hours are for "numeric normalisation… preventing common hallucinations in number and
date transcription." That is precisely this category: teach the model **what to write**, not **how
to hear**. The 0.03% is the literature's consensus sweet spot, arrived at independently.

⭐ Note for the repo: **Burin Naowarat et al. 2023** (Interspeech) is a Thai-affiliated group doing
exactly this — training contextual-biasing adapters on synthesised audio, with a projection module
plus consistency regularisation to align synthetic encoder features to real ones. Same performance
using **half the real speech**; **16% WER reduction** when a full real training set is extended
with synthetic. This is the closest published work to a Thai synthetic-data programme.

## 7. 🔴 The `smartturn/` question — where this gets uncomfortable

Everything above is about **transcription**. `smartturn/` is not transcription — it is a
paralinguistic classifier asking *has this person finished their turn?* Whether the TTS-for-ASR
conclusions transfer is a separate question, and the turn-taking literature answers it badly for us.

**The load-bearing finding — Ekstedt et al. 2023.** They evaluated three commercial and two
open-source TTS systems for their ability to *produce* turn-taking cues, scored by a Voice Activity
Projection model. Result:

> TTS, trained on read or spontaneous speech, produce **strong turn-hold but weak turn-yield
> cues** … commercial TTS "often produce ambiguous signals."

**Read that against `smartturn/data_v3/`.** We are training a turn-*end* detector on audio
generated by a system that is measurably weak at producing turn-*end* cues. The label file says
"this turn is finished"; the waveform may not carry the acoustic evidence that a human would
produce. That is a much sharper concern than a general "synthetic data is risky" — it names the
exact property that is missing and the exact task that needs it.

It is **not disqualifying**, and two things soften it:

- The **ETD Dataset** (SpeculativeETD, Ok et al. 2025), the first public end-turn-detection
  dataset, deliberately mixes **TTS-generated synthetic speech with real web-collected speech** —
  so the field treats synthetic as a legitimate *ingredient* for this exact task.
- Our own signal is largely **lexical** (Thai final particles ครับ/ค่ะ/นะ), and lexical content
  survives synthesis perfectly. It is the *acoustic* turn-yield cue that TTS renders weakly.

**But it does force one rule:** you may train on synthetic; you may not **evaluate** on synthetic.
An eval set of real Thai callers is the non-negotiable part, and `CLAUDE.md` already concedes the
current corpus cannot evaluate turn-taking (particle hit 3/22).

## 8. ⭐ Two findings that support the thesis claim directly

**Turn-taking models do not cross languages.** Inoue et al. 2024 trained Voice Activity Projection
models on English, Mandarin and Japanese: **a monolingual VAP model trained on one language does
not make good predictions on other languages**, while a multilingual model matches monolingual
performance everywhere. This is published, independent, peer-reviewed support for what we measured
on Thai with Smart Turn v3.2 (1/4, confidently wrong) — and it means our result is an *instance of
a known phenomenon*, not an anomaly, which is a stronger position for a thesis, not a weaker one.

**Smart Turn v3 is independently weak, not just weak on Thai.** Helwani et al. 2026 report their
hierarchical end-of-turn model at **87.7% recall vs Smart Turn v3's 58.9%**, at **36 ms median
detection latency vs 800–1300 ms**, using 1.14 M parameters. Easy Turn (Li et al. 2025) likewise
outperforms Smart Turn V2 and released a **1,145-hour** turn-taking training set. So the baseline
we found failing is one that others also find failing — on English.

## 9. What Typhoon actually did instead, and the lever we already own

Typhoon's answer to Thai data scarcity was **not synthesis**. It was **GigaSpeech2**: 30,000 hours
crawled from unlabelled YouTube across Thai, Indonesian and Vietnamese, pseudo-labelled by
Whisper, force-aligned with MMS, then iteratively refined with noisy student training — reducing
Thai WER **25–40% versus Whisper large-v3 with 10% of the parameters**. That corpus is
**10,329.5 of Typhoon's 10,999 hours (94%)**. Real audio, machine-labelled. Their own three-way
Whisper-Large majority vote with human review only on numeral-containing transcripts is the same
philosophy applied to labelling.

⭐ **The one asset we have that this literature says is the top lever.** §4 item 1 says speaker
diversity matters more than anything else, and that many mediocre voices beat few good ones.
VoxCPM2 ships **natural-language voice design** — a new speaker identity from an English text
description, zero-shot, no reference audio (`LITERATURE.md` T16, and
`voxcpm/ref_designed_soft_warm_48k.wav` is one already made). That is an **unbounded speaker
generator**, free, already installed. Every synthetic corpus in this repo so far has been built on
essentially one voice. That is the single cheapest change available, and it is the change the
literature ranks first.

---

## 10. Standing conclusions — so this ground is not re-searched

- 🔴 **TTS audio is for vocabulary, not acoustics.** Every large published win is a rare-word,
  OOV, named-entity or numeric-normalisation win. Nobody has published a large win from
  synthesising *general* speech to teach a model how humans sound.
- 🔴 **Do not cite a TTS-augmentation result obtained on Whisper as applying to Typhoon.**
  Encoder-decoder benefits; transducer reportedly does not (§5).
- ⚠️ **The 8–12× exchange rate and the 1.46 WER-ratio floor are the two numbers to quote.** Both
  are from setups with far more data than we have; treat them as optimistic bounds.
- ⚠️ **This survey searched Consensus only.** GitHub, model cards and vendor blogs were not swept,
  at his instruction. An implementation detail could be missing.

## 11. Open questions this survey did NOT answer

1. Does the transducer-vs-encoder-decoder split (§5) hold for **modern** transducers, or was it a
   2021 artefact? Rossenbach's finding is five years old and predates flow-matching TTS.
2. Is there any published work on synthetic audio for **turn-taking in a tonal language**? Nothing
   surfaced. Thai turn-taking + synthetic data appears to be genuinely unexamined.
3. What is the real:synthetic exchange rate for a **classification** task like end-of-turn, as
   opposed to transcription? Every number in §2 is a WER/CER number.
4. Is the Easy Turn 1,145-hour trainset public, and does it contain any Thai?
