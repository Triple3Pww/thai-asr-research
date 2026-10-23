# Design — teaching Thai to SenseVoice: one model, one pass, text + emotion + events

**Date:** 2026-09-19
**Status:** proposed, nothing implemented
**Scope:** the ASR accuracy research line. Self-contained in this repo.
**Decision taken by him, 2026-09-19:** *"what if I want to use sensevoice as base"* — after the
alternative (a new head on Typhoon's frozen encoder) was presented and measured. **This spec builds
the SenseVoice route. It does not re-argue the choice**, but §2 records the price honestly so the
cost is visible in the record rather than in a chat log.

---

## 0. NEXT SESSION STARTS HERE — research Thai emotion detection

**Decided 2026-09-26 by him:** *"next session we will focus on research the thai emotion
detection."* Not more measurement on what we have — research.

### Where the emotion line actually stands

| | |
|---|---|
| our model, THEIR protocol, 3 s cap | **WA 77.18 ± 1.98 / UA 73.18 ± 2.52** |
| our model, uncapped | **WA 82.22 ± 1.38 / UA 79.82 ± 2.72** |
| THAI-SER's published baseline | 67.34 ± 3.05 / 62.61 ± 3.19 |
| the only public Thai SER model | **55.15 ± 4.74 / 45.84 ± 3.66** (`F25`) |
| checkpoint | `checkpoints/thaiser_head_3s_v1.pt` (51 KB, head + stats) |
| corpus | **full 29 shards**: 11,301 clips / 200 actors / 100 sessions / 12.37 h |

Findings to read first: **`F18`** (the fine-tune ladder), **`F21`** (separate head AND encoder),
**`F22`** (4 seconds), **`F23`** (their protocol, +9.84), **`F25`** (the public model), **`F26`**
(the checkpoint). `F19`/`F20` are the pseudo-labelling and intentless-corpus findings.

### 🔴 The three questions the research should answer

**1. Does any of this survive non-acted speech?** This is the one that decides whether the project
is real. Every number above is professional actors performing in a studio. `F20` measured this
model family labelling read Common Voice sentences as HAPPY at 0.997 confidence, with **no
correlation to the emotion of the words** (chi-square 5.93 on 9 df). Three concrete assets already
sit on disk or one step away:
- **2,516 Zoom clips** already cached in `ft_cache_full` — the corpus authors' OWN held-out
  out-of-domain set. Needs one `mic_zoom` cache pass. **Cheapest real answer available.**
- **40 clips in `D:/thai-ser-listen/`**, still unlistened — the only human ground truth in the
  entire natural-speech analysis.
- **EMOLA** (`A62`): 8,987 turns of Thai TV-drama dialogue. Conversational, in situation, not
  prompt-reading. Not on HuggingFace; request-only from NECTEC.

**2. What is the honest position against the literature?** `A60` records published Thai SER at
~80% (5-class) with **suspected speaker leakage** -- the corpus authors' own speaker-independent
baseline on the same data is 59.80. Unproven; both full texts paywalled. And `A61` is a 2024 Chula
thesis that already found our result's shape, so the honest framing is **"confirms and extends"**,
not "first". Settling this needs: the 5-class re-run (their `four_emotions.yaml` has the 4-class
config; the 5-class one is `baseline.yaml`), and a read of the two paywalled papers.

**3. Is the architecture conclusion still right at scale?** `F21` says separate head AND separate
encoder, measured on 1,256 clips. The full corpus is now 9x larger. 🔴 And `F23` found this model
loses **3.2 WA / 6.0 UA across microphone channels** — unpriced deployment risk for the loop,
where the mic is neither `con` nor `clip`.

### Cheap runs that are set up and unrun

1. `mic_zoom` cache pass → score the 2,516 OOD clips. Hours.
2. **Full fine-tune under their 8 folds.** `F23`'s 77.18 is the FROZEN rung; `F18` says the full
   fine-tune is worth several points. Our headline is currently our weakest configuration.
3. 5-class run, for `A60`.
4. Re-run `F21`'s encoder comparison on the full corpus.

### ⚠ Environment notes that will bite

- **D: has ~8 GB free.** THAI-SER took 12 GB and its cache 5.7. Anything large needs space cleared.
- Two emotion venvs: `D:/sensevoice-spike/.venv` (CPU) and `D:/sensevoice-gpu/.venv` (cu130, ~19x).
- **Two caches, deliberately:** `ft_cache` (1,256 clips) is what `F18`–`F22` were measured on and
  must not be overwritten; `ft_cache_full` (11,301 clips) is the full corpus. Set `E2V_CACHE`.
- 🔴 `funasr` dithers at inference; set `dither=0` for any SenseVoice arm.

### What NOT to do

- **Do not re-run the fine-tune ladder to confirm it.** Recorded with seeds, sweep and selection.
- **Do not quote "best Thai emotion model" unqualified** — `F25` scope is HuggingFace only, and
  everything is acted speech.
- **Do not pseudo-label a corpus without reading `F19` and `F20` first.**

---
## 0b. 🔴 THE ONE-PASS GOAL NOW CARRIES A PRICE (measured 2026-09-22)

This spec is built on *"one model, one pass, text + emotion + events"*. `F21` measured what that
costs the emotion output, and the number belongs next to the goal rather than buried in a findings
file. Everything below is on the same 410 actor-held-out clips.

| architecture | emotion accuracy |
|---|---|
| **shared CTC head** (this spec's slot 2, as SenseVoice ships it) | **0.4951** — SAD recall **0.019**, 40% `EMO_UNKNOWN` |
| separate head on a frozen ASR encoder — *the one-pass option* | **0.7000** |
| separate head on a frozen emotion encoder | 0.7805 |
| **separate, fully fine-tuned emotion encoder** — *two passes* | **0.8488** |

🔴 **The shared head is disqualified outright**: its own encoder carries SAD at 0.830 and the head
emits 0.019. 🔴 **The shared *encoder* is disqualified too** — a swept head (`linear/mlp128/mlp512`)
closed only 2 of the 10 points, paired McNemar **p = 0.0032**. 🔴🔴 **And the reason is structural:**
`F18` shows emotion needs a FULL fine-tune to reach 0.8488, which cannot be done to a shared encoder
without destroying its ASR. The thing that makes emotion work is the thing sharing forbids.

**So the price of one pass is ~15 points of emotion accuracy**, paid to save ~20 ms on the GPU and
164 M params of VRAM. That is a real trade and it may still be worth taking — but it is now a choice
made with a number, which is what this section exists to record.

⭐ **What survives untouched:** the reason SenseVoice was chosen as the **ASR** base (multilingual,
code-switching, `F17`) is independent of this and still stands. What is undermined is only the
"one head emits everything" half of the design.

⭐ **And the ASR encoder is not wasted.** It beats emotion2vec at SAD (0.792 vs 0.604) while losing
on NEUTRAL; fusing its already-computed pooled features into the emotion head lifts macro-recall
0.7532 → 0.8041. ⚠ Marginal (p = 0.0488 over three tried combinations) but nearly free.

⚠ **The door that is NOT closed:** everything above is **frozen** encoders. **Multi-task fine-tuning**
— one encoder trained on ASR and emotion jointly with a forgetting control (`S10`) — is unmeasured,
and it is the only route that could deliver a genuine single pass at good accuracy. If the one-pass
architecture matters, that is the experiment to run, not a frozen shared encoder.

**Operational constraint from `F22`:** emotion accuracy plateaus at **4 s** of audio and the encoder
runs at 50 fps. NEUTRAL is decidable at 0.5 s and ANGRY at 1 s, but HAPPY needs the full 4 s. Below
3 s the model is overconfident by 11-17 points, so any confidence gate needs a minimum-duration
guard. A loop that must answer in under a second can say *"something is up"*, not *which* emotion.

---

## 1. The target, in one paragraph

`FunAudioLLM/SenseVoiceSmall` emits four answers and a transcript from **one** encoder pass —
language, audio event, emotion, digits-or-words, then the text — because all five come out of the
same CTC head at fixed output positions. It does this for Chinese, Cantonese, English, Japanese and
Korean. **It does not do Thai at all**: zero Thai tokens in its 25,055-token vocabulary and no `th`
slot in its language list. Measured here 2026-09-19, it labels Thai speech as Cantonese and emits
confident garbage. The target is to add Thai as a sixth language **without** losing the four
paralinguistic outputs or the other five languages, and to measure what it costs on all three axes.

---

## 2. What this route costs, recorded once so it stays visible

| | Typhoon + new head (rejected) | **SenseVoice base (chosen)** |
|---|---|---|
| Thai ASR starting point | CER **9.92** TVSpeech, working today | **zero Thai**, starts from nothing |
| tokenizer surgery | none | required, plus a wider output layer |
| risk to the working `:8007` loop | none (encoder frozen) | none (separate model) but no reuse either |
| iteration speed | minutes on cached features | hours to days per run |
| what you gain | Thai + emotion + events | the same **plus 5 more languages** |
| realistic Thai CER | 9.92 unchanged | 🔴 **unknown and probably poor — see the data note below** |

🔴🔴 **THE DATA GAP, COUNTED 2026-09-19 AND WORSE THAN ASSUMED.** Typhoon reached CER 9.92 with
**11,000 hours**. The real Thai audio actually available to this project is:

| source | unique hours | distinct sentences | on disk? |
|---|---|---|---|
| **Common Voice 17 Thai, train** | **37.5 h** | **32,823** (one per clip) | transcripts yes, audio no |
| Common Voice 17 Thai, dev + test | 29.4 h | 22,084 | test tar only (321 MB) |
| LOTUSDIS train (5 mics collapsed) | ~17.6 h | 22,596 | ✅ 13 GB local |
| this repo's verified synthetic corpus | 10.57 h | 685 texts | ✅ `D:/thai-synth-corpus` |
| **realistic training pool** | **~65 h** | **~78,000** | |
| *CV17 Thai, ALL clips incl. unvalidated* | *423 h / 363,362 clips* | — | downloadable, CC0, ungated |

⭐ **Counted 2026-09-19 from the cached transcripts, not estimated.** This is far better than a first
pass suggested, and the reason is **text diversity**, which `H39` says is the axis that actually pays:
Common Voice Thai train is **32,823 clips carrying 32,823 distinct sentences** — every clip a
different sentence — where LOTUSDIS repeats 22,596 sentences across 5 mic copies each.

🔴 **It is still ~170x short of Typhoon's 11,000 h**, and SenseVoice starts from *zero* Thai rather
than from a working Thai model. Teaching Thai from scratch on ~65 h will not approach CER 9.92.
**A Thai CER anywhere near Typhoon's 9.92 is not a plausible outcome and must never be a success
criterion.** What this route buys is *one model doing six languages and four paralinguistic jobs*,
and the thesis claim has to live there (§7).

⭐ **The one unlock, and it should be decided before Phase 1 starts.** GigaSpeech2-th is **10,000 h**
of Thai -- essentially Typhoon's own corpus -- gated but free, and ~600 GB against 67 GB of free
disk. **It does not have to be stored.** HuggingFace `datasets` streams it, so the disk stops being
the blocker and only access approval and wall-clock remain. **Request access first**; if it is
granted, Phase 1 is a different and far more viable project than the §5.1 table describes.

---

## 3. Decisions taken (do not silently reverse these)

| # | Decision | Why |
|---|---|---|
| **S1** | **Extend the tokenizer with Typhoon's own 2,048-token Thai BPE**, never train a fresh one | It is CC-BY-4.0, proven on 11k h of Thai, and — decisively — it keeps this repo's label conventions identical to every other arm. A new BPE would make `F1`/`F2` re-verification necessary from scratch |
| **S2** | **Extend, never replace.** New vocab = 25,055 + 2,048 + 1 (`<\|th\|>`) | Exactly what Typhoon did to Nemotron: *"the multilingual tokenizer is extended (not replaced) … keeping the other languages recoverable"* |
| **S3** | **Emotion labels stay supervised in every training batch** | Emotion is output position 2 of the *same* `ctc_lo` as the text. Thai-only ASR training degrades it with nothing to stop it, and nothing errors |
| **S4** | **Freeze `encoders0` (1) + `encoders` (49) + `after_norm`; train `tp_encoders` (20) + `tp_norm` + `ctc_lo` + the new embedding rows.** ⚠️ `after_norm` closes stack 1 and must be frozen WITH it — it is only 1,024 params, but leaving it trainable lets the "frozen" encoder's output scale drift and the do-no-harm gate quietly stops holding. That is exactly the model's own module boundary — 50 frozen, 20 trainable, no arbitrary cut | Limits forgetting, fits 16 GB comfortably, makes a run hours instead of days. The split is where SenseVoice already puts its `after_norm`, so it is the architecture's seam rather than a guess. Unfreezing more is an ablation, not the default |
| **S5** | **The five original languages are MEASURED before and after, every run** | You cannot preserve what you cannot measure. A "we kept the other languages" claim without a before/after table is not a claim |
| **S6** | **Thai labels decomposed, SARA AM folded on both sides of every CER, unspaced number WORDS** | `F1` (26.6% of mean CER) and `F2`. One definition: `adapter/_thai.py`. Non-negotiable |
| **S7** | **`thai-contextasr-bench` is NEVER training data** | CC-BY-NC-4.0. Needs a hard guard in the training script, not a comment |
| **S8** | **THAI-SER split is ACTOR-disjoint, never clip-disjoint** | 200 actors repeating 3 scripted sentences; a clip split leaks the speaker and every number becomes meaningless |
| **S9** | **The research claim is the ABSTENTION behaviour, not the CER** | §7. The CER will lose to Typhoon; the abstention result is measured, ours, and unclaimed |
| **S10** | 🔄 **`tp_encoders` trains under an EWC penalty — CONDITIONAL on the FLEURS slice landing in Phase 0. Fallback is L2-SP.** | EWC weights each parameter's spring by how much the *old* languages depend on it, which requires running their audio through the model. **That data is not on disk** (Phase 0). Without it the honest fallback is L2-SP — the same springs at uniform stiffness, no importance estimate, no old-language data — which is weaker but far better than nothing. `H19` (93 cites, on a transducer, −57% on target with no degradation elsewhere); `H25` uses parameter regularisation as one of four components |
| **S11** | 🔄 **HYPOTHESIS, gated on `P0.3` — not a decision.** Adapters in the frozen 50, *if* the probe shows the frozen features cannot carry Thai | A hard freeze protects the old languages by refusing to adapt *at all*, and it blocks exactly the early-to-middle layers where new phonology has to land (`H5`, `H20`). Zero-init bottleneck adapters (512→32→512, 33,792 params each, `H3`) keep the original weights untouched **and** let the low layers move; `H14` reports 10–17% WER reduction with −1% out of domain. ⚠️ Measured here 2026-09-11: `set_enabled_adapters` is module-level, so Thai and non-Thai **cannot share a batch**. 🔴 **And they are NOT free at inference** — 50 layers each gain two extra matrix multiplies, permanently; an earlier draft drew them as a training-only side node, which was wrong |
| **S12** | **Decode with Typhoon's Thai 4-gram in shallow fusion** | 🔴 CTC is conditionally independent per frame — it carries **no internal language model**, which is the whole reason it transcribes worse than an RNN-T (13.46 vs 9.92). `A35`, 172 cites. ⭐ The file is already on disk (`ngram_4gram_bpe2048.nemo`) and is built on the *same* 2,048 Thai BPE that `S1` grafts in, so it needs no training. ⚠️ The repo's negative result for fusion (`D2`) was measured on an **RNN-T that already has an internal LM** and does not transfer. 🔴 **Three things the 2026-09-19 literature sweep added** (`A53`–`A55`): (a) the `.nemo` is a NeMo artefact and SenseVoice runs under funasr — the **`.arpa` is the portable route**, via a CTC beam decoder such as FlexCTC (`A55`), so "no training" is true but "drop it in" is not; (b) this LM is **subword-level**, and `A54` measures subword n-grams as the *weaker* kind precisely for languages **without word boundaries** — Thai is one, so ⑦ has a ceiling worth measuring; (c) its text is Typhoon's ASR transcripts plus LLM-generated carrier sentences, and **whether any of `thai-contextasr-bench`'s 682 sentences are in it is unchecked** — see `R7` |
| **S13** | **Slot 2 is trained by distilling a Thai-tuned emotion2vec, and carries a TUNABLE abstention threshold** | `F8b` measured the ranking: re-ranking one model's own candidates **−1**, biasing its scores **+8**, a second model's opinion of the audio **+99**. One model doing everything is the weakest row. Distillation buys the second opinion at training time and costs nothing at inference. The threshold turns §7 from one observed point into a **precision–coverage curve** (`A49`, Qiu 2021) |

---

## 4. What this rests on — measured here, not assumed

| fact | number | source |
|---|---|---|
| SenseVoice zero-shot Thai emotion, 4-way | **0.4761**, refusing on **43.6%** | `results/probe_emotion_thai.json` |
| …but **84.3% precise on the 709 clips where it commits** | higher than the teacher's 78.6% | same |
| SenseVoice's own frozen encoder, linear probe, Thai emotion | **0.6829** | `results/probe_encoder_emotion.json` |
| 🔴 its encoder reads SAD at **0.830**, its shipped head emits SAD at **0.060** | the head is the bottleneck | both files |
| emotion2vec_plus_large zero-shot Thai, 4-way | **0.7381**, refusing on 6.1% | `results/probe_emotion_thai.json` |
| mel floor (no encoder at all) | **0.5756** | `results/probe_encoder_emotion.json` |
| human ceiling, annotators vs assigned, same subset | **0.9286** | same |
| THAI-SER published human accuracy (5-way, whole corpus) | 0.772 | arXiv 2507.09618 |
| SenseVoice architecture, read not guessed | 70 SAN-M layers, d_model 512, LFR 7/6 → 60 ms frames, CTC-only, 234 M | `config.yaml` + `model.py` |
| Typhoon, the number to be honest about | CER 6.89 gigaspeech2 / **9.92 TVSpeech** | `typhoon-asr-qwen-1.7b-ctx` card |

⭐ **The `0.060 → 0.830` gap is the whole reason this design is viable.** The ears already hear Thai
emotion. Only the mouth is broken, and the mouth is what a fine-tune rewrites.

---

## 5. The plan

### Phase 0 — surgery and a do-no-harm baseline (≈3 days)

**Build:** `adapter/sv_extend.py`.

1. Merge Typhoon's `tokenizer.model` (2,048 Thai BPE) into
   `chn_jpn_yue_eng_ko_spectok.bpe.model`. Add `<\|th\|>`. New vocab **27,104**.
2. Grow `ctc_lo` from `[25055, 512]` to `[27104, 512]`. 🔴 **Weights zero AND bias ≈ −30 on the
   new rows.** Zero weights alone is NOT enough and an earlier draft of this spec was wrong about
   it: `ctc_lo` is `torch.nn.Linear`, so it *has* a bias, and zero-weight + zero-bias gives a new
   Thai token a logit of exactly **0** — not −∞. Competing tokens routinely score negative, so a
   brand-new Thai row at 0 can outrank a real Chinese one at −2 and win. A large negative bias
   makes the new rows unreachable until training moves them, which is what Gate 0 actually needs.
3. Add a `th` entry to `lid_dict` and one row to `nn.Embedding(16, 560)` → `(17, 560)`.

🔴 **GATE 0, and it is a hard one.** After surgery and *before any training*, the model's output on
the five original languages must be **byte-identical** to the shipped model on a fixed probe set.
If it is not, the surgery is wrong and nothing downstream means anything. The negative-bias init
in step 2 is what makes this achievable; if it still fails, stop.

**Also built here:** `adapter/sv_eval_multi.py` — the before/after harness for the five original
languages (S5). Without it, Phase 1 cannot report its own damage.

### ⭐ P0.3 — the probe that decides whether ② (adapters) happens at all

**His question, 2026-09-19:** *"if we will finetune the tp_encoder, so what the benefit of adapter"*.
It is the right question and `S11` could not answer it, because adapters add only **1.69 M** trainable
parameters against the **~63 M** already trainable in `tp_encoders` — **2.7%**. They are not capacity.
Their entire claim is *location*: that layers 1–50, trained on zh/yue/en/ja/ko and never on Thai, may
have **collapsed Thai distinctions that no amount of work in layers 51–70 can recover**.

That is a claim about the frozen features, so measure the frozen features. **Same harness as
`adapter/probe_encoder_emotion.py`**, different target:

> Freeze the encoder, take its per-frame output on Thai audio, and fit a **linear probe to predict
> the Thai character at each frame**. Run it on SenseVoice's frozen encoder and on Typhoon's, which
> demonstrably does carry Thai. Actor-disjoint, identical pooling, the encoder is the only variable.

| result | decision |
|---|---|
| SenseVoice ≈ Typhoon | the low layers already carry Thai — 🟢 **SKIP ②.** Train the top 20 only; no adapters, no inference cost, no batching constraint |
| SenseVoice ≪ Typhoon | the distinctions really are collapsed — 🔴 **② is justified** and its costs are worth paying |

**Cost:** half a day, no training, no GPU, and the extraction code already exists.

🔴 **Until this runs, `S11` is a HYPOTHESIS, not a decision.** Nothing in Phase 1 should assume
adapters.

**PREREQUISITE — pull FLEURS. Counted 2026-09-19; this replaces an earlier hand-wave about
"five example mp3s", which was a smoke test, not an evaluation.**

`google/fleurs` is **CC-BY-4.0 and ungated**, and it covers every language this design touches:

| config | language | size | rows |
|---|---|---|---|
| `cmn_hans_cn` | Chinese | 3.20 GB | 4,600 |
| `yue_hant_hk` | Cantonese | 2.54 GB | 3,120 |
| `en_us` | English | 2.36 GB | 3,643 |
| `ja_jp` | Japanese | 2.44 GB | 3,208 |
| `ko_kr` | Korean | 2.28 GB | 2,915 |
| **five old languages** | | **12.82 GB** | 17,486 |
| `th_th` | ⭐ Thai | 3.06 GB | 4,062 |

🔴 **It is NOT on disk.** Only ~389 KB of FLEURS metadata is cached. Free space at the time of
writing: C 24.1 GB · D 24.6 GB · E 12.0 GB — so the full 12.82 GB fits, but it is real space on a
box already carrying THAI-SER and LOTUSDIS.

⭐ **Take only what is needed.** The forgetting harness needs the **test** splits, not the configs;
EWC's Fisher estimate (`S10`) needs a **few hundred utterances per language**, not thousands. Both
jobs together are roughly **2 GB**, and `datasets` streaming can serve them with **nothing written to
disk at all**. Pull the slice in Phase 0; do not mirror the corpus.

⭐ **`th_th` is a bonus worth taking: a SECOND Thai test set**, read speech, human-transcribed, from a
different source than Common Voice. This repo currently has no way to check whether a Thai result
generalises past the corpus it was tuned on.

⚠️ **Before any Thai number from FLEURS is quoted, check its conventions against `S6`.** If it writes
digits where Typhoon writes number words, its CER needs the same treatment as everything else here
(`F2`, `F17`, `adapter/ortho_rescore.py`). Unverified.

**If a language still cannot be measured, say so in the record and mark it unmeasured — do not
quietly drop it.**

### Phase 1 — Thai ASR, the expensive part (≈2–3 weeks)

**Data on disk or obtainable (§5.1):**

| source | size | licence | note |
|---|---|---|---|
| 🔴 LOTUSDIS | 13 GB / 160,915 utts / 114.2 h — but **only ~17.6 h of UNIQUE audio** | CC-BY-SA-4.0 | **COUNTED 2026-09-19, and it is far smaller than it looks.** Every utterance is recorded on **5 microphones** (`con123`, `lav123`, `bt3m`, `bt10m`, `jbl`, exactly 32,183 each). Train is 120,245 rows / 88.1 h but **24,026 unique recordings and 22,596 distinct sentences** ⇒ ~17.6 h of distinct speech. ⭐ The 5 channels are genuine *channel* augmentation (and the untested row-3 axis of the old spec), but they add **zero new text** — and `H39` says text diversity is what pays |
| Common Voice 17 Thai | small | CC0 | the repo already uses 150 clips of it for `arm1_baseline` |
| this repo's synthetic corpus | 10.57 h, 4,688 verified keeps | ours | `H28` says pretrain on synthetic **first**, then real |
| 🔴 GigaSpeech2-th | 10,000 h | gated **and ~600 GB** | **out of reach**: 67 GB free across all drives |

**Recipe:** synthetic first, then real (`H28`), S4's freeze, emotion supervised throughout (S3).

**Gates, in order:**
- **G1.1** Thai CER on the repo's own 150 Common Voice clips, SARA AM folded, beats **1.0** (i.e.
  it produces Thai at all). Trivial, but it is the "did the tokenizer surgery work" check.
- **G1.2** The five original languages degrade by **less than 10% relative** on Phase 0's harness.
  🔴 **If they degrade more, the multilingual claim — the entire reason for choosing this base — is
  gone, and the design must be revisited before more time is spent.**
- **G1.3** Emotion on THAI-SER (actor-disjoint) has not fallen below the **0.4761** zero-shot number.

### Phase 2 — emotion and events, his three-step pipeline (≈1–2 weeks)

This is his original proposal, and it is the right shape now that the head is known to be the
bottleneck.

1. **Fine-tune `emotion2vec_plus_large` on THAI-SER**, actor-disjoint (S8). Zero-shot is 0.7381;
   in-corpus fine-tuning should beat it. **This model is the teacher and the label source.**
2. **Pseudo-label** the Phase 1 Thai audio with it — exactly what Alibaba did at 30 M entries
   (`H40`), and the same instinct as this repo's readback panel (`H36`).
3. **Train SenseVoice's positions 1–2** (event, emotion) on those labels, jointly with the Thai text.

🔴 **The known defect, stated up front.** THAI-SER is **acted** — professional actors, three fixed
sentences in the scripted half. LOTUSDIS is spontaneous. A teacher trained on the first will be
confidently wrong on the second and nothing in the pipeline can detect it. This is `F5`'s shape and
the same reasoning behind the spec's *"Layer 6 must not use Typhoon"* rule. **Mitigation:** hold out
the improvised half of THAI-SER as a harder validation set (both models already score lower on it —
SenseVoice 0.412 vs 0.554, emotion2vec 0.699 vs 0.786), and report the pseudo-label agreement rate
against it.

⚠️ There is **no second Thai emotion corpus** for cross-corpus validation. `Vorrapard/thai-emotion`
is `n<1K` and a single CSV. Until one exists, every emotion number here is in-corpus.

### Phase 3 — the measurement that is actually ours (≈1 week) — **the contribution**

See §7.

---

## 5b. The seven changes, ordered by return on effort

Not by position in the stack. The ledger rows are in `LITERATURE.md`; the ordering is ours.

| # | change | decision | what it fixes | cost |
|---|---|---|---|---|
| **⑦** | **Thai 4-gram at decode** | `S12` | CTC has no internal language model, and Thai training data is ~65 h, not 11,000 | 🟢 zero training; file on disk. ⚠️ needs a CTC beam decoder (`.arpa`, not `.nemo`) and the `R7` leak check first |
| **②** | adapters, not a bare freeze | `S11` | freezing layers 1–50 blocks where new phonology must land — **unproven** | 🔴 **gated on `P0.3`**; 1.69 M params (2.7% of trainable) + permanent inference cost |
| **⑤** | distil emotion2vec into slot 2 | `S13` | one model doing everything is `F8b`'s weakest row (−1 / +8 / **+99**) | training only; free at inference |
| **④** | reweight the slot 0–3 loss | `S3` | emotion is one token against a whole transcript; the text gradient drowns it (`H17`) | trivial |
| **⑥** | tunable abstention on slot 2 | `S13` | §7's claim is currently one uncontrolled point, not a curve | small |
| **③** | EWC on the trained stack | `S10` | nothing currently *prevents* forgetting — `S5` only measures it afterwards | moderate; 🔴 **needs the Phase 0 FLEURS slice** |
| **①** | synthetic first, then spontaneous | — | Common Voice is read speech; the loop hears conversation, measured **2.2× harder** (0.0518 vs 0.1125). `H28` measured the reverse order as worse; `H39` says text diversity is the axis that pays | ordering only |

⭐ **⑦ first.** It costs nothing, the file is already on this machine, and CTC is precisely the
architecture where a language model should pay. ⚠️ And it is the one arm where this repo holds a
*negative* result — `D2`, measured on an RNN-T that already carries an internal LM — which makes
running it informative rather than routine.

🔴 **③ is gated on Phase 0.** If the FLEURS slice is not pulled, EWC degrades to L2-SP (`S10`) and
the forgetting claim in `S5` cannot be made at all.

**Diagram:** the before/after node graphs, the adapter detail and this table are drawn at
`https://claude.ai/artifact/VfvHMRKGFZ7JnS7JTXHEBd`.

---

## 6. Protocol rules that must hold in every run

Each cost a session to learn. Violating one invalidates the run.

1. **Fold SARA AM on BOTH sides of every Thai CER**; write labels **decomposed**. `unicodedata`
   does nothing. Worth 26.6% of mean CER. One definition: `adapter/_thai.py`.
2. **Unspaced Thai number WORDS, never digits.** Structural, not stylistic — Typhoon's tokenizer
   has zero digit tokens.
3. **Bracket everything compared back-to-back in one window**; discard the first sweep after any
   server start. This box drifts 25–45% across an evening.
4. **Run the control even when nothing changed**, and pin anything that moves a number
   (`--align dp`).
5. **Actor-disjoint splits on THAI-SER, always** (S8).
6. **Capture stderr first, conclude second.** A silent exit-0 on this box is as often the page file
   as it is VRAM.
7. **`thai-contextasr-bench` never enters training** (S7), enforced by an assertion.

---

## 7. What may be claimed — and it is not the CER

🔴 **The Thai CER will lose to Typhoon** (§2). A thesis resting on it fails on contact.

⭐⭐ **What is ours, measured today, and unclaimed anywhere:**

> SenseVoice abstains on **43.6%** of Thai emotion clips while being **84.3% precise** on the rest —
> *better* precision than the specialist teacher. And the refusal is not deafness: a linear probe on
> its own frozen encoder reads SAD at **0.830** where its shipped head emits **0.060**.

So the research question Phase 3 answers is:

> **When a multilingual speech-understanding model is extended to an unseen language, does its
> learned abstention survive — and is coverage bought by trading away precision?**

Nobody has asked it, because nobody has run a multilingual paralinguistic head on an unseen language
and scored **refusal separately from error**. The measurements needed are already defined:
refusal rate, precision-when-committing, and per-class recall, before and after Phase 2.

**Secondary, publishable regardless of the above:** the first Thai numbers for `emotion2vec+`'s
unnamed-language universality claim (`A51`), and a measured before/after forgetting table for adding
a sixth language to a five-language model (`S5`) — which Typhoon's Nemotron card asserts and does not
quantify.

---

## 8. Risks and kill criteria

| # | risk | how it shows | what we do |
|---|---|---|---|
| **R1** | 🔴 **Catastrophic forgetting of the five languages** — the whole reason for this base | Phase 0's harness shows >10% relative degradation | **Stop and reconsider the base.** Mitigations, now decided rather than listed: adapters in the frozen stack (`S11`), EWC on the trained stack if the FLEURS slice lands, L2-SP if it does not (`S10`). 🔴 **All three depend on Phase 0 actually pulling old-language audio** — without it, forgetting can be neither prevented by EWC nor measured at all |
| **R2** | 🔴🔴 **THE DOMINANT RISK. Only ~28 h of unique Thai exists locally** (§5.1), against Typhoon's 11,000 h | Thai CER never becomes usable at all | **STREAM GigaSpeech2-th** rather than storing it -- 10,000 h, gated but free, and `datasets` streaming removes the 600 GB disk problem entirely. Request access before Phase 1. If access is refused, accept a poor CER, say so plainly, and lean entirely on §7. **Do NOT quietly drop the CER gate** |
| **R3** | **Emotion degrades as Thai is learned** — they share `ctc_lo` | G1.3 fails | S3 is the mitigation; if it is insufficient, weight the emotion loss or freeze position 1–2 rows of `ctc_lo` during text training |
| **R4** | **Pseudo-label domain shift**, acted → spontaneous | Phase 2's improvised-half agreement is poor | Report it; fall back to THAI-SER-only supervision, which is smaller but honest |
| **R5** | **The tokenizer surgery silently breaks the five languages** | Gate 0 is not byte-identical | Zero-init is designed to prevent this. If it still fails, the merge is wrong — fix before proceeding |
| **R6** | Box contention with the live loop | both want the GPU | Schedule; never break rule 3 to save time |
| **R7** | 🔴🔴 **The 4-gram may have seen the benchmark.** Its card says "no evaluation data" — but that means *Typhoon's* eval sets. `thai-contextasr-bench` is a Wayu/Paxa artefact (`A43`), not theirs, and every number in this repo is scored on it | reference sentences from the 682 real utts appear as high-order n-grams in `ngram_4gram_bpe2048.arpa` | **CHECK BEFORE ⑦ IS RUN, NOT AFTER.** The ARPA is plain text on disk and the references are in `results/*_hyps.jsonl`; grep a sample. If they are present, ⑦'s numbers on this benchmark are a leak and cannot be quoted |

---

## 9. Open questions, in order

1. ~~How many hours is LOTUSDIS actually?~~ **ANSWERED 2026-09-19: 114.2 h of rows, ~17.6 h of
   UNIQUE audio.** 160,915 rows = 32,183 utterances x **5 microphones**; train alone is 120,245 rows /
   88.1 h but 24,026 unique recordings and 22,596 distinct sentences. 🔴 **This is the number that
   reshapes the whole spec** -- see the revised §2 and R2. The 5 channels remain valuable as channel
   augmentation; they are worth nothing as text diversity.
2. **Can GigaSpeech2-th be streamed rather than stored, and will access be granted?** This is now
   the single question that decides whether Phase 1 is viable. **Ask before building anything.**
3. **Do FLEURS' Thai transcripts follow this repo's conventions (`S6`)?** If `th_th` writes digits
   where Typhoon writes number words, every CER from it needs the `F17` treatment before it can be
   compared to anything here. Cheap to check from the transcripts alone, no audio needed.
4. **Can SenseVoice transcribe a full English sentence today?** Its card says yes; nothing here has
   tested it. It is the Phase 0 baseline for `en` and it is untested.
5. **Does Typhoon's Thai BPE merge cleanly into a SentencePiece model with 25,055 existing tokens?**
   Assumed in S1, unverified.
6. **What is the real forgetting rate at each freeze depth?** S4 picks 50/70 by reasoning, not
   measurement. One ablation answers it.
7. ~~Does SenseVoice's encoder carry Thai emotion?~~ **ANSWERED 2026-09-19: yes, 0.6829 linear
   probe, and SAD at 0.830 where the head emits 0.060.**

---

## 10. References

`results/probe_emotion_thai.json` · `results/probe_encoder_emotion.json` ·
`FunAudioLLM/SenseVoiceSmall` (`config.yaml`, `model.py`, FunASR `postprocess_utils.py`) ·
FunAudioLLM report arXiv 2407.04051v3 §3.1, §4.7 · THAI-SER arXiv 2507.09618 ·
`typhoon-ai/typhoon-asr-streaming-nemotron-0.6b` (tokenizer-extension precedent) ·
`LITERATURE.md` `A50`–`A52`, `H28`, `H36`, `H39`–`H40`, `H19` · `RESEARCH-FINDINGS.md` `F1`, `F2`,
`F5` · `HISTORY.md` 2026-09-19 (later) and (night).
