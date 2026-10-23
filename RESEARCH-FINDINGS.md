# RESEARCH-FINDINGS — everything we found

**Written 2026-09-12.** Companion: `RESEARCH-PROGRESS.md`. Per-claim detail lives in `LITERATURE.md`;
the append-only history in `HISTORY.md`. This file is the synthesis — the things that changed what
we would do.

---

# PART 1 — Our own measurements

## ⭐⭐ F1. Thai SARA AM silently inflates every Thai CER number

The vowel in `บำรุง` has **two encodings that render identically**:

```
U+0E33           composed     ำ
U+0E4D U+0E32    decomposed   ํ + า    (NIKHAHIT + SARA AA)
```

🔴 **Unicode gives `U+0E33` no canonical decomposition, so `unicodedata.normalize` does nothing.**
Measured: NFC changed **0 of 150** clips.

Meanwhile:

| | decomposed | composed |
|---|---|---|
| Typhoon's hypotheses | **57** | 0 |
| Common Voice references | 0 | **57** |

**47 of 150 clips (31%) carried the penalty.** Folding it:

| | mean CER | median | perfect |
|---|---|---|---|
| unfolded | 0.0706 | 0.0405 | 57/150 |
| **folded** | **0.0518** | **0.0000** | **85/150** |

**26.6% of the mean and 100% of the median was an encoding difference, not a recognition error.**

🔴 **Why it is a trap, not an inaccuracy.** Our generated text banks used the composed form
exclusively (987 occurrences). Train a decoder on those labels, score with an unfolded metric, and
the arm posts a **~26% general-CER "improvement" that is purely a spelling change** — and the
adoption gate accepts it for a reason with nothing to do with the intervention.

**Consequence beyond this repo: cross-paper Thai ASR CER comparison is not sound.** Every Thai ASR
leaderboard inherits this. It appears in **no paper** across four literature sweeps.

**Fix:** fold on *both* sides when scoring; write labels in the decomposed form the model emits.
One definition: `adapter/_thai.py`.

## F2. Typhoon writes number WORDS, unspaced — never digits

Measured over **53 real Thai turns** already in `live/_out/turns-*.jsonl` (no server start needed —
the live panel's `asked` field *is* :8007's output):

| | Typhoon | our banks as generated |
|---|---|---|
| ASCII digits | **0 / 53** | **279 / 279** labels |
| Thai number words | 7 transcripts | — |
| any space at all | **1 / 53** (mean 0.02) | mean **4.64** per label |

The decisive row is a phone number read aloud — exactly where a digit-writing ASR would emit digits:
`ชื่อพอร์ตศูนย์แปดสามศูนย์หนึ่งแปดเก้าห้าสามเจ็ดครับ`. It wrote words. Dates too: `วันที่ยี่สิบเอ็ด`.

**Cost:** all 279 phase-2 labels were in the wrong convention, and the **69 postcode rows lost their
purpose entirely** — with the incumbent already writing number words there is no digit-normalisation
gap to teach, and a 5-digit entity can never appear in a correct label. 35 salvaged (their sentence
also names a district), 34 dropped.

⚠️ **So the banks make no claim about numeric normalisation.** What survives is out-of-vocabulary
*vocabulary*, which is the side of the line synthetic data actually wins on.

**F1 and F2 together are the general lesson:** *three* invisible text conventions (digits, spacing,
SARA AM) each would have produced a confident wrong answer — two by making a good method look bad,
one by making a useless method look good.

## F3. The rare-word gap is field-wide, not our model's fault

On the identical 682 real utterances, with the benchmark's own scorer:

- **B-CER is 2.01–3.04× U-CER for every one of 7 systems** — RNNT, CTC and encoder-decoder alike.
  Ours 2.48×, the best 2.01×.
- **The field's best model still misses 57% of named entities** unbiased (recall 0.435).
- Entity characters were only **3,014 of 69,432** (~4%) of the text, so total failure on entities
  barely moves overall CER. **The damage is real and statistically invisible at once** — which is
  exactly why the B/U split is the right instrument.

🔄 **Re-tested 2026-09-17 under orthographic normalisation (`F17`) — the finding holds, the table
it sits on does not.** The B/U ratio stays 2.07-3.08x across all seven, so "universally 2-3x
harder" is not a spelling artefact. But the CER ordering in that same table scrambles (Kendall
0.429, 6 of 21 pairs) once digit-vs-word convention is neutralised, and **our rank goes 7th to
5th**. Quote F3 for the B/U ratio. Do not quote its CER column as a ranking of accuracy.

## F4. 🔴 Synthetic speech did NOT train a Thai endpointer — and the control is still missing

Four fine-tunes on two generations of VoxCPM data, on 400 real Thai pauses: **0.425 / 0.435 / 0.435 /
0.450 EER against 0.4175 untrained**; pause duration alone 0.419. Every arm at or below baseline.

⚠️ **Nobody trained the same model on *real* pauses.** So "synthetic cannot train it" and "this model
cannot learn it" are still not separable. Cheap to fix, and it decides whether a thesis claim is
legitimate.

## F5. Including synthetic evaluation audio makes every model look ~2× better

Same model, same condition, only the corpus differs:

| | theirs (all 2,062 utts) | ours (682 real) |
|---|---|---|
| CER | 0.0483 | **0.0924** |
| entity recall | 0.459 | 0.408 |

**The benchmark's 1,380 TTS utterances are substantially easier than its 682 real ones.** Quoting
their published number beside ours would have shown a recall gap of 0.13 where the real gap is 0.08.
⭐ And it is quiet independent support for this repo's own thesis: **synthetic speech is measurably
easier than real speech, even as evaluation material.**

## F6. Two bugs found by running code, not reading it

- **`--per-text 8` gave one text eight clips in the *same voice*** — the knob multiplied random seeds
  instead of speakers, which would have defeated both the strongest lever in this literature and the
  ratio sweep. The first fix left distinctness to chance (8 distinct for only 365 of 440 texts); a
  coprime stride made it **guaranteed**.
- **Matched pairs were split across voices**, making voice a confound in the one comparison the
  `pair_id` exists to isolate.

## F7. Disk and memory mechanics on this box

- The WSL vhdx is **already NTFS-sparse** and WSL's ext4 **discards**, so deleting N GB inside returns
  ~N GB to Windows — **no compaction, no admin**. But it **flushes on `wsl --terminate`**, not
  reliably on delete. Measure after terminating or you will conclude the deletion failed.
- 🔴 **Do not use `wsl --manage --set-sparse true --allow-unsafe`.** WSL disables it citing data
  corruption, and it is unnecessary — the disk is already sparse.
- Measure allocated size with `GetCompressedFileSizeW`, **not** `os.path.getsize` — the logical size
  never moves.
- The page file trap: **commit limit = RAM + page file**, it is **auto-managed and shrinks to match
  free disk**, and VoxCPM reserves **~22.7 GB of commit** on load. Freeing app memory gives it back.

---

# PART 2 — What the literature says (4 Consensus sweeps, 40 new rows)

## ⭐⭐ F8. We are decoding greedy with no language model, and that is 14–30% relative WER

`typhoon-asr-realtime`'s config is `strategy: greedy_batch`; `asr/server.py` calls `transcribe()`
with defaults. **And NVIDIA already shipped the fix inside the NeMo we have installed** — all four
verified present:

| capability | published effect | in NeMo 3.0.0 |
|---|---|---|
| ALSD++ / AES++ fast batched beam (`A30`) | **14–30% rel. WER** vs greedy, system slowdown only **10–20%** | ✅ `malsd_batch`, `maes_batch` |
| **NGPU-LM** — n-gram *inside greedy* (`A31`) | **>50% of the beam gap for <7% overhead** | ✅ `NGramGPULanguageModel.from_arpa()` |
| n-gram shallow fusion (`A35`, 172 cites) | best first-pass method | ✅ `ngram_lm_model`/`ngram_lm_alpha` |
| unigram rare-word reward (`A32`) | rare words **+3.7%**, general flat, additive with rescoring | ✅ fusion hooks |

**This is the largest unexploited accuracy lever in the repo and it needs no training.**

### ✅ F8 was tested 2026-09-12 (late). Two of the four rows survived; the headline one did not.

| capability | published | **measured here** |
|---|---|---|
| batched beam (`A30`) | 14–30% rel. WER, slowdown 10–20% | ✅ **−8.9% CER**, RTF 0.012 → 0.016. Beam 2 = beam 8 |
| **NGPU-LM in greedy** (`A31`) | >50% of the beam gap at <7% | ❌ **−0.3% (noise) at 2.6× RTF.** Did not reproduce |
| shallow fusion (`A35`) | best first-pass method | ⚠️ **best CER/U-CER of any arm, RTF 0.136, and no effect on entities** |
| unigram reward (`A32`) | rare words +3.7%, general flat | ▶ not yet run — now the next arm |

🔴 **The `A31` failure has one honest excuse and one that does not hold.** The latency half: NeMo
warns *"Triton is disabled. Version without Triton is not compatible with Cuda graphs"*, Triton has
no Windows wheel, and a control (beam with graphs forced off, no LM) shows **3.6× of the slowdown is
the missing graphs, 2.3× is the LM itself**. The accuracy half has no excuse — it simply did not
close the gap on this model and corpus.

## 🔴🔴 F8b. No decode-time arm moved entity recall. 144 → 143 → 143.

✅ **SOLVED 2026-09-24 — see `F24`.** Everything below stands as written and was true of the five arms it describes. What it could not know is that the answer was not a better decode-time trick on `typhoon-asr-realtime` but **a different model Typhoon had already shipped**: `typhoon-asr-qwen-0.6b-ctx`, prompt-based contextual biasing, which takes the same 439 gold mentions from **144 to 207**. 🔴 The lesson is the ordering, not the number — this repo cited that model's card for its CER table and never ran the model.

Across greedy, beam, and beam+4-gram on the same 682 held-out utterances, entity recall did not
move, and the LM arm — which posted the **best U-CER of the session** (0.10146) — left **B-CER
*worse* than beam alone** (0.25705 vs 0.25574).

**The reading:** re-ranking a transducer's own candidates improves the text it was already close to
getting right, and cannot recover a name whose subwords the acoustic model never emitted. Entity
recall is bounded upstream of the decoder. **This is the measurement that promotes D4 (the boosting
tree, which injects the phrases) and D6 (the splice, which adds a second acoustic opinion) over
anything else on the decode side** — and it was an assumption until today.

✅ **D4 confirmed the reading and bounded it.** Injecting the phrases *does* move recall —
.328 → .362 with a perfect list, the first arm to move it at all — but by **+8 mentions over a
same-size distractor list**, roughly 1.1×. So re-weighting the model's own candidates helps and has
a low ceiling.

⭐⭐ **D6 closed it.** A *second acoustic opinion* — Typhoon's CTC head spotting entities on the
shared frozen encoder, spliced into our RNN-T transcript — moves recall **.328 → .554, +99 of 439
gold mentions**, with U-CER, CER and B-CER all falling, at RTF 0.028 → 0.032. The progression is the
finding:

| mechanism | Δ recall |
|---|---|
| re-rank one model's own candidates (D1/D2/D3) | **−1 mention** |
| bias one model's own scores (D4) | +8 |
| **a second model's opinion of the audio (D6)** | **+99** |

**Entity recall is bounded upstream of the decoder — and the way past that bound is a second model,
not a better search.** Measured, not argued.

## 🔴 F8c. Two harness defects were under the numbers we had already published

**(a) `:8007` has always run on the GPU.** `ASRModel.from_pretrained()` without `map_location`
lands on `cuda:0`; the server's docstring said CPU for a month and three documents repeated it.
Forcing `cuda` reproduces the recorded arm-1 baseline **byte-identically** (150/150 hypotheses);
forcing `cpu` changes exactly one, for the better. Every "RTF on CPU" figure was a GPU figure,
including *"~115M at RTF 0.007 on CPU against ~1.5B Whispers on GPU"* — that was GPU vs GPU.
⭐ **The silver lining is a real lever:** measured on purpose, the CPU costs RTF 0.027 vs 0.010
(~96 ms on a 6 s turn) and **returns ~0.95 GB of VRAM**, against the 2.9 GB the streaming ASR needs.

**(b) The benchmark's B-CER/U-CER depend on an optional package.** `thai-contextasr-bench` calls
`kaldialign` a *faster* path; it is also a *different* one. Byte-identical hypotheses, deterministic
scorer, three repeats each: **B-CER 0.26178 (DP) vs 0.25050 (kaldialign)** — 4.3% relative, on the
metric the whole plan is gated on. CER and recall do not move, because they need no span attribution.
Our ASR venv has kaldialign; the VoxCPM venv does not; everything published came from the latter, so
the existing table is internally consistent — but one arm run from the other venv would have produced
a phantom 0.011 win. Fixed by `adapter/_score.py`, which pins the backend and refuses the wrong one.

⭐ **Both defects share a shape worth naming:** neither produced an error, and both were found only
because a *control* was re-run rather than assumed. The first surfaced as one changed hypothesis in
150; the second as a number that would not reproduce against a file that had not changed.

## ⭐⭐ F8d. The benchmark's B-CER/U-CER split is not comparable across bias-list sizes

Third methodological defect in this benchmark, and the largest. It is the sibling of `F1` (SARA AM)
and, like it, it makes published numbers non-comparable rather than merely noisy.

**The rule.** An inserted run is charged to the biased bucket *"iff its string matches (⊆ or ⊇,
len≥2) any form of any entity in the **provided bias list** (gold + distractors)"* — the scorer's own
docstring, a character-level analogue of Le et al.'s (2104.02194) word-level "inserted word is in the
bias list".

**Why it breaks in Thai specifically.** The original rule operates on *words*. Thai has no word
boundaries, so the benchmark's unit is a **character**, and the threshold is len≥2. A two-character
Thai fragment is a substring of *something* in a 500-entry list nearly always.

**The measurement.** On `dis@500` — a condition whose bias list contains **no gold entities at all**:

- **650 of 3,222 inserted characters (20.2%) are charged to B**, in 154 runs.
- **126 of those 154 runs (82%) are two or three characters long.**
- B-CER therefore reads **0.49588 against the unbiased 0.26178** — nearly double, from a list that
  cannot possibly have helped or hurt the model find a gold entity.

**The proof that the model did not change.** Total insertions are flat across every condition —
3036 / 2992 / 2933 / 3024 / 2938 — while the split moves monotonically with list size:

| condition | total ins | → B | → U |
|---|---|---|---|
| none | 3036 | **0** | 3036 |
| bias@0 | 2992 | 208 | 2784 |
| bias@50 | 2933 | 364 | 2569 |
| bias@500 | 3024 | 703 | 2321 |
| dis@500 | 2938 | **652** | 2286 |

**Consequences, in order of how much they cost.**

1. 🔴 **The over-biasing gate this project adopted from `H21` — "recall may rise only if U-CER does
   not" — can be passed by changing nothing**, because U-CER falls automatically as the bias list
   grows. Any arm reporting U-CER against a `none` baseline is reporting partly an artefact.
   ✅ The fix is free: compare only **equal-list-size** conditions, `bias@N` against `dis@N`.
2. **B-CER cannot be compared across systems evaluated under different conditions**, which is what a
   contextual-biasing leaderboard is for.
3. It inflates exactly the metric that makes biasing look effective, in the direction that flatters
   the method.

⚠️ **Scope.** Measured on one model's hypotheses over 682 real utterances, on this benchmark's
scorer. The rule is in the published scorer, so the mechanism generalises to anyone using it; the
*magnitude* is ours.

⭐ **What it is worth.** Alongside `F1`, this is a second concrete "Thai CER numbers are not what they
look like" result, and it comes from the same habit: a control that should have been boring
(`dis@500` exists to measure hallucination, and the fire rate was a clean 0.0002) turned out to
falsify the metric instead.

## ⭐⭐ F9. Typhoon's own biasing can be reused — via two models, not one

`typhoon-asr-realtime-nemo-ctc` (135M, CC-BY-4.0) ships **`word_boost.py` and `kws_spot.py`** and
reaches **entity recall .158 → .852 with zero biasing training**, staying flat to .757 at 500
distractors — beating every 1.5–2B model.

✅ **The tokenizers are byte-identical** (`88e04338795f42f0492c953e461ac785`, hash-verified). It
declares `base_model: typhoon-asr-realtime` and is our **frozen 17 encoder layers + 4 new + a CTC
head**.

🔴 **But their spotter cannot run on our model.** From their code: it needs a per-frame `[T, V]` CTC
posterior matrix (`lp.argmax(axis=1)`, a blank index). A transducer emits `[T, U, V]` conditioned on
its own prefix. **Hence the two-model splice** — RNN-T for the transcript (better CER 6.89/9.92 vs
8.54/13.46), CTC for the spots.

🔴🔴 **CORRECTED 2026-09-19 — "unmeasured by anyone" was wrong, and so was the novelty claim under it.** This line read *⚠️ Unmeasured by anyone: what splicing CTC spots into an RNN-T transcript does to U-CER.* It is measured, and the whole mechanism is published: **CTC-WS**, Andrusenko et al. 2024 (arXiv 2406.07096, `LITERATURE.md` `A46`) — CTC log-probs matched against a context graph, candidates *"replace their greedy recognition counterparts in corresponding frame intervals"*, and *"a Hybrid Transducer-CTC model enables the CTC-WS application for the Transducer model."* **It ships in the NeMo we run.** Its GPU boosting-tree successor, TurboBias (`A48`), is arm D4. He asked the question that found this (*"is RNN-T + CTC spot really our idea?"*) and the answer is no.

✅ **What survives, and it is not nothing.** The Thai measurement stands exactly as recorded — recall .328 → .554, +99 of 439 gold mentions, U-CER/CER/B-CER all falling, RTF 0.028 → 0.032 — and so does `F8b`'s progression (re-rank −1, bias +8, second opinion +99), which is a finding about where the bound sits, not about who invented the splice. **Quote D6 as the first Thai measurement of a published method, never as a method.**

⚠️ **Only the abstracts were read** (Consensus, 2026-09-19). Two questions need the full papers before anything further is claimed: whether `A46` reports a **threshold study** — the tau sweep found ~45 gold mentions between −1.5 and −2.5 that no constant can reach — and whether their *jointly-trained* Hybrid Transducer-CTC differs materially from our *two released checkpoints sharing a frozen encoder*.

## 🔴 F10. Both of our candidate thesis directions are crowded

| | what we believed | what the literature says |
|---|---|---|
| synthetic audio teaches rare words | an open architecture problem | **~40 papers.** `H19` did it on an **RNN-T, −57% on OOV**, 2020, 93 cites. `H16` did it on **road names and addresses with no real audio at all** (57.6% → 16.5%). `H25` did it with **encoder freezing** at **>65%**. `H15` published our own "attention instead of n-gram" reframe in **2023**. Our adapter's **6% is the smallest number in the family** |
| Thai end-of-turn via sentence-final particles | our opening | 🔴 **`E2` claims "the FIRST systematic study of Thai end-of-turn detection" and names "Thai-specific linguistic cues (e.g. sentence-final particles)"** — a Thai group, 2025. `E41` (2026) already puts turn-taking on a **frozen ASR encoder**. `E52` ships a **1,145-hour** dataset |

🔴 **The ledger had under-recorded `E2` for two sessions.** `E:\Claude\visualllm-business\HANDOFF.md` built the "stronger"
direction on ground a 2025 paper had taken.

⭐ **What survives as novelty: none of it is Thai.** So the honest claim in these families is
**"first Thai", not "new method"** — a legitimate paper, a weak architecture contribution.

## ⭐ F11. Two rows that argue FOR keeping the synthetic-audio pipeline

A correction to a wrong inference made the same day — that an n-gram over the same text would make
`colab/` redundant:

- **`H26`** compared text-only adaptation against TTS audio head-on: **text loses, "even when the TTS
  engine is sub-optimal"**.
- **`H27`** (97 cites) compared TTS gains against feeding the same text in as an LM: **mostly
  independent — they stack.**

**So a text-side baseline is an arm to ADD, not an alternative to choose.** And `A33` (93 cites)
reports 15.4–34.5% from deep fusion **over a baseline that already used shallow fusion** — the same
pattern.

## F12. Four more that are immediately actionable

| | finding |
|---|---|
| `H36` | **Filter synthetic data, don't use all of it.** A similarity scorer held full accuracy on **<30%** of the TTS data; `WAVe` cut training steps 34% and cross-domain WER 13.54% → 6.89%. ⭐ `smartturn/verify.py` already reads every clip back through Typhoon — that readback should be a **filter**, not a report |
| `H28` | **Pretrain on synthetic first, then transfer to real** — the reverse order is worse. The dominant error is **prosody mismatch**, which `voxcpm/_squeeze.py` controls |
| `H16` | ⚠️ **Multiple TTS systems beat one TTS with many voices** — 9.8% relative at equal data size. `colab/gen_colab.py` is built the weaker way |
| `H29` | **How much Thai was VoxCPM2 trained on?** Synthetic speech can only approach real data if the TTS itself saw a large enough corpus. Unasked, and it caps everything downstream |

## F13. S1 — "freeze the encoder" — is published, not a hypothesis

`H25` (`SynthASR`, 68 cites) uses **encoder freezing** as a component of a recipe teaching an **RNN-T**
a new vocabulary from synthetic speech at **>65% relative with no degradation elsewhere**. Our
architecture, our task shape. And `H19` offers a fourth alternative — **EWC on the encoder** — also
with a number.

## ❌ F14. The adversary arm is cut

- `H4` (JODAL), its headline support: **no number, 0 citations**, mathematics journal, **no transducer
  tested**. `H13` gives 4–5%.
- 🔴 **`H9` — the same study that gave the adversary its tap point — says representation separability
  *"does not directly predict downstream ASR gains."*** Its premise, contradicted by its own source.
- `H5`, from that study, found the **opposite** instinct worked: make synthetic audio **messier**.

Two players, no published number, a contradicted premise, eight cheaper alternatives.

## F15. Closed ground — do not propose these as contributions

**Contextual biasing, in every form.** Typhoon shipped it **four ways on 2026-09-08** (`A26`), and
`H15`–`H23`, `H30`–`H35` cover attention-over-a-list, retrieval to 200k entries, tries, dynamic
vocabularies, and the over-biasing gate. **Adjacent, and taken.**

## F16. The process lesson, stated once

Four times this session, reading the actual artefact overturned a belief: the checkpoint config
(batch norm → arm 4 exists), the turn log (number convention), the CER rows (SARA AM), and their
`word_boost.py` (CTC-only). And three times a Consensus sweep overturned a plan.

🔴 **And twice I asserted novelty that a search would have disproved** — the attention reframe
(`H15`, 2023) and the latency-breakdown claim (`E61`, 2021). The repo's own rule already covers it:
*before asserting that something does not exist, say what was checked.*

## ⭐⭐ F17. A third of the benchmark's "error" is spelling convention — and it reorders the table

`A40` (Bhogale et al. 2026) reported that accounting for permissible orthographic variation
narrowed a Gemini-vs-Canary gap from 18.1 points to 11.5. That is a reason to distrust any raw-CER
ranking in a language with real spelling variation, and `F3`'s "field-wide" claim rests on exactly
such a ranking. All seven hypothesis files were already on disk, so the test cost no GPU and no
decoding: **only the scoring convention changed, identically for every system**
(`adapter/ortho_rescore.py`, `adapter/_ortho.py`, 2026-09-17).

**Where the error mass actually is.** Aligning all 7 systems against the 682 real references gives
56,232 error characters. Classified by what the disagreement *is*:

| family | share | normalised here? |
|---|---|---|
| script: Latin ↔ Thai, outside gold spans | **20.4%** | ❌ no — see below |
| number: digits ↔ Thai words | **9.1%** | ✅ the `num` rung |
| catastrophic runs ≥ 60 chars (decoder collapse) | 3.0% | a real failure, never |
| hesitation markers dropped by the model | 1.0% | the `filler` rung, off by default |
| everything else | 66.5% | — |

**≈30% of the measured error mass is convention, before one recognition error is counted.**

**The ranking does not survive; the finding does.** Re-scoring at the `num` rung:

| | Kendall τ vs baseline | pairs swapped |
|---|---|---|
| ranking by CER | **0.429** | **6 of 21** |
| ranking by entity recall | 1.000 | 0 — invariant by construction |
| B-CER/U-CER range | 2.01–3.04 → **2.07–3.08** | — |

So:

- 🔴 **The CER column's ordering is partly an artefact of whose number convention matches the
  corpus typist's.** `typhoon-whisper-large-v3` goes 3rd → 1st; **ours goes 7th of 7 → 5th of 7**,
  CER 0.11248 → 0.09493, **−15.6% relative, with no change to the model at all.** The references
  rewritten per system say why: 75 for typhoon-whisper and 75 for ours (both write number words,
  per `F2`) against 6 for `qwen3_asr_0p6b` (writes digits, so it already agreed with the corpus and
  had nothing to gain — it gains 0.7%, against our 15.6%).
- ✅ **`F3` itself survives, slightly strengthened.** The B/U ratio stays 2.0–3.1× for all seven,
  and the table as published is sorted by entity recall, which cannot move — recall is computed
  from the untouched hypothesis. The claim "entity spans are universally 2–3× harder" is not a
  convention artefact.

**The big family is the one left alone.** Script disagreement is 20.4%, twice the number family,
and the `ent` rung dents it by 0.0002 CER — it fires on 2 of 682 utterances for our system. The
benchmark's own dual-form defence is already healthy (439/439 gold spans located; its `cap=32`
identity-only fallback fires on only 9 of 298 entity-bearing utterances), so what leaks past it is
**un-annotated loanwords** — `prawat`/`ประวัติ`, `japanesecoaster`/`แจแปนิสโคสเตอร์` — whose pairing
no inventory in this corpus declares. Normalising those needs Thai↔Latin transliteration, which is
a guess, and a guess here forgives real errors. Measured and left alone on purpose.

⚠️ **This is not a way to make our numbers look better, and must never be quoted as one.** The
rewrite only re-renders the *reference* into a form the hypothesis contains **verbatim**; it cannot
turn a wrong transcription into a right one, and it under-corrects — `1979` heard as
`หนึ่งพันเก้าร้อยแปดสิบเก้า` still eats all 26 characters rather than the one digit actually missed.
Every number in this repo's records remains the `none` rung. What F17 licenses is one sentence in
the thesis: *a Thai CER ranking that has not been normalised for orthography is not a ranking of
accuracy.*

### Two collision bugs, both caught after the selftest passed

- **The `ent` rung rewrote `คลินตัน` → `คLinตัน`.** The corpus has an entity `ลิน`/`Lin`; `Lin`
  sits inside the hypothesis `Clint Clint`. Same mechanism turned `นิวซีแลนด์` into `Newซีแลนด์`
  via `นิว`/`New`. 447 of the corpus's forms are 3–4 Thai characters — each a syllable that turns
  up inside unrelated words. The benchmark carries a guard for precisely this (citing `ไลน์` inside
  `เดทไลน์`); I arrived at it independently, by **spot-checking six real rewrites by eye after a
  26-case selftest had passed green**.
- **The `filler` rung ate `อ่าน`, "to read".** A substring `.replace` of `อ่า` turns `ผู้อ่าน`
  (reader) into `ผู้น`, 35+ times. The harness caught this one, not the eye: entity recall is
  supposed to be invariant across rungs, and it moved 0.003 for one system. Bounded to whitespace
  now — which leaves most genuine unspaced fillers in place, so the rung is worth almost nothing
  and is off by default. **That is the answer to "should we strip fillers", not an assumption.**

---

## ⭐⭐ F18. The emotion teacher fine-tunes, and ONLY the full unfreeze earns its cost

Measured 2026-09-22, `adapter/finetune_emotion2vec.py`, 410 actor-held-out THAI-SER clips.
`emotion2vec_plus_large` zero-shot scores **0.7317**; fine-tuned end to end it scores **0.8488**,
macro-recall **0.6848 → 0.8570**, four points off the **0.9286** human ceiling on the same subset.

**Where the gain lives is the useful half of the finding.** Four rungs, paired McNemar on identical
clips: every rung beats zero-shot (head +20 clips p=0.011, top2 +21 p=0.014, top4 +29 p=0.0009,
full +48 p<0.0001) — but **head, top2 and top4 are statistically indistinguishable from each other**
(head vs top4 p=0.31), while **full separates from all three** (vs head p=0.0046, vs top4 p=0.0295).
Refitting a 4,100-parameter head captures most of the easy gain; 25 M and 50 M unfrozen parameters buy
nothing measurable; 164 M buys a real, separable jump. **There is no middle rung worth running.**

⭐ **The repair is concentrated in SAD: recall 0.396 → 0.906.** That is why macro-recall moves further
than accuracy, and it matters because SAD is the class SenseVoice's shipped head emits at **0.019** on
these same clips while its own encoder carries it at **0.830**. The broken class is the fixable one.

🔴🔴 **A single learning-rate guess would have inverted this finding.** `full` dev accuracy by
backbone lr is **non-monotonic**: 1e-5 → 0.7762, 3e-5 → **0.7119**, 1e-4 → **0.8881**. The middle value
is worse than the smallest. This session's own first guess was 1e-5, which reports "full fine-tuning is
no better than a frozen head" — the opposite conclusion, from a defensible-looking run. **Sweep the
backbone lr or do not claim a fine-tuning result.** The sweep only became affordable because the work
moved to the GPU (9-minute CPU epoch → ~30 s; 18.9× fwd+bwd, 37× on the inference pass).

⚠ **It is balanced on early stopping.** 426 training clips from SEVEN actors against 164 M parameters:
train loss hits **0.0009–0.0014** within 8 epochs and the dev-selected checkpoints are epochs **3** and
**7**. Every number here depends on dev-side epoch selection; without it the rungs read worse than
zero-shot. **Report the selection rule alongside the score or the score is not meaningful.**

⚠ **And it is still acted speech.** Fine-tuning gains +8.4 points on improvised turns but +15.7 on
scripted, so the impro/script gap nearly quadruples (0.026 → 0.098). The model is improving faster at
acted delivery than at spontaneous delivery, and **nothing here is measured on genuinely natural Thai.**

---

## ⭐⭐ F19. A good emotion teacher is still a bad LABELLER on mostly-neutral speech

🔴 **Read `F20` first.** Its arithmetic below is correct and still applies to a corpus whose speakers actually feel something. But it assumes every clip HAS a true emotion, and `F20` measured a corpus where they do not — there the prior-shift correction has nothing to solve for. `F19` is the right model for *spontaneous* speech; it is the wrong model for *read* speech.

His question, 2026-09-22: *"what if we use this emotion model to label all the dataset which might have
most sound be the natural, will that be a problem"*. Measured, not argued:
`adapter/probe_pseudolabel_yield.py` → `results/pseudolabel_yield_metrics.json`.

**The teacher does not change; the corpus does.** It mislabels a neutral clip as HAPPY 13.8% of the
time — harmless when NEUTRAL is a third of the corpus, fatal when it is 85%, because the leaked
neutrals then outnumber the true happy clips **3:1**. Pseudo-label precision by corpus mix:

| corpus | NEUTRAL | HAPPY | ANGRY | SAD |
|---|---|---|---|---|
| THAI-SER as measured (35/27/25/13) | 0.921 | 0.592 | 0.878 | 0.600 |
| spontaneous-ish (85/6/5/4) | 0.991 | **0.232** | 0.780 | 0.317 |
| read corpus (95/2/1.5/1.5) | 0.997 | **0.092** | 0.597 | 0.148 |

🔴 **Accuracy is the wrong instrument here and stays flattering throughout.** A student distilled on
such labels learns "say neutral", scores ~0.87, and is useless.

⭐ **Confidence partially rescues it — but only AFTER the fine-tune, which is the real value of `F18`'s
head rung.** Confidence-as-error-detector AUROC is 0.737 zero-shot, 0.777 fine-tuned. With per-class
thresholds chosen on **dev** and confirmed once on test, on an 85%-neutral corpus:

| teacher | NEUTRAL | HAPPY | ANGRY | SAD |
|---|---|---|---|---|
| zero-shot | t=0 → 0.991 | 🔴 **never usable at any threshold** | t=0 → 0.780 | t=0.6 → 0.489 (dev said 0.772 — did not transfer) |
| fine-tuned | t=0 → 0.986 | t=0.99 → **1.000** | t=0.6 → 0.693 | t=0.9 → 0.835 |

**The shipped head's HAPPY labels are unusable at every confidence level; the refit head's reach 1.000.**
The fine-tune's product is CALIBRATION, not accuracy — which is invisible if you only read `F18`'s
accuracy column.

🔴 **One threshold cannot serve four classes.** A single global rule is pinned by the worst class
(HAPPY needs t=0.99) and at that threshold ANGRY — already clean at t=0.6 — has stopped being emitted:
**49 correct per 100 k against 1,569**. Per-class thresholds, always.

🔴🔴 **THE DECISIVE NUMBER IS YIELD, AND IT CLOSES THE ROUTE FOR THE MINORITY CLASSES.** At those
thresholds, per 100 k clips: NEUTRAL 70,931 correct, ANGRY 4,118, SAD 1,660, **HAPPY 327**. Applied to
**every Thai clip reachable on this box** — 32,183 unique LOTUSDIS recordings **plus** 32,823 Common
Voice Thai train clips, 65,006 total:

| | pseudo-label the ENTIRE Thai corpus on this box | **download the other 26 THAI-SER shards** |
|---|---|---|
| HAPPY | **213** | **~3,402** |
| ANGRY | 2,677 | ~2,793 |
| SAD | 1,079 | ~1,933 |
| label quality | projection from an acted-speech teacher | human majority vote, agreement-filtered |

⭐ **Pulling shards you already have access to beats pseudo-labelling every Thai clip on the box by
16× on HAPPY, at ~400–600 MB per shard.** That is the answer to the question.

✅ **What pseudo-labelling is still the only route to:** NEUTRAL at 0.986 precision and 71 k per 100 k,
on **natural** speech — which THAI-SER structurally cannot provide, being 100% acted studio
performance. So: THAI-SER for the minority classes, pseudo-labels for natural-speech NEUTRAL and
acoustic coverage, **soft labels not hard**, per-class thresholds, and never as an evaluation set.

⚠ **Three caveats that bound all of the above.** (1) Every projection is OPTIMISTIC — `P(pred|true)` is
measured on acted speech where the teacher is at its best (`F18`: impro 0.804 vs script 0.903, and a
raw-mel floor of 0.5756 says much of the apparent skill is reading exaggerated actor prosody).
(2) Dev→test threshold transfer is shaky at this sample size: zero-shot SAD fell 0.772 → 0.489,
fine-tuned ANGRY 0.769 → 0.693, missing its own bar. (3) **The 85%-neutral prior is ASSUMED, not
measured** — nobody has measured the emotion prior of LOTUSDIS or Common Voice Thai, and measuring it
with this teacher would be circular. A hand-check of a few hundred sampled clips would fix that and is
the cheapest thing that would firm up the whole analysis.

---

## ⭐⭐ F20. A corpus with no emotional INTENT has no emotion to label — and the teacher labels it anyway

**His observation, 2026-09-22:** *"I think the dataset didnt make it to have emotion in it so, some
voice sound sad but the transcript is not."* It is correct, it was measured, and it **supersedes the
framing of `F19`** (which is left standing — its arithmetic is right, its assumption is not universal).

`adapter/probe_natural_prior.py` → `results/natural_prior_{metrics.json,hyps.jsonl}`.
2,000 Common Voice 17 Thai test clips (2.77 h of read speech) scored by the fine-tuned teacher, plus
300 of their transcripts labelled for emotion **from the text alone**, by a separate model with no
access to the audio or to the audio predictions.

| | NEUTRAL | HAPPY | ANGRY | SAD |
|---|---|---|---|---|
| **text** says (300 sentences) | **0.893** | 0.063 | 0.017 | 0.027 |
| **audio** says (same 300) | 0.553 | 0.207 | 0.090 | 0.150 |
| `F19` had **assumed** | 0.850 | 0.060 | 0.050 | 0.040 |

🔴 **They disagree by 34 points AND do not correlate clip-by-clip.** Chi-square of independence:
**5.93 on 9 df**, against a critical 16.92 — no detectable association. Only **14 of 300** clips have
both audio and text non-neutral, and those agree on the class 7 times, which is chance. Every per-class
lift sits inside noise; the largest is audio-ANGRY predicting *sad* text at 2.78×, which is nonsense.

⭐ **The mechanism, and it is not "the model is bad".** A volunteer reading a prompt has no emotion to
detect. Asked for a label anyway, the model returns the only thing in the signal — voice timbre,
recording conditions, speaking rate, fatigue — mapped onto emotion words. Caught at its clearest:

```
0.997 HAPPY   โซเดียมคลอไรด์เป็นแค่เกลือธรรมดา      "Sodium chloride is just ordinary salt"
0.995 HAPPY   ฟาโรห์หลายคนปกครองอาณาจักรอียิปต์โบราณ   "Many pharaohs ruled ancient Egypt"
0.915 ANGRY   คนขับรถม้าวิ่งกลับไปที่ตัวบ้าน     "The carriage driver ran back to the house"
```

Encyclopedia facts at 99.5% HAPPY. **Confidently wrong is the worst failure mode for pseudo-labelling**,
because confidence thresholding — `F19`'s rescue — protects against *random* error and **concentrates**
systematic error. Thresholds tuned on acted speech make this worse, not better.

🔴 **Why this supersedes `F19`'s framing.** `F19` treated the problem as CLASS PRIOR SHIFT: the
teacher is fixed, only the mix of true classes changes, so a confusion matrix corrects it. That assumes
every clip HAS a true emotion. In an intentless corpus most do not, so there is no prior to recover and
no matrix that can fix it. It also explains the anomaly `F19`'s method produced: black-box shift
estimation returned 66% neutral for a corpus whose text is 89% neutral — the correction was solving for
a quantity that does not exist.

⭐⭐ **THE CONSEQUENCE FOR CORPUS CHOICE, which is the actionable half.** The criterion is not hours
and not audio quality — it is **whether the speaker had a reason to feel something**:

| | emotional intent | verdict |
|---|---|---|
| conversational / broadcast / YouTube (GigaSpeech2) | real, spontaneous | ⭐ best available |
| acted studio (THAI-SER) | performed, but present and labelled | ✅ usable, with the acted caveat (`F18`) |
| read prompts (Common Voice) | **none** | ❌ not emotion data at any scale |

⭐ **This RAISES the value of the 10,000 h corpus rather than lowering it.** The worry was that a big
corpus would be mostly neutral. Spontaneous-but-neutral is still real emotion data — a neutral clip
from someone genuinely talking is a valid NEUTRAL example. Read-and-intentless is not data at all.
Common Voice ranks last despite having the cleanest audio on the box.

⚠ **Limits of this measurement, stated because they bound it.** Text emotion is a PROXY, not ground
truth — a neutral sentence can be spoken emotionally. In a corpus where nobody has a reason to feel
anything that is the argument rather than a hole, but it is not the same as human listening labels;
`D:/thai-ser-listen/` holds 40 extracted clips for exactly that check and **nobody has listened yet**.
The text labels come from one model in one pass, not a panel. And 300 sentences with only 8 non-neutral
text labels cannot detect a *weak* association — only a strong one, and there is none.

🔴 **A self-correction worth keeping.** Mid-session I showed six SAD-predicted clips whose text was
genuinely sad ("I don't think I'll ever see my old friends again") and argued readers must be
unconsciously performing the line. That was six examples ranked by confidence out of 2,000 — the top of
a ranking will always look like a pattern. At scale the effect is **absent** (lift 1.67× on 45 clips,
chi-square n.s.). **Cherry-picking the head of a confidence ranking manufactures a story; this repo has
`F8c` because that kind of thing survives into published numbers.**

---

## ⭐⭐ F21. Emotion needs its own head AND its own encoder — the one-pass goal costs 15 points

Measured 2026-09-22. `adapter/probe_shared_encoder_cost.py` →
`results/shared_encoder_cost_metrics.json`. All on the same 410 actor-held-out THAI-SER clips.

**Half of this was already settled and is restated because it is the premise:** SenseVoice's shipped
CTC head emits SAD at **0.019** recall while a single affine layer on **that same model's own frozen
encoder** reads SAD at **0.830**, with 40% `EMO_UNKNOWN` abstention. One emotion token competing in a
25,055-token softmax, contributing one position of gradient out of hundreds, drifts to the safe
default. **A shared text+emotion head is not a candidate.**

🔴 **THE NEW RESULT: the encoder cannot be shared either, and it is not a head-capacity artefact.**
The 2026-09-19 figure came from a deliberately weak single affine layer, so it was only a lower
bound. Sweeping head capacity (`linear / mlp128 / mlp512`) and weight decay on the dev actors was the
test of whether a real head closes the gap. **It closed 2 points of 10.**

| encoder | best head (dev-chosen) | test acc | macro-R | NEU | HAP | ANG | SAD |
|---|---|---|---|---|---|---|---|
| raw log-mel, no encoder | mlp128 | 0.5951 | 0.5920 | .717 | .500 | .490 | .660 |
| Typhoon (ASR) | mlp512 | 0.6780 | 0.6904 | .628 | .673 | .725 | **.736** |
| SenseVoice (ASR) | mlp128 | 0.7000 | 0.7136 | .676 | .700 | .686 | **.792** |
| **emotion2vec** | **linear** | **0.7805** | 0.7532 | .848 | .727 | .833 | .604 |
| all three fused | mlp128 | 0.8146 | 0.8041 | .834 | .764 | .882 | .736 |
| emotion2vec fully fine-tuned (`F18`) | linear | **0.8488** | **0.8570** | .848 | .782 | .892 | **.906** |

⭐ **The tell is which head won.** emotion2vec's best head was the plain **linear** one — extra
capacity bought it nothing, because its features are already linearly separable for emotion. The ASR
encoders *needed* MLPs and still lost: thinner information, more entangled. Paired McNemar, best ASR
encoder vs emotion2vec: **p = 0.0032**.

🔴🔴 **AND THE STRUCTURAL ARGUMENT CLOSES IT.** `F18` showed that what makes emotion good is a
**full fine-tune** (0.8488 vs 0.7805 frozen). You cannot fully fine-tune a shared encoder for emotion
without destroying its ASR. **The thing that makes emotion work is precisely the thing sharing
forbids** — that is not a tradeoff to tune, it is structural.

⭐ **The useful surprise: the ASR encoders are COMPLEMENTARY, not merely worse.** They beat
emotion2vec at SAD (0.792 / 0.736 vs **0.604**) while losing on NEUTRAL. Fusing lifts macro-recall
**0.7532 → 0.8041** and SAD to 0.811 (Typhoon+emotion2vec). Since the ASR encoder runs anyway for the
text, its pooled features are already computed — feeding them to the emotion head is nearly free.
⚠ But the fusion is **marginal**: three encoders reach p = 0.0488 against emotion2vec alone, and
three combinations were tried, so it does **not** survive multiple-comparison correction. The
macro-recall gain is the convincing half, not the accuracy. And **fine-tuning one encoder (0.8488)
beats fusing three frozen ones (0.8146)** — priority is fine-tuning, not fusion.

**The price, stated so the spec's goal carries a number.** Best frozen ASR encoder **0.7000** against
a fine-tuned emotion encoder **0.8488**: **~15 points of accuracy is what one encoder pass costs**,
paid to save ~20 ms on the GPU and 164 M params of VRAM. The 2026-09-19 spec states *"one model, one
pass, text + emotion + events"* with no price attached. This is the price.

⚠ **Scope.** Everything above is **frozen** encoders. **Multi-task fine-tuning** — one shared encoder
trained on ASR and emotion jointly with a forgetting control (`S10`) — is **unmeasured**, and it is
the only route that could deliver a genuine single pass at good accuracy. Do not read this finding as
closing that door; read it as closing the *frozen-shared-encoder* door.

---

## ⭐⭐ F22. Emotion needs 4 seconds — but "is something wrong" needs one

Measured 2026-09-22. `adapter/probe_emotion_duration.py` → `results/emotion_duration_metrics.json`.

🔴 **The table this repo already had was CONFOUNDED.** The 2026-09-19 by-duration-band split
(emotion2vec 0.656 / 0.697 / 0.770 / 0.737 for <1s / 1-2s / 2-4s / >4s) compared **different clips**,
not the same clips at different lengths — short THAI-SER utterances are different sentences by
different actors carrying a different emotion mix. Its apparent peak at 2-4 s and drop beyond 4 s read
as "too much audio hurts". It does not.

**The fix: truncate the SAME clips.** 212 test clips of at least 4 s, each cut from the **start** to
every window, so speakers, emotions and labels are identical at every point and only the amount of
audio varies. The curve is then **monotonic**:

| window | encoder frames | accuracy | macro-R | confidence − accuracy |
|---|---|---|---|---|
| 0.5 s | 24 | 0.5943 | 0.5707 | **+0.168** |
| 1 s | 49 | 0.6226 | 0.6032 | +0.146 |
| 1.5 s | 74 | 0.6509 | 0.6391 | +0.146 |
| 2 s | 99 | 0.6745 | 0.6690 | +0.161 |
| 3 s | 149 | 0.7217 | 0.7145 | +0.110 |
| **4 s** | 199 | **0.7925** | 0.7872 | +0.044 |
| full (6.2 s mean) | 312 | 0.7925 | 0.7886 | +0.076 |

**4 s is the plateau** — 4 s and full-clip agree to four decimals, so beyond 4 s more audio adds
nothing. The encoder runs at **50 frames per second**. Below the plateau the cost is ~7 points at 3 s,
~12 at 2 s, ~20 at 0.5 s.

⭐⭐ **THE PER-CLASS SPLIT IS THE ACTIONABLE PART, and it has a mechanism.**

| emotion | recall at 0.5 s | within 2 pts of full at |
|---|---|---|
| NEUTRAL | 0.937 | **0.5 s** |
| ANGRY | 0.740 | **1 s** |
| SAD | 0.300 | **2 s** |
| HAPPY | 0.306 | **4 s / full** |

The fast pair is separable by **energy level alone** — neutral sits mid, angry sits high, and that is
audible in half a second. The slow pair needs **contour over time**: happy and angry are *both*
high-energy so only pitch shape separates them, and sad and neutral are *both* low-energy for the same
reason. **Energy is instantaneous; contour costs seconds.** For a live loop: *"something is up"* is
available in ~1 s, *which* emotion is not available until ~4 s.

🔴 **A trap that invalidates `F19`'s thresholds on short audio.** Below 3 s the model is
**overconfident by 11-17 points** — at 0.5 s it reports 0.763 mean confidence while scoring 0.594.
Per-class confidence thresholds assume confidence tracks correctness; under ~3 s it does not. Any
pseudo-labelling or selective-prediction rule must carry a minimum-duration gate.

⚠ **Two limits.** These are **truncated long clips**, not naturally short utterances — a 1 s cut of a
6 s sentence is a fragment, a naturally 1 s utterance is a complete thought, and they need not behave
alike. And the cut is from the **start**, which is what a live loop has but is the harder case: the
emotional peak of an utterance is usually not at its beginning, so a centred or end-anchored window
should do better. That is one cheap follow-up and it is not run.

---

## ⭐⭐ F23. Under THAI-SER's OWN protocol the margin is +9.8, not +17.5 — and the protocol found a channel bug

Measured 2026-09-24. `adapter/probe_thaiser_protocol.py` → `results/thaiser_protocol_metrics.json`.
The full 29-shard corpus was pulled for this (`adapter/fetch_thaiser_shards.py`): **11,301 clips,
200 actors, 100 sessions, 12.37 h**, against the 1,256 clips / 22 actors every earlier emotion
number used.

**Why this arm exists.** `F18` reports 0.8488 on an actor-holdout split and `A50` reports the
corpus authors' 4-class baseline at WA 67.34. Those are close enough in kind to tempt a
"+17.5 points" claim and **not** close enough in protocol to earn it. So their protocol was read
out of their own code (`tann9949/thaiser-experiments`, `experiment/data/thaiser.py` +
`config/sample-normalize/baseline/four_emotions.yaml`) and adopted wholesale.

| | WA | UA |
|---|---|---|
| **ours**, `mic_con` (their training mic) | **77.18 ± 1.98** | **73.18 ± 2.52** |
| ours, `mic_clip` (cross-channel) | 73.95 ± 2.96 | 67.19 ± 3.28 |
| **their CNN+LSTM baseline** | 67.34 ± 3.05 | 62.61 ± 3.19 |
| **delta on their mic** | **+9.84** | **+10.57** |

8 folds over studio sessions, all scored, none skipped; agreement **>= 0.71** (not our 0.70);
4 classes; Zoom excluded; **3-second cap** as their packer applies.

🔴🔴 **THE PROTOCOL COST US ~8 POINTS OF THE APPARENT MARGIN, AND THAT IS THE FINDING.**
The naive comparison (`F18` 0.8488 vs their 67.34) reads +17.5. Matching their protocol gives
**+9.84**. **A cross-paper accuracy comparison that does not reproduce the other side's
windowing is not a comparison.**

✅ **DECOMPOSED, because "most of it is the cap" was first written here as an inference and
then measured (2026-09-24).** Re-running the identical 8 folds with the cap removed:

| their 8 folds, `mic_con` | WA | UA |
|---|---|---|
| with their 3 s cap | 77.18 ± 1.98 | 73.18 ± 2.52 |
| **without the cap** | **82.22 ± 1.38** | **79.82 ± 2.72** |
| the cap alone costs | **5.04** | **6.64** |

So of the 7.7-point drop from `F18`'s 84.88, **5.04 is the 3-second cap (65%)** and **2.66 is
everything else** -- their folds, their 0.71 threshold, their training mic, combined. The
inference was right. ⭐ And that 2.66 is itself worth keeping: it says `F18`'s actor-holdout
split was NOT meaningfully optimistic against the corpus authors' own fold design.

🔴🔴 **AND +9.84 IS A FLOOR, NOT A CEILING -- this arm fits a LINEAR HEAD ON A FROZEN
ENCODER.** It is `F18`'s *head* rung, not the full fine-tune. Their 67.34 is a CNN+LSTM trained
from scratch, so the comparison as it stands is **our weakest rung against their trained
system**. `F18` measured the full fine-tune as worth several points over the head rung
(0.7805 → 0.8488 on its own split), and **a full fine-tune under their protocol is UNRUN**.
Do not describe +9.84 as our best result; describe it as the conservative one.

✅ **What survives is smaller and far more trustworthy.** +9.84 WA is about **3x their own
fold-to-fold standard deviation**, reported over 8 folds with a spread rather than one split. And
`F18`'s 0.8488 came from a model trained on **426 clips from 7 actors** with every rung balanced
on dev-side early stopping; each fold here trains on **~7,300 clips from ~190 actors**. The
number went down and the confidence in it went up.

⭐⭐ **THE UNASKED-FOR FINDING: THIS MODEL IS CHANNEL-SENSITIVE.** Their protocol tests on both
`con` and `clip`, so training on `mic_con` and testing on `mic_clip` is measured for free:
**-3.2 WA and -6.0 UA.** UA takes double the hit, so **channel mismatch damages the minority
classes twice as hard as the majority** -- the same asymmetry `F19` found for prior shift.
🔴 Every emotion number in this repo before today was measured on `mic_clip`, and this model
trains on `mic_con`. For the voice loop, where the microphone is neither of these, a 6-point UA
drop from channel alone is a deployment risk that nothing in the plan currently accounts for.

⚠ **What matching the DATA protocol does NOT fix.** Their baseline is a CNN+LSTM trained from
scratch on 64-bin fbank with VTLP; ours is a 164 M self-supervised encoder with a fitted head.
Matching the protocol makes the **numbers** comparable, not the **systems**. The honest thesis
sentence is about what large-scale self-supervised pretraining transfers to a low-resource tonal
language, not about our engineering beating theirs.

⚠ Still unrun: the **2,516 Zoom clips** now in the cache -- the corpus authors' own out-of-domain
set, and the closest thing to non-studio speech this project has (`F20`). They need a `mic_zoom`
cache pass; the current cache holds only `mic_clip` and `mic_con`.

---

## ⭐⭐ F24. Contextual biasing DOES move entity recall — 144 → 207 — and it survives the dis@N control

Measured 2026-09-24. `adapter/arm1_rareword_qwenctx.py` → `results/arm1_rareword_qwenctx_*.json`.
`typhoon-ai/typhoon-asr-qwen-0.6b-ctx` (0.78 B, prompt-based biasing, a LoRA fine-tune of
`Qwen/Qwen3-ASR-0.6B-hf`), 682 real-split utterances, the benchmark's own frozen scorer.

**This closes `F8b`**, which recorded five decode-time arms failing to move entity recall off 144
of 439 gold mentions.

| system / condition | CER | U-CER | B-CER | **entity recall** |
|---|---|---|---|---|
| `typhoon-asr-realtime`, none (`F8b`'s baseline) | 0.1125 | 0.1057 | 0.2618 | **0.3280** (144) |
| `qwen-0.6b-ctx`, none | 0.0997 | 0.0937 | 0.2214 | 0.3690 (162) |
| `qwen-0.6b-ctx`, **dis@50** (control) | 0.1013 | 0.0920 | 0.2912 | 0.3645 (160) |
| `qwen-0.6b-ctx`, **bias@50** | **0.0983** | **0.0899** | 0.2702 | **0.4715** (207) |

✅ **IT SURVIVES THE CONTROL THAT WOULD HAVE KILLED IT.** `F8d` is why `dis@50` was run: the
benchmark's scorer charges insertions to B-CER on a >=2-char substring match against the LISTED
forms, so a longer list can deflate U-CER and flatter recall for free. Measured here, the
**list-length artifact alone is -0.0046 recall** — essentially zero. Against `dis@50`, the same
50-entry list with **no gold entity in it**, biasing gains **+0.1071 recall / +47 mentions**.
That is real biasing, not scorer inflation, and it is the first arm in this repo to clear that bar.

⭐ **The harm term does not just hold, it improves.** At the same list length U-CER goes
**0.0920 → 0.0899** and B-CER **0.2912 → 0.2702**. Every earlier arm had to argue that a recall
gain had not damaged general text; this one does not. Overall CER **0.0983** against the old
baseline's **0.1125** — the model is better before biasing starts.

**Distractor false-alarm rate 0.0002**, matching the card's claimed <=0.0004. The mechanism is
visible though: `ใส่แกง` (curry) was corrupted to `ใส่แก๊ง` (gang) by a listed entity pulling a correct
common word off target. Rare, but real, and it is what `dis@50` exists to price.

🔴🔴 **THE LESSON IS THE ORDERING, NOT THE NUMBER.** The problem five purpose-built arms failed
to move was solved by a model **Typhoon had already published** — tagged `contextual-biasing`
and `hotwords`, with a paper — which this repo **cited for its CER table and never ran**
(`docs/superpowers/specs/2026-09-19-...md` quotes `typhoon-asr-qwen-1.7b-ctx`'s card). Before
building the next arm, enumerate what the vendor already ships and run it. Hours, not weeks.

⚠ **Three caveats that bound the claim.** (1) 95% CI on bias@50 recall is **[0.4214, 0.5283]**,
so +0.107 carries about ±0.05. (2) This is a **different model**, so the jump from 0.3280 mixes
"biasing works" with "qwen-ctx is the better ASR" — **the clean isolation of biasing is the
bias@50 vs dis@50 row**, not the comparison to the old baseline. (3) It ran **in-process**
(`qwen3_asr` is not NeMo, `:8007` cannot serve it), so it never crossed the PCM16 wire that
changes 3.4% of transcripts on the NeMo path.

⚠ `check_hyps` exits 1 on all three conditions, complaining of 1,380 missing rows. That is
**expected**: the benchmark holds 2,062 utterances and only the **682 real-split** ones are run,
because scoring a synthetic-data method on the OmniVoice TTS split would be circular (`S7`).
The original `arm1_rareword.py` has the same behaviour.

---

## ⭐⭐ F25. The only public Thai SER model, benchmarked for the first time: WA 55.15 — and it does not load

Measured 2026-09-26. `adapter/bench_public_thai_ser.py` →
`results/bench_public_thai_ser_metrics.json`. `A59` established that the Thai speech-emotion model
landscape is **one checkpoint**: `awghuku/wav2vec2-base-thai-ser`. It ships no model card, no
metrics and no dataset declaration, so **nobody had ever published a number for it.** Now someone
has.

| system | WA | UA |
|---|---|---|
| `awghuku/wav2vec2-base-thai-ser` | **55.15 ± 4.74** | **45.84 ± 3.66** |
| THAI-SER's published baseline (`A50`) | 67.34 ± 3.05 | 62.61 ± 3.19 |
| ours, frozen head, their 3 s cap (`F23`) | **77.18 ± 1.98** | **73.18 ± 2.52** |
| ours, frozen head, uncapped | **82.22 ± 1.38** | **79.82 ± 2.72** |

Identical 8 folds, agreement, mic and 4 classes as `F23`. **It loses to the corpus authors' own
baseline by 12 points.**

✅ **Every fairness choice was made in its favour, so a loss cannot be blamed on the harness.**
Its 5th logit (frustrated) is DROPPED rather than scored wrong, so a frustrated prediction never
costs it; and it was given the best of all **120** possible label orderings (see below).

🔴🔴 **AND THE SCORE IS AN UPPER BOUND.** Its train/test split is undocumented. The model is
named `-thai-ser` and was plainly trained on this corpus, so **our test folds may be its training
data**. A contaminated upper bound that still loses by 12 points is decisive; it must never be
quoted as its honest generalisation.

### Two defects that matter more than the score

🔴 **1. It cannot be loaded as published.** Its config carries
`"id2label": {"0": 0, ...}` — integer VALUES where strings are required — and `"label2id"` with
string values where ints are required. Current `transformers` validates strictly and refuses:
`StrictDataclassFieldValidationError: Field 'id2label' ... doesn't match any type in
(dict[int, str], dict[str, str], NoneType)`. The benchmark repairs the config in memory to measure
it at all.

🔴🔴 **2. Its label order is undocumented AND counter-intuitive, so the obvious guess produces
garbage.** `id2label` is `{0,1,2,3,4}` with no emotion names. Fitted on the TRAIN folds and stable
across all 8, the true mapping is **`ANGRY=0, NEUTRAL=1, HAPPY=2, SAD=4`**. Assume instead the
corpus authors' own documented column order (`neutral, angry, happy, sad, frustrated`, from
`thaiser-experiments` `score_cols`) and you get **WA 8.56%** — far below the 35% majority-class
floor, with nothing to tell you it is wrong. ⚠ The mapping was fitted on train folds only and
applied unchanged to test: fitting it on test would be choosing the decoding on the scoring data.

⭐ **What this establishes.** Public Thai speech-emotion recognition is not a field with a
state of the art we are competing against — it is a field with **one broken, undocumented,
unbenchmarked checkpoint**. The competitive bar is the corpus authors' published baseline (67.34),
which is a paper, not a downloadable model.

⚠ **Scope, stated because `A59`'s rule applies.** "The only public Thai SER model" means the only
one found on **HuggingFace**, by search and by enumerating 7 Thai AI orgs. GitHub, ModelScope and
university-hosted models are **unchecked**. Ask him before writing "the only" in a thesis.

---

## ✅ F26. There is now a checkpoint — the emotion result stopped being a recipe

Until 2026-09-26 **every emotion number in this repo came from weights held in a Python variable
and discarded when the process exited.** `F18`'s 0.8488, `F23`'s 77.18 — all reproducible from
script, seed and config, and none of them an artifact anyone could be handed. "Our model" was not
a true description of anything on disk.

`adapter/probe_thaiser_protocol.py --save-checkpoint` now writes
`checkpoints/thaiser_head_3s_v1.pt` (51 KB), and it is deliberate about **which part is ours**:

| | |
|---|---|
| backbone | `emotion2vec/emotion2vec_plus_large`, **unmodified, frozen** — not ours |
| ours | the 4-class head (`4 x 2048` + bias) and the feature standardisation it expects |
| carried in the payload | backbone id, class order, mean/std stats, the full protocol, the scores, and the caveats |

⭐ The artifact being **51 KB** is the honest summary of the contribution: a linear read-out on a
public 164 M-parameter encoder, plus the protocol discipline to know what it is worth.

⚠ The payload embeds its own warning: these are **acted studio speech** scores (`F20`: this model
family mislabels natural read speech), with a measured **3.2 WA / 6.0 UA** drop across microphone
channels (`F23`). 🔴 And the **full fine-tune** — `F18`'s best rung, and the better model — still
has no checkpoint and has never been run under their protocol.

---

# PART 3 — The "beat Typhoon in real time" line (2026-10-03 → 2026-10-05)

His direction, in two steps: *"the asr model have better score than the best typhoon model right now"*,
then *"make the a model that beat typhoon best score that can be use in realtime."* Full record, with
every run and its caveats: `idea-stage/IDEA_REPORT.md`. All CERs below are the 682 real
`thai-contextasr-bench` utterances, condition `none`, the benchmark's frozen scorer, under the
**canon normaliser** (`adapter/eval_canon.py`: one function on reference AND hypothesis — digit runs
→ Thai number words, separators dropped) unless marked "own tokeniser".

## ⭐⭐ F27. Eight public Thai ASRs, voted, beat the best Typhoon — but only without hand-picking

Token-level ROVER over all 8 public systems (`adapter/pilot_rover.py`): **CER 0.0678 vs
typhoon-whisper-large-v3 0.0735 (−7.8%)**, paired bootstrap P = 0.0015, wins in all three source
groups. Two traps were walked into and caught:

- **Convention.** A hand-picked 5-system vote looked **−20%** on raw CER; most of it was the vote
  landing on digits, the corpus typist's convention, against Typhoon's number words (`F2`/`F17`).
- **Selection on the test set.** Choosing the best of 219 pools on two source groups and scoring on
  the third (`adapter/pilot_rover_cv.py`) **loses** to Typhoon. Only "every system, no selection" wins.

Zero-shot LLM correction on top (qwen3.5:9b, `adapter/pilot_ger.py`) reached 0.0640 and beat the bar
on all four metrics, but on a test-selected pool, and its gain sits on read speech only. A
prefix-completion probe found **0 of 120** benchmark sentences memorised. Neither system is real-time.

## 🔴 F28. Distilling that teacher into a real-time student failed four times — the data was not fresh

10,199 YODAS2 Thai segments (13 h, CC-BY-3.0) labelled by all 8 teachers; students
typhoon-asr-realtime (3 runs) and typhoon-whisper-turbo (decoder-only): **0.0943 / 0.0940 / 0.0978
vs base 0.0945; turbo 0.0816 vs base 0.0803.** Two causes, both measured:

- **Round 1's filter removed the teaching signal**: confidence over all 8 systems *including the
  student* meant "the student already agrees" (3.4% CER from its own labels).
- **YODAS is not fresh to Typhoon**: on held-out videos, against uploader captions, base turbo
  **0.074** beats the 8-model teacher **0.079** (large-v3: 0.147). Learning from a worse teacher can
  only pull it down.

Consistent with the literature (`A63`–`A65`): every published distillation win starts from a WEAK
student with hundreds+ of hours; over a 10,000 h baseline, fresh pseudo-labels give 2–11%.
🔴 **Rule: before training on any corpus, check the model under training has not already seen it** —
the freshness gate, which later caught NECTEC memorising Porjai (`F30`).

## ⭐⭐ F29. His N-best picker works — the ceiling is in DIVERSITY, the gain is in a TRAINED picker

His idea (*"pass top possibility and pass it to llm to pick the right context"*), then *"try both"*.

| ceiling (perfect picker, own tokeniser; bar reads 0.0838) | sentence | per spot |
|---|---|---|
| turbo top-5 | 0.0810 | 0.0785 |
| turbo + typhoon-asr-realtime top-5 | 0.0717 | 0.0636 |
| + NECTEC top-5 | — | read 0.0373 / talk 0.0779 |

One model's top-5 are near-copies of one hearing; a second model hears differently (ตลาด**วโรรส**).
Voting rules lose to turbo alone; a zero-shot 9B picker only ties it. **Trained pickers**
(Qwen3-0.6B, digits-only structured output, `adapter/train_picker.py`):

| picker | sources | trained on | benchmark CER | talk | read |
|---|---|---|---|---|---|
| p1 | turbo + realtime | Porjai 16k | 0.0786 | — | — |
| p3 | + NECTEC | synthetic 4,214 | 0.0727 | **0.1287** | 0.0657 |
| p4 | + NECTEC | synthetic + Porjai | **0.0681** | 0.1344 | 0.0598 |
| *NECTEC alone* | | | *0.0676* | *0.1379* | ***0.0583*** |
| *bar* | | | *0.0735* | *0.1428* | *0.0618* |

⭐ **A picker trained on synthetic TTS audio transfers to real speech** and is the best system of all
on conversation (−10% vs the bar). Real read speech in training moves it toward NECTEC and the net is
a tie with NECTEC alone at 3–4× the latency. Doubling synthetic data: −3.2% only.

## ⭐⭐ F30. A public 115M real-time model already ties the best Typhoon — NECTEC's

`nectec/Pathumma-stt-th-fastconformer-rnnt-large` (2026-09-06, CC-BY-4.0, NeMo FastConformer-RNNT),
found by enumerating 11 Thai orgs on HF: **CER 0.0676 at beam 5, 113 ms/turn; 0.0695 greedy, 109 ms**
vs typhoon-whisper-large-v3 0.0735 and typhoon-asr-realtime 0.0945. 🔴 **A tie, not a win**: paired
bootstrap P(not better) = 0.084 (CI of the difference [−0.0097, +0.0015]); worse on entities (B-CER
0.213 vs 0.180, recall 0.374 vs 0.408).

Contamination checked as far as possible: the benchmark's `c-` clips are the val/test splits of
`shunyalabs/thai-speech-dataset` (public, no licence stated); **0 of 426 sentences occur in its train
split**; NECTEC's CER there (0.058) is 4× its CER on Porjai, which it clearly memorised (0.015). Not
excluded: training on Shunya val/test audio — only NECTEC can say.

✅ **Deployed**: `asr/run.ps1` serves it by default since 2026-10-05 (`-Model typhoon` reverts);
over HTTP 48/50 transcripts match offline, 133 ms round trip. It writes digits and composed SARA AM.

## F31. Four infrastructure defects, found by running, not reading

| defect | effect | fix |
|---|---|---|
| NeMo opens manifests with `open()` and no encoding | cp1252 on Windows dies on the first Thai byte | `PYTHONUTF8=1`; `finetune_rt.py` refuses to start without it |
| `warprnnt_numba` under numba 0.67 | "Signature mismatch" on training step 1 | — |
| NeMo's pure-PyTorch RNNT loss | correct but **100 s per batch** (16 h/epoch) | `adapter/_rnnt_fast.py`: u-recursion as one `logcumsumexp` per frame — matches NeMo to 2e-6, **0.86 s** |
| Qwen3's 151k-vocab logits in picker training | VRAM overflow spills to RAM → session reaped | LM head on answer positions only + gradient checkpointing → 5.7 GB |

---

# PART 4 — The "DeepSeek for Thai ASR" line (2026-10-06 → 08)

His direction: *"like the come of deepseek — cheaper and can perform like the high-end model."* Full
record: `idea-stage/IDEA_REPORT_2026-10-06.md`; narrative: `NARRATIVE_REPORT.md`. The bar is
typhoon-whisper-large-v3 (1.55B). Benchmark numbers: 682 real utts, canon scorer, greedy.

## ⭐⭐ F32. Fine-tuning on a fresh corpus teaches its TYPIST — rewrite the labels into the base model's convention first

NECTEC on 16.6 h of LOTUSDIS (fresh by the gate) with its own labels: in-domain test **−25%**,
benchmark **0.0695 → 0.0913 (+31%)**, monotone in the interpolation weight — not forgetting. LOTUSDIS
writes loanwords in Latin script (27,585 letters vs NECTEC's 678 on the benchmark), spoken spellings
(เค้า 1,490× vs 5; มั้ย 1,601× vs 3; นึง 666× vs 4), never ๆ, and fillers. `adapter/lotusdis_convention.py`
rewrites labels into **what the base model already emits** (rules read from its outputs, not from test
references): the same recipe then lands at 0.0715 (α .75) and α 1 at 0.0762. This is `F2`/`F17`
arriving through training labels. 🔴 **Rule: before training on any corpus, diff its label convention
against the base model's output convention, not just its freshness.**

## ⭐ F33. 115M ties 1.55B on two sets; 6 GPU-minutes buys ~5% on long-form media — borderline

| vs typhoon-whisper-large-v3 | benchmark (P) | TVSpeech 570 clips (P) |
|---|---|---|
| NECTEC 115M | −4.4% (.08) tie | +3.3% (.85) tie |
| + 1 epoch conv-LOTUSDIS, α .75, seed 1 | −4.0% (.11) tie | −4.8% (.054; Holm .108) |
| same, seed 2 | −3.8% (.12) tie | **−5.9% (.023; Holm .046)** |

One epoch = ~6 min on an RTX 5060 Ti. The media gain replicates in direction and size; significance
is borderline. Both seeds lower benchmark name recall (0.367 → ~0.35). A pooled two-set test
(P = 0.026) was computed after the per-set results were seen and is **withdrawn** (Stage-3 review).
🔴 The bar on TVSpeech is **0.0785 only when the audio is cut at pauses into ≤ 15 s pieces**: in one
30 s window typhoon-whisper's decoder runs out on Thai and cuts 170/570 clips mid-word (0.2004).

## ⭐⭐ F34. The name splice works ACROSS models — NECTEC + Typhoon's CTC spotter beats the 0.78B list model

D6 needs no shared encoder or tokenizer: spots are placed by frame index (both models 0.08 s) and
inserted as text. bias@50 vs dis@50 (same-size decoys), canon:

| | CER | U-CER | B-CER | recall |
|---|---|---|---|---|
| NECTEC + splice, dis@50 | 0.0726 | 0.0635 | 0.2765 | 0.349 |
| **NECTEC + splice, bias@50** (pads 1:2) | **0.0727** | 0.0655 | **0.2238** | **0.574** |
| typhoon-asr-qwen-0.6b-ctx, bias@50 (0.78B) | 0.0791 | 0.0704 | 0.2628 | 0.472 |

Better than the vendor's list-reading model on all four (unpadded version: CER and recall
P < 0.0005). The first run's B-CER did not fall at all — the splice left old-word tails beside the
inserted name; pads (frames claimed before/after a spot, chosen on the synth split) fixed it at a
U-CER cost of +0.003. **Offline**: two encoders + a Python Viterbi ≈ 350 ms/turn vs 113 ms, and it
failed its pre-registered C1–C3 (two of which wrongly compared bias@50 with Whisper's `none`).
Next: a CTC head on NECTEC's frozen encoder (one pass).

## F35. What did not work, and four harness defects

**Killed:** ROVER of small models (never beats NECTEC — it *is* the consensus); routing uncertain
turns to Whisper (oracle −17%, NECTEC's own confidence finds −0.7%); sequence-level RL (pre-registered
gate, best SFT 0.0686 > 0.066); GPU boosting tree on NECTEC (+0.03 recall, CER up — `F8b` again);
**synthetic entity audio** (7.9 h: synthetic held-out −72%, real benchmark flat, recall ↓ — it learned
the TTS voice, not the names; it helps a picker, `F29`, not an ASR).

| defect | effect | fix |
|---|---|---|
| `pin_memory=True` in `finetune_rt.py` | ~25 MB/step leak → 30.9 GB, commit limit, 10× slowdown, shells reaped | `pin_memory=False`, no CUDA-graph decoder in training; per-epoch saves |
| one 30 s Whisper window on Thai | 170/570 TVSpeech clips cut, bar 0.2004 | `fw_manifest.py --vad-max-s 15` (timestamps and `max_new_tokens` both tried, both no effect) |
| `eval_canon.py` hardcoded `none` | every bias@N / dis@N file scored as zero rows | scores each file's own condition |
| splice by token start frame only | old-word tails survive → B-CER flat | `--pad-start/--pad-end` |

## ⭐ F36. The name splice in ONE encoder pass — a 2.6M-parameter CTC head on NECTEC's frozen encoder

`adapter/ctc_head.py`: NECTEC's encoder frozen (eval mode; the transcript is byte-for-byte NECTEC),
features cached once (2.05 M frames ≈ 45 h: conv-LOTUSDIS + Porjai), then a 512→1024→2049 head
trained with CTC in minutes. Held-out-speaker greedy CER of the head **alone: 0.109** (NECTEC's own
decoder: 0.107) — the encoder already "knows" the text; the head only reads it. Chosen on val
(linear 12 ep 0.145, linear 30 ep 0.129, MLP 30 ep 0.109). Spotting = Typhoon's released lattice
spotter, unchanged, on the head's posteriors (`arm_d6_splice.py --spotter head`), τ = −1.5 chosen on
the synth split, pads 1:2 (`F34`). Real split, canon:

| bias@50 | encoders | CER | U-CER | B-CER | recall |
|---|---|---|---|---|---|
| one-pass head, dis@50 (decoys) | 1 | 0.0737 | 0.0626 | 0.3245 | 0.360 |
| **one-pass head, bias@50** | **1** | 0.0748 | 0.0682 | **0.2139** | 0.567 |
| two-model splice (`F34`) | 2 | 0.0727 | 0.0655 | 0.2238 | 0.574 |
| typhoon-asr-qwen-0.6b-ctx (0.78B) | 1 | 0.0791 | 0.0704 | 0.2628 | 0.472 |

Paired bootstrap (5,000): beats the 0.78B vendor model on CER (P = 0.0002) and recall (P < 0.0002);
recall over own decoys P < 0.0002, CER vs decoys a tie (P = 0.16); vs the two-model splice recall equal
(P = 0.69) but **CER worse** (P = 0.996 that it is not lower) — the head accepts more borderline spots,
costing U-CER +0.003. **+91 names over decoys from one encoder pass.**

**Latency** (`adapter/latency_onepass.py`, 677 turns, mean 9.7 s, batch 1): NECTEC alone 45 ms p50;
one pass + head + spotting **192 ms p50 / 302 p90, of which 154 ms is the Python keyword Viterbi**
over 50 entries. The second encoder is gone; what is left is CPU code, not model — vectorising the
spotter is the remaining real-time work.

🔴 **A NeMo 3.0.0 defect, found because the first run was wrong:** `ASRModel.transcribe()` returns
with `model.training == False` but **`model.encoder.training == True`**. A direct encoder call after
it runs dropout and **updates BatchNorm running statistics** — the first head run silently drifted
NECTEC's own transcript in 481/682 utterances (CER 0.0695 → 0.0972 with almost no spots). Fix: put
the encoder and preprocessor in eval before any direct call; verified 80/80 transcripts identical.
That run is quarantined in `results/_invalid_ctchead_h3_run1/`. Any arm that mixes `transcribe()` with
direct module calls is exposed.

## ⭐⭐ F37. The one-pass name splice runs at NECTEC's own speed — and the A2 tune does not stack with it on the benchmark

**Real-time, verified.** `adapter/kws_fast.py` ports Typhoon's keyword Viterbi to numba line by line
(same quirks, same float32/float64 arithmetic): **0 mismatches in 3,503 real (posterior, keyword)
pairs** (`--check`), and the full real bias@50 run with `--fast-spot` is **byte-identical** to the slow
one. On a 9.7 s turn, 50 keywords: 187 ms → 11.5 ms. End to end (`latency_onepass.py --fast-spot`,
677 turns, batch 1): **one encoder pass + RNN-T greedy + CTC head + spotting = 42.9 ms p50 / 65 ms p90,
spotting 8.8 ms**, against NECTEC's own `transcribe()` at 43.0 / 62 ms in the same run (that call
carries NeMo's per-call overhead, which the direct path skips). **The `F36` system — better than the
0.78B vendor list model on all four metrics — now costs no more time per turn than NECTEC alone.**

**A2 + names (head on the conv-LOTUSDIS encoder, `ctchead_a2h3`, val greedy 0.098; τ = −1 chosen on
synth).** Real split, canon:

| bias@50 | CER | U-CER | B-CER | recall |
|---|---|---|---|---|
| base NECTEC + head (`F36`, τ −1.5) | 0.0748 | 0.0682 | **0.2139** | **0.567** |
| A2 + its head (τ −1) | **0.0741** | **0.0661** | 0.2427 | 0.519 |
| A2 + its head, dis@50 | 0.0723 | 0.0620 | 0.2950 | 0.344 |
| qwen-0.6b-ctx (0.78B) | 0.0791 | 0.0704 | 0.2628 | 0.472 |

vs base+head: CER tie (P = 0.20), **recall lower** (P = 0.9998 not higher); vs qwen-0.6b-ctx: CER and
recall still better (P < 0.0002 / 0.0008); vs own decoys: recall +0.175. Partly a different τ (the synth
sweep picked a stricter one for A2), partly A2's lower native name recall (`F33`). **On this benchmark
the two improvements do not stack**; A2's advantage is long-form media speech (TVSpeech, `F33`), which
has no bias lists to test names on. Product reading: base NECTEC + head when a list exists; A2 is a
separate choice for media-style audio.

## ⭐⭐ F38. NECTEC's ear is better than its score — ~40–50% of its error is writing, ~30% is names, tone is 8%

Measured 2026-10-08 for the "invent an ASR architecture from zero" question, so the design would be
aimed at real errors (`adapter/probe_error_anatomy.py`, `results/anatomy_nectec_metrics.json`;
682 real utts, own Levenshtein after `canon()` + SARA AM fold, Latin entity spans replaced by their
annotated Thai form — own-scorer CER 0.0778 vs the benchmark scorer's 0.0676; read the SHARES).

| slice of NECTEC's 5,566 character edits | share |
|---|---|
| same loanword in Latin vs Thai script (`wifi`/`ไวไฟ`, `maria`/`มาเรีย`) | **23.1%** |
| right sound, wrong spelling (vanish under royin romanisation, pure-Thai utts) | **21.5%** (crude consonant-class key: 17.9%) |
| inside named-entity spans (5.7% of the text) | **29.0%** |
| tone mark only | **8.3%** |
| particles / fillers | 4.5% |
| edits in bursts ≥ 4 chars (whole words) | **61.1%** |
| 5-best ORACLE over NECTEC's own beam | CER −4.6% only |

**What it changed:** an explicit tone branch is capped at ~8% (the Mandarin 30% oracle does not
transfer); re-ranking one model's beam is capped at ~5%; the big targets are *writing* (script +
spelling, fixable from text) and *names* (fixable by memory). This is the basis of the Ear–Memory–Pen
design in `idea-stage/IDEA_REPORT_2026-10-08.md` — a design, **not yet a result**. ⚠️ The 23% script
slice is partly the benchmark's own inconsistency (references mix Latin and Thai for loanwords), so a
script-blind metric must come before any model claim. ⚠️ Royin romanisation drops tone and vowel
length, so 21.5% is an UPPER bound on "spelling, not hearing".

> 🔴 **F38 CORRECTED by F39 (same day):** the "~20% right sound, wrong spelling" slice was an artifact of
> the coarse sound keys (royin and the consonant-class key drop tone and vowel length). With a TONED G2P
> key, Thai spelling is almost fully determined by sound (E1 below), and a homophone re-speller gains
> nothing (E3). Those errors are hearing errors — tone and vowel length — not writing errors.

## 🔴 F39. Ear–Memory–Pen, tested: the Ear loses to the draft, the Pen has nothing to fix — but the sound metric reorders the field

His *"do it"* on the `F38` design (2026-10-08). Every step was pre-registered with a stop condition in
`idea-stage/IDEA_REPORT_2026-10-08.md` §7. One GPU job at a time; all on the 682 real utts.

**E0 — script-blind + sound metric** (`adapter/eval_soundblind.py`, new venv `E:/thainlp-spike`, tltk G2P;
`results/eval_soundblind_metrics.json`). Script-blind scoring removes ~10% of CER **for every system alike**
(NECTEC 0.0778 → 0.0701), so the CER ranking barely moves. The **syllable (sound) error rate reorders it**:

| system | CER | SylER-sb |
|---|---|---|
| Pathumma-whisper-large-v3 | 0.0814 | **0.1145** |
| NECTEC + LOTUSDIS-conv (A2) | 0.0785 | 0.1149 |
| NECTEC | **0.0778** | 0.1293 |
| biodatlab-whisper-large-v3 | 0.1096 (9th) | 0.1371 (4th) |
| typhoon-whisper-large-v3 | 0.0808 | 0.1534 |

⭐ "Who hears best" ≠ "who spells best". **A2 hears 11% better than base NECTEC at a CER tie** — consistent
with its TVSpeech gain (`F33`), which CER on the benchmark could not show. ⚠️ `same_word` (Latin↔Thai
match) is a heuristic with visible false accepts (`moneyback ~ มันไอ้แบล็ค`); SylER carries tltk G2P noise.

**E1 — how ambiguous is sound → spelling** (`adapter/syllable_labels.py ambiguity`, 371,505 word tokens of
LOTUSDIS + Porjai training text; `results/p2g_ambiguity_metrics.json`). 21.6% of tokens have a TONED sound with
2+ spellings, but writing the most frequent spelling per sound is wrong on only **1.25%** of tokens. Drop tone
and 52.9% become ambiguous. **Thai writing is nearly a function of toned sound.**

**E2 — the Ear** (`adapter/syllable_head.py`, `checkpoints/sylhead_s2.pt`: 1024-hidden head, 4,000 toned
syllables + `<unk>`, frozen NECTEC encoder, cached features). Val syllable error 0.106 (LOTUSDIS held-out
speakers). On the benchmark, against the reference's G2P: **head 0.164 vs G2P(NECTEC's own transcript) 0.115**
(A2 0.102, Pathumma-whisper 0.0965). **Stop condition met: the Ear reads the sound worse than the draft
implies.** First head (s1, 3 tokens/syllable) failed for a mechanical reason — ~15 tokens/s vs NECTEC's
12.5 frames/s, CTC could not fit it (val 0.38).

**E3 — the Pen** (`adapter/pen_lattice.py` + `adapter/pen_score.py`; homophone dictionary from pythainlp's
word list + training transcripts only; Qwen3-0.6B sentence log-prob; switch threshold cross-fitted on
two halves). Every threshold made CER worse (0.0778 → 0.0790–0.0830); cross-fit chose "never switch" on both
folds: **0 utterances changed**. With the Ear's alternatives added (86 more slots): worse at every threshold.

**E4 — the Memory, ceiling only.** NECTEC misses 284 of 439 entity mentions. pythainlp's 81k names/words
contain 59 of them; **Thai Wikipedia's 305,554 titles contain 85 (30%)**. Not built: spotting a 305k catalog
per turn needs a retrieval shortlist (`A87`) the box does not have, and the 50-name spotter already costs 8.8 ms.

**What it changed:** the "decouple hearing from writing" architecture is **falsified for Thai on this box** —
the writing half has almost nothing to do once tone is heard. What survives: (1) the sound-level metric as a
diagnostic, which separates hearing from spelling and shows A2 genuinely hears better; (2) names remain the
largest addressable slice, and the lever is catalog-scale retrieval, not a new decoder.

## 🔴 F40. Six real-time ways to beat NECTEC without a name list — all lose on the benchmark

His directive (2026-10-08): *"keep loop doing research, do testing until do success."* Success was fixed
first: real-time (≤ ~2× NECTEC), **no list given**, official canon CER below NECTEC's **0.0695** with
paired-bootstrap P < 0.05 (`adapter/paired_vs.py`, 10k resamples over utterances). Each arm's control
reproduced NECTEC byte-for-byte (682/682) before its number was read.

| # | idea | arm | official CER | P(not better) | why it lost |
|---|---|---|---|---|---|
| 1 | catalog → auto name list → one-pass splice (`catalog_retrieve.py`, Thai Wikipedia 243,695 titles) | `loop1_catalog_k50` | 0.1736 | 1.0 | 1,517 false spots; only 15 of 299 missed names were even retrievable from the draft |
| 2 | session memory (earlier turns' names as the list) | — | — | — | ceiling: **3** of 299 missed names appear correctly earlier in the session |
| 3 | session test-time adaptation, encoder LayerNorms, entropy of the CTC head, causal (`tta_session.py`) | lr 1e-4 × 3 / 1e-5 × 1 | 0.0789 / 0.0713 | 1.0 / 0.90 | entropy minimisation drifts; worse the harder it is pushed |
| 4 | step-synchronous RNN-T ensemble, base + LOTUSDIS tune (`rnnt_ensemble.py`) | `loop4_ens_b_l2a100` | 0.0821 | 1.0 | a fine-tune emits tokens on different frames; per-frame averaging splices half-tokens |
| 4b | ROVER over base + two LOTUSDIS tunes (`pilot_rover.py --add`) | `loop4b_rover_nec_l2_l3` | 0.0718 | 0.96 | both tunes write LOTUSDIS's convention; the vote drifts with them |
| 5 | A2's ear + NECTEC's pen: keep NECTEC where the two sound the same (`ear_pen_merge.py`) | `loop5_merge_nec_a2` | 0.0713 | 0.61 | 532 of 567 differences are real SOUND differences, which are not net-better |

**What it changed:** within the 115M NECTEC family and the data on disk, every real-time recombination
lands at or above NECTEC on this benchmark. The per-utterance oracle says the room is real (base + the
LOTUSDIS tune −11.7%, base + Pathumma-whisper-large −20%), but no label-free selector found it. ⚠️ The
diversity that exists is in models outside the real-time budget (Pathumma-whisper-large-v3) or in data
we do not have. ⚠️ Arms 3–5 had one pre-registered setting each plus at most one exploratory; none was
tuned on the benchmark.

## ⭐ F41. Three ears in 481 ms: NECTEC drafts, two large Whispers verify, ROVER votes — passes the loop's criteria, with a selection caveat

After `F40` he relaxed the budget to ~300–500 ms (2026-10-08). Success criteria, fixed before any run:
p50 ≤ 500 ms/turn, no list, official canon CER < NECTEC's 0.0695 with paired P < 0.05, TVSpeech not worse.

**The architecture.** NECTEC transcribes first (~43 ms) and its text is the DRAFT for two large Whispers
(Pathumma-whisper-th-large-v3, typhoon-whisper-large-v3) that run concurrently in two worker processes and
**speculative-decode** against it (`adapter/spec_whisper.py`): encode once, verify the whole draft in one
decoder pass, take Whisper's own token at the first mismatch, re-sync the draft by suffix match, KV cache
cropped on rejection. Each emitted token is Whisper's own greedy argmax, so it is lossless (39/40 identical to
`generate`; the 40th differs by a trailing space). Then ROVER over the three (`pilot_rover.rover`, uniform).

| | latency / turn | benchmark official CER | P(not better) | TVSpeech CER |
|---|---|---|---|---|
| NECTEC alone | 43 ms | 0.0695 | — | 0.0811 |
| Pathumma-L greedy (CTranslate2) | 1,027 ms | 0.0727 | — | 0.1159 |
| typhoon-L greedy (CTranslate2) | 1,044 ms | 0.0737 | — | 0.0785 |
| Pathumma-L int8 | 952 ms | 0.0740 | — | — |
| Pathumma-L speculative (HF, NECTEC draft) | 264 ms (p90 523) | — | — | — |
| **3-ear system, end to end** (`pipeline_latency.py`) | **481 ms p50** (p90 908) | **0.0652 (−6.2%)** | **0.013** | **0.0803 (−0.9%, P .36)** |

Entity recall 0.3667 → 0.3690. Within 500 ms *without* the large Whispers nothing passed: ROVER of
{NECTEC, A2, turbo, distill-medium} −1.5% (P .54).

🔴 **The caveat that decides how to quote it:** the pool {NECTEC, Pathumma-L, typhoon-L} was chosen after
scoring ~15 ROVER pools on this benchmark. A Bonferroni correction over all of them puts P near 0.2, so the
benchmark gain is **not robust to the selection**, and the held-out set (TVSpeech) shows a tie, not a gain.
Quote it as: *meets the pre-set criteria; the benchmark gain is unconfirmed on a second set.* ⚠️ p90 is
908 ms — over budget on long or disagreeing turns. ⚠️ TVSpeech accuracy uses CTranslate2 greedy with
≤15 s VAD cuts (`fw_manifest.py --vad-max-s 15`), not the HF speculative path; the latency was measured on
the benchmark only.

## 🔴 F42. Toward CER 0.060 (−14% vs NECTEC) at ≤500 ms: the room exists, no judge finds it

His target (2026-10-08): official CER **0.060** on the 682 real utts, still ≤500 ms/turn. Protocol, fixed
first and committed (`494d951`): the 34 recording sessions split into **dev** (14 sessions, 339 utts —
including the 240-utt session) and a locked **test** half (20 sessions, 343). Tune on dev only; the test
half is scored once (`paired_vs.py --half test --final`, every use logged to `results/test_evaluations.log`).
Because the halves differ in difficulty (NECTEC dev 0.0506), the target is the RELATIVE cut: test CER ≤
0.863 × NECTEC's. **The test half has not been scored.**

A harness fix came first: `adapter/official_utt.py` gives per-utterance errors from the benchmark's own
`score_utterance` (accepts Thai OR Latin entity forms) and sums exactly to the official CER (0.0695, 0.06517
reproduced). A plain Levenshtein had shown a selection at −4.4% that was −0.7% officially.

| dev-half result (official, vs NECTEC 0.0506) | rel | P(not better) |
|---|---|---|
| per-utterance ORACLE over {NECTEC, Pathumma-L, typhoon-L} | −24.5% | (ceiling) |
| … over all 12 available ears | −42% | (ceiling) |
| 3-ear ROVER (`F41` system) | −3.1% | 0.76 |
| ROVER + Qwen3-0.6B settling disagreement regions (`rover_lm.py`, γ/δ grid) | worse at every setting | — |
| sentence-level product of experts: every ear scores every candidate (`poe_rescore.py`) | −2.3% | 0.26 |
| best single ear (lot3_a75, plain scorer) | −2.9% | — |

**Why the judges fail:** each candidate is some ear's own greedy output, so each ear's likelihood favours its
own text — the "experts" re-cast the same votes. The oracle gap needs information none of the three models
carries about the others' mistakes: a judge TRAINED on many (ears' outputs, truth) pairs from non-benchmark
audio (the `F29` picker line, now with large-Whisper diversity and acoustic-likelihood features), or better ears.

## 🔴 F43. A judge trained on LOTUSDIS learns "trust NECTEC" — because NECTEC has heard LOTUSDIS

The trained-judge route from `F42` (2026-10-08 → 09, `adapter/judge.py`). Training data: LOTUSDIS train+test
clips joined into 4,314 benchmark-like turns (12.7 h, median 10.8 s; never the benchmark); the three ears run
exactly as deployed (NECTEC, then Pathumma-L / typhoon-L speculative on NECTEC's draft); every ear scores
every candidate (+ the vote); a 32-unit MLP picks one, trained on expected errors, early-stopped on 10% of
the LOTUSDIS turns (`checkpoints/judge_v1.pt`).

| | LOTUSDIS held-out turns | benchmark DEV half (official) |
|---|---|---|
| always NECTEC | 0.1195 | 0.05059 |
| always ROVER | 0.1325 | — |
| per-turn oracle | 0.0996 | (−24.5%) |
| **trained judge** | **0.1099 (−8%)** | **0.05058 (−0.0%, P .50)** — picks NECTEC 226/339 |

**Why:** on LOTUSDIS NECTEC is the best single ear by a wide margin (0.1195 vs typhoon-L 0.1481, Pathumma-L
0.1697) — it has likely seen part of LOTUSDIS train (`F30` follow-up), the Whispers have not. The judge learns
that prior and carries it to the benchmark, where NECTEC's lead is much smaller. The freshness gate
(`CLAUDE.md`) applies to a JUDGE's training data too: it must be fresh to EVERY ear it judges, not only to one.
The test half was not scored. Also recorded: the run was reaped three times for low system memory; the
scoring step held all audio in RAM — fixed to stream and checkpoint per model.

## 🔴 F44. Fresh data does not save the judge — its skill does not transfer to the benchmark

Follow-up to `F43` (2026-10-09). GigaSpeech2-th dev+test (`speechcolab/gigaspeech2`, CC-BY-4.0, 16,928 human-
refined clips, 20 h) passed the freshness gate for ALL three ears on 300+300 clips (dev: NECTEC .122,
Pathumma-L .120, typhoon-L .113; test: .148/.148/.146 — none near-perfect, none dominant). Joined into 7,099
~10 s turns by consecutive segments of one video; same pipeline as `F43` (`JUDGE_DATA=gs2 adapter/judge.py`;
ears and scoring one large model per process, scoring in 1,500-turn slices — the box reaped every longer
process for low memory).

| judge | its own held-out (always-best / judge / oracle) | benchmark DEV half, official vs NECTEC 0.05059 |
|---|---|---|
| LOTUSDIS (`F43`) | 0.1195 (NECTEC) / 0.1099 / 0.0996 | 0.05058 (P .50) — picks NECTEC 226/339 |
| **GigaSpeech2** | 0.0927 (ROVER) / **0.0871** / 0.0753 | **0.05137 (+1.5%, P .64)** — picks typhoon-L 282/339 |
| **pooled, no which-ear feature** | 0.1078 (ROVER) / 0.0997 / 0.0845 | **0.05059 (±0.0%, P .51)** — picks spread 182/100/39/18 |

**What it changed:** each judge learns the best ear OF ITS TRAINING DOMAIN (NECTEC on LOTUSDIS, typhoon-L on
GigaSpeech2) and carries that prior to the benchmark, where the ranking differs. Removing the ear-identity
feature and pooling the two domains removes the prior — and the gain with it: the per-utterance evidence the
judge sees (each ear's likelihood of each candidate, cross-candidate agreement, length) does not predict, on
the benchmark, which transcript is right. Every route to the dev oracle (−24.5%) tried so far — vote, LM,
product of experts, three trained judges — lands within ±3% of NECTEC. **The test half remains unscored.**

## ⭐ F45. A better EAR, not a better judge: 18 h of fresh GigaSpeech2 buys −6.8% on the dev half at NECTEC's own speed

After three judges tied NECTEC (`F43`/`F44`), the lever changed to the ear itself (2026-10-09). GigaSpeech2-th
dev+test, fresh to NECTEC (`F44`), rewritten into NECTEC's convention (`adapter/gs2_convention.py`): SARA AM
composed, whitelisted reduplication → ๆ, **clips dropped where NECTEC writes Latin** (GS2 spells loanwords in Thai;
the benchmark references use MORE Latin than NECTEC — 15.9 vs 9.5 per 1k chars — so training toward Thai script
would move away from them), 20-char leak guard vs the benchmark (313 clips dropped), 5% of videos held out.
15,286 clips / 17.9 h. The A2 recipe: 1 epoch, LR 5e-5, interpolate with the base (`adapter/night_gs2.sh`).

| | GS2 held-out videos | benchmark DEV half (official, NECTEC 0.05059) |
|---|---|---|
| base NECTEC | 0.0743 | 0.05059 |
| GS2 α .25 / .50 / .75 / 1 | .0672 / .0612 / .0550 / .0515 | .04863 / **.04764 (−5.8%, P .008)** / .04764 / .04806 |
| GS2 + conv-LOTUSDIS (35 h) α .25 / .50 / .75 / 1 | .0658 / .0586 / .0545 / .0549 | .04836 / .04782 / **.04716 (−6.8%, P .013)** / .04941 |
| ROVER of 3 tuned variants + Pathumma-L + typhoon-L | — | .04645 (−8.2%, P .002) — ~550 ms, over budget |

**What it changed:** fresh, convention-matched human data moves the ear on the benchmark — the first single
real-time model to beat NECTEC significantly on held-out benchmark sessions, at the same 115M / ~43 ms.
In-domain the full tune keeps improving (−31%), but the benchmark peaks at α .5–.75: past that the model
drifts toward GigaSpeech2's style. Voting NECTEC variants adds little (same family, same errors).
⚠️ α was chosen on the dev half (allowed by the `F42` protocol); **the test half is still unscored**.
⚠️ One seed. TVSpeech not yet measured for these models.

## F46. More fresh data is not automatically better: scripted food orders hurt; elderly speech is not fresh

Search for more fresh human-labelled Thai on Hugging Face (2026-10-09/10; 66 Thai ASR datasets listed). Checked:
`Phonsiri/thai-elderly-speech` (3,840 labelled clips, 5.3 h) — **NECTEC CER 0.0031 → not fresh** (Typhoon 0.03 on the
same clips, so it is trained-on, not merely easy); `KittipatPaisanpudinun/Thai-Food-Ordering-Dataset` (CC-BY, 12.3 h
scripted reads) — fresh (0.059); `AIxBlock` call-center — metadata only, no audio; `ThinkingMachinesDataScience/
Ratchada-STT` — gated (terms not accepted); `hotdogs/thai-speech-20k` — derived from GigaSpeech2.

Food added to GS2 + conv-LOTUSDIS (digits → words, ≤10 s: 3,313 clips / 5.8 h), same recipe, batch 8 × accum 2:
dev half α .25/.5/.75/1 = −2.6% / −4.7% / −5.1% / −4.1% — **worse at every α** than without it (best −6.8%).
Narrow scripted domain pulls the model away from conversational benchmark speech.

**TVSpeech for the `F45` models** (independent set, nothing chosen on it): base NECTEC 0.0811 → GS2 α .5 **0.0713
(−12.1%, P < 0.0001)**, GS2+LOT α .75 **0.0734 (−9.5%, P = 0.0001)** — both now below typhoon-whisper-large-v3 (0.0785).

Harness lessons: (1) dense-text clips at batch 16 overflow 16 GB VRAM and Windows spills the overflow into system
RAM (17 GB working set) — use batch 8 × accum 2 for such data; (2) the low-memory reaper kills the SHELL, not its
python child — an orphaned trainer kept running and starving the box; check for orphans after every reap.

## 🔴 F47. A second ear fused at the ENCODER does not help either — a frozen Whisper-large adds nothing a GS2-adapted block does not, and costs Latin

His ask (2026-10-10): CER 0.05 "by architecture". `F42`–`F44` said text-level combiners cannot reach the ears' oracle
because the evidence of who heard right is gone once each ear has written; the test was to fuse EARLIER
(`adapter/fuse_ears.py`, `idea-stage/IDEA_REPORT_2026-10-10.md`). Pathumma-whisper-th-large-v3's encoder (frozen,
**72 ms** per 30-s window here, last layer pooled ×4 to NECTEC's 12.5 fps — frame counts equal on 654/682) feeds a
2-layer block whose zero-initialised output is added to NECTEC's frozen encoder output; NECTEC's decoder/joint read it.
Trained on cached features of GS2 dev+test (`F45` manifests, fresh to both ears), 4 epochs. Harness control: the
untrained block reproduces NECTEC on **682/682**. Control arm = the identical block with the Whisper input zeroed.

| arm (decoder/joint frozen, LR 5e-5) | GS2 held-out val (NECTEC .0759) | benchmark DEV half, own scorer (NECTEC .05016) | P(not better) |
|---|---|---|---|
| **f2** fusion, Whisper on | **.0530 (−30%)** | .05611 (**+11.6%**) | .996 |
| **c2** same block, Whisper zeroed | **.0524 (−31%)** | .05118 (+1.9%) | .71 |
| f2, residual × α .25 / .5 / .75 | — | +4.8% / +2.3% / +2.3% | ≥ .75 |
| c2, residual × α .25 / .5 / .75 | — | +0.8% / +3.0% / **−2.9%** | .18 at best |
| `gs2lot_a75` (`F45`, for reference) | — | −6.8% (official) | .013 |

**Why:** (1) in-domain the second ear buys nothing over the same block without it — the whole −30% is GS2 adaptation;
a short 1,200-step probe had suggested otherwise (−17.7% vs −1.4%) because its control ran at a different LR, so
the probe was not a control. (2) On the benchmark the fused model writes **60 Latin letters on the dev half where
NECTEC writes 214** (spaces 537 → 27): GS2's references contain zero Latin, and `gs2_convention.py` dropped every clip
where NECTEC writes Latin, so a block trained on GS2 learns "never Latin" — and the Whisper branch, the strongest
learner, learns it hardest. The benchmark references use Latin for loanwords. (3) Probes on the way: updating
NECTEC's decoder/joint on frozen features made GS2 val WORSE (.0759 → .082 in 800 steps); fusion at LR 3e-4 diverged.

**What it changed:** the pre-registered stop fired (fusion not better than its control); the end-to-end version was not
run. Feature-level fusion of a frozen large Whisper, trained on 18 h, does not unlock the `F42` oracle on this box.
The reusable lesson is about the DATA: **GigaSpeech2-th teaches Thai-script loanwords**; any component trained on it
must be blended back (as `F45`'s α does) or given Latin-bearing data, or it erases the Latin the benchmark rewards.
⚠️ One seed; the test half was not touched.

## F48. GigaSpeech2-th TRAIN (machine labels) adds nothing over the 18 h human GS2 — "NECTEC is wrong" selects wrong labels

Loop toward CER 0.05 (`idea-stage/LOOP_2026-10-10.md`), 2026-10-10/11. GS2 train shard 0 (~51 h) passed the freshness
gate (NECTEC .0333, typhoon-L — known to train on it — .0397). NECTEC transcribed all 49.6k clips (`gs2train_prep.py`);
21.5k had any error. All arms = the `F45` gs2lot mix + the extra clips, same recipe (1 epoch, LR 5e-5, bs 8 × accum 2),
α-blended with the base; benchmark DEV half, official CER (NECTEC .05059, `gs2lot_a75` .04716).

| arm | extra data | best α, dev | vs NECTEC | vs `gs2lot_a75` |
|---|---|---|---|---|
| `gs2s0` | 21.5k "hard" + 15% easy (27.9 h) | .25: .04862 | −3.9% (P .009) | +3.1% ❌; α .75 / 1: **+9.4% / +25%** vs NECTEC |
| `gs2s0c` | only the 3,818 hard clips (4.2 h) where Pathumma-L agrees with the label (CER ≤ .02, closer than NECTEC) | .75: .04692 | −7.3% (P .008) | −0.5% (P .41) = tie |
| no training: NECTEC's hyp where it writes Latin, else `gs2lot_a75` | — | .04721 | −6.7% | +0.1% (P .44) = tie |

**Why:** the machine labels were filtered to segments an ASR already agreed on, so on the clips NECTEC misses the
LABEL is usually what is wrong: an independent ear confirms only **18%** of them. Training on the unfiltered set
teaches the label errors (the full-strength model is +25%); the confirmed set is too small to add to 18 h of human
GS2. And the Latin that GS2 erases (`F47`) cannot be bought back by routing: in the utterances where NECTEC writes
Latin, its Thai errors cancel the gain.

**What it changed:** with five tries inside ±1% of `gs2lot_a75`, the data/routing levers on the 115M NECTEC have
saturated at ~−7% on dev; the 0.05 target (−28%) is out of their reach. ⚠️ One seed each; test half untouched.

## F49. Two epochs is the one lever that worked (−3%, two seeds); fourteen other loop tries tie or lose — 115M NECTEC + this data saturates near dev .046

24-hour loop toward dev CER .0364 (`idea-stage/LOOP_2026-10-10.md`, 2026-10-11). Benchmark DEV half, official
per-utterance CER (NECTEC .05059). The test half was not scored (the gate never passed).

| try | dev (best α) | vs best |
|---|---|---|
| **`r2ep`: the `F45` gs2lot mix, 2 epochs instead of 1 (bs 8 × 2), α .75** | **.04561** | **new best: −3.3% vs `gs2lot_a75` (P .042), −9.8% vs NECTEC (P .001)** |
| same, seed 7 (replicate) | .04595 | +0.7% — the 2-epoch gain is real; the 1-epoch seed-7 run was .0489 (seed noise ≈ ±3.5%) |
| 3 epochs | .04672 | +2.4% — 2 is the peak |
| two-seed soup of the 2-epoch tunes | .04598 | tie |
| + 19.6k Pathumma-L-confirmed GS2-train clips (shards 0–4, 21.9 h), 2 epochs | .04862 | +6.6% |
| + YODAS2 uploader captions, two-ear confirmed (3,936 clips, 4.3 h, 280 with Latin) | .04779 | +4.8% |
| + FLEURS-th train (6.7 h, Latin names), 1 epoch | .04706 | (vs gs2lot) tie |
| re-weight the mix: GS2 ×2 / LOTUSDIS ×2 | .04644 / .04768 | +1.8% / +4.5% |
| separate encoder / decoder α (5 splits) | ≥ .04628 | worse than the uniform α .75 |
| malsd beam 4 | .04553 | tie (p50 91 ms) |
| + token 4-gram over 2.47 M GS2-train transcripts (`build_lm_arpa.py`), α .2 / .4 / .6 | .04503 / .0558 / .0897 | −1.3% (P .29), then worse; p50 **622 ms** (no CUDA graphs on Windows) |
| drop read-speech false starts (`restart_fix.py`) | .0443 at a threshold picked ON dev | ≤ 3%, biased; hurts the a half |
| `typhoon-asr-streaming-nemotron-0.6b` (641M, released 2026-09-15) as a base | .0971 | +92% vs NECTEC — real acoustic errors |
| name list (existing `F36` splice): bias@50 vs dis@50 | .05484 vs .05529 | −0.8% CER, recall .42 → .61 |

**Why it saturates:** (1) every dataset ADDED to the 18 h human GS2 + 12 h conv-LOTUSDIS mix — machine-labelled,
confirmed, human-captioned, read speech — costs the conversational "a" half, whatever its labels; so does
re-weighting the mix or training past 2 epochs. (2) The read "c" half (60% of the errors) is dominated by foreign names
the references write in Latin, which the official scorer already accepts in Thai OR Latin for annotated entities —
so even a name list buys < 1% CER. (3) Decode-side levers (beam, n-gram) add ≤ 1.3% and the LM breaks the latency
budget. The best model `checkpoints/r2ep_a75.nemo` is −9.8% dev and −5.7% on TVSpeech (.0765 vs .0811), ~125 ms
p50 — real-time and not worse on TVSpeech, but 25% short of the 0.05 target.

**Harness lessons (each cost a restart):** on Windows WDDM a full VRAM spills into system RAM instead of raising
OOM, so torch's cache never frees and the commit limit dies — `finetune_rt.py --vram-frac` (default .9; .8 for the
47k-clip mixes) and a cap in any long transcription loop; the low-memory reaper kills only the outer shell, so queued
inner loops and python children survive and must be killed before a restart; never wait on a log line an earlier run
may already have written; NeMo checkpoints store decoding enums by NAME ("LATE", "LM_WEIGHTED_FULL"), which the
enum constructors reject once an n-gram LM is attached — lowercase them (`eval_rt_local.py`).

## ⭐ F50. On the PUBLIC test sets Typhoon reports, our 115M models beat every open Thai model we ran — and no test sentence was in our training (2026-10-12)

**Question (his):** read the published Typhoon and NECTEC ASR papers, see which test sets they use, compare, and make sure that test
data was not in our training.

**What they test on.** Typhoon (arXiv 2601.13044 "Typhoon ASR Real-time", 2609.14991 "Typhoon ASR Streaming", all `typhoon-ai/*`
cards): the **Typhoon ASR Benchmark** = `gigaspeech2-typhoon` (1,000 utts; its ids are GigaSpeech2-**train** ids, 1000/1000 found in
`train_refined.tsv`) + **TVSpeech** (570 utts, 3.75 h) + **FLEURS-th test**, plus an Isan held-out set. NECTEC: the FastConformer card
reports **no** numbers; Pathumma-whisper-large's card has a commented-out WER table on CV18, Gowajee, LOTUS-TRD, Thai Dialect, Elderly,
GigaSpeech2, FLEURS, Distant Meeting, Podcast. Neither paper states a train/test decontamination for its own models.

**Leak audit** (`LOOP` row 16; normalised text, exact + 20-char runs + longest common substring; GS2 ids). Best models' training
(GS2 official dev+test + LOTUSDIS-lav train + val): Typhoon GS2 bench **0/1000 ids**, longest shared run 25 chars; TVSpeech 0 exact,
longest 36 of 284 chars; FLEURS test 0 exact, 25 chars — stock phrases, not sentences. 🔴 **The GigaSpeech2 OFFICIAL test split IS in our
training** (by design, `F45`): never report it. The benchmark's own read-speech clips reuse FLEURS sentences — **dev half from
FLEURS-dev (209/240), test half from FLEURS-test (127/186)** — so tuning on the dev half never saw a FLEURS-test sentence.

**Scores** (one scorer, `eval_manifest` = `gate_lotusdis.norm`, pooled CER, one window 2026-10-12 00:47–01:07; Whispers with VAD ≤ 15 s
past 28 s; paired bootstrap 2,000):

| model | params | TVSpeech (570) | FLEURS-th test (1,021) |
|---|---|---|---|
| **gs2a_a50** (ours, GS2-only α .5) | 115M | **.0713** | **.0595** |
| **r2ep_a75** (ours, dev-best) | 115M | .0765 | .0612 |
| typhoon-whisper-large-v3 | 1.55B | .0785 | .0690 |
| NECTEC FastConformer (our base) | 115M | .0811 | .0617 |
| typhoon-asr-realtime | 115M | .0828 | .0892 |
| Pathumma-whisper-large-v3 | 1.55B | .1159 | .0618 |

gs2a_a50 beats every model on both sets (P ≤ .001) except Pathumma-whisper on FLEURS (−3.7%, P .096). r2ep_a75 beats NECTEC, typhoon-RT and
Pathumma-whisper on TVSpeech, typhoon-whisper-large and typhoon-RT on FLEURS; ties typhoon-whisper-large on TVSpeech (−2.4%, P .21) and
NECTEC/Pathumma on FLEURS. **The dev-half winner is not the public-set winner**: gs2a_a50 (dev −5.8%) generalises better than r2ep_a75
(dev −9.8%) — the extra LOTUSDIS + second epoch fit the benchmark's conversational style. ⚠️ gs2a's α was picked looking at TVSpeech
among others (`F45`); FLEURS test is the one fully fresh comparison. ⚠️ Typhoon's published CERs use their own normaliser (TVSpeech:
typhoon-whisper-large 6.32, typhoon-RT 9.99, Pathumma-whisper 10.36; FLEURS orig: 9.98 / 13.87 / 6.29) — compare models only within
one scorer. Not covered: the Typhoon GS2 bench (its audio sits in 191 GS2-train shards, we hold 6), commercial APIs, NECTEC's own
training data (unknown — it could contain any public set).
