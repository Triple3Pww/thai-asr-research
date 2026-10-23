# Idea Discovery Report — a "DeepSeek moment" for Thai ASR

**Direction** (his words, 2026-10-06): *"I want our project to be like the revolutional of asr(stt)
like the come of deepseek that cheaper and can perform like the highend model."*
**Run**: `deepseek-asr-2026-10-06` (pipeline) / `idea-deepseek-asr-2026-10-06` (this stage) ·
**Executor**: claude-opus-5-5 · **Reviewer**: Gemini 3.8 Flash via `gemini-review` (no Codex MCP on this box)
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine

> 🔴 A plan and a pilot record, not a finding. The earlier run (2026-10-03..05, the one `F27`–`F30`
> cite) stays in `IDEA_REPORT.md`, unedited.

## Executive Summary

No significant "small clearly beats big" yet; three things that are real. (1) **Parity, fairly
measured on two sets**: NECTEC 115M ties typhoon-whisper-large-v3 on the benchmark and on TVSpeech
(once Whisper's long-clip truncation was found and fixed). (2) **A 6-GPU-minute adaptation** —
one epoch of LOTUSDIS with labels rewritten into the base model's own convention, α .75 — is −4.8% /
−5.9% vs Whisper on TVSpeech across two seeds (P .054 / .023; Holm .108 / .046), a tie on the
benchmark, and costs name recall. Raw labels instead: +31% on the benchmark. (3) **With a name list**,
NECTEC + Typhoon's 115M CTC spotter spliced by frame beats the vendor's 0.78B list model on all four
metrics at bias@50 — offline (≈ 350 ms/turn). Killed: voting, routing, RL (gate), boosting tree,
synthetic entity audio. Stage-3 review: 4/10 not ready, fixes applied. Next: a CTC head on NECTEC's
frozen encoder. Full narrative: `NARRATIVE_REPORT.md`.

## Literature Landscape

**What "DeepSeek" means, translated to ASR.** DeepSeek's moment was not one trick: an open model,
trained for a small fraction of the incumbents' cost, that reached their quality, using
(a) a strong pre-trained base, (b) cheap targeted post-training, and (c) sequence-level RL on a
verifiable reward (R1). The ASR analogue: an open ~100M real-time model, adapted on one consumer
GPU in hours, that **significantly** beats the 1.5B-class model. The reward in ASR is already
verifiable: −CER against a label.

**Where Thai stands (measured here, `F30`).** The base already exists: NECTEC's 115M
FastConformer-RNNT ties typhoon-whisper-large-v3 (CER 0.0676 vs 0.0735, P = 0.084) and is worse on
names (B-CER 0.213 vs 0.180). Typhoon's own paper makes the same "45× cheaper, comparable accuracy"
claim for its 115M model [T1]. **So "small ties big" is already published; "small clearly beats
big" is not.** That gap is the target.

| route | claimed number | source |
|---|---|---|
| MWER / sequence-level training on N-best | −8.2% rel. WER (attention S2S); −8.0% (RNN-T EP); SCST −8.7% / −7.8% | [C2] Prabhavalkar 2018, [C7] Li 2020, [C15] Chen 2022 |
| RL with an LLM as reward, unlabelled data | −21% **entity** WER vs self-training | [C14] Ling 2025 |
| GRPO vs SFT on **synthetic-only** speech | WER 36.71 → 22.09 (−40% vs SFT); SFT→GRPO −45% — LLM-based ASR | [W1] arXiv 2607.08409 (Jul 2026) |
| Weight interpolation (fine-tuned ⊕ base) | forgetting 3.4% vs 49% for LoRA fine-tune-on-all | [C10] Plantinga, SLT 2024 |
| Update only encoder linears | prevents forgetting (LibriSpeech → AMI) | [C4] Takashima 2022 |
| Text normalisation instead of scaling | 115M ≈ Whisper-large-v3 at 45× less compute | [T1] Typhoon ASR Real-time, arXiv 2601.13044 |
| Distillation into a small student | within 1% WER at 51% params — **from a stronger teacher** | [C1] Distil-Whisper |

**Structural gap.** Sequence-level RL for ASR has been shown on English and on LLM-based ASR, and
(2026) to extract far more from synthetic speech than SFT does. No result was found applying it to a
small Thai transducer, or measuring whether it closes the **name** gap that synthetic entity audio
targets. ⚠ "Not found" = Consensus (3 queries) + one web search, not an enumeration.

## Ranked Ideas

Pilots score the 682 real `thai-contextasr-bench` utterances with `eval_canon.py` (one normaliser on
both sides) unless stated. NECTEC numbers are **greedy** (0.0695) when compared with a fine-tune,
because `eval_rt_local.py` decodes greedy; beam-5 base is 0.0676.

### ❌ I0 — Small models voting together (zero training) — PILOT NEGATIVE, ELIMINATED
Token ROVER (`pilot_rover.py`) with NECTEC added to the pool:

| pool | CER | B-CER | recall |
|---|---|---|---|
| NECTEC alone (beam 5) | **0.0676** | 0.2132 | 0.3736 |
| NECTEC + tw + pat (two 1.5B Whispers) | 0.0690 | **0.1859** | 0.3713 |
| NECTEC + rt + q06 + qctx + q17 (all ≤2B) | 0.0690 | 0.2153 | 0.3462 |
| NECTEC + rt + q06 (all ≤0.8B) | 0.0728 | 0.2290 | 0.3485 |

**NECTEC already is the field's consensus**: every vote loses to it on CER. Combined with `F27`
(8-model ROVER 0.0678), the cheap win cannot come from combining models. One useful by-product:
voting with the Whispers fixes names (B-CER 0.213 → 0.186) — the name knowledge exists in the big
models and not in NECTEC.

### 🔴 I1 — Fresh-domain SFT of NECTEC on LOTUSDIS (raw labels) + interpolation — PILOT NEGATIVE, CAUSE FOUND
16.6 h near-mic conversational Thai (`adapter/lotusdis_extract.py`: mic `lav123`, 20,646 clips,
speaker-held-out val 605, 23 clips dropped by the 20-char benchmark leak guard). Fresh by the gate
(NECTEC test 0.205 / train 0.124). `lot1`: 2 epochs, LR 5e-5, 2,581 steps, ~2 h on the 5060 Ti
(slowed by a parallel job); `avg_nemo.py` interpolations. Greedy throughout; base greedy = 0.0695.

| α (share of the tune) | LOTUSDIS val | LOTUSDIS test | **bench CER** | B-CER | recall |
|---|---|---|---|---|---|
| 0 (base NECTEC) | 0.1137 | 0.2033 | **0.0695** | 0.2121 | 0.3667 |
| 0.25 | 0.0985 | 0.1784 | 0.0699 | 0.2125 | 0.3713 |
| 0.50 | 0.0782 | 0.1594 | 0.0724 | 0.2300 | 0.3713 |
| 0.75 | 0.0684 | 0.1555 | 0.0755 | 0.2383 | 0.3736 |
| 1.00 | 0.0696 | 0.1533 | 0.0913 | 0.2643 | 0.3850 |

**−25% in domain, +31% on the benchmark, monotone in α — interpolation does not rescue it.** The
diff (335 utts worse, 137 better; 1,606 of the +2,194 char errors in the read-speech `c` group) is a
**convention shift**, the `F2`/`F17` trap arriving through training labels: LOTUSDIS writes loanwords
in Latin script (`Website`, `India`, `UNESCO` — 27,585 Latin letters vs NECTEC's 678 on the bench),
spoken spellings (เค้า 1,490× vs 5, มั้ย 1,601× vs 3, นึง 666× vs 4), never ๆ (0 vs 50), and
fillers (เอ่อ 732× vs 3). The model learned the typist, and the typist is not the benchmark's.
Latency unchanged (~50 ms p50 greedy, batch 1).

### 🟡 I1b — the same, labels rewritten into the BASE model's own convention — PILOT NULL
`adapter/lotusdis_convention.py`: drop Latin rows (4,012 / 20,646), เค้า→เขา, มั้ย→ไหม, นึง→หนึ่ง,
25 whitelisted reduplications XX → "X ๆ", drop เอ่อ; 3,743 rows relabelled, 16,634 kept. Target =
what NECTEC already emits, read from its outputs — not tuned on the test references. `lot2`.
🔴 **1 epoch, not 2**: epoch 2 was stopped at 31% because of a host-memory leak (below); the finished
epoch-1 checkpoint is `lot2`. val/test below are scored against the converted labels.

| α | LOTUSDIS val | LOTUSDIS test | **bench CER** | B-CER | recall | Δerrors a / c |
|---|---|---|---|---|---|---|
| 0 (base) | 0.1069 | 0.1904 | **0.0695** | 0.2121 | 0.3667 | — |
| 0.25 | 0.0943 | 0.1690 | **0.0686** | 0.2101 | 0.3667 | −43 / −35 |
| 0.50 | 0.0802 | 0.1580 | 0.0713 | 0.2248 | 0.3576 | |
| **0.75 ← val choice** | **0.0703** | 0.1465 | 0.0715 | 0.2235 | 0.3440 | +13 / +9 |
| 1.00 | 0.0731 | 0.1451 | 0.0762 | 0.2273 | 0.3440 | +119 / +332 |

The convention fix removed most of the harm (α = 1: 0.0913 → 0.0762, though with half the epochs).
What is left is domain style, not spelling: added fillers (อะ, เนี่ย) and tone-mark changes, mostly
on read speech. The val-chosen model is **worse** than base on the benchmark; α = 0.25 is better in
both source groups but by 0.0009 — inside any CI.

### 🔴 I3 — + synthetic entity audio (conv LOTUSDIS + 4,201 VoxCPM clips) — PILOT NEGATIVE ON REAL SPEECH
`lot3`, 2 full epochs after the leak fix (~20 min).

| α | LOTUSDIS val | synth held-out (60 unseen texts) | **bench CER** | B-CER | recall |
|---|---|---|---|---|---|
| 0 (base) | 0.1069 | 0.0390 | **0.0695** | 0.2121 | 0.3667 |
| 0.25 | 0.0930 | 0.0224 | 0.0696 | 0.2101 | 0.3667 |
| 0.50 | 0.0742 | 0.0118 | **0.0695** | 0.2080 | 0.3576 |
| **0.75 ← val choice** | **0.0665** | 0.0108 | 0.0711 | **0.2054** | 0.3554 |
| 1.00 | 0.0670 | 0.0110 | 0.0737 | 0.2114 | 0.3349 |

**−72% on held-out synthetic entity audio, zero on real speech.** Entity recall on the benchmark
*falls* (0.367 → 0.335 at α = 1). B-CER moves 0.212 → 0.205 at the val choice, a 3% fraction of
the 0.032 gap to typhoon-whisper-large-v3. The model learned the TTS voice and its entity *types*,
not Thai names in real mouths — the synthetic held-out number is self-confirming (`F5`, `S7`).
Synthetic data helped the *picker* (`F29`) because a picker ranks the ASR's real hearings; it
does not help the *ASR* hear.

### ⛔ I2 — sequence-level RL — KILLED BY THE PRE-REGISTERED GATE
Rule (reviewer, set before any result): kill if the best SFT+interpolation top-1 > 0.066. Best
measured: **0.0686** (lot2 α 0.25); the val-chosen models are 0.0711–0.0715. Killed without being
built. The oracle half of the rule was not needed.

### 🟢 TVSpeech (independent set, Typhoon's robustness track) — lot2 helps where the benchmark could not see it
570 in-the-wild YouTube clips, 3.75 h, ≤ 30 s each; `eval_manifest.py`, one normaliser, greedy.

| model | TVSpeech CER |
|---|---|
| base NECTEC | 0.0811 |
| lot2 α 0.25 / **0.50** / 0.75 (val choice) / 1.0 | 0.0745 / **0.0723** / 0.0747 / 0.0797 |
| lot3 α 0.25 / 0.50 / 0.75 / 1.0 | 0.0778 / 0.0766 / 0.0786 / 0.0814 |
| ~~typhoon-whisper-large-v3, one 30 s window~~ | ~~0.2004 / 0.2036~~ — **harness artifact, discarded** |
| **typhoon-whisper-large-v3, audio cut at pauses into ≤ 15 s pieces** | **0.0785** |

#### The bar, measured properly (2026-10-07) — and what it took
Two wrong diagnoses first, both recorded: (1) "448-token cap, use timestamps" — timestamps gave the
identical 0.2004; (2) "faster-whisper's 224-token default, raise `max_new_tokens`" — 0.2036, still
capped. Measured cause: typhoon-whisper generates Thai in fragment tokens, so ~29 s of Thai in ONE
decoder window exhausts the 448-token context at ~224 canonical tokens (170 / 570 clips cut
mid-word, every output ≤ 225 canonical tokens). faster-whisper's own `vad_filter` does not help — it
re-joins its chunks and still decodes 30 s windows (20 cut clips: 0.323 → 0.316). Cutting the audio
at pauses into ≤ 15 s pieces and decoding each alone (`fw_manifest.py --vad-max-s 15`) fixes it
(same 20 clips → 0.069, *better* than NECTEC's 0.084 there). That is the bar used below — the
fairest version of the big model this repo can run, not the weakest.

| vs typhoon-whisper-large-v3 0.0785 (570 clips, 5,000 paired resamples) | CER | rel. | P(not better) | 95% CI of diff |
|---|---|---|---|---|
| base NECTEC | 0.0811 | +3.3% | 0.85 | [−0.002, +0.007] |
| **lot2 α 0.75 (val-chosen)** | **0.0747** | **−4.8%** | **0.048** | [−0.009, +0.001] |
| lot2 α 0.50 (*not* pre-registered — shown, not claimed) | 0.0723 | −7.8% | 0.004 | [−0.011, −0.001] |

#### ⚠️ Both sets together — WITHDRAWN as a claim after the Stage-3 review (kept as a record)
Stage-3 review (Gemini 3.8 Flash, thread `839525848fd84784a26b3cadf40e8a99`, **4/10 not ready**):
pooling chosen after seeing the per-set results is a forking path, and TVSpeech's 2/3 of the
characters drowns the benchmark's null. Holm over the two pre-stated per-set tests (P .10, .048) →
TVSpeech P = 0.096, not significant. What stands: parity on both sets; a −4.8% uncorrected
long-form-media gain from 6 GPU-minutes. The original text follows, unedited.
Benchmark, same scorer family (per-utterance, own normaliser): base −4.4% (P = 0.08), lot2 α 0.75
−4.0% (P = 0.10) — both ties there. Pooled over both independent sets, resampled within each set:

| pooled (682 + 570) | CER | vs bar 0.0817 | P(not better) |
|---|---|---|---|
| base NECTEC 115M | 0.0823 | +0.6% | 0.63 |
| **lot2 α 0.75** | **0.0780** | **−4.6%** | **0.026** |

**A 115M real-time model, adapted for ~6 GPU-minutes on one RTX 5060 Ti (one epoch, 1,040 steps
of 16.6 h convention-matched LOTUSDIS) and interpolated with its base at the α chosen on held-out
speakers, beats typhoon-whisper-large-v3 (1.55B) at P = 0.026.** The base model alone does not
(P = 0.63): it wins the benchmark and loses TVSpeech; the cheap adaptation makes it win both.
🔴 Caveats that bound it: pooling was decided *after* the per-set numbers were seen (a forking
path — the pre-registered primary was the benchmark alone, where it is a tie); TVSpeech carries
~2/3 of the characters; lot2 is 1 epoch because a memory leak stopped epoch 2; one seed.

**lot2 at the val-chosen α is −8% on TVSpeech (−11% at α 0.5)**, where on the 682 benchmark utts it
was +3%. Not a convention gift: TVSpeech's labels write the *spoken* forms (มั้ย, เค้า) and lot2 was
trained toward ไหม / เขา, i.e. *away* from them. In-the-wild long-form media speech is closer to
LOTUSDIS than the benchmark's short turns are. lot3's synthetic audio again costs, not helps.

🔴 **The bar's 0.2004 is ours, not Whisper's** (card: 6.32%). 14.3 of its 20 points are deletions in
runs ≥ 20 chars (NECTEC: 0.3): decoding without timestamps, a ~29 s Thai clip hits Whisper's
448-token decoder cap (≈ 1 token per Thai char) and the rest is dropped. Re-run with timestamps
(long-form seek) queued in `adapter/night_tvbar.sh`. Until then **no NECTEC-vs-bar claim on TVSpeech**.

### ⭐ I6 — Names at decode time from a second model: NECTEC + Typhoon's CTC spotter, spliced by frame — POSITIVE ON NAMES, TIE ON CER, NOT YET REAL-TIME
`arm_d6_splice.py --mode rnnt --rnnt-model <NECTEC .nemo>` (new flag). NECTEC writes the transcript;
`typhoon-asr-realtime-nemo-ctc` (115M, separate pass) scores every bias-list entry with Typhoon's
released keyword spotter; accepted spots replace NECTEC's tokens in the same frames. **No shared
encoder or tokenizer is needed** — the splice is by frame index (both 0.08 s: NECTEC `window_stride`
0.01 × `subsampling_factor` 8, checked) and inserts the keyword as text. τ = −1.5, the published
default, never tuned on real. Canon scorer (`eval_canon.py`, now scoring each file's own condition):

| bias@50 unless stated | params | CER | U-CER | B-CER | **recall** |
|---|---|---|---|---|---|
| NECTEC + D6, **none** (harness control) | 115M | 0.0695 | 0.0630 | 0.2137 | 0.362 |
| NECTEC + D6, **dis@50** (same-size decoys) | 2×115M | 0.0726 | 0.0629 | 0.2875 | 0.355 |
| **NECTEC + D6, bias@50** | 2×115M | **0.0728** | **0.0625** | 0.2876 | **0.576** |
| typhoon-asr-qwen-0.6b-ctx (reads the list) | 0.78B | 0.0791 | 0.0704 | **0.2628** | 0.472 |
| typhoon-asr-realtime + D6 (`F8b`'s arm) | 2×115M | 0.0938 | 0.0857 | 0.2625 | 0.554 |
| Typhoon's released CTC spotter pipeline | 115M | 0.1203 | 0.1102 | 0.3202 | 0.485 |
| *typhoon-whisper-large-v3, none (takes no list)* | 1.55B | *0.0735* | *0.0682* | *0.1798* | *0.408* |

- The control reproduces plain NECTEC exactly (0.0695), so the harness adds nothing by itself.
- **+97 names over the same-size decoy list** (157 → 254 of 439), U-CER flat-to-better. Decoy
  false-fire rate 0.0014.
- **Paired bootstrap** (2,000 utterance resamples; own per-utterance scorer — exact-substring recall,
  no list-insertion charge — so absolute values differ from the table):
  vs typhoon-whisper-large-v3: recall higher **P < 0.0005**, CER lower P = 0.11 (**tie**);
  vs qwen-0.6b-ctx at bias@50: CER **and** recall better, both **P < 0.0005**;
  vs own dis@50: CER lower P = 0.024, recall P < 0.0005.
- 🟡 **B-CER does not move** (0.2875 → 0.2876) although recall rises by 97 mentions. Under the
  independent scorer the same splice *lowers* CER vs decoys (0.0874 → 0.0853), so this is most likely
  `F8d`'s list-insertion charge, which grows with the list for decoys and real names alike — not
  shown here, only inferred; an entity-level diff is owed before anyone quotes B-CER from this arm.
- 🔴 **Latency**: RTF 0.0364 vs NECTEC's 0.0117 → ≈ 350 ms vs 113 ms per 9.7 s turn (+240 ms). The
  arm is unoptimised (a second full encoder, a Python Viterbi per keyword); a CTC head on NECTEC's own
  frozen encoder would remove the second encoder pass.

**Against the pre-registered criteria (`refine-logs/FINAL_PROPOSAL.md`), strictly as written:**
C1 recall ≥ 0.420 ✅ (0.576) and > dis@50 by ≥ 0.05 ✅ (+0.221) — but B-CER ≤ 0.170 ❌ (0.288);
C2 CER ≤ 0.0680 ❌ (0.0728); C3 ≤ +50 ms ❌ (+240 ms est.). **It fails as written.** Two of those
three thresholds compare a bias@50 number with the bar's `none` number, which `F8d` and CLAUDE.md
forbid — that was a defect in my criteria, written before the numbers, and it is recorded rather
than quietly moved. The claim that survives at the same N: **2×115M with a list finds significantly
more names than every list-capable Thai system run here, including the 0.78B vendor model, and
significantly more than the 1.55B Whisper, while tying it on CER.**

#### I6b — the B-CER puzzle solved: splice boundaries (2026-10-07)
Entity-level diff of bias@50 vs dis@50: the splice dropped only tokens whose *start* frame lay
inside a spot, so the tail of the old word survived beside the inserted name — "ปัญญาวิจินธนสาร
**ศาสตร์**", "ธงชัยศรีสุขประเสริฐ **ริฐ**". New `--pad-start/--pad-end` (frames) also claim
neighbouring tokens. Chosen **on the synth split only** (bias@50, 7 settings; CER 0.0607 → 0.0542,
B-CER 0.166 → 0.128 at pad 1:2), then run once on real:

| canon | CER | U-CER | B-CER | recall |
|---|---|---|---|---|
| pad 0:0, dis@50 / bias@50 | 0.0726 / 0.0728 | 0.0629 / 0.0625 | 0.2875 / 0.2876 | 0.355 / 0.576 |
| **pad 1:2, dis@50 / bias@50** | 0.0726 / **0.0727** | 0.0635 / 0.0655 | 0.2765 / **0.2238** | 0.349 / **0.574** |
| qwen-0.6b-ctx bias@50 (0.78B) | 0.0791 | 0.0704 | 0.2628 | 0.472 |

B-CER now falls **−0.053 against the same-size decoys** and beats the vendor's list-reading model
on every metric. The pads cost U-CER (+0.003: they swallow a neighbouring character now and then),
so CER stays flat. The `F8d` list artifact is visible in our own rows: decoys alone lift B-CER
0.214 → 0.277; that gap, not the model, is most of the distance to the bar's 0.180 at `none`.

### ❌ I2b — D4 GPU boosting tree on NECTEC — DEAD ON DEV, as `F8b` predicted
Synth split, bias@50, sweep α:score ∈ {0.5:1, 1:1, 1:3, 2:3, 2:5, 4:5}: recall 0.380 → 0.408 at best
(1:3) while CER 0.036 → 0.052; 2:3 → CER 0.26, 4:5 → 0.91 (text destroyed). The real split was not
spent on it. (An earlier status line read "CER 0.906 = broken run" — it was only the last row of the
sweep printed by `tail`.)

### ❌ I5 — Route hard utterances to the big model ("big model only for hard audio") — PILOT NEGATIVE
Zero-training, from files on disk (gate normaliser, so absolute values differ from canon; relative
holds): NECTEC 0.0830, tw 0.0887, **per-utterance oracle router 0.0691 (−17%)** — the two models'
errors are complementary. But NECTEC's own confidence (top-1 − top-2 beam score) cannot find them:
routing the least-confident 10% to tw gives −0.7%, 20–50% is *worse* than NECTEC alone. A learned
router is the N-best picker of `F29`, which ties. Eliminated.

### Harness defect found under the runs: a host-memory leak in `finetune_rt.py`
`pin_memory=True` with variable-length audio grew the training process by ~25 MB/step (lot2: 30.9 GB
private, commit 55.5 / 56.9 GB, throughput 2.9 → 0.26 it/s in epoch 2). It is also why `lot1` took
~2 h instead of ~15 min, and why Claude Code reaped background shells. Fix: `pin_memory=False` and
plain greedy (no CUDA-graph decoder) for the in-training WER log; per-epoch `.nemo` saves added.
After the fix lot3 held flat (24–30 GB private, a fixed CUDA reservation) at ~2.8 it/s.
⚠ Checkpoints saved by the fixed script carry `use_cuda_graph_decoder = False`, so their greedy
latency (~50 ms here) is not comparable with the base's CUDA-graph latency — CER is unaffected.

### I2 — Sequence-level RL (MWER / GRPO-style) on NECTEC's own N-best — GATED ON I1/I3
Headroom measured: NECTEC top-5 oracle **0.0597** vs top-1 0.0676 (−11.7%). Reward = −CER vs
label on LOTUSDIS + synthetic. Built only if the SFT models move the oracle too (reviewer's rule,
below).

### I3 — Close the name gap with fresh synthetic entity audio — PILOT QUEUED
LOTUSDIS + 4,201 synthetic clips (7.9 h, 245 entities; 13 clips of 6 entities that occur in the
benchmark dropped), SARA AM re-composed to NECTEC's convention
(`E:/Claude/datasets/lotusdis_lav/manifest_train_plus_synth.jsonl`). Targets B-CER 0.213 → ≤0.180.

### I4 — "ThaiSeek-115M": I1 + I3 → (I2 if gated in) → interpolate, cost reported
The deliverable if the parts work: one open 115M checkpoint, real-time, with its total training
cost in GPU-hours on a 16 GB card, that beats typhoon-whisper-large-v3 **significantly** on the
benchmark **and** on an independent set.

### Eliminated before piloting
| idea | why |
|---|---|
| distil the 8-model ensemble | `F28`: done four times, no gain; the teacher is no better than NECTEC (0.0678 vs 0.0676) |
| fine-tune a big Whisper | not cheap, not real-time — contradicts the direction |
| n-gram shallow fusion | `D2`/`D3`: noise on the sister RNN-T |
| bigger N-best picker | `F29`: ties NECTEC at 3–4× latency |

## Novelty Verification

Reviewer (Gemini 3.8 Flash, thread `e6800f81f17b46178dd5ddfc7fbe33ed`), verdict
**PROCEED WITH CAUTION**. Adopted in full:

- **Algorithmic novelty ≈ 0.** "GRPO on an RNN-T N-best" *is* MWER [C2]; calling it GRPO is
  marketing. Frame as an **empirical recipe**: can sequence-level training make a small Thai
  transducer learn from synthetic audio where SFT cannot — the [W1] question, asked of a 115M
  transducer instead of an LLM-ASR.
- **We start at a tie**, and Typhoon already published "115M ≈ Whisper-large-v3" [T1]. A 0.003 win
  on 682 utterances is not a DeepSeek moment.
- **682 utterances is under-powered** (< 1,000 character errors; 5% rel. ≈ 35 edits). ⇒ **Added
  TVSpeech** (Typhoon's own robustness track, 570 utts / 3.75 h, CC-BY; their card: tw 6.32%) as
  an independent second test set, and α / checkpoint are chosen on **LOTUSDIS val, never on a
  test set**.
- **MWER on a 5-best: expect ≤ 2–3% rel.**, not ≥ 5% — 5% would need 43% of the oracle gap.
- **Cheapest discriminating experiment**: SFT (I1+I3) + interpolation sweep, measuring top-1 CER,
  B-CER **and the top-5 oracle**. **Kill I2** if SFT+soup top-1 > 0.066 *or* the oracle does not
  drop below 0.056. If SFT+soup alone reaches ≤ 0.063 and fixes names, the recipe needs no RL.

## External Critical Review

Reviewer: Gemini 3.8 Flash, thread `3f875f07f4eb4027af52dad46d8f82c0`, given every pilot above
including the negatives. Verdict **PROCEED — with decode-time contextual biasing on frozen NECTEC**,
and only that. Its reading, which matches the numbers:

1. **Weight-space adaptation of the 115M model is exhausted** on what this box has: every tune
   (raw, convention-matched, + synthetic) is ≤ base on the benchmark once α is chosen on val.
2. **The deficit was misdiagnosed as general.** NECTEC already beats the bar on CER (0.0695 greedy
   vs 0.0735); its *entire* deficit is named entities (B-CER 0.212 vs 0.180, recall 0.367 vs
   0.408) — open-vocabulary name memory that a 115M model does not get from 24 h of fine-tuning.
3. **Synthetic entity audio taught the TTS, not the names** (held-out synth −72%, real recall ↓).
4. **The one positive signal in the whole record is biasing** (`F24`: entity recall 144 → 207 on a
   Qwen-ctx model, surviving the dis@N control) — so put the names in at decode time, not in weights.

Its kill criteria (adopted): on the real split, NECTEC + biasing must reach **B-CER ≤ 0.170 AND
recall ≥ 0.420** (beating the bar's 0.180 / 0.408), **CER ≤ 0.0680**, and keep latency within
+50 ms — and the gain must survive `dis@N` at the same N.

**My addition, which the reviewer did not weigh:** `F8b` — NeMo's GPU boosting tree on
typhoon-asr-realtime, a 115M FastConformer-RNNT almost identical to NECTEC, moved recall 144 → 143.
The prior for the same mechanism on NECTEC is poor. It is still the cheapest test on the table (no
training, `arm_d4_boosting.py --model`), so it runs; if it fails, the route that worked in `F24` is
a *model that reads the list* (Qwen-ctx), and the real-time question becomes whether a small
list-reading model can be made fast.

## References

- [C1] [Distil-Whisper](https://consensus.app/papers/details/544b08e5b9105caba1906e6f21d19f28/?utm_source=claude_code) (Gandhi et al., 2023)
- [C2] [Minimum Word Error Rate Training for Attention-Based Sequence-to-Sequence Models](https://consensus.app/papers/details/9b6191ff02ec5ef0bcff5d279e6ce63e/?utm_source=claude_code) (Prabhavalkar et al., ICASSP 2018)
- [C4] [Updating Only Encoders Prevents Catastrophic Forgetting of End-to-End ASR Models](https://consensus.app/papers/details/df6e1e9068fa5289804d6fcd58a5c459/?utm_source=claude_code) (Takashima et al., Interspeech 2022)
- [C7] [Towards Fast and Accurate Streaming End-To-End ASR](https://consensus.app/papers/details/2825604e098851d2b4f63c604bcb952a/?utm_source=claude_code) (Li et al., ICASSP 2020)
- [C10] [Parameter Averaging Is All You Need To Prevent Forgetting](https://consensus.app/papers/details/ecc40d8e65a35899a2cf39ac405c5fc7/?utm_source=claude_code) (Plantinga et al., SLT 2024)
- [C14] [Customizing Speech Recognition Model with Large Language Model Feedback](https://consensus.app/papers/details/74e2a92c91c1521fa3e02282ba220842/?utm_source=claude_code) (Ling et al., ASRU 2025)
- [C15] [Self-Critical Sequence Training for Automatic Speech Recognition](https://consensus.app/papers/details/620e56734a4452c099bc467c3fdc508d/?utm_source=claude_code) (Chen et al., ICASSP 2022)
- [T1] [Typhoon ASR Real-time: FastConformer-Transducer for Thai ASR](https://consensus.app/papers/details/5f6dff9194fd55329698dad4e71f4cd1/?utm_source=claude_code) (Sirichotedumrong et al., 2026)
- [W1] [When Synthetic Speech Is All You Have: Better Call GRPO](https://www.alphaxiv.org/abs/2607.08409) (arXiv 2607.08409, 2026)

<!-- ARIS_IDEA_DISCOVERY_EVIDENCE_GATE:START -->
## Evidence Gate
**Status:** PASS

All required stage records, review receipts, artifacts, and report sections are present.
<!-- ARIS_IDEA_DISCOVERY_EVIDENCE_GATE:END -->

## Follow-up (2026-10-07, his "do it") — I6c: the splice in ONE encoder pass — POSITIVE, NOT YET REAL-TIME

A 2.6M CTC head on NECTEC's frozen encoder replaces Typhoon's second 115M model (`adapter/ctc_head.py`,
`arm_d6_splice.py --spotter head`). Full numbers and the NeMo `transcribe()` encoder-train-mode
defect that invalidated the first run: **`F36`**. In one line: bias@50 CER 0.0748 / B-CER 0.214 /
recall 0.567, better than qwen-0.6b-ctx (0.78B) on all four (P ≤ 0.0002), recall equal to the
two-model splice, CER slightly worse than it; 192 ms/turn of which 154 ms is the Python spotter.

## Follow-up (2026-10-08, his "do both") — real-time reached; A2 does not stack (`F37`)
`kws_fast.py` (numba, bit-identical) brings the one-pass splice to **42.9 ms p50/turn = NECTEC's own
speed**. A head on the A2 encoder: bias@50 CER 0.0741 (tie with base+head), recall 0.519 (lower than
0.567) — still better than qwen-0.6b-ctx on all four. Details: `F37`.
