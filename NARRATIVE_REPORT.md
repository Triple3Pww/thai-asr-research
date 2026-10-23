# Narrative Report — a cheap Thai ASR that beats the 1.5B model (2026-10-06 → 07)

**Direction** (his words): *"like the come of deepseek — cheaper, and can perform like the high-end
model."* **Pipeline run** `deepseek-asr-2026-10-06`. Full record: `idea-stage/IDEA_REPORT_2026-10-06.md`;
plan and tracker: `refine-logs/`. Reviewer: Gemini 3.8 Flash (`gemini-review`; no Codex on this box).

> 🔄 **What came next (2026-10-08 → 10, `F38`–`F46`, `idea-stage/IDEA_REPORT_2026-10-08.md`):** the A2 recipe
> applied to fresh GigaSpeech2-th (+ LOTUSDIS) is the first real-time model to beat NECTEC on held-out benchmark
> sessions (dev half −6.8%, P .013) and on TVSpeech (−9.5 to −12.1%) — the TVSpeech gain this report called
> borderline is now clear. This report's own numbers below are unchanged.
>
> Point-in-time. One seed unless stated. The bar is **typhoon-whisper-large-v3 (1.55B)**, the best
> plain Thai transcriber the vendor ships.

## 1. Problem and the claim that survived

Make a small, open, real-time Thai ASR that **significantly** beats the 1.5B-class model, cheaply.
The starting point was already a tie: NECTEC's open 115M FastConformer-RNNT matched the bar on the
benchmark (`F30`), and Typhoon has published the same "115M ≈ Whisper-large-v3" claim for its own
model. "Ties" was not new; "clearly beats" was the target.

**The honest answer: a significant "small clearly beats big" was NOT reached.** What was reached,
after the Stage-3 review (Gemini, 4/10 "not ready" on the first draft — its fixes are applied below):

**A1 — parity, measured fairly, on two sets.** NECTEC 115M matches typhoon-whisper-large-v3 (1.55B)
on general Thai CER at >10× less compute: benchmark −4.4% (P = 0.08), TVSpeech +3.3% (P = 0.85).
Ties on both. That repeats Typhoon's own published "115M ≈ Whisper-large-v3" claim — not new.

**A2 — a 6-GPU-minute adaptation moves long-form media speech, not the benchmark.** One epoch of
16.6 h LOTUSDIS with labels rewritten into the base model's own spelling convention, interpolated
75/25 (α on held-out speakers): TVSpeech **−4.8% vs Whisper, P = 0.048 uncorrected**; benchmark a tie
(P = 0.10) and slightly *worse than base NECTEC* (0.0715 vs 0.0695). With Holm correction over the
two pre-stated per-set tests the TVSpeech result is P = 0.096 — **not significant**. A pooled test
(P = 0.026) was computed but chosen after seeing the per-set results; it is withdrawn as a claim.

**B — with a name list (bias@50), against list-aware systems only.** NECTEC's transcript + Typhoon's
released 115M CTC keyword spotter, spliced by audio frame (two 115M models, no training) beats the
vendor's **0.78B list-reading model (qwen-0.6b-ctx) on all four metrics** at bias@50 and beats its own
same-size decoy control. It is **not** compared with Whisper, which takes no list (`F8d`). It is an
**offline proof-of-concept** (≈ 350 ms/turn vs 113 ms) and **failed its pre-registered C1–C3**; the
boundary pads that fixed B-CER are exploratory recovery after that failure.

## 2. Method

| piece | what | file |
|---|---|---|
| fresh data | LOTUSDIS lavalier mic, 20,646 clips, speaker-held-out val, 20-char leak guard vs the benchmark | `adapter/lotusdis_extract.py` |
| convention match | drop Latin-script rows; เค้า→เขา, มั้ย→ไหม, นึง→หนึ่ง; whitelisted reduplication → ๆ; drop เอ่อ — rules read from the BASE model's outputs, not test refs | `adapter/lotusdis_convention.py` |
| adaptation | RNN-T fine-tune, LR 5e-5, bf16, batch 16; per-epoch saves | `adapter/finetune_rt.py --base` |
| anti-forgetting | linear interpolation with the base, α picked on val | `adapter/avg_nemo.py`, `adapter/eval_chain_lot.sh` |
| names | cross-model frame splice, pads 1/2 frames chosen on the synthetic split | `adapter/arm_d6_splice.py --rnnt-model --pad-*` |
| scoring | canon normaliser on both sides; own per-utterance scorer for paired bootstraps | `adapter/eval_canon.py`, `adapter/eval_manifest.py` |

## 3. Results

### A1 / A2 (all greedy, no list)

| | benchmark 682 (canon) | TVSpeech 570 | (pooled, post hoc — not a claim) |
|---|---|---|---|
| typhoon-whisper-large-v3 (1.55B) | 0.0735 | 0.0785 | — |
| NECTEC 115M | 0.0695 | 0.0811 (+3.3%, P .85) | +0.6%, P .63 |
| NECTEC + LOTUSDIS-conv, α .75 | 0.0715 | 0.0747 (−4.8%, P .048; Holm .096) | −4.6%, P .026 |
| NECTEC + LOTUSDIS **raw labels**, α 1 | 0.0913 | — | — |
| + 7.9 h synthetic entity audio (lot3), α .75 | 0.0711 | 0.0786 | — |

### B (bias@50 vs its own same-size decoy control and the vendor's list model; canon) — Whisper row for reference only, not a comparison

| | CER | U-CER | B-CER | recall |
|---|---|---|---|---|
| NECTEC + splice, dis@50 | 0.0726 | 0.0635 | 0.2765 | 0.349 |
| **NECTEC + splice, bias@50** | **0.0727** | 0.0655 | **0.2238** | **0.574** |
| qwen-0.6b-ctx (0.78B), bias@50 | 0.0791 | 0.0704 | 0.2628 | 0.472 |
| typhoon-whisper-large-v3, none (takes no list) | 0.0735 | 0.0682 | 0.1798 | 0.408 |

## 4. What failed, and what each failure taught (as valuable as the wins)

| tried | result | lesson |
|---|---|---|
| small models voting (ROVER) | never beats NECTEC alone | NECTEC already *is* the consensus |
| fine-tune on raw LOTUSDIS | in-domain −25%, benchmark **+31%** | the model learns the typist: Latin loanwords, spoken spellings — `F2`/`F17` through training labels |
| synthetic entity audio | synth held-out −72%, real flat, recall ↓ | TTS teaches the voice, not the names |
| route uncertain turns to Whisper | oracle −17%, own confidence −0.7% | the errors are complementary but invisible to the small model |
| sequence-level RL | killed by the pre-registered gate | — |
| GPU boosting tree | +0.03 recall, CER up | re-weighting a model's own guesses cannot add a name it never heard (`F8b`) |
| pre-registered C1–C3 for Claim B | failed as written | two thresholds compared bias@50 with `none` — the criteria broke the repo's own `F8d` rule |

## 5. Harness defects found under the numbers

1. **`pin_memory=True` leaked ~25 MB/step** in `finetune_rt.py` → 30.9 GB private, 10× slowdown,
   Claude Code shells reaped. Fixed.
2. **The bar's first TVSpeech score (0.2004) was ours**: one 30 s Thai window exhausts Whisper's
   decoder (Thai decodes to fragment tokens), cutting 170/570 clips mid-word. Two wrong diagnoses
   (timestamps; `max_new_tokens`) before the fix: cut audio at pauses into ≤ 15 s pieces.
3. **`eval_canon.py` scored every non-`none` run as zero rows.** Fixed to score each file's condition.
4. **The splice left old-word tails beside inserted names** — the reason B-CER did not move. Pads.

## 6. Limitations and follow-ups

- **One seed** for Claim A (a second seed was running at the time of writing — §7).
- Pooling the two sets was decided after seeing them; on the benchmark alone Claim A is a tie.
- Claim B needs a name list at run time, and ≈ 350 ms/turn. **Next: a CTC head on NECTEC's frozen
  encoder** (one encoder pass) — the real-time version of Claim B, and the obvious way to combine A+B.
- 25% of gold entities are Latin-spelled and unspottable by a Thai-character spotter.
- LOTUSDIS is CC-BY-SA-4.0; a released checkpoint inherits share-alike.

## 7. Second seed and external review

**Seed 2** (`lot2s2`, same recipe, `--seed 7`, 1 epoch, α .75; ~9 min train + eval):

| | LOTUSDIS val | benchmark canon CER / recall | bench vs bar (P) | TVSpeech | TVSpeech vs bar (P) | Holm (2 tests) |
|---|---|---|---|---|---|---|
| base NECTEC | 0.1069 | 0.0695 / 0.367 | −4.4% (.08) | 0.0811 | +3.3% (.85) | — |
| seed 1 (`lot2`) | 0.0703 | 0.0715 / 0.344 | −4.0% (.11) | 0.0747 | −4.8% (.054) | .108 ✗ |
| seed 2 (`lot2s2`) | 0.0729 | 0.0702 / 0.349 | −3.8% (.12) | **0.0739** | **−5.9% (.023)** | **.046 ✓** |

(Bootstraps: 5,000 paired resamples, own per-utterance scorer; seed 1's TVSpeech P moved .048 → .054
between bootstrap RNG seeds — that is how close to the line it sits.) **The long-form-media gain
replicates in direction and size across two training seeds (−4.8%, −5.9%); its significance is
borderline (one seed survives Holm, one does not).** The benchmark stays a tie for both, and both
seeds *lower* benchmark name recall (0.367 → 0.344 / 0.349) — conversational adaptation costs names.

**External review** (Stage 3, Gemini 3.8 Flash): round 1 thread `839525848fd84784a26b3cadf40e8a99`,
**4/10, not ready** — (1) the pooled P = 0.026 was post-hoc → withdrawn, Holm applied; (2) Claim B
must not be compared with Whisper (no list) → decoupled; (3) Claim B is not real-time and failed its
pre-registration → stated as an offline proof-of-concept. All three applied above. A second review
round was not possible: the `gemini-review` MCP server disconnected after round 1. **Stage 3 therefore
did not reach its positive stop** (score ≥ 6 and ready/almost); this report is the honest
not-ready state, not a submission draft.

## 8. Done after this report: the one-pass name splice (`F36`, 2026-10-07)

A 2.6M CTC head on NECTEC's **frozen** encoder (trained in minutes on cached features) replaces the
second 115M model. bias@50: CER 0.0748, B-CER 0.214, recall 0.567 — **better than the 0.78B vendor
list model on all four metrics (P ≤ 0.0002)**, same recall as the two-model splice, CER slightly worse.
192 ms/turn (NECTEC alone 45 ms in the same run); **154 ms of it is the Python keyword spotter**, so
the remaining real-time work is code, not model. The first run was invalid (NeMo leaves the encoder in
training mode after `transcribe()`); fixed and verified before the rerun.

### 8b. Real-time reached, and the stack tested (`F37`, 2026-10-08)

The keyword spotter ported to numba (bit-identical on 3,503 pairs, byte-identical end to end): the
**one-pass name splice runs at 42.9 ms p50 per turn — NECTEC's own speed** — while beating the 0.78B
vendor list model on all four metrics. A head on the A2 encoder ties base+head on CER but recovers
fewer names (0.519 vs 0.567): the two improvements do not stack on the benchmark.

**Where this leaves the direction.** A real-time, 115M-class, open system now: (a) ties the 1.55B
Whisper on plain transcription on two sets; (b) with a name list, beats the vendor's 0.78B list model
on every metric at the same speed as plain NECTEC; (c) with 6 GPU-minutes of adaptation, edges ahead
of Whisper on long-form media speech (borderline significance). That is a cheap model performing like
the high-end ones — not yet a significant overall "beats" on plain transcription.

## 9. Recommended next steps (ranked)

1. **Deploy (b) behind `:8007` as an opt-in mode** (a per-request name list), measured over the wire.
2. **A third and fourth seed of A2 + one more long-form test set** (e.g. GigaSpeech2-th test via
   `typhoon-ai/gigaspeech2-typhoon` ids) — decides whether the −5% media gain is real at P < 0.05
   corrected, without new data.
3. Product check: transcribe some of the voice loop's own conversations with base vs A2 — the
   loop's audio is closer to TVSpeech/LOTUSDIS than to the benchmark.

## Figure / table inventory
All tables above exist as numbers in `results/`; no figures exist yet. Natural figures: a cost–CER
plot (params × latency vs CER, both sets), and the α-sweep curves (val vs benchmark vs TVSpeech).
