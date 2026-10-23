# Idea Discovery Report — toward CER 0.05 by architecture

**Direction** (his words, 2026-10-10): *"do more research to make our asr reach 0.05, I want it to be more architechure improve."*
**Run**: `fuse-ears-2026-10-10` (`.aris/runs/`). **Pipeline**: research-lit (Consensus) → ideas → novelty → review → pilot.

> 🔴 A plan and a pilot record, not a finding. Nothing here moves `RESEARCH-FINDINGS.md` until a named run does.

## 0. What "0.05" means here (interpretation, stated so it can be corrected)

The `F42` protocol is in force: tune on the **dev** half (339 utts), score the **test** half (343) once.
`gs2lot_a75` already reads **0.0472 on the dev half** — but dev is the easy half (NECTEC 0.0506 there, ~0.088 on
test). Reading "0.05" as a dev-half number would be met by an existing model and mean nothing. So:

* **Target = official CER 0.050 on all 682 real utts ≈ −28% vs NECTEC's 0.0695.** On the test half that is the
  relative cut **test CER ≤ 0.72 × NECTEC's test CER**, scored once with `paired_vs.py --half test --final`.
* That is beyond the 3-ear dev oracle of text-level selection (−24.5%, `F42`) — it cannot come from picking
  among existing transcripts. It needs a better ear.

## 1. Where the room is (from the record, not re-derived)

| fact | source | implication |
|---|---|---|
| 61% of NECTEC's edits are whole-word bursts; names 29%, script 23%, tone 8% | `F38` | a better *hearing* model, not a speller (`F39`: writing is ~a function of toned sound) |
| oracle {NECTEC, Pathumma-L, typhoon-L} −24.5% dev; all 12 ears −42% | `F42` | the ears are complementary |
| ROVER, LM, PoE, 3 trained judges: all within ±3% | `F42`–`F44` | once each ear has WRITTEN its text, the evidence of who heard right is gone |
| fresh, convention-matched GS2 → −6.8% dev, −9.5..12.1% TVSpeech at 43 ms | `F45`/`F46` | GS2 dev+test is fresh to all three ears and moves the ear |
| Pathumma-L is the best *sound-level* ear (SylER .1145 vs NECTEC .1293) | `F39` | the ear to fuse |
| Whisper-large ENCODER = 72 ms / 30-s window here (measured 2026-10-10) | probe | the decoder, not the encoder, is what cost ~1 s |

## 2. Literature (Consensus first, 3 calls)

* Multi-encoder multi-resolution E2E ASR (two heterogeneous encoders, hierarchical attention): **−18.0..32.1% WER**
  on WSJ / CHiME-4 — [Li et al. 2019, TASLP](https://consensus.app/papers/details/45734f6f52e65e5abc15328180326211/?utm_source=claude_code).
* Two-stage multi-stream: train only the fusion module on frozen per-stream encoders: **−8.2..32.4%** —
  [Li et al. 2020, ICASSP](https://consensus.app/papers/details/92397235c0985a128fd7e323a783dfeb/?utm_source=claude_code).
* Fusing parallel pretrained speech encoders for LLM-ASR improves every scenario tested (no single number in abstract) —
  [Poncelet et al. 2026](https://consensus.app/papers/details/6467482220e1573a824022f44af07a90/?utm_source=claude_code).
* Thai: Whisper-small encoder + Mamba encoder fused, CommonVoice-th WER 10.5 vs Thonburian 11.2 (**−6%**) —
  [Lin et al. 2025, ICICC](https://consensus.app/papers/details/abc1652c06ad5b929a96a790c6e48420/?utm_source=claude_code).
* Joint CTC/RNN-T score combination: CER 4.52 → 4.38 (**−3%**) AISHELL-1 —
  [Kwok et al. 2024](https://consensus.app/papers/details/7ea0d5448e2559f0a832360d9aa3ccb2/?utm_source=claude_code).

Novelty, honestly: multi-encoder fusion is **published**; the open question is only whether it works *here* — a
frozen 1.55B Whisper encoder feeding a 115M Thai RNN-T, real-time, trained on 18 h fresh to both ears, on
conversational Thai with names. As a contribution it is engineering; as a lever toward 0.05 it is the one the
`F42`–`F44` evidence points at.

## 3. Ideas, ranked

| # | idea | ceiling / prior | cost | verdict |
|---|---|---|---|---|
| **1** | **FUSE** — Pathumma-L encoder (frozen, pooled ×4) fused into NECTEC's encoder output by a zero-init block; NECTEC's RNN-T decoder/joint fine-tuned; GS2 fresh data | lit −8..32%; oracle says room ≥ 24% | ~120 ms/turn; cached features, hours of GPU | **pilot first** |
| 2 | FUSE-E2E — same, NECTEC encoder unfrozen (= `F45` recipe + a second ear) | stacks with `F45` if 1 is positive | 1 night | only if 1 passes |
| 3 | joint CTC+RNN-T beam (the `F36` CTC head as a second scorer) | capped by beam oracle −4.6% (`F38`) | small | low ceiling, not first |
| 4 | GS2 **train** split (~10k h, auto labels) as data scale | unknown; freshness to NECTEC unchecked | days | data, not architecture — parallel lever |
| 5 | three-ear FUSE (+ typhoon-L encoder) | +? over 1 | +72 ms | after 1 |

## 4. Pre-registered pilot (idea 1) — criteria fixed before any training

Arms (all on GS2 train `manifest_train_conv`, val = GS2 held-out videos `manifest_val_conv`, same seed/steps):
* **f1** fusion, Whisper on; **c1** control, identical block, Whisper input zeroed.
* Harness control **done**: identity (zero-init) fusion reproduces NECTEC on **682/682** benchmark utts.

| gate | test | pass |
|---|---|---|
| P1 — does the second ear add anything | f1 vs c1, benchmark **dev** half, official CER, paired bootstrap | f1 better, P < 0.05 |
| P2 — is it a better ear than the best we have | f1 vs `gs2lot_a75` (0.04716) dev | f1 ≤ 0.0472 |
| P3 — real-time | NECTEC + Whisper encoder + block, p50 | ≤ 500 ms |
| P4 — held-out second set | TVSpeech vs NECTEC | not worse |
| FINAL | test half, once | ≤ 0.72 × NECTEC test |

**Stop condition:** f1 ≤ c1 + 1% relative on dev → a frozen Whisper encoder adds nothing a fine-tuned decoder can
use; recorded as a finding, idea 2 not run.

## 5. RESULT (2026-10-10) — stop condition met → `F47`

| gate | result |
|---|---|
| P1 fusion beats control (dev) | ❌ f2 .05611 vs c2 .05118 (own scorer); in-domain f2 .0530 vs c2 .0524 — equal |
| P2 ≤ .0472 | ❌ best f2 (α .5/.75) +2.3% vs NECTEC |
| P3 latency | ✅ (encoder 72 ms; not end-to-end measured — moot) |

Idea 2 (end-to-end) not run, as pre-registered. Cause and data lesson: `RESEARCH-FINDINGS.md` `F47`.
