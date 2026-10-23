# Final Proposal — "ThaiSeek": a 2×115M real-time Thai ASR that beats the 1.5–2B class on names

**Run**: `deepseek-asr-2026-10-06` · **Date**: 2026-10-06 · Source: `idea-stage/IDEA_REPORT_2026-10-06.md`

## Problem anchor (frozen)

His direction: *"like the come of deepseek — cheaper, and can perform like the high-end model."*
Operationally: an **open, real-time (≤ ~150 ms/turn) system of ~100M-class models** that
**significantly** beats typhoon-whisper-large-v3 (1.55B) on `thai-contextasr-bench` real split
(682 utts), and is at least competitive with the vendor's best *biased* model
(typhoon-asr-qwen-1.7b-ctx, 2.07B), confirmed on an independent set (TVSpeech).

## What the pilots settled (do not reopen without new evidence)

- NECTEC 115M already **beats the bar on general CER** (0.0695 greedy vs 0.0735). Its whole
  deficit is **named entities** (B-CER 0.212 vs 0.180; recall 0.367 vs 0.408).
- **Weight-space adaptation is exhausted** on this box's data: LOTUSDIS (raw, convention-matched)
  and + synthetic entity audio all end ≤ base on the benchmark once α is chosen on val.
- **Combination without context is exhausted**: ROVER never beats NECTEC; confidence routing fails.
- **RL killed** by the pre-registered gate.

## Method thesis

Put the names in at **decode time, from a second acoustic opinion, not into weights**: NECTEC
writes the transcript; Typhoon's 115M CTC keyword spotter (their released pipeline, `A26`) scores
each bias-list entry against the audio and the accepted spots are spliced into NECTEC's text by
frame index (both models: 0.08 s frames). This is `D6`, which on typhoon-asr-realtime moved entity
recall .328 → .554 (+99 mentions) at RTF 0.028 → 0.032 — applied to the stronger transcriber.

**Dominant contribution:** the empirical claim that a **cross-model** (no shared encoder or
tokenizer) frame-level splice works, making the name fix a plug-in for *any* RNN-T on the same
frame grid — and that 2 × 115M with a context list beats 1.5–2B models on names at real-time cost.

**What it is not:** a new algorithm (the spotter and splice are published / already in this repo);
a no-context result (it needs a bias list — the benchmark's `bias@N` track).

## Claims and the evidence each needs

| claim | evidence | kill |
|---|---|---|
| C1 names: NECTEC+D6 bias@50 beats the bar on names | B-CER ≤ 0.170 and recall ≥ 0.420, and **beats dis@50** at the same N | either fails, or the gain vs dis@50 < +0.05 recall |
| C2 no harm | CER ≤ 0.0680 and U-CER ≤ base at bias@50; dis@50 false-alarm ≈ 0 | CER > 0.0680 |
| C3 real-time | total latency ≤ base + 50 ms | > +50 ms |
| C4 generalises | TVSpeech: NECTEC CER vs bar (no list) reported regardless of direction | — (reported, not gated) |

## Out of scope / deferred

- Training a CTC head on NECTEC's frozen encoder (one encoder pass instead of two) — the
  engineering version of C3, only worth doing if C1/C2 pass with the two-model splice.
- Transliterating Latin-spelled entities (25% of gold mentions are unspottable by a Thai-character
  spotter, `D6` note) — must be applied to gold *and* distractors or it is an oracle leak.
