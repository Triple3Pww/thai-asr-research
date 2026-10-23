# Experiment Tracker — ThaiSeek (NECTEC + cross-model CTC splice)

Updated 2026-10-07 09:45. Plan: `EXPERIMENT_PLAN.md`. Numbers: canon scorer, 682 real utts.

| # | run | result | status |
|---|---|---|---|
| E0 | NECTEC through D4 code path, `none` | raw-scorer CER 0.0745 / recall 0.364 (= D6 none) | ✅ |
| E1 | D4 boosting sweep, synth split | best recall 0.408 (from ~0.38) at CER 0.036 → 0.052; strong settings destroy text | ❌ dead on dev, real not spent |
| E2 | D6 NECTEC `none` | CER **0.0695** = plain NECTEC → harness control passes | ✅ |
| E3 | D6 NECTEC `dis@50` | CER 0.0726, U 0.0629, B 0.2875, recall 0.355 | ✅ |
| E4 | D6 NECTEC `bias@50` | CER 0.0728, U 0.0625, B 0.2876, **recall 0.576** | ✅ |
| E5 | canon + paired bootstrap | recall > tw P<0.0005, CER tie P=0.11; beats qwen-0.6b-ctx@50 on CER and recall P<0.0005 | ✅ |
| E6 | latency | RTF 0.0364 vs 0.0117 (≈ +240 ms / 9.7 s turn) | ❌ C3 as written |
| E7 | TVSpeech | bar 0.0785 (audio cut ≤ 15 s; runs at 0.2004/0.2036 were a one-window truncation artifact); NECTEC 0.0811 (+3.3%, P .85); **lot2 α.75 0.0747 (−4.8%, P .048)** | ✅ |
| E7b | both sets pooled, stratified bootstrap | base +0.6% (P .63); **lot2 α.75 −4.6% vs tw, P = 0.026** — post-hoc pooling, flagged | ✅ |
| E7c | second seed of lot2 (`lot2s2`, seed 7) | bench 0.0702 (tie, P .12); **TVSpeech 0.0739, −5.9%, P .023 (Holm .046)**; seed 1 re-bootstrapped P .054 (Holm .108); recall falls both seeds | ✅ replicates in direction, borderline significance |
| E7d | Stage-3 review (Gemini) | 4/10 not ready: pooled claim withdrawn, B decoupled from Whisper, B = offline PoC; round 2 impossible (MCP disconnected) | ⚠ no positive stop |
| E9 | CTC head on frozen NECTEC encoder (h3, MLP) | val greedy 0.109; bias@50 CER .0748 B .214 recall .567; beats qwen-0.6b-ctx all four; 192 ms (154 spotter) | ✅ `F36` (run 1 invalid: NeMo encoder train-mode, quarantined) |
| E11 | numba spotter (`kws_fast.py`) | 0/3,503 mismatches; end-to-end byte-identical; **one-pass splice 42.9 ms p50 = NECTEC alone** | ✅ `F37` |
| E12 | head on A2 encoder (`ctchead_a2h3`) | bias@50 CER .0741 (tie w/ base+head), recall .519 (lower), beats qwen-0.6b-ctx | ✅ does not stack, `F37` |
| E8 | entity diff of splice | old-word tails survive beside inserted names | ✅ |
| E8b | splice pads, chosen on synth (1:2), real once | bias@50 B-CER 0.288 → **0.224**, recall 0.574, U-CER +0.003, CER flat; beats qwen-0.6b-ctx@50 on all four | ✅ |

**Gate verdict (as written): FAIL on C1-B-CER, C2, C3; PASS on C1-recall and the dis@50 control.**
The B-CER and CER thresholds compared bias@50 with the bar's `none` — an `F8d` violation in the
criteria themselves; recorded, not moved.

Next, in order of value:
1. ~~E8~~ done (pads).
2. ~~E9~~ `F36`, ~~E11~~ ~~E12~~ `F37`. Next: E13 serve the one-pass splice on `:8007` (opt-in list), wire-measured.
3. E10 — bias@N sweep (10/100/250) beside dis@N, real split, τ fixed.
