# RESEARCH-PROGRESS — what is actually done

**Written 2026-09-12.** Companion: `RESEARCH-FINDINGS.md` (what we learned). This file is the honest
state of what was actually done.

⚠️ **Section 5, the pending to-do list, was removed on 2026-09-15** when this repo was reduced to
evidence only — it was a plan, not a measurement. It is in git: `git show 4734fc5:RESEARCH-PROGRESS.md`.

🔄 **SUPERSEDED 2026-10-05 — models HAVE now been trained** (the line below stood until then):
typhoon-asr-realtime students s1–s3 and a typhoon-whisper-turbo decoder student (all **no gain**, `F28`),
and four N-best pickers p1/p3/p3half/p4 (Qwen3-0.6B; best ties NECTEC, `F29`). Still true: **no ASR
model in this repo has been trained on the synthetic corpus as ASR training data** — the synthetic clips
trained only the picker. Section **§0** below has the real-time line.

🔄 **SUPERSEDED AGAIN 2026-10-07** — the sentence above stood until then. NECTEC was fine-tuned on
LOTUSDIS (`lot1`, `lot2`, `lot2s2`) and on LOTUSDIS **+ 4,201 synthetic clips** (`lot3`) — **the synthetic
corpus HAS now trained an ASR, and gave no gain on real speech** (synthetic held-out −72%, benchmark
flat, recall down; `F35`). CTC heads were trained on NECTEC's frozen encoder (`F36`). Section **§00**
below has the 2026-10-06 → 08 line.

🔄 **AND AGAIN 2026-10-09/10** — NECTEC fine-tuned on fresh GigaSpeech2-th (+ LOTUSDIS, + food orders):
the first real-time single model to beat NECTEC on held-out benchmark sessions (dev −6.8%) and on TVSpeech
(−9.5 to −12.1%). Plus a syllable head, three trained judges (no gain). Section **§000** below.

🔴 ~~**Nothing has been trained.** Everything below is measurement, tooling, or literature.~~

🔄 **THE CORPUS EXISTS AS OF 2026-09-18.** Generated overnight on this box, unattended:

| bank | clips | audio | wall | silent | cheap-check keep |
|---|---|---|---|---|---|
| `phase2_rare` (245 texts x 8 voices) | 1,960 | 4.98 h | 192 min | 0 | **98.1%** |
| `phase1_known` (440 texts x 8 voices) | 3,520 | 5.59 h | 219 min | 0 | **97.8%** |
| **total** | **5,480** | **10.57 h** | 6.9 h | **0** | **97.9%** |

VoxCPM2, RTF 0.65, both phases exit 0. Rejects are `internal_stall` 1.6-1.7%, `repetition`
0.1-0.4%, `too_short/long_for_text` 0.1-0.2%. ⭐ **0 of 245 and 0 of 440 texts lost every voice** --
each text has at least one usable clip. Lives on `D:/thai-synth-corpus/` (E: had 12 GB free),
1.2 GB as tar shards plus an extracted copy.

### The rare bank, read back by the full panel (2026-09-18)

`phase2_rare`, 1,960 clips, five ASRs -- biodatlab / Pathumma / Qwen3 / typhoon-whisper voting,
`typhoon-asr-realtime` recorded but not voting.

| outcome | clips | |
|---|---|---|
| **keep** | **1,502** | **76.6%** |
| `entity_unheard` | **227** | 🔴 **the largest single reason** -- the rare name was destroyed inside a sentence that passed |
| `panel_rejected` | 194 | readback CER over 0.10 by majority |
| `internal_stall` / `too_short` / `repetition` | 37 | cheap checks |

⭐ **The entity gate, written the day before, rejected more clips than the CER gate it sits behind
(227 vs 194).** Those 227 would otherwise have trained the model on a wrong spelling for a rare
name, which is worse than not training on them.

⭐ **238 clips are `teach` cases** -- the jury heard the sentence correctly and the model under test
could not. That is the highest-value subset in the corpus, and it is 12% of it.

🔴 **15 of 245 rare texts (6.1%) lost every one of their 8 voices.** `ครน` and `ดุซงญอ` failed in
8 of 8; `ส้มโอนครชัยศรี`, `ส้านหอม`, `สก๊อตช์ไบรต์`, `สะเนียน`, `ลําสินธุ์`, `สรรพยา`, `โล๊ะจูด` in 7 of 8. **This is
`H37`'s rare-word ceiling, measured**: names no amount of regeneration will teach VoxCPM to say.
The other 230 texts keep a median of 7 usable voices out of 8.

Per model, over the 1,923 clips that reached the panel:

| | accept | CER median |
|---|---|---|
| typhoon-whisper | 92.8% | 0.013 |
| Pathumma | 90.7% | 0.033 |
| biodatlab | 89.8% | 0.035 |
| Qwen3 | 79.3% | 0.032 |
| *target (not voting)* | *82.4%* | *0.049* |

⚠️ `phase1_known`'s panel run **died at 01:59 on 2026-09-19**, killed mid-run at 1,600 of 3,442 on
the fourth model -- the box ran out of memory, the same pressure that killed a monitor task the
same night. Its cheap-check numbers above stand; no ASR has read it back.

🔴 **No model has been trained on any of this.** 🔄 *(2026-10-05: still true for ASR training; the
synthetic clips have since trained the N-best picker only — §0.)*

## 000. "Invent an ASR from zero" → "reach CER 0.060" — measured 2026-10-08 → 2026-10-10

Synthesis `F38`–`F46`; every run and the design `idea-stage/IDEA_REPORT_2026-10-08.md`. From 2026-10-08 evening on,
**the 682 real utts are split by recording session** (`results/split_dev_test.json`, commit `494d951`): tune on
**dev** (339), the **test** half (343) is scored once at the end — **not yet scored**. All benchmark numbers below
are official canon CER via `adapter/official_utt.py`; NECTEC dev = 0.05059.

### The table that answers his question (target 0.060 ≈ −13.7% vs NECTEC, real-time)

| system | per turn | dev half | TVSpeech | verdict |
|---|---|---|---|---|
| NECTEC (`:8007`) | 43 ms | 0.05059 | 0.0811 | the bar |
| Ear–Memory–Pen: syllable head / homophone Pen (`F39`) | — | Pen changed 0 utts | — | falsified |
| six no-list ideas: catalog lists, session memory, TTA, RNN-T ensemble, ROVER-NECTEC-family, ear/pen merge (`F40`, full set) | ≤ 2× | worse / tie | — | lost |
| NECTEC-drafted speculative Pathumma-L + typhoon-L + ROVER (`F41`) | 481 ms p50 | −3.1% (full set −6.2%) | tie | selection caveat |
| ROVER + LM tie-break / sentence product of experts (`F42`) | — | worse / −2.3% | — | no |
| trained judges: LOTUSDIS / GigaSpeech2 / pooled no-identity (`F43`/`F44`) | — | 0.0% / +1.5% / 0.0% | — | no |
| ⭐ **NECTEC + 1 epoch GS2 (conv), α .5** (`F45`) | 43 ms | **−5.8%** (P .008) | **0.0713 (−12.1%)** | win |
| ⭐ **NECTEC + 1 epoch GS2 + conv-LOTUSDIS, α .75** (`F45`) | 43 ms | **−6.8%** (P .013) | **0.0734 (−9.5%)** | best dev |
| … + food orders ≤ 10 s (`F46`) | 43 ms | −5.1% best | — | worse than without |
| ROVER: 3 tuned variants + Pathumma-L + typhoon-L | ~550 ms | −8.2% (P .002) | — | over budget |

### Built (all in `adapter/` unless noted)

| file | what |
|---|---|
| `probe_error_anatomy.py` | NECTEC's error by Thai character class, script, entity span, burst length (`F38`) |
| `eval_soundblind.py` | script-blind CER + toned-syllable error rate (tltk G2P), thainlp venv (`F39` E0) |
| `syllable_labels.py`, `syllable_head.py` | G2P sound units, P2G ambiguity, the syllable CTC head (`sylhead_s2.pt`) |
| `pen_lattice.py`, `pen_score.py` | homophone lattice + Qwen3-0.6B cross-fitted re-speller (no gain) |
| `catalog_retrieve.py`, `tta_session.py`, `rnnt_ensemble.py`, `ear_pen_merge.py` | the `F40` arms; `arm_d6_splice.py --lists-file`, `pilot_rover.py --add` |
| `spec_whisper.py`, `pipeline_latency.py` | lossless NECTEC-drafted speculative decoding (KV-cached) and the concurrent 3-ear pipeline |
| `rover_manifest.py`, `pool_search_dev.py`, `rover_lm.py`, `poe_rescore.py` | ROVER on any manifest, dev pool search, LM tie-break, sentence-level PoE |
| `official_utt.py`, `paired_vs.py --half {dev,test} [--final]` | official per-utterance errors; dev/test-aware paired bootstrap that refuses the test half without `--final` |
| `judge.py` (`JUDGE_DATA` = lotusdis or gs2) | build turns → ears → scores → train → apply; one large model per process, scoring in 1,500-turn slices |
| `gs2_convention.py`, `night_gs2.sh`, `night_gs2lot.sh`, `night_gs2lotfood.sh` | GigaSpeech2 → NECTEC convention; the fine-tune → interpolate → score chains |

### Checkpoints and data added

`checkpoints/`: `rt_student_gs2a` + `gs2a_a{25,50,75}`, `rt_student_gs2lot` + `gs2lot_a{25,50,75}`,
`rt_student_gs2lotfood` + `gs2lotfood_a*`, `sylhead_s1/s2.pt`, `judge_v1.pt`, `judge_gs2.pt`, `judge_pooled_noid.pt`.
`E:/Claude/datasets/`: `gigaspeech2_th/` (dev+test 20 h, fresh to all three ears), `judge/`, `judge_gs2/`,
`thai_extra/` (food 16 kHz, elderly), `pen/`. `E:/fw-spike/models/nectec_Pathumma-whisper-th-large-v3-ct2`.

### Defects found under the numbers

- **Plain Levenshtein misled dev tuning** (−4.4% that was −0.7% officially): entity Latin/Thai acceptance → `official_utt.py`.
- **F38's "~20% spelling" was an artifact** of tone-free sound keys; with toned G2P it is hearing error (`F39`).
- **A 3-tokens-per-syllable CTC head cannot fit** NECTEC's 12.5 frames/s (val 0.38 → whole-syllable tokens 0.106).
- **Step-synchronous RNN-T ensembling splices half-tokens** — a fine-tune emits on different frames (`F40` #4).
- **Speculative decoding hit Whisper's 448-position limit** (CUDA assert) and stopped early on failed re-sync — both fixed.
- **The low-memory reaper** killed runs 8+ times: scorer holding all audio in RAM; two large models loaded in one
  process; dense-text clips at batch 16 overflowing VRAM into system RAM; and **orphaned trainers** that outlive
  their killed shell. Fixed by streaming, one model per process, scoring slices, batch 8 × accum 2, an in-job guard.

### Open, in order

1. Score the **test half once** with the chosen model (`gs2lot_a75` or `gs2a_a50`) — the honest benchmark number.
2. Ratchada-STT (MIT, gated — he must accept the terms) → freshness gate.
3. Recipe: 2 epochs / LR / second seed for the GS2 tune; the GS2 tune as the draft ear in the `F41` pipeline.

## 00. The "DeepSeek for Thai ASR" line — measured 2026-10-06 → 2026-10-08

Synthesis `F32`–`F37`; narrative `NARRATIVE_REPORT.md`; every run `idea-stage/IDEA_REPORT_2026-10-06.md`;
plan and tracker `refine-logs/`. 682 real utts, canon, greedy unless stated; TVSpeech = Typhoon's 570-clip
robustness set (`E:/asr-bench/typhoon-eval/`).

### The table that answers his question

| system | size | per turn | bench CER | TVSpeech | bias@50 CER / recall |
|---|---|---|---|---|---|
| typhoon-whisper-large-v3 (the bar) | 1.55B | — | 0.0735 | 0.0785 ¹ | — (takes no list) |
| NECTEC FastConformer (`:8007`) | 115M | 43 ms ² | 0.0695 (tie) | 0.0811 (tie) | — |
| NECTEC + 1 epoch conv-LOTUSDIS, α .75 (A2) | 115M | same | 0.0715 / 0.0702 (2 seeds, tie) | **0.0747 / 0.0739** (P .054 / .023) | — |
| typhoon-asr-qwen-0.6b-ctx | 0.78B | — | 0.0806 | — | 0.0791 / 0.472 |
| NECTEC + Typhoon CTC spotter, 2 encoders (`F34`) | 2×115M | ~350 ms | — | — | 0.0727 / 0.574 |
| ⭐ **NECTEC + CTC head, one pass + numba spotter (`F36`/`F37`)** | 115M + 2.6M | **43 ms** ² | 0.0695 (unchanged) | — | **0.0748 / 0.567** |

¹ only with audio cut at pauses into ≤ 15 s pieces (`fw_manifest.py --vad-max-s 15`); one 30 s window → 0.2004.
² `latency_onepass.py`, batch 1, 9.7 s mean turn: NECTEC `transcribe()` 43.0 ms p50; one-pass splice 42.9 ms p50 / 65 p90.

### Built (all in `adapter/` unless noted)

| file | what |
|---|---|
| `lotusdis_extract.py`, `lotusdis_convention.py` | LOTUSDIS lavalier → manifests (speaker-held-out val, leak guard); labels rewritten into NECTEC's convention |
| `finetune_rt.py` (extended) | `--base` any `.nemo`, `--val`, `--seed`, per-epoch saves; **pin_memory leak fixed** |
| `avg_nemo.py`, `eval_chain_lot.sh` | base ⊕ tune interpolation; serial α sweep (val → test → bench) |
| `eval_manifest.py`, `fw_manifest.py` | any manifest, NeMo or external hyps, one scorer; Whisper with `--vad-max-s` cutting |
| `eval_canon.py` (fixed) | scores each file's own condition (bias@N / dis@N were silently zero rows) |
| `arm_d6_splice.py` (extended) | `--rnnt-model`, `--pad-start/--pad-end`, `--spotter head --head`, `--fast-spot` |
| `arm_d4_boosting.py` (extended) | `--model` (+ tag prefix so it never overwrites typhoon's `d4_*`) |
| `ctc_head.py` | cache a frozen encoder's features, train a CTC head on them (`--base` for any encoder) |
| `kws_fast.py` | Typhoon's keyword Viterbi in numba; `--check` proves bit-identity |
| `latency_onepass.py` | one encoder pass → RNN-T + head + spotting, timed against `transcribe()` |
| `night_*.sh` | the serial overnight chains (deepseek, bias, pad, tvbar1–3, seed, head, a2head, fastspot) |

### Checkpoints and data added

| path | what |
|---|---|
| `checkpoints/rt_student_lot1/lot2/lot2s2/lot3*.nemo`, `lot*_a25/50/75.nemo` | the LOTUSDIS tunes and their interpolations (~7.8 GB) |
| `checkpoints/ctchead_h1/h2/h3.pt`, `ctchead_a2h3.pt` | CTC heads (h3 = 512→1024→2049 on base NECTEC; a2h3 on the A2 encoder) |
| `E:/Claude/datasets/lotusdis_lav/` | 20,646 / 605 / 1,000 clips + `*_conv` manifests (16,634 train after Latin drop) |
| `E:/Claude/datasets/ctchead_cache*/` | frozen-encoder features, 2.05 M frames each (base and A2), ~2 GB each |
| `E:/asr-bench/typhoon-eval/` | TVSpeech (570 clips, wavs + manifest) and gigaspeech2-typhoon (ids + text only) |
| `E:/fw-spike/models/typhoon-whisper-large-v3-ct2/` | the bar under CTranslate2 fp16 |
| `results/_invalid_ctchead_h3_run1/` | the head run spoiled by NeMo's encoder train-mode defect — kept as evidence |

### Defects found under the numbers

| defect | effect | fix |
|---|---|---|
| `pin_memory=True` in training | ~25 MB/step host leak → commit limit, 10× slowdown, reaped shells | off |
| Whisper on one 30 s Thai window | 170/570 TVSpeech clips cut mid-word (0.2004) | cut audio ≤ 15 s at pauses |
| `eval_canon.py` hardcoded `none` | every bias/dis run scored as zero rows | file's own condition |
| splice by token start frame only | old-word tails beside inserted names, B-CER flat | pads 1:2 |
| NeMo 3.0.0 `transcribe()` leaves `encoder.training = True` | direct encoder calls drift BatchNorm — 481/682 transcripts changed | `.eval()` before direct calls |

### Open, in order

1. **Serve the one-pass splice on `:8007`** as an opt-in per-request name list; measure over the wire.
2. **Seeds 3–4 of A2 + GigaSpeech2-th test** (needs its audio) — settles the TVSpeech gain at P < 0.05 corrected.
3. **The loop's own conversations**, base vs A2 — the product test the benchmark cannot give.
4. Stage-3 review round 2 (Gemini MCP reconnected 2026-10-08) on the updated narrative.

---

## 0. The real-time line — measured 2026-10-03 → 2026-10-05

Synthesis in `RESEARCH-FINDINGS.md` `F27`–`F31`; every run, caveat and command in `idea-stage/IDEA_REPORT.md`.
682 real utts, condition `none`, frozen scorer, **canon** normaliser (`adapter/eval_canon.py`).

### The table that answers his question

| system | CER | B-CER | recall | per turn | real-time? |
|---|---|---|---|---|---|
| **NECTEC Pathumma FastConformer, beam 5** | **0.0676** | 0.213 | 0.374 | **113 ms** | ✅ deployed on `:8007` |
| picker p4 (turbo+realtime+NECTEC, Qwen3-0.6B) | 0.0681 | 0.210 | 0.380 | ~460 + 367 ms | ⚠ |
| ROVER over all 8 public systems | 0.0678 | 0.193 | 0.417 | seconds | ❌ |
| zero-shot GER over a 5-pool ROVER ⚠ test-selected | 0.0640 | 0.174 | 0.458 | seconds | ❌ |
| **typhoon-whisper-large-v3 (the bar)** | **0.0735** | 0.180 | 0.408 | — | ❌ |
| typhoon-whisper-turbo (CTranslate2 fp16) | 0.0803 | 0.213 | 0.369 | 339 ms | ⚠ |
| typhoon-asr-realtime (the old `:8007`) | 0.0945 | 0.259 | 0.328 | 65 ms | ✅ |

NECTEC vs the bar: paired bootstrap **P(not better) = 0.084** → a tie, not a significant win.

### Built (all in `adapter/` unless noted)

| file | what |
|---|---|
| `eval_canon.py` | one normaliser for reference and hypothesis — the scorer every number above uses |
| `pilot_rover.py`, `pilot_rover_cv.py`, `pilot_mbr_combine.py`, `pilot_ger.py`, `probe_contamination.py` | system combination, its honest CV, LLM correction, memorisation probe |
| `yodas_segment.py`, `teacher_label.py`, `teacher_combine.py` | YODAS2 Thai → 8-teacher labels (leak guard, teach mode) |
| `finetune_rt.py`, `_rnnt_fast.py`, `eval_rt_local.py`, `finetune_whisper.py` | student training (fast RNNT loss), in-process eval with PCM16 + latency |
| `bench_student.py`, `bench_fw.py` | per-turn latency under HF and CTranslate2 (`E:/fw-spike/.venv`) |
| `nbest_fw.py`, `nbest_rt.py`, `nbest_oracle.py`, `nbest_lattice_oracle.py`, `candidate_gain.py` | N-best lists and the picker ceilings |
| `porjai_extract.py`, `picker_lib.py`, `picker_data.py`, `train_picker.py`, `pilot_picker.py` | the picker, data to training |
| `night_picker3.sh`, `night_picker4.sh` | the two overnight runs |
| `asr/server.py`, `asr/run.ps1` | `.nemo` loading; `-Model nectec` (default) / `typhoon` |

### Data now on disk

| path | what |
|---|---|
| `E:/Claude/datasets/yodas2_th/` | YODAS2 `th000` shards 0–12, 10,199 + 37,757 segments, 8-teacher labels for shards 0–2 |
| `E:/Claude/datasets/porjai/` | Porjai Thai-central shards 0–1: 16,000 train / 1,000 held-out (by sentence), 241 leak-dropped |
| `E:/Claude/datasets/synth/` | synthetic corpus as picker data: 4,214 train / 424 held-out (21 leak + 29 entity dropped) |
| `E:/Claude/datasets/shunya_transcripts.json` | transcripts of the benchmark's `c-` source, for the overlap check |
| `E:/asr-spike/models/` | NECTEC `.nemo` |
| `checkpoints/` | `rt_student_s1..s3.nemo`, `whisper_student_w1/`, `picker_p1`, `picker_p3`, `picker_p3half`, `picker_p4` |

### Open, in order

1. **The loop's own conversations, transcribed** — the only test that decides whether the picker's
   conversational edge (p3: −7% vs NECTEC on talk) is worth ~370 ms.
2. Picker latency: one forward pass scoring options instead of `generate` (est. 30–60 ms).
3. NECTEC's training data — ask NECTEC whether Shunya val/test audio was used.

---

## 1. Measured — real numbers, reproducible

### Arm 1 — the baseline, COMPLETE (both halves of the gate)

**General accuracy**, 150 Common Voice 17.0 Thai clips (CC0, ground-truth sentences), via
`adapter/arm1_baseline.py`:

| CER mode | mean | median | p90 | perfect |
|---|---|---|---|---|
| raw (whitespace only) | 0.0723 | 0.0430 | 0.2000 | 55/150 |
| + punctuation | 0.0706 | 0.0405 | 0.2000 | 57/150 |
| **+ SARA AM folded — quote this** | **0.0518** | **0.0000** | 0.1667 | **85/150** |

15.0 min of audio, **on the GPU** (2026-09-12: `:8007` never ran on the CPU -- `from_pretrained` without `map_location` lands on `cuda:0`; see `HISTORY.md` 2026-09-12 late), **RTF 0.009-0.011**. Deliberate CPU: **RTF 0.027**. → `results/arm1_typhoon_shipped.json`

**Rare-word accuracy**, `wayu-ai/thai-contextasr-bench` with **the benchmark's own scorer**,
condition `none`, **682 real-split utterances** (439 gold entity mentions), via
`adapter/arm1_rareword.py`:

| metric | value | 95% CI |
|---|---|---|
| CER overall | 0.1125 | [0.1022, 0.1239] |
| **U-CER** — outside entity spans, the harm term | **0.1057** | — |
| **B-CER** — inside entity spans | **0.2618** | [0.2330, 0.2927] |
| **entity recall** | **0.328** | [0.2751, 0.3802] |

1.83 h of audio, **RTF 0.007-0.008 on the GPU** (not CPU -- see above). 144 of 439 gold mentions recovered.
→ `results/arm1_rareword_metrics.json`

### ⭐⭐ Contextual biasing, measured at last — `typhoon-asr-qwen-0.6b-ctx`, 2026-09-24

`adapter/arm1_rareword_qwenctx.py` → `results/arm1_rareword_qwenctx_{metrics,hyps,run}.json`.
682 real-split utterances, benchmark's own scorer, 3 conditions, 2,046 generations, ~2.4 h on GPU.
Reasoning in `F24`; **this closes `F8b`**.

| system / condition | CER | U-CER | B-CER | entity recall |
|---|---|---|---|---|
| `typhoon-asr-realtime`, none | 0.1125 | 0.1057 | 0.2618 | 0.3280 (144/439) |
| `qwen-0.6b-ctx`, none | 0.0997 | 0.0937 | 0.2214 | 0.3690 (162) |
| `qwen-0.6b-ctx`, dis@50 | 0.1013 | 0.0920 | 0.2912 | 0.3645 (160) |
| `qwen-0.6b-ctx`, **bias@50** | **0.0983** | **0.0899** | 0.2702 | **0.4715 (207)** |

- ✅ **bias@50 vs dis@50, the only legitimate comparison (`F8d`): +0.1071 recall, +47 mentions**,
  with U-CER -0.0021 and B-CER -0.0210 at the same list length. The list-length artifact by
  itself (none → dis@50) is **-0.0046 recall** — essentially nil.
- Distractor fire rate **0.0002** (card claims <=0.0004). CI95 on bias@50 recall [0.4214, 0.5283].
- ⚠ Different model, so 0.3280 → 0.4715 mixes biasing with a better ASR; quote the dis@50 delta.
- ⚠ In-process (`qwen3_asr` is not NeMo), so no PCM16 wire; `check_hyps` exit 1 is the expected
  real-split-only complaint.

### The field, rescored on the identical 682 utterances

`adapter/compare_baselines.py` — their published transcripts, their scorer, our subset:

| model | CER | U-CER | B-CER | recall | B/U |
|---|---|---|---|---|---|
| `biodatlab/whisper-th-large-v3-combined` | 0.1068 | 0.1021 | 0.2054 | **0.435** | 2.01 |
| `typhoon-whisper-large-v3` | 0.0924 | 0.0878 | 0.1829 | 0.408 | 2.08 |
| `qwen3-asr-1.7b` | 0.0860 | 0.0790 | 0.2251 | 0.367 | 2.85 |
| `nectec/Pathumma-whisper-th-large-v3` | **0.0814** | **0.0751** | 0.2062 | 0.351 | 2.74 |
| `typhoon-isan-asr-whisper` | 0.1059 | 0.0992 | 0.2371 | 0.349 | 2.39 |
| **`typhoon-asr-realtime` (ours, ~115M, GPU)** | 0.1125 | 0.1057 | 0.2618 | 0.328 | 2.48 |
| `qwen3-asr-0.6b` | 0.1051 | 0.0966 | 0.2940 | 0.253 | 3.04 |

⚠️ **Not size-matched** — ours is ~115M against ~1.5B Whispers. 🔴 **Both sides are GPU**: the "on CPU" in the original of this line was wrong, corrected 2026-09-12. Ours at RTF 0.007-0.008; forced to the CPU it is RTF 0.027, still 37x real time.

### The same table, re-scored under permissible orthographic variation (2026-09-17)

`adapter/ortho_rescore.py --rungs none num ent` — same hypotheses, same scorer, same alignment
backend (`dp`), same 682 utterances. **Only the scoring convention changed, identically for all
seven.** Rung `num`: a digit run in the reference may be read as digits, as a cardinal, or
digit-by-digit, and the rendering the hypothesis *contains verbatim* is the one scored against.

| model | CER `none` | CER `num` | Δ rel | refs rewritten |
|---|---|---|---|---|
| `typhoon-whisper-large-v3` | 0.0924 | **0.0743** | −19.6% | 75 |
| `nectec/Pathumma-whisper-th-large-v3` | 0.0814 | 0.0797 | −2.1% | 12 |
| `qwen3-asr-1.7b` | 0.0860 | 0.0819 | −4.7% | 24 |
| `typhoon-isan-asr-whisper` | 0.1059 | 0.0878 | −17.0% | 74 |
| **`typhoon-asr-realtime` (ours)** | 0.1125 | **0.0949** | **−15.6%** | 75 |
| `biodatlab/whisper-th-large-v3-combined` | 0.1068 | 0.1041 | −2.5% | 21 |
| `qwen3-asr-0.6b` | 0.1051 | 0.1043 | −0.7% | 6 |

**Kendall τ 0.429 against the table above — 6 of 21 pairs swap.** Ours goes 7th of 7 → 5th of 7
with no change to the model. The "refs rewritten" column is the mechanism: the systems that gain
are the ones that write number *words* where the corpus typed digits (`F2`). Entity recall is
identical at every rung — it is computed from the untouched hypothesis, and the script asserts it.

B-CER/U-CER range **2.01–3.04 → 2.07–3.08**, so `F3`'s ratio claim is not a convention artefact;
its CER ordering is. Full table including the `ent` and `filler` rungs →
`results/ortho_rescore_real.json`. Read `adapter/_ortho.py` before quoting any of it: 🔴 **every
number recorded elsewhere in this repo is the `none` rung, and stays that way.**

### D1 — batched beam search, measured 2026-09-12 (late)

**One config value** (`strategy: greedy_batch` → `malsd_batch`), no training, no new data. Both
arms bracketed in one window with a greedy control at each end; the control reproduced **exactly**
(0.0518 both times), so none of this is box drift.

**General CER**, Common Voice 150, SARA AM folded, GPU:

| arm | mean | median | p90 | max | perfect | RTF |
|---|---|---|---|---|---|---|
| D0 greedy_batch (control, start) | 0.0518 | 0.0000 | 0.1667 | 0.8000 | 85 | 0.011 |
| D0 greedy_batch (control, end) | **0.0518** | 0.0000 | 0.1667 | 0.8000 | 85 | 0.010 |
| **D1 malsd_batch, beam 2** ⭐ | **0.0472** | 0.0000 | 0.1667 | **0.5161** | 86 | 0.016 |
| D1 malsd_batch, beam 4 | 0.0478 | 0.0000 | 0.1562 | 0.5323 | 85 | 0.016 |
| D1 malsd_batch, beam 8 | 0.0476 | 0.0000 | 0.1562 | 0.5323 | 85 | 0.016 |
| maes_batch, beam 2 | 0.0494 | 0.0000 | 0.1667 | 0.5323 | 85 | **0.079** |

**−8.9% relative.** ⭐ **Beam width buys nothing past 2, and costs nothing either** — RTF is 0.016
at widths 2/4/8, because batch-1 decoding on a GPU is launch-bound. `maes_batch` is worse on both
axes. ⚠️ What moves is the **worst** clip (max CER 0.80 → 0.52): beam fixes greedy's catastrophic
failures, which is also why a median already at 0.0000 cannot move.

**Rare-word**, `thai-contextasr-bench` 682 real utts, condition `none`, **align `dp` on both sides**:

| | D0 greedy | D1 malsd b2 | Δ rel |
|---|---|---|---|
| CER | 0.11248 | **0.10995** | −2.3% |
| **U-CER** (harm term) | 0.10571 | **0.10335** | **−2.3%, DOWN** |
| B-CER | 0.26178 | 0.25574 | −2.3% |
| entity recall | 0.32802 (144/439) | 0.32574 (143/439) | −0.7% (one mention) |
| RTF | 0.008 | 0.013 | +63% |

**The gate passes in the easy direction** — the harm term falls. Recall is flat: beam search is not
a rare-word method, and this is the measurement that says so instead of an assumption. The gain is
uniform across all three CERs, the signature of a decoder improvement rather than an entity one.

🔴 **It first read CER 1.0000 on all 150 clips.** The checkpoint ships
`beam.return_best_hypothesis: false`, so a beam decoder returns the whole N-best **list** and
`_transcribe` read `.text` off a list and got `""`. Nothing errored. `asr/server.py` now forces the
flag and logs any empty transcript.

### D2 / D3 — n-gram fusion, measured 2026-09-12 (late)

Typhoon's own 4-gram, downloaded and built: `ngram_4gram_bpe2048.arpa` (2.7 GB text, 104.8 M
n-grams) → `ngram_4gram_bpe2048.nemo` (2.23 GiB) via `adapter/build_ngram.py`, which re-verifies the
tokenizer hash against the *served* checkpoint and refuses to build on a mismatch.
🔴 The build peaked at ~11 GB private bytes and ended with **0.7 GB RAM / 0.9 GB commit free** —
the page file is still the unfixed 7.76 GB. Load the `.nemo`, never the ARPA (17 min to parse).

**Dev set: the 150 Common Voice clips. Held-out: the 682-utt benchmark.** α was selected on dev and
reported once on held-out, which is what `A38` asks for.

| arm | dev CER (fold) | vs D0 | perfect | dev RTF |
|---|---|---|---|---|
| D0 greedy_batch (control ×3: 0.011 / 0.010 / 0.012) | 0.0518 | — | 85 | 0.010–0.012 |
| **D1 malsd_batch beam 2** | **0.0472** | **−8.9%** | 86 | 0.016 |
| D1 beam 4 / beam 8 | 0.0478 / 0.0476 | −7.7 / −8.0% | 85 | 0.016 |
| maes_batch beam 2 | 0.0494 | −4.6% | 85 | 0.079 |
| D2 greedy + 4-gram, α 0.15 / **0.30** / 0.50 | 0.0522 / **0.0516** / 0.0546 | +0.8 / **−0.3** / +5.4% | 86/85/80 | 0.030 |
| D3 beam 2 + 4-gram, α 0.05 / **0.10** / 0.15 / 0.30 | 0.0489 / **0.0463** / 0.0463 / 0.0490 | −5.6 / **−10.6** / −10.6 / −5.3% | 87/**90**/90/87 | 0.13 |
| *control:* beam 2, no LM, CUDA graphs **off** | 0.0472 | −8.9% | 86 | **0.056** |

**Held-out** — `thai-contextasr-bench`, 682 real utts, condition `none`, align `dp`:

| arm | CER | U-CER | B-CER | entity recall | RTF |
|---|---|---|---|---|---|
| D0 greedy | 0.11248 | 0.10571 | 0.26178 | 0.32802 (144/439) | 0.008 |
| **D1 beam 2** ⭐ | 0.10995 | 0.10335 | **0.25574** | 0.32574 (143/439) | **0.013** |
| D3 beam 2 + 4-gram α 0.10 | **0.10814** | **0.10146** | 0.25705 | 0.32574 (143/439) | 0.136 |

🔴🔴 **The n-gram LM improves ordinary text and does NOTHING for named entities.** U-CER is the best
of any arm while **B-CER is worse than beam alone** and recall is **bit-identical** to D1's. Entity
recall across every decode-time arm tried: **144 → 143 → 143.** That is the case for D4 and D6 being
the only remaining arms that touch the metric this plan is built on.

✅ **The 11× RTF is attributed, with a control.** Beam 2 with graphs forced off and no LM scores
CER 0.0472 — identical to graphs-on, so graphs are pure execution — at RTF 0.056. So **losing CUDA
graphs costs 3.6× and the LM's own arithmetic costs 2.3×**; only the second is intrinsic. The first
is a Windows artefact: NeMo prints *"Triton is disabled. Version without Triton is not compatible
with Cuda graphs"* and Triton has no Windows wheel. `asr/server.py` disables graphs automatically
when a fusion model is loaded (otherwise `malsd_batch` + LM dies in capture with
`RuntimeError: Capture must end on the same stream it began on`) and exposes `ASR_CUDA_GRAPHS`.

**Verdict.** ✅ D1 adoptable (+4 ms on a 6 s utterance). ❌ D2 rejected — noise at 2.6× RTF, and
`A31`'s ">50% of the beam gap at <7% overhead" did not reproduce on either axis. ⏸️ D3 parked — best
CER and U-CER of the session, but RTF 0.136 is ~816 ms on a 6 s utterance and it buys nothing on
entities. Re-cost on Triton, or keep for an offline transcription path.

All arms: `results/decoding_arms_summary.json`.

### ⭐⭐ D6 — the two-model splice, measured 2026-09-12 (night). The arm that worked.

`adapter/arm_d6_splice.py`. Typhoon's released CTC spotter finds entities on the shared frozen
encoder's per-frame posteriors; **our** RNN-T writes the transcript; spots are spliced in by frame
overlap. 682 real utts, align `dp`, tau **−1.5** (their recommended default, not tuned here).

| system | recall | gold | CER | U-CER | B-CER | RTF |
|---|---|---|---|---|---|---|
| RNN-T, no list (D0) | 0.32802 | 144/439 | 0.11248 | 0.10571 | 0.26178 | 0.028 |
| RNN-T + boosting tree (D4) | 0.35080 | 154/439 | 0.11270 | 0.10103 | 0.37307 | 0.021 |
| CTC alone, no list | 0.15718 | 69/439 | 0.14440 | 0.13465 | 0.33421 | 0.005 |
| CTC + spot, dis@50 | 0.15490 | 68/439 | 0.14617 | 0.13108 | 0.44018 | 0.019 |
| CTC + spot, bias@50 (**theirs**) | 0.48519 | 213/439 | 0.13911 | 0.12917 | 0.33015 | 0.019 |
| RNN-T + spot, dis@50 (control) | 0.32802 | 144/439 | 0.11500 | 0.10450 | 0.34710 | 0.032 |
| **RNN-T + spot, bias@50 (OURS)** | **0.55353** | **243/439** | **0.11162** | **0.10351** | **0.27493** | 0.032 |

**On the equal-list-size comparison the corrected gate demands — `bias@50` vs `dis@50`:
recall +99 mentions (+69% rel), and U-CER, CER and B-CER all FALL.** Every axis moves the right way
at once. `dis@50` lands on **exactly** the unbiased recall (144/439, bit-identical to D0), so 50 pure
decoys are provably inert and the whole gain is the gold entries.

✅ **Their published baseline reproduces here.** `A26` reports **.158** unbiased for
`typhoon-asr-realtime-nemo-ctc`; their model, their spotter, their splice on our 682 gives
**0.15718**. So the harness is validated against an external number.
⚠️ Their *boosted* .852 does **not** reproduce — we get .485 on the same pipeline. They score all
2,062 utts including the 1,380 TTS ones; we use only the 682 real. Synthetic speech is cleaner and
its entities came from the same list, which is exactly `F5`. **Read .852 as a synthetic-inclusive
figure; the real-speech number is ours.**

🔴 **THE WORD "CONTRIBUTION" IN THE NEXT PARAGRAPH IS RETRACTED, 2026-09-19.** The splice is **CTC-WS** (Andrusenko et al. 2024, arXiv 2406.07096, `LITERATURE.md` `A46`), published and shipped inside NeMo, and it covers the transducer case explicitly. The paragraph is kept verbatim below because the *measurements* in it are correct and reproducible; only the claim of authorship was false. See `RESEARCH-FINDINGS.md` `F9`.

⭐ **Keeping the better transcript is what this arm measured.** Their pipeline splices into the CTC greedy
text (CER 0.14440); splicing the *same spots* into the RNN-T text gives **+30 more gold mentions and
−20% CER simultaneously**. It is a defined operation, not a string hack: one frozen encoder, frames
aligned at ×8 / 0.08 s, tokenizers byte-identical. Cost: **RTF 0.028 → 0.032, ~14%, for a second
model** — the spotter is 1–2 ms per keyword and the encoder work is the same shape.

**The session's thesis, now closed:**

| arm | mechanism | Δ recall |
|---|---|---|
| D1 / D2 / D3 | re-rank one model's own candidates | **−1 mention** |
| D4 | bias one model's own scores | +8 |
| **D6** | **a second model's opinion of the audio** | **+99** |

⚠️ **Three measured limits.** (1) 🔴 **25% of gold mentions are unspottable by construction** — 108 of
439 are spelled in Latin and score −16 to −20, where Thai entities score −0.64 to 0.0 against decoys
at −2.4 to −4.8. The 0.554 is achieved with a quarter of the targets invisible; transliteration is
the obvious next arm, and must cover distractors too or it is an oracle leak. (2) `tau` is theirs,
unswept. (3) Only the 50-entry ladder was run; 500 is untested for this arm.

🔴 **Loading defect:** their `KeywordSpotter` calls `ASRModel.restore_from`, abstract in NeMo 3.0.0
*and* fatal on this 2.5.0 checkpoint (`Missing key key_phrase_items_list`). Worked around by
restoring `EncDecCTCModelBPE` against a generated patched config, leaving their released files
byte-identical so they stay quotable.

All numbers: `results/d6_splice_summary.json`.

### D4 — context biasing via the boosting tree, measured 2026-09-12 (night)

`adapter/arm_d4_boosting.py`. NeMo's `GPUBoostingTreeModel`, one tree rebuilt per utterance from the
benchmark's frozen bias list (3–8 ms per tree). Hyperparameters selected on the **synth** split
(1,380 clips pulled tonight, ~630 MB), reported once on **real**.

**Held-out, 682 real utts, α 1.0 / context_score 0.5:**

| condition | recall | gold | U-CER | B-CER* | CER | fires | RTF |
|---|---|---|---|---|---|---|---|
| none (control) | 0.32802 | 144/439 | 0.10571 | 0.26178 | 0.11248 | — | 0.028 |
| **bias@0** (gold only) | **0.36219** | **159/439** | 0.10196 | 0.31384 | 0.11124 | — | 0.018 |
| bias@50 | 0.35080 | 154/439 | 0.10103 | 0.37307 | 0.11270 | 0.0003 | 0.021 |
| bias@500 | 0.34169 | 150/439 | 0.09725 | 0.49528 | 0.11426 | 0.0002 | 0.031 |
| dis@500 (no gold) | 0.32346 | 142/439 | 0.09830 | 0.49588 | 0.11500 | 0.0002 | 0.031 |

⭐ **The first arm to move entity recall** after three that did not (144 → 143 → 143). No
hallucination: distractor fire rate 0.0003. General CER flat.

🔴 **Quote the same-list-size comparison, not the `none` one.** At N=500, gold-in-list vs
pure-distractor: **0.34169 vs 0.32346, +8 mentions.** That is the honest effect; the +15 against
`none` is partly the attribution artefact below. ⚠️ And either way it is ~1.1×, where Typhoon's own
`nemo-ctc` reports **.158 → .852**. Their baseline is far worse and their mechanism is different
(CTC frame posteriors), so it is not like-for-like — but it is the reason **D6 is now the decisive
arm**: re-weighting this model's own candidates has a low ceiling.

**Dev frontier** (400 synth utts, bias@50) — recall is buyable, the question is the price:

| α:score | none | 0.25:1 | 0.5:1 | **1:0.5** | 1:1 | 1:1.5 | 1:3 | 1:5 |
|---|---|---|---|---|---|---|---|---|
| U-CER | 0.02866 | 0.02774 | 0.02841 | **0.02857** | 0.03360 | 0.03949 | 0.10076 | 0.37996 |
| recall | 0.29204 | 0.31121 | 0.32596 | **0.33628** | 0.36136 | 0.39528 | 0.44100 | 0.45133 |

`1:0.5` is the last point where the harm term does not rise. At `1:5` the decoder emits almost
nothing but entity names. `H21`'s over-biasing gate is the dominant feature of this curve.

### 🔴🔴 Defect 3 — B-CER/U-CER are not comparable across bias-list SIZES

`dis@500`'s list contains **no gold entities**, yet B-CER nearly doubles (0.26178 → 0.49588). The
counts show total insertions are **flat** and merely re-attributed:

| condition | total ins | → B | → U |
|---|---|---|---|
| none | 3036 | **0** | 3036 |
| bias@50 | 2933 | 364 | 2569 |
| bias@500 | 3024 | 703 | 2321 |
| dis@500 | 2938 | **652** | 2286 |

The scorer charges an inserted run to B *"iff its string matches (⊆ or ⊇, len≥2) any form of any
entity in the **provided bias list**"* — a character-level copy of Le et al.'s word-level rule.
Measured on `dis@500`: **650 of 3,222 inserted chars (20.2%) charged, in 154 runs, 126 of them (82%)
two or three characters long.** Thai has no word boundaries and the unit is a character, so a
2-char fragment is a substring of something in a 500-entry list nearly always.

**Consequence: growing a bias list deflates U-CER and inflates B-CER for free**, so the plan's gate
("recall up, U-CER must not rise") can be passed by changing nothing. It is valid only between
conditions of equal list size — `bias@N` vs `dis@N`.

### Two NeMo defects worked around, one wire-format fact

1. **`GPUBoostingTreeModel`'s CUDA path is Triton-only with no availability check** — raises
   `ModuleNotFoundError` mid-decode on Windows. Its n-gram sibling checks and falls back; this one
   does not. `use_triton = False` selects the shipped PyTorch path.
2. **Hot-swapping `computer.fusion_models[i]` silently does nothing** (cached at construction) —
   verified byte-identical output after swapping in a gold tree. `change_decoding_strategy` per
   utterance is the only path that works.
3. **The PCM16 wire changes 3.4% of transcripts.** The in-process control differed from the server on
   23 of 682 hypotheses — not CUDA graphs, not the tree, but the 16-bit round-trip every other arm
   goes through (confirmed 23/23). Aggregate CER does not move (0.11248 vs 0.11249), so it is
   invisible in a metrics table. The harness now quantises and reproduces D0 **exactly**.

All numbers: `results/d4_boosting_summary.json`.

### ⭐⭐ Emotion transfer to Thai, zero-shot — measured 2026-09-19. No training, no GPU.

`adapter/probe_emotion.py` → `results/probe_emotion_thai.json`. THAI-SER (`airesearch/thai-ser`,
CC-BY-SA-4.0, ungated), shards 0/10/20 of 29. **1,256 clips** kept of 2,882 scanned (`majority_emo`
in the 4 classes SenseVoice can express, annotator `agreement ≥ 0.7`), 5,912 s, 22 actors, plus 760
held-out `Frustrated`. CPU only, 12 minutes.

| system | accuracy (4-way) | 95% CI | macro-recall | non-committal |
|---|---|---|---|---|
| majority-class floor (always NEUTRAL) | 0.3304 | — | — | — |
| **SenseVoiceSmall** | **0.4761** | [0.448, 0.504] | 0.4322 | 🔴 **43.6%** |
| **emotion2vec_plus_large** | **0.7381** | [0.714, 0.762] | 0.7194 | 6.1% |
| human, annotators vs assigned | 0.9286 | — | — | — |
| human, leave-one-annotator-out | 0.9511 | — | — | — |

⚠️ **Three ceilings, none interchangeable.** Both of ours are on the filtered 4-way subset, which is
*selected* for annotator agreement and therefore easier than the corpus. THAI-SER's published human
figure is **0.772**, 5-way over everything (arXiv 2507.09618). Quote the matching one or none.

⭐⭐ **The result is the refusal rate, not the accuracy.** SenseVoice emits `EMO_UNKNOWN` on **547 of
1,256 (43.6%)**, which can never score correct, so refusals alone cap it at 0.5645. **On the 709
clips where it commits it is right 84.3% — higher precision than the teacher's 78.6%.** It does not
misread Thai emotion, it abstains on it.

**Arousal transfers; valence does not.** Per-class recall ANGRY **0.668**, NEUTRAL **0.646**,
HAPPY 0.355, SAD **0.060**. The confusion is concentrated: `SAD→EMO_UNKNOWN` 151 of 200 (76%),
`HAPPY→EMO_UNKNOWN` 209 of 352 (59%), against `NEUTRAL→NEUTRAL` 268 and `ANGRY→ANGRY` 193.

✅ **The teacher clears the bar the student does not.** `emotion2vec_plus_large` — the model that
pseudo-labelled SenseVoice's 30 M SER entries (`H40`) — claims to *"overcome the effects of
language"* and **names no languages**; it now has a Thai number, 0.7381 at a 6.1% refusal rate. So
the capability is genuinely language-transferable, and what SenseVoice lost it lost in distillation
into a 5-language CTC vocabulary, not in the acoustics.

**Three secondary results, free:**

- ⭐ **The audio-event head transfers outright** — 1,245 of 1,256 clips tagged `<|Speech|>`, plus one
  `<|Laughter|>` and one `<|Cry|>`.
- 🔴 **SenseVoice calls Thai Cantonese.** No `th` slot exists in its 6-way LID, so it picks the
  nearest tonal neighbour and emits confident garbage:
  `<|yue|><|NEUTRAL|><|Speech|><|woitn|>婆话 look下你正话狂啲鞋摆嘅佢嘅 bow` on Thai speech.
  Samples kept in `sensevoice_raw_samples`.
- **2 s is the knee.** SenseVoice 0.344 (<1 s) / 0.288 (1–2 s) / **0.537** (2–4 s) / 0.493 (>4 s);
  emotion2vec 0.656 / 0.697 / 0.770 / 0.737. Prosody needs time, and THAI-SER's mean clip is 2.72 s.
  🔴 **CONFOUNDED — superseded by `F22` (2026-09-22).** These bands compare *different clips*, not
  the same clips at different lengths, so the apparent peak at 2–4 s and the drop beyond 4 s are a
  property of which utterances happen to be short, not of duration. Truncating the SAME 212 clips
  gives a **monotonic** curve that plateaus at **4 s**. The original numbers are left as written;
  do not quote them as a duration effect.
- **Script beats impro for both** — SenseVoice 0.554 vs 0.412, emotion2vec 0.786 vs 0.699. ⚠️ The
  improvised half is the honest number, and the gap is `F5`'s shape again.

🔴 **What this does NOT say.** Nothing here tested **fine-tuning** SenseVoice on Thai — the tokenizer
has no Thai, and emotion is position 2 of the *same* `ctc_lo` as the text, so Thai-only fine-tuning
would degrade it unless emotion labels stay supervised. And this is **acted** emotion from
professional actors, three fixed sentences in the script half; spontaneous Thai affect will be worse.
⚠️ One mic channel (`mic_clip`) of four; the other three are on disk and unrun.

**Environment, deliberately separate:** `D:/sensevoice-spike/.venv`, CPU torch. SenseVoice pins
`numpy<=1.26.4` and `E:/asr-spike/.venv` runs numpy 2.4.6 under NeMo 3.0.0 — installing funasr there
would downgrade numpy beneath NeMo. asr-spike verified unchanged after the run.


### ⭐⭐ Which frozen encoder carries Thai emotion — linear probe, measured 2026-09-19 (night)

`adapter/probe_encoder_emotion.py` → `results/probe_encoder_emotion.json`. Same 1,256 THAI-SER
clips as the arm above, **actor-disjoint** split (22 actors → 8 held out, 410 test clips),
regularisation chosen on dev actors and reported once on test. Mean+std pooling, identical for
every encoder; the encoder is the only thing that varies. CPU only, GPU untouched.

🔴 **The instrument is a LINEAR probe — one affine layer, deliberately too weak to be clever.** It
measures whether emotion is *present* in the representation, not what is achievable. **Every number
is a lower bound**; a deeper head can only do better, which is why a low score is the informative one.

| encoder | dim | test accuracy | 95% CI | macro-recall | gain over floor |
|---|---|---|---|---|---|
| majority class | — | 0.3537 | — | — | — |
| **mel — no encoder at all** | 160 | **0.5756** | [0.528, 0.623] | 0.5676 | — |
| **typhoon** (frozen, 17 layers, 11k h Thai) | 1024 | **0.6512** | [0.605, 0.697] | 0.6652 | **+0.0756** |
| **sensevoice** (frozen, 70 layers, no Thai) | 1024 | **0.6829** | [0.638, 0.728] | 0.7008 | +0.1073 |
| **emotion2vec** (frozen, trained for this) | 2048 | **0.7707** | [0.730, 0.811] | 0.7437 | +0.1951 |

✅ **Typhoon's ASR encoder is not empty of emotion** — 0.6512 against a 0.5756 floor and a 0.3537
majority class. ⚠️ **But the gap to emotion2vec is real and the CIs do not overlap** (0.697 < 0.730),
so the ~12-point price of reusing the ASR encoder for a single-pass design is measured.

🔴 **The mel floor is uncomfortably high at 0.5756**, so the encoders add only +7.6 / +10.7 / +19.5
over doing nothing. Almost certainly because THAI-SER is **acted** — actors project with large
energy and pitch swings that mean+std pooling of a spectrogram captures directly. **On spontaneous
Thai the floor should collapse and the ranking may reorder; none of this is measured there.**

**Per-class recall — read the macro column, not the accuracy column:**

| encoder | NEUTRAL | HAPPY | ANGRY | SAD |
|---|---|---|---|---|
| mel | 0.676 | 0.500 | 0.510 | 0.585 |
| typhoon | 0.600 | 0.618 | **0.725** | 0.717 |
| sensevoice | 0.697 | 0.600 | 0.676 | **0.830** |
| emotion2vec | **0.848** | **0.709** | 0.814 | 0.604 |

⚠️ Typhoon is **worse than raw mel on NEUTRAL** (0.600 vs 0.676) while beating it on ANGRY and SAD;
its accuracy is dragged by the largest class. And emotion2vec is the **worst of the three encoders
at SAD** (0.604) despite winning overall — the reference model is not uniformly better.

### ⭐⭐ The finding that was not the question: SenseVoice's SAD failure is in the HEAD

| | SAD recall |
|---|---|
| SenseVoiceSmall, full model, its own CTC head (morning) | **0.060** |
| linear probe on **SenseVoice's own frozen encoder** (night) | **0.830** |

**The information is in the encoder at 0.830 and the shipped head emits it at 0.060.** So the 43.6%
`EMO_UNKNOWN` abstention is not deafness to Thai emotion — it is a CTC head that shares its
vocabulary and output layer with the text declining to emit a token its own encoder has evidence for.
Corroborated from the other side: Typhoon's encoder, with *no* emotion training at all, reads SAD at
0.717.

⚠️ **Different denominators** — the morning figure is all 1,256 clips, the probe is 410 clips from 8
held-out actors. 0.060 → 0.830 is far outside split noise, but **re-scoring the morning arm on the
identical 410 clips is one cheap command and should be run before this is quoted.**

⭐ **Consequence: a new head on a frozen encoder is the right shape**, because the head is the part
that is broken — and that holds whichever encoder is chosen.


### The same arm re-scored on the 410 actor-held-out clips (2026-09-19, ⚠️ provenance unresolved)

The morning arm scored all 1,256 clips; the encoder probe held out 8 actors / 410 clips. Comparing
`0.060` against `0.830` across those two denominators was the one loose end in the finding above.
Re-scored on the **identical 410 clips, same 8 actors** — the gap widens rather than closes:

| | shipped **head** | linear probe on its own **encoder** |
|---|---|---|
| NEUTRAL | 0.607 | 0.697 |
| HAPPY | 0.391 | 0.600 |
| ANGRY | 0.696 | 0.676 |
| **SAD** | **0.019** | **0.830** |
| overall accuracy | 0.4951 | 0.6829 |

Refusal rate **40.0%** (164/410) and precision-when-committing **0.8286** on 245 clips — both within a
whisker of the full-corpus 43.6% / 0.8434. **One of 53 sad clips is called sad by the shipped model;
a single affine layer on the same features gets 44 of them.**

✅ **PROVENANCE RESOLVED 2026-09-22 — both halves reproduced, and the residue was a real defect.**
The file appeared at 21:40 on 2026-09-19 from a task nobody in-session launched, and the script on disk
had no actor filter and no `_hyps.jsonl` writer, so it could not have written it. Rather than settle who
ran what, the numbers were re-derived from scratch through an independent code path
(`adapter/finetune_emotion2vec.py`, its own audio cache, its own loader call, its own pooling):

| half | verdict |
|---|---|
| **emotion2vec 0.7317** | **exact** — all **410/410** per-clip predictions identical, identical confusion matrix, identical per-class table, on the same 410 clip ids in the same order |
| **SenseVoice 0.4951** | **408/410** — accuracy `0.495122`, macro-recall `0.428188` and SAD recall `0.018868` identical to the last digit; **2 clips differ** |

🔴 **Those 2 clips are not a discrepancy, they are a HARNESS DEFECT — see §1b, SenseVoice dithers at
inference.** The same clip, same model, same audio, six repeats, returns
`HAPPY, EMO_UNKNOWN, HAPPY, EMO_UNKNOWN, EMO_UNKNOWN, EMO_UNKNOWN`. So the 2026-09-19 file was a real
run whose script was not saved, and **this table may now be quoted** — with the caveat that SenseVoice
per-clip hypotheses are reproducible only with `dither=0`.

### ⭐⭐ THE TEACHER WAS FINE-TUNED — measured 2026-09-22. Phase 2 step 1, and it worked.

`adapter/finetune_emotion2vec.py` → `results/finetune_e2v_{zeroshot,head,top2,top4,full}_*.json`.
The question from the spec's §0: **does fine-tuning `emotion2vec_plus_large` on THAI-SER beat its
zero-shot 0.7317 on the same 410 actor-held-out clips?** Yes, by **+11.7 points**.

Four rungs, each changing exactly one thing from the one below, all scored on the identical 410 clips
of actors `001 004 108 111 114 182 185 188`. The split is imported from `probe_encoder_emotion.py` and
**asserted** at run time, so it cannot drift. Backbone learning rate swept over `1e-5 / 3e-5 / 1e-4`,
selected on the **dev** actors, test touched **once** by the dev-chosen winner.

| rung | trainable | test acc | 95% CI | macro-recall | NEU | HAP | ANG | **SAD** |
|---|---|---|---|---|---|---|---|---|
| majority class | — | 0.3537 | — | — | — | — | — | — |
| **zero-shot** (the bar) | 0 | 0.7317 | [0.689, 0.775] | 0.6848 | 0.800 | 0.700 | 0.843 | **0.396** |
| head (backbone frozen) | 4.1 K | 0.7805 | [0.740, 0.821] | 0.7571 | 0.834 | 0.718 | 0.853 | 0.623 |
| top2 | 25.2 M | 0.7829 | [0.743, 0.823] | 0.7598 | 0.855 | 0.709 | 0.833 | 0.642 |
| top4 | 50.4 M | 0.8024 | [0.764, 0.841] | 0.7830 | 0.869 | 0.682 | 0.902 | 0.679 |
| **full** | **164.0 M** | **0.8488** | [0.814, 0.883] | **0.8570** | 0.848 | 0.782 | 0.892 | **0.906** |
| human ceiling, same subset | — | 0.9286 | — | — | — | — | — | — |

⭐⭐ **SAD is the whole story: 0.396 → 0.906.** The class the shipped model could barely find is the
one full fine-tuning repairs, and it is why macro-recall (+0.172) moves further than accuracy (+0.117).

🔴 **Paired McNemar on the same 410 clips — the test that matters, because the CIs overlap.**

| comparison | net clips | p | |
|---|---|---|---|
| zeroshot vs head | +20 | 0.0105 | significant |
| zeroshot vs top2 | +21 | 0.0139 | significant |
| zeroshot vs top4 | +29 | 0.0009 | significant |
| **zeroshot vs full** | **+48** | **<0.0001** | significant |
| head vs top2 | +1 | 1.0000 | — |
| head vs top4 | +9 | 0.3057 | — |
| top2 vs top4 | +8 | 0.3742 | — |
| **head vs full** | **+28** | **0.0046** | significant |
| **top4 vs full** | **+19** | **0.0295** | significant |

⚠ **head, top2 and top4 are statistically indistinguishable from each other.** Partial unfreezing is
not separable from refitting a 4,100-parameter head at this sample size. **`full` is separable from all
three.** It is the only rung that earns its cost.

🔴🔴 **THE LEARNING RATE IS NON-MONOTONIC, AND ONE GUESS WOULD HAVE INVERTED THE ANSWER.**
`full` dev accuracy by backbone lr: **1e-5 → 0.7762, 3e-5 → 0.7119, 1e-4 → 0.8881.** The middle value is
worse than the smallest. The session's original single guess was `1e-5`, which would have reported
"full fine-tuning ≈ the frozen head" and closed the question the wrong way. **The sweep is the finding
as much as the number is.** `top4` behaves the same way (1e-5 → 0.674 dev, 1e-4 → 0.817).

⚠ **It overfits almost instantly and only early stopping saves it.** 426 training clips from SEVEN
actors against 164 M parameters: `top4` reached train loss **0.0014** by epoch 8 and its dev-selected
checkpoint was **epoch 3**; `full` bottomed at **0.0009**. Every number above depends on dev-side epoch
selection. Without it these rungs would read far worse than zero-shot.

**Improvised vs scripted — the axis that matters for the real target.** `impro` is the closest thing to
natural speech in an acted corpus, and it is where every model is weakest.

| rung | impro (n=225) | script (n=185) | gap |
|---|---|---|---|
| zero-shot | 0.7200 | 0.7459 | 0.026 |
| head | 0.7422 | 0.8270 | 0.085 |
| top4 | 0.7600 | 0.8541 | 0.094 |
| **full** | **0.8044** | **0.9027** | **0.098** |

⚠ **Fine-tuning helps improvised speech (+8.4 points) but helps scripted speech more (+15.7), so the
gap nearly quadruples.** The model is getting better at acted delivery faster than at spontaneous
delivery. **Nothing here is measured on genuinely natural Thai**, and that gap is the direction the
error will grow when it is.

✅ **Against the corpus authors' own trained baseline** (`A50`, CNN+LSTM, **WA 67.34 ± 3.05 / UA 62.61 ± 3.19**, Table 12): the fine-tuned
teacher reaches **0.8488 / macro-recall 0.8570** and sits **4 points off the human ceiling (0.9286)**.
✅ **Verified 2026-09-22 from the full text: their baseline is agreement-filtered too** (*"filtered out ... left us with a corpus that was 7 hours and 43 minutes long"*), and **UA is macro-recall, WA is accuracy** — so the metrics and the filter match and the comparison is fairer than the 2026-09-19 note assumed. The gap is **+17.5 accuracy / +23.1 macro-recall** against their ±3-point fold variance, i.e. 5–7 SD — too large for split luck. ⚠ What still stands: **one split vs 8-fold CV**, and a **164 M SSL encoder vs a from-scratch CNN+LSTM**, which is the expected direction and makes this a finding about what pretraining transfers, NOT a claim that our system is better engineered. 🔴 **And the filtered-corpus size still does not reconcile** — they report 7 h 43 m, the same filter on our shards scales to ~23 h (~18 h sans Zoom). Until that is explained, our subset is not provably their subset.

**Environment, and a device bracket that had to pass first.** Runs on `D:/sensevoice-gpu/.venv`
(`torch 2.14.0+cu130`) — deliberately the **same torch version** as the CPU venv so the device is the
only variable. ✅ **The GPU reproduced the zero-shot bar exactly before any rung was trusted on it:**
identical accuracy on train/dev/test, identical confusion matrices, **all 410 per-clip predictions
identical**, pooled features differing by at most `1.8e-3`. Measured speedup on real work: the
1,256-clip zero-shot pass went **900 s → 24 s (37×)**; fwd+bwd per clip **1.31 s → 58 ms (18.9×)**.
A 9-minute CPU epoch became ~30 s, which is the only reason a 9-run lr sweep was affordable at all.

### ⭐⭐ Can the teacher LABEL a mostly-neutral corpus? — measured 2026-09-22

`adapter/probe_pseudolabel_yield.py` → `results/pseudolabel_yield_metrics.json`. Full reasoning in
`F19`; the measured core, on the same 410 actor-held-out clips:

- **Confidence carries real but limited signal.** AUROC as a detector of its own errors: **0.737**
  zero-shot, **0.777** fine-tuned. (0.5 would have closed the selective-prediction route entirely.)
- **Raising the confidence cutoff trades quantity for purity**, measured on the fine-tuned head:
  t=0 keeps 410/410 at 78% right → t=0.9 keeps 233 at 91% → t=0.99 keeps 79 at 94%.
- **Per-class thresholds chosen on DEV, confirmed once on test, projected onto an 85%-neutral corpus:**
  fine-tuned teacher gives NEUTRAL 0.986 (t=0), HAPPY **1.000** (t=0.99), ANGRY 0.693 (t=0.6),
  SAD 0.835 (t=0.9). The zero-shot teacher **cannot reach a usable HAPPY precision at any threshold**.
- 🔴 **Yield is what kills it for the minority classes.** Per 100 k clips: NEUTRAL 70,931 correct,
  ANGRY 4,118, SAD 1,660, **HAPPY 327**. Over every Thai clip on this box (65,006 — LOTUSDIS unique
  plus Common Voice th train) that is **213 clean HAPPY**, against **~3,402** in the 26 un-downloaded
  THAI-SER shards.
- ⚠ Projections assume an 85%-neutral prior that **nobody has measured** for LOTUSDIS or Common Voice
  Thai, and use `P(pred|true)` measured on **acted** speech. Both push the real numbers downward.

### ⭐⭐ The emotion prior of READ Thai, measured — and it is not a prior at all (2026-09-22)

`adapter/probe_natural_prior.py` → `results/natural_prior_{metrics.json,hyps.jsonl}`. Full reasoning in
`F20`. 2,000 Common Voice 17 Thai test clips (2.77 h), 39 s on the GPU, plus 300 transcripts labelled
for emotion **from text alone** by a separate model that never saw the audio.

| | NEUTRAL | HAPPY | ANGRY | SAD |
|---|---|---|---|---|
| text-only labels (300) | **0.893** | 0.063 | 0.017 | 0.027 |
| the teacher, audio-only (same 300) | 0.553 | 0.207 | 0.090 | 0.150 |
| the teacher, audio-only (all 2,000) | 0.586 | 0.186 | 0.080 | 0.148 |
| `F19`'s **assumed** prior | 0.850 | 0.060 | 0.050 | 0.040 |

🔴 **chi-square of independence 5.93 on 9 df (critical 16.92) — no association.** 14 of 300 clips are
non-neutral on both sides and they agree on the class 7 times, which is chance.

- Actual yield at `F19`'s dev-chosen thresholds on 2,000 real clips: NEUTRAL 1,171 kept, SAD 165,
  ANGRY 111, **HAPPY 29**. ⚠ "Kept" is not "correct" — there are no labels here, and `F20` argues most
  of the non-neutral keeps are artifacts held at high confidence.
- Confusion-corrected (black-box shift) estimate returned 0.663 neutral — against text's 0.893. The
  correction assumes every clip has a true emotion; in a read corpus most do not.
- ✅ **The head used here was verified exact**, recovered by least squares and reproducing the recorded
  0.7805 on THAI-SER test with 100% argmax agreement, so these numbers are the `F18` head's.
- 🔴 **LOTUSDIS audio is NOT on this box** — 9 KB of metadata only (checked 2026-09-22). The
  `D:/thai-ser` acted corpus and Common Voice test are the only Thai audio available locally.
- 40 clips extracted to `D:/thai-ser-listen/` for the human listening check. **Not yet done** — it is
  the only non-circular validation of any of this.

### ⭐⭐ What sharing the ASR encoder costs the emotion head — measured 2026-09-22

`adapter/probe_shared_encoder_cost.py` → `results/shared_encoder_cost_metrics.json`. Reasoning in
`F21`. Same 410 actor-held-out clips; head capacity (`linear / mlp128 / mlp512`) and weight decay
chosen on the dev actors, test scored once.

| encoder | best head | test acc | macro-R | SAD |
|---|---|---|---|---|
| raw log-mel, no encoder | mlp128 | 0.5951 | 0.5920 | 0.660 |
| Typhoon (ASR) | mlp512 | 0.6780 | 0.6904 | 0.736 |
| SenseVoice (ASR) | mlp128 | 0.7000 | 0.7136 | 0.792 |
| **emotion2vec** | **linear** | **0.7805** | 0.7532 | 0.604 |
| all three fused | mlp128 | 0.8146 | 0.8041 | 0.736 |
| emotion2vec fully fine-tuned | linear | **0.8488** | **0.8570** | 0.906 |

- 🔴 A swept head closed **2 points of the 10-point gap**; paired McNemar best-ASR vs emotion2vec
  **p = 0.0032**. The gap is real, not a head-capacity artefact.
- ⭐ emotion2vec's winning head was **linear** — its features are already linearly separable for
  emotion; the ASR encoders needed MLPs and still lost.
- ⭐ The ASR encoders **beat** emotion2vec at SAD (0.792 / 0.736 vs 0.604) while losing on NEUTRAL.
  Fusion lifts macro-recall 0.7532 → 0.8041. ⚠ Fusion p = 0.0488 across three tried combinations —
  does not survive multiple-comparison correction.
- **The price of one pass: ~15 points** (0.7000 vs 0.8488), for ~20 ms and 164 M params.
- ⚠ **Frozen encoders only.** Multi-task fine-tuning of a shared encoder is unmeasured.

### ⭐⭐ How much audio emotion2vec needs — measured 2026-09-22, replacing a confounded table

`adapter/probe_emotion_duration.py` → `results/emotion_duration_metrics.json`. Reasoning in `F22`.
212 test clips of ≥ 4 s, each truncated **from the start** to every window, so only duration varies.

| window | frames | accuracy | macro-R | conf − acc |
|---|---|---|---|---|
| 0.5 s | 24 | 0.5943 | 0.5707 | +0.168 |
| 1 s | 49 | 0.6226 | 0.6032 | +0.146 |
| 2 s | 99 | 0.6745 | 0.6690 | +0.161 |
| 3 s | 149 | 0.7217 | 0.7145 | +0.110 |
| **4 s** | 199 | **0.7925** | 0.7872 | +0.044 |
| full (6.2 s mean) | 312 | 0.7925 | 0.7886 | +0.076 |

- **4 s is the plateau**; 4 s and full-clip agree to four decimals. Encoder runs at **50 fps**.
- **Per class:** NEUTRAL saturates at 0.5 s, ANGRY at 1 s, SAD at 2 s, **HAPPY not until 4 s**.
  Energy separates the fast pair instantly; happy-vs-angry and sad-vs-neutral need pitch **contour**,
  which costs seconds. *"Something is up"* is available in ~1 s; *which* emotion is not until ~4 s.
- 🔴 **Below 3 s the model is overconfident by 11–17 points**, so `F19`'s confidence thresholds are
  invalid there. Any selective-prediction rule needs a minimum-duration gate.
- ⚠ Truncated long clips ≠ naturally short utterances, and the cut is from the start (the harder
  case). A centred/end-anchored window is unrun.

### ⭐⭐ Our model under THAI-SER's OWN 8-fold protocol — measured 2026-09-24

`adapter/probe_thaiser_protocol.py` → `results/thaiser_protocol_metrics.json`. Reasoning in `F23`.
Protocol read out of the corpus authors' own code, not inferred: 8 folds over **studio sessions**
(5 test + 5 val each), agreement **>= 0.71**, 4 classes, Zoom excluded, **train `mic_con` / test
`[con, clip]`**, **3-second cap**.

🔴 This required the **full 29-shard corpus** (`adapter/fetch_thaiser_shards.py`, 12.71 GB):
**11,301 clips / 200 actors / 100 sessions / 12.37 h**, cached to `D:/sensevoice-spike/ft_cache_full`
— a SEPARATE directory, because overwriting `ft_cache` would make `F18`–`F22` unreproducible.
On the old 3 shards only 12 sessions existed and **7 of their 8 folds had zero test clips**.

| | WA | UA |
|---|---|---|
| ours, `mic_con` | **77.18 ± 1.98** | **73.18 ± 2.52** |
| ours, `mic_clip` (cross-channel) | 73.95 ± 2.96 | 67.19 ± 3.28 |
| their CNN+LSTM (`A50`) | 67.34 ± 3.05 | 62.61 ± 3.19 |

Per fold (`mic_con` WA): 0.766, 0.800, 0.760, 0.795, 0.776, 0.782, 0.751, 0.745 — train 7,018–7,613,
test 383–614. All 8 scored, none skipped.

- 🔴 **The naive comparison overstates by ~8 points.** `F18`'s 0.8488 vs 67.34 reads +17.5; under
  their protocol it is **+9.84**. ✅ **Decomposed by re-running the folds uncapped:** without the
  3 s cap the same arm scores **WA 82.22 ± 1.38 / UA 79.82 ± 2.72**, so the cap alone costs
  **5.04 WA (65% of the drop)** and their remaining protocol costs 2.66.
- 🔴 **+9.84 is a FLOOR:** this arm is a linear head on a FROZEN encoder (`F18`'s head rung),
  compared against their from-scratch CNN+LSTM. A full fine-tune under their protocol is unrun.
- ⭐ **Channel sensitivity, measured for free:** `con`→`clip` costs **3.2 WA / 6.0 UA**. UA takes
  double the hit. Every prior emotion number here was `mic_clip`.
- ⚠ The **2,516 Zoom clips** in the new cache are unscored — they need a `mic_zoom` cache pass.

### ⭐⭐ The only public Thai SER model, benchmarked — 2026-09-26

`adapter/bench_public_thai_ser.py` → `results/bench_public_thai_ser_metrics.json`. Reasoning in
`F25`. `awghuku/wav2vec2-base-thai-ser` scored under the identical `F23` protocol.

| system | WA | UA |
|---|---|---|
| `awghuku/wav2vec2-base-thai-ser` | **55.15 ± 4.74** | **45.84 ± 3.66** |
| THAI-SER published baseline | 67.34 ± 3.05 | 62.61 ± 3.19 |
| ours, 3 s cap | 77.18 ± 1.98 | 73.18 ± 2.52 |
| ours, uncapped | 82.22 ± 1.38 | 79.82 ± 2.72 |

- **First published number for that model by anyone** — it ships no card, no metrics, 19 downloads.
- ✅ Fairness: its frustrated logit is dropped (never scored wrong) and it got the best of 120
  label orderings. It still loses to the corpus baseline by 12 points.
- 🔴 **Upper bound** — its train/test split is undocumented; our test folds may be its train data.
- 🔴 **It does not load on current transformers** (`id2label` integer values fail strict
  validation); the benchmark repairs the config in memory.
- 🔴 **Label order undocumented and counter-intuitive:** true mapping `ANGRY=0, NEUTRAL=1,
  HAPPY=2, SAD=4` (stable across all 8 folds); the corpus authors' documented column order gives
  **WA 8.56%**, below the 35% majority-class floor.

### ✅ A checkpoint exists at last — `checkpoints/thaiser_head_3s_v1.pt` (2026-09-26)

Every emotion number before this date came from weights discarded at process exit. `F26`.
51 KB: the **unmodified public `emotion2vec_plus_large`** backbone is named but not included;
what is saved is our 4-class head (`4 x 2048`), the feature mean/std it expects, the protocol,
the scores and the caveats. 🔴 The **full fine-tune** (`F18`'s best rung) still has no checkpoint
and has never been run under their protocol.

### THAI-SER's own published baseline — the number our zero-shot results should be read against

From the corpus paper (`arXiv:2507.09618`, `A50`): a CNN+LSTM (64 mel filterbanks, 3-second chunks,
VTLP augmentation, **8-fold speaker-independent** cross-validation, Zoom sessions excluded), **trained
on THAI-SER**, evaluated on the same four emotions this repo scores:

| system | 4-emotion accuracy | trained on Thai? |
|---|---|---|
| THAI-SER's own CNN+LSTM baseline | **WA 67.34 ± 3.05**, UA 62.61 ± 3.19 | ✅ yes |
| **`emotion2vec_plus_large`, zero-shot** | **0.7317** (410 held-out) · 0.7381 (1,256) | ❌ **no** |
| linear probe on SenseVoice's frozen encoder | 0.6829 | ❌ no |
| SenseVoice's shipped head | 0.4951 | ❌ no |
| human ceiling (this repo, same subset) | 0.9286 | — |
| human recognition accuracy (their paper, 5-way) | 0.772 | — |

⭐⭐ **A model that has never heard Thai, with no training at all, matches or beats the corpus authors'
trained baseline.** ⚠️ Not like-for-like — they run 8-fold CV over studio sessions, we use 8 held-out
actors from 3 of 29 shards — but the margin survives that difference and is the strongest single
argument for the emotion half of this project.

🔴 **Corpus reliability, and a number that did not reproduce.** The raw corpus scores Krippendorff's
alpha **0.413**, *below* the 0.667 recommendation; filtering to `agreement >= 0.71` lifts it to
**0.692**, and the authors state that filter as an instruction. ✅ Every probe here filtered at 0.70.
⚠️ The paper reports the filtered corpus as **7 h 43 m**; measured on our three shards the same filter
keeps **54.6% of clips and 55.0% of the hours** (2,882 → 1,575; 3.85 h → 2.12 h), scaling to ~23 h, and
~18 h if Zoom is also dropped. **The gap is unexplained.** Quote "agreement-filtered", never an hour count.


---

## 1b. 🔴 Harness defects found under those numbers (two here, a third and fourth under D4 above)

### `:8007` runs on the GPU, and always has — every "CPU" figure in these docs is a GPU figure

`asr/server.py` called `ASRModel.from_pretrained(MODEL_NAME)` with no `map_location`; NeMo defaults
that to `cuda:0`. Confirmed by `next(m.parameters()).device`, and by the fact that forcing
`map_location="cuda"` reproduces the recorded arm-1 baseline **byte-identically** (150/150
hypotheses) while forcing `cpu` changes exactly one (for the better) and moves mean CER to 0.0513.

| | GPU (what has always run) | CPU (measured on purpose) |
|---|---|---|
| general CER, fold | 0.0518 | 0.0513 |
| RTF, Common Voice | 0.010–0.011 | **0.027** |
| VRAM held | **~0.95 GB** | 0 |

~66 ms → ~162 ms on a 6 s utterance, still 37× real time. **The ~0.95 GB is a real lever** toward
the streaming ASR's 2.9 GB, and it was inside the 2026-09-06 "box is full at 14.6 of 16.3 GB"
measurement unattributed. `ASR_DEVICE` is now explicit and `/health` reports the live device.

### 🔴 SenseVoice DITHERS AT INFERENCE — its per-clip output is not reproducible (found 2026-09-22)

Found by chasing a 2-clip difference between a re-score and the 2026-09-19 run instead of rounding it
away. `funasr`'s `WavFrontend` is configured **`dither = 1.0`** and applies it unconditionally, so
random noise is added to the waveform before the filterbank on **every inference call**.

Same clip, same model, same audio, six repeats:

```
s001_con_actor001_script1_1_3a  ->  HAPPY, EMO_UNKNOWN, HAPPY, EMO_UNKNOWN, EMO_UNKNOWN, EMO_UNKNOWN
s074_con_actor188_script1_2_3b  ->  ANGRY, ANGRY, ANGRY, ANGRY, ANGRY, EMO_UNKNOWN
```

Setting `frontend.dither = 0.0` makes both perfectly stable across six repeats.

**Ruled out, in order:** it is not thread non-determinism (it persists at `torch.set_num_threads(1)`);
it is not dropout (every one of the 211 dropout modules reports `training=False`, and `specaug` **is**
correctly gated by `self.training` in `sense_voice/model.py:832`). Seeding `torch` and `numpy` before
each call makes it deterministic, which is what pointed at a live random op in the frontend.

✅ **Scope — this does NOT touch the repo's ASR numbers.**

| stack | dither at inference | |
|---|---|---|
| **NeMo** (Typhoon, `:8007`, every ASR arm here) | guarded: `if self.training and self.dither > 0`, commented *"only in training mode for eval determinism"* | ✅ deterministic |
| **funasr `WavFrontend`** (SenseVoice) | `dither=1.0`, ungated | 🔴 **nondeterministic** |
| **emotion2vec** | no `WavFrontend` — raw waveform + `layer_norm` | ✅ deterministic |

⭐ That third row is the confirmation rather than a guess: it is exactly why emotion2vec reproduced
**410/410** while SenseVoice reproduced **408/410** in the same provenance check.

**Magnitude:** ~0.5% of clips flip, and the flips landed on the `EMO_UNKNOWN` boundary — which is
precisely the abstention behaviour the SenseVoice analysis is about. Aggregate metrics survived
unchanged in the one paired observation available. **But per-clip hypotheses are not reproducible
unless dither is disabled, and byte-exact re-runs are this repo's whole discipline** (`F8c`).
**Any future SenseVoice arm must set `dither=0` and say so in its `_run.json`.**

### The benchmark's B-CER/U-CER depend on whether `kaldialign` is installed

Same scorer, one byte-identical hypothesis file, three repeats each (the scorer is deterministic —
seeded bootstrap):

| align backend | CER | U-CER | B-CER | recall |
|---|---|---|---|---|
| DP (no kaldialign) | 0.11248 | 0.10571 | **0.26178** | 0.32802 |
| kaldialign | 0.11248 | 0.10622 | **0.25050** | 0.32802 |

CER and recall need no span attribution and do not move; B-CER/U-CER do, because they attribute
each edit inside/outside a gold span **via the alignment**, and the two implementations break ties
differently. `E:/asr-spike/.venv` has kaldialign, `E:/voxcpm-spike/.venv` does not — **everything
published above was scored by the VoxCPM interpreter, so all of it is on the DP path and mutually
consistent**, but a later arm run from the other venv would have shown a 0.011 B-CER "win" that is
pure tie-breaking, on the metric the whole experiment is gated on.
**Fixed:** `adapter/_score.py` pins the backend, defaults to `dp`, and refuses to run if it got the
other branch. `arm1_rareword.py` scores only through it and records `align_backend`.

### The checkpoint, read rather than guessed

`typhoon-asr-realtime.nemo`, unpacked: `EncDecRNNTBPEModel`, ConformerEncoder **17 layers,
d_model 512**, `dw_striding` ×8, `rel_pos`, 8 heads, conv kernel 9, `att_context_size [-1,-1]`,
`conv_norm_type: **batch_norm**` with **51 running-stat buffers**, `strategy: **greedy_batch**`.

| | params |
|---|---|
| encoder (frozen under S1) | **108.78 M** (94%) |
| decoder + joint (trainable in phase 2) | **6.65 M** (6%) |
| bottleneck-32 adapter only | **0.57 M** (0.5%) |
| total | **115.45 M** |

## 2. Built and verified

| artefact | what | verification |
|---|---|---|
| `adapter/_thai.py` | the Thai normaliser — SARA AM folding for scoring, decomposition for labels | the numbers in §1 |
| `adapter/arm1_baseline.py` | general CER, three normalisation modes, all reported | ran, 150 clips |
| `adapter/arm1_rareword.py` | rare-word half via the benchmark's own scorer | ran, 682 utts; denominators hand-checked (`gold: [144, 439]`) |
| `adapter/compare_baselines.py` | the field rescored like-for-like | reproduced the table exactly |
| `adapter/make_text_banks.py` | the two text banks via `agy` (Gemini) | 440 + 245 rows |
| `adapter/check_text_banks.py` | cross-bank auditor | ran on both |
| `adapter/fix_label_convention.py` | relabels into Typhoon's conventions | 3 axes, assertions pass |
| `colab/gen_colab.py` | Colab TTS generator — tar shards to Drive, resumable | **resume tested**: 25 of 46, re-ran, finished at 46, **zero duplicate gen_ids** |
| `colab/Generate_Thai_Synthetic.ipynb` | 16-cell notebook | valid JSON, 16 cells |
| `%TEMP%\colab-bundle.zip` | upload bundle | **52.0 MB, 153 entries** |

### The text banks

| bank | rows | content |
|---|---|---|
| `adapter/texts/phase1_known.json` | **440** | everyday spoken Thai, 40 each over 11 call-centre themes, no digits, no proper nouns, 440/440 unique, **one** repeated opening in the whole bank |
| `adapter/texts/phase2_rare.json` | **245** | one rare entity per row — `place` 155, `product` 90, **245/245 distinct**, entity present in label 245/245 |

At `--per-text 8`: **5,480 clips ≈ 10.2 h of audio ≈ 1.09 GB ≈ ~3 h of generation.** Voices are
assigned so a text's N clips are N **different** speakers (coprime stride — guaranteed, not chance).

### Data pulled

| | size | licence | where |
|---|---|---|---|
| `wayu-ai/thai-contextasr-bench` — real split + scorer + 6 baselines | 247 MB | **CC-BY-NC-4.0** — eval only, never training or product | `E:/asr-bench/thai-contextasr-bench` |
| `typhoon-asr-realtime-nemo-ctc` + `word_boost.py` + `kws_spot.py` | 516 MB | CC-BY-4.0 | `E:/asr-bench/nemo-ctc` |

## 3. Housekeeping done

- **C: 11.9 → 50.53 GB free (+38.6 GB).** WSL caches ~5 GB, three dead quantisation exports 24 GB,
  a duplicate CosyVoice3 checkpoint 12.9 GB. Mechanism: the vhdx is already NTFS-sparse and WSL's
  ext4 discards, so deleting inside returns space to Windows — **flushed by `wsl --terminate`**.
- ✅ Confirmed nothing broke: the CosyVoice delete was a **duplicate**; `run_vllm_server.sh:45` loads
  v3 from E:, still present.

## 4. Literature — the ledger grew a lot

| section | before | after |
|---|---|---|
| `A` — ASR | 29 | **38** (`A29` our benchmark result; `A30`–`A38` decoding/accuracy) |
| `E` — endpointing | 58 | **67** (`E59`–`E67` joint ASR+endpointing architecture) |
| `H` — synthetic speech | 14 | **36** (`H15`–`H36` contextual biasing + synthetic) |

All via Consensus, so every row is citable. Corrections written into `A26`, `E2`, `H3` and the
design spec's §2.

## 5. Known traps carried forward

🔴 **The box drifts 25–45% across an evening, unequally between engines.** Bracket everything being
compared back-to-back in one window; discard the first sweep after any server start.

🔴 **Start order: llama-server BEFORE VoxCPM**, or VoxCPM takes the VRAM and llama-server dies loudly.

🔴 **Capture stderr first, conclude second.** Every wrong turn in the 2026-09-08 session came from
reading an empty log as evidence.

⚠️ **`check_hyps.py` reports FAIL on our 682-utt subset by design** — it wants all 2,062. The scorer
handles the subset correctly; the denominators were verified by hand.
