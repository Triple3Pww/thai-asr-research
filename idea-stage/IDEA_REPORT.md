# Idea Discovery Report — beat the best Typhoon ASR

**Direction** (his words, 2026-10-03): *"the asr model have better score than the best typhoon model right now."*
**Date**: 2026-10-03 · **Run**: `idea-beat-typhoon-2026-10-03` (inside pipeline run `beat-typhoon-2026-10-03`)
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline

> ➡️ The next run (2026-10-06/07, "DeepSeek for Thai ASR") is `IDEA_REPORT_2026-10-06.md`; this file is unchanged below.

> 🔴 This file is a **plan and a pilot record**, not a finding. Nothing here moves a row in
> `LITERATURE.md` or a line in `RESEARCH-FINDINGS.md` until a run named in a `RESULT:` block does.

## Executive Summary

_(filled at Phase 5)_

## Literature Landscape

### What "the best Typhoon" is — read from the vendor's own cards (2026-10-03)

`typhoon-ai` has published **16 ASR checkpoints**; 8 were released **after** this repo's 7-system
table (enumerated via the HF API, `sort=lastModified`, per the CLAUDE.md rule). Their own cards
score them on **`wayu-ai/thai-contextasr-bench`, all 2,062 utts (real + TTS split)** — which this
repo does not use (TTS split is circular for a synthetic-data method, `S7`). So their numbers are
**not** ours, but the ordering is informative:

| model | params | bench CER% none / recall | best biased CER% / recall | gigaspeech2 / TVSpeech CER% |
|---|---|---|---|---|
| **typhoon-whisper-large-v3** | 1.55B | **4.83** / .459 | N/A (no list support) | **4.69 / 6.32** |
| typhoon-asr-qwen-1.7b-ctx | 2.07B | 5.11 / .441 | **3.39 / .754** (bias@0, gold) · 4.30 / .640 (bias@50) | 4.86 / 6.87 |
| typhoon-asr-qwen-0.6b-ctx | 0.80B | 5.71 / .397 | 4.01 / .668 | 5.06 / 7.28 |
| typhoon-whisper-medium | 769M | 7.05 / .383 | 4.92 / .658 | 4.81 / 7.66 |
| typhoon-asr-realtime-nemo-ctc | 135M | 9.56 / .158 | 5.97 / **.852** | 8.54 / 13.46 |
| typhoon-asr-streaming-nemotron-0.6b | 0.6B | — | — | 9.3 / 14.1 (streaming @1040 ms) |

**So, by the vendor's own measurement: on plain transcription the best Typhoon is still
`typhoon-whisper-large-v3`; with a bias list it is `typhoon-asr-qwen-1.7b-ctx`.** Every Typhoon
model is trained on **~10,800–11,000 h** of Thai (>95% GigaSpeech2-th).

### The routes, and what the literature says each is worth

| route | evidence | fits a 16 GB box + 10.57 h synthetic? |
|---|---|---|
| **Out-train it** (fine-tune on our corpus) | `F10`: ~40 papers on synthetic rare-word training; best real/synth ratio `H39` 1.08 — but on 960 h-scale English. Typhoon already holds 1,000× our real data | ❌ for **general CER**. ✅ plausible for **entity recall** only, and `F10` says that ground is crowded |
| **System combination** (ROVER / MBR over several ASRs) | ROVER over Whisper + Canary + Parakeet: WER 6.26 → 3.92 (−37%) vs Whisper-large-v3 alone; hybrid+E2E ROVER −14% / −20% noisy. Gain is driven by **system diversity** (theory + measurements) | ✅ zero training; 7 diverse Thai systems' hypotheses are **already on disk** |
| **Generative error correction** (LLM reads N-best, writes the transcript) | HyPoradise (NeurIPS 2023): fine-tuned GER **beats the N-best oracle** of re-ranking; zero-shot LLM is weaker. Thai: Typhoon2-8B as an ASR *decoder* gave only −3.4% rel. CER on GigaSpeech2 (arXiv 2505.22063) | ⚠ an 8B LLM fits in 16 GB at 4-bit; training data for a GER needs N-best on *real* audio we lack labels for |
| **Shallow fusion** of an n-gram | `A53`: up to 51% rel. on low-resource CTC/attention — 🔴 `D2`/`D3` found it noise on our RNN-T | ⚠ untested on Whisper here |
| **Contextual biasing** | `F15`: closed as a *contribution*; `F24`: qwen-0.6b-ctx bias@50 recall 0.4715 on our split | usable as a component, never as the claim |

### Structural gap

No Thai paper found combines **multiple independent public Thai ASRs** — despite at least six
strong, architecturally diverse ones now existing (two Whisper fine-tunes from different groups,
two Qwen3-ASR sizes, Typhoon's own RNN-T/CTC/Qwen line). ⚠ "Not found" here means three web
searches and the ledger, not an enumeration — see Novelty Verification.

Sources: [HyPoradise](https://arxiv.org/abs/2309.15701) ·
[Thai LLM-based ASR, Typhoon2 decoder](https://arxiv.org/pdf/2505.22063) ·
[Sommelier (ROVER w/ Whisper+Canary+Parakeet)](https://arxiv.org/pdf/2603.25750) ·
[NTT CHiME-8 system](https://arxiv.org/pdf/2502.09859) ·
[typhoon-asr-qwen-1.7b-ctx card](https://huggingface.co/typhoon-ai/typhoon-asr-qwen-1.7b-ctx) ·
[typhoon-whisper-medium card](https://huggingface.co/typhoon-ai/typhoon-whisper-medium)

## Ranked Ideas

All pilots: 682 real utts, condition `none`, scored by the benchmark's frozen scorer (`_score.py
--align dp`) at the F17 **`none`** and **`num`** rungs. **`num` is the comparison that counts** —
see the convention trap below. Pool keys: `tw` typhoon-whisper-large-v3, `pat` Pathumma-large-v3,
`q17`/`q06` Qwen3-ASR 1.7B/0.6B, `isan` typhoon-isan-whisper, `bio` biodatlab-large-v3,
`qctx` typhoon-asr-qwen-0.6b-ctx, `rt` typhoon-asr-realtime (ours).

### 🏆 I1 — ROVER over every public Thai ASR, no pool selection — PILOT POSITIVE
Token-level ROVER (`adapter/pilot_rover.py`) in the scorer's own token space; pivot = MBR medoid;
uniform votes; label-free.

| system | CER none | **CER num** | U-CER num | B-CER num | recall |
|---|---|---|---|---|---|
| **ROVER all-8** | 0.0835 | **0.0687** | **0.0624** | 0.1956 | **0.4169** |
| typhoon-whisper-large-v3 (the bar) | 0.0924 | 0.0743 | 0.0688 | **0.1829** | 0.4077 |
| Pathumma-large-v3 | 0.0814 | 0.0797 | 0.0734 | 0.2056 | 0.3508 |

**−7.5% relative CER vs the bar**, paired bootstrap (own tokeniser, 2,000 draws) **P(ROVER ≥ tw) =
0.0015**, and it wins in **all three source groups** (conversational `a` 0.134 vs 0.143,
`c-test` 0.081 vs 0.085, `c-validation` 0.042 vs 0.044). 🔴 **B-CER is worse** (0.196 vs 0.183):
voting dilutes the one system that is best on names. Recall still edges up because recall is
per-mention, B-CER per-character.

### 🔴 The two traps this pilot walked into, and how each was caught
1. **Convention.** On raw CER, ROVER over a hand-picked 5 looked like **−20%** (0.0741 vs 0.0924).
   Most of that was the vote landing on **digits**, the corpus typist's convention, against
   Typhoon's number words (`F2`/`F17`) — `c-validation-0023` alone swings 101 tokens. At `num` the
   same pool is 0.0683: real, but ~8%, not 20%.
2. **Selection on test.** The best of 219 pools was chosen on all 682 utterances. Re-done honestly
   (`adapter/pilot_rover_cv.py`: choose the pool on two source groups, score on the third), the
   CV-chosen ROVER is **0.0866 — WORSE than Typhoon (0.0846)**, P = 0.93. Pool selection overfits,
   most of all on the conversational group (tw+pat+bio: 0.154 vs tw 0.143). **The only honest
   configuration is "every system, no selection" — which is the one that wins.**

### I2 — Zero-shot generative error correction over the ROVER consensus — PILOT WEAK POSITIVE (n=90)
`adapter/pilot_ger.py`: a local LLM (Ollama) gets the ROVER line + every hypothesis, writes one
line; falls back to ROVER if it drifts >30% from it (1 of 90 did). Same 90 utterances:

| | CER none | CER num | B-CER num | recall |
|---|---|---|---|---|
| GER qwen3.5:9b over 5-pool ROVER | **0.0653** | **0.0602** | **0.1795** | 0.3934 |
| 5-pool ROVER | 0.0687 | 0.0636 | 0.1963 | 0.3934 |
| typhoon-whisper-large-v3 | 0.0765 | 0.0652 | 0.1808 | 0.3443 |

⚠ n=90 and **the 5-pool was test-selected** — full 682 runs on the all-8 (unselected) pool are in
flight. ⚠ `typhoon2.1-gemma3-12b` (the Thai LLM) **does not load** on this Ollama
(`selectattr: unknown test 'tool_calls'` in its chat template, raw mode too) — untested, not ruled out.

### I3 — Distil the ensemble into ONE model (pseudo-label real Thai audio with I1/I2, fine-tune) — NOT PILOTED (>2 h)
Answers his literal ask — *"the asr model"*, singular — which I1/I2 do not: they are systems of
8 models, 8× the compute of the bar. Needs unlabelled real Thai audio outside every pool member's
training set (risk: GigaSpeech2 is already Typhoon's data).

### I4 — Fine-tune typhoon-whisper-large-v3 on the synthetic corpus — NOT PILOTED, DEPRIORITISED
10.57 h synthetic vs ~11,000 h real behind the bar; `F10` crowded; plausible only for entity recall.

### Eliminated
- **MBR selection** (pick one system's whole line): superseded by ROVER before it finished; can only
  ever reach the per-utterance oracle of whole lines.
- **CV-selected pools**: lose to Typhoon (above).
- **Agreement-weighted ROVER**: 0.0879 none vs 0.0876 uniform — no gain.

## Novelty Verification

**Reviewer:** Gemini 3.8 Flash (High) via `gemini-review` MCP (agy backend), thread
`876be4d3f0504ba492e7421af9254342` — cross-family; Codex MCP is not installed on this box.
**Verdicts:** I1 ROVER **ABANDON as a research claim** · I2 GER **PROCEED WITH CAUTION** ·
I3 distillation **PROCEED** · I4 synthetic FT **ABANDON** · overall **PROCEED WITH CAUTION (pivot to I3)**.

| idea | method novel? | closest work |
|---|---|---|
| I1 ROVER | ❌ | Fiscus 1997; CNC (Mangu 2000); Whisper+Canary+Parakeet ROVER recipes |
| I2 GER | ❌ | HyPoradise (NeurIPS 2023); multi-ASR + LLM fusion (arXiv 2506.11089); West Frisian GER (2605.19711) |
| I3 distil | ❌ method, ✅ as an empirical Thai system | arXiv 2506.11089 (multi-ASR fusion + SpeechLLM correction → pseudo-labels) is the **closest and must be cited** |
| I4 | ❌ | SynthASR; Rossenbach 2020 — and `F10` |

So, consistent with `F10`: **"first Thai", never "new method".** None of the reviewer's
concurrent-work examples were checked against the bench by the reviewer; its claim that Typhoon2-Audio
/ Qwen-Audio would beat a cascade is **an assertion, not a measurement** — `typhoon-asr-qwen-*-ctx`
IS such a model and is in the pool at 0.0806.

### The reviewer's three threats, each answered by a run

**A. "`num` is a moving target."** Fair. `adapter/eval_canon.py` applies ONE function to reference
and hypothesis alike (separators dropped, every digit run → Thai number words — Typhoon's own
convention, so the bar is not penalised). Frozen scorer otherwise. 6 entity spans touched.

| system (canon) | CER | U-CER | B-CER | recall |
|---|---|---|---|---|
| GER qwen3.5:9b over 5-pool ROVER ⚠ pool test-selected | **0.0640** | **0.0585** | **0.1735** | **0.4579** |
| ROVER all-8 (no selection) | 0.0678 | 0.0617 | 0.1933 | 0.4169 |
| **typhoon-whisper-large-v3 (bar)** | 0.0735 | 0.0682 | 0.1798 | 0.4077 |
| Pathumma | 0.0762 | 0.0701 | 0.2016 | 0.3508 |

The ordering under `canon` matches `num`. ⭐ **GER is the first configuration to beat the bar on
all four metrics, B-CER included** — the LLM undoes the entity damage ROVER's vote does.

**B. Contamination.** `adapter/probe_contamination.py`: the LLM is given the first half of each
REFERENCE sentence and asked to continue it. **0 of 120 near-verbatim** (continuation CER <0.1);
mean continuation CER `a` 0.899 · `c-test` 0.854 · `c-validation` 0.901 — the public-text groups
are no more memorised than the in-house conversational control. ⚠ **But the GER gain is
concentrated on the read-speech groups** (CER canon, own tokeniser):

| group | GER | ROVER-8 | tw |
|---|---|---|---|
| `a` conversational (256) | 0.1363 | **0.1342** | 0.1428 |
| `c-test` (186) | **0.0720** | 0.0809 | 0.0858 |
| `c-validation` (240) | **0.0342** | 0.0397 | 0.0423 |

Not memorisation — but GER helps **written-register** Thai and does nothing for conversation.
That bounds the claim.

**C. Test-set selection.** Already answered in the pilot: CV-chosen pools lose; only the
no-selection all-8 pool is reported as a system. The GER row above still uses the test-selected
5-pool — the all-8 GER run is the one that counts.

## 🔄 Direction 2 (2026-10-03, later): *"make the a model that beat typhoon best score that can be use in realtime."*

The ROVER/GER system is 8 models + an LLM: not one model, not real-time. So it becomes the
**teacher**, and a single fast **student** is trained on its labels (I3, the reviewer's PROCEED).

**Which student — measured, not assumed** (batch 1, fp16, RTX 5060 Ti, 682 real utts, CER canon):

| candidate | CER | p50 / p90 per turn | verdict |
|---|---|---|---|
| typhoon-whisper-turbo (809M) | 0.0789 | 1,229 / 2,101 ms (RTF 0.129) | ❌ too slow for the loop, and still loses to the bar |
| **typhoon-asr-realtime (115M RNNT)** — `eval_rt_local.py`, PCM16 wire | 0.0945 | **65 / 298 ms** (RTF 0.015) | ✅ the student |
| typhoon-whisper-large-v3 (the bar) | 0.0735 | not finished — killed at 47 min; slower than turbo by construction | ❌ |

⚠ typhoon-asr-realtime is **full-context** (`att_context_size [-1,-1]`): it transcribes a finished
turn, it does not stream. That is how the voice loop uses it, so "real-time" here means per-turn latency.

**Student gap:** 0.0945 → must beat 0.0735 (−22%). Teacher sits at 0.0640–0.0678.

**Teacher data:** `espnet/yodas2` `th000` shards 0–2 (CC-BY-3.0, ungated; GigaSpeech2 is gated and
there is no HF token on this box) → 10,199 caption segments, 2–20 s, **13.0 h**
(`adapter/yodas_segment.py`). ⚠ Benchmark `a-` clips are GigaSpeech2 **test**, cut from YouTube —
same platform as YODAS — so `teacher_combine.py` drops any segment sharing a 20-char run with any
of the 2,062 benchmark references.

### Student round 1 — NEGATIVE (2026-10-03)

All 8 teachers over 10,199 YODAS segments (13.0 h; ~5 h GPU, RTF 0.04–0.06 per Whisper/Qwen
teacher). `teacher_combine.py` kept **8,636 segments / 10.88 h** — leak guard **27** (YouTube
overlap with the benchmark is real), disagreement 1,526, shape 10. Fine-tune: 3 epochs, LR 5e-5,
bs 8×2, fast RNNT loss, 1,589 steps, ~20 min.

| | CER canon | U-CER | B-CER | recall | p50 / p90 per turn |
|---|---|---|---|---|---|
| typhoon-asr-realtime (in-process, PCM16) | 0.0945 | 0.0873 | 0.2586 | 0.3280 | 65 / 298 ms |
| **student s1** | **0.0943** | 0.0866 | 0.2563 | **0.2984** ↓ | 59 / 66 ms |

511/682 transcripts changed, but they cancel: read-speech groups improve ~4%, the GigaSpeech2 `a`
group gets worse (0.156 → 0.158), and entity recall drops 3 points.

🔴 **Why: the confidence filter removed the teaching signal.** Disagreement was measured over all
8 systems, the student included — so "confident" meant "the student already agrees". On the held-out
teacher split the untouched student was only **3.4% CER from its own labels**. The 1,526 dropped
segments were the ones that could teach. Round 2 (`--student rt`): confidence and the label both
come from the 7 OTHER systems; train on every segment the student gets wrong (>5% vs label) plus an
equal number it gets right → **3,366 teach + 3,366 easy, 8.65 h**.

### Rounds 2–4 — all NEGATIVE; the cause found (2026-10-03/04)

| run | student | data | CER canon | recall |
|---|---|---|---|---|
| base | typhoon-asr-realtime | — | 0.0945 | 0.3280 |
| s1 | rt | all-8 ROVER, 10.9 h | 0.0943 | 0.2984 |
| s2 | rt | teach mode (7 judges, student-wrong + replay), 8.65 h | 0.0940 | 0.3007 |
| s3 | rt | same, LR 3e-4 | 0.0978 | 0.2802 |
| base | typhoon-whisper-turbo (CT2 fp16, 339 ms p50) | — | 0.0803 | 0.3690 |
| w1 | turbo, decoder-only | all-8 ROVER, 10.9 h, 2 ep | 0.0816 | 0.3371 |

**Why — measured against the YODAS uploaders' captions** (8,030 segments where some system is
within 30% of the caption, i.e. the caption is a transcript and not spam):

- Over all segments the teacher IS better than the bar: ROVER-8 **0.119** vs typhoon-whisper-large-v3 0.142.
- But on the 190 held-out-video segments of w1: **base turbo 0.074** · ROVER-8 0.079 · w1 0.096 · large-v3 0.147.
  **Untouched turbo beats the teacher on YODAS**, so distilling the teacher into it can only pull it down —
  which is what happened. Turbo is 2× better than large-v3 here while slightly *worse* on the benchmark;
  its card lists "Internal Curated Public Media" in its ~11k h. 🔴 **Inference, not verified:** Typhoon's
  newer models have seen YouTube Thai of this kind, possibly these videos. 4 held-out videos only.

**Conclusion:** YODAS + this teacher cannot produce a single real-time model that beats the bar. Four
runs, two students, two label schemes, two learning rates — the failure is the data source, not a knob.

### 🧭 Direction research (2026-10-04) — Consensus, 9 queries; why the four rounds failed and what can work

**1. Why 10.9 h could not move it — the literature's scale.** Every published win distils into a
student that started WEAK, with hundreds+ of hours: transducer KD from a big teacher gave 6.8–11.5%
with no extra data and **38.5–48.2% only once 860 h of unlabelled audio were added**, over a 100 h
baseline [Yang 2021]; a ROVER of diverse CTC + RNN-T teachers distilled into a streaming student
gave **−13% to −41% WER** [Doutre 2021]. For a baseline already trained on **10,000 h**, large-scale
semi-supervised training gave only **2–11%** relative, and *"the audio attributes, transcription
quality and importance of the fresh data are more important than the increased data quantity"*
[Long 2019]; 40,000 h of selected audio gave 17% over a CTC baseline [Swarup 2020]. Our students
sit on ~11,000 h. **13 h is three orders of magnitude short, and YODAS is not "fresh" to Typhoon**
(our held-out-video result). The four negatives are what the literature predicts.

**2. Labels matter as much as size — for Typhoon, explicitly.** Typhoon's own paper: *"rigorous
text normalization can match the impact of model scaling"* — their 115M model reaches large-v3-level
accuracy on their benchmark through consistent number/ๆ conventions [Sirichotedumrong 2026].
Strong-teacher pseudo-labels can beat human labels for a streaming student (−13.6% rel.) [Hwang 2022].

**3. Decode-time levers, measured here on the benchmark (canon):**

| typhoon-asr-realtime | CER | Δ |
|---|---|---|
| greedy (shipped) | 0.0946 | — |
| `malsd` beam 2 | 0.0914 | −3.4% |
| beam 2 + 4-gram α 0.1 | 0.0898 | −5.1% |

NeMo's own beam paper claims 14–30% over greedy [Grigoryan 2025]; on Thai/this bench it is ~5%.
**Not enough alone**; it stacks with anything else. Whisper-family: n-gram fusion up to 51% in-distribution
in low-resource languages [de Zuazo 2025] — **untested on turbo**, and turbo is only 8.5% from the bar.

**4. Fresh, human-labelled Thai that Typhoon's cards do not list** (cards list GigaSpeech2,
Common Voice, "internal curated public media"):

| corpus | size | style | licence | status |
|---|---|---|---|---|
| **Porjai Thai-central** (`CMKL/Porjai-Thai-voice-dataset-central`) | **~700 h, 335,674 utts, 7.5 GB** | read | CC-BY-SA 4.0, ungated | not downloaded |
| LOTUSDIS | 114 h | conversational, far-field | CC-BY-SA 4.0 | **on disk** |
| LOTUS-TRD | 180 h | read, 4 dialects | CC-BY-SA 4.0 | not located on HF |

⚠ Read speech is the benchmark's `c-` half (Shunya) — Porjai is a different corpus, but the 20-char
leak guard still runs before any training. ⚠ "Not listed on the card" ≠ "not trained on": check by
measuring base turbo / base rt on a Porjai held-out slice first — if they are already near 0, it is
not fresh (the YODAS lesson).

**Ranked directions (no GPU spent yet):**

| | direction | cost | expected | why |
|---|---|---|---|---|
| **D1** | **turbo + decode-time**: beam 5, then Thai n-gram fusion | hours | −3 to −10%? needs −8.5% | cheapest; untested; stacks with D2 |
| **D2** | **turbo + fresh human labels**: Porjai 700 h (+ LOTUSDIS), with YODAS teacher-label replay, decoder-first then LoRA-encoder | 1–2 days | the only route the literature backs at this baseline scale | *fresh* data + *human* labels + replay [Qian 2024, Hsiao 2025] |
| D3 | rt + beam/LM + Porjai | 1–2 days | 22% gap — unlikely to close | for the ~65 ms tier, not the bar |
| ✗ | more YODAS + teacher | 16+ GPU-days per 1,000 h | not fresh to Typhoon | ruled out by the held-out-video test |

**Gate before D2:** download one Porjai shard, run base turbo on it. If turbo is already as good on
Porjai as on YODAS, Porjai is not fresh and D2 is cut before any training.

### 💡 His idea: the ASR returns its top possibilities, an LLM picks (2026-10-04)

*"is the asr can return the possibility, what if we pass like top possibility and pass it to llm to pick
the right context?"* — N-best picking. Then *"try both"*: merge two models' lists, and pick per spot.

**Ceilings** (perfect picker, reads the reference; own tokeniser, where the bar reads 0.0838):

| candidates | pick a sentence | pick per spot |
|---|---|---|
| turbo top-5 (459 ms) | 0.0810 | 0.0785 |
| typhoon-asr-realtime top-5 (127 ms) | 0.0914 | 0.0894 |
| **both merged (10)** | 0.0717 | **0.0636 (−24% vs bar)** |

Turbo's own 5 are near-copies of one hearing (only 138/682 lists contain the exact sentence); a
second model hears differently (`a-0-104`: only realtime offers ตลาด**วโรรส**). Diversity is the lever.

**Real pickers, official scorer, canon** (`adapter/pilot_picker.py`, word-sized regions — single-letter
spots were unjudgeable: the LLM kept the vote on 14/14):

| system | CER | recall |
|---|---|---|
| bar, typhoon-whisper-large-v3 | **0.0735** | 0.408 |
| LLM picker, qwen3.5:9b zero-shot, ~500 ms | 0.0805 | 0.365 |
| turbo top-1 | 0.0810 | 0.362 |
| probability-weighted vote over the 10 | 0.0833 | 0.360 |
| count vote over the 10 | 0.0841 | 0.353 |

The zero-shot LLM beats both votes (+4%) but only ties turbo's own top-1; voting rules lose to it
outright, because they cannot tell WHEN the weaker model's minority answer is the right one. That
is a learned judgement: **the next step is a TRAINED picker** — run turbo + realtime N-best on
human-labelled audio that is not the benchmark (Porjai), label each region with the option closest
to the human transcript, fine-tune a small LLM (~0.5–1.7B) to emit the choice + confidence.

### 🎯 The TRAINED picker (2026-10-04) — first positive on the real-time line

Data: Porjai Thai-central (CC-BY-SA, ungated), 2 shards → **16,000 train / 1,000 held-out clips**, split
by SENTENCE (Porjai repeats sentences across speakers); **241 clips dropped by the 20-char leak guard**
against benchmark text. Freshness gate passed: base turbo 0.0589 / realtime 0.0728 on held-out Porjai
(not memorised), merged per-spot ceiling 0.0323. N-best: turbo top-5 (CT2) + realtime top-5 (NeMo
`malsd_batch`) on all 17k clips. Labels: per region, the option closest to the human transcript
(`picker_lib.oracle_choice`) — **11,204 of 36,576 regions (31%) are not the vote's answer**.
Picker: **Qwen3-0.6B, full fine-tune**, 2 epochs, answers only digits (`adapter/train_picker.py`;
first run reaped for RAM — 151k-vocab logits overflowed VRAM into system RAM; fixed by running the LM
head on answer positions only + gradient checkpointing → 5.7 GB peak).

| held-out Porjai (sentences never trained on) | CER |
|---|---|
| vote | 0.0662 |
| turbo top-1 | 0.0589 |
| **trained picker** | **0.0533** (−9.5% vs turbo) |
| perfect picker | 0.0362 |

| benchmark, 682 real, canon (never trained on) | CER | recall |
|---|---|---|
| **bar, typhoon-whisper-large-v3** | **0.0735** | 0.408 |
| **trained picker** | **0.0786** | 0.369 |
| zero-shot qwen3.5:9b picker | 0.0805 | 0.365 |
| turbo top-1 | 0.0810 | 0.362 |
| vote | 0.0833 | 0.360 |

**By group (own tokeniser)** — where the bar's lead actually is:

| group | picker | turbo | bar |
|---|---|---|---|
| `a` YouTube talk | 0.1328 | **0.1321** | 0.1428 |
| `c-test` read | 0.0956 | 0.0965 | **0.0858** |
| `c-validation` read | 0.0534 | 0.0602 | **0.0423** |

⭐ **Turbo already beats the bar on conversational YouTube speech.** The whole gap is READ speech,
where large-v3 is far ahead; the picker (trained on read speech) closes 11% of it on c-validation and
almost none on c-test. ⚠ Picker latency is **632 ms** (HF `generate`, 28 layers, batch 1) — not yet
real-time; a single forward pass over the option tokens would remove the generation loop.

### 🏁 A third option source — and a public real-time model that beats the bar alone (2026-10-05)

Literature (Consensus): fused error ≈ mean error − diversity (Audhkhasi 2014); lists from systems
trained on DIFFERENT DATA are the most diverse (Audhkhasi 2013); different architectures 22–33%.
HF enumeration of 11 Thai orgs found `nectec/Pathumma-stt-th-fastconformer-rnnt-large` (2026-09-06,
CC-BY-4.0, NECTEC data, NeMo FastConformer-RNNT — one `.nemo` whose name differs from the repo, so
`from_pretrained` fails; `restore_from` the downloaded file).

**Speed gate (held-out Porjai, top-5, batch 1):** NECTEC FastConformer **92 ms** · biodatlab
distill-whisper-small 561 ms · distill-medium 702 ms · Pathumma-whisper-medium 2,006 ms → only NECTEC passes.

🔴 **Porjai is NOT fresh for the non-Typhoon models:** top-1 on held-out Porjai NECTEC **0.0154**,
Pathumma-medium 0.0248, biodatlab-medium 0.0275 vs turbo 0.0589 — biodatlab's card thanks SLSCU (Porjai's
makers), NECTEC lists "open-source" data. Porjai cannot rank them, and a picker trained on Porjai with
NECTEC options would learn "always trust NECTEC".

**On the benchmark (official scorer, canon):**

| | CER | B-CER | recall | per turn |
|---|---|---|---|---|
| **NECTEC FastConformer, top-1 of beam 5** | **0.0676** | 0.2132 | 0.374 | **113 ms** |
| bar, typhoon-whisper-large-v3 | 0.0735 | 0.1798 | 0.408 | — |

| own tokeniser | talk (`a`) | read (`c`) |
|---|---|---|
| turbo top-1 | **0.1321** | 0.0765 |
| NECTEC top-1 | 0.1379 | **0.0583** |
| ceiling turbo5+rt5, per spot | 0.0917 | 0.0532 |
| **ceiling + NECTEC5, per spot** | **0.0779** | **0.0373** |

⭐ **A single public real-time model beats the bar on CER by 8%** — but worse on names, one run, no
significance test, and ⚠ **training data undisclosed**: if NECTEC trained on Shunya (the benchmark's
read-speech source) its read-speech lead is partly memorisation. Not verified; ask before claiming.

### ⚙️ Deployed + verified (2026-10-05)

- **`asr/run.ps1` now serves NECTEC by default** (`-Model typhoon` restores the old one; `server.py`
  loads a local `.nemo`). End-to-end over HTTP: 48/50 transcripts identical to the offline beam-5
  run (2 beam ties), **133 ms round trip p50**, 240 ms max. CLAUDE.md warns that the arms' control
  is now a different model unless started with `-Model typhoon`.
- **Significance vs the bar** (paired bootstrap, 5,000 draws, own tokeniser): NECTEC beam-5 0.0799 vs
  0.0838, **P(not better) = 0.084**, 95% CI of the difference [−0.0097, +0.0015]; greedy 0.0816, P=0.22.
  **It ties the bar; it does not significantly beat it.**
- **Contamination check:** the benchmark's `c-` clips are the validation/test splits of
  `shunyalabs/thai-speech-dataset` (public, 2,602 / 439 / 1,021, **no licence stated**). **0 of 426**
  benchmark read sentences occur in Shunya TRAIN, and val/test share 0 sentences with train. NECTEC's
  CER on those clips (0.058) is 4× its CER on the Porjai it memorised (0.015) — the profile of unseen
  data. Not excluded: NECTEC training on Shunya val/test audio; only NECTEC can say.

### 🧪 3-source picker trained on SYNTHETIC audio (his choice, overnight 2026-10-05)

Synthetic corpus = the one labelled audio none of turbo / realtime / NECTEC has heard. Guards: 21 clips
dropped by the 20-char leak guard, 29 whose rare entity is a benchmark entity → **4,214 train / 424
held-out, split by sentence**. Regions over turbo5 + realtime5 + NECTEC5, digits canonicalised (NECTEC
writes digits). Only **15%** of synthetic regions have a non-vote answer (Porjai: 31%) — clean TTS audio
makes the models agree. Picker p3: Qwen3-0.6B, 3 epochs; held-out synthetic vote 0.0184 → **0.0156**.

| benchmark, canon | CER (official) | talk (own tok.) | read (own tok.) |
|---|---|---|---|
| **NECTEC alone** | **0.0676** | 0.1379 | **0.0583** |
| **picker p3 (3 sources, synthetic-trained)** | 0.0727 | **0.1287** | 0.0657 |
| bar, typhoon-whisper-large-v3 | 0.0735 | 0.1428 | 0.0618 |
| vote over the 3 sources | 0.0757 | 0.1366 | 0.0700 |
| picker p1 (2 sources, Porjai-trained) | 0.0786 | — | — |

⭐ The synthetic-trained picker **transfers to real speech** (beats its own vote by 4%) and is **the best
system of all on conversational speech** (−10% vs the bar, −7% vs NECTEC). 🔴 It under-trusts NECTEC on
read speech — on synthetic audio NECTEC was not exceptional, so the picker never learned how good it is on
real read sentences. P(not better than bar) = 0.38. Picker latency 370 ms (HF `generate`) — still the
engineering item.

### 📉 More data? Learning curve + real speech added (2026-10-05 morning)

| picker | train data | benchmark CER | talk | read | P(not better than bar) | P(not better than NECTEC) |
|---|---|---|---|---|---|---|
| p3half | 2,084 synthetic | 0.0751 | — | — | — | — |
| p3 | 4,214 synthetic | 0.0727 | **0.1287** | 0.0657 | 0.36 | 0.97 |
| **p4** | 4,214 synthetic + 16,000 Porjai | **0.0681** | 0.1344 | 0.0598 | 0.087 | 0.56 |
| NECTEC alone | — | **0.0676** | 0.1379 | **0.0583** | 0.077 | — |

- **Doubling synthetic data: −3.2%** on the benchmark (0.0751 → 0.0727), flat on held-out synthetic
  (0.0158 → 0.0156). At ~7 GPU-hours per 5k clips, more synthetic is NOT the cheap lever.
- **Adding real read speech moved the picker toward NECTEC** — read 0.0657 → 0.0598, talk 0.1287 → 0.1344 —
  and the net is a **tie with NECTEC alone** at 3–4× its latency (picker 367 ms on top of the candidates).
- ⭐ **Conclusion of the real-time line, as of now:** the deployable answer is **NECTEC alone** (already the
  `:8007` default): ties typhoon-whisper-large-v3 on CER (P=0.077), ~110 ms. The picker's only edge is
  CONVERSATIONAL speech (p3: −7% vs NECTEC, −10% vs the bar) — the speech the voice loop actually hears —
  but the benchmark's conversational half is 256 YouTube clips, not the loop's audio. **Measuring on the
  loop's own conversations is the open question** that would decide whether the picker earns its latency.

**Two infrastructure defects found and fixed on the way:**
- NeMo reads manifests with `open()` and no encoding → cp1252 on Windows → dies on the first Thai
  byte. `finetune_rt.py` refuses to start without `PYTHONUTF8=1`.
- The shipped RNNT loss `warprnnt_numba` fails numba 0.67's typing pass on step 1
  (`Signature mismatch: 2 argument types given`). NeMo's pure-PyTorch loss is correct but loops per
  (t,u) cell: **100 s per batch**, 16 h an epoch (a 20-step smoke on the 200 *shortest* clips hid
  this). `adapter/_rnnt_fast.py` vectorises the u-recursion with `logcumsumexp` — one Python step
  per frame — matches NeMo's reference to **2e-6** (loss and gradient) and runs the worst-case batch
  (16 × 20 s, U=100, V=2049) in **0.86 s**. Training now 2.9 it/s.
- NeMo's batched greedy decoder (CUDA graphs) hit `cudaErrorIllegalAddress` on the long-segment
  batches at bs=16; the remaining 488 segments re-run at bs=1.

## ⏸ Paused 2026-10-03 — three runs killed for low memory

Claude Code reaped all three background jobs when the box ran critically low on RAM (1.4 GB free
afterwards). Not restarted — waiting for him. Partial files are resumable (`pilot_ger.py` skips
done rows; the qwen arm does not):

| run | got to | resumes? |
|---|---|---|
| `ger_all8_qwen35_9b` — **the honest GER number, the one that matters** | never started | — |
| `ger_typhoon25_4b` (5-pool) | 330 / 682 | ✅ |
| `bar_qwen17ctx` (typhoon-asr-qwen-1.7b-ctx, none/bias@50/dis@50) | 393 / 682 of `none` | ❌ restarts |

## External Critical Review

_(Phase 4)_
