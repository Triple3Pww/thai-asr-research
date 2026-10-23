# HISTORY — the ASR research log

**Newest first. Append-only: old entries stay exactly as written, including the ones that turned
out to be wrong** — a claim that was later disproved is the most useful kind of record.

---

## 2026-10-17 -- disk cleanup: 79 checkpoints deleted (35 GB), E: was at 56 MB free

At his request ("clean the repo, it take too much space"). `checkpoints/` was 60 of the repo's 61 GB. Deleted, full list in
`logs/checkpoints_deleted_2026-10-17.txt`:
- 31 per-epoch snapshots `rt_student_*_epN.nemo` (14 GB) -- every run kept its final `rt_student_X.nemo`.
- 48 alpha blends `*_a25/_a50/_a60/_a75/_a90`, `r2ep_e*d*` (21 GB) -- their scores are in `results/`; any one is rebuilt
  exactly with `adapter/avg_nemo.py --base <NECTEC .nemo> --tuned checkpoints/rt_student_X.nemo --alpha A [--alpha-dec D]`.

Kept: all 24 final students, the 13 soups, the headline blends `gs2lot_a75`, `gs2a_a50`, `r2ep_a75`, `lot2_a75`, every `.pt`
head, the pickers and `whisper_student_w1`. `checkpoints/` is now 26 GB; E: has 35 GB free.

---

## 2026-10-12 -- public test sets: our 115M models lead every open Thai model we ran; no test sentence in our training (`F50`)

Loop closed at 00:36 (CronDelete a891aa7d; last try LR 3e-5 lost, +3.6%). At his request, read the Typhoon and NECTEC papers/cards:
Typhoon tests on its GS2 bench (GS2-train ids) + TVSpeech + FLEURS-th; NECTEC's FastConformer publishes nothing. Leak audit clean for
both best models (the GS2 OFFICIAL test is in our training by design -- never report it). One-window scores: gs2a_a50 TVSpeech .0713 /
FLEURS .0595, best of six incl. typhoon-whisper-large-v3 (.0785 / .0690) and Pathumma-whisper-large (.1159 / .0618); r2ep_a75 .0765 / .0612.

---

## 2026-10-11 (evening) -- 24 h loop: 2 epochs is a real -3% (new best `r2ep_a75`, dev .0456); 14 other tries tie or lose (`F49`)

His instruction 00:29: keep the loop running until he says stop. New best `checkpoints/r2ep_a75.nemo` (gs2lot mix, 2 epochs,
alpha .75): dev .04561, -9.8% vs NECTEC, replicated on a 2nd seed (.04595); TVSpeech .0765 vs .0811. Everything else --
more GS2-train, YODAS2 captions, FLEURS, re-weighting, 3 epochs, soups, layer-wise alpha, beam + a 2.5 M-line n-gram,
false-start removal, the new 0.6B Typhoon streaming model, a name list -- tied or lost. Gate (.0364) not reached; test half
untouched. Corrected on the way: a name list is NOT the remaining lever (bias@50 vs dis@50 = -0.8% CER).

---

## 2026-10-11 -- GS2 train machine labels: no gain; five loop tries all tie the best (`F48`)

His instruction: loop research -> test until success (target dev .0364). Shard-0 clips NECTEC misses made the
model worse (the labels are wrong there: Pathumma-L confirms only 18%); the confirmed 4.2 h tied `gs2lot_a75`
(dev .04692 vs .04716, P .41); routing NECTEC's Latin back in also tied. Loop paused for his direction. Test half untouched.

---

## 2026-10-10 -- encoder-level fusion of NECTEC + Whisper-large: no gain over its control, loses Latin (`F47`)

He asked for CER 0.05 by architecture. Ran the pipeline: Consensus lit (multi-encoder fusion, -8..32% elsewhere), idea
report with pre-registered gates, Gemini review (3.5/10; padding bug fixed, decoder-only control added), cached both
encoders on GS2 + the benchmark, identity check 682/682. Fusion and its no-Whisper control both reach -30% on GS2
held-out, but on the benchmark dev half fusion is +11.6% and the control +1.9%; no residual scale makes fusion beat
NECTEC. Cause found: GS2 has no Latin, so the block stops writing loanwords in Latin (214 -> 60 letters). Stop
condition met; the end-to-end variant was not run. Test half untouched.

## 2026-10-10 -- docs brought up to date with F38-F46

He asked to update all .md files. README (banner, headline numbers, rules, layout, environment), CLAUDE.md (best
real-time model, the dev/test protocol, the freshness/batch/orphan rules, new data, results naming and count 758),
RESEARCH-PROGRESS (new section 000), HARNESS (the 2026-10-08 -> 10 evaluation harness), LITERATURE (A90-A91 and
RESULT notes on A72/A75/A80/A84/A86/A87, no status moved), NARRATIVE_REPORT (pointer to what came next) and the
2026-10-08 idea report (section 9, the loop). RESEARCH-FINDINGS already carried F38-F46. No numbers changed.

## 2026-10-10 -- the F45 models win TVSpeech by 9.5-12%; extra data search: elderly not fresh, food hurts (`F46`)

TVSpeech (independent): GS2 alpha .5 -12.1%, GS2+LOT alpha .75 -9.5% vs base NECTEC, both below typhoon-large.
Hugging Face sweep of Thai ASR datasets: elderly speech was already heard by NECTEC; food-ordering (fresh) made the
dev half worse at every alpha. Two reaper incidents traced to VRAM overflow spilling into RAM and an orphaned
trainer; fixed with batch 8 x accum 2 and an in-job memory guard.

## 2026-10-09 -- fine-tuning NECTEC on fresh GigaSpeech2: -6.8% on the dev half at 43 ms (`F45`)

He said do it. GS2 dev+test rewritten into NECTEC's convention (Latin-NECTEC clips dropped, leak-guarded),
17.9 h; 1-epoch A2 recipe. Dev half: GS2 alpha .5 -5.8% (P .008); GS2+conv-LOTUSDIS alpha .75 -6.8% (P .013);
best vote with the Whispers -8.2% but ~550 ms. Target -13.7% not reached. Test half still unscored. One reaper
stop after training finished -- resumed under his standing permission.

## 2026-10-09 -- GigaSpeech2 judge (fresh to all ears) and a pooled identity-free judge: both tie NECTEC on dev (`F44`)

He logged in to Hugging Face (token pasted in chat -- advised to revoke it). GigaSpeech2-th dev+test downloaded
(20 h), passed the freshness gate for all three ears. Judge trained on 7,099 turns: own held-out -6% vs ROVER,
benchmark dev +1.5% (learned "trust typhoon-L"). Pooled LOTUSDIS+GS2 without the which-ear feature: dev
exactly NECTEC. Ears/scoring re-engineered to one model per process and 1,500-turn slices after repeated
low-memory reaps. He said restarts of approved runs need no permission (memory saved).

## 2026-10-09 -- the trained judge ties NECTEC on dev: LOTUSDIS taught it to trust NECTEC (`F43`)

4,314 LOTUSDIS turns through the three deployed ears, all candidates scored by all ears, a small MLP judge.
LOTUSDIS held-out -8% vs always-NECTEC, but benchmark dev 0.05058 vs 0.05059 -- NECTEC has likely heard
LOTUSDIS, so the judge learned its prior. Test half untouched. Three low-memory reaps along the way (VALORANT
open once; then my own scorer holding all audio in RAM -- fixed).

## 2026-10-08 -- target CER 0.060 at <=500 ms: dev/test split locked; vote, LM tie-break, sentence PoE all ~-2 to -3% on dev (`F42`)

He set the target at 0.060 and chose "split by session" after asking what it meant. Split committed before
any tuning (494d951); test half untouched. Fixed the harness: per-utterance OFFICIAL errors
(`official_utt.py`). On dev: oracle -24.5% over 3 ears, but ROVER -3.1%, ROVER+LM worse, sentence-level
product of experts -2.3%. Stopped a pool search that ranked with the superseded plain scorer.

## 2026-10-08 -- budget relaxed to ~500 ms: NECTEC-drafted speculative Whispers + ROVER pass the criteria (`F41`)

He chose "allow slower (~300-500 ms)". Large Whispers cost ~1 s/turn (int8 no help). New piece:
`spec_whisper.py` -- NECTEC's transcript as the speculative draft for a large Whisper, lossless, 264-297 ms.
End to end (`pipeline_latency.py`: NECTEC + Pathumma-L + typhoon-L concurrently + ROVER): 481 ms p50,
benchmark CER 0.0652 vs 0.0695 (P .013), TVSpeech 0.0803 vs 0.0811 (tie). Meets the pre-set criteria, but
the pool was picked after ~15 pools were scored on the benchmark -- not robust to that selection, and the
held-out set shows no gain. Not deployed.

## 2026-10-08 -- the loop: six real-time ideas to beat NECTEC without a list, all lost (`F40`)

He asked to keep looping research + tests until success. Success fixed first: real-time, no list, official
canon CER < 0.0695 with paired P < 0.05. Tried, each with a byte-exact control: catalog-retrieved name lists
from Thai Wikipedia (0.1736, false spots), session name memory (ceiling 3 names), session test-time
adaptation (0.0789 / 0.0713), a step-synchronous RNN-T ensemble (0.0821, emission-timing mismatch), ROVER of
NECTEC-family models (0.0718), and an ear/pen merge of A2 + NECTEC (0.0713). Stopped at the pipeline's stall
rule (>= 4 consecutive failures -> hand the structural choice to him) rather than tune on the benchmark.

## 2026-10-08 -- Ear-Memory-Pen tested and falsified; F38's "spelling" slice was tone and vowel length (`F39`)

He said *"do it"*. E0: a script-blind + syllable-level metric (`eval_soundblind.py`, new venv
`E:/thainlp-spike`) -- script-blind takes ~10% off every system alike; the SOUND metric reorders the field
(Pathumma-whisper and A2 hear best, NECTEC spells best). E1: with TONED G2P, the most frequent spelling per
sound is wrong on only 1.25% of words. E2: a syllable head on frozen NECTEC reads sound worse (0.164) than
G2P of NECTEC's own text (0.115) -- the pre-registered stop. E3: a Qwen3-0.6B homophone re-speller changed 0
utterances under cross-fitting. E4: Thai Wikipedia titles cover 30% of NECTEC's missed names; not built.
**This overturns part of the entry below:** `F38`'s "~20% right sound / wrong spelling" was an artifact of
tone-free sound keys. Nothing deployed; `:8007` unchanged.

## 2026-10-08 -- "invent a new ASR from zero": measured the error first, then designed Ear-Memory-Pen (`F38`)

He asked (via `/aris:research-pipeline`) to *"start from 0 and invent the whole new architecture of asr"*,
research across every useful field, and report the possibility. Ten Consensus searches (hearing biology,
predictive coding, tonal phonetics, Thai syllable structure, syllable tokenisers, LLM decoders, retrieval,
NAR refiners, analysis-by-synthesis, test-time adaptation, two-stage P2G) + one web search. Before
inventing, `adapter/probe_error_anatomy.py` measured NECTEC's errors: 23% Latin-vs-Thai script, ~20%
right-sound-wrong-spelling, 29% inside names, 8% tone-only, 5-best oracle only -4.6% (`F38`). Proposed
architecture: frozen NECTEC Ear + syllable (onset/rhyme/tone) head, a sound-keyed name Memory, and a
Pen trained on Thai TEXT only that writes spelling and chooses script. Two-stage sound->spelling is
published elsewhere (Mandarin, Polish/German, Vietnamese-English); the Thai composition is not found.
**Stages 2-3 not run** -- no model trained; next step E0 is a script-blind metric (no GPU).
Report: `idea-stage/IDEA_REPORT_2026-10-08.md`.

## 2026-10-08 -- the name splice is real-time; the A2 tune does not stack with it (`F37`)

He said *"do both"*. (1) `adapter/kws_fast.py`: Typhoon's keyword Viterbi in numba, bit-identical on
3,503 pairs and byte-identical end to end; the one-pass pipeline (transcript + 50-name splice) now runs
**42.9 ms p50** per 9.7 s turn, the same as NECTEC's own `transcribe()`. (2) A CTC head on the A2
(conv-LOTUSDIS) encoder: CER tie with base+head, fewer names (0.519 vs 0.567); still beats the 0.78B
vendor list model. The two do not stack on the benchmark.

---

## 2026-10-07 -- the name splice in one pass: a 2.6M head on NECTEC's frozen encoder (`F36`)

He said *"do it"* to the ranked next step. `adapter/ctc_head.py` caches NECTEC's frozen encoder
features and trains a small CTC head (held-out greedy CER 0.109, NECTEC's decoder 0.107). With
Typhoon's spotter on its posteriors: bias@50 recall 0.567 (+91 names over decoys), beats
qwen-0.6b-ctx (0.78B) on all four metrics, one encoder pass; CER slightly worse than the two-model
splice. 192 ms/turn, 154 of it Python spotting. The first run was invalid -- NeMo's `transcribe()`
leaves the encoder in training mode and my direct encoder calls drifted BatchNorm stats (481/682
transcripts changed); quarantined, fixed, verified, rerun. Also my slip: an over-broad `rm` deleted
the head's training record; rebuilt from the checkpoint's own log.

---

## 2026-10-06/07 -- "DeepSeek for Thai ASR": parity on two sets, a borderline media gain, a name splice that beats the vendor's list model

He asked for *"the revolutional of asr like the come of deepseek — cheaper and can perform like the
high-end model"* and left me to run it overnight (`/aris:research-pipeline`, run `deepseek-asr-2026-10-06`,
reviewer Gemini). Results → `F32`–`F35`; record `idea-stage/IDEA_REPORT_2026-10-06.md`; `NARRATIVE_REPORT.md`.

- Small-model voting and confidence routing: dead. NECTEC already is the consensus.
- LOTUSDIS fine-tune with raw labels: −25% in domain, **+31% on the benchmark** — the model learned
  the typist (`F32`). Labels rewritten into NECTEC's own convention: tie on the benchmark,
  **−4.8% / −5.9% vs Whisper on TVSpeech over two seeds** (P .054 / .023), ~6 GPU-minutes (`F33`).
- Synthetic entity audio: −72% on synthetic, nothing on real. RL: killed by its own gate.
- **NECTEC + Typhoon's CTC spotter, spliced by frame**: beats qwen-0.6b-ctx (0.78B) on all four
  metrics at bias@50; offline, ≈ 350 ms (`F34`).
- Mistakes of mine, recorded: two jobs run side by side filled RAM and got reaped; I told him the
  first fine-tune had died when it was still running, then launched a duplicate (stopped in a
  minute); two wrong diagnoses of the Whisper truncation before the real one; pre-registered
  criteria that broke the repo's own `F8d` rule; a pooled P = 0.026 chosen after seeing the data —
  withdrawn after the Stage-3 review (4/10, not ready).

---

## 2026-10-05 -- a real conversation corpus was already on disk: LOTUSDIS is fresh enough to train on

He asked *"do more research on how can we improve asr on the resourse that we have"*. Consensus-first
sweep (six queries) -> `LITERATURE.md` `A72`–`A77`. Two things came out of looking at the box, not
the papers:

1. **NECTEC's FastConformer is a pure transducer** (`EncDecRNNTBPEModel`, 17 layers, d512, 2,048
   unigram pieces, full context, no CTC head). So the boosting tree (`A32`, arm D4) ports directly,
   but the D6 second-opinion splice -- the one arm that doubled entity recall -- has no CTC branch to
   spot with. Every decode-time arm in `F8`–`F16` was measured on typhoon-asr-realtime, never on NECTEC.
2. **`E:/Claude/datasets/LOTUSDIS/` (13 GB) has been there since 2026-08-26**, fetched for the
   endpointing line and never used for ASR: 114 h of real three-party Thai meetings, CC-BY-SA,
   human transcripts. Freshness gate (`adapter/gate_lotusdis.py` -> `results/gate_lotusdis.json`,
   300 lavalier clips per split): **NECTEC test 0.205 / train 0.124; typhoon-rt test 0.272 / train
   0.207.** Test is fresh to both (ratio 0.75 vs the benchmark's 0.72); NECTEC has probably seen some
   of train (−39% vs −24%) but not memorised it. Both read real conversation at **3× their benchmark
   error** -- the headroom the read-speech benchmark does not show.

Nothing trained. The ranked plan is in the session report: fine-tune NECTEC on LOTUSDIS train;
a CTC head on NECTEC's frozen encoder to bring back D6; then text-only predictor adaptation and
per-speaker adaptation.

---

## 2026-10-03 → 10-05 -- beating the best Typhoon in real time: our students failed, his picker worked, and NECTEC had already done it

He asked for *"the asr model have better score than the best typhoon model right now"*, then
*"make the a model that beat typhoon best score that can be use in realtime"*. Three days, run
through the ARIS research pipeline and then by hand. Synthesis: `F27`–`F31`. Record:
`idea-stage/IDEA_REPORT.md`. State: `RESEARCH-PROGRESS.md` §0.

### Day 1 -- combining beats the bar; the first two "wins" were traps

Typhoon had shipped 8 ASR models since our 7-system table; their own cards still put
typhoon-whisper-large-v3 first (`A68`). A vote over the 8 public systems beat it (0.0678 vs 0.0735,
`F27`). On the way, **two wins that were not**: a hand-picked 5-system vote looked −20% on raw CER
and was mostly number convention, and the best of 219 pools, chosen on the test set, lost once
chosen honestly by cross-validation. A cross-family reviewer (Gemini; Codex is not installed)
called the methods not novel and pushed toward distillation into one model.

### Day 2 -- four distillation runs, zero gain, one cause

YODAS2 Thai, labelled by all 8 teachers, into typhoon-asr-realtime (3 recipes) and turbo: nothing
moved (`F28`). The fix for round 1 (confidence measured on the student too) changed nothing. The
cause was found, not guessed: **on held-out videos base turbo beats the teacher**. YODAS is not
fresh to Typhoon. Building the training loop on this box also needed a new RNNT loss
(`adapter/_rnnt_fast.py`): NeMo's numba loss fails on numba 0.67 and its PyTorch one took 100 s a batch.

He then made **Consensus the research tool** (CLAUDE.md, memory, and patches to the ARIS
`research-lit` / `novelty-check` skills, which live in the plugin cache). Its first sweeps put
numbers on the failure: distillation wins start from weak students with hundreds of hours (`A64`,
`A65`).

### Day 2, evening -- his idea

*"is the asr can return the possibility, what if we pass like top possibility and pass it to llm to
pick the right context?"* An N-best picker. One model's top-5 were near-copies (ceiling −2.4% vs
the bar); merging two models and picking per spot pushed the ceiling to −24% (`F29`). A zero-shot
9B LLM only tied turbo; a **trained** Qwen3-0.6B picker on Porjai (700 h read Thai, fresh for
Typhoon, `A67`) beat turbo. He asked for better candidate models, research first: diversity and
different training data matter most (`A63`). Enumerating 11 Thai orgs found
**`nectec/Pathumma-stt-th-fastconformer-rnnt-large`**.

### Day 3 -- NECTEC alone ties the bar; deployed; the picker trained on synthetic audio transfers

NECTEC's real-time model scores **0.0676 at 113 ms**: a tie with large-v3 (P = 0.084), worse on
names (`F30`). Contamination checked as far as the public record allows (0/426 benchmark sentences
in Shunya train). **He asked to put it in the loop**: `asr/run.ps1` now serves it by default
(`-Model typhoon` reverts). CLAUDE.md warns that the arms' `:8007` control is now a different
model.

NECTEC had memorised Porjai, so the 3-source picker trained overnight on the **synthetic corpus**
instead (his choice), the only labelled audio no model had heard. It transfers to real speech and is
the best system of all on conversation (−10% vs the bar), but under-trusts NECTEC on read
sentences. Adding Porjai gave a tie with NECTEC alone. Doubling synthetic data: −3.2%, so no more
was generated.

### What is open

1. Transcribed conversations from the loop itself -- the only test of whether the picker's
   conversational edge is worth ~370 ms.
2. Picker latency (one scoring pass instead of `generate`).
3. Whether NECTEC trained on Shunya val/test audio -- ask NECTEC.

---

## 2026-09-26 -- the only public Thai SER model is broken and scores 55; and our result stopped being a recipe

He asked the sharp version of the question: *"so right now our model is the best public thai model
that detect that emotion?"* The answer required two things that did not exist yet.

### 🔴 First: there was no model. There was a recipe.

Checked before answering: **no `.pt` / `.safetensors` / `.ckpt` anywhere.** Every emotion number in
this repo -- `F18`'s 0.8488, `F23`'s 77.18 -- came from weights held in a Python variable and
discarded when the process exited. Reproducible from script + seed + config, and **not an artifact
anyone could be handed.** "Our model" was not a true description of anything on disk.

✅ Fixed: `probe_thaiser_protocol.py --save-checkpoint` now writes
`checkpoints/thaiser_head_3s_v1.pt` (51 KB). It is deliberate about which part is ours -- the
backbone is the **unmodified public `emotion2vec_plus_large`**, named but not included; what is
saved is the 4-class head (4 x 2048), the feature standardisation it expects, the protocol, the
scores and the caveats. ⭐ 51 KB is the honest summary of the contribution: a linear read-out on a
public 164 M encoder, plus the protocol discipline to know what it is worth. (`F26`)

### F25 -- the only public Thai SER model, benchmarked for the first time by anyone

`A59` found the landscape is one checkpoint: `awghuku/wav2vec2-base-thai-ser`, no card, no metrics,
19 downloads, untouched since 2023-04-28. So nobody -- including its author -- had ever published a
number for it.

| system | WA | UA |
|---|---|---|
| `awghuku/wav2vec2-base-thai-ser` | **55.15 ± 4.74** | **45.84 ± 3.66** |
| THAI-SER published baseline | 67.34 ± 3.05 | 62.61 ± 3.19 |
| ours, their 3 s cap | 77.18 ± 1.98 | 73.18 ± 2.52 |
| ours, uncapped | 82.22 ± 1.38 | 79.82 ± 2.72 |

**It loses to the corpus authors' own baseline by 12 points** -- and every fairness choice was made
in its favour: its frustrated logit is dropped rather than scored wrong, and it was given the best
of all 120 possible label orderings. 🔴 The score is still an **upper bound**, because its
train/test split is undocumented and our test folds may be its training data.

🔴🔴 **Two defects beat the score for importance.**
1. **It does not load on current transformers.** `"id2label": {"0": 0, ...}` -- integer values where
   strings are required -- fails strict config validation. The benchmark repairs it in memory.
2. **Its label order is undocumented AND counter-intuitive.** Fitted on train folds, stable across
   all 8: `ANGRY=0, NEUTRAL=1, HAPPY=2, SAD=4`. Assume the corpus authors' own documented column
   order instead and you get **WA 8.56%** -- below the 35% majority-class floor, with nothing to
   signal the error.

⭐ **What this establishes:** public Thai SER is not a field with a state of the art we are racing.
It is a field with one broken, undocumented, unbenchmarked checkpoint. The real bar is the corpus
authors' published baseline, which is a paper and not a downloadable model.

### What can now be said, and what still cannot

✅ **Defensible:** *benchmarked under the THAI-SER authors' own protocol on their full corpus, ours
scores WA 77.18 / UA 73.18 (82.22 / 79.82 uncapped) against their published baseline's 67.34 /
62.61 and the only existing public Thai SER model's 55.15 / 45.84 -- the first published evaluation
of that model by anyone.* Three systems, one protocol, eight folds, standard deviations throughout.

⚠️ **Still not defensible:** "best Thai emotion model" unqualified. Everything is **acted studio
speech**; the 2,516 Zoom clips are unscored; `F20` showed this family mislabelling natural read
speech; only HuggingFace was searched; and the number we quote is our FROZEN rung, not `F18`'s
full fine-tune, which remains unrun under their protocol and uncheckpointed.

---

## 2026-09-24 -- F8b is dead: biasing moves entity recall 144 -> 207, and our emotion number survives their protocol

Two lines of work, both of which shrank a claim and strengthened it.

### F24 -- the problem five arms could not move was solved by a model Typhoon already shipped

He asked whether we still need to fine-tune SenseVoice or Typhoon now that the emotion model is
good. The answer split in two: emotion is done (`F21` -- separate encoder, so neither ASR model is
involved), and the ASR question was still open. Enumerating `typhoon-ai` on the HF API surfaced
**`typhoon-asr-qwen-0.6b-ctx`** -- tagged `contextual-biasing`, `hotwords`, with a paper -- which
**this repo had cited for its CER table and never run.**

`adapter/arm1_rareword_qwenctx.py`, 682 real-split utterances, 3 conditions, 2,046 generations,
~2.4 h on the GPU, benchmark's own frozen scorer:

| system / condition | CER | U-CER | B-CER | entity recall |
|---|---|---|---|---|
| typhoon-asr-realtime, none | 0.1125 | 0.1057 | 0.2618 | 0.3280 (144/439) |
| qwen-0.6b-ctx, none | 0.0997 | 0.0937 | 0.2214 | 0.3690 (162) |
| qwen-0.6b-ctx, dis@50 | 0.1013 | 0.0920 | 0.2912 | 0.3645 (160) |
| **qwen-0.6b-ctx, bias@50** | **0.0983** | **0.0899** | 0.2702 | **0.4715 (207)** |

✅ **It survives the control that would have killed it.** `F8d` says `bias@N` only means anything
against `dis@N` at the same N. The list-length artifact alone is **-0.0046 recall** -- nil. Against
dis@50, biasing gains **+0.1071 recall / +47 mentions**, with U-CER -0.0021 and B-CER -0.0210.
First arm in this repo to clear that bar. Distractor fire rate .0002, matching their <=.0004 claim.

🔴🔴 **The lesson is the ordering.** Five purpose-built decode-time arms failed at this. The fix was
a download. **Enumerate what the vendor ships and run it before building the next arm.**

⚠️ Caveats kept with the number: CI95 [0.4214, 0.5283]; it is a DIFFERENT model so 0.3280 -> 0.4715
mixes biasing with a better ASR (quote the dis@50 delta); and it runs in-process, never crossing
the PCM16 wire.

### F23 -- our emotion number under THAI-SER's OWN protocol: +9.8, not +17.5

He asked whether their dataset is public and whether we can evaluate on it. It is -- CC-BY-SA-4.0,
ungated, 29 shards -- and **so is their experiment code**, which turned an approximate comparison
into an exact one. Pulled the full corpus (`adapter/fetch_thaiser_shards.py`, 12.71 GB):
**11,301 clips / 200 actors / 100 sessions / 12.37 h**, against the 1,256 clips / 22 actors every
earlier emotion number used. On the old 3 shards **7 of their 8 folds had zero test clips.**

Their protocol, read out of `experiment/data/thaiser.py` and `four_emotions.yaml`: 8 folds over
studio sessions, agreement >= 0.71 (we used 0.70), train `mic_con` / test `[con, clip]` (we used
`mic_clip`), Zoom excluded, **3-second cap**.

| | WA | UA |
|---|---|---|
| ours, `mic_con` | **77.18 ± 1.98** | **73.18 ± 2.52** |
| ours, `mic_clip` (cross-channel) | 73.95 ± 2.96 | 67.19 ± 3.28 |
| their CNN+LSTM | 67.34 ± 3.05 | 62.61 ± 3.19 |

🔴 **The naive pairing would have read +17.5; the protocol-matched answer is +9.84.** Written first
as "most of that is the 3-second cap" -- an inference -- then **measured** by re-running the folds
uncapped: 82.22 WA, so **the cap alone costs 5.04 (65%)** and their remaining protocol 2.66.
⭐ That 2.66 also says `F18`'s actor-holdout split was not meaningfully optimistic.

🔴 **And +9.84 is a FLOOR:** this arm fits a linear head on a FROZEN encoder -- `F18`'s head rung --
against their from-scratch CNN+LSTM. The full fine-tune under their protocol is unrun.

⭐⭐ **Unasked-for finding: the model is channel-sensitive.** `con` -> `clip` costs **3.2 WA / 6.0 UA**.
UA takes double the hit, so channel mismatch damages the minority classes twice as hard -- the same
asymmetry `F19` found for prior shift. Every emotion number here before today was `mic_clip` and
this model trains on `mic_con`; for the loop, where the mic is neither, that is unpriced risk.

### The Thai emotion landscape, surveyed (A59-A62)

- 🔴 **The MODEL landscape is empty.** One Thai SER model on HF (`awghuku/wav2vec2-base-thai-ser`,
  19 downloads, 2023, **no model card at all**), and zero emotion models across `airesearch` (200
  repos), `typhoon-ai` (82), `biodatlab` (25), `pythainlp` (22), `nectec` (20), `wayu-ai`, `VISAI-AI`.
  **`airesearch` built THAI-SER and released no model.** ⚠️ Not checked: GitHub, ModelScope; and
  three org handles returned 0 repos, meaning wrong handle, not absence.
- The **literature** is active: 80.36% and 80.59% published on this corpus (5-class) -- but the
  corpus authors' own speaker-independent baseline is 59.80 WA on the same data, and the 2021 paper
  describes a plain 70/30 split with no speaker-independence. **Suspected speaker leakage**, not
  proven (paywalled).
- ⭐ **EMOLA** (`A62`): a SECOND Thai emotion corpus, 8,987 turns from TV drama. Drama dialogue sits
  between `F20`'s intentless read speech and THAI-SER's studio prompts. Not on HF.
- ⭐ `A61`: a Chula 2024 thesis already found our exact result shape -- cross-lingual alone does not
  beat same-corpus Thai, but pretrain-large-then-finetune does. **"Confirms and extends", not "first".**

### Left ready, unrun

- **2,516 Zoom clips** now in `ft_cache_full` -- the corpus authors' own out-of-domain set, and the
  closest thing to non-studio speech this project has (`F20`). Needs a `mic_zoom` cache pass.
- The **full fine-tune under their 8 folds**, which would turn `F23`'s floor into the real number.
- **40 clips in `D:/thai-ser-listen/`** still unlistened -- still the only human ground truth in the
  natural-speech analysis.

---

## 2026-09-22 (late) -- the architecture question: separate head, separate encoder, and 4 seconds

He asked the design question directly -- *"make emotion in the same head like the same or we need
specific head for train to train emotion"* -- then *"how long audio need for emotion2vec for an
prediction"*. Both were answered with measurements rather than argument.

### F21 -- the encoder cannot be shared either, and it is not a head-capacity artefact

The shared CTC head was already dead: 0.019 SAD recall against 0.830 from a linear probe on its own
encoder. The open half was the ENCODER. The 2026-09-19 gap came from a deliberately weak single
affine layer, so sweeping head capacity (`linear / mlp128 / mlp512`) was the test of whether a real
head closes it. **It closed 2 points of 10.**

| encoder | best head | test acc | macro-R | SAD |
|---|---|---|---|---|
| Typhoon (ASR) | mlp512 | 0.6780 | 0.6904 | 0.736 |
| SenseVoice (ASR) | mlp128 | 0.7000 | 0.7136 | 0.792 |
| emotion2vec | **linear** | 0.7805 | 0.7532 | 0.604 |
| all three fused | mlp128 | 0.8146 | 0.8041 | 0.736 |
| emotion2vec fine-tuned | linear | **0.8488** | **0.8570** | 0.906 |

⭐ **The tell is which head won.** emotion2vec's best head was the plain linear one -- extra capacity
bought it nothing, its features are already linearly separable for emotion. The ASR encoders NEEDED
MLPs and still lost. Paired McNemar best-ASR vs emotion2vec **p = 0.0032**.

🔴🔴 **And the structural argument closes it:** `F18` says what makes emotion good is a FULL
fine-tune, and you cannot fully fine-tune a shared encoder without destroying its ASR. The thing that
makes emotion work is exactly the thing sharing forbids.

⭐ **The surprise: ASR encoders are COMPLEMENTARY.** They beat emotion2vec at SAD (0.792 / 0.736 vs
0.604) while losing on NEUTRAL -- fusing lifts macro-recall 0.7532 -> 0.8041. Since the ASR encoder
runs anyway, its pooled features are free. ⚠️ Fusion p = 0.0488 over three tried combinations, so it
does not survive multiple-comparison correction, and fine-tuning one encoder (0.8488) still beats
fusing three frozen ones (0.8146).

**The price of the spec's one-pass goal is now a number: ~15 points** (0.7000 vs 0.8488), for ~20 ms
and 164 M params. ⚠️ All frozen -- multi-task fine-tuning stays unmeasured and is the one route that
could still deliver a real single pass.

### F22 -- 4 seconds, and a confounded table corrected

The by-duration-band numbers already in `RESEARCH-PROGRESS.md` compared **different clips**, so their
peak at 2-4 s and drop beyond 4 s read as "too much audio hurts". Truncating the SAME 212 clips from
the start gives a **monotonic** curve that **plateaus at 4 s** (4 s and full-clip agree to four
decimals). The old line is left as written with a correction note beneath it.

⭐⭐ **Per class, with a mechanism:** NEUTRAL saturates at **0.5 s**, ANGRY at **1 s**, SAD at **2 s**,
HAPPY not until **4 s**. The fast pair is separable by energy alone; happy-vs-angry and sad-vs-neutral
are each an energy tie, so only pitch CONTOUR separates them and contour costs seconds. For the loop:
*"something is up"* in ~1 s, *which* emotion not until ~4 s.

🔴 **Below 3 s the model is overconfident by 11-17 points** (0.763 confidence at 0.594 accuracy on
0.5 s), so `F19`'s per-class confidence thresholds are invalid there. Any selective-prediction rule
needs a minimum-duration gate.

### What their own papers actually report (A56-A58)

He asked what SenseVoice and emotion2vec claim. Read 2026-09-22:

- **SenseVoice** (arXiv 2407.04051, Table 8): CASIA UA/WA 96.0 Large, IEMOCAP 73.9/75.3 Large,
  CASIA 70.0/70.0 Small. 🔴 **No Thai, no SEA language** in the SER evaluation at all.
- **emotion2vec base** (arXiv 2312.15185): IEMOCAP WA 71.79 / UA 72.94, pre-trained on **262 h** of
  English emotion data, tested on 10 languages -- 🔴 **Thai not among them**. Cross-lingual WA runs
  Mandarin 49.15 to Bangla 90.91, median ~74.
- 🔴🔴 **`emotion2vec_plus_large`, the model we actually use, publishes NO accuracy for any
  language.** 42,526 h of pseudo-labels, a radar-chart image, no table. The 71.79 belongs to *base*.

⭐ So our Thai frozen-head **0.7805** sits above their cross-lingual median and the fine-tuned
**0.8488** lands between their German and Urdu: **Thai is a typical unseen language for this model,
not an unusually hard one.** And since no Thai figure exists for either model, ours are the first --
a cleaner claim than "we beat them".

### Artefact

- **Architecture comparison, old vs proposed, as node graphs with every rung measured:**
  `https://claude.ai/artifact/3oJhNQaN8YZphwxdtXiuP8`

---

## 2026-09-22 (evening) -- he spotted that the corpus has no emotion IN it, and the measurement agreed

One observation from him reshaped the pseudo-labelling plan: *"I think the dataset didnt make it to
have emotion in it so, some voice sound sad but the transcript is not."*

### What was asked, and what was found first

He asked for the emotion class proportions of SenseVoice's training data, then a plan. **The
proportions are not published.** The FunAudioLLM report (arXiv 2407.04051) states the SER data
*"comprised 30 million entries"* pseudo-labelled by open-source models -- naming `emotion2vec_plus_large`,
the same teacher this repo fine-tuned -- and gives **no per-class counts, no class-imbalance
discussion, no balancing strategy, and never documents `EMO_UNKNOWN` at all**. That absence is the
finding: a 30 M-sample emotion set built by pseudo-labelling, shipped with no distribution, whose head
we have measured abstaining 40% of the time and emitting SAD at 0.019 recall on Thai.

### Then the corpus was measured instead of assumed

`adapter/probe_natural_prior.py`. 2,000 Common Voice 17 Thai test clips, 2.77 h, 39 s on the GPU.
🔴 Along the way: **LOTUSDIS audio is NOT on this box** -- 9 KB of metadata only. The acted THAI-SER
shards and Common Voice test are the only Thai audio available locally.

| | NEUTRAL | HAPPY | ANGRY | SAD |
|---|---|---|---|---|
| text-only labels, 300 sentences | **0.893** | 0.063 | 0.017 | 0.027 |
| the teacher, audio-only, same 300 | 0.553 | 0.207 | 0.090 | 0.150 |
| `F19` had assumed | 0.850 | 0.060 | 0.050 | 0.040 |

🔴 **chi-square 5.93 on 9 df against a critical 16.92 -- no association between what the voice is
labelled and what the words say.** 14 of 300 clips are non-neutral on both sides; they agree on the
class 7 times, which is chance.

⭐ **He was right about the mechanism.** A volunteer reading a prompt has no emotion to detect; asked
for a label anyway the model returns timbre, recording conditions and fatigue dressed as emotion:
`"Sodium chloride is just ordinary salt"` -> **HAPPY 0.997**. `"Many pharaohs ruled ancient Egypt"` ->
**HAPPY 0.995**.

### Why this supersedes the framing of F19, which stays on the record

`F19` modelled this as class prior shift -- teacher fixed, class mix changes, confusion matrix
corrects it. That assumes every clip HAS a true emotion. Here most do not, so there is no prior to
recover, which also explains why the black-box-shift correction returned 66% neutral for a corpus
whose text is 89% neutral: it was solving for a quantity that does not exist. `F19`'s arithmetic is
still right for spontaneous speech. `F20` records where it does not apply.

⭐⭐ **The actionable consequence: corpus choice is about INTENT, not hours or audio quality.**
conversational/YouTube > acted > read. Common Voice ranks last despite the cleanest audio.
**This raises the value of the 10,000 h GigaSpeech2 corpus rather than lowering it** -- his original
worry was that a big corpus would be mostly neutral, but spontaneous-and-neutral is real emotion data
(a valid NEUTRAL example from someone actually talking), while read-and-intentless is not data at all.

### 🔴 A self-correction, recorded because that is what this file is for

Mid-session I showed six SAD-predicted clips whose text really was sad -- *"I don't think I'll ever see
my old friends again"* -- and argued that readers must be unconsciously performing the line, so SAD
might be salvageable where HAPPY and ANGRY were not. **That was six clips ranked by confidence out of
2,000.** The head of a confidence ranking always looks like a pattern. Measured at scale the effect is
absent (lift 1.67x on 45 clips, chi-square n.s.). His hypothesis was right and my caveat was noise.

### Also this session

- `adapter/probe_pseudolabel_yield.py` (`F19`) and `adapter/probe_natural_prior.py` (`F20`) added.
- The `F18` head was verified **exact** where reused -- recovered by least squares, reproducing 0.7805
  on THAI-SER test with 100% argmax agreement.
- 40 clips extracted to `D:/thai-ser-listen/` for a human listening check. **Nobody has listened yet;
  it remains the only non-circular validation of any of the natural-speech numbers.**
- `agy` headless silently returns a permission error instead of labels when it decides to browse
  (`read_url` auto-denied). Half the batches failed that way; adding *"do not use tools"* fixed it.
  Worth remembering for any future `agy` batch job.

---

## 2026-09-22 (afternoon) -- the teacher was fine-tuned: 0.7317 -> 0.8488, and only the FULL unfreeze earns it

Phase 2 step 1 of the spec, executed. A measurement session: four rungs, a learning-rate sweep, a GPU
migration with a bracket, a provenance question closed, a harness defect found, and his question about
pseudo-labelling answered with arithmetic instead of an opinion.

### The result

`emotion2vec_plus_large` fine-tuned on THAI-SER, scored on the same 410 actor-held-out clips every
earlier number uses. **Zero-shot 0.7317 -> full fine-tune 0.8488**, macro-recall 0.6848 -> **0.8570**,
four points off the 0.9286 human ceiling.

| rung | trainable | test acc | macro-recall | SAD recall |
|---|---|---|---|---|
| zero-shot | 0 | 0.7317 | 0.6848 | **0.396** |
| head (frozen backbone) | 4.1 K | 0.7805 | 0.7571 | 0.623 |
| top2 | 25.2 M | 0.7829 | 0.7598 | 0.642 |
| top4 | 50.4 M | 0.8024 | 0.7830 | 0.679 |
| **full** | 164.0 M | **0.8488** | **0.8570** | **0.906** |

⭐⭐ **SAD is the story: 0.396 -> 0.906.** The class the shipped model could barely find is the one the
fine-tune repairs, which is why macro-recall moves further than accuracy.

🔴 **Paired McNemar says the middle rungs are worthless.** Every rung beats zero-shot significantly, but
**head / top2 / top4 are statistically indistinguishable from one another** (head vs top4 p=0.31) while
**full separates from all three** (vs head p=0.0046, vs top4 p=0.0295). A 4,100-parameter head captures
the easy gain; 25 M and 50 M buy nothing measurable; 164 M buys a real jump. There is no middle rung.

### 🔴🔴 One learning-rate guess would have inverted the answer

`full` dev accuracy by backbone lr is **non-monotonic**: 1e-5 -> 0.7762, 3e-5 -> **0.7119**, 1e-4 ->
**0.8881**. The middle value is worse than the smallest. **This session's own first guess was 1e-5**,
which reports "full fine-tuning is no better than a frozen head" -- the opposite conclusion, from a run
that would have looked perfectly defensible. The sweep is as much the finding as the number is.

⚠️ And it balances on early stopping: 426 clips from SEVEN actors against 164 M parameters, train loss
0.0009-0.0014 within 8 epochs, dev-selected checkpoints at epochs 3 and 7.

⚠️ Still acted speech. Fine-tuning gains +8.4 points on improvised turns but +15.7 on scripted, so the
impro/script gap nearly **quadruples** (0.026 -> 0.098). Getting better at acting faster than at
speaking. Nothing here is measured on natural Thai.

### The GPU, and the bracket that had to pass before it could be used

The work moved to a **separate** `D:/sensevoice-gpu/.venv` on `torch 2.14.0+cu130` -- deliberately the
same torch *version* as the CPU venv so the device is the only variable, and separate rather than an
in-place upgrade because the CPU venv produced the bar and every encoder-probe number on record.

✅ **The device was bracketed like a dependency.** GPU zero-shot reproduced 0.7317 / 0.6848 with
identical accuracy on train, dev *and* test, identical confusion matrices, and **all 410 per-clip
predictions identical**; pooled features differ by at most 1.8e-3. Measured: the 1,256-clip pass went
**900 s -> 24 s (37x)**, fwd+bwd **1.31 s -> 58 ms (18.9x)**. Batching to bs=8 would have been 55x and
was **declined** -- it needs padding plus a mask in the mean-pool, and a pooled vector that silently
averages over pad frames is the exact class of defect this repo keeps finding under its numbers.

### ✅ The unexplained 2026-09-19 files are explained -- and the residue was a real defect

`results/probe_emotion_thai_testactors.json` could not have come from the script on disk. Rather than
settle who ran it, both halves were re-derived through an independent code path:

- **emotion2vec 0.7317 -- exact.** All **410/410** per-clip predictions identical, same clip ids in the
  same order, identical confusion and per-class tables.
- **SenseVoice 0.4951 -- 408/410.** Accuracy, macro-recall and SAD recall identical to the last digit;
  two clips differ.

🔴 **Those two clips are `funasr` dithering at inference.** `WavFrontend` is configured `dither = 1.0`
and applies it **unconditionally**, so noise is added to the waveform on every call. Six repeats of one
clip: `HAPPY, EMO_UNKNOWN, HAPPY, EMO_UNKNOWN, EMO_UNKNOWN, EMO_UNKNOWN`. `dither=0.0` is perfectly
stable. Ruled out first: not threads (persists single-threaded), not dropout (all 211 modules report
`training=False`, and `specaug` **is** correctly gated). ✅ **NeMo guards its own dither with
`if self.training`** -- commented *"only in training mode for eval determinism"* -- so every ASR number
in this repo is unaffected, and emotion2vec has no `WavFrontend` at all, which is precisely why it
reproduced 410/410 where SenseVoice reproduced 408/410. **Any future SenseVoice arm sets `dither=0`.**

### ⭐⭐ His question: pseudo-labelling a mostly-neutral corpus

*"what if we use this emotion model to label all the dataset which might have most sound be the
natural, will that be a problem"* -- yes, and it is computable. The teacher mislabels a neutral clip as
HAPPY 13.8% of the time; harmless at a 1/3 neutral prior, fatal at 85% where the leaked neutrals
outnumber true happy clips 3:1. HAPPY pseudo-label precision falls **0.592 -> 0.232 -> 0.092** as the
corpus goes from THAI-SER's mix to spontaneous to read. Accuracy stays flattering the whole way.

⭐ Confidence partially rescues it, **but only after the fine-tune** -- the zero-shot head's HAPPY labels
are unusable at *every* threshold while the refit head's reach 1.000 at t=0.99. **The fine-tune's real
product is calibration, not accuracy.** And one global threshold cannot serve four classes: pinned by
HAPPY at t=0.99, ANGRY collapses from 1,569 correct per 100 k to 49.

🔴🔴 **The decisive number is yield.** Per 100 k clips the fine-tuned teacher yields 327 correct HAPPY.
Over **every** Thai clip on this box (65,006 = LOTUSDIS unique + Common Voice th train) that is **213
clean HAPPY** -- against **~3,402** sitting in the 26 THAI-SER shards not yet downloaded, at ~400-600 MB
each, already human-labelled. **Pulling shards beats pseudo-labelling the entire Thai corpus by 16x.**
✅ What pseudo-labelling alone can give is NEUTRAL at 0.986 on **natural** speech, which an acted corpus
structurally cannot.

### What was built

- `adapter/finetune_emotion2vec.py` -- cache / zeroshot / sensevoice re-score / four training rungs,
  dev-side lr sweep, split asserted against the recorded eight actors, `--device cpu|cuda`.
- `adapter/probe_pseudolabel_yield.py` -- selective prediction and prior-shift projection.
- `D:/sensevoice-spike/ft_cache/` -- 1,256 clips decoded once, three mic channels (`mic_zoom` is null
  on every studio session; it exists only for the 20 held-out Zoom sessions).

### Corrections made in-session, recorded because the record is the point

1. Off `top2` alone I said unfreezing "buys one clip" and was wrong -- `top4` and `full` both gain.
2. Then off `top4` I said unfreezing "does pay" -- also wrong, McNemar puts it at p=0.31.
3. The honest version is the one above: nothing between a frozen head and a full unfreeze is separable.
4. I predicted `full` was "as likely to be a warning as a win" on 426 clips. It was the clear winner.

---

## 2026-09-22 -- SenseVoice chosen as the base, the data counted honestly, and THAI-SER read end to end

A design session, not a measurement one. Four things came out of it: a spec, a corrected picture of how
much Thai data actually exists on this box, a literature sweep that put a ceiling on the cheapest
planned change, and a full read of the THAI-SER paper that hands us a baseline to beat.

### The decision

He chose **SenseVoice as the base**, after the alternative was built and measured against it. The
alternative -- a new head on Typhoon's frozen encoder -- keeps Thai ASR at CER 9.92 for free and
trains in minutes on cached features; the SenseVoice route starts Thai from zero and buys five other
languages. He was shown both with numbers and picked the second. **This spec does not re-argue it**:
`docs/superpowers/specs/2026-09-19-sensevoice-thai-design.md`.

⚖️ **His argument for it, which survived scrutiny:** a multilingual model that also speaks Thai is
worth more than a Thai-only one, and code-switching is the measured largest error family (`F17`:
Latin<->Thai script is **20.4%** of this benchmark's error mass). ⚠️ Partly undercut the same session:
`typhoon-asr-realtime` already emits Latin inline on **89 of 682 (13.0%)** real Thai utterances
(`BTS`, `MRT`, `MK`, `Truemany`), so Thai-with-English-words is already handled. And Typhoon shipped a
declared **th+en** model in July (`typhoon-asr-streaming-nemotron-0.6b`) that this repo has never run.

### 🔴 The Thai data, counted rather than assumed

| source | what was believed | what it is |
|---|---|---|
| **LOTUSDIS** | 13 GB / 160,915 utts / "spontaneous Thai" | 🔴 **~17.6 h of UNIQUE audio.** 114.2 h of rows = 32,183 utterances x **5 microphones** (`con123`, `lav123`, `bt3m`, `bt10m`, `jbl`, exactly 32,183 each). Train alone: 120,245 rows / 88.1 h but **24,026 unique recordings, 22,596 distinct sentences** |
| **Common Voice 17 Thai** | "small" | ⭐ **37.5 h train / 32,823 clips / 32,823 DISTINCT SENTENCES** -- one per clip. Plus 29.4 h dev+test. 423 h / 363,362 clips exist in total. CC0, ungated, **transcripts cached, audio not downloaded** |
| **GigaSpeech2-th** | "he has it" | ❌ **He does not.** What is cached is `typhoon-ai/gigaspeech2-typhoon` at **93 KB** -- one test parquet, their *benchmark*. The 10,000 h training corpus (`speechcolab/gigaspeech2`) is gated and ~600 GB; no HF token found on the box |
| **FLEURS** | not considered | ⭐ all six languages available (`cmn_hans_cn` `yue_hant_hk` `en_us` `ja_jp` `ko_kr` **`th_th`**), CC-BY-4.0, ungated, **12.82 GB for the five old ones**. 🔴 **Only ~389 KB of metadata is on disk** |

⭐ **The interesting part is sentences, not hours.** Common Voice Thai train carries **32,823 distinct
sentences across 32,823 clips** -- every clip a different sentence -- where LOTUSDIS repeats 22,596
sentences five times each. Given `H39` (*text diversity beats audio duration*), Common Voice is the
more valuable corpus here despite being read speech. **Realistic training pool: ~65 h**, against
Typhoon's 11,000.

### The n-gram sweep -- change ⑦ is better supported than claimed, and also capped

He asked directly: *"is 4 gram is really a good solution, can you do research"*. Rows `A53`-`A55`.

✅ **Better supported than the single `A35` it was argued from.** Whisper-LM (arXiv 2503.23542) reports
**up to 51% relative in-distribution and 34% out-of-distribution** from *statistical* LMs on
low-resource languages, beating LLMs at the same job. An n-gram **beat an RNN-LM** on a low-resource
CTC model (Amazigh). A 35 h Komi system gained **>25% CER**. And Toshniwal 2018 (173 cites) still
holds: *"for first-pass decoding, the rather simple approach of shallow fusion performs best."*

🔴 **But the artefact on this disk is the weaker kind for Thai.** `A54` (Tian et al., IEEE SPL 2022)
measures **subword** n-grams as inferior to word-level ones *specifically for languages without word
boundaries* -- 14.8% relative CER on Mandarin. Ours is `ngram_4gram_bpe2048`: subword, 2,048 BPE.
Thai has no word boundaries. ⑦ stays first because it is free; it is not the ceiling.

⚠️ **And two practical facts the diagram had glossed:** the `.nemo` is a NeMo artefact while SenseVoice
runs under funasr -- the **`.arpa` is the portable route** (FlexCTC, `A55`, is CTC-native) -- and the
LM's text is *"the ASR training transcripts plus a synthetic in-domain corpus (English code-switch
terms inserted into LLM-generated Thai carrier templates)"*. ⭐ That second half is
`make_text_banks.py`'s method, arrived at independently by SCB DataX.

🔴🔴 **New risk `R7`: the 4-gram may have seen the benchmark.** Its card says *"no evaluation data"* --
meaning **Typhoon's** eval sets. `thai-contextasr-bench` is a Wayu/Paxa artefact (`A43`), not theirs,
and every number in this repo is scored on it. **Check the ARPA against the 682 reference sentences
BEFORE running ⑦**, not after. Plain text on disk; grep is enough.

### ⭐⭐ THAI-SER read end to end -- and it hands us a baseline we already beat

`arXiv:2507.09618` / *Language Resources and Evaluation* 60(3):59.

**Their trained baseline, on the same four emotions we score:** CNN+LSTM (64 mel filterbanks, 3-second
chunks, VTLP, 8-fold **speaker-independent** CV, Zoom excluded) reaches **WA 65-70%, UA 58-67%**.

| | |
|---|---|
| their trained CNN+LSTM, 4 emotions | **WA 65-70%** |
| **our `emotion2vec_plus_large`, ZERO-SHOT** | **0.7317** (410 actor-held-out) / **0.7381** (1,256) |
| linear probe on SenseVoice's frozen encoder | 0.6829 |

⭐ **A model that has never heard Thai, with no training, matches or beats their trained baseline.**
⚠️ Not like-for-like -- 8-fold CV over studio sessions vs our 8 held-out actors -- but the gap survives it.

**The annotation is human, and rigorously so.** 1,759 applicants, **984 passed (56%)**. Pretest =
tutorial video + 10 sample clips (>=5 correct) + a **trick question** (an animal sound hidden at a
random point in the video, answerable only if you watched). Then per batch: 3-8 annotators per
utterance with majority vote, **2 hidden gold clips** chosen by the studio directors, and **1 hidden
duplicate** for self-consistency. ⚠️ Annotators hear the voice only -- **no facial expression**.

🔴 **The raw corpus is not reliable.** Krippendorff's alpha **0.413**, below the 0.667 recommendation.
Filtering to agreement **>= 0.71** lifts it to **0.692**, and they state it as an instruction:
*"we strongly recommend that any researcher using the THAI-SER corpus remove utterances with an
agreement score lower than 0.71."* ✅ Our probes filtered at 0.70 -- right by instinct.

⚠️ **A number I could not reproduce, recorded as unreproduced.** The paper says filtering leaves
**7 h 43 m**. Measured on our three shards: agreement >=0.71 keeps **54.6% of clips and 55.0% of the
hours** (2,882 -> 1,575; 3.85 h -> 2.12 h), which scales to ~23 h, not 7.7 h. Excluding Zoom would give
~18 h. The gap is unexplained -- possibly their internal agreement score differs from the released
`agreement` column. 🔴 **I asserted "the reliable part is 7h43m" before measuring, then had to walk it
back. Quote "agreement-filtered", never an hour count.**

**Other findings worth keeping:** *frustrated* is confused with angry, sad **and** neutral -- holding it
out rather than mapping it to ANGRY was right. Neutral is recognised 78% -> 93% after filtering; happy
and frustrated sit ~61-62% and gain 18%/15%. **High-intensity acting agrees better than low-intensity**,
which the authors flag as a limit on acted corpora. **Zoom's 20 sessions are deliberately held out as
an out-of-domain challenge set** -- an untouched robustness axis. Cross-corpus, THAI-SER-trained models
beat IEMOCAP-trained ones on Emo-DB and EMOVO, and **training on improvised data generalises better**.

### Four corrections to the design, all found by him asking what a box meant

1. 🔴 **Gate 0 was wrong.** "Zero-init the new `ctc_lo` rows" does not make the other five languages
   bit-identical: `ctc_lo` is `torch.nn.Linear` and **has a bias**, so zero weights give a new Thai
   token a logit of exactly **0, not −∞** -- and competing tokens routinely score negative, so a new
   row at 0 can beat a real Chinese one at −2. Fixed: **weights zero AND bias ≈ −30**.
2. 🔴 **Adapters are NOT free at inference.** Drawn as a training-time side node; in fact all 50 layers
   gain two extra matrix multiplies, permanently.
3. 🔴 **EWC needs old-language audio to weight its springs** -- and that audio is not on disk (FLEURS,
   above). `S10` is now conditional, with **L2-SP** as the no-data fallback.
4. 🔄 **`S11` (adapters) demoted from decision to HYPOTHESIS**, gated on a new **`P0.3`** probe. His
   question: *"if we will finetune the tp_encoder, so what the benefit of adapter"*. The honest answer
   is that adapters add **1.69 M** against **~63 M** already trainable -- **2.7%** -- so they are not
   capacity, only *location*. Whether layers 1-50 collapsed Thai distinctions is measurable: a linear
   probe predicting **Thai characters** off frozen features, SenseVoice vs Typhoon. Half a day, same
   harness as `probe_encoder_emotion.py`.

⚠️ Also corrected: `S4` now freezes `after_norm` with stack 1. It is 1,024 params, but leaving it
trainable lets the "frozen" encoder's output scale drift and the do-no-harm gate silently stops holding.

### 🔴 Unexplained files, left uncommitted on purpose

`results/probe_emotion_thai_testactors.json` and `..._hyps.jsonl` appeared at **21:40 on 2026-09-19**
from a task this session did not launch, and `adapter/probe_emotion.py` was modified at **21:33**.

**Verified:** the result is internally coherent (confusion sums to 410, accuracy recomputes exactly,
schema matches this script's nine keys), and the actor set is **exactly** the eight held out by
`probe_encoder_emotion.py` -- `001 004 108 111 114 182 185 188`.

🔴 **But the script on disk cannot have produced it**: no actor filter, no `_hyps.jsonl` writer. So the
numbers are right and **not reproducible from the code that exists**. By `F8c`'s standard that is
disqualifying until explained. **Its numbers are recorded in `RESEARCH-PROGRESS.md` with that caveat
attached and must not be quoted without it.**

### Artefacts

- **Architecture diagram** (before/after stacks, node graphs, adapter detail, the seven changes):
  `https://claude.ai/artifact/VfvHMRKGFZ7JnS7JTXHEBd`
- **Paper list for his advisor** (five papers, each with the decision it supports):
  `https://claude.ai/code/artifact/6e6933fd-318a-44c5-b100-b399c662a728`

---

## 2026-09-19 (night) -- the encoder probe: he was half right, and SenseVoice's SAD failure is a HEAD problem

He pushed back on the architecture: *"isnt if we finetune on model that good at emotion it will
transfer to thai better?"* -- i.e. build on an encoder that already knows emotion rather than
bolting a head onto Typhoon's ASR encoder, which was never taught any. The argument has a
mechanism and it is his, not mine: **Thai tone is LEXICAL so Typhoon's encoder must preserve pitch
CONTOUR, but speaker REGISTER is pure nuisance for transcription and the encoder is rewarded for
discarding it -- and register is the emotional signal.** That is the same split the design spec
makes in `D5`. So the thing that makes Typhoon good at Thai could be the thing that kills its
emotion information.

Neither of us knew the magnitude, so it was measured. `adapter/probe_encoder_emotion.py` ->
`results/probe_encoder_emotion.json`.

**Instrument: a LINEAR probe.** Freeze the encoder, mean+std pool its frames, fit ONE affine layer
to predict the emotion. Deliberately too weak to be clever -- if a single layer can read emotion
off the features then the emotion is genuinely PRESENT, not reconstructed by the probe. **So every
number here is a LOWER BOUND**: a deeper head can only do better, which is why a low score is the
informative one.

Same 1,256 THAI-SER clips as the morning probe, **actor-disjoint** split (22 actors -> 8 held out,
410 test clips), regularisation chosen on dev actors and reported once on test, identical pooling
everywhere. The encoder is the only thing that varies.

| encoder | dim | test accuracy | 95% CI | macro-recall | gain over floor |
|---|---|---|---|---|---|
| majority class | -- | 0.3537 | -- | -- | -- |
| **mel, no encoder at all** | 160 | **0.5756** | [0.528, 0.623] | 0.5676 | -- |
| **typhoon** (frozen, 17 layers) | 1024 | **0.6512** | [0.605, 0.697] | 0.6652 | **+0.0756** |
| **sensevoice** (frozen, 70 layers) | 1024 | **0.6829** | [0.638, 0.728] | 0.7008 | +0.1073 |
| **emotion2vec** (frozen, trained for this) | 2048 | **0.7707** | [0.730, 0.811] | 0.7437 | +0.1951 |

✅ **HE WAS RIGHT ABOUT THE DIRECTION AND WRONG ABOUT THE MAGNITUDE.** Typhoon's encoder is NOT
empty of emotion -- 0.6512 against a 0.5756 mel floor and a 0.3537 majority class. But the gap to
emotion2vec is real and the confidence intervals **do not overlap** (0.697 < 0.730), so the 12-point
price of using the ASR encoder is measured, not hand-waved. **One pass is viable; it costs
something.**

🔴 **THE MEL FLOOR IS THE UNCOMFORTABLE NUMBER.** Raw log-mel statistics with NO encoder reach
**0.5756**, so the encoders are adding +7.6 / +10.7 / +19.5 points over doing nothing at all.
⚠️ Almost certainly because THAI-SER is **acted**: professional actors project with large obvious
energy and pitch swings that mean+std pooling of a spectrogram captures directly. On spontaneous
speech the floor should collapse and the encoder gaps should widen. **Nothing here is measured on
spontaneous Thai, and the whole ranking could compress or reorder there.**

### ⭐⭐ The finding that was not the question: SenseVoice's SAD failure is in the HEAD, not the ears

Put this morning's full-model result beside tonight's probe on **that same model's own encoder**:

| | SAD recall |
|---|---|
| SenseVoiceSmall, full model, its own CTC head (2026-09-19 morning) | **0.060** |
| linear probe on SenseVoice's own frozen encoder (tonight) | **0.830** |

**The SAD information is sitting in SenseVoice's encoder at 0.830 recall, and its own output head
emits it at 0.060.** One affine layer recovers what the shipped model refuses to say. The 43.6%
`EMO_UNKNOWN` abstention measured this morning is therefore **not** the model failing to hear Thai
emotion -- it is the CTC head, which shares its vocabulary and output layer with the text, declining
to emit a token its encoder has the evidence for.

⚠️ **Scope, stated because the numbers come from different subsets.** The morning figure is over all
1,256 clips; the probe is 410 clips from 8 held-out actors. They are not the same denominator. But
0.060 -> 0.830 is far outside anything a split difference produces, and the direction is
corroborated by Typhoon's encoder also reading SAD well (0.717) while having no emotion training at
all. **Re-scoring the morning arm on the identical 410 clips is one cheap command and should be run
before this is quoted anywhere.**

⭐ **Consequence for the architecture, and it is the opposite of what the morning suggested.** The
morning result said "SenseVoice abstains on Thai." Tonight says "the encoder knows; the head is the
bottleneck." A **new head on a frozen encoder is therefore the right shape**, because the head is
exactly the part that is broken -- and that is true whichever encoder is chosen.

### Per-class, where each encoder earns its keep

| encoder | NEUTRAL | HAPPY | ANGRY | SAD |
|---|---|---|---|---|
| mel | 0.676 | 0.500 | 0.510 | 0.585 |
| typhoon | 0.600 | 0.618 | **0.725** | 0.717 |
| sensevoice | 0.697 | 0.600 | 0.676 | **0.830** |
| emotion2vec | **0.848** | **0.709** | 0.814 | 0.604 |

⚠️ **Typhoon is WORSE than the raw mel floor on NEUTRAL** (0.600 vs 0.676) while beating it
comfortably on ANGRY and SAD. Its accuracy is dragged by the largest class, which is why its
macro-recall (0.6652) sits closer to SenseVoice's (0.7008) than the headline accuracies suggest.
Read the macro column, not the accuracy column, when choosing the encoder.

⭐ **And emotion2vec is the WORST of the three encoders at SAD** (0.604, against SenseVoice 0.830 and
Typhoon 0.717) despite winning overall. The reference model is not uniformly better, which is an
argument for measuring per class before committing to an encoder.

### Engineering notes

- **Two venvs, unavoidable.** Typhoon needs NeMo (`E:/asr-spike`, which has **no torchaudio**);
  emotion2vec and SenseVoice need funasr (`D:/sensevoice-spike`, no librosa). `to_wav16k` now tries
  torchaudio and falls back to librosa so the same code runs under either interpreter.
- The Typhoon loader prints `encoder layers=17 d_model=512` and uses the same model id as
  `asr/server.py:49`, so a probe cannot quietly measure a different checkpoint from the one every
  other arm scores -- the 2026-09-17 failure that cost two full runs.
- 1256/1256 clips extracted cleanly on all four encoders. CPU only; the GPU was never touched.

---

## 2026-09-19 (later) -- emotion DOES transfer to Thai zero-shot, but through the TEACHER, not SenseVoice

He asked whether fine-tuning SenseVoice would give us emotion detection and whether it would
transfer to Thai, then said "do it all. I will go eat." So it was measured instead of argued.
`adapter/probe_emotion.py`, `results/probe_emotion_thai.json`.

**Setup.** THAI-SER (`airesearch/thai-ser`, CC-BY-SA-4.0, ungated, 27,854 utts / 41h36m / 200
actors). Shards 0/10/20 of 29 pulled -- spread, not the first three, so actors and rooms vary.
2,882 rows scanned -> **1,256 kept** (majority_emo in the 4 classes SenseVoice can express,
annotator agreement >= 0.7), 5,912 s of audio, 22 actors, plus **760 held-out `Frustrated`** as a
diagnostic because SenseVoice has no class for it. Zero training, zero GPU, CPU only.

🔴 **A separate venv, and that was not caution theatre.** SenseVoice pins `numpy<=1.26.4`;
`E:/asr-spike/.venv` runs **numpy 2.4.6** under NeMo 3.0.0. Installing funasr there downgrades
numpy beneath NeMo. Built `D:/sensevoice-spike/.venv` with CPU torch instead; asr-spike verified
unchanged afterwards (torch 2.13.0+cu130, CUDA True, numpy 2.4.6, NeMo 3.0.0).

### The numbers

| system | accuracy (4-way) | 95% CI | macro-recall | non-committal |
|---|---|---|---|---|
| majority-class floor (always NEUTRAL) | 0.3304 | -- | -- | -- |
| **SenseVoiceSmall** (the student) | **0.4761** | [0.448, 0.504] | 0.4322 | **43.6%** |
| **emotion2vec_plus_large** (the teacher) | **0.7381** | [0.714, 0.762] | 0.7194 | 6.1% |
| human ceiling, annotators vs assigned | 0.9286 | -- | -- | -- |
| human ceiling, leave-one-annotator-out | 0.9511 | -- | -- | -- |

⚠️ Both ceilings are measured on **this filtered 4-way subset**, which is selected for annotator
agreement and therefore easier than the corpus. THAI-SER's own published human accuracy is
**0.772**, 5-way over everything (arXiv 2507.09618). The three are not interchangeable and all
three are recorded so nobody later quotes whichever flatters a model.

⭐⭐ **THE RESULT IS THE REFUSAL RATE, NOT THE ACCURACY.** SenseVoice emits `EMO_UNKNOWN` on
**547 of 1,256 clips (43.6%)**, and that token can never be scored correct, so refusals alone cap
it at 0.5645. **On the 709 clips where it does commit, it is right 84.3%** -- higher precision
than emotion2vec's 78.6% when committing. The student is not confused by Thai; it **abstains** on
Thai, and is unusually trustworthy when it does not.

**Where it abstains is specific.** From the confusion matrix: `HAPPY->EMO_UNKNOWN` 209 of 352
(59%) and `SAD->EMO_UNKNOWN` 151 of 200 (76%), against `NEUTRAL->NEUTRAL` 268 and
`ANGRY->ANGRY` 193. Per-class recall: ANGRY 0.668, NEUTRAL 0.646, HAPPY 0.355, **SAD 0.060**.
So arousal survives the language change and valence does not -- high-arousal states (angry,
neutral-as-default) transfer, low-arousal sadness is essentially invisible.

✅ **The teacher answers the question the student cannot.** `emotion2vec_plus_large`'s card claims
to be *"overcoming the effects of language ... to achieve universal, robust emotion recognition"*
and **names no languages**. It now has a Thai number: **0.7381**, refusing on only 6.1%. So the
capability is genuinely language-transferable; what SenseVoice lost, it lost in distillation into
a 5-language CTC vocabulary, not in the acoustics.

### Three secondary results, all free

⭐ **The audio-event head transfers outright.** 1,245 of 1,256 clips tagged `<|Speech|>`, plus one
`<|Laughter|>` and one `<|Cry|>`. Event detection is not language-bound, which is what one would
expect and is now measured rather than assumed.

🔴 **SenseVoice calls Thai CANTONESE.** Raw outputs are `<|yue|>` far more often than anything
else, with `<|en|>` and `<|ja|>` appearing too, and the transcript is confident garbage --
`<|yue|><|NEUTRAL|><|Speech|><|woitn|>婆话 look下你正话狂啲鞋摆嘅佢嘅 bow` on Thai speech. The LID
head has six slots and no `th`, so it picks the nearest tonal neighbour. Sample outputs kept in
`sensevoice_raw_samples` in the result JSON.

**Duration gates the signal, and 2 s is the knee.** SenseVoice: 0.344 under 1 s, 0.288 at 1-2 s,
**0.537 at 2-4 s**, 0.493 over 4 s. emotion2vec: 0.656 / 0.697 / 0.770 / 0.737. Prosody needs
time; THAI-SER's mean clip is 2.72 s and its minimum is 0.53 s.

**Acted-and-scripted is easier than improvised**, for both: SenseVoice 0.554 script vs 0.412
impro, emotion2vec 0.786 vs 0.699. ⚠️ The impro half is the more honest number and the gap is the
same shape as `F5` -- an easier evaluation split flattering every system on it.

**Held-out `Frustrated` behaves sensibly:** 429 of 760 -> `EMO_UNKNOWN`, 179 -> NEUTRAL, 96 ->
HAPPY, 52 -> ANGRY. An emotion with no class mostly triggers the refusal rather than being forced
into a neighbour.

### Two bugs, both caught before the real run

🔴 **The token regex was `[A-Z_]+`, uppercase only.** Emotion tokens are all-caps so arm 1 was
unaffected, but every EVENT token is mixed case (`<|Speech|>`, `<|Laughter|>`) and read back as
"no event emitted" -- a 100%-`NONE` distribution in the 20-clip smoke test, which is the kind of
result that is wrong rather than interesting. Fixed to `[A-Za-z_]+` before anything was recorded.

⚠️ **`pq.read_table` on one shard peaked at 1.5 GB resident**, and three shards would have been
~4 GB on the box whose commit limit killed a job the same night. Rewritten to stream row groups
(100 rows, ~40 MB peak). The full run held ~3.6 GB, which is the two fp32 models, not the data.

### What it does and does not license

✅ **Thai has a measured paralinguistic baseline it did not have this morning**, on an open
CC-BY-SA corpus, reproducible on CPU in 12 minutes.
✅ **A useful product signal exists for free:** 84.3%-precise emotion on the ~56% of Thai turns
SenseVoice will commit to, from a model that has never heard Thai, at no training cost.

🔴 **It does NOT say fine-tuning SenseVoice on Thai is a good idea.** Nothing here tested that, the
tokenizer surgery and forgetting problem are untouched, and the emotion head shares `ctc_lo` with
the text so Thai-only fine-tuning would degrade position 2 unless emotion stays supervised.
🔴 **And it is ACTED emotion from professional actors**, three fixed sentences in the script half.
Real spontaneous Thai affect is not measured and will be worse.
⚠️ Single mic channel (`mic_clip`); the other three are downloaded and unrun, which is a free
channel-robustness axis nobody has spent.

---

## 2026-09-19 -- he asked whether the D6 splice is ours. It is not: it is CTC-WS, and it ships in our NeMo

His question, after four days of me calling D6 "the arm that worked" and proposing a contribution on
top of it: *"is RNN-T + CTC spot really our idea?"* Two Consensus sweeps, and the answer is no.

**`A46` -- Andrusenko et al. 2024, arXiv 2406.07096, "Fast Context-Biasing for CTC and Transducer ASR
models with CTC-based Word Spotter."** Their abstract, sentence for sentence against D6: match CTC
log-probabilities against a compact context graph to detect candidates; *"the valid candidates then
replace their greedy recognition counterparts in corresponding frame intervals"*; *"a Hybrid
Transducer-CTC model enables the CTC-WS application for the Transducer model."* And:
*"publicly available in the NVIDIA NeMo toolkit."*

🔴 **So the mechanism was already inside the toolkit this repo runs.** Typhoon's `kws_spot.py` is an
implementation of it, not their invention either. `A47` (Tsai et al. 2026) is its streaming
extension. `A48` -- **TurboBias**, Andrusenko again, ASRU 2025 -- is `GPUBoostingTreeModel`, which is
arm **D4**. Both of this repo's biasing arms are one author's line.

🔴🔴 **And the architecture contribution I had recommended two messages earlier is also published.**
`A49`, Huang et al. ASRU 2025: an attention-based biasing decoder scoring candidate phrases from
ASR-encoder acoustics, per-token discriminative objective *"encourages higher scores for ground-truth
phrases while suppressing distractors"*, used to filter candidates and compute the fusion bonus,
*"modular and can be used with any ASR system."* That is the "learned arbiter" box I drew for him,
one year earlier. Neighbours at `A49`: Liu 2021 (keyword-confidence module) and Qiu 2021 (learned
word-level confidence).

**This is the third time in this repo and the pattern is identical each time** -- `H15` (the
attention reframe, published 2023), `E61` (the latency-breakdown claim, 2021), and now this. `F16`
already states the rule. What is new is that the rule did not fire on its own: **he fired it.**

✅ **What survives, stated so it cannot drift back into a novelty claim.** The Thai measurement is
untouched -- recall .328 -> .554, +99 of 439 gold mentions, U-CER/CER/B-CER all falling at
RTF 0.028 -> 0.032, `bias@50` against `dis@50` -- and so is `F8b`'s progression: re-rank one model's
candidates **-1 mention**, bias its scores **+8**, take a second model's opinion of the audio
**+99**. That is a finding about where the bound sits. **Quote D6 as the first Thai measurement of a
published method, never as a method.**

⚠️ **Only abstracts were read.** Two questions need the full `A46`/`A47` before anything is built:
(a) do they report a **threshold study**? The 2026-09-17 tau sweep found ~45 gold mentions between
-1.5 and -2.5 that no constant can take, because the loosening that admits them also admits the
fabrications -- that is the one shape of gap still standing. (b) Does their *jointly-trained* Hybrid
Transducer-CTC differ materially from our *two separately released checkpoints sharing a frozen
encoder*? Reading absence-of-mention as absence-of-result is exactly how `A44` went wrong four days
ago.

**Corrections written in place, not deleted:** `RESEARCH-FINDINGS.md` `F9` (the line "Unmeasured by
anyone: what splicing CTC spots into an RNN-T transcript does to U-CER" was doubly wrong) and
`RESEARCH-PROGRESS.md`'s D6 section, where the paragraph beginning "The contribution is keeping the
better transcript" is kept verbatim with its authorship claim retracted above it. The measurements in
both are correct and reproducible; only the authorship was false.

### The same session: the Typhoon line is adaptation end to end, and two of its models were unknown here

Enumerating `typhoon-ai` on HuggingFace (82 models, 19 speech) to answer "what are their best models"
turned up two things the repo did not have.

🔴 **`typhoon-asr-streaming-115m` and `typhoon-asr-streaming-nemotron-0.6b` (2026-07-07, `-ngram`
variants 2026-09-15) have never been touched here.** The 115m card states plainly that the
full-context base *"collapses to 62.8% CER when forced to stream at 1040 ms"* -- that base is
`typhoon-asr-realtime`, the model on `:8007`. It is fine as used (whole utterance at a time), but
`A1`'s "packaged as streaming and is not" now has a vendor-supplied number attached, and a real
streaming replacement exists. Nemotron-0.6b claims CER **14.1%** TVSpeech / **9.3%** GigaSpeech2 at
1040 ms. Unverified here.

⚠️ **And I told him Typhoon ships no medium/large CTC pair. Wrong** -- `typhoon-whisper-medium-ctc`
and `typhoon-whisper-large-v3-ctc` (2026-09-06/08) are exactly that: 29 M character-CTC heads on
frozen Whisper encoders, which is `A27`, already in the ledger. The search that missed them was
`search=typhoon-asr` against models named `typhoon-whisper`. Same failure shape as `A44`: a partial
enumeration reported as a complete one.

**The lineage, from every card's `base_model` field.** `typhoon-asr-realtime` declares
`nvidia/stt_en_fastconformer_transducer_large` -- an **English** checkpoint -- and its config matches
NeMo's stock FastConformer Large recipe on every value (17 layers, d_model 512, 8 heads, kernel 9,
dw_striding x8, pred/joint hidden 640, 1 LSTM layer, xscaling true). The whole catalogue is three
moves: fine-tune someone's checkpoint (asr-realtime, all whispers, qwen-ctx via merged LoRA), freeze
their own encoder and bolt on a head (nemo-ctc, whisper-*-ctc), or swap a structural component and
warm-start (streaming-115m). **Their contribution across the line is the ~11k-hour Thai data pipeline
and a Thai tokenizer, not an architecture** -- which is what this repo wrote on 2026-09-11 and then
drifted away from.

⚖️ **Read both ways.** The bar in Thai speech AI is not "invent a layer" -- the strongest lab here
published on adaptation. But freeze-and-bolt-on is *their house move*, done twice this month, so it
is by definition a few days of their work.

### And a synthetic-data row worth more than the question that found it

Chasing "is SenseVoice's training data real or synthetic?" into the FunAudioLLM report
(arXiv 2407.04051v3) produced `H39` and `H40`.

🔴 **The provenance question has no published answer.** Section 3.1 on SenseVoice's data is four
sentences: ~300,000 h for Small over 5 languages, +100,000 h of multilingual for Large, and nothing
about where any of it came from. Section 3.2 documents CosyVoice's collection in detail. They
documented the TTS model's data and not the ASR model's. What *is* machine-generated is the
**labels** -- 150 M AED entries via `audioset_tagging_cnn`, 30 M SER via `emotion2vec_plus_large` --
which is pseudo-labelling on audio of unstated origin, not synthetic audio. Same shape of gap as
`H29`. **Pseudo-labelled is not synthetic, and the distinction must survive into the thesis.**

⭐⭐ **`H39` is the find.** Their §4.7 Table 13, CosyVoice used as an ASR data generator on
LibriSpeech, scored on **human-uttered** test sets: synthetic-only **3.00** against 960 h of real at
**2.79** on test-clean -- a real/synthetic ratio of **~1.08**, far past the 1.46 in `H29`'s line. And
their stated conclusion is about text, not audio: adding synthetic speech built from *MLS* sentences
took real+synthetic from 2.56 to **2.04 / 4.53**, and *"this may indicate that the text diversity is
more critical for the ASR task than the duration of speech itself."*

➡️ **That aims at Phase 1b.** `H7`'s exchange rate sweeps hours; `H39` says the axis that paid was
the **text bank**. The current design contains no arm that holds hours fixed and varies text.
🔴 **It is not transferable as written**: English read speech, their own TTS judged by its own
authors, and **the paper never states which ASR architecture was trained** -- so `D4` ("every arm
runs on the transducer") cannot be checked against it, which is the whole reason `H2`/`H10` exist.
`UNTESTED`, and it stays there until something runs here.

---

## 2026-09-18 (evening) -- the standard augmentation does NOT damage Thai, and rows 4-5 lose their premise

Before building a tone-preserving pitch transform, test the assumption under it: does the careless
one hurt? `adapter/probe_pitch.py`, 40 real Common Voice Thai sentences x 8 conditions = 320 clips,
read back by biodatlab. Real audio, so TTS wobble cannot be blamed. Two instruments: ordinary CER,
and **TONE-CER** -- edit distance over the tone marks alone, separable because Thai writes tone as
characters (`_thai.tone_sequence`, 211 tone marks per condition).

| condition | CER vs artifact floor | TONE-CER vs floor | tone/CER |
|---|---|---|---|
| roundtrip (vocoder, 0 semitones) | +0.0000 | +0.0000 | 1.38 |
| shift -4 semitones | +0.0753 | +0.0714 | 1.07 |
| shift -2 | +0.0552 | +0.0357 | 0.91 |
| shift +2 | +0.0558 | +0.0319 | 0.86 |
| shift +4 | +0.0659 | +0.0677 | 1.14 |
| **speed 0.90** | **-0.0025** | **-0.0142** | 1.00 |
| **speed 1.10** | **-0.0054** | **-0.0021** | 1.60 |

🔴 **SPEED PERTURBATION -- the augmentation actually in everyone's ASR recipe (Ko et al. 2015) --
COSTS NOTHING IN THAI.** Both directions land at or below the artifact floor on CER and on
TONE-CER. The premise of row 5, *"if this damages Thai where it was merely ineffective in
English, that is the finding"*, is **not supported** by the test that was supposed to establish it.

🔴 **And the phase-vocoder shifts damage tone and text at the SAME rate** (tone/CER 0.86-1.14, and
the untouched clips already sit at 1.28-1.38). Tone-specific damage would show as tone degrading
FASTER. It does not. This looks like general degradation, not a tonal-language effect.

⚠️ **THE LIMITATION THAT STOPS THIS BEING CONCLUSIVE, AND IT IS MINE.** The artifact control runs
the vocoder at **zero** semitones, so it measures the floor at zero and nowhere else. A phase
vocoder distorts more the further it stretches, so the +0.055-0.075 CER at ±2-±4 is an unknown mix
of "pitch shifting hurts Thai" and "librosa's pitch shifting hurts audio". Separating them needs a
clean vocoder -- WORLD/`pyworld` -- which is absent from both venvs. **So the honest reading is:
the standard recipe is safe (strong, it needs no vocoder at all), and naive pitch shift is
unproven either way (weak).**

➡️ Rows 4-5 need reshaping before anything is built for them. Not killed -- the "isolate one axis"
argument from this morning still stands, and that is now their only support.

---

## 2026-09-18 -- he asked whether I had tried the voice-design prompt. I had asked it the wrong thing

His question: *"do you try gen voice using the prompt func of vox"*. I had -- `noclone_*` in
`probe_control.py` is exactly design mode -- but I asked that knob to *"speak slowly"* and then
reported "control text does not work". `voxcpm/cli.py` has one documented example of it:
`--control "warm female voice"`. It is an instruction about WHO IS SPEAKING, and I tested tempo.
The honest claim from that probe was narrower: it does not control speaking RATE.

Worth testing properly, because reference cloning is capped at the 150 Common Voice clips this
repo owns, and a voice designed from a sentence would be an unbounded speaker supply -- and for
ASR training data speaker diversity is worth more than tempo.

**`adapter/probe_voice.py`, 4 texts x 8 conditions, paired by text so the same seed faces every
instruction. Readback median CER 0.000, 29 of 32 kept -- the words are right, so the pitch means
what it says.**

🔴 **The gender instruction does nothing.** "deep male voice" minus "warm female voice" = **-4.4
Hz**, paired. A real male/female gap is 80-100 Hz, and the random seed alone moves pitch **37.8
Hz**. An order of magnitude the wrong way. Only "young child" separates (+50 Hz mean) and it is
unreliable: +153 Hz on one sentence, +1.8 Hz on the next. A Thai instruction scored within 2.3 Hz
of the English one -- neither works, so neither wins. **The seed picks the voice.**

✅ **The free variety is the finding.** Design mode invents a speaker per draw; across four
sentences pitch ran 105-251 Hz with the words intact. VoxCPM2 is an unbounded speaker supply that
simply cannot be steered. For a corpus with 150 reference speakers that is worth more than it
sounds, and it costs a different seed.

So Phase 2 cannot hold a speaker fixed while varying something else, and cannot request a
demographic. It can set rate by reference selection (1.42x) and sample speakers freely by seed.

⚠️ Pitch is a proxy: it proves voices separated, it cannot prove two clips at the same pitch are
the same speaker. Still needs an ear.

---

## 2026-09-17 (last) -- R2 answered: reference selection controls, control text does not

He chose "probe all three cheaply first" over committing eight weeks to a guess. `probe_control.py`:
8 texts x 9 conditions, one attribute measurable without a model (speaking rate), and a NOISE FLOOR
as a condition -- same text, same voice, new seed -- because VoxCPM samples at every step and a
mechanism that moves rate less than the dice do is not a control.

**The answer.** Noise floor 0.96 ch/s.

| mechanism | result | |
|---|---|---|
| B reference selection | 11.83 -> 16.82 ch/s, **5.2x noise**, readback clean | ✅ USABLE |
| C control + cloning | 0 of 8 clips clean, readback CER median 0.157 | ❌ the model SPEAKS the instruction |
| C control, no cloning | 15.77 -> 15.82 ch/s, 0.0x noise, readback clean | ❌ consumed and ignored |

`cli.py:123` refuses to combine `--control` with `--prompt-text`, and that refusal is **real at the
model level**: the readback of `(พูดช้าๆ)สวัสดีค่ะ...` literally begins `พูดช้า ๆ สวัสดีค่ะ`. The
prefix is only stripped in voice-design mode, where it then does nothing to the rate.

⚠️ **B's usable range is 1.42x, not the bank's 3.3x.** The references span 3.95-13.10 ch/s; the
clips made from them span 11.83-16.82. VoxCPM2 borrows a reference's rate and compresses it toward
its own default. Phase 2 must be planned around the achieved range.

🔴 **TWO FULL RUNS, 32 MINUTES, WERE SPENT MEASURING THE WRONG MODEL.** I hardcoded
`openbmb/VoxCPM-0.5B`; `colab/gen_colab.py:87` uses `openbmb/VoxCPM2`, the 2B 30-language
checkpoint that is the only one with Thai (`H29`). The 0.5B produced **19 of 72 catastrophic**
clips (0.46 s and 41 s for the same 75-character sentence) and of course ignored `(control)`, which
`cli.py:376` calls "Control instruction for VoxCPM2". I then spent a third run testing whether my
*settings* caused the instability, and a fourth diagnostic on whether my *reference choice* did --
two hypotheses that a single grep for the model id would have pre-empted. VoxCPM2 on the same
material: **1 of 72**. `probe_control.py` now IMPORTS `VOXCPM_MODEL`, `CFG_VALUE` and `TIMESTEPS`
from `gen_colab.py` and prints them, so a probe can no longer disagree with the corpus about what
it is testing.

🔴 **And the full corpus run would have crashed 40 seconds in, before any of this.** torchaudio
2.11 routes `load()` through torchcodec, whose FFmpeg DLLs are not on this box, and VoxCPM loads
every reference wav that way -- the model loads fine, then the first clip dies with
`libtorchcodec_core5.dll`. Shimmed to soundfile in `probe_control.py` AND `colab/gen_colab.py`.

**Also corrected mid-probe:** rate was first computed from clip length, which made `base_a` 21.3
ch/s and `base_b` 3.6 -- a 6x "effect" that was trailing silence. `gen_colab.py:113` already said
so: *"VoxCPM samples its own trailing pause (measured 0-2380 ms per reply), so without this the
clip length carries information about the draw rather than about the speech."* The denominator is
the voiced span now.

---

## 2026-09-17 (latest) -- the gate was passing clips whose one important word was destroyed

Following `H37`'s warning that no self-distillation overcomes a TTS's rare-word ceiling, I tested
the ceiling on the 12 clips already generated. All the data was on disk: each clip carries four
independent ASR readbacks plus the rare entity it was built around. No GPU, no generation.

**First answer, and it was wrong.** Asking "does the entity appear verbatim in the readback"
said **4 of 12 unsayable**, with all four jurors agreeing on a wrong string -- which reads as
VoxCPM mispronouncing rather than `H38`'s false alarms.

**Second answer, after scoring the entity by distance instead of exact match: 0 of 12 unsayable.**
Every one of those four was **one tone mark or one letter** out -- `อุไดเจริญ` heard as `อุได้เจริญ`
(0.11), `กาเยาะมาตี` as `กาเยาะมาตรี` (0.10). An exact-substring test on Thai rare names reproduces
the exact error `F17` is about, and I nearly wrote its conclusion into the record on the same day I
committed the finding that names it. The ceiling is not firing hard on this sample.

**What the pilot did find is a real gate defect.** All 12 clips passed `--cer-max 0.10`. Scored on
the entity alone, two were wrecked: `พลับพลาไชย` -> `พัฒนาชัย` / `พรรคพราชัย`, entity-CER 0.40-0.60
for three of four jurors, inside a sentence still scoring under 0.10 **because the name is 10
characters of 85**. For `phase1_known` that is harmless; for `phase2_rare` the entity IS the clip,
and a sentence built around a rare name that does not contain a sayable version of it teaches the
model a wrong spelling for a sound. That is worse than not training on the clip at all.

**New: `--entity-max 0.25 --entity-votes 2` in `verify_synth.py`**, plus `infix_cer` in `_thai.py`
(edit distance to the best-matching stretch of a hypothesis, so one word can be scored inside a
sentence). Swept against the 12 real clips: rejects exactly the 2 damaged ones, and the verdict is
identical anywhere in 0.15-0.25 -- a plateau, not a knife edge. 4 new selftest cases, every one a
real row from those clips, including the two that set the threshold from both sides.

⚠️ 17% is a pilot rate on 12 clips and the threshold has still not met an ear. `H38` puts
TTS-derived ASR false alarms at 21-34%, which is larger than the effect being gated on.

---

## 2026-09-17 (late) -- a third of the benchmark's "error" is spelling, and it reorders the table

Acting on `A40`, which I had written into `LITERATURE.md` that morning and not tested. Its claim:
accounting for permissible orthographic variation narrowed a Gemini-vs-Canary gap from 18.1 points
to 11.5. `F3`'s "the rare-word gap is field-wide" rests on a raw-CER ranking of seven Thai systems,
and Thai has exactly the variation that paper normalises. All seven hypothesis files were already
on disk, so the test cost no GPU and no decoding.

**New: `adapter/_ortho.py` (the rewriter + selftest) and `adapter/ortho_rescore.py` (the driver).**
`thai_cardinal` / `thai_digitwise` moved out of `verify_synth.py` into `_thai.py` when `_ortho.py`
became a second caller -- two copies of a number speller is how a convention rule drifts apart.

**The finding.** Neutralising one family -- digit runs against Thai number words -- reorders the
table: Kendall tau **0.429**, 6 of 21 pairs swapped. Ours goes **7th of 7 to 5th of 7**, CER
0.11248 -> 0.09493 (-15.6%), with no change to the model. Classifying all 56,232 aligned error
characters: script 20.4%, numbers 9.1%, decoder collapse 3.0%, fillers 1.0% -- **~30% convention**.
`F3`'s ratio claim survives (B/U 2.01-3.04 -> 2.07-3.08); its CER ordering does not. `A39` and
`A40` moved to `CONFIRMED`; `F17` written; `RESEARCH-PROGRESS.md` carries the re-scored table.

**Two bugs, and they were caught by different instruments -- which is the point.**

- The `ent` rung rewrote `คลินตัน` into `คLinตัน`. The corpus has an entity `ลิน`/`Lin` and `Lin`
  sits inside the hypothesis `Clint Clint`; the same mechanism made `นิวซีแลนด์` into `Newซีแลนด์`.
  447 corpus forms are 3-4 Thai characters. **A 26-case selftest had passed green**; this came out
  of printing six real rewrites and reading them. The benchmark carries a guard for exactly this
  collision, citing `ไลน์` inside `เดทไลน์` -- I rebuilt it without knowing.
- The `filler` rung ate `อ่าน`, "to read": a substring replace of `อ่า` turns `ผู้อ่าน` into `ผู้น`,
  35+ times. **The harness caught this one, not the eye** -- entity recall is invariant across rungs
  by construction, the driver asserts it, and it moved 0.003 for one system. Bounded to whitespace
  now, which leaves most unspaced fillers alone, so the rung is worth almost nothing and is off by
  default. That is the measured answer to "should we strip fillers".

**And the first draft's selection rule was the third bug.** It picked the reference rendering with
the smallest edit distance to the hypothesis; its own selftest killed it, on reference
`Khlong Toei` against a hypothesis saying `บางกะปิ` -- a flatly wrong place -- which it rewrote to
`คลองเตย` because a Thai candidate is nearer to Thai output than a Latin one. The rule is now exact
verbatim presence: the reference may only be re-rendered into a form the hypothesis actually
contains. It under-corrects, and that is the direction to be wrong in.

🔴 **Every number recorded in this repo stays the unnormalised `none` rung.** This is a caveat on
reading a Thai CER table, not a restatement of our results. Also corrected today: the first page of
`RESEARCH-PROGRESS.md` still said no synthetic audio existed, which stopped being true yesterday.

---

## 2026-09-17 (night) — the first synthetic audio, and the filter was the thing that was broken

🔴 **The repo's headline changes today: synthetic audio now exists.** Twelve clips from
`phase2_rare`, generated **locally** -- `colab/gen_colab.py` is plain Python and Colab was only ever
where a free GPU was. 12 distinct voices, 0 silent, RTF 0.698, so the full 5,480-clip corpus is
about **7 wall hours on this box**.

### VoxCPM's Thai is faithful, and the panel says so with five voices

| juror | accepts | CER median |
|---|---|---|
| typhoon-whisper | 100% | **0.006** |
| qwen3-asr | 100% | 0.037 |
| pathumma | 92% | 0.042 |
| biodatlab | 100% | 0.050 |
| **target** (the model being trained) | **75%** | **0.077** |

⭐ **Our best juror reads the generated Thai at CER 0.006 -- better than the 0.94% VoxCPM2 claims on
its own benchmark.** So `R1` (does VoxCPM say what it was told?) is answered in the affirmative for
this bank, and `H29`'s intelligibility half is confirmed independently rather than taken from a
card. ⚠️ n=12. This establishes the pipeline, not a rate.

⭐⭐ **The teach set is 3 of 12 (25%)** -- clips every juror heard and `typhoon-asr-realtime` could
not. Its errors are concrete and repeat: it writes **ไปรัศนี for ไปรษณีย์** (postal code) and
**บางลักษณ์ for บางรัก**, where qwen scores 0.000 on the same audio. That is the training signal the
design is built to isolate, and it appeared in the first twelve clips ever generated.

### 🔴 BUT THE FIRST RUN REJECTED 12 OF 12, AND THE FILTER WAS AT FAULT BOTH TIMES

**Fault 1, the repetition detector.** It took the MAXIMUM pairwise similarity between mel windows.
Real speech always has two similar windows -- same speaker, recurring phonemes, pauses -- so it was
measuring "is this audio", not "is this a loop". Measured: real Common Voice scored **median 0.9983**
against synthetic's 0.9973, i.e. **real speech looked MORE repetitive than a TTS clip**, and the
0.985 threshold rejected everything. Rewritten as a **lag search** -- a loop is consecutive windows
repeating at one fixed period, so score each lag by its mean similarity down the whole diagonal.
Now: real speech tops out at 0.489, synthetic at 0.620, a real clip tiled 3x scores **0.999**.
Threshold 0.80 sits in the gap, calibrated rather than guessed.

**Fault 2, the tail check.** It required the reference's last characters VERBATIM, and Thai has
accepted spelling variants:

    label  ...ใช่ไหมครับ      biodatlab  ...ใช่มั้ยครับ       same particle
    label  ...เท่าไรคะ        biodatlab  ...เท่าไหร่คะ         same word, colloquial

Nothing was dropped; both were rejected. Rewritten to compare the tail with a tolerant CER rather
than match it -- a deletion sends that toward 1.0, a respelling leaves it small. ⚠️ Verified it
still catches the real case on the same clip: pathumma ended `...ใช่ไหม` where the label ends
`...ใช่ไหมครับ`, a genuinely dropped ครับ.

### ⭐ THE PATTERN, WHICH IS NOW WORTH STATING AS A RULE

**Four Thai text conventions have each manufactured a wrong answer in this repo:** SARA AM (`F1`),
digits (`F2`), script (killed the vocabulary hypothesis 2026-09-15), and now **colloquial spelling**.
Every one looks exactly like a recognition error and is not. Assume a fifth exists.

**And three tools failed today in one shape: passing their own tests while being wrong.** The
repetition detector tested that it FIRES, never that it stays QUIET. The tail check tested a dropped
tail, never a respelled one. `compare_baselines` had a three-line comment in its sibling about
pinning the backend and no pin of its own. 🔴 **The common cause is testing the failure you imagined
and not its opposite.** Both directions are pinned now, in `--selftest`.

---

## 2026-09-17 (later still) — spontaneous Thai costs 4.2x, and the capture device costs 13x

He said it first: Typhoon "even miss the normal conversation context". The instrument had hinted at
it for a month and nobody followed up -- 0.0518 on Common Voice against 0.1125 on the benchmark --
but those two differ in corpus, speakers, channel AND register at once, so the gap could not be
attributed to anything. `A19` says only that a Thai read-vs-spontaneous gap exists; nobody publishes
what it costs. New arm: `adapter/arm_spontaneous.py`.

### Why LOTUSDIS makes this controlled rather than merely comparative

Its test split holds **5,516 rows per device across five devices, row-aligned** -- verified
5516/5516 identical on sentence, speaker_id and duration, and re-verified on the sampled rows
before scoring rather than trusting row order. The same utterance was recorded simultaneously on
all five, so those rows differ in **exactly one factor**. That is the only single-variable
comparison in this repo.

### The two numbers

**Read -> spontaneous: CER 0.0518 -> 0.2174, a 4.2x penalty.** 150 utterances >= 3 s, same
`_thai.py` fold as `arm1_baseline`, same box, same server, same window.

| device | CER | median | p90 | perfect | vs read |
|---|---|---|---|---|---|
| `lav123` | 0.2174 | 0.1667 | 0.4861 | 12/150 | 4.2x |
| `con123` | **0.2054** | 0.1578 | 0.4615 | 16/150 | 4.0x |
| `jbl` | 0.2391 | 0.1816 | 0.5167 | 9/150 | 4.6x |
| `bt3m` | **0.6808** | 0.6958 | **1.0000** | **0/150** | **13.1x** |
| `bt10m` | **0.6401** | 0.6540 | 0.8868 | **0/150** | **12.4x** |

⭐⭐ **The device costs more than the register does.** Three devices land at 0.21-0.24 and two at
0.64-0.68 -- unusable, with **zero perfect transcripts out of 150** and a p90 of 1.0000, meaning a
tenth of those utterances are transcribed with no correct content at all.

🔴 **It is NOT distance, and the data says so directly: `bt3m` is WORSE than `bt10m`.** A distance
explanation requires the opposite. Whatever separates the two groups is a property of the capture
path, not of how far away the speaker was.

### What is NOT known, and was nearly asserted

The label -> device mapping is published nowhere. The paper (arXiv 2509.18722) says only "nine
independent single-channel devices spanning six microphone types at distances from 0.12 m to 10 m";
neither it nor the dataset card says what `bt3m` or `jbl` IS. **The first version of this arm
printed "bluetooth speaker at 3 m" as a description** -- that was read off the label string and
presented as fact. Corrected to flag every reading as a guess. The grouping is the finding; the
cause is not yet known.

### Three traps this arm had to avoid

* **The split opens with scripted greetings** (`สวัสดีครับ M ศูนย์สามครับ`) -- a calibration
  preamble. Sampling from the top would have measured read speech and labelled it spontaneous.
* **Median duration is 1.69 s** against Common Voice's ~6 s, where CER is mostly edge effects.
  `--min-dur 3.0` leaves 488 candidates per device; the residual mismatch (3.8 s vs 6.0 s) is
  printed rather than hidden.
* **The same utterance ids go to every device.** Sampling each independently would have put
  different speech behind each row and destroyed the control silently.

⚠️ **Scope.** One model, 150 utterances, LOTUSDIS is 3-party meeting speech and not dyadic
telephone (`A21`), and duration is still unmatched. This is the direction and rough size, not a
benchmark result.

### Why it matters more than it looks

The product is a **call-centre** loop: the caller is on a phone, which is a codec and a channel, not
a lavalier. Every accuracy number this repo has produced -- 0.0518, 0.1125, the whole D1-D6 ladder --
was measured on clean-channel audio. **The largest effect measured on this box all day is the one
nobody was working on.**

---

## 2026-09-17 (later) — the tau sweep: Typhoon's default is the knee, and the headroom above it is real

**`tau` had never been changed on this box.** `-1.5` came from Typhoon's README and every D6 number
in this repo used it. Swept `-0.5 / -1.0 / -2.0 / -2.5`, each paired with `dis@50` at the same tau,
because `F8d` means an unpaired row is an artefact rather than a weak result. One window, eleven
runs, a closing control.

✅ **The window is certified.** The closing control (`bias@50`, `tau -1.5`, run last) reproduced the
opening run byte-exactly -- recall 0.55353, CER 0.11162, U-CER 0.10351, B-CER 0.27493 -- forty
minutes and eleven model loads apart. Without that the table below would be unreadable.

| tau | recall bias | recall dis | delta | U-CER bias | CER bias | B-CER bias |
|---|---|---|---|---|---|---|
| -0.5 | 0.3964 | 0.3280 | +30 | 0.10292 | 0.11274 | 0.32240 |
| -1.0 | 0.4806 | 0.3280 | +67 | 0.10280 | 0.11154 | 0.29239 |
| **-1.5** | **0.5535** | **0.3280** | **+99** | **0.10351** | **0.11162** | **0.27493** |
| -2.0 | 0.6241 | 0.3075 | +139 | 0.10642 | 0.11686 | 0.32162 |
| -2.5 | 0.6560 | 0.2870 | +162 | 0.11545 | 0.13158 | 0.44264 |

### The finding is in the decoy column, not the recall column

**Recall rises monotonically as tau loosens, and so does harm.** Read across the `dis@50` column
instead:

| tau | dis@50 recall | what a list with NO gold in it does |
|---|---|---|
| -0.5, -1.0, -1.5 | **144/439**, three times, bit-identical | nothing. The decoys are provably inert |
| -2.0 | 135/439 | **destroys 9 mentions the model had right unaided** |
| -2.5 | 126/439 | **destroys 18**, and B-CER reaches 0.683 |

🔴 **`-1.5` is the largest |tau| at which pure decoys are inert.** Past it the spotter starts
finding phrases that are not there and splicing them over correct text. Typhoon's default is not
an arbitrary constant -- it is the knee, and this is the first measurement on this box that shows
why.

⭐ **It also passes the over-biasing gate and nothing looser does.** At `-1.5` the biased arm's
U-CER is **0.10351**, *below* the 0.10571 unbiased baseline. At `-2.0` it is 0.10642 and at `-2.5`
0.11545 -- recall bought with harm, which is exactly what `H21`'s gate exists to refuse.

### What it means for the learned-fusion question

The spec's open question 6 asked whether sweeping `tau` leaves anything for a learned rule. The
answer is better than either outcome expected:

- **A better CONSTANT does not exist.** `-1.5` is already the best fixed threshold under the gate.
- **But there are ~45 more gold mentions between `-1.5` and `-2.5`** (243 -> 288), and they are
  real gold, not decoy noise -- the paired delta grows the whole way.
- **They are unreachable by any constant**, because the same loosening that admits them admits the
  fabrications.

⭐ **So this ARGUES FOR a learned, input-dependent threshold rather than against one** -- take the
loose threshold only where the evidence supports it. That is a sharper motivation than the spec had
before, and it is measured.

⚠️ **Scope.** One model, one bias-list size (50), `dis@50` as the control, 682 real utts, align dp.
`N=500` is untested for this arm, and the tau/list-size interaction is unmeasured.

---

## 2026-09-17 — the move is verified, and the control found a fifth harness defect

**Why anything ran.** The repo was made self-contained on 2026-09-16 -- `adapter/`, `asr/` and
`colab/` moved in, output paths were rewritten, `ADAPTER_OUT` was demoted from requirement to
override. Nothing about the models or the audio changed, so every arm had to reproduce exactly.
That is this repo's own rule -- *run the control even when nothing changed* -- applied to a
reorganisation rather than to an experiment.

### It did not run at all the first time

`arm1_baseline.py` had `REFS = REPO / "smartturn" / "refs"`, and `smartturn/` stayed in the
voice-loop repo. Before the move I had verified that `adapter/` imports nothing from that repo and
that nothing there imports it. Both true, **and insufficient: that check saw CODE dependencies and
not DATA ones.** The 150 Common Voice clips every baseline is scored against were a path, not an
import, so no grep over imports could have found them.

Fixed by **copying, not moving** -- the endpointer line uses the same clips as VoxCPM cloning
references, so a move would have broken that line to fix this one. They are immutable input, never
edited, so the copies cannot drift. Now `data/refs/`, 151 files, 83 MB.

### Then both halves of arm 1 reproduced exactly

| | recorded | 2026-09-17 |
|---|---|---|
| general CER, fold | 0.0518 / 0.0000 / p90 0.1667, 85 perfect | **identical**, 16/16 metrics, 150/150 transcripts byte-identical |
| rare-word | CER 0.1125, U-CER 0.1057, B-CER 0.2618, recall 0.3280 | **identical**, 24/24 metrics incl. CIs, 682/682 hypotheses byte-identical |

RTF 0.009 and 0.007. `results/` worked as the default output with no environment variable.

### 🔴 THE FIFTH HARNESS DEFECT: `compare_baselines.py` never pinned the alignment backend

Run from the documented ASR venv, the seven-system comparison table came back with **CER and
entity recall identical for every system, and U-CER and B-CER different for every system.** Ours
read U-CER 0.1062 / B-CER 0.2505 where the record says 0.10571 / 0.26178 -- an exact match for the
**kaldialign** figures recorded under defect 2 on 2026-09-12.

The cause: `compare_baselines.py:74` called the benchmark's `scorer/score.py` **directly**, the one
hole `adapter/_score.py` exists to close -- while `arm1_rareword.py` carries a comment saying
*"Through `_score.py`, never score.py directly."* The recorded table was produced from the VoxCPM
venv, which has no `kaldialign`, so it is on the DP path and internally consistent. The documented
command in the documented venv silently produced a different one.

⚠️ **This is the 2026-09-12 defect recurring in a script that was never audited for it.** Finding a
defect does not fix the sibling that shares its shape.

**Fixed:** routed through `_score.py`, `--align` added and defaulting to `dp`, and the backend is
now printed with every table. Re-run pinned, all seven systems reproduce exactly, B/U ratio range
2.01-3.04 as recorded.

### Two stale labels, both from before `F8c`

`arm1_baseline.py` printed `:8007, CPU` two lines below a health check reporting `device: cuda`,
and `compare_baselines.py` labelled our row `~115M RNNT, CPU`. Both now read the device from
`/health` or state GPU explicitly -- a label that cannot go stale beats one that happens to be
right today.

### What this leaves

Verified after the move: both arm-1 halves, `compare_baselines`, the `results/` default, and
`_score.py`'s pinning. Still unrun since the move: `arm_d4_boosting` and `arm_d6_splice`.
🔴 **Nothing has been trained and no synthetic audio has been generated** -- that remains true.

These entries were **copied**, not moved, out of `E:\Claude\visualllm-business\UPDATES.md` when the
ASR research split into its own repo on 2026-09-15. That file is append-only by its own rule, so
nothing was deleted from it; it remains the complete history of the voice loop, and this file is
the ASR-relevant subset of it. Entries about TTS, LLM serving, the loop's latency and the live
panels were left behind and are deliberately not duplicated here.

New ASR work gets its entry **here**, at the top, under a `## YYYY-MM-DD — one-line title`.

---

## 2026-09-15 — the repo is reduced to evidence: the plans come out

**His call, same day as the split:** *"remove the plan from it, just data needed."* This repo now
holds what was **found and measured**, not what was going to be done next.

**Removed** (all recoverable — `git show 4734fc5:<path>`, and `git show 4734fc5 --stat` lists them):

- `RESEARCH-PLAN.md` — the arms in order, the architecture, §4 on what may and may not be claimed.
- `HANDOFF.md` — the 2026-09-12 ASR handoff. Its §7i and the earlier `▶ NEXT` blocks were the
  roadmap; its measurements are duplicated in `RESEARCH-PROGRESS.md` and `RESEARCH-FINDINGS.md`,
  which is why it could go without losing a number.
- `docs/specs/` (5) and `docs/plans/` (2) — the adapter design, the accuracy plan, the project
  direction, the two endpointer designs, the two endpointer plans.
- `RESEARCH-PROGRESS.md` §5, the pending to-do table (§6 "Known traps carried forward" survives as
  §5 — traps are evidence, not plan).
- `LITERATURE.md` §3, the reading queue.

**Kept, deliberately:** `RESEARCH-FINDINGS.md` `F12` ("four more that are immediately actionable")
and `F14` ("the adversary arm is cut"). Both read as *next actions* but are findings and a decision
record respectively — removing them would lose why an arm was dropped, which is the kind of thing
this repo exists to remember.

**Repointed, so nothing dangles:** `HARNESS.md` and the endpointing-instrument note now cite the
removed specs by their git ref rather than a path that no longer resolves. In the loop repo,
`adapter/make_text_banks.py`, `adapter/arm_d6_splice.py` and `colab/README.md` were repointed the
same way — `arm_d6_splice.py`'s control reference moved from `RESEARCH-PLAN.md` §1 to
`RESEARCH-FINDINGS.md` `F8d`, which is where that measurement actually lives.

🔴 **Don't re-add a roadmap here.** Planning belongs where the work happens, in
`E:\Claude\visualllm-business\`. ⚠️ Nothing was measured or re-derived; this was a deletion.

## 2026-09-12 (night, cont.) — D6: the splice works, and it is 12× D4

`adapter/arm_d6_splice.py`. Typhoon's released CTC spotter finds the entities; **our** RNN-T writes
the transcript; the spots are spliced into it by frame overlap. All numbers on the 682 real utts,
align `dp`, tau −1.5 (their recommended default, not tuned by us).

| system | recall | gold | CER | U-CER | B-CER | RTF |
|---|---|---|---|---|---|---|
| RNN-T, no list (**D0**) | 0.32802 | 144/439 | 0.11248 | 0.10571 | 0.26178 | 0.028 |
| RNN-T + boosting tree (**D4**) | 0.35080 | 154/439 | 0.11270 | 0.10103 | 0.37307 | 0.021 |
| CTC alone, no list | 0.15718 | 69/439 | 0.14440 | 0.13465 | 0.33421 | 0.005 |
| CTC + spot, **dis@50** (control) | 0.15490 | 68/439 | 0.14617 | 0.13108 | 0.44018 | 0.019 |
| CTC + spot, **bias@50** (theirs) | 0.48519 | 213/439 | 0.13911 | 0.12917 | 0.33015 | 0.019 |
| RNN-T + spot, **dis@50** (control) | 0.32802 | 144/439 | 0.11500 | 0.10450 | 0.34710 | 0.032 |
| **RNN-T + spot, bias@50 (OURS)** | **0.55353** | **243/439** | **0.11162** | **0.10351** | **0.27493** | 0.032 |

### ⭐⭐ The headline, on the comparison the corrected gate demands

**`bias@50` vs `dis@50` — same list size, same audio, same everything:**

| | dis@50 (50 decoys) | bias@50 (50 incl. gold) | Δ |
|---|---|---|---|
| entity recall | 0.32802 (144) | **0.55353 (243)** | **+99 mentions, +69% rel** |
| U-CER (harm) | 0.10450 | **0.10351** | **down** |
| CER | 0.11500 | **0.11162** | down |
| B-CER | 0.34710 | **0.27493** | down |

**Every number moves the right way at once.** This is the first arm where the harm term falls *and*
recall leaps, on the equal-list-size comparison rather than against `none` — which matters because
`dis@50` lands on **exactly** the unbiased recall (0.32802, 144/439, bit-identical to D0), so 50
pure decoys are provably inert and the entire gain is the gold entries.

### ✅ Independent validation: their published baseline reproduces on our harness

`A26` reports entity recall **.158** for `typhoon-asr-realtime-nemo-ctc` unbiased. Running their
model, their spotter, their splice, on our 682: **0.15718 (69/439)**. That is their number to three
decimal places, from a harness they have never seen — so when this harness then says something
different from their card, the difference is the setup, not a bug here.

⚠️ Their **boosted** figure is .852; ours is **.485** for the same pipeline. Not a contradiction to
chase blindly: they score all 2,062 utterances including the 1,380 TTS ones, and we deliberately use
only the 682 real. Synthetic speech is cleaner and its entities were generated from the same list, so
a much higher boosted recall there is exactly what `F5` predicts. **The honest reading is that .852
is not reachable on real audio**, and the number to quote for real Thai speech is ours.

### ⭐ The contribution: keep the better transcript, take only the spots

Their pipeline splices into the **CTC greedy** transcript, which costs accuracy everywhere else —
CTC alone reads CER 0.14440 against the RNN-T's 0.11248. Splicing the *same spots* into the RNN-T
transcript instead wins on every axis at once:

| | theirs (CTC + spot) | ours (RNN-T + spot) |
|---|---|---|
| recall | 0.48519 | **0.55353** |
| CER | 0.13911 | **0.11162** |
| U-CER | 0.12917 | **0.10351** |
| B-CER | 0.33015 | **0.27493** |

**+30 more gold mentions and −20% CER at the same time**, because the two models share one frozen
encoder: the frames line up (both ×8 subsampling, 0.08 s), the tokenizers are byte-identical, and the
CTC head's only job becomes *where*, never *what*. Cost: RTF 0.032 vs 0.028 — **a second model for
~14% more compute**, because the encoder work is the same shape and the spotter is 1–2 ms per keyword.

### Versus everything else tried today

| arm | mechanism | Δ recall vs baseline |
|---|---|---|
| D1 beam / D2 n-gram / D3 beam+LM | re-rank the model's own candidates | **−1 mention** |
| D4 boosting tree | bias the model's own scores toward listed phrases | +8 (same-size cmp) |
| **D6 splice** | **a second model's opinion of the audio** | **+99** |

The session's thesis, now measured end to end: **re-weighting one model has a low ceiling; a second
acoustic opinion does not.** D4 predicted this and D6 confirms it at 12× the effect.

### ⚠️ Three honest limits, all measured

1. 🔴 **25% of the gold mentions are unspottable by construction.** 108 of 439 gold `span_text` are
   spelled in **Latin**, and the spotter scores those at **−16 to −20** — a Thai speaker says the Thai
   pronunciation, and a Latin string has no acoustic path through a Thai-character vocabulary. Thai
   entities score −0.64 to 0.0 against decoys at −2.4 to −4.8, cleanly separated at their tau of −1.5.
   **So the 0.554 is achieved while a quarter of the target set is invisible**, and transliterating
   Latin entries is the obvious next arm. ⚠️ It must be applied to distractors too — only *gold*
   entities carry `form_th` in the annotations, so expanding from those alone is an oracle leak.
2. `tau` was left at their recommended −1.5 and **not tuned by us**. It is the obvious free knob, and
   it should be swept on the synth split, never on these 682.
3. Only `bias@50` / `dis@50` were run. The 500-entry ladder is the interesting stress case and is
   untested for this arm.

### Loading defect, worked around without touching their code

Their `KeywordSpotter` calls `ASRModel.restore_from`, which is abstract in NeMo 3.0.0 *and* chokes on
this NeMo-2.5.0 checkpoint with `ConfigAttributeError: Missing key key_phrase_items_list` —
`BoostingTreeModelConfig` grew a field after the checkpoint was exported. Fixed by restoring the
concrete `EncDecCTCModelBPE` against a generated `model_config.patched.yaml`, leaving
`kws_spot.py` / `word_boost.py` byte-identical to what Typhoon released so they stay quotable.

All numbers: `adapter/_out/d6_splice_summary.json`.

---

## 2026-09-12 (night) — D4: biasing WORKS, it is the first arm to move recall, and it exposed a third scorer defect

`adapter/arm_d4_boosting.py`. NeMo's `GPUBoostingTreeModel`, one tree rebuilt per utterance from the
benchmark's own frozen bias list (3–8 ms per tree, negligible). **Hyperparameters selected on the
synth split, reported once on real** — the discipline `A38` asks for, and the reason the synth audio
(1,380 clips, ~630 MB) was pulled tonight rather than spending the 682.

### ✅ The first arm that moves entity recall

Held-out, 682 real utts, **α 1.0 / context_score 0.5** selected on dev:

| condition | recall | gold found | U-CER | CER | fires | RTF |
|---|---|---|---|---|---|---|
| none (control) | 0.32802 | 144/439 | 0.10571 | 0.11248 | — | 0.028 |
| **bias@0** (gold only) | **0.36219** | **159/439** | 0.10196 | 0.11124 | — | 0.018 |
| bias@50 | 0.35080 | 154/439 | 0.10103 | 0.11270 | 0.0003 | 0.021 |
| bias@500 | 0.34169 | 150/439 | 0.09725 | 0.11426 | 0.0002 | 0.031 |
| dis@500 (**no gold in list**) | 0.32346 | 142/439 | 0.09830 | 0.11500 | 0.0002 | 0.031 |

**+15 gold mentions with a perfect list (+10.4% relative), decaying to +6 at 500 distractors**, with
general CER flat and a distractor fire rate of **0.0003** — it does not hallucinate listed entities.
After three arms that moved recall 144 → 143 → 143, this one moves it.

⚠️ **But it is far from the published effect.** Typhoon's own `nemo-ctc` reports **.158 → .852**.
Ours goes .328 → .362. Their baseline is much worse than ours and their spotter is a different
mechanism (CTC frame posteriors), so it is not a like-for-like gap — but a 5× and a 1.1× are not the
same result, and the splice (D6) is the arm that actually reproduces their setup.

### The frontier, and why we sat at the cautious end

Dev (400 synth utts, seeded sample, bias@50). Recall is buyable — the question is what it costs:

| α:score | U-CER | vs none | recall | vs none |
|---|---|---|---|---|
| none | 0.02866 | — | 0.29204 | — |
| 0.25:1 | 0.02774 | −3% | 0.31121 | +6.6% |
| 0.5:1 | 0.02841 | −1% | 0.32596 | +11.6% |
| **1:0.5 (selected)** | **0.02857** | **−0.3%** | **0.33628** | **+15.1%** |
| 1:0.75 | 0.03063 | +7% | 0.34808 | +19.2% |
| 1:1 | 0.03360 | +17% | 0.36136 | +23.7% |
| 1:1.5 | 0.03949 | +38% | 0.39528 | +35.4% |
| 1:3 | 0.10076 | +252% | 0.44100 | +51.0% |
| 1:5 | 0.37996 | +1226% | 0.45133 | +54.5% |

**1:0.5 is the last point where the harm term does not rise.** Past it every recall point is bought,
and by α1/s5 the decoder emits little but entity names — verified by eye on a single clip, which
came back as `บางกระปิ บางกระปิ บางกระปิ … Khlong Toei Khlong Toei`. `H21`'s over-biasing gate is not
theoretical on this model; it is the dominant feature of the curve.

### 🔴🔴 DEFECT 3 — B-CER and U-CER are not comparable across bias-list SIZES

`dis@500` is the condition that gives it away: its list contains **no gold entities at all**, yet
B-CER goes **0.26178 → 0.49588**, nearly doubling. The counts say why:

| condition | total insertions | charged to **B** | charged to U |
|---|---|---|---|
| none | 3036 | **0** | 3036 |
| bias@0 | 2992 | 208 | 2784 |
| bias@50 | 2933 | 364 | 2569 |
| bias@500 | 3024 | 703 | 2321 |
| **dis@500 (no gold)** | 2938 | **652** | 2286 |

**Total insertions are flat.** They are only being *re-attributed*. The scorer's rule, in its own
words, is that an inserted run belongs to B *"iff its string matches (⊆ or ⊇, len≥2) any form of any
entity in the **provided bias list** (gold + distractors)"* — a character-level analogue of Le et
al.'s word-level "inserted word is in the bias list". Measured on `dis@500`: **650 of 3,222 inserted
characters (20.2%) are charged to B, in 154 runs, of which 126 (82%) are two or three characters
long.**

🔴 **In Thai this rule is much looser than the word-level original it copies.** There are no word
boundaries, the unit is a character, and a 2-character fragment is a substring of *something* in a
500-entry list nearly always. So:

- **B-CER inflates with list size**, independent of the model.
- **U-CER deflates with list size**, for the same reason — which is why it "improved" in every biased
  condition above. That improvement is not real.
- 🔴 **The plan's own gate — "recall up, U-CER must not rise" — is therefore UNSAFE as written.** An
  arm can pass it by growing its bias list and changing nothing else. The gate only holds
  **between conditions with the same list size**, which is how the numbers above must be read: the
  D4 comparison that means something is `bias@N` vs `dis@N` at equal N, not against `none`.

That comparison, at N=500: recall **0.34169 vs 0.32346** — the gold list is worth **+8 mentions** over
a same-sized list of pure distractors, on identical audio with identical attribution. **That is the
honest D4 effect**, and it is smaller than the +15 the `none` comparison suggests.

### Two more NeMo defects, both worked around

1. 🔴 **`GPUBoostingTreeModel`'s CUDA path is Triton-only with no availability check.**
   `advance()` dispatches to `_advance_triton` whenever `use_triton and device == cuda`, and Triton
   has no Windows wheel, so it raises `ModuleNotFoundError` **mid-decode**. Its sibling
   `NGramGPULanguageModel` checks and falls back with a warning; this one does not. Forcing
   `use_triton = False` selects the shipped `_advance_pytorch` path and it works.
2. 🔴 **Hot-swapping `computer.fusion_models[i]` silently does nothing** — the computer caches its
   fusion models at construction. Verified: swapping in a gold-entity tree produced byte-identical
   output. `change_decoding_strategy` per utterance is the only path that takes effect.

### ✅ And a fourth thing, found by the control: the PCM16 wire changes 3.4% of transcripts

The in-process harness first reproduced D0 to 4 decimal places but **not exactly** — 23 of 682
hypotheses differed. It was neither CUDA graphs nor the placeholder tree (all three in-process
variants agreed with each other and none with the server). It is the **16-bit wire format**: every
other arm reaches the model as PCM16 over HTTP, and feeding `soundfile`'s float32 straight in changes
those 23 transcripts — confirmed **23/23** by replaying exactly those clips through the round-trip.
Aggregate CER does not move (0.11248 vs 0.11249) and recall is identical, so it is invisible in a
metrics table and visible only per-utterance. `arm_d4_boosting.py` now quantises, and the control
reproduces D0 **exactly**: 0.11248 / 0.10571 / 0.26178 / 0.32802.

All numbers: `adapter/_out/d4_boosting_summary.json`.

---

## 2026-09-12 (late, cont.) — D2 and D3: the n-gram LM is real, entity-blind, and too slow here

Continues the entry below. Typhoon's 4-gram is downloaded, built, and measured on both halves.
**A31's headline did not reproduce; A35's did, at a price the loop cannot pay.**

### The LM, and what it cost to get

`ngram_4gram_bpe2048.arpa` — 2,045 unigrams / 1.96 M bigrams / 30.9 M trigrams / **71.9 M 4-grams**,
104.8 M total. `adapter/build_ngram.py` re-verifies the tokenizer hash against the *served*
checkpoint (refuses to build on a mismatch), converts once, and prints memory either side.

🔴 **The build ran the box to the edge, exactly where `CLAUDE.md` says it would:** 17 min, peak
~11 GB private bytes, ending at **RAM free 0.7 GB / commit free 0.9 GB** from 12.9 / 9.2. The
page file is still the unfixed 7.76 GB. Output: `ngram_4gram_bpe2048.nemo`, 2.23 GiB — load that,
never the ARPA, or every server boot pays the 17 minutes again.

🔴 **NeMo warns at build: *"Triton is disabled. Version without Triton is not compatible with Cuda
graphs; decoding can be slow."*** Triton has no Windows wheel. This is not cosmetic — see below.

### 🔴 D2 — n-gram inside GREEDY. `A31`'s claim does not reproduce, on either axis.

`A31` (NGPU-LM) promises **>50% of the beam gap at <7% overhead**. Measured, 150 Common Voice clips:

| α | CER (fold) | vs greedy | RTF |
|---|---|---|---|
| — (greedy, control ×3) | 0.0518 | — | 0.010 / 0.011 / 0.012 |
| 0.15 | 0.0522 | +0.8% | 0.030 |
| **0.30 (best)** | **0.0516** | **−0.3%** | 0.030 |
| 0.50 | 0.0546 | +5.4% | 0.030 |

**It closed ~0% of the beam gap** (beam alone buys −8.9%), and the overhead is **2.6×, not 7%**.
⚠️ The latency half has an honest excuse — no Triton, so no CUDA graphs on the fused path. **The
accuracy half does not.** Re-open on Linux; do not carry `A31`'s number forward as if it were ours.

### ⚠️ D3 — beam + n-gram. Best CER of any arm, and it still does nothing for entities.

α swept on the 150 clips as **dev**, reported once on the 682-utt benchmark as **held-out** — the
discipline `A38` asks for, and the fragility it warns about showed up:

| α (dev) | 0.00 | 0.05 | **0.10** | 0.15 | 0.30 |
|---|---|---|---|---|---|
| CER | 0.0472 | 0.0489 | **0.0463** | 0.0463 | 0.0490 |
| perfect | 86 | 87 | **90** | 90 | 87 |

Non-monotonic — 0.05 is worse than both neighbours. ⚠️ And **the LM raises the worst clip** while
fixing several small ones: max CER 0.5161 → 0.6452.

**Held-out, `thai-contextasr-bench` 682 real utts, condition `none`, align `dp`:**

| arm | CER | U-CER | B-CER | recall | RTF |
|---|---|---|---|---|---|
| D0 greedy | 0.11248 | 0.10571 | 0.26178 | 0.32802 (144/439) | 0.008 |
| **D1 beam 2** | 0.10995 | 0.10335 | **0.25574** | 0.32574 (143/439) | **0.013** |
| D3 beam 2 + 4-gram α 0.10 | **0.10814** | **0.10146** | 0.25705 | 0.32574 (143/439) | **0.136** |

🔴🔴 **THE RESULT THAT MATTERS: the LM improves ordinary text and does NOTHING for named entities.**
U-CER is the session's best (−4.0% on D0) while **B-CER is *worse* than beam alone** and recall is
**bit-identical to D1's, 143 of 439**. A 4-gram trained partly on code-switched entity templates
still cannot recover entities, because recall is bounded by whether the acoustic model emits the
right subwords at all — which is the whole argument for D6, the two-model splice, and against
spending more on LM fusion.

### ✅ The RTF blow-up is attributed, with a control

An 11× slowdown needs a control before it is blamed on the LM. Beam 2, **no LM, CUDA graphs forced
off**: CER **0.0472 — identical** to graphs-on (so graphs are pure execution, no accuracy effect),
RTF **0.0561**.

| | RTF | ×D0 |
|---|---|---|
| beam 2, graphs on | 0.0157 | 1.3× |
| beam 2, graphs **off** | 0.0561 | 4.7× |
| beam 2 + LM (graphs off, forced) | 0.1294 | 11× |

**So losing CUDA graphs costs 3.6× and the LM's own arithmetic costs 2.3×.** Only the second is
intrinsic; the first is a Windows/Triton artefact. `asr/server.py` now turns graphs off
automatically whenever a fusion model is loaded (`malsd_batch` + LM otherwise dies inside capture
with `RuntimeError: Capture must end on the same stream it began on`) and exposes `ASR_CUDA_GRAPHS`
to force them back once Triton exists.

### Verdict

- ✅ **D1 (beam 2) is the adoptable win** — −8.9% dev CER, −2.3% held-out, +4 ms on a 6 s utterance.
- ❌ **D2 (n-gram in greedy) is rejected** — noise, at 2.6× RTF.
- ⏸️ **D3 is parked, not rejected** — genuinely the best CER and U-CER, but **RTF 0.136 is ~816 ms
  on a 6 s utterance** in a loop chasing 500 ms voice-to-voice, and it buys nothing on the metric
  the plan targets. Re-cost it on Triton, or keep it for an offline/batch transcription path where
  latency is not the constraint.
- ▶ **The entity gap survives every decode-time arm tried.** Recall moved 144 → 143 → 143. That is
  the argument for D4 (the boosting tree) and D6 (the splice) being the only arms left that target it.

All arms: `adapter/_out/decoding_arms_summary.json`.

---

## 2026-09-12 (late) — D1 is measured, and the harness had TWO silent defects under it

**Ran: D1 (batched beam search), a beam-width sweep, and the rare-word half — all bracketed in one
window with a greedy control at each end.** Two things broke before a number could be trusted, and
both had been quietly producing the numbers this repo already published.

### 🔴 DEFECT 1 — `:8007` has been running on the GPU since it was written. Every "CPU" claim is wrong.

`asr/server.py`'s docstring says *"on the CPU"* and *"The CPU is not a compromise, it is the point"*.
It called `ASRModel.from_pretrained(MODEL_NAME)` with **no `map_location`**, and NeMo defaults that to
`cuda:0` when torch sees a card. Measured directly: `next(m.parameters()).device` → **`cuda:0`**.

Proof it was always so, not a change: forcing `map_location="cuda"` reproduces the recorded arm-1
baseline **byte-identically** — 150 of 150 hypotheses, CER 0.0723 / 0.0706 / **0.0518**, 55/57/85
perfect. Forcing `cpu` changes **one** hypothesis of 150 (`ประโยชน์` → `ประโยค`, which is correct) and
moves mean CER to 0.0513.

**What this invalidates:** every "RTF 0.007 / 0.009 **on CPU**" in `RESEARCH-PROGRESS.md`,
`CLAUDE.md` and `adapter/arm1_*.py` is a **GPU** number, including the line *"ours is ~115M at RTF
0.007 on CPU against ~1.5B Whispers on GPU"* in the seven-system comparison — that was GPU vs GPU.
The 2026-08-17 verdict *"Typhoon on CPU 138–155 ms beats faster-whisper-large-v3-turbo 279–308 ms on
the GPU"* is the same error, one month older.

**What the CPU actually costs, now that it has been measured on purpose:** RTF **0.027 vs 0.010–0.011**
on GPU, i.e. **2.6×**, and ~66 ms → ~162 ms on a 6 s utterance. Still 37× faster than real time.
**And the GPU copy holds ~0.95 GB of VRAM** (5893 MiB with the server up vs 4942 MiB idle, nothing
else of ours loaded) — VRAM that the 2026-09-06 "the box is full at 14.6 of 16.3 GB" measurement
included without knowing what it was. Moving the ASR genuinely to CPU is therefore a **real ~0.95 GB
lever** toward the streaming ASR's 2.9 GB, priced at ~96 ms per turn.

`ASR_DEVICE` is now an explicit env var and `/health` reports the live device, so this cannot be
ambiguous again.

### 🔴 DEFECT 2 — the benchmark's B-CER/U-CER depend on whether an optional package is installed.

`thai-contextasr-bench`'s `metrics.py:15` says *"`kaldialign` is used if importable (**faster**), else
a deterministic O(nm) DP"*. "faster" implies agreement. **They disagree.** Same scorer, **one
byte-identical hypothesis file**, three repeats each (the scorer itself is fully deterministic —
seeded bootstrap, `seed=2026`):

| align backend | CER | U-CER | B-CER | recall |
|---|---|---|---|---|
| DP (no kaldialign) | 0.11248 | 0.10571 | **0.26178** | 0.32802 |
| kaldialign | 0.11248 | 0.10622 | **0.25050** | 0.32802 |
| | same | +0.0005 | **−0.0113 (4.3% rel)** | same |

CER and recall need no span attribution and do not move. B-CER and U-CER attribute each edit to
inside/outside a gold-entity span **via the alignment**, and the two implementations break ties
differently. `E:/asr-spike/.venv` has kaldialign (since 2026-08-17); `E:/voxcpm-spike/.venv` does not.
Everything published so far was scored by the VoxCPM interpreter, so **the arm-1 row and all seven
comparison rows are on the DP path and mutually consistent** — but running a later arm from the ASR
venv would have shown a 0.011 B-CER "win" that is pure tie-breaking, on the exact metric the
experiment is gated on.

**Fixed:** `adapter/_score.py` pins the backend by poisoning `sys.modules`, defaults to `dp` (what
every existing number used), verifies the branch it actually got, and refuses to run if it got the
other one. `arm1_rareword.py` now scores only through it and records `align_backend` in a sidecar.

### ✅ D1 — batched beam search. Everything improves, recall flat, RTF +50%.

`strategy: malsd_batch`, one config value, no training, no new data. 🔴 It first read **CER 1.0000 on
all 150 clips** — the checkpoint ships `beam.return_best_hypothesis: false`, so a beam decoder hands
back the whole N-best **list** and `_transcribe` read `.text` off a list and got `""`. Nothing errored.
That reads exactly like "beam search destroys Thai". Now forced true, and an empty transcript logs.

**General CER** — Common Voice 150, SARA AM folded, GPU, one window:

| arm | mean | median | p90 | max | perfect | RTF |
|---|---|---|---|---|---|---|
| D0 greedy_batch (control, start) | 0.0518 | 0.0000 | 0.1667 | 0.8000 | 85 | 0.011 |
| D0 greedy_batch (control, end) | **0.0518** | 0.0000 | 0.1667 | 0.8000 | 85 | 0.010 |
| **D1 malsd_batch, beam 2** | **0.0472** | 0.0000 | 0.1667 | **0.5161** | 86 | 0.016 |
| D1 malsd_batch, beam 4 | 0.0478 | 0.0000 | 0.1562 | 0.5323 | 85 | 0.016 |
| D1 malsd_batch, beam 8 | 0.0476 | 0.0000 | 0.1562 | 0.5323 | 85 | 0.016 |
| maes_batch, beam 2 | 0.0494 | 0.0000 | 0.1667 | 0.5323 | 85 | **0.079** |

**−8.9% relative at beam 2**, and the control reproduced **exactly** at both ends of the window, so
this is not drift. ⭐ **Beam width buys nothing past 2** (0.0472 / 0.0478 / 0.0476 is noise) and costs
nothing either — RTF is 0.016 at every width, because batch-1 decoding on a GPU is launch-bound, not
compute-bound. **`maes_batch` is worse on both axes**; `malsd_batch` is the right beam.
⚠️ The single largest change is the **worst** clip: max CER 0.80 → 0.52. Greedy's catastrophic
failures are what beam fixes, which is also why the median (already 0.0000) cannot move.

**Rare-word half** — `thai-contextasr-bench`, 682 real utts, condition `none`, **align `dp` both
sides**:

| | D0 greedy | D1 malsd b2 | Δ rel |
|---|---|---|---|
| CER | 0.11248 | **0.10995** | −2.3% |
| **U-CER** (the harm term) | 0.10571 | **0.10335** | **−2.3%, it goes DOWN** |
| B-CER | 0.26178 | 0.25574 | −2.3% |
| entity recall | 0.32802 (144/439) | 0.32574 (143/439) | −0.7%, one mention |
| RTF | 0.008 | 0.013 | +63% |

**The gate is passed in the direction nobody has to argue about:** the harm term does not rise, it
falls. Recall is flat — beam search is not a rare-word method, and this is the measurement that says
so rather than an assumption. 🔴 The gain is uniform (−2.3% on all three CERs), which is the signature
of a *decoder* improvement, not an entity one.

### ✅ D2's blocker cleared for 350 KB instead of 462 MB

The plan said hash-check the tokenizer before trusting Typhoon's shipped 4-gram. Two things found:

1. **The 4-gram is not in the model repo.** `typhoon-asr-streaming-115m` ships only a `.nemo`. The LM
   lives in a **separate** repo, `typhoon-asr-streaming-115m-ngram` — `ngram_4gram_bpe2048.arpa`,
   **2.7 GB**, CC-BY-4.0, KenLM 4-gram over the 2048 Thai BPE vocab, and its card says
   **"Tokenizer-locked: it works only with the matching model's tokenizer."** Found by enumerating the
   org (82 models), which is this repo's own standing lesson.
2. ✅ **The lock is satisfied.** A `.nemo` is an uncompressed tar, so **HTTP range requests** read its
   header index and pulled just the 280,149-byte tokenizer out of the middle of a 462 MB file:
   `typhoon-asr-streaming-115m`, `typhoon-asr-realtime` (ours, from the served checkpoint) and
   `typhoon-asr-realtime-nemo-ctc` are all **md5 `88e04338795f42f0492c953e461ac785`, byte-identical**.

⚠️ The paired 0.6B LM (`ngram_4gram_agg.nemo`, 2.4 GB) is over a **15,135** aggregate vocab and is
**not** ours — and its card explains why it ships as a binary: that vocab *"corrupts the ARPA
`chr()` token encoding"*. Only the bpe2048 ARPA is usable here.

---

## 2026-09-12 (night) — BOTH directions are crowded, and the ledger had under-recorded why

**A direction document exists: `specs/2026-09-12-project-direction.md`.** Prompted by "right now our
direction is blur". Two more Consensus sweeps (Thai rare-word ASR; turn-taking 2023+). Nothing built.

🔴🔴 **THE CORRECTION THAT MATTERS MOST.** `E2` (Popit et al. 2025) was in the ledger as "a 0.6B
classifier does Thai end-of-turn at F1 0.881". Its abstract actually claims *"the **first** systematic
study of Thai text-only end-of-turn detection"* using ***"Thai-specific linguistic cues (e.g.
sentence-final particles)"***. **So the Thai sentence-final-particle end-of-turn angle is published
and claimed — by a Thai group, in 2025.** `HANDOFF.md` has been treating it as this repo's opening for
two sessions. `E2` is now annotated.

⭐ **What is still ours there is narrow but real:** theirs is **text-only over YODAS subtitles**; ours
is **real audio** (13,602 LOTUSDIS decision moments; particle rates 23.3%/26.3% at turn ends vs 8.7%
mid-turn, measured on speech) inside a **latency-instrumented loop**. Claim the audio and the systems
evaluation — never the particle insight.

🔴 **And the turn-taking section needed nothing new**, which is itself the finding: `E41` (JAL-Turn,
2026) already puts turn-taking on a **frozen ASR encoder** with automatic label derivation; `E52`
(Easy Turn) ships a **1,145-hour** dataset; `E12` and `E15` ship more test sets. All were already
rowed. **The "nobody does turn-taking" framing is out of date.**

🔴 **Thai rare-word ASR also has pre-neural history**: Thai named-entity ASR via class-based LMs over
**multiple-sized subword units** (Interspeech 2008) and a hybrid word + **pseudo-morpheme** lexicon
cutting Thai OOV below 1% (EURASIP 2016). Old and weak, but the word-boundary angle is not virgin
ground either.

### So: no architecture gap in either direction, and that is why the direction felt blurry

The direction doc's answer is **not "pick one"** — it is that the five things this repo uniquely holds
are all **measurements**, not architectures:

- ⭐⭐ **M1 — Thai SARA AM silently inflates every Thai CER number.** Two renderings, identical on
  screen, **NFC does not bridge them**; 47 of 150 clips; folding moves mean CER 0.0706 → 0.0518 and
  median 0.0405 → **0.0000**. In no paper from any sweep. **Cross-paper Thai ASR CER comparison is
  therefore not sound**, and that is a methodological result, not a footnote.
- **M2** — a per-stage Thai voice-agent latency breakdown from true end-of-speech; four published
  breakdowns exist and **none includes endpointing**
- **M3/M4** — `typhoon-asr-realtime` on a public Thai rare-word benchmark with the field rescored on
  identical audio, and **B-CER 2.01–3.04× U-CER across all 7 systems**
- **M5** — the endpointer negative result, ⚠️ still missing its control

**Recommended order: the Thai evaluation paper FIRST (≈80% already measured), then a borrowed-and-cited
architecture (frozen encoder + small head) judged on end-to-end latency rather than F1** — because
every architecture claim is measured with the M1 instrument, and publishing one on an unfolded CER
would be publishing an artifact. Stop positioning the bottleneck adapter as the headline; keep
`colab/` as an arm (`H26`/`H27` justify it).

## 2026-09-12 (evening) — the contextual-biasing + synthetic-speech literature, 22 new rows

**Nothing measured. A Consensus sweep that changes which direction the thesis should take**, prompted
by his question "can we use attention instead of n-gram". Two searches, `LITERATURE.md` **`H15`–`H36`**
(the H series went 14 → 36). §2.7 was the thinnest section in the ledger and is now the thickest.

### 🔴 The finding: this family is crowded, and our method is its weakest member

| method | reported gain | row |
|---|---|---|
| synthetic-only rare-word adaptation, **road names and addresses** | **WER 57.6% → 16.5%, no real audio at all** | `H16` |
| TTS for OOV on an **RNN-T** (our architecture), 93 cites | **−57% on OOV utterances**, no degradation | `H19` |
| `SynthASR` — RNN-T, new vocabulary, 68 cites | **>65% relative**, no degradation elsewhere | `H25` |
| zero-shot synthetic-pronunciation trie | **B-WER −43%**, U-CER flat, **no training** | `H18` |
| alternate-spelling prediction, **no TTS needed** | rare-word recall **+34.7%**, OOV **+97.2%** | `H32` |
| attention-over-a-list family | **19–60%** on rare words | `H22` |
| trie + shallow fusion + neural LM, **143 cites** | 19.5% relative | `H33` |
| **this repo's planned adapter (`H3`)** | **6% relative** | — |

🔴🔴 **`H15` is this repo's own 2026-09-12 reframe, published in 2023.** "Train an attention-based
contextual biasing adapter on synthetic audio" — with a projection module and consistency
regularisation to close the real/synthetic gap — Naowarat et al., Interspeech 2023. It was proposed
here as novel in the same session. It is not.

🔴 **`H19` is the third strike on `H2`.** The design's premise is that transducers do not benefit
from synthetic data. `H10` already contradicted it; `H19` is a **93-citation** paper showing an RNN-T
gaining **57%** on exactly our target class, in 2020. Treat the premise as dead, not contested.

🔴 **`H16` says `colab/gen_colab.py` is built the weaker way.** Measured: condition **multiple TTS
systems** on one speaker, rather than one TTS on many speakers — different systems occupy different
speaker-embedding distributions, worth **9.8% relative** at equal data size. We use one TTS (VoxCPM)
across 150 references: the arrangement they measured as worse. A second Thai TTS does not exist here
(F5/JaiTTS archived); VoxCPM voice-design is one system wearing different voices, not two systems.

### ⭐ But two rows from the same sweep RESTORE part of the plan

🔴 **A wrong inference made earlier the same day is corrected.** It was argued here that an n-gram LM
over the same text might make the synthetic-audio pipeline redundant ("arm 0"). Measured otherwise:

- **`H26`** compared text-only adaptation against TTS audio head-on — **text loses, "even when the TTS
  engine is sub-optimal"**, and the authors note this challenges the forgetting/misalignment
  literature that motivated text-only methods.
- **`H27`** (97 cites) compared TTS gains against integrating the same text as an LM: **mostly
  independent — they stack.**

**So a text-side baseline is an arm to ADD, not an alternative to choose between.** `colab/` keeps its
justification.

⭐ **And `H25` retires an open question in the design.** S1 ("the encoder is frozen throughout") was
labelled a deliberate untested deviation from `H3`. `SynthASR` (68 cites) uses **encoder freezing** as
one component of a recipe that teaches an RNN-T a new vocabulary from synthetic speech at **>65%
relative with no degradation**. The freeze is published, on our architecture, for our task shape.

### ⭐⭐ What is actually left: `H24` — none of it is Thai

Every row H15–H23 and H30–H36 is English (LibriSpeech, Earnings21, SPGISpeech, ConEC), Japanese, or
Singapore English (NSC). There is **no Thai contextual-biasing-from-synthetic result**, no Thai
synthetic-to-real exchange rate (`H7`), and the Thai-specific obstacles appear nowhere in any of them:
no word boundaries, **two renderings of SARA AM that no NFC pass reconciles** (measured here the same
day at 26.6% of mean CER), a convention of unspaced number *words*, and Latin↔Thai code-switching in
real entity mentions.

🔴 **So the honest claim in this family is "first Thai", not "new method".** A legitimate paper; a weak
answer to an advisor who rejected "generate synthetic data and fine-tune" and asked for an
architecture change. Written into the design spec §2 so it cannot be forgotten.

⚖️ **Against that, the turn-taking direction is untouched by anyone:** ten Typhoon ASR releases and
none does it, four published voice-agent stage breakdowns and none includes it, the freeze-and-bolt-on
pattern published **three** times now as precedent (`A28`, `H25`, `H30`), and this repo holds the only
Thai sentence-final-particle quantification on real audio. Its blocker is data — `A16` — not
architecture. **The safe route is no longer the safer bet.**

### Four rows that are immediately actionable regardless of direction

- **`H36`** — filter synthetic data, do not use all of it. A similarity scorer kept accuracy at
  **under 30%** of the TTS data; `WAVe` word-alignment filtering cut training steps 34% and cross-domain
  WER 13.54% → 6.89%. ⭐ `smartturn/verify.py` already reads every clip back through Typhoon — these
  rows say that readback should be a **filter**, not a report.
- **`H28`** — pretrain on synthetic **first**, then transfer to real; the reverse order is worse. And
  the dominant error mode is **prosody mismatch**, which is the one thing `voxcpm/_squeeze.py` controls.
- **`H32`** — alternate spellings double OOV recall and need no TTS. `A29`'s benchmark already ships
  `form_th_alt` and `latin_alts` per entity, so the variants exist for free.
- **`H29`** — asks a question nobody here has: **how much Thai was VoxCPM2 trained on?** Synthetic
  speech can only approach real data if the TTS itself saw a large enough corpus. If the Thai share is
  small, that caps what any architecture here can achieve.

## 2026-09-12 (later still) — ARM 1 IS COMPLETE, and the rare-word gap is now a number

**Both halves of §8's gate now exist**, which means later arms can finally be judged.
`adapter/arm1_rareword.py`, on `wayu-ai/thai-contextasr-bench` using **the benchmark's own scorer**,
condition `none`, **682 real-split utterances** (439 gold entity mentions, 379 distinct):

| metric | value | 95% CI |
|---|---|---|
| CER overall | 0.1125 | [0.1022, 0.1239] |
| **U-CER** — outside entity spans, the HARM term | **0.1057** | — |
| **B-CER** — inside entity spans | **0.2618** | [0.2330, 0.2927] |
| **entity recall** | **0.328** | [0.2751, 0.3802] |

682 utts, 1.83 h of audio, **RTF 0.007** on CPU. 144 of 439 gold mentions recovered.

### ⭐ Rescored the benchmark's six published baselines on the SAME 682 utts

Their repo ships baseline hyp files, so this is like-for-like — identical audio, identical scorer,
identical condition:

| model | CER | U-CER | B-CER | recall |
|---|---|---|---|---|
| `biodatlab/whisper-th-large-v3-combined` | 0.1068 | 0.1021 | 0.2054 | **0.435** |
| `typhoon-whisper-large-v3` | 0.0924 | 0.0878 | 0.1829 | 0.408 |
| `qwen3-asr-1.7b` | 0.0860 | 0.0790 | 0.2251 | 0.367 |
| `nectec/Pathumma-whisper-th-large-v3` | 0.0814 | 0.0751 | 0.2062 | 0.351 |
| `typhoon-isan-asr-whisper` | 0.1059 | 0.0992 | 0.2371 | 0.349 |
| **`typhoon-asr-realtime` (ours)** | 0.1125 | 0.1057 | 0.2618 | **0.328** |
| `qwen3-asr-0.6b` | 0.1051 | 0.0966 | 0.2940 | 0.253 |

⭐⭐ **Two findings, and the first one is the thesis.**

1. **B-CER is 2.01–3.04× U-CER for EVERY model tested** (ours 2.48×, the lowest 2.01×). Entity spans
   are universally two to three times harder than the rest of the audio, across RNNT, CTC and
   encoder-decoder architectures alike. **The rare-word gap is field-wide, not a deficiency of the
   model we happen to run.**
2. **The best model in the field still misses 57% of named entities unbiased.** Nobody has solved
   this. Our 0.328 is 6th of 7, but closing to 0.435 only *matches* the field's best — beating it
   would be new.

⚠️ **Not size-matched.** Ours is a ~115M-class RNNT at RTF 0.007 on **CPU**; the Whispers are
~1.5B on GPU. Being 6th of 7 on recall while being the smallest and ~100× cheapest is a different
statement from being worst.

### 🔴 Four constraints on this number — all recorded in the result JSON

1. **Real split only, 682 of 2,062.** The other 1,380 are OmniVoice TTS with cloned voices, and
   this experiment is *about* whether synthetic speech can teach an ASR — evaluating on synthetic
   audio would be circular. S7 says real speech. Not a convenience; a correctness requirement.
2. **Condition `none` only.** `typhoon-asr-realtime` takes audio and returns text; it has no bias
   list input. `none` is also the right condition — unbiased rare-word accuracy is exactly what the
   adapter is supposed to improve.
3. 🔴 **The authors declare split A may be contaminated** (old YouTube audio) and prescribe
   context-vs-no-context **deltas** on identical audio as the honest headline. Our model has one
   condition, so we cannot form that delta. The absolute number therefore carries a contamination
   risk we cannot remove. It is valid for what §8 needs — arm-vs-arm on identical audio and harness,
   where contamination is constant — and is **not** a claim about Typhoon's true ability.
4. ⚠️ **CC-BY-NC-4.0, NonCommercial.** Fine for evaluation and a thesis. **Not** fine as training
   data or inside a product. This is a business repo; keep the corpus on the evaluation side.

🔴 **`A26`'s `.158` unbiased figure is NOT our 0.328.** That is `typhoon-asr-realtime-nemo-ctc` — a
CTC variant with four added layers — measured across all 2,062 utts including the TTS ones. Ours is
the plain RNNT on :8007 over the 682 real only. Two different models on two different subsets; do
not read the gap as disagreement.

### Why `check_hyps.py` reports FAIL, and why that is fine

It demands all 2,062 utterances and we deliberately supplied 682, so it prints
`MISSING rows: 1380`. The scorer itself handled the subset correctly — verified against the
annotations: `gold: [144, 439]`, and 439 is exactly the gold-mention count of those 682 utts. Their
README warns the scorer "silently under-reports" on gaps, which is why that denominator was checked
by hand rather than trusted.

### Where arm 1 stands

✅ **Complete.** Harm term on two corpora (Common Voice CER **0.0518**; benchmark U-CER **0.1057** —
different corpora, not a contradiction) and the rare-word metric at **recall 0.328 / B-CER 0.2618**.
The gate §8 describes can now reject or accept an arm. Nothing is trained yet.

## 2026-09-12 (later) — ARM 1 MEASURED, and a second invisible convention mismatch

**Arm 1 of the adapter design exists.** `adapter/arm1_baseline.py`, 150 Common Voice 17.0 Thai
clips (CC0, ground-truth sentences) through `typhoon-asr-realtime` on :8007. Design S5 makes this
non-negotiable and first, because the 2026-08-27 endpointer table could not separate "the method
failed" from "this setup cannot learn" — nobody had measured the untrained model.

| CER mode | mean | median | p90 | max | perfect |
|---|---|---|---|---|---|
| raw (whitespace only) | 0.0723 | 0.0430 | 0.2000 | 0.800 | 55/150 |
| + punctuation | 0.0706 | 0.0405 | 0.2000 | 0.800 | 57/150 |
| **+ SARA AM folded — QUOTE THIS** | **0.0518** | **0.0000** | 0.1667 | 0.800 | **85/150** |

150 clips, 15.0 min of audio, CPU, **RTF 0.009**. Result JSON with every row:
`adapter/_out/arm1_typhoon_shipped.json`.

### ⭐ THE SECOND FINDING: Typhoon writes Thai SARA AM DECOMPOSED, and NFC does not bridge it

The vowel in `บำรุง` has two encodings that **render identically**:

    U+0E33          composed     ำ
    U+0E4D U+0E32   decomposed   ํ + า    (NIKHAHIT + SARA AA)

🔴 Unicode gives U+0E33 **no canonical decomposition**, so `unicodedata.normalize` does nothing:
measured, NFC changed **0 of 150** clips. Meanwhile Typhoon's hypotheses were **57 decomposed / 0
composed** and the Common Voice references **0 decomposed / 57 composed** — the two sides never
agree, and **47 of 150 clips (31%)** carried the penalty.

Folding it is worth **26.6% of the mean CER and 100% of the median** (0.0706 → 0.0518; median
0.0405 → **0.0000**, i.e. the median clip is a perfect transcript), and 57 → 85 perfect.

🔴 **This was a trap, not just an inaccuracy.** The generated text banks used the composed form
exclusively (987 occurrences, 0 decomposed). Train the decoder on those labels, score with an
unfolded metric, and the arm posts a ~26% general-CER **improvement that is purely a spelling
change** — and §8's gate adopts it for a reason with nothing to do with synthetic audio. Same class
of error as labelling numbers as digits when Typhoon writes words, found the same way, in the same
hour. **Two of the three convention axes would each have produced a confident wrong answer.**

Fixed on both sides, which is the only safe arrangement:
- **scoring** folds SARA AM on *both* operands (`adapter/_thai.py:for_scoring`)
- **labels** are written in the decomposed form Typhoon emits (182 + 299 occurrences converted), so
  fine-tuning does not silently change the model's output spelling
- `text` — what VoxCPM speaks — keeps the composed form, because that is ordinary written Thai

All three axes now verified: labels have **0 spaces, 0 digits, 0 composed SARA AM**; entity present
in label 245/245; both banks still plan to 8 distinct voices per text.

### What arm 1 is NOT

🔴 **Only the harm term.** §8 needs two numbers together and **rare-word accuracy is absent** —
that needs a real Thai corpus containing rare words, and `wayu-ai/thai-contextasr-bench` is still
not pulled. `rare_word_accuracy: null` is written into the result JSON rather than omitted, so the
gap cannot be forgotten. **Arm 1 is not complete.**

⚠️ Three limits on the corpus, all recorded in the JSON: the clips are **denoised**, so this is
clean-speech CER and a telephone number would be worse; they are Common Voice **read speech**, not
call-centre speech; and they contain **no rare words**, which is why they cannot serve the other
metric. Valid as a baseline for later arms on the same corpus and harness — not a production
accuracy claim.

### Reading the tail

mean 0.0518 against median 0.0000 is a tail, and it is mostly the metric: references under 30
chars have median 0.0345 but mean 0.1054, because a handful of catastrophes dominate a short
denominator. The worst clip (CER 0.800) is an RNNT **repetition loop** — ref `ครับ ครับ ผมจะใจเย็นลง`,
hyp repeated `ครับ` six times. ⭐ And one error is exactly the failure the whole experiment targets:
`โรงพยาบาลบำรุงราษฎร์` → `โรงพยาบาลบํารุงราช`, a **named entity** truncated. Anecdotal, but it is
the thesis in one row.

## 2026-09-12 — the two adapter text banks, and Typhoon writes number WORDS not digits

**Built the two text banks the adapter design needs, then found the labels were in the wrong
convention and fixed them.** Nothing was generated as audio and nothing was trained. New dirs:
`adapter/` (the banks and their tooling) and `colab/` (the generator that will speak them).

### ⭐ THE FINDING: `typhoon-asr-realtime` writes Thai number WORDS, unspaced — never digits

Measured against **53 real Thai turns** already sitting in `live/_out/turns-*.jsonl`. No server
start was needed: the live panel's path is mic → Silero → Typhoon (:8007) → llama-server → VoxCPM,
so the log's `asked` field *is* Typhoon's own output. This is the cheapest possible check and it
had never been run.

| | Typhoon | the banks as generated |
|---|---|---|
| ASCII digits | **0 / 53** | **279 / 279** labels |
| Thai number words | 7 transcripts | — |
| any space at all | **1 / 53** (mean 0.02/transcript) | mean **4.64** spaces per label |

The decisive row is a phone number read aloud — exactly the case where a digit-writing ASR would
emit digits: `ชื่อพอร์ตศูนย์แปดสามศูนย์หนึ่งแปดเก้าห้าสามเจ็ดครับ`. It wrote words. Dates likewise:
`วันที่ยี่สิบเอ็ด`, `วันที่ยี่สิบ`.

🔴 **Why this would have produced a confident wrong answer.** Design §5.3 phase 2 trains the
**decoder**. Training it on digit labels teaches it to *change convention*, which raises **general
CER** — the harm term in §8's adoption gate — and would have read as a **leak** from the synthetic
audio when it was really a relabelling. The arm would have been rejected for the wrong reason, and
nothing in the harness would have said so.

⚠️ **Scope.** This is `typhoon-asr-realtime` as deployed on :8007 — the model in the loop and the
one the adapter would train, so it is the right target. Other Typhoon checkpoints are unchecked.
The number-bearing subset is 7 of 53 turns from one speaker: small, but unanimous on digits (0/53)
and it includes the hardest case.

### What the finding cost, and what it killed

`adapter/fix_label_convention.py` applies three consequences:

1. **`label_text` is unspaced Thai words in BOTH banks.** 🔴 Phase 1 needed it too, which is easy
   to miss because that bank has no numbers: phase 1 trains the adapter through a **frozen**
   decoder, so a target disagreeing with the decoder's spacing makes the loss dominated by an error
   the adapter cannot fix — the exact failure the two-phase split exists to prevent. 344 spaces
   removed over 440 rows.
2. **Entities reduced to their Thai core** — `ควอนตัมชาร์จ 160` scores as `ควอนตัมชาร์จ`, because
   the model number is a normalisation artifact, not a rare word. 30 entities affected.
3. 🔴 **The 69 `postcode` rows lost their purpose entirely.** With the incumbent already writing
   number words there is no digit-normalisation gap to teach, and a 5-digit `entity` can never
   appear in a correct label, so §8 could not score them. **35 salvaged** (their sentence also
   names a district, so the entity was re-pointed at that place name), **34 dropped**.

**So these banks make no claim about numeric normalisation.** That was the postcode rows' only
lesson and the incumbent has it already. What survives is out-of-vocabulary **vocabulary**, which
is the side of the line synthetic data actually wins on — consistent with the whole TTS-for-ASR
survey (`notes/2026-09-07-tts-for-asr-survey.md`): synthetic teaches what to write, never how
speech sounds.

### The banks

| bank | rows | content |
|---|---|---|
| `adapter/texts/phase1_known.json` | **440** | everyday spoken Thai, 40 each over 11 call-centre themes, no digits, no proper nouns |
| `adapter/texts/phase2_rare.json` | **245** | one rare entity per row — `place` 155, `product` 90, all 245 entities distinct |

Written by `agy` (Gemini 3.5 Flash) per the house rule that gemini writes the Thai. Audited by
`adapter/check_text_banks.py`: phase 1 is 440/440 unique with exactly **one** repeated 18-char
opening in the whole bank (i.e. not templates); phase 2 has `entity` present in `label_text`
245/245. `*.raw.json` keeps the untouched model output so the relabelling is reversible.

⚠️ **A known flaw that deliberately was NOT fixed:** a place name can be paired with the wrong
postal code (one row gave บางใหญ่ as 11120; that is Pak Kret, Bang Yai is 11140). Province prefixes
*are* reliable — spot-checked Chiang Rai 57, Phatthalung 93, Suphanburi 72, Phayao 56, Nakhon
Ratchasima 30000. It does not affect training, because audio and label agree and the pairing is
internally consistent; geography is irrelevant to what the ASR learns. It *would* break any use of
these rows as fact.

### `colab/` — the generator, because the box has no room to generate

`colab/gen_colab.py` + `Generate_Thai_Synthetic.ipynb` + `README.md`. A port of `smartturn/gen.py`,
not a wrapper: no `app.py` import, clips packed into tar **shards** on Drive, resume driven by the
manifests on Drive rather than the VM's disk (which is wiped on disconnect). Verified by stubbing
the model: ran `--limit 25` of 46, re-ran, it skipped the 25 and finished at 46 with **zero
duplicate gen_ids**; manifest matches tar contents; scratch cleaned; no stray `.part`.

🔴 **Storage was never the reason to go to Colab, and the original request said it was.** Measured
from the corpora already here, 16 kHz Thai costs **~115 MB/hour** — 100 hours is 11.5 GB, and a
5 TB Drive would hold more than Typhoon's entire 11,000-hour corpus. The real reasons are that the
GPU is full at one concurrent call and generation is GPU-hours.

🔴 **Two bugs in `gen_colab.py` found by running it, not reading it.** (a) `--per-text 8` gave one
text eight clips in the **same voice** — the knob multiplied seeds instead of speakers, which would
have defeated both the strongest lever in the TTS-for-ASR literature and the §7 ratio sweep.
(b) The first fix hashed `(group, k)` and left distinctness to chance: 8 distinct voices for only
365 of 440 texts, 7 for 74, 6 for one. Replaced with a coprime stride (`base(group) + k*7 mod 150`),
which makes it **guaranteed** — 8 distinct voices for all 440, spread 22–24 per speaker, and
matched `pair_id` pairs still share a voice within each k. A third property the port had to keep:
clip seeds derive from the clip **id**, never its position, so a resumed run reproduces an
uninterrupted one.

### Still blocking the experiment — unchanged by any of this

The rare-word **evaluation corpus** (`wayu-ai/thai-contextasr-bench`) is not pulled; **Thai
telephone-channel RIRs** still gate arm 3, the arm that sets §2's bar; and **arm 1, the untrained
baseline, still has to run first** (S5). Generating audio unblocks none of them.

## 2026-09-11 (later) — the adapter design's pre-build checks, answered against the checkpoint

**Nothing trained, nothing measured through the loop.** This was §10 of
`specs/2026-09-11-synthetic-adapter-asr-design.md` — "check these before building" — actually
checked, by unpacking `typhoon-asr-realtime.nemo` and reading the NeMo installed in
`E:/asr-spike/.venv`. Six questions; **three answered, three still open**, and the answers move the
build order.

**1. ⭐ Arm 4 exists — Typhoon's encoder uses BATCH norm.** This was the one 🔴 question in §10,
because `H8` (SYNT++, 13% ASR / 18% KWS, the best-evidenced cheap arm) needs normalisation that
carries running statistics, and Conformers often use layer norm, which has nothing to split.
`encoder.conv_norm_type: batch_norm`, and the checkpoint carries **51 running-stat buffers** —
`running_mean`, `running_var`, `num_batches_tracked` for all **17** layers, at
`encoder.layers.N.conv.batch_norm.*`, shape `(512,)`. So there are 17 places to split real from
synthetic. **The cheapest high-evidence arm is unblocked.**

**2. ❌ Per-sample adapter toggling does not exist — the §5.2 fallback is mandatory.**
`set_enabled_adapters(name, enabled)` writes `adapter_cfg[name]['enabled']`, a **module-level**
config flag read at forward time. Real and synthetic therefore cannot share a batch: two forward
passes, or separate batches. This is a data-loader constraint, not a nice-to-have.

**3. ✅ The encoder's shape, measured not guessed.** `d_model 512`, `n_layers 17`, encoder
**108.78 M** params — the spec's 512/109 M figures from model cards were right. Whole checkpoint
**115.45 M** = encoder 108.78 + decoder 4.59 (RNNT, `pred_hidden 640`, 1 LSTM layer) + joint 2.05
(`joint_hidden 640`, 2048 classes) + preprocessor 0.02. `ConformerEncoder`, `dw_striding` ×8,
`rel_pos`, 8 heads, ff×4, conv kernel 9, `att_context_size [-1,-1]` (full-context, as already known).

**4. ⭐ The adapter needs no custom code.** `ConformerEncoderAdapter` already exists
(`conformer_encoder.py:1276`), and `ConformerLayer` is an `AttentionAdapterModuleMixin` that calls
`forward_enabled_adapters` at **two** tap points per layer — once after multi-head attention, once
after the whole layer. NeMo's `LinearAdapter` is §5.1's design exactly: LayerNorm →
`Linear(512,32,bias=False)` → swish → `Linear(32,512,bias=False)`, residual-**added** via
`ResidualAddAdapterStrategy`, and its docstring states it *"explicitly initializes its final layer
with all zeros"*. **S3 (zero-init) is free and already implemented.**

**5. 🔴 §5.1's parameter arithmetic was wrong and is corrected in place.** A bottleneck-32 adapter
is **33,792** params (16,384 + 16,384 + a 1,024-param LayerNorm), not "~65 k each". So **0.574 M /
0.53%** of the encoder at one tap per layer, or **1.149 M / 1.06%** using both NeMo tap points. The
spec's "~1 M total, about 1%" was right only because a 2× per-adapter error cancelled against
assuming one tap — the two-tap choice is now an explicit decision instead of an accident.

**What this changes about what to do next.** Arm 4's blocker is cleared, so it should run first
among the cheap arms. **Arm 3 — the RIR arm, the one that sets §2's bar — is now the only arm
blocked on an input nobody owns: Thai telephone-channel room impulse responses.** Finding those is
the critical path for this whole direction, not anything about the adapter. Still open alongside it:
the rare-word evaluation corpus (`wayu-ai/thai-contextasr-bench` not yet pulled or sized) and the
bottleneck width, which is a knob to sweep rather than defend.

**Also fixed:** `LITERATURE.md` rows `H3`, `H4` and `H7` pointed at arm numbers from the design's
**first** draft (H3 → "arm 3", actually arms 5 and 7; H4 → "arm 6", actually arm 8; H7 → "arm 4",
actually arm 6). The §11 revision renumbered the arms and the ledger was never resynced.

## 2026-09-11 — the Typhoon line surveyed properly, an architecture thesis designed, and three stale claims killed

**No code ran. No measurement was taken.** This was a research and design session; everything
below is reading, reasoning, or a document.

### 🔴 The session's own failure, first, because it is the reusable part

I read arXiv 2601.13044 in full plus two model cards and reported confidently that **Typhoon "has
never modified a model architecture — every improvement is data, backbone swap, curriculum or
benchmark."** I then recommended **contextual biasing** as his thesis contribution on the strength
of it sitting on their *future work* list.

His entire reply: *"read all in https://huggingface.co/typhoon-ai, there are a lot more"*.

The org holds **82 models**. Ten are ASR, and **six shipped in the previous nine weeks** — none
in the paper, none surfaced by four web searches. They falsified both claims at once, including
**contextual biasing, which shipped three days before I called it novel.**

**The rule, now in `CLAUDE.md`:** when the question is *what is a lab doing / where are they
going*, **enumerate the org first** —
`curl -s "https://huggingface.co/api/models?author=<org>&limit=1000"`, sort by `lastModified`,
read the recent cards. A paper is a snapshot eight months stale. This is the **third** time his
knowledge of the Thai landscape beat a search (Instadesk, then the streaming ASR, now this).

### What the Typhoon line actually does — ten models, four moves

**The corpus is the asset; the models are disposable.** Fix the labels (their own ablation: same
Whisper-large-v3 architecture, their normalised data, TVSpeech **10.36% → 6.32%**); swap or
convert the backbone; change the training schedule (Isan two-stage, worth **5.57 points**); change
the evaluation (a benchmark, a spelling standard, a transcription convention, a phonetic
dictionary). Full write-up in `CLAUDE.md` §"The Typhoon ASR line"; rows `A24`–`A28`.

🔴 **Three claims in this repo were wrong or stale and are now corrected:**

| was | is |
|---|---|
| "Typhoon uses 3.2 TTS h of 10,999 = **0.03%**" | true of the *general corpus only*. The **Isan stage is 62.48 h of 303 = 20.6%** — ~700× the ratio. The *mechanism* claim survives: in both stages the TTS exists only for numeric/date normalisation |
| "no small streaming Thai ASR fits beside the loop" | **`typhoon-asr-streaming-115m`** exists (Jul 2026, CC-BY-4.0), TVSpeech **19.4% @1040 ms**. 115M fits where the 0.6B's 2.9 GB does not — **may unblock next-session step 1 without the 9b→4b swap** |
| "one encoder, two modes is a config change" | the full-context RT model **collapses to 62.8% CER if forced to stream**. It is a retrain, and Typhoon already did it |

⭐ **Two things they published that this repo should use.** `typhoon-whisper-large-v3-ctc` is a
**29 M character-CTC head on a frozen encoder** giving forced alignment at **2.0 ms jitter** and
**pause-instrument r = 0.988** (MFA-Thai 0.767) — a far better pause instrument than the Silero
VAD our endpointer corpus was built on. ⚠️ It fixes label *precision*, not label *validity*.
And the **freeze-and-bolt-on-a-head pattern is now their house move**, done twice in September for
two capabilities — **turn-taking is the one they have not done.**

### 🔴 Our own synthetic data did NOT train an endpointer — and this was already measured

He asked whether we actually know synthetic data can train an endpointer. Checking rather than
answering from memory: **we measured it twice and it did not work.** On 400 real Thai pauses,
same seed, same set:

| model | trained on | EER |
|---|---|---|
| Smart Turn v3.2, **off the shelf** | — | **0.4175** |
| v3.2 frozen / full | v1 synthetic (11,971 clips) | 0.425 / 0.435 |
| v3.2 frozen / full | **v2** synthetic (2,375, hesitation spoken) | 0.435 / 0.450 |
| whisper-tiny × 4 | v1 / v2 | 0.470–0.495 |

Chance is 0.50; pause duration alone is 0.419. **Every fine-tune landed at or below both the
untrained model and a plain silence timer, and the "better" v2 data was worse than v1.** G16
predicted it: TTS makes strong turn-*hold* and weak turn-*yield* cues.

⚠️ **The control was never run.** Nobody trained the same model on *real* Thai pauses, so
"synthetic cannot train an endpointer" and "this model cannot learn Thai turn-taking from
anything" are not yet separable. `smartturn/out_v3/` is empty — v3's 3,188 clips were generated
and never trained. Row `H1`.

### Endpointing literature, swept via Consensus — 7 new rows, one of which bites

`E50`–`E57`. Most of the sweep matched rows already in the file. The ones that did not:

- 🔴 **`E50` contests `E9`.** Sato et al. 2024 re-ran the VAP cross-language experiment across
  *multiple datasets per language* and found performance depends on **the dataset, not the
  language** — specifically on how speech-segmentation labels are provided — and that a
  model trained in one language **can** transfer. E9 is this repo's published cause for Smart
  Turn's Thai failure. **Rebuild that argument on our own measured Thai numbers.**
- ⭐ **`E51`** Fukuda et al. 2025 (ASRU) do joint predictive ASR + turn-taking in one model. **The
  closest published precedent to the joint-head direction — must be cited and differentiated.**
- ⭐⭐ **`E55` is the academic frame.** Turn-taking *timing* is universal (Stivers 2009, 10
  languages, ±250 ms) but the *cues* are language-specific — and Japanese has 25 years of
  work on sentence-final particles marking transition-relevance places (Tanaka 2000, 181 cites).
  **Thai ครับ/ค่ะ/นะ/เหรอ is the direct analogue and has no equivalent literature** — while this
  repo holds the only quantification on real Thai audio.
- ⭐ **`E56`** Corps et al. 2022 (*Cognition*): "speech segments derived from transcribed corpora do
  not necessarily correspond to turns". **Our LOTUSDIS structural failure, published and citable.**
- ⭐ **`E57`** Thai Discourse Treebank — 148 Thai discourse connectives. Our group-C failures were
  dangling function words; this is the inventory for the half particles cannot catch.

### Designed: the adapter thesis

`docs/superpowers/specs/2026-09-11-synthetic-adapter-asr-design.md`. Freeze the encoder, route
**only synthetic audio** through a ~1 M adapter so the acoustic lesson lands somewhere deletable
while the vocabulary lesson lands in the shared decoder. **Two-phase, and phase 1 freezes the
decoder** — his call, and it closed a real hole: with the decoder free it is the path of least
resistance and soaks up the poison instead of forcing the adapter to work. Phase 1's second
objective is a discriminator (gradient reversal), **but only after a one-directional probe says
there is TTS-ness left** — measure before taking on two-player instability.

🔴 **Arm 1 (untrained baseline) is non-negotiable and runs first.** The `smartturn` table above is
exactly the failure it prevents: every arm below baseline, and no way to tell "the method failed"
from "this setup cannot learn".

⭐ **`H7` — there is no Thai synthetic-to-real exchange rate** (EN 7–8×, ES 9×,
FR 12×). It falls out of the ratio sweep for free and is publishable on its own.

### Built: `lessons/adapter.html`

Six interactive parts covering every mechanism in the design — what the encoder produces, what
frozen means, what the adapter adds, the nonlinearity, zero-init, and the two lanes end to end.
🔴 **Two explanation attempts failed before this** (matrix algebra in prose; then a 32-detector
figure). What worked was **one number and one slider**, showing that the graph is a straight line
without the nonlinearity and has a corner with it. Same lesson as the design work: start smaller
than feels necessary.

### ⚠️ Bookkeeping defect found

The TTS-for-ASR survey note and the `project-synthetic-speech-training` memory both cite rows
**`G1`–`G24` in `LITERATURE.md` §2.9. Those rows are not in the file** — the evidence
lives only in `notes/2026-09-07-tts-for-asr-survey.md`. Either transcribe them or stop citing
G-numbers.

---

## 2026-09-07 — TTS-for-ASR surveyed: Typhoon's 0.03% synthetic is the field's consensus, not an accident

**Nothing was measured on this box.** This is a reading entry: 6 Consensus queries, ~120 unique
records, **24 new `UNTESTED` rows (G1–G24) in `LITERATURE.md` §2.9**, synthesis in
`docs/superpowers/notes/2026-09-07-tts-for-asr-survey.md`. He asked for Consensus specifically and
ruled out web search, so GitHub / model cards / vendor blogs are **unswept ground** for this topic.

**What prompted it.** Typhoon ASR Real-time's paper (Table 2) breaks its corpus down as
GigaSpeech2 10,329.5 h + internal curated media 631.0 h + Common Voice 17.0 35.4 h + **internal
TTS (Orpheus) 3.2 h**, total **10,999.1 h / 9,971,887 utterances**, model 114 M params. The TTS
slice is **0.03%** and exists for one job the paper names: *"enforcing strict adherence to our
normalization standards, preventing common hallucinations in number and date transcription."* The
Isan adaptation is separate — 303 h, of which 62.5 h (21%) is TTS, again for numeric formatting.
The streaming 0.6B is NVIDIA's Nemotron 3.5 ASR Streaming checkpoint with a Thai BPE extension,
adapted on the same ~11 k corpus. `smartturn/data_v3/` is ~100% VoxCPM-generated, so the question
was whether 0.03% is a finding or an artefact of available data.

**It is a finding, and it is quantitative.** Synthetic-to-real exchange rate for ASR parity is
**7× clean / 10× noisy** (Voicebox, G1) and **8× EN / 9× ES / 12× FR** (XTTS, G2) — it worsens with
distance from English, and no Thai figure exists. The published floor for a fully-synthetic corpus
is a real:synthetic **WER ratio of 1.46** (G3), i.e. +46% errors, and that paper says "a large gap
remains". Every large published *win* is a vocabulary win — OOV 57% relative (G10), medication
names >65% (G11), rare words 57.6% → 16.5% WER with no real audio at all (G8). **TTS teaches a
model what to write, not how to hear.**

**The mechanism is diversity and acoustic irregularity, NOT naturalness.** TTS judged natural by
humans still trains bad ASR (G3). Voice-cloned audio of *lower* quality but *more voices* beats
higher quality with few voices, and low-variability synthetic **saturates** (G7). Convolving with
**room impulse responses** closes the gap "not by making synthetic speech cleaner or more natural,
but by reproducing the acoustic irregularities of real recordings" — with layer selection that
matched a fully-real baseline on **25% of the real speech** (G6).

🔴 **Two rows that bear directly on this repo.**

1. **G4 — the architecture trap.** Synthetic data substantially helps *attention encoder-decoder*
   ASR but reportedly has **"no relevant effect" on CTC and transducer** models (Rossenbach 2021).
   **Typhoon is a FastConformer-Transducer.** So a TTS-augmentation result obtained on Whisper
   does **not** transfer to our ASR, and Typhoon may have used almost no synthetic audio because
   its architecture could not spend it. ⚠️ 2021 finding, and **G22 partly cuts against it**
   (67 h real + 636 h simulated beat a 2,700 h zero-shot model, on a FastConformer both sides).
2. **G16 — the `smartturn/` risk, and it is specific.** TTS systems produce **strong turn-HOLD but
   weak turn-YIELD cues** (Ekstedt 2023, 3 commercial + 2 open TTS scored by a VAP model). We are
   training a turn-*end* detector on audio from a class of system measurably weak at producing
   turn-*end* cues. Softened by our signal being largely **lexical** — Thai final particles survive
   synthesis intact — and by the ETD Dataset (G20) deliberately mixing synthetic with real for this
   exact task. **The rule it forces: you may train on synthetic, you may not evaluate on it.**

⭐ **Two findings that strengthen the endpointing thesis rather than weaken it.** G17: a monolingual
VAP turn-taking model trained on one language **does not predict well on other languages**
(Inoue 2024, English/Mandarin/Japanese) — published, peer-reviewed, independent support for our
Smart Turn v3.2 Thai failure, which makes our result an instance of a known phenomenon instead of
an anomaly. G19: an independent group measures **Smart Turn v3 at 58.9% recall against their
87.7%**, at 36 ms vs 800–1300 ms — so the baseline we found failing is failing for others, on
English.

⭐ **The top lever in this literature is one already installed.** G7/G8/G9 all rank speaker
diversity first, and say many mediocre voices beat few good ones. VoxCPM2's natural-language
**voice design** (`LITERATURE.md` T16) is an unbounded speaker generator requiring no reference
audio — and every synthetic corpus in this repo so far has been built on essentially one voice.
Untested here; it is the cheapest change the survey suggests.

**Also recorded:** Typhoon's labelling pipeline is three Whisper-Large models transcribing in
parallel with majority voting, humans reviewing **only** transcripts containing Arabic numerals or
unusual punctuation. And Typhoon's actual answer to Thai data scarcity was **GigaSpeech2** (G21) —
30,000 h crawled from YouTube, Whisper-pseudo-labelled, MMS force-aligned, noisy-student refined —
which is 94% of its corpus. **Pseudo-labelled real audio, not synthesis.**

⚠️ Their CER figures (0.0984 realtime; 9.3–14.1% streaming) are **not** comparable to our measured
0.032 / 0.117 / 0.192 — different test sets, different axis. Do not put them in one table.

---

## 2026-09-06 — a streaming Thai ASR EXISTS, is wired on :8011, and does not fit on this box

⭐ **This overturns the repo's own headline finding.** `docs/superpowers/notes/2026-08-17-streaming-thai-asr-spike.md`
concluded "no streaming Thai ASR exists", and `CLAUDE.md` carried it as a red-flag fact. That was
true of `typhoon-asr-realtime` and is still true of it — its checkpoint is full-context
(`att_context_size [-1,-1]`) despite the name. It is **not** true of the field. The old entry is
not edited; it is marked superseded by this one.

🔴 **And I did not find it — he did.** A targeted web search *and* a re-fetch of the model card
both missed it; his entire message was the URL. Second time this has happened (Instadesk, 2026-08-27).
The generalised rule is now in the `feedback-ask-him-for-the-vendor-list` memory: **before asserting
that something does not exist in Thai speech AI, say what was checked and ask him.** A repo doc
restating an absence is not evidence the absence still holds.

**The model.** `typhoon-ai/typhoon-asr-streaming-nemotron-0.6b` — `EncDecRNNTBPEModelWithPrompt`,
cache-aware FastConformer-Transducer, 0.6B, built on NVIDIA Nemotron 3.5 ASR, OpenMDW-1.1. It is
prompt-conditioned: `set_inference_prompt("th-TH")` is **required**, and NeMo 3.0.0 in
`E:/asr-spike/.venv` already has the class — no upgrade was needed.

⭐ **Measured here, three `_spike_probe` clips, CER against the same references:**

| | long11 | solo14 | long19 | mean |
|---|---|---|---|---|
| `typhoon-asr-realtime` 115M CPU, offline (2026-08-17 window) | 0.032 | 0.236 | 0.084 | **0.117** |
| this model 0.6B GPU, offline | 0.123 | 0.183 | 0.141 | **0.149** |
| this model 0.6B GPU, streaming @240 ms | 0.117 | 0.229 | 0.229 | **0.192** |

🔴 **The incumbent's column is from a DIFFERENT measurement window** — per the drift warning, that
comparison is indicative, not a verdict. Bracket both in one window before quoting it. The
direction (streaming is less accurate) was stable across every configuration tried.

⭐ **Look-ahead is not a quality knob on this evidence.** `att_context_size [56, R]`, R ∈ {3,6,13}
→ 240/480/1040 ms. Mean CER **0.192 / 0.193 / 0.195** — flat. So the cheapest is taken and 240 ms
is the default. **R=0 (80 ms) crashes** (`cannot reshape tensor of 0 elements`, zero-size cache).
Do not raise R hoping for accuracy.

⭐ **Speed:** ~34 ms per encoder step against a 320 ms chunk on the GPU, 29 ms p50 / 40 ms p95
round-trip over HTTP per 160 ms chunk. First partial at **~700–950 ms** of audio through the panel.

🔴 **THREE FEEDING BUGS, all silent, all in how NeMo's own buffer is driven.** Each produced a
plausible-looking wrong answer rather than an error:

1. `CacheAwareStreamingAudioBuffer.append_audio()` defaults to `stream_id=-1`, which creates a
   **new batch row per call** — it is built for batching files, not appending to a stream. Feeding
   75 slices made 75 one-frame streams and the encoder never ran once. Pass `stream_id=0`.
2. Running the mel preprocessor on each arriving 160 ms slice **independently** injects an edge
   artefact per slice. Symptom: transcript froze at ~8 s and was badly wrong. Fix: preprocess the
   whole stream and append only the new frames, holding back the last 4 (they sit against the
   padded edge and would be frozen in wrong).
3. **The one that mattered.** The buffer's `__iter__` advances `buffer_idx += shift_size`
   *whether or not that many real frames exist*. A short chunk therefore feeds zero-padding AND
   skips the frames that arrive next. Symptom: a fluent, grammatical Thai transcript **missing
   most of the words** (29 chars against 154) — the most convincing wrong output of the session.
   Fix: step the encoder only when a full chunk of real frames is available. CER 0.903 → 0.117.

NeMo's own `transcribe_simulate_cache_aware_streaming` **refuses this model class**, so there was
no reference implementation to copy — hence the above.

🔴 **IT DOES NOT FIT ON THIS BOX, ON EITHER DEVICE.** Measured, and it cost VoxCPM:

| | needs | free with the loop up | outcome |
|---|---|---|---|
| GPU | **2.9 GB** VRAM (10609 → 7684 MiB when stopped) | 1.7 GB | VoxCPM needs ~6.9 GB, had 5.6, **died silently** |
| CPU | ~4.5 GB RAM (a 2nd NeMo process) | ~5.0 GB | `import nemo` → **MemoryError** |

⭐ **The VoxCPM failure mode is worth its own line: exit code 0, no traceback, no message.**
`:8005` simply never appeared and the only symptom was `ConnectionError` at the panel. Restarting
it with :8011 stopped and nothing else changed: up in ~110 s. **A silent exit-0 death during model
load on this box means VRAM, and it will not tell you.** VoxCPM alone holds ~6.9 GB VRAM and
**8.9 GB of system RAM**.

So `live/run-thai.ps1` does **not** start :8011 by default — it is `-Stream`, opt-in, with the
numbers written into the script. The route to having both is to free VRAM; the 9b → 4b LLM swap
is the obvious candidate and is wanted anyway (`HANDOFF.md` §10).

**Built:** `asr/server_streaming.py` (:8011, stdlib HTTP, session-keyed `POST /stream?sid&final`,
plus a one-shot `/asr` for parity), `asr/run-streaming.ps1`, `live/thai.py` streaming client on a
queue + worker so the mic path never blocks, `live/thai.html` shows the growing partial dimmed.
Verified: the panel degrades cleanly with :8011 down (turn completes, partial empty, error named).

🔴 **The partial is a DISPLAY, not the transcript of record.** :8007 still produces `asked`
because it is more accurate. They visibly disagree on screen; that is the point of dimming it.

---

## 2026-08-27 — the endpointer instrument is built, and it says the LLM solves the blocker

Executed `docs/superpowers/plans/2026-08-27-thai-endpointer-instrument.md` tasks 1-13.
**219 tests**, 12 commits. Two corpora built, three baselines measured, one hypothesis killed.

### ✅ Phase 0 — Typhoon CAN read far-field meeting audio

`loop.bench.eot_cer`, 90 utterances over 6 sessions: mean CER **0.171**, median 0.138 →
`GO-WITH-CAVEAT`. On the 83% of references that can be scored fairly, **0.123** — inside GO.

🔴 **Two bugs in the measurement itself, both inflating it, found before trusting the number.**
The first read was 0.185 and its worst cases were annotation artefacts, not audio:

- `#word#` (7,945 in the corpus) wraps **real spoken Thai** — `#อย่างนี้#` is "like this". The
  normaliser left the `#` in, charging two character errors per marker against transcripts that
  were correct.
- `[WORD]` (7,371) is **also speech**: `[OK]` `[MENU]` `[BUFFET]` `[M]` `[1]` are code-switched
  words the speaker said, written in Latin script while Typhoon emits Thai. Deleting them scored
  a correct transcript as an insertion; keeping them would score it as a substitution. Neither is
  fair, so they are now flagged and reported separately (17% of utterances).
- A 2.3 s interval can hold a 3-character interjection, where one substitution is 0.33. Added a
  **10-character reference floor**, fixed on arithmetic before the re-run, not after seeing it.

One case had Typhoon **more accurate than the human reference** — it heard `ซีรีส์` where the
annotator omitted it.

### The corpus

`loop/bench/eot_build.py` — Silero finds the pauses, the TextGrid says who held the floor,
Typhoon transcribes the turn so far. **13,602 rows** (v1 labels) / **10,925** (v2), ~25 min a
build, session-disjoint 70/10/10 split committed as `split.json`.

### 🔴 Phase 2 — the latency blocker is dead, the accuracy claim was a label defect

`loop.bench.eot_eval --split test`, 1,589 held-out rows from 9 unseen sessions:

| baseline | finds real turn ends | interrupts mid-turn |
|---|---|---|
| timer @600 ms (today) | 34.1% | **15.9%** |
| timer @400 ms | 54.6% | 34.2% |
| particle grammar | 17.1% | 6.0% |
| **resident LLM** | **85.4%** | **58.1%** |

✅ **E4's 569 ms is gone: p95 check cost 108 ms** (+88 ms STT = 196 ms) against a 200 ms window.
It was ollama's tax, and ollama is gone.

✅ **Two independent agreements say the instrument is sound**: the timer read **39.9% EER**
against the 41.9% from the 8-session probe, and the particle grammar read **17.1% recall**
against the 23.3% particle ceiling measured on 4,688 floor changes.

🔴 **The 58.1% is not the model's fault.** The false alarms are complete Thai sentences a
meeting speaker continued past — `แต่ว่าเราชอบทํากินเองที่บ้านมากกว่า` ("but we prefer cooking
at home") — labelled INCOMPLETE only because nobody interrupted. The rate was **flat across
pause length** (59% at 0–300 ms, 49% at 1–2 s): the signature of a label problem.

⚠️ **The gate was inadequate, and that is the more useful finding.** Spec §6 gated on EER and
latency with **no ceiling on cut-off rate**, and compared a binary rule's single operating point
against a swept threshold's EER. It would have adopted a 58% interrupter over today's 15.9%.
**Any future gate needs a cut-off ceiling.**

### ❌ The strict relabel FAILED its own test, and introduced a circularity

`loop/bench/eot_label.py` — three buckets instead of two: COMPLETE only if someone else holds
the floor ≥1.0 s and the original does not reclaim within 2.0 s; INCOMPLETE only if the same
speaker is back within 500 ms; **everything else dropped**. Thresholds fixed before the run.
Uses only timestamps and speaker identity — filtering with the LLM's own verdicts would be
marking its homework with its own answers.

🔴 **Its primary prediction did not happen. The LLM's false alarm went 58.1% → 59.3%, recall
identical at 85.4%.** Relabelling changed nothing about the text-only judgement.

🔴 **And it made every gap-based rule circular.** `MAX_BREATH_MS = 500` caps INCOMPLETE gaps at
**480 ms**, so any threshold ≥500 ms scores **0.0% false alarm and 100% precision by
construction**. The flattering `timer @600ms → 0.0%` and `LLM and gap>=500ms → 100%` rows in
that table are artefacts and must not be quoted.

🔴 **Conclusion: LOTUSDIS cannot answer this question, and it is structural, not fixable.** In a
three-party meeting a speaker continues because nobody interrupted them; nothing in the
recording says whether they would have yielded had an agent spoken. "The floor did not change"
is not "the caller was not finished," and no re-derivation from these timestamps bridges that.

### ⭐ The transfer test is the real result — the LLM solves the blocker

`loop.bench.eot_transfer` on `corpus_tts/` (18 items, turn ends true by construction):

| | group A (particle) | **group B (no particle)** | group C (mid-thought pause) | total |
|---|---|---|---|---|
| particle grammar | 6/6 | **0/6** | 6/6 | 12/18 |
| **resident LLM** | 6/6 | **6/6** | 1/6 | **13/18** |

✅ **The adapter is verified**: the particle grammar reproduced 6/6 and 0/6 exactly, which is the
2026-08-18 measurement on an unchanged corpus.

⭐ **Group B is the reason route C is not the default** (`HANDOFF.md` §7): a caller who finishes
*without* a closing particle looks identical to one mid-thought. **The LLM gets all six** —
`พรุ่งนี้หน้าร้านเปิดทําการกี่โมง` and five like it, where the grammar gets none.

🔴 **But it fails group C 5 of 6, and those are real errors, not label artefacts.** The misses
end on dangling function words — `...จะส่งถึงภายใน` ("will arrive within"), `...มันเป็น` ("it
was"), `คือเงินที่ขอคืนไป` (a bare noun phrase). Genuinely incomplete Thai read as complete.

**They are exactly complementary**: the grammar wins continuations and is blind to particle-less
endings; the LLM wins endings and is blind to continuations. Neither is sufficient alone.

⚠️ **Prompt engineering moves along the curve, not up it.** A v2 prompt naming the dangling-word
failure explicitly scored **14/18** — group C 1→3 but group B 6→5. On 18 items that is noise,
and further tuning would fit it. This independently supports E2's finding that SFT beats
prompting on the accuracy-latency trade-off.

### 🔴 Separate finding: `MAX_WAIT_MS = 1500` cuts off mid-thought pausers by itself

Group C's pauses run **1200–3000 ms**, so on 5 of 6 the escape hatch fires *before* the caller
resumes — **regardless of what the judge said**. Both judges show an 83.3% cut-off rate on that
subset for this reason alone. The successful route-C run of 2026-08-18 used `max_wait 3500`;
`endpoint_semantic.py` ships 1500. A judge being right cannot help if the safety timer commits
the turn first.

### 🔴 THE STRUCTURAL RESULT — text cannot separate "finished" from "pausing"

Two follow-up fixes were proposed and **both were measured wrong**. What they proved instead is
the most durable finding of the session.

**Grammar and LLM cannot be combined.** On `corpus_tts`, the `(grammar, llm)` signal is:

| group | signal |
|---|---|
| A — finished, with particle | `(True, True)` |
| **B — finished, NO particle** | **`(False, True)`** |
| **C — paused mid-thought** | **`(False, True)`** (5 of 6) |

B and C are **identical**. So `OR` scores 13/18 (the LLM alone) and `AND` scores 12/18 (the
grammar alone). No boolean combination of the two adds anything, ever.

🔴 **The reason is structural: at the decision point, "I have finished and used no particle" and
"I am pausing mid-thought" are the same text.** The only thing that separates them is TIME — C
resumes, B does not. **No text-only endpointer can distinguish them, at any accuracy.** This is
also why the LOTUSDIS false-alarm rate was flat across pause length (59% at 0–300 ms, 49% at
1–2 s): text cannot see time.

**And raising `max_wait` does not help once the LLM is judging.** The safety timer only fires
when the judge says *not finished*; on group C the LLM says *finished*, so it commits at 200 ms
and `max_wait` never gets a turn. Raising 1500 → 3500 saves group C **only under the particle
grammar**, which misses all of group B anyway. The two proposed fixes cancel each other.

Measured on `corpus_tts` (group C pauses: 1200/1500/1800/2100/2400/3000 ms):

| max_wait | group B waits | group C cut off |
|---|---|---|
| 1500 ms (shipped) | 1500 ms | 5/6 |
| 2500 ms | 2500 ms | 1/6 |
| **3500 ms** | **3500 ms** | **0/6** |

### ⭐ The trade, priced on 13,602 real Thai rows

| | share |
|---|---|
| turn ends carrying **no** closing particle — only the LLM catches these | **73.7%** |
| mid-turn pauses longer than **1200 ms** — where firing at 200 ms does clear damage | **1.63%** |
| (turn ends WITH a particle) | 26.3% |

✅ That 26.3% independently reproduces the **23.3%** particle rate measured on 4,688 floor
changes — a third internal agreement that the instrument is sound.

**So text-only endpointing forces a choice, and the data does not make it:**

- **LLM judge** — 73.7% of callers answered ~1300 ms sooner; the ~1.6% who pause long mid-thought
  are interrupted.
- **Particle grammar + long `max_wait`** — nobody is interrupted; 73.7% of callers wait 1.5–3.5 s.

73.7% against 1.6% argues hard for the LLM, but **being cut off hurts more per incident than
waiting**, and by how much is a product judgement, not a measurement. ⚠️ And the 1.63% comes
from meeting audio whose labels we have already shown do not mean what we need — it is the right
order of magnitude, not a number to bet on.

**What would actually break the tie is a model that sees text AND time together**, which no
baseline here does: the timer sees only time, the grammar and the LLM see only text. That is the
strongest remaining argument for phase 3 — not better text accuracy, but a *joint* feature.

---

### Where this leaves phase 3

**Do not train on LOTUSDIS labels** — they encode "will this meeting speaker keep talking",
which is not the product's task. What the corpus IS good for is unchanged: 10,925 rows of real
Thai transcribed by the deployed recogniser, which is the right *text* distribution (N4) even
where the labels are wrong.

---

## 2026-08-27 — the first Thai pause distribution, and a design for training the endpointer

**No code changed and nothing was trained.** This session measured the corpus, read the text-side
literature, and wrote a design. Continues the 2026-08-27 reading session below.

### ⭐ The measurement — the first pause numbers for Thai conversation in this repo

The 2026-08-26 entry below withdrew a pause-distribution table as "unrecoverable from the
TextGrids" and said **VAD on the audio is the only route**. That route was taken.

Silero over 8 LOTUSDIS sessions (~2.0 h), every pause labelled by the TextGrid floor:

| | n | p10 | p25 | **p50** | p75 | p90 |
|---|---|---|---|---|---|---|
| same speaker resumes (INCOMPLETE) | 1,137 | 160 | 224 | **288** | 448 | 608 ms |
| floor changes hands (COMPLETE) | 396 | 192 | 256 | **416** | 640 | 896 ms |

204 unattributable (overlap/noise).

🔴 **Pause duration alone separates the two classes at 41.9% EER**, best threshold 325 ms —
barely above chance. Published acoustic-feature work reaches **20% EER at 100 ms latency**
(row E26), so roughly half the error sits in signal a pause-length timer throws away.

The threshold curve, on real Thai:

| window | mid-turn pauses that fire wrongly | cost on every true turn end |
|---|---|---|
| 200 ms | 77.1% | +200 ms |
| 400 ms | 30.5% | +400 ms |
| **600 ms (deployed)** | **10.9%** | **+600 ms** |
| 800 ms | 3.4% | +800 ms |

⚠️ **Not comparable to the repo's "18% of callers cut off"** — that denominator is utterances,
this one is pauses.

### The annotation is structurally blind, and now it is quantified

Of 21,998 speech intervals, **15,270 of the 21,908 consecutive pairs abut exactly**
(`xmax == next xmin`): 13,288 same-speaker (annotator segmentation, **not pauses**) and 1,982
zero-gap floor changes. Only 6,638 pairs carry a timed silence, and those are long — same-speaker
p50 **2207 ms**, floor-change p50 **3815 ms**. Below 800 ms: 422 + 147 rows in the entire corpus.
The 400/600 ms decision lives in a range the TextGrids cannot see.

🔴 **Consequence for training:** naively labelling "same speaker resumes = INCOMPLETE" from the
annotation yields 17,220 rows of which 13,288 have **no pause at all** — they are not endpointing
decisions, because nothing ever went quiet. The decision moments must come from VAD.

### Literature — 15 rows added, E36–E50

Consensus sweep aimed at the text route: 6 queries, ~110 abstracts. Full list in `LITERATURE.md`.
All `UNTESTED`. The four that changed the design:

- ⭐⭐ **E36 (de Ruiter 2006, *Language*, 400 cites)** — flatten pitch to completely flat and
  turn-end projection accuracy is **unchanged**; remove recognisable words while keeping
  intonation and it **degrades significantly**. Lexicosyntactic content is *necessary and possibly
  sufficient*; intonation is neither. **This is a post-hoc vindication of the text-only call, and
  a second post-mortem for E1** — Smart Turn was reading the channel that carries less signal.
  ⚠️ Counterweighted by **E37** (Bögels 2021): button presses do land at intonational boundaries.
  Cite both or neither.
- 🔴 **E38 (Montenegro 2021)** — an EOT classifier trained on clean text **degrades on ASR text**,
  and their fix is an error simulator matched to the recogniser's rate. We own the audio *and* the
  recogniser, so the design transcribes the corpus with Typhoon instead of simulating. Governs how
  the training set is built.
- ⭐ **E41 (JAL-Turn 2026)** — a **frozen shared ASR encoder** lets turn-taking run in parallel
  with recognition at **no added end-to-end latency**, and describes an automatic label pipeline
  from real dialogue corpora. Kills the main objection to the text route.
- ✅ **E40 (Raju 2023, ASRU)** — "two-pass endpointing": first pass proposes, an **EP Arbitrator**
  verifies; improves the cutoff/latency trade-off **regardless of the first-pass model**.
  `endpoint_semantic.py` is this architecture, published.

Also: **E42/E43** put *projection* (E8/E11's answer to reactive endpointing) within reach of the
text route with no new model — an LM's P(end-of-utterance K tokens ahead). **E44** (TurnGPT) is
the architecture template. **E47/E32** argue the target should be regression on time-to-resume,
not a binary. ⚠️ **E49** reports 99.65% on Thai sentence completeness — **do not quote it**, it is
2,000 hand-written sentences in a low-tier venue.

### Built — the design

`docs/superpowers/specs/2026-08-27-thai-endpointer-training-design.md`.

Four phases, and **phase 2 gates phase 3**: the resident 9b already scores 22/24 zero-shot (E4)
and its only disqualification was 569 ms, of which ~360 ms was ollama's tax — now deleted. Asking
it costs **zero extra VRAM** because the weights are already loaded; it costs a slot
(`-np 1` queues 3–6× worse) and per-slot context (65–79 vs ~93 tok/s). **If it clears the gate,
no model gets trained at all.**

Phase 0 is a **go/no-go**: Typhoon reads clean Thai at CER 0.032, and LOTUSDIS is a far-field mic
with three people in the room. CER ≤ 0.15 continues, > 0.30 stops the route. Measuring that before
building anything on the transcripts is the whole point of having it as phase 0.

**The deliverable is a table, not a model** — timer / particle grammar / resident 9b / trained
model, one row each, on EER, cut-off rate, realised ms per turn, and transfer.

🔴 **The transfer test may be the real result.** Everything trains on three-party meeting audio
about prompted topics; the product is two-party phone calls. `corpus_tts/` and the frozen corpus
stay held out, reported **broken down by utterance type, never averaged** — a caller reading an
account number digit by digit has no completeness cue at any point, and an averaged number hides
exactly that.

---

### Audio conditioning — recorded as a branch, deliberately not built

He asked whether normalising or enhancing the corpus audio would help. It became a conditional
branch off phase 0 (plan Task 5, step 7) rather than work, for three reasons:

1. **The classifier reads text and never sees a waveform**, so audio reaches it through exactly
   one channel: Typhoon's CER. Phase 0 already measures that, so the branch is settled by a
   number we are about to have. Building it first would be spending on an unconfirmed problem.
2. 🔴 **"Better" is the wrong direction.** Far-field and telephone audio are *differently*
   broken -- reverb and distance against band-limiting and codec crush. Cleaning the corpus
   yields wideband studio speech, a THIRD domain, not our deployment one. The move that targets
   the gap is **telephone-band augmentation** (300-3400 Hz + a telephony codec), which makes the
   audio worse on purpose so Typhoon's errors on the corpus resemble its errors on a real call.
   That is the audio twin of E38/E39. ⚠️ Our own reasoning -- **the supporting literature is
   unread and no ledger row was written**, because a row needs a source and a number.
3. 🔴 **Whatever wins must also run live.** Conditioning only the training audio recreates by
   hand the exact mismatch N4 exists to prevent.

⚠️ And the honest caveat: this addresses the **smaller** of the two domain gaps. The larger one
is content -- a caller reading an account number has no completeness cue at any sample rate.

---

## 2026-08-26 — the literature is indexed, and the Thai conversation corpus is on disk

**No code changed and nothing was measured on the GPU.** This session was reading and data
acquisition. Three artefacts, one dataset, and two withdrawn claims.

### Built

- **`LITERATURE.md`** — the claim ledger, now **101 rows**: 82 `UNTESTED`, 11 `REJECTED`,
  5 `CONFIRMED`, 3 `RULED-OUT`, 2 `BLOCKED`.
- **`docs/superpowers/notes/2026-08-25-speech-chatbot-methodology-and-latency.md`** — how the
  field builds speech chatbots, what they measure, and what that says about our numbers.
- **`docs/superpowers/notes/2026-08-25-bibliography.md`** — 148 papers + 17 code/model links,
  grouped and starred. An index, not a reading list.
- Consensus MCP wired at user scope and authenticated (20 results/query).

### The findings that change the roadmap

🔴 **Reactive endpointing may be a category ceiling.** Human turn gaps are ~200 ms but language
*production* takes >600 ms, so humans **predict** turn ends rather than react (Levinson &
Torreira 2015). Every endpointer we own is reactive. Udupa et al. 2026 forecasts endpoints
**2.56 s ahead** for **−505 ms** at +28.4% compute, **with no streaming-ASR dependency** — the
largest single item in the file, aimed at the stage that is 47% of our loop. Rows E8, E11.

✅ **Smart Turn's Thai failure has a published cause, and it is not tonality.** Monolingual
turn-taking models simply do not transfer; multilingual training fixes it, and Mandarin (tonal)
is in their set (Inoue 2024). Confirmed twice, on two tasks. Rows E9, E21.

⭐ **There is a whole family of non-model endpointers faster than ours.** Arsikere et al. 2014:
cheap pre-pausal acoustic features (lengthening, pitch, rate, spectral constancy) discriminate
final from non-final pauses at **20% EER / 100 ms latency**, using *no ASR, no speaker info, no
training data in the target language*. Their baseline was a 500 ms threshold at a 36% cutoff
rate — the same wall as our 600 ms. ⚠️ Do **not** quote 36% against our 18%: different corpora,
and theirs was elicited to provoke the failure. Rows E26–E34.

✅ **A cascaded architecture is independently vindicated, twice.** Thai-Duplex-Bench (JCSSE
2026) has a cascaded baseline beating both GPT-Realtime-Mini and Gemini Live on Thai
service-role adherence (97.0% strict pass); Qiu et al. 2026 open with *"fully self-hosted
end-to-end solutions are not yet available."* Rows D3, S14.

⚠️ **Our 1375 ms is competitive with shipped systems measured the same way** — Vapi p50 1558 ms,
Retell 1740 ms, GPT-Realtime-Mini 1.5 s in Thai. Always attach the caveat: their numbers carry a
telephony leg ours excludes, ours is a 22-item quiet-box corpus. 🔴 And one result argues
*against* the 500 ms target — optimal dialogue response measured at **600–1500 ms** (Cao 2024).
Rows S7, S13.

🔴 **JaiTTS-v1.0 is released and built on VoxCPM** — CER 1.94% vs human 1.98%. The memory index
saying "unreleased" is stale, and `_archive/` investigated a *different*, F5-based JaiTTS. Row T4.

### Acquired — a real Thai conversation corpus

**LOTUSDIS** (NECTEC, CC-BY-SA 4.0), now on disk:

```
E:\Claude\datasets\LOTUSDIS\                    13 GB   160,915 utterance rows + transcripts
E:\Claude\datasets\LOTUSDIS_sessions\wav\      4.5 GB   105 full sessions, Con123 mic
E:\Claude\datasets\LOTUSDIS_sessions	extgrid\ 1.5 MB   90 annotated timelines
E:\Download\wav.zip                              54 GB   full archive, 12 mics, 100 GB unpacked
```

Verified: all 90 TextGrids pair with audio and **90/90 durations agree within 1 s**. The CSV's
`_chunkN.wav` paths give exact utterance ordering, so audio↔timeline alignment needs no text
matching (a 100%-match text join was built first — correct but the hard way).

**10,599 turn boundaries are derivable with zero annotation**: same speaker resumes = INCOMPLETE
(5,638), floor changes hands = COMPLETE (4,961).

### 🔴 Two claims withdrawn in-session

1. **A pause-distribution table** ("a 600 ms window would cut off 94.9% of Thai mid-turn
   pauses") — **wrong and unrecoverable from the TextGrids.** The annotation marks short pauses
   as **inline `<n>` tokens inside an interval's text** (38,844 of them) and only gives long
   silences their own timed row (10,599, p10 = 813 ms). Everything under ~800 ms — exactly the
   range where the 400/600 ms decision lives — has no timestamp. **VAD on the audio is the only
   route.**
2. **"114 hours of Thai."** It is **32,183 utterances ≈ 22.8 h recorded on 5 microphones.**
   Every mic has exactly 32,183 rows, and 22.8 h matches the TextGrid timeline total
   independently. Legitimate augmentation for ASR; **not** 114 h of distinct speech, and the
   50–200 h labelled target should be read against ~23 h.

### ⭐ The measurement worth keeping

**The Thai final-particle hypothesis, on 4,688 real floor changes:**

| | particle in final ~8 chars |
|---|---|
| before the floor changes hands | **1,093 / 4,688 = 23.3%** |
| before the same speaker continues | 1,496 / 17,220 = 8.7% |

The signal is real (2.7x enrichment) and **far too weak to endpoint on** — `looks_finished`
would miss three-quarters of genuine Thai turn ends. This confirms the synthetic corpus's
group-B result on real data and puts a measured ceiling on the particle grammar.

### Caveats on the corpus

3 speakers in 87 of 90 sessions (4 in the rest), 41 prompted topics, **Central Thai only**,
far-field meeting register, 96/24 kHz mono (resampling to 16 kHz is mandatory). ⚠️ The CSV
reports **465 distinct speaker labels** across 90 three-person sessions — that fits neither
session-scoped (~270) nor global (~86, per the paper). **Do not rely on speaker identity until
that is understood.**

---

## 2026-08-25 — first real literature pass: the field says *predict*, and a Thai benchmark backs our architecture

Consensus MCP wired in (remote server, user scope) + web search. 6 paper queries, 4 web queries.
Full note: `docs/superpowers/notes/2026-08-25-speech-chatbot-methodology-and-latency.md`.
**15 new ledger rows, all `UNTESTED`** — nothing here was run on this box, and two of the load-bearing
papers are 0-citation 2026 preprints.

**1. 🔴 Reactive endpointing may be a category ceiling, not a tuning problem.** Levinson & Torreira
2015: human turn gaps run ~200 ms but language *production* takes >600 ms — so humans **predict**
turn ends rather than react to them. Every endpointer we have built, and every one we surveyed in
August, is reactive. Row E11. And Udupa et al. 2026 ("Endpoint Anticipation") forecasts end-of-turn
**up to 2.56 s ahead**, speculatively runs LLM+TTS on partial context, and reports a **505 ms
average latency cut for 28.4% more compute** — beating VAP baselines, **with no streaming-ASR
dependency.** Against a stage that is 47% of our loop, that is the largest actionable item in the
repo. Row E8.

**2. ✅ Smart Turn's Thai failure now has a published cause, and it is not tonality.** Inoue et al.
2024: a monolingual turn-taking model **does not transfer** to another language, while a
multilingual one matches monolingual performance on all of its languages — and Mandarin, tonal, is
in their set. Our 1/4 was predictable. Row E9, amending the tonal-language doubt recorded on
2026-08-18.

**3. ✅ A Thai benchmark now independently supports locked decision 5.** `Thai-Duplex-Bench`
(Popit et al., JCSSE 2026 — same first author as the Thai EOT paper) — 555 items, five
customer-service domains: GPT-Realtime-Mini median TTA **1.5 s**, Gemini Live **4.2 s**, and the
**cascaded reference baseline takes the highest strict pass rate at 97.0%** on service-role
adherence. "Semantic stability remains difficult for current end-to-end speech models." Row D3.

**4. ⚠️ Our 1375 ms is competitive with shipped commercial systems, measured the same way.**
Openbenchmarks times TTFAB as caller-stops-speaking → agent audio starts off the call recording —
**our clock's origin.** Vapi p50 **1558 ms** (382 turns), Retell p50 **1740 ms**. Row S7.
🔴 **Always quote the caveat:** their numbers carry a telephony leg ours excludes, ours is a
quiet-box 22-item lab corpus, and this box drifts 25–45%. The reframe that is fair: the shipped
state of the art on real calls is 1.5–1.7 s, so 500 ms would be ~3x better than anything sold —
not "a cut we might not reach".

**5. 🔴 JaiTTS-v1.0 is released, and it is built on VoxCPM.** CER **1.94% against human ground
truth's 1.98%**; 283 wins / 58 losses in 400 pairwise comparisons vs commercial flagships; handles
numerals and Thai-English code-switching with no text normalisation — the exact class of bug that
broke `cut_clause()` on `09.00 น.`. **Two repo documents are stale**: the memory index says
"unreleased", and `_archive/` investigated a *different*, F5-based JaiTTS. It does **not** fix the
consent problem (still a cloning model, still needs a reference clip). Row T4 — A/B it blinded with
a control before believing any of that.

**6. Row A2 has published recipes now** — chunk-limited attention + mode-consistency regularisation
for RNNT (open-sourced), a 140 ms look-ahead transducer, and an attention-free dual-mode Mamba
encoder. Rows A6–A8.

**Amended, not added:** `LITERATURE.md` §1's claim that "nobody publishes latency from true
end-of-speech" was **too strong and is struck** — TTFAB does exactly that. What survives is the
250–600 ms *budget* vs 1558–1740 ms *measured* gap, and that no such number exists for a
self-hosted Thai stack.

---

## 2026-08-25 — `LITERATURE.md`: the reading gets a ledger, and it starts almost empty

**Why now.** The repo has two good surveys already —
`docs/superpowers/notes/2026-08-17-streaming-thai-asr-spike.md` and
`2026-08-18-endpointing-state-of-the-art.md` — and no index over them. So a paper read in one
session cannot be found in the next, and there is nowhere to put a claim that has not been tested
yet without it looking like a finding.

`LITERATURE.md` (repo root, beside `HANDOFF.md`) is a **claim ledger**, not a survey. One row per
external claim, each carrying the number claimed, a five-value status
(`UNTESTED` / `CONFIRMED` / `REJECTED` / `BLOCKED` / `RULED-OUT`), and the measurement that would
settle it here. Grouped by loop stage. Rejected rows are never deleted, same reason this file
keeps its wrong entries.

**Back-filled 26 rows from what was already measured** — nothing new was read. **11 carry a
`REJECTED` verdict** (2 of them partial: TRT-LLM wins prefill, vLLM wins at concurrency 4), and
**8 of the 11 overturn a published or vendor claim** — Smart Turn on Thai, Typhoon's
`att_context_size`, TRT-LLM FP8 on the hybrid, TensorRT on the DiT, vLLM vs llama.cpp, the clause
cut by ear, Thai fine-tunes at 16 GB. The other 3 kill folklore (Whisper chunking, waveform
metrics, bytes-to-decode proportionality). **That ratio is the argument for the file existing**,
and it is also the warning: this box disagrees with the field more often than not, so an untested
row is worth very little here.

**The queue is ranked by what changes what we build tomorrow, not by interest.** Top item is
NeMo cache-aware streaming fine-tuning (unblocks partials → unblocks prefill-during-speech, which
is currently a no-op); then Voice Activity Projection, the endpointing family we never read after
Smart Turn died zero-shot.

🔴 **One row is a live problem, not a reading task: the 500 ms target is asserted, not sourced.**
The usual citation is a ~200 ms cross-linguistic turn-gap result (Stivers et al., PNAS 2009) —
**whether Thai is in that language set is unverified.** Row E7.

§4 fixes the capture template's one non-optional field: **what a paper measures latency FROM.**
Almost none say it in the abstract, and if they measure from VAD fire their number cannot be
compared to ours at all (our clock's origin is hand-marked true EOS).

---

## 2026-08-18 — the LLM can judge end-of-turn (92%), but 569 ms too slowly

Asked the resident `qwen3.5:9b` one token over 24 cases with known answers:

```
accuracy 22/24    latency median 569 ms   p95 601 ms
  group B 6/6  <- the particle grammar scores 0/6 here
  C-heads 4/6  <- the grammar scores 6/6 here
```

✅ **Accuracy is not the ceiling.** It reads "พรุ่งนี้หน้าร้านเปิดทำการกี่โมง" as a finished
question with no particle -- exactly what `looks_finished` cannot do, zero-shot.

🔴 **569 ms for ONE TOKEN, against a 200 ms window.** And it is the runtime: ollama charges
~370 ms per request on a resident model, and 4b was only 47 ms faster than 9b, so a smaller
model does not help. Each check also occupies the GPU that should be prefilling the reply.

This costs out the classifier swap instead of leaving it hopeful: the paper's fine-tuned
**Qwen3-0.6B at 90 ms** is 6x faster, fits the window, and runs on CPU. A tiered interim
(grammar first, LLM only when it says unfinished) would fire group B at ~+670 ms instead of
never -- trading two group-C saves for six group-B saves. Detail:
`docs/superpowers/notes/2026-08-18-endpointing-state-of-the-art.md`

## 2026-08-18 — a long pause makes Typhoon silently drop the rest of the sentence

Found while testing a cheaper idea (transcribe only the TAIL of a turn, since Thai puts the
turn-end signal at the end). On the longest corpus item the tail and the full buffer disagreed,
and **the full buffer was wrong**: a 10.2 s utterance came back as its first 4 seconds, with
everything after a 3 s pause missing.

Not a length limit -- shorten the gap and the words return. The threshold is sharp:

```
internal silence  200-1200 ms   6/6 transcripts keep the words after the pause
internal silence      1600 ms   5/6
internal silence 2000-3000 ms   4/6
```

🔴 **A correctness bug, and a silent one** -- no error, just a shorter transcript, so the LLM
answers a question whose second half it never got. **And it collides with the semantic
endpointer**, which exists precisely to let callers pause mid-thought: fixing endpointing
without fixing this would have shipped a worse bug than it cured.

Fix: `cap_internal_silence()` in `loop/impl/stt_typhoon.py`, default **800 ms**, applied inside
`_post` so both `finish()` and the endpointer's mid-pause checks get it. Same shape as
`voxcpm/_squeeze.py`, applied to the input instead of the output -- silence carries no words, so
shortening it cannot lose any. **6/6 after the fix, up from 4/6.** Five tests pin that speech
samples are never touched.

⚠️ Measured on synthetic audio with digitally silent gaps; real call silence has room tone.
Re-measure on human audio before trusting the threshold. Detail:
`docs/superpowers/notes/2026-08-18-typhoon-stt-wired.md`

## 2026-08-18 — semantic endpointing works, and the corpus cannot prove it

`loop/impl/endpoint_semantic.py`, config `semantic_vad`, `loop.bench.semantic_sweep`, 9 tests.
Silero at 200 ms proposes, the Thai transcript disposes, `max_wait_ms` commits regardless so it
cannot hang. Transcription starts at silence onset, so it lands before the window expires.

```
 clock@600   median  +615ms   CUT OFF 4/22
 clock@400   median  +420ms   CUT OFF 5/22
  semantic   median +1055ms   CUT OFF 0/22   never fired 20/22
```

**0 cut-offs, all four chronic items carried through their interior pause.** The mechanism is
right. But 20 of 22 never committed, and the diagnosis is the real finding:

🔴 **The transcript ends on a sentence-final particle in only 3 of 22 items** -- because these
clips are cut out of longer scraped speech and END MID-THOUGHT. The corpus measures latency from
a known instant well, and **cannot evaluate turn-taking at all**. That applies retroactively to
every endpointing result taken from it, including the Smart Turn probe's 11/22 at true EOS.

(Artefact worth knowing: `build_corpus.py` pads exactly 1500 ms after true EOS and `max_wait_ms`
defaulted to 1500, so the clip ends on the frame the escape hatch would fire. Default should be
~1200 ms.)

So: the particle grammar is the FLOOR of this approach and it is too low (the Thai paper's
fine-tuned classifier reaches F1 0.881 on the same signal, and `looks_finished` is one injectable
function). **A turn-taking corpus is now the blocker for both remaining routes** -- judging
semantic endpointing and fine-tuning Smart Turn for Thai both need audio whose items are real
turns. YODAS-th subtitle boundaries give exactly that, for free. Detail:
`docs/superpowers/notes/2026-08-18-endpointing-state-of-the-art.md`

## 2026-08-18 — Smart Turn v3.2 does not work on Thai zero-shot

`loop/bench/smartturn_probe.py`, 22 items, against ground truth we already own:

```
at true EOS       said complete    11/22
at an early fire  said incomplete   1/4    misses were CONFIDENT: 0.873, 0.897, 0.979
```

On three of the four chronic items it asserted "complete" where the caller kept talking for
another 1.6-2.2 s. Probabilities are polarised, so it is not hedging on Thai -- it is confidently
wrong. **Zero-shot adoption is out.**

Caveat in the model's favour: our corpus is scraped speech, not conversational turn-taking, so
"true EOS" need not be a prosodic turn end -- that inflates the 11/22. It does not touch the
early-fire result, where the speaker demonstrably continues.

The audio route therefore needs a **Thai fine-tune** (train.py + public datasets + YODAS-th
labels from subtitle timings) -- a project, not an afternoon. **Route C (Typhoon's ~100 ms flush
at candidate endpoints, overlapped with the window, judged by the particle grammar already in
`loop/clause.py`) is untouched by this result and is now the cheapest next move.** Detail:
`docs/superpowers/notes/2026-08-18-endpointing-state-of-the-art.md`

## 2026-08-18 — the 600 ms endpointer already cuts off 18% of callers

`loop/bench/endpoint_sweep.py` replays every corpus item at eight redemption windows against
the hand-marked `true_eos_ms`. Silero's per-frame probabilities are cached and only the decision
logic is replayed, so every window sees identical model output.

```
window   median latency   cut off   of which mild (<500ms)
 150ms       180ms          8/22            2
 300ms       320ms          6/22            1
 400ms       420ms          5/22            1
 600ms       615ms          4/22            0     <- today
```

🔴 **Four items are cut at EVERY window, by 1.6-4.5 seconds.** Their interior pauses exceed any
window worth having (the worst would need 4.5 s of tolerance). So interrupting the caller is not
a risk of shortening the window -- **it is the current behaviour at 600 ms, on 18% of the corpus,
and it had never been reported.**

Against that floor the window buys little: **600 -> 400 ms costs exactly one more item, clipped
by 200 ms, and saves ~195 ms on every turn.** 300 ms is where it turns bad (the next item loses
1.5 s). **Recommended 400 ms**, to be adopted with the run that measures it end to end.

🔴 **Harness hole, now fixed.** When the endpointer fires before the marked EOS, `Session.run`
breaks out of the loop, `true_eos` is never marked, and the fallback silently re-origins the
clock to the early fire -- so a truncated turn produced a normal-looking record with no fault.
One of the six items the bench runs by default is cut by 1.6 s, and every table printed so far
included it silently. `premature_endpoint` is now a fault kind and reports how early it fired.

This turns the case for the limited-context ASR fine-tune from an architectural preference into a
measured one: the four chronic items are exactly what semantic endpointing exists for, and no
amount of window tuning reaches them. Detail:
`docs/superpowers/notes/2026-08-18-endpointing-window-sweep.md`

## 2026-08-17 — no streaming Thai ASR exists

`typhoon-asr-realtime` is the best Thai ASR measured — **CER 0.032 vs Whisper's 0.130**, and
~2× faster on **CPU** than Whisper on the GPU. But its released checkpoint is **full-context**
(`att_context_size [-1,-1]`, `att_context_style regular`); NeMo cache-aware streaming needs
`chunked_limited`. **The model card says streaming, the config says otherwise, and only the
config is evidence.**

Also: **Whisper code-switches out of Thai.** On an 18 s probe with `language="th"` set it
emitted Japanese, Vietnamese and Russian inside one Thai sentence (CER 0.762). Typhoon on the
same audio: clean Thai, 0.084.

Consequence: the budget's 60 ms `stt_flush` line is fiction; read it as ~145 ms. Fixing it is a
limited-context fine-tune — the same shape of work as the RSFT loop.

---
