# asr-research — Thai ASR accuracy

**Self-contained since 2026-09-16.** Docs, literature, measured results **and the code** live here.
Nothing in this repo depends on `E:/Claude/visualllm-business`.

> 🔄 **As of 2026-10-05, models HAVE been trained** — ASR students (all no gain, `F28`) and four
> N-best pickers (`F29`). Read `RESEARCH-PROGRESS.md` (§00, §0) before quoting anything as a result.
>
> 🔄 **2026-10-06 → 08 ("DeepSeek for Thai ASR", `F32`–`F37`, `NARRATIVE_REPORT.md`):** NECTEC was
> fine-tuned on LOTUSDIS (and once on LOTUSDIS + synthetic — **the synthetic corpus has now trained an
> ASR, with no gain on real speech**), and CTC heads were trained on its frozen encoder.
>
> 🔄 **2026-10-08 → 10 ("invent an ASR from zero" → "reach CER 0.060", `F38`–`F46`):** an Ear–Memory–Pen
> design was proposed and **falsified** (`F38`/`F39`); six real-time no-list ideas lost (`F40`); NECTEC-drafted
> speculative Whispers + ROVER reached −6.2% at 481 ms (`F41`); three trained judges tied NECTEC (`F42`–`F44`);
> ⭐ **fine-tuning NECTEC on fresh, convention-matched GigaSpeech2 (+ LOTUSDIS) is the first real-time win on
> held-out benchmark sessions: dev half −6.8%, TVSpeech −9.5 to −12.1%** (`F45`/`F46`). The benchmark's
> **test half is locked and still unscored** (`results/split_dev_test.json`, commit `494d951`).
>
> 🔄 **THE 5,480-CLIP CORPUS EXISTS as of 2026-09-18** — 10.57 h of audio, 0 silent, 97.9% past
> the cheap checks, on `E:/thai-synth-corpus/` (this said `D:` until 2026-10-05; it is not there).

⚠️ **The live voice loop starts its ASR from here now.** `asr/run.ps1` serves `:8007`, which
`E:/Claude/visualllm-business` calls over HTTP. That is the one direction of dependency and it
runs product → research, never the other way.

---

## Read in this order

🔄 **This repo is evidence only.** The forward-looking material — `RESEARCH-PLAN.md`, the ASR
`HANDOFF.md`, `docs/specs/` and `docs/plans/`, the pending to-do list and the reading queue — was
removed on 2026-09-15 at his request. None of it is lost: `git show 4734fc5:<path>` brings any of it
back, and `git show 4734fc5 --stat` lists the lot. What is here is what was **found and measured**.

| file | what it is |
|---|---|
| **`RESEARCH-FINDINGS.md`** | the synthesis — the things that changed what we would do. **Start here.** |
| **`idea-stage/IDEA_REPORT_2026-10-08.md`** | the 2026-10-08 → 10 line: the error anatomy, the Ear–Memory–Pen design, why it failed, and the loop that followed (`F38`–`F46`) |
| **`NARRATIVE_REPORT.md`** | the 2026-10-06 → 08 "cheap model like the high-end one" run, told once, with what failed — the shortest way into `F32`–`F37` |
| **`RESEARCH-PROGRESS.md`** | the honest state: what was measured and what was built, with the harness defects found underneath |
| `LITERATURE.md` | the claim ledger — every external claim that could change this work, each with a status saying whether *this box* has tested it. 🔴 **A row being written is not a finding** |
| `HARNESS.md` | the `adapter/` experiment harness: what each arm script does and how to run it |
| `HISTORY.md` | append-only log, newest first, **including the claims later disproved** |
| `docs/notes/` | the six deep survey notes — streaming Thai ASR ×2, endpointing state of the art, the endpointing instrument, the TTS-for-ASR survey, the 145-paper bibliography |
| `literature-review/` | the advisor deliverable — the `.docx`, its builder, and the two presentation decks |
| `lessons/` | two interactive walkthroughs (`stt.html`, `adapter.html`) — open in a browser |
| `docs/superpowers/specs/` | the forward-looking designs. ⭐ **`2026-09-19-sensevoice-thai-design.md` §0 is where the next session starts** |
| `results/` | raw arm outputs: `*_metrics.json`, `*_hyps.jsonl`, `*_run.json`, plus the `probe_*` spikes |

## The headline numbers, so nothing has to be re-derived

- ⭐⭐ **The best real-time model (2026-10-09/10, `F45`/`F46`): NECTEC 115M fine-tuned 1 epoch on fresh
  GigaSpeech2-th dev+test (17.9 h, labels rewritten into NECTEC's own convention, Latin-NECTEC clips dropped)
  + convention-matched LOTUSDIS, interpolated α .75 with the base** (`checkpoints/gs2lot_a75.nemo`): benchmark
  **dev half 0.04716 vs 0.05059 (−6.8%, P .013)**; TVSpeech **0.0734 vs 0.0811 (−9.5%, P .0001)**. The GS2-only
  α .5 model (`gs2a_a50.nemo`): dev −5.8% (P .008), **TVSpeech 0.0713 (−12.1%)** — both below
  typhoon-whisper-large-v3 (0.0785). Same 115M, ~43 ms. 🔴 **The test half is unscored**; the target he set,
  CER 0.060 (−13.7%), is **not** reached. Not deployed — `:8007` still serves plain NECTEC.
- 🔴 **What did NOT reach it** (`F38`–`F44`): an explicit tone branch (tone is 8% of NECTEC's error), a
  sound→spelling "Pen" (toned sound fixes Thai spelling on 98.75% of words), catalog name lists (false spots),
  test-time adaptation (worse), step-synchronous RNN-T ensembling (emission timing), LM tie-breaking, sentence
  product-of-experts, and three TRAINED judges — every one within ±3% of NECTEC on the dev half, although a
  per-utterance oracle over three ears is −24.5%. A judge learns its training domain's best ear and carries
  that prior (`F43`/`F44`).
- ⭐ **Large Whispers can be made ~3× faster losslessly**: NECTEC's transcript as the speculative draft for
  Pathumma-/typhoon-whisper-large (`adapter/spec_whisper.py`), 1,030 → 264–297 ms/turn, output identical to
  greedy on 39/40 (`F41`).

- ⭐⭐ **The cheap system (2026-10-08, `F36`/`F37`): NECTEC 115M + a 2.6M CTC head on its frozen
  encoder, one pass, runs at NECTEC's own speed (42.9 ms p50/turn) and, given a 50-name list, beats
  typhoon-asr-qwen-0.6b-ctx (0.78B) on all four metrics** — bias@50 CER 0.0748 vs 0.0791, B-CER 0.214
  vs 0.263, recall 0.567 vs 0.472 (P ≤ 0.0002). Without a list NECTEC ties typhoon-whisper-large-v3 on
  the benchmark **and** TVSpeech; a 6-GPU-minute LOTUSDIS tune is −4.8% / −5.9% on TVSpeech over two
  seeds (borderline, `F33`). **Not deployed yet** — `:8007` serves plain NECTEC.
- 🔴 **Two traps found on the way:** fine-tuning on a corpus teaches its label *convention* (raw
  LOTUSDIS: −25% in domain, **+31% on the benchmark**, `F32`); and typhoon-whisper on one 30 s Thai window
  truncates (TVSpeech 0.2004; 0.0785 with audio cut ≤ 15 s, `F33`).

- ⭐⭐ **The real-time answer (2026-10-05): NECTEC's `Pathumma-stt-th-fastconformer-rnnt-large`
  ties the best Typhoon at real-time speed** — CER **0.0676** at 113 ms/turn vs typhoon-whisper-large-v3
  **0.0735** and typhoon-asr-realtime **0.0945** (682 real utts, canon normaliser). A tie, not a win
  (paired P = 0.084), and worse on names. **Now the `:8007` default** (`./asr/run.ps1 -Model typhoon`
  for the old model). Our own attempts: distillation into a real-time student — no gain in 4 runs
  (`F28`); an N-best picker trained on synthetic audio — best of all on conversation, ties NECTEC
  overall (`F29`). → `RESEARCH-FINDINGS.md` `F27`–`F31`, `idea-stage/IDEA_REPORT.md`

- **Arm 1, Typhoon as shipped:** general CER **0.0518 mean / 0.0000 median** over 150 Common Voice
  clips (`results/arm1_typhoon_shipped.json`). ⚠️ Denoised read speech, no rare words — a baseline
  for later arms on the same harness, **not** a production number.
- **The rare-word gap is field-wide, not our model's defect.** On `thai-contextasr-bench` (682 real
  utts, the benchmark's own scorer) `typhoon-asr-realtime` reads entity recall **0.328**, B-CER
  **0.2618** inside gold-entity spans vs U-CER **0.1057** outside — **2.48×**. Rescoring their six
  published baselines on the same 682: recall 0.253–0.435, and **B-CER is 2.01–3.04× U-CER for
  every model**, across RNNT, CTC and encoder-decoder. **The field's best still misses 57% of named
  entities unbiased.**
- **D6 — a second acoustic opinion — is the arm that worked:** recall **.328 → .554** (+99
  mentions) with the harm term falling, for 14% more compute. Every pure re-weighting arm before it
  moved recall by ~nothing (144 → 143 → 143).
- ⭐⭐ **Emotion transfers to Thai zero-shot — and SenseVoice's refusal is a HEAD problem, not an ear
  problem.** On 1,256 agreement-filtered THAI-SER clips, `emotion2vec_plus_large` scores **0.7381**
  4-way with no Thai training, at or above THAI-SER's own *trained* CNN+LSTM baseline (WA 67.34 +/- 3.05).
  `SenseVoiceSmall` scores 0.4761 **but abstains on 43.6% of clips** and is **84.3% precise when it
  commits**. A linear probe on its own frozen encoder reads SAD at **0.830** where its shipped head
  emits **0.060** — the information is there and the output layer will not say it. Human ceiling on
  the same subset: **0.9286**. → `results/probe_emotion_thai.json`, `results/probe_encoder_emotion.json`
- **Thai SARA AM is worth 26.6% of mean CER** and is a spelling convention, not accuracy. Fold it on
  **both sides** of any Thai CER. `adapter/_thai.py` is the one definition.

## Rules that already cost a session each

- 🔴 **Tune on the dev half, score the test half once** (`F42`): `results/split_dev_test.json` splits the
  682 real utts by recording session (dev 339 / test 343); `adapter/paired_vs.py --half test` refuses without
  `--final` and logs every use. Judge relative cuts — the halves differ in difficulty (NECTEC dev 0.0506).
- 🔴 **Tune with official per-utterance errors, not plain Levenshtein** (`adapter/official_utt.py`): the
  benchmark scorer accepts an entity in Thai OR Latin form; a plain scorer showed −4.4% for a pick that was
  −0.7% officially.
- 🔴 **A judge's (or student's) training data must be fresh to EVERY model involved** (`F43`), and its
  labels rewritten into the base model's convention — including script choice for loanwords (`F45`).
- 🔴 **Dense-text clips at batch 16 overflow 16 GB VRAM and Windows spills it into system RAM** — train
  such data at batch 8 × accum 2; after a low-memory reap, **kill the orphaned python child** (`F46`).

1. **Fold SARA AM on both sides of every Thai CER**, and write training labels **decomposed**. A
   model fine-tuned on composed labels posts a ~26% "win" that is purely a spelling change.
2. **Typhoon writes unspaced Thai number WORDS, never digits** — measured over 53 real turns
   (0 of 53 contain an ASCII digit). Labels and scoring must match that convention.
3. **`:8007` runs on the GPU and always has.** Every "RTF on CPU" figure in the older docs is a GPU
   figure. `ASR_DEVICE=cpu` forces CPU: RTF 0.027 vs 0.010, ~96 ms more per turn, ~0.95 GB of VRAM
   back.
4. **The over-biasing gate in the benchmark's scorer is broken as written** — insertions are charged
   to B on a ≥2-char substring match against the *listed* forms, so growing a bias list deflates
   U-CER for free. **Only compare `bias@N` against `dis@N` at the same N.**
5. **Pin anything that moves a number.** B-CER/U-CER shift 4.3% depending on whether `kaldialign`
   is installed.
6. **This box drifts 25–45% across an evening and not equally between engines.** Bracket everything
   being compared back-to-back in one window; discard the first sweep after any server start.
7. **`thai-contextasr-bench` is CC-BY-NC-4.0** — evaluation and thesis only, never training data or
   product. Real split only; the TTS split makes evaluating a synthetic-data method circular.
8. **Before asserting a Thai speech model or vendor does not exist, say what you checked and ask
   him.** That assumption has been wrong three times. Enumerate the org first:
   `curl -s "https://huggingface.co/api/models?author=typhoon-ai&limit=1000"`, sort by
   `lastModified`. A paper is a snapshot months stale.

## Layout

| dir | what |
|---|---|
| `adapter/` | the experiment harness — arm scripts, the scorer, the Thai normaliser, the text banks |
| `asr/` | the ASR servers: `server.py` (`:8007` — NECTEC FastConformer by default since 2026-10-05, `-Model typhoon` for full-context Typhoon) and `server_streaming.py` (`:8011`), with their `run*.ps1` |
| `idea-stage/` | `IDEA_REPORT.md` — the 2026-10-03→05 "beat Typhoon in real time" line; `IDEA_REPORT_2026-10-06.md` — the 2026-10-06→08 "DeepSeek" line; `IDEA_REPORT_2026-10-08.md` — the 2026-10-08→10 "from zero → 0.060" line; every run and caveat |
| `refine-logs/` | the 2026-10-06 proposal, experiment plan and tracker (pre-registered criteria included, and where they failed) |
| `checkpoints/` | the emotion head, the (no-gain) ASR students, the N-best pickers `picker_p1/p3/p3half/p4`, the LOTUSDIS tunes `rt_student_lot*` / `lot*_a*`, the CTC heads `ctchead_h1/h2/h3/a2h3.pt`; since 2026-10-09 the GigaSpeech2 tunes `rt_student_gs2a` / `gs2a_a*`, `rt_student_gs2lot` / `gs2lot_a*` (⭐ `gs2lot_a75`, `gs2a_a50`), `gs2lotfood_*` (no gain), the syllable head `sylhead_s2.pt` and the judges `judge_v1.pt` / `judge_gs2.pt` / `judge_pooled_noid.pt` (no gain) |
| `colab/` | the Colab TTS generator for synthetic audio — notebook + `gen_colab.py` |
| `results/` | 758 raw outputs (2026-10-10), `<arm>_<condition>_{metrics,hyps,run}.json`; the real-time line adds `*_nbest.jsonl` (top-5 lists), `picker_*`, `nectec_*`, `rover_*`; the 2026-10-08→10 line adds `anatomy_*`, `eval_soundblind_*`, `sylhead_*`, `pen_v0_*`, `loop*_*`, `spec_*`, `fw_*large*`, `poe_*`, `judge_*`, `gs2*_bench`, `gs2_tvspeech_*`, `paired_*_{dev,test}`, `split_dev_test.json`; the DeepSeek line adds `lot*`, `nec_d6*`, `ctchead_*`, `tvspeech_*`, `deepseek_*`, `latency_onepass_*`; `_invalid_ctchead_h3_run1/` is quarantined evidence |
| `docs/notes/` | six deep survey notes |
| `docs/superpowers/specs/` | the design specs |
| `lessons/`, `literature-review/` | two interactive walkthroughs, and the advisor deliverable |

## Running an arm

Start the ASR server, then run the arm. **`results/` is the default output** — no environment
variable needed, because the code and the results are in the same repo now.

```powershell
$py = "E:/asr-spike/.venv/Scripts/python.exe"

./asr/run.ps1                                       # :8007, leave it running

& $py adapter/arm1_baseline.py                                  # general CER, 150 Common Voice clips
& $py adapter/arm1_rareword.py --split real --align dp          # rare-word gate, 682 real utts
& $py adapter/compare_baselines.py --split real                 # rescore the benchmark's 6 baselines
& $py adapter/arm_d4_boosting.py --split real --condition bias@50 --alpha 1 --score 0.5
& $py adapter/arm_d6_splice.py  --mode ctc --condition bias@50 --tau -1.5

& $py adapter/ortho_rescore.py                                  # the 7 systems under orthographic normalisation
& $py adapter/_ortho.py --selftest                              # 26 cases, and two of them are why it exists
```

`ortho_rescore.py` needs **no server and no GPU** — it re-scores hypothesis files already on disk.
It is the one arm that changes the scoring convention rather than the model, so read `F17` and
`adapter/_ortho.py` before quoting it. 🔴 **Every other number in this repo is the unnormalised
`none` rung, and stays that way.**

`ADAPTER_OUT` still overrides the output dir, for a scratch run that must not touch the record.

✅ **All five arms were re-run on 2026-09-16 after the move and reproduced byte-exactly** —
arm1_baseline 16/16 metrics and 150/150 transcripts, arm1_rareword 24/24 and 682/682,
compare_baselines 7/7 systems, plus d4 and d6. The "not run since the split" warning that stood
here is retired.

Rebuilding the text banks (needs the `agy` CLI; `fix_label_convention.py --apply` is **not**
optional polish — a freshly generated bank is in the wrong label convention):

```powershell
& $py adapter/make_text_banks.py --phase 1 --batch 40
& $py adapter/fix_label_convention.py --apply
& $py adapter/check_text_banks.py adapter/texts/phase2_rare.json --samples 12
```

## Environment

- Python for ASR work is **`E:/asr-spike/.venv/Scripts/python.exe`** (NeMo 3.0.0, torch 2.13.0+cu130,
  **CUDA available** — it is not a CPU-only venv). Keep it separate from `E:/voxcpm-spike/.venv`:
  installing NeMo beside VoxCPM risks a torch downgrade that breaks the live TTS.
- **`E:/thainlp-spike/.venv`** (2026-10-08): pythainlp + tltk (Thai G2P with tone), numpy<2 / pandas<3 — text
  only; runs `eval_soundblind.py`, `syllable_labels.py`, `pen_lattice.py`. **`E:/fw-spike/.venv`** runs Whisper
  under CTranslate2 (`bench_fw.py`, `fw_manifest.py`); Pathumma-whisper-large-v3 was converted there on 2026-10-08.
- **New data (2026-10-09/10):** `E:/Claude/datasets/gigaspeech2_th/` (GigaSpeech2-th dev+test, 20 h, CC-BY,
  `manifest_*_conv.jsonl` = NECTEC-convention training labels), `judge/` and `judge_gs2/` (joined ~10 s turns +
  three ears' transcripts and scores), `thai_extra/` (food-ordering 16 kHz copies — fresh but no gain; elderly
  speech — NOT fresh to NECTEC), `pen/` (homophone dictionary, Thai Wikipedia titles).
- GPU: RTX 5060 Ti, 16 GB, and **the box is full on both kinds of memory** when the loop is up.
  Check `nvidia-smi` and the Windows commit limit before assuming a model will load.
