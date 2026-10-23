# Thai ASR Research — making speech recognition work for real Thai conversation

A self-directed research project (Aug–Oct 2026) on **Thai automatic speech recognition (ASR)** for a
real-time voice agent: how accurate can a model be while still answering in tens of milliseconds, and
why do Thai ASR systems miss names and rare words?

Everything here was run on **one consumer GPU (RTX 5060 Ti, 16 GB)**. Every number comes from a script
in this repo and a results file in `results/`. That includes the ideas that failed: those are logged
as carefully as the wins.

## Key results

| Finding | Number |
|---|---|
| A **115M-parameter real-time model** (NECTEC FastConformer-RNNT), fine-tuned on convention-matched GigaSpeech2-th + LOTUSDIS, beats the much larger `typhoon-whisper-large-v3` on TVSpeech | CER **0.0713–0.0734** vs **0.0785**, at ~43 ms/turn |
| Same model on the held-out dev half of `thai-contextasr-bench` | **−6.8 %** CER vs base (paired P = .013) |
| A 2.6M-param CTC head on the frozen encoder + a 50-name list beats a 0.78B context-aware model on all four metrics | bias@50 CER 0.0748 vs 0.0791, entity recall 0.567 vs 0.472 |
| Speculative decoding: the small model drafts for a large Whisper → **3.5–3.9× faster**, same output | 1,030 → 264–297 ms/turn, identical on 39/40 |
| The rare-word gap is field-wide: every published baseline makes **2–3× more errors inside named entities** than outside | B-CER / U-CER = 2.01–3.04× |
| Emotion recognition transfers to Thai zero-shot (`emotion2vec_plus_large`) | 0.738 4-way, ≥ a trained Thai baseline (0.673) |

> **Honest caveats.** The benchmark's **test half is locked and still unscored**: all tuning used the
> dev half only. The self-set target (CER 0.060) was **not** reached.

## What did *not* work (and why that matters)

- Distilling large models into a real-time student: no gain in 4 runs.
- An explicit tone branch, a sound→spelling "Pen" model, test-time adaptation, LM tie-breaking, and three
  trained judges: all within ±3 % of the baseline, even though a per-utterance oracle over three models is
  −24.5 %. A judge learns its training domain's best model and carries that prior.
- Training on a 10.6-hour **synthetic** Thai corpus (5,480 TTS clips) gave no gain on real speech.

## Traps found along the way

- **Spelling conventions can fake a win.** Thai SARA AM accounts for 26.6 % of mean CER. Fine-tuning on
  raw LOTUSDIS looked −25 % in-domain and was **+31 % worse** on the benchmark. It had learned a label
  *convention*, not accuracy. Normalise both sides (`adapter/_thai.py`).
- **Tune on dev, score test once.** The split is by recording session, and `adapter/paired_vs.py` refuses to
  touch the test half without an explicit `--final` flag.
- **Use the benchmark's official scorer.** A plain Levenshtein scorer showed −4.4 % for a change that was
  −0.7 % officially.

## Repo map

| Path | What |
|---|---|
| [`RESEARCH-FINDINGS.md`](RESEARCH-FINDINGS.md) | The synthesis: findings F1–F46. **Start here.** |
| [`NARRATIVE_REPORT.md`](NARRATIVE_REPORT.md) | One experiment line told end to end, failures included |
| [`idea-stage/`](idea-stage) | Idea reports: hypotheses, runs, and why each was kept or killed |
| [`LITERATURE.md`](LITERATURE.md) · [`docs/notes/`](docs/notes) | Claim ledger + six survey notes (streaming Thai ASR, endpointing, a 145-paper bibliography) |
| [`adapter/`](adapter) | Experiment harness: arm scripts, scorer, Thai normaliser, fine-tuning |
| [`asr/`](asr) | The ASR servers (offline + streaming) used by the live voice loop |
| [`results/`](results) | Metrics and run records for every arm |
| [`lessons/`](lessons) | Two interactive HTML walkthroughs of the pipeline |
| [`HISTORY.md`](HISTORY.md) | Append-only lab log, newest first, **including claims later disproved** |
| [`NOTEBOOK.md`](NOTEBOOK.md) | The working lab notebook (internal paths, run commands) |

## Stack

Python · PyTorch · NVIDIA NeMo (FastConformer RNN-T/CTC) · Whisper / CTranslate2 · faster-whisper ·
SenseVoice · emotion2vec · Google Colab (synthetic data generation)

## Notes on data

Model checkpoints, audio, and raw transcripts are **not** included. `thai-contextasr-bench` is
CC-BY-NC-4.0 and was used for evaluation only, never for training. GigaSpeech2 and LOTUSDIS are used under
their own licenses. Only aggregate metrics are published here.

---

Built by **Chanachon Ruanpheng**, ICT student at Mahidol University. Code is MIT-licensed (see `LICENSE`).
