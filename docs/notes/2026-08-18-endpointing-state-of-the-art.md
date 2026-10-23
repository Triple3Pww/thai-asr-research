# How to actually fix endpointing — a survey, and what it changes

**Date:** 2026-08-18
**Follows:** `E:/Claude/visualllm-business/docs/superpowers/notes/2026-08-18-endpointing-window-sweep.md`, which measured that the inherited 600 ms
window cuts off 18% of corpus utterances and that no window setting reaches them.

⚠️ **EVERYTHING BELOW IS CLAIMS FROM PAPERS AND VENDOR BLOGS. None of it is measured on this
box.** This repo has been burned once by a research sweep that confidently inverted its own
ear-verified findings, and once by a model card that said "streaming" over a full-context
checkpoint. Treat every number here as a lead, not a fact. The recommended first move below is
deliberately an experiment, not an adoption.

## The field has moved, and this repo was solving it the expensive way

"Wait for silence" is the 2023 answer. Production voice agents now run a **turn-detection model**
alongside the VAD, and they split into two families:

| family | input | needs streaming ASR? | examples |
|---|---|---|---|
| **text / semantic EOU** | the partial transcript | **yes** | LiveKit turn-detector, TEN Turn Detection, Turnsense, the Thai paper below |
| **audio / acoustic EOU** | the raw waveform | **NO** | Pipecat **Smart Turn v3.x**, VAP-style models |

The plan of record here — fine-tune Typhoon to limited-context attention so it emits partials —
buys into the *first* family, and it is the most expensive item on the roadmap. **The second
family sidesteps the blocker entirely**, and that is the finding worth acting on.

## Smart Turn v3.x — the interesting one

Pipecat/Daily's open-source turn detector, current as of the v3.2 release.

- **Audio only.** It reports whether a speaker has finished by analysing the raw waveform, not
  the transcript. No ASR partials required — the exact constraint that blocks us.
- **Whisper Tiny encoder + a linear classifier, ~8M params.** 8 MB int8 ONNX / 32 MB fp32.
- **12 ms CPU inference**, claimed. Against a 643 ms stage, that is free.
- **BSD-2-Clause**, weights + training code (`train.py`) + datasets all public
  (`pipecat-ai/smart-turn-data-v3.2-train` / `-test` on HF).
- Input contract: **16 kHz mono PCM, up to 8 s**, one call to `predict_endpoint()`.
- v3.2 specifically improved **short utterances** (miscategorised 40% less often) and **noisy
  audio** — both directly relevant to a phone line, and to Thai replies that are one particle long.

🔴 **Thai is NOT in its 23 languages** (Arabic, Bengali, Chinese, Danish, Dutch, German, English,
Finnish, French, Hindi, Indonesian, Italian, Japanese, Korean, Marathi, Norwegian, Polish,
Portuguese, Russian, Spanish, Turkish, Ukrainian, Vietnamese). Verify that list from the model
card before relying on it.

**But the base encoder is Whisper Tiny, which was trained on Thai audio**, so the representation
exists even though the classifier never saw Thai turn boundaries. Whether that generalises is an
empirical question this repo can answer in an afternoon — see the recommended experiment.

### How it actually works (read from `inference.py` and the README, not the blog)

```
16 kHz mono PCM of the WHOLE turn so far
   -> truncate to the LAST 8 seconds
   -> Whisper feature extractor -> log-mel, padded to exactly 8 s with the ZEROS AT THE FRONT
      and the audio at the end, do_normalize=True
   -> Whisper Tiny encoder -> a single linear classifier -> sigmoid
   -> P(turn complete);  prediction = 1 if p > 0.5
```

**It does not replace the VAD, it adjudicates for it.** Silero keeps its job of noticing that
speech stopped; Smart Turn only runs *during silence*, and answers the question the VAD cannot:
was that a finished thought or a breath? If the caller resumes mid-inference, you re-run on the
**entire turn** — it needs full context and is explicitly "not designed to run on very short
audio segments". Audio from previous turns is not included.

So the cost lands where we were already waiting: nothing during speech, ~10-65 ms inside a pause.
That fits `SileroEndpointer` as an adjudicator behind its existing fire, which is a small change
to one stage rather than a new one.

What it reads is **prosody** — final lengthening, whether pitch falls or stays level, the timing
signature of a hesitation. Their own test pair makes the target behaviour concrete: *"I can't
seem to, um ..."* (incomplete) vs *"I can't seem to, um, find the return label."* (complete).
Same words, same filler, opposite answer — and no window setting can separate them.

⚠️ **The specific doubt for Thai: it is a tonal language**, so pitch contour carries lexical
meaning rather than only phrasing, and that is the cue this model leans on. Cutting the other
way, **Chinese and Vietnamese are both in its 23 languages and both are tonal**. Which effect
wins is exactly what the zero-shot experiment below settles.

Roadmap item worth tracking for a call-centre agent: **text conditioning**, so the model can be
told the caller is reciting a phone number or an address -- the case where a caller pauses
between digits and gets cut off.

## The Thai paper — the same idea, from the text side

**"Thai Semantic End-of-Turn Detection for Real-Time Voice Agents"**, Popit, Rungseesiripak,
Charattrakool, Ruangtanusak (arXiv 2510.04016). *To their knowledge the first systematic study of
Thai EOT.* It validates the exact hypothesis this repo has been reasoning from — **Thai
sentence-final particles are a strong turn-end cue** — and prices it:

| model | F1 | accuracy | latency | size |
|---|---|---|---|---|
| fine-tuned Llama3.2-Typhoon2-1B | **0.881** | 0.874 | 0.11 s | 1B |
| fine-tuned Qwen3-0.6B | 0.866 | 0.861 | 0.09 s | 0.6B |
| fine-tuned mDeBERTa-v3-base | 0.861 | 0.855 | 0.29 s | 276M |

Two things to take from it beyond the numbers:

1. **The training data is public and needs no human labelling.** They built it from **YODAS**
   (a YouTube subtitle corpus) Thai subtitles — subtitle boundaries *are* utterance boundaries, so
   positive and negative examples fall out of the timings automatically. That same corpus feeds an
   audio-side fine-tune just as well.
2. **No code or weights are announced.** The paper is a method and a baseline, not a download.

## The others, and why they are not it

- **LiveKit turn-detector** — 0.1B distilled from Qwen2.5-7B, CPU, strong numbers (9.9% false
  cutoffs at a 300 ms budget vs Deepgram Flux's 12.9%). **Text-only, 14 languages, no Thai**, and
  a proprietary LiveKit Model License. Wrong on all three counts for us.
- **TEN Turn Detection / Turnsense** — text-side EOU models; same partials dependency.
- **Deepgram Flux and other cloud ASR with built-in turn detection** — ruled out by spec §2 D2
  (no cloud hop inside the loop), regardless of quality.

## What this changes

**The limited-context Typhoon fine-tune is no longer the only path to semantic endpointing, and
it is the most expensive one.** Three routes now exist, and they are not mutually exclusive:

**A. Shorten the window to 400 ms.** Already measured: −195 ms, one extra mild clip in 22. Do it
regardless; it is a config change and it is orthogonal to everything below.

**B. Audio-side turn detection (Smart Turn v3.x).** No ASR involvement at all, 12 ms CPU, and the
GPU never sees it. If it works on Thai — zero-shot or after a fine-tune on YODAS-th — it attacks
the 18% cut-off floor *and* lets the window shrink further, because the model, not the clock,
decides.

**C. Transcript-side EOT using the ASR we already have.** ⭐ This one is available **today**, with
no new model: **Typhoon's whole-utterance flush is ~100 ms on the CPU.** So at each candidate
endpoint, transcribe the buffer so far and ask whether it ends like a finished Thai sentence —
the particle grammar in `loop/clause.py` already encodes that, and the paper says a small
fine-tuned classifier does it at F1 0.88 if the grammar is not enough. Crucially the check can be
**started at silence onset and overlapped with the redemption window**, so on a normal turn it
costs nothing: by the time the window expires, the verdict is already in hand.

The old note said "chunking buys nothing" — that was measured on **Whisper**, which re-transcribes
at 300 ms and keeps no state. At Typhoon's ~100 ms the arithmetic is different, and that
conclusion should not be carried over unexamined.

## Recommended first move — an afternoon, CPU only, decisive either way

**Run Smart Turn v3.2 zero-shot against the corpus, at the exact frames where Silero fires.**

The infrastructure exists: `endpoint_sweep.py` already knows every item's hand-marked EOS and
every window's firing frame, and the four chronic items give a ready-made hard set. For each
firing point, feed the preceding ≤8 s of 16 kHz audio to `predict_endpoint()` and ask whether it
agrees the turn is over.

- **If it says "not finished" on the four chronic items** — an 8 MB model on the CPU fixes what no
  window can, today, with no training and no GPU.
- **If it fails** — that is a measured answer about Thai generalisation, it costs one afternoon,
  and route B becomes a fine-tune with a public data path (YODAS-th) rather than a guess. Route C
  remains available regardless.

Either way the result is worth more than more reasoning about it. **Do not adopt anything in this
note on the strength of the note.**

## RESULT: Smart Turn v3.2 zero-shot on Thai FAILS (measured 2026-08-18)

Run, not argued -- `loop/bench/smartturn_probe.py`, 22 corpus items, int8 CPU model:

```
at true EOS       said complete    11/22    coin flip
at an early fire  said incomplete   1/4     and the 3 misses were CONFIDENT: 0.873, 0.897, 0.979
```

On three of the four chronic items it asserted "turn complete" at a point where the caller went
on speaking for another 1.6-2.2 s. Its probabilities are polarised (0.006-0.984), so it is not
uncertain about Thai -- it is confidently wrong, which is the worse failure and rules out
zero-shot adoption outright.

⚠️ **Caveat on the 11/22, in fairness to the model:** the corpus is scraped Thai speech, not
conversational turn-taking, so "true EOS" is where the clip's speech ends and need not be a
prosodic turn end. That inflates the miss rate at EOS. **It does not explain the early-fire
result** -- there the speaker demonstrably continues, and the model said finished anyway.

**Verdict: the audio route needs a Thai fine-tune, not a download.** The path is known and cheap
by this repo's standards (`train.py`, public datasets, YODAS-th for labels that fall out of
subtitle timings) -- but it is now a project, not an afternoon. **Route C is unblocked and
unaffected**, and is the cheaper next move.

## ROUTE C IS BUILT — and the corpus cannot validate it (measured 2026-08-18)

`loop/impl/endpoint_semantic.py` + config `semantic_vad` + `loop/bench/semantic_sweep.py`.
Silero at a 200 ms candidate window proposes; the Thai transcript disposes; `max_wait_ms`
commits regardless so it can never hang. Transcription starts at silence onset, so on a normal
turn it finishes before the window does. 9 unit tests over the decision logic.

Replayed over all 22 items against the same hand-marked EOS:

```
   clock@600   median  +615ms   CUT OFF 4/22   worst -4510ms
   clock@400   median  +420ms   CUT OFF 5/22   worst -4710ms
    semantic   median +1055ms   CUT OFF 0/22   never fired 20/22   checks med 1 max 2
```

**Zero cut-offs — including all four chronic items.** That is the mechanism working: every one
of them was carried through its interior pause instead of being severed. But 20 of 22 never
committed, and the diagnosis kills the celebration:

🔴 **The transcript ends on a Thai sentence-final particle in only 3 of 22 items.** Not a bug in
`looks_finished` — look at what the ASR returns at true EOS:

```
...เพราะเธออยากให้เรา          ...ถ้าสมมุติว่าเธอไม่ทําจริงแล้ว
...เขาไม่ได้ชอบให้ผู้ชายเลี้ยงเรา   ...หนูไม่ค่อยรู้จักเขามาก่อนเลย
```

**These utterances do not end. They are clipped out of longer scraped speech.** The corpus was
built to measure latency from a known instant, and it does that well — but its items are not
turns, so **it cannot evaluate turn-taking at all**, for this route or any other. That limitation
now applies retroactively to every endpointing conclusion drawn from it, including the
Smart Turn probe's 11/22 at true EOS.

(Also artefactual: `never fired` is inflated because `build_corpus.py` pads exactly 1500 ms after
true EOS and `max_wait_ms` defaulted to 1500 — the clip ends on the same frame the escape hatch
would fire. A production `max_wait` must sit below the audio available, and the sweep's default
should drop to ~1200 ms.)

**What this does and does not settle:**

- ✅ The mechanism is right: reading the words carries a caller through a pause the clock cuts.
  0/22 cut off vs the clock's 4-5.
- ❌ A **particle grammar alone is too narrow** — the floor of this approach, and the floor is
  low. arXiv 2510.04016 reports F1 0.881 for a fine-tuned classifier on the same signal, which
  is the ceiling, and `looks_finished` is a single injectable function so that swap costs
  nothing structurally.
- 🔴 **A turn-taking corpus is now the blocker for BOTH remaining routes.** Judging semantic
  endpointing, or fine-tuning Smart Turn for Thai, both need audio whose items are real turns
  with real ends. That is the next piece of work, and it is the same YODAS-th subtitle-boundary
  trick the Thai paper used -- boundaries come free from the timings.

## THE TTS TURN-TAKING CORPUS, AND WHAT IT SETTLES (2026-08-18)

`loop/bench/build_corpus_tts.py` -> `loop/bench/corpus_tts/`, 18 synthesised items whose turn
ends are known by construction. Thai drafted with the `gemini` skill (caller-side, spoken
register), three groups of six:

| group | what it is | what an endpointer should do |
|---|---|---|
| **A** | ends on a sentence-final particle | fire fast |
| **B** | finished, but **no particle** | fire fast -- the grammar's blind spot, included on purpose |
| **C** | **pauses mid-thought 1.2-3.0 s**, then continues | hold on, then fire at the real end |

⚠️ **Synthetic, and it cannot prove endpointing works on real callers** -- the mid-thought
pauses were inserted by us at lengths we chose, so anything tested against them is tested
against our own assumption. It is a positive control and a regression test. The full caveat is
in the builder's docstring, and it applies to every number below.

### The result

```
                          group A        group B          group C (the chronic case)
clock@600                 +640           +640             CUT OFF 6/6, by 2.4-6.3 s
clock@400                 +440           +440             CUT OFF 6/6, by 2.6-6.5 s
semantic, max_wait 1200   +240 .. +290   +1240 fallback   CUT OFF 6/6
semantic, max_wait 3500   +240 .. +290   never fires      SAVED 6/6, fires +400 .. +600
```

✅ **The mechanism works, and this is the first clean evidence of it.** On group A the semantic
endpointer fires **400 ms earlier than the inherited clock** and correctly. On group C with a
long enough max_wait, **all six callers are carried through a 1.2-3.0 s mid-thought pause and
the turn ends ~400-600 ms after they actually finish** -- where every clock setting severed them
by seconds. That is the failure mode no window can reach, fixed.

🔴 **And the grammar's ceiling is now a measured number, not a worry.** `looks_finished` cannot
distinguish group B (finished, no particle) from group C (genuinely unfinished): both look
unfinished, so ONE timer cannot serve both. Short max_wait rescues B and severs C; long max_wait
rescues C and strands B. **That is not a tuning problem, it is the grammar being too blunt** --
exactly the gap arXiv 2510.04016's fine-tuned classifier closes at F1 0.881, and `looks_finished`
is a single injectable function so the swap costs nothing structurally.

### A harness bug this found, worth more than the run

The first pass reported `semantic` firing at exactly max_wait on all 18 items. Cause: **a replay
feeds 20 ms frames in microseconds**, so the 200 ms window expired long before a ~100 ms
transcription could return, the verdict was always `None`, and the endpointer silently degraded
into a bare max_wait timer. It was measuring nothing.

Fixed with `wait_for_verdict=True`, set by the sweep only: it blocks at the candidate until the
transcript lands, which is what happens in real time anyway (the check starts at silence onset
and finishes inside the window). **The live config must never set it.** A replay that is faster
than real time does not merely mis-time an async design -- it can disable it, silently.

### What to do next, in order

1. **Swap the particle grammar for a classifier.** It is the one change that separates B from C,
   it is one function, and the paper prices it at F1 0.881. Needs Thai EOT training data.
2. **Two timers, not one**, as an interim: fire fast when the transcript looks finished; extend
   only when it looks unfinished. Cheap, and it recovers most of B without losing C.
3. **Then re-measure on human audio.** Everything above is synthetic and stays provisional
   until it is not.

## Can the LLM we already have judge end-of-turn? Yes, but not in time (2026-08-18)

Asked `qwen3.5:9b` through ollama, one token, `think: False`, temperature 0, over the 18 TTS
items plus the 6 group-C heads (24 cases with a known answer):

```
accuracy 22/24 (92%)      latency median 569 ms   p95 601 ms   (first call 9205 ms, cold)
  group A  6/6      group B  6/6  <- the particle grammar scores 0/6 here
  group C  6/6      C-heads  4/6  <- the grammar scores 6/6 here
```

✅ **The task is not hard for a language model, zero-shot.** It correctly reads
"พรุ่งนี้หน้าร้านเปิดทำการกี่โมง" as a finished question with no particle -- precisely the case
`looks_finished` cannot see. That is the grammar's blind spot closed with no training at all.

🔴 **569 ms for ONE TOKEN is the blocker, and it is the runtime, not the task.** The candidate
window is 200 ms, so even started at silence onset this adds ~470 ms. And we already know the
cause: ollama charges ~370 ms per request on an already-resident model, and `qwen3.5:4b` measured
only 47 ms faster than `9b` -- **a smaller model does not fix it, because the tax is not the
model.** Two further costs: each check occupies the GPU that should be prefilling the reply, and
group C needed 8-17 checks per turn.

**What it settles:** the classifier swap is now costed rather than hoped for. A general LLM gets
~92% zero-shot; the ceiling is not accuracy. arXiv 2510.04016's fine-tuned **Qwen3-0.6B at 90 ms**
is 6x faster than what we measured, fits inside the window, and runs on CPU where it never
competes with the reply. That is the shape to build.

**A tiered interim, available now:** grammar first (free, catches group A), LLM only when the
grammar says unfinished. Group B would fire at ~+670 ms instead of never -- about clock speed,
but without the clock's cut-off risk. It costs two group-C saves to buy six group-B saves, so it
is a real trade and not a free win.

## Sources

- Smart Turn v3 announcement — https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/
- Smart Turn v3.2 — https://www.daily.co/blog/smart-turn-v3-2-handling-noisy-environments-and-short-responses/
- Model card — https://huggingface.co/pipecat-ai/smart-turn-v3
- Training code — https://github.com/pipecat-ai/smart-turn
- Thai EOT paper — https://arxiv.org/abs/2510.04016 and https://huggingface.co/papers/2510.04016
- LiveKit turn-detector — https://huggingface.co/livekit/turn-detector
- LiveKit on turn detection — https://livekit.com/blog/turn-detection-voice-agents-vad-endpointing-model-based-detection
