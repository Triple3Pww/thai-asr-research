# Latency in Self-Hosted Thai Voice Agents — the 15-minute cut
### Supervisor meeting, 2026-09-02. 9 slides + 4 backup. Cut from `lit-review-presentation-2026-09-01.md` (12 slides).

> **How to use this file.** One `##` heading per slide. **Bold lines are the slide text**; `Notes:`
> is what to say; `⏱` is the time budget. ⚠️ marks the figures a supervisor is most likely to push
> on — the honest answer is written next to each.
>
> **All numbers measured 2026-09-01, `loop.bench.report`, frozen corpus n=22 per pass, realtime
> replay, origin hand-marked true EOS, config `typhoon_llamacpp_300` (300 ms window, designed voice,
> reference mode) against `typhoon_llamacpp` (600 ms), alternated back-to-back.**
>
> ✅ **Three passes per arm, across two independent windows** (the servers died and were restarted
> between them). 300 ms: WIRE p50 **771.1 / 754.7 / 791.5 ms** — spread **4.9%**. 600 ms:
> **1060.5 / 1097.4 / 1092.6** — **3.5%**. `endpointing` reproduces to **1.1 ms**
> (330.4 / 331.5 / 331.4). **The 771 ms and 474 ms quoted below are medians of three passes.**
> On a box that drifts 25–45% across an evening, that spread is the evidence the table is real.
> Raw records: `loop/bench/_out_pres/`.
>
> **The rule that governs this deck:** every number is either measured on this box by
> `loop.bench.report`, or cited to a paper. **Nothing from the `live/` panel is quoted** — it shares
> the box with the three servers it drives and brackets nothing. The 12-slide version quoted 797 ms
> from the panel; that number is not in this deck.

**9 slides / ~13 min speaking, leaving ~2 min of the slot for questions.**

---

## Slide 1 — Title ⏱ 30 s

**Latency in Self-Hosted Thai Voice Agents**
**A stage-level literature review — and what it does not contain**

Chanachon Ruanpheng · 2026-09-02

Notes: One sentence of framing, then move on. *"I set out to compare my system against published
voice-agent latency numbers, and found that the comparison mostly cannot be made. That absence is
the result."* Do not preview the findings — slide 4 is the payload and it lands harder cold.

---

## Slide 2 — The system under study ⏱ 75 s

**A Thai voice agent that must answer in under 500 ms, entirely self-hosted**

- Audio in → audio out, five stages, one consumer GPU (RTX 5060 Ti, 16 GB)
- Self-hosting is not a preference: a cloud hop inside the loop costs more than the whole budget
- Thai is low-resource — most published components do not support it at all

Notes: The point to land is that **the two constraints are one constraint**. Thai quality and low
latency both force self-hosting, and self-hosting is what makes almost every published number
inapplicable. The 500 ms target comes from human turn-taking research, not from marketing.

⚠️ *"Why not just use a cloud API?"* — the round trip alone exceeds the budget, and Thailand's PDPA
restricts cross-border transfer of call audio for exactly the customers this targets.

---

## Slide 3 — Method ⏱ 90 s

**A claim ledger, not a summary**

- **155 rows**, each one external claim **with a number in it**, plus a status
- `UNTESTED` (the default) · `CONFIRMED` · `REJECTED` · `BLOCKED` · `RULED-OUT`
- **A claim is not a finding.** A row changes status only when a measurement on this hardware moves it
- This sweep: **13 queries → ~150 unique records → 42 rows kept → 15 papers read in full**
- **What was discarded is logged by category**, so the same ground is not searched twice

Notes: This is the slide that earns the right to be believed later. Emphasise that ~110 papers were
deliberately discarded *with reasons recorded*, and that **rejected rows are kept permanently** —
they turn out to be the most useful ones. Two standing discards worth naming aloud: LLM-serving
latency research does not apply at one concurrent call, and there is no transport literature for
voice agents at all (the WebRTC corpus is video-conferencing QoE, a dead end).

---

## Slide 4 — The main result ⏱ 2 min · **the payload**

**Of four published stage breakdowns, NOT ONE includes endpointing**

| paper | stage table? | endpointing line | clock origin stated? |
|---|---|---|---|
| Qiu 2026 (enterprise tutorial) | ✅ | ❌ absent | ❌ |
| Ethiraj 2025 (NetoAI telecom) | ✅ | ❌ absent | partial |
| ChipChat 2025 (Apple, ASRU) | ✅ | ❌ absent (starts at mel) | ❌ |
| X-Talk 2025 | ❌ none | — | states it *excludes* VAD |

> X-Talk says it outright: *"the frontend VAD uses a 500 ms end-of-utterance silence threshold…
> **and this fixed delay is not included in latency calculation**."*

Notes: **Slow down here.** Every published "sub-second cascade" is sub-second on a clock that starts
*after* the endpointer fired. Read the X-Talk quote aloud — it is the strongest single piece of
evidence precisely because they are not hiding it; they simply do not consider it latency.

⚠️ *"Is this just four papers?"* — yes, and say so plainly: these are the four that publish a stage
table at all, which is itself the finding. Out of ~150 records screened, four had a breakdown to
inspect.

---

## Slide 5 — Why that matters here ⏱ 2 min

**Endpointing is our largest single stage — so their clocks start after our biggest cost**

| stage | ms (p50) | who studies it |
|---|---|---|
| **endpointing — deciding the caller stopped** | **330.4** | ASR / turn-taking |
| STT flush | 52.5 | speech recognition |
| LLM first token | 128.9 | inference serving |
| LLM clause generation | 205.5 | inference serving |
| TTS first audio | 77.6 | speech synthesis |

- Each stage is rigorously studied **in isolation, by a different community, in a different unit**
- Four incompatible units for "latency": first-packet/TTFB · p90 emission · *algorithmic* (chunk-derived, not wall-clock) · RTF (throughput — **cannot see first-token at all**)

Notes: The units point explains why papers disagree with each other. Concrete example: i-LAVA
concludes TTS dominates; our table says endpointing does. **Both are right** — they measure RTF over
a whole utterance, we measure time to first audio. A units artefact, not a hardware one.

And the rigour is not missing from the field, only from the whole-loop papers: Bijwadia et al. 2022
defines endpoint latency from **ground-truth end-of-query** — EP50 390 / EP90 740 ms, and
270 / 570 ms for their jointly-trained model at no accuracy cost.

⚠️ *"So your endpointer is slower than Google's."* Yes. But theirs is short-query voice search, where
the caller stops and is done; ours is conversation, where Thai mid-turn and turn-final pauses are
only **41.9%-EER separable by duration** — measured here, 1,137 / 396 pauses, LOTUSDIS.

---

## Slide 6 — Our measurements, beside theirs ⏱ 2 min

**On the span they actually measure, we are comparable — and the comparison had to be constructed**

| | **ours** | Qiu | NetoAI | ChipChat |
|---|---|---|---|---|
| total as published | **771 ms** | 755 ms | 934 ms | ~920 ms |
| **post-endpoint (comparable span)** | **474 ms** | 755 ms | 678 ms | ~920 ms |
| clock origin | **hand-marked true EOS** | not stated | "per utterance" | not stated |
| hardware | RTX 5060 Ti | cloud STT/TTS | not stated | Mac Studio M2 Ultra |

Notes: **Say the caveat before she asks for it.** This is reported-vs-reported across different
hardware, languages and workloads — positioning, not a controlled experiment. Nobody else's hardware
was bracketed against ours, and it could not be.

⚠️ **The honest half:** a shorter endpointing window buys speed by cutting callers off earlier — in
this same run the endpointer fired early on **7 of 22 utterances at 300 ms against 4 of 22 at
600 ms**. *A lazier endpointer always
scores better on this metric.* That is exactly why the clock must start at hand-marked true
end-of-speech, and why both windows were measured back-to-back in one session rather than quoted
from different evenings.

---

## Slide 7 — What I am building now ⏱ 60 s

**Training a Thai turn detector, because the zero-shot one fails**

- The published audio turn-detector (Smart Turn v3.2) is **wrong on Thai, and confidently** — p = 0.87–0.98 on speakers who continued for another 1.6–2.2 s
- Its own training set is **82.7% synthetic — in train *and* test**, so its published accuracy was never calibrated against human speech
- Built a Thai fine-tune: 1,220 LLM-drafted caller utterances spoken by our own TTS, speaker- **and** text-disjoint split. The **23% particle-final ratio is measured**, on 4,688 real Thai floor changes — the data is synthetic, the *distribution* is not
- **First runs regressed below their own control. Diagnosing — not yet a result.**

Notes: Keep this short and do not oversell it. Say plainly that the first fine-tunes came out
**worse than a linear probe on frozen features**, which the evaluation script itself defines as
*"something is wrong, not that the idea failed"* — so we cannot yet separate *synthetic Thai does
not transfer* from *the pipeline has a defect*.

⚠️ **Be precise about "real" if she asks — the training clips are entirely synthetic** (LLM text,
our TTS voicing it). The only measured-from-real quantity is the particle ratio. Claiming real
caller data here would be false, and it is also the point: **our training set has the same
synthetic-data weakness we just criticised in the published model.** Say that before she finds it.

The reason it is in the talk at all is the tie back to slide 4: **the same synthetic-data blind spot
found in the literature was then hit in our own work.** If she asks for numbers, that is backup
slide C — do not put the EER table on screen unprompted.

---

## Slide 8 — Limitations, stated first ⏱ 90 s

**What these numbers cannot support**

- n = 22 utterances per pass on a **frozen, scraped** corpus — not a standard benchmark, so the absolute values are ours alone
- Cross-system comparison is **reported-vs-reported** — never bracketed against their hardware
- **This machine drifts 25–45% across an evening.** Only same-window comparisons are quoted
- Our test corpus is **2–3× easier** than the ASR model's own benchmark (our CER 3.2% vs their published 6.81–10.65%) — so accuracy figures here are optimistic
- Synthetic and scraped audio, not real telephone calls

Notes: Presenting limitations before being asked is the single strongest move available in this
meeting. **The drift number is worth saying aloud** — it invalidates most casually-collected
benchmark tables, including two of my own earlier ones, and it is the reason slide 6 was run as an
alternating bracketed pair in one window instead of two runs on different nights.

---

## Slide 9 — Where this goes ⏱ 90 s

**Next — and one question for you**

- **Publishable now:** the endpointing-measurement gap, and the first per-stage table for a self-hosted Thai loop measured from hand-marked true end-of-speech
- **Endpoint anticipation** — forecasting turn-end rather than detecting it — reports **−505 ms**, the largest published win against our largest stage
- **Streaming Thai ASR does not exist.** It is the blocker for the overlapped architecture, and it is a limited-context fine-tune, not a download
- **The voice no longer has a consent problem** — a described voice, cloned from one generated sample, working in Thai

> **Is the endpointing-measurement gap worth a short paper on its own?**

Notes: Close on the question, not on the list. It is the one thing only she can answer, the data
already exists, and if she says yes the next meeting is about venue and scope — a much better
meeting than another status update.

---
---

# Backup slides — do not present; answer *from* them

## Backup A — the four incompatible units, in full

first-packet / TTFB (TTS) · p90 emission latency (streaming ASR) · algorithmic latency
(chunk-derived, **not** wall-clock) · RTF (throughput; structurally blind to first-token).
Two papers disagree about which stage dominates purely because of this.

## Backup B — the Thai last mile

**Model cards claiming a language are not evidence.**

- Of five published streaming-TTS systems at 54–102 ms first-packet, **four cannot speak Thai**; the fifth is **2.5× worse** at Thai than the model already in use
- Independent benchmark, Thai row (WER / speaker-similarity): MiniMax **2.70 / 80.0** · **VoxCPM 2 — 2.96 / 84.0** (best open system) · Fish-Audio **4.23** · ElevenLabs **73.94 / 58.8**
- Voice-design instructions are obeyed in English and Chinese; a **Thai** instruction is read aloud verbatim (4 phrasings, 4 failures)

⚠️ ElevenLabs at 73.94 WER is the memorable number, but **do not generalise it to "cloud TTS fails at
Thai"** — MiniMax is cloud and tops the row. The correct claim is that *English-centric vendors* fail
Thai.

## Backup C — the Smart Turn fine-tune, with numbers

```
pause duration (baseline)      41.0% EER
Smart Turn v3.2 zero-shot      41.8%
frozen linear probe, last 1 s  34.5%   <- the control
our fine-tunes            42.5-49.5%   <- WORSE than the control, all four runs
target                        <=15%
```

Two dataset versions; v2 (3,188 texts, the hesitation written into the text) came out **worse** than
v1 (1,220 texts). Both banks are LLM-drafted and voiced by our own TTS — **fully synthetic
training audio**. Evaluated on 400 **real** Thai pauses, session-disjoint folds, LOTUSDIS.

⚠️ Far-field 3-party meeting audio is close to the worst case for this question — every number is
pessimistic by an unknown amount, and is used for **comparison between models on identical clips,
never as an absolute**.

## Backup D — every figure, mapped to its source

| claim | number | source |
|---|---|---|
| our loop, end to end | 771 ms p50, from hand-marked true EOS | `loop.bench.report`, bracketed pair 2026-09-01 |
| Google production endpointer | EP50 390 / EP90 740 ms | Bijwadia 2022, SLT (arXiv 2211.00786) |
| jointly-trained endpointer | EP50 270 / EP90 570 ms, no WER cost | same |
| Hinglish EOS detection | 2858 → 1525 ms; noisy 5080 → 3096 | Goyal 2023 (arXiv 2305.18596) |
| enterprise cascade | STT 337 / LLM 337 / TTS 219; 755 ms | Qiu 2026 (arXiv 2603.05413) |
| telecom cascade | ASR 49 / LLM 670 / TTS 286; 934 ms | Ethiraj 2025 (arXiv 2508.04721) |
| on-device cascade | ~920 ms cumulative, M2 Ultra | ChipChat, ASRU 2025 (arXiv 2509.00078) |
| VAD excluded by design | "not included in latency calculation" | X-Talk (arXiv 2512.18706) |
| Thai TTS benchmark | VoxCPM 2 2.96 WER / 84.0 SIM; ElevenLabs 73.94 | dots.tts (arXiv 2606.07080) Table 3 |
| Thai pause separability | 41.9% EER, n = 1,137 / 396 | ours, LOTUSDIS, 2026-08-27 |
| endpoint window trade | 4/22 clipped at 600 ms, 6/22 at 300 ms | ours, `loop.bench.endpoint_sweep` |
| Smart Turn v3 synthetic share | 82.7% of 270,429 train clips; test split 81.4% | pipecat-ai/smart-turn-v3 dataset card |
