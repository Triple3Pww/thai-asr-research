# Literature Review — Latency in Self-Hosted Thai Voice Agents
### Presentation content, 2026-09-01. 12 slides. Every number here is sourced.

> **How to use this file.** One `##` heading per slide. **Bold lines are the slide text**; the
> `Notes:` block is what to say. Nothing here is unverified — figures marked ⚠️ are the ones a
> supervisor is most likely to push on, and the honest answer is written next to each.

---

## Slide 1 — Title

**Latency in Self-Hosted Thai Voice Agents**
**A stage-level literature review, and what it does not contain**

Chanachon Ruanpheng · 2026-09-01

Notes: One sentence of framing — *"I set out to compare my system against published voice-agent
latency numbers, and found that the comparison mostly cannot be made. That absence is the result."*

---

## Slide 2 — The system under study

**A Thai voice agent that must answer in under 500 ms, entirely self-hosted**

- Audio in → audio out, five stages, one consumer GPU (RTX 5060 Ti, 16 GB)
- Self-hosting is not a preference: a cloud hop inside the loop costs more than the budget
- Thai is a low-resource language — most published components do not support it

Notes: Explain *why* self-hosted and Thai are coupled: both force the same architecture. The 500 ms
target is a design constraint taken from human turn-taking, not a marketing number.

---

## Slide 3 — Method

**A claim ledger, not a summary**

- **155 rows**, each = one external claim **with a number in it**, plus a status
- Statuses: `UNTESTED` (default) · `CONFIRMED` · `REJECTED` · `BLOCKED` · `RULED-OUT`
- **A claim is not a finding.** A row changes status only when a measurement on this hardware moves it
- This sweep: **13 queries → ~150 unique records → 42 rows kept → 15 papers read in full**

Notes: This is the slide that earns credibility. Emphasise that ~110 papers were deliberately
discarded and that the *discards are logged by category* so the search is not repeated. Rejected
rows are kept permanently — they are the most useful ones.

---

## Slide 4 — What a voice agent's latency is made of

**Five sequential stages, and each is a separate research literature**

| stage | what it costs us | who studies it |
|---|---|---|
| endpointing — deciding the caller stopped | 300 ms | ASR / turn-taking |
| STT | 65 ms | speech recognition |
| LLM first token | 119 ms | inference serving |
| LLM clause generation | 186 ms | inference serving |
| TTS first audio | 92 ms | speech synthesis |

Notes: The point to land: **each stage is rigorously studied in isolation, by a different
community, in a different unit.** That is the root of everything on the next slide.

---

## Slide 5 — Finding 1: the units do not agree

**Four incompatible units for "latency", and papers disagree because of it**

- **first-packet / TTFB** — TTS
- **p90 emission latency** — streaming ASR
- **algorithmic latency** — chunk-derived, *not wall-clock*
- **RTF (real-time factor)** — throughput, and **cannot see first-token at all**

> Two papers disagree about which stage dominates. One measures RTF over a whole utterance;
> we measure time to first audio. Both are right.

Notes: Concrete example — i-LAVA concludes TTS dominates; our table says endpointing does. The
disagreement is a units artefact, not a hardware one.

---

## Slide 6 — Finding 2: the gap (the main result)

**Of four published stage breakdowns, NOT ONE includes endpointing**

| paper | has stage table? | endpointing line | clock origin stated? |
|---|---|---|---|
| Qiu 2026 (enterprise tutorial) | ✅ | ❌ absent | ❌ |
| Ethiraj 2025 (NetoAI telecom) | ✅ | ❌ absent | partial |
| ChipChat 2025 (Apple, ASRU) | ✅ | ❌ absent (starts at mel) | ❌ |
| X-Talk 2025 | ❌ none | — | states it *excludes* VAD |

> X-Talk says it outright: *"the frontend VAD uses a 500 ms end-of-utterance silence threshold…
> **and this fixed delay is not included in latency calculation**."*

Notes: **This is the contribution.** Every published "sub-second cascade" is sub-second on a clock
that starts *after* the endpointer fired — while endpointing is the largest single stage in our
system. So published numbers and ours have never been comparable.

---

## Slide 7 — Finding 3: the discipline exists, but not where it is needed

**The endpointing literature measures from true end-of-speech. The voice-agent literature does not.**

- Bijwadia et al. 2022 defines endpoint latency as *"the time between when the user stops speaking
  (**ground truth EOQ**) and the detected EOQ"* — **EP50 390 ms / EP90 740 ms**, and **270 / 570 ms**
  for their jointly-trained model, at no accuracy cost
- Goyal et al. 2023, same clock: **2858 → 1525 ms** (and **5080 → 3096 ms on noisy audio**)

Notes: So the rigour is not missing from the field — it is missing from the *whole-loop* papers,
which inherit none of it. ⚠️ Expect: *"so your endpointer is slower than Google's."* Yes — 300 ms
window vs their trained 270 ms — but theirs is short-query voice search where the caller stops and
is done; ours is conversation, where Thai mid-turn and turn-final pauses are only 41.9%-EER
separable by duration.

---

## Slide 8 — Our measurements, beside theirs

**On the span they actually measure, we are 1.4–1.9× faster**

| | **ours** | Qiu | NetoAI | ChipChat |
|---|---|---|---|---|
| total as published | **797 ms** | 755 ms | 934 ms | ~920 ms |
| **post-endpoint (comparable)** | **~497 ms** | 755 ms | 678 ms | ~920 ms |
| clock origin | **hand-marked true EOS** | not stated | "per utterance" | not stated |
| hardware | RTX 5060 Ti | cloud STT/TTS | not stated | Mac Studio M2 Ultra |

Notes: Say the caveat before they ask it: this is **reported-vs-reported**, across different
hardware, languages and workloads — positioning, not a controlled experiment. ⚠️ And the 300 ms
endpointing window buys speed by **cutting callers off earlier** (6/22 utterances clipped vs 4/22
at 600 ms). A lazier endpointer always scores better on this metric; that is precisely why the
clock must start at true end-of-speech.

---

## Slide 9 — Finding 4: the Thai last mile

**Model cards claiming a language are not evidence**

- Of five published streaming-TTS systems at 54–102 ms first-packet: **four cannot speak Thai**,
  and the fifth scores **2.5× worse** at Thai than the model already in use
- In an independent benchmark's Thai row (WER / speaker-similarity):
  MiniMax **2.70 / 80.0** · **VoxCPM 2 — 2.96 / 84.0** ← best of the open systems ·
  Fish-Audio **4.23** · **ElevenLabs 73.94 / 58.8**
- Voice-design instructions are obeyed in **English and Chinese**; a **Thai** instruction is read
  aloud verbatim (4 phrasings, 4 failures)

Notes: ElevenLabs at 73.94 WER is the memorable one — a leading commercial TTS is effectively
unusable in Thai. ⚠️ Do **not** generalise it to "cloud TTS fails at Thai": MiniMax is cloud and
tops the row. The correct claim is that English-centric vendors fail Thai.

---

## Slide 10 — What this repository knows that the literature does not

**Five results with no published counterpart**

1. **No published stage breakdown contains an endpointing line** — 0 of 4, verified in full text
2. **Thai mid-turn vs turn-final pauses are barely separable by duration — 41.9% EER**
   (1,137 / 396 pauses, LOTUSDIS corpus)
3. **The best Thai ASR is packaged as streaming and is not** — its own paper reports **no latency
   figure at all**, and never states its attention-context configuration
4. **Zero-shot audio turn-detection fails on Thai, and fails confidently** — wrong at 0.87–0.98
   probability on speakers who continued for another 1.6–2.2 s
5. **No per-stage latency exists for any self-hosted Thai loop** — this is the first

Notes: Frame these as *publishable*, and note each is a measurement, not a reading.

---

## Slide 11 — Limitations, stated first

**What these numbers cannot support**

- **n = 5** for the end-to-end table; published figures are medians over larger sets
- **Cross-system comparison is reported-vs-reported** — never bracketed against their hardware
- This machine **drifts 25–45% across an evening**; one configuration read 205 ms and 179 ms ten
  minutes apart. Only same-window comparisons are quoted
- **Our test corpus is 2–3× easier than the ASR model's own benchmarks** (our CER 3.2% vs their
  published 6.81–10.65%), so accuracy figures here are optimistic
- Synthetic and scraped audio, not real telephone calls

Notes: Presenting limitations before being asked is the single strongest move available. The drift
number is worth saying aloud — it invalidates most casually-collected benchmark tables.

---

## Slide 12 — Where this goes

**Next**

- **Publishable now:** the endpointing-gap result, and the first per-stage table for a self-hosted
  Thai loop measured from hand-marked true end-of-speech
- **Endpoint anticipation** (forecasting turn-end rather than detecting it) reports **−505 ms** —
  the largest published win against our largest stage
- **Streaming Thai ASR does not exist** and is the blocker for the overlapped architecture — a
  limited-context fine-tune, not a download
- **A voice with no consent problem**: a described voice, cloned from one generated sample, tested
  and working in Thai

Notes: Close on the question worth a supervisor's attention: *is the endpointing-measurement gap
worth a short paper on its own?* The answer looks like yes, and the data already exists.

---

## Appendix — figures a supervisor may probe

| claim | number | source |
|---|---|---|
| our loop, end to end | 797 ms p50, n=5, from true EOS | `loop/bench/report.py` |
| Google production endpointer | EP50 390 / EP90 740 ms | Bijwadia 2022, SLT (arXiv 2211.00786) |
| jointly-trained endpointer | EP50 270 / EP90 570 ms, no WER cost | same |
| Hinglish EOS detection | 2858 → 1525 ms; noisy 5080 → 3096 | Goyal 2023 (arXiv 2305.18596) |
| enterprise cascade | STT 337 / LLM 337 / TTS 219; 755 ms measured | Qiu 2026 (arXiv 2603.05413) |
| telecom cascade | ASR 49 / LLM 670 / TTS 286; 934 ms | Ethiraj 2025 (arXiv 2508.04721) |
| on-device cascade | ~920 ms cumulative, M2 Ultra | ChipChat, ASRU 2025 (arXiv 2509.00078) |
| VAD excluded by design | "not included in latency calculation" | X-Talk (arXiv 2512.18706) |
| Thai TTS benchmark | VoxCPM 2 2.96 WER / 84.0 SIM; ElevenLabs 73.94 | dots.tts (arXiv 2606.07080) Table 3 |
| Thai pause separability | 41.9% EER, n=1,137/396 | ours, LOTUSDIS, 2026-08-27 |
