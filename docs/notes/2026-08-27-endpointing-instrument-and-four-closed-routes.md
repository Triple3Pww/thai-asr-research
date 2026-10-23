# The endpointing instrument, and four routes closed by one fact

**2026-08-27.** Executed the endpointer-instrument plan, tasks 1–13.
**249 tests, 22 commits, nothing trained.**

This note is the complete record of what was tested and what was found. `E:\Claude\visualllm-business\HANDOFF.md` §7 is the
short version; `HISTORY.md` is the chronology; this is the detail, including every wrong turn,
because most of what was learned came from those.

---

## 0. The one-paragraph result

The instrument was built and it works. The resident LLM's old 569 ms blocker is **dead** (108 ms
p95) and it solves **group B** — the single stated reason route C is not the default. But
**text cannot separate "finished" from "pausing"**, structurally and at any accuracy; and **every
free source of ground truth is now measured and closed**: LOTUSDIS labels, a strict relabel of
them, YODAS captions, and the hand-crafted acoustic shortcut. The bottleneck is not the model,
the architecture, or the prompt. **It is the recording.**

---

## 1. Phase 0 — can Typhoon read far-field meeting audio?

`loop.bench.eot_cer`, 90 utterances over 6 LOTUSDIS sessions.

| | CER |
|---|---|
| all utterances | mean **0.171**, median 0.138, p75 0.214, p90 0.435 |
| references with no `[WORD]` code-switch tag (83%) | mean **0.123**, median 0.105 |

**Verdict `GO-WITH-CAVEAT`** (thresholds fixed before the run: ≤0.15 GO, ≤0.30 caveat, else STOP).

### 🔴 Two bugs in the measurement, both inflating it, found before the number was trusted

The first read was 0.185. Its worst cases were annotation artefacts, not audio:

- **`#word#` wraps real spoken Thai.** `#อย่างนี้#` is "like this". 7,945 in the corpus. The
  normaliser left the `#` in, charging two character errors per marker against transcripts that
  were correct.
- **`[WORD]` is also speech.** `[OK]` `[MENU]` `[BUFFET]` `[ONLINE]` `[M]` `[1]` — code-switched
  words the speaker said, written in Latin script while Typhoon emits Thai. 7,371 of them.
  Deleting them scored a correct transcript as an insertion; keeping them would score it as a
  substitution. **Neither is fair**, so they are now flagged and reported separately.
- **A 2.3 s interval can hold a 3-character interjection** (`อ๋อ` heard as `เออ` = 0.67). Added a
  **10-character reference floor**, fixed on arithmetic before the re-run, not after seeing it.

One case had **Typhoon more accurate than the human reference** — it heard `ซีรีส์` where the
annotator omitted it.

### The audio-conditioning branch was NOT run

Spec Task 5 step 7 makes a bake-off (plain / enhanced / telephone-band) conditional on landing in
0.15–0.30. It landed there, and was **skipped deliberately**: on the 83% of references that can
be scored fairly the CER is 0.123, and the remaining 17% fail because the reference is in the
wrong *script*. No filter fixes a Latin-script ground truth. Recorded rather than run.

---

## 2. The corpus

`loop/bench/eot_build.py` — Silero finds the pauses, the TextGrid says who held the floor,
Typhoon transcribes the turn so far. **~25 min a build** (the spec's "2–3 h" estimate was
conservative by 6×).

| build | rows | COMPLETE | empty transcripts |
|---|---|---|---|
| v1 (loose labels) | **13,602** | 2,388 (17.6%) | 1.4% |
| v2 (strict labels) | 10,925 | 2,255 (20.6%) | 1.2% |

Session-disjoint 70/10/10 split, `split.json` **committed** so a rebuild cannot redraw it.
Rows are git-ignored — rebuild before quoting any number.

**The text comes from Typhoon, never from the TextGrid** (E38). Training on the annotation would
let a model learn "ends with `<n>`" — 38,844 of those markers exist and none survive to
inference.

---

## 3. Phase 2 — the baselines, on 1,589 held-out rows from 9 unseen sessions

| baseline | finds real turn ends | interrupts mid-turn | EER |
|---|---|---|---|
| timer @400 ms | 54.6% | 34.2% | 39.9% |
| timer @600 ms *(deployed today)* | 34.1% | **15.9%** | 39.9% |
| particle grammar | 17.1% | 6.0% | 44.5% |
| **resident LLM** | **85.4%** | **58.1%** | 36.4% |

✅ **E4's 569 ms is dead: p95 check cost 108 ms** (+88 ms Typhoon flush = 196 ms) against a
200 ms window. It was ollama's per-request tax. Warm calls 130–156 ms; the cold first call read
339 ms and was discarded.

⚠️ Measured at `-np 1` on an **idle box**. Spec §6 also requires a measurement under the loop's
own load, and **that was not done.** The `-np 2` comparison was not run either.

### 🔴 The 58.1% is a label defect, not a model defect

The false alarms are complete Thai sentences a meeting speaker continued past:

> `สวัสดีค่ะเอฟศูนย์สองค่ะ` — "Hello, F02 here."
> `แต่ว่าเราชอบทํากินเองที่บ้านมากกว่า` — "but we prefer cooking at home."

Labelled INCOMPLETE only because nobody interrupted. **The false-alarm rate is flat across pause
length** — 59% at 0–300 ms, 57% at 300–600, 56% at 600–1000, 49% at 1–2 s. That flatness is the
signature: the model is reading grammar, and grammar says "complete" mid-monologue.

### ⚠️ The gate was inadequate — the most portable lesson in this note

Spec §6 gated adoption on **EER and latency, with no ceiling on cut-off rate**, and compared a
**binary rule's single operating point** against a **swept threshold's** EER. Apples to oranges.
It technically **passed** the resident LLM — a 58% interrupter, against today's 15.9%.

**Any future gate needs a cut-off ceiling, and must not compare a binary rule's EER to a swept one's.**

---

## 4. The strict relabel — proposed, built, and FAILED

`loop/bench/eot_label.py`. Three buckets instead of two, thresholds fixed before the run:

- **COMPLETE** — another speaker holds the floor ≥1.0 s **and** the original does not reclaim
  within 2.0 s (excludes backchannels and quick reclaims)
- **INCOMPLETE** — same speaker back within 500 ms (a breath)
- **everything else — DROPPED**, because a speaker who finishes a sentence and continues because
  nobody interrupted is neither

Uses only timestamps and speaker identity. Filtering with the LLM's own verdicts would be marking
its homework with its own answers.

🔴 **Its primary prediction did not happen. False alarm 58.1% → 59.3%; recall identical at
85.4%.** Relabelling changed nothing about the text-only judgement.

🔴 **And it introduced a circularity.** `MAX_BREATH_MS = 500` caps INCOMPLETE gaps at **480 ms**,
so any rule thresholding at ≥500 ms scores **0.0% false alarm and 100% precision by
construction**. The flattering `timer @600ms → 0.0%` and `LLM and gap≥500ms → 100%` rows are
artefacts. **Do not quote them.**

---

## 5. The transfer test — and the result that matters

`loop.bench.eot_transfer` on `corpus_tts/` (18 items, turn ends true **by construction**).

| | A: finished, particle | **B: finished, NO particle** | C: paused mid-thought | total |
|---|---|---|---|---|
| particle grammar | 6/6 | **0/6** | 6/6 | 12/18 |
| **resident LLM** | 6/6 | **6/6** | 1/6 | **13/18** |
| LLM, prompt v2 | 6/6 | 5/6 | 3/6 | 14/18 |

✅ **The adapter is verified**: the particle grammar reproduced 6/6 and 0/6 exactly — the
2026-08-18 measurement on an unchanged corpus.

⭐ **Group B is the single stated reason route C is not the default** (`E:\Claude\visualllm-business\HANDOFF.md` §7). The LLM
gets all six, including `พรุ่งนี้หน้าร้านเปิดทําการกี่โมง`.

🔴 **It fails group C 5 of 6, and those are real errors.** The misses end on dangling function
words: `...จะส่งถึงภายใน` ("will arrive within"), `...มันเป็น` ("it was"), `คือเงินที่ขอคืนไป`
(a bare noun phrase). Genuinely incomplete Thai read as complete.

⚠️ **Prompt engineering moves along the curve, not up it.** v2 named the dangling-word failure
explicitly: group C 1→3 but group B 6→5, net 13→14 of 18. On 18 items that is noise, and further
tuning would fit it. Independently supports E2's finding that SFT beats prompting.

---

## 6. 🔴 The structural result

**At the decision point, "I have finished and used no particle" and "I am pausing mid-thought"
are the SAME TEXT.**

| group | `(grammar, LLM)` |
|---|---|
| A | `(True, True)` |
| **B** | **`(False, True)`** |
| **C** | **`(False, True)`** (5 of 6) |

So `OR` = 13/18 (the LLM alone), `AND` = 12/18 (the grammar alone). **No boolean combination adds
anything, ever.** This is not an accuracy problem and a better text model does not fix it. It is
also why the LOTUSDIS false-alarm rate was flat across pause length: **text cannot see time.**

**And raising `max_wait` does not help once the LLM is judging.** The safety timer only fires when
the judge says *not finished*; on group C the LLM says *finished* and commits at 200 ms first.
The two obvious fixes cancel each other. Both were proposed, both were measured wrong.

Measured (group C pauses: 1200/1500/1800/2100/2400/3000 ms):

| max_wait | group B waits | group C cut off |
|---|---|---|
| 1500 ms *(shipped)* | 1500 ms | 5/6 |
| 2500 ms | 2500 ms | 1/6 |
| 3500 ms *(the 2026-08-18 run used this)* | 3500 ms | 0/6 |

### ⭐ The trade, priced on 13,602 rows

| | share |
|---|---|
| turn ends carrying **no** closing particle — only the LLM catches these | **73.7%** |
| mid-turn pauses longer than 1200 ms — where firing at 200 ms does clear damage | **1.63%** |
| (turn ends **with** a particle) | 26.3% |

**73.7% against 1.6% argues hard for the LLM. The data does not settle it**, because being cut
off hurts more per incident than waiting, and by how much is a product judgement. ⚠️ The 1.63%
comes from meeting audio whose labels do not mean what we need.

---

## 7. The acoustic shortcut — E26/E27/E28 do NOT reproduce here

`loop/bench/eot_prosody.py`. Five hand-crafted pre-pausal features (pitch slope, final pitch vs
utterance median, energy slope, voiced-tail length, voiced-frame count) on **400 real Thai
pauses** from LOTUSDIS test sessions — 200 final, 200 non-final.

🔴 **For this question those labels are RIGHT**, unlike the semantic one: E26's task is literally
"discriminate final pauses from non-final pauses", which is exactly what they encode.

| | EER (50% = chance) |
|---|---|
| pitch slope | 47% |
| final pitch | 48% |
| energy slope | 47% |
| voiced tail | 49% |
| voiced-frame count | 46% |
| **all five, 5-fold cross-validated** | **46%** ← E26 claims **20%** |
| **pause duration alone** | **38%** |
| prosody + duration | 43% ← **worse than duration alone** |

Also run on `corpus_tts` first: no separation there either (best single feature 75% on 12 items),
but **that corpus is TTS-generated and cannot answer a prosody question at all.**

### What this does NOT show

- **It does not refute E26.** These are five features reimplemented from paper abstracts, missing
  their spectral constancy and speaking rate.
- **Far-field three-party audio is near the worst case for prosody** — reverb smears exactly
  these contours. A close-mic phone call is a different signal.

**What it does show:** the *cheap* path is closed. E26's selling point was no training and no
target-language data — free signal sitting in the audio. On our audio it is not there.

---

## 8. YODAS — E3 tested and REJECTED, for 2 MB instead of 47 GB

`espnet/yodas` and `espnet/yodas2`, `th000` shard 0, **text only** — the audio was never needed.

**The test reused a baseline this repo already owns.** On 4,688 real Thai floor changes a turn end
carries a final particle **23.3%** of the time; a mid-turn point **8.7%**.

| | particle rate at boundary |
|---|---|
| real Thai **turn end** | **23.3%** |
| YODAS2 boundaries | 16.9% |
| YODAS v1 boundaries | 12.3% |
| real Thai **mid-turn** | **8.7%** |

Between the two, well short of a turn end. 🔴 And **56% of consecutive v1 segments have a zero
gap** — the captions tile the audio, so more than half the boundaries have no silence at all. An
endpointer decides *at a pause*.

🔴 **The disqualifying fact is structural and identical to the LOTUSDIS one: YODAS carries no
speaker information and the content is single-speaker video.** With no second speaker there is no
floor to change, so nothing defines "the caller finished" as opposed to "a sentence ended."

⚠️ **Two snap judgements corrected, both from reading the first file.** Shard 0 opens on Thai
gambling SEO spam in uniform 2 s chunks. Neither generalises: the modal duration covers **2.4%**
of segments (p10 1380 / p50 3180 / p90 6420 ms — natural), and gambling keywords hit **0.7%**
(v1) / **1.7%** (v2). **The corpus is fine; it is the wrong shape.**

✅ Still useful as a large free Thai ASR corpus with human captions, if a *sentence-completeness*
classifier is ever wanted.

---

## 9. Why the harness can be trusted where the labels could not

Four independent agreements with prior measurements:

| the harness said | prior measurement | |
|---|---|---|
| timer EER **39.9%** | 41.9% (8-session probe) | ✅ |
| particle recall **17.1%**, particle rate **26.3%** | 23.3% (4,688 floor changes) | ✅ |
| transfer adapter: grammar **6/6 A, 0/6 B** | 6/6, 0/6 (2026-08-18, unchanged corpus) | ✅ exact |
| duration EER **38%** (prosody probe) | 39.9% (1,589 rows) | ✅ |

That is why the 58.1% was attributed to the labels rather than to the instrument.

---

## 10. Everything that went wrong, and it is most of what was learned

| mistake | how it was caught |
|---|---|
| **The gate had no cut-off ceiling** and compared a binary rule's operating point to a swept EER | reading the cut-off column after the gate "passed" |
| **CER normaliser deleted spoken words** (`[WORD]`) and scored `#` as errors | inspecting the worst 5 cases instead of trusting the mean |
| **Duration floor let 3-char interjections through** | same inspection |
| **Strict relabel failed its own prediction** | it predicted a collapse; measured 58.1 → 59.3 |
| **Strict relabel introduced a circularity** | a rule scoring 100% precision, which is never real |
| **Both follow-up fixes cancelled each other** | measuring instead of reasoning |
| **"Combine grammar and LLM"** — impossible, identical signals | printing the raw signal patterns |
| **EER function reported 100%** for a discrete feature | 100% EER is arithmetically impossible |
| **llama-server never started**; a "4/6" result was entirely the failure path | all answers False at a constant ~2050 ms |
| **Dates stamped 2026-08-26** while every commit said 08-27 | `date` |
| **`git add -A loop/` committed 4 unreviewed pre-existing files** | reading `git show --stat` |

✅ **The failure direction held throughout**: a dead llama-server produced "not finished", never a
cut-off caller. That was designed in and it worked.

---

## 11. What is open, and what it costs

🔴 **Nothing measured here sees text AND time together.** The timer sees only time; the grammar
and the LLM see only text. Given §6, **a joint model is the only thing that can beat this trade**
— which is a *different design* from spec §6 phase 3 (text-only, trained on labels now known to
be wrong). **Phase 3 as specified is dead.**

**Four free routes to ground truth are now closed, all by the same fact:**

| route | why it failed |
|---|---|
| LOTUSDIS labels | meeting monologue — the speaker continues because nobody interrupted |
| strict relabel | could not fix it: 58.1% → 59.3% |
| YODAS captions | single-speaker video — no floor to change |
| acoustic shortcut (E26) | 46% EER on real Thai; duration alone (38%) beats it |

### The next step, and the one requirement that makes it cheap

**~2 hours of two-party Thai calls, each speaker on their OWN CHANNEL.**

🔴 **The channel separation is the whole point.** With one channel per speaker, "who was talking
when" falls out of a VAD per channel — no annotator, the same free-labels trick, pointed at audio
where "the other person spoke" actually means something. **A mono recording throws the labels
away and the session has to be redone.**

A few hundred turns is enough to *score* a decision function honestly, which is the milestone-1
deliverable. Training needs a few thousand; that is later.

✅ **Everything built today runs on it unchanged** — the builder, the split, both eval levels, the
transfer harness, the prosody probe. Swapping the corpus path is the whole change. That was the
point of building an instrument rather than a model.

---

## 12. Files

```
loop/bench/eot_textgrid.py   parse a TextGrid; attribute a pause to a speaker
loop/bench/eot_audio.py      96k/24k -> 16k mono (resample_poly, not stride)
loop/bench/eot_label.py      the strict rule, and why the obvious one failed
loop/bench/eot_rows.py       Row schema, jsonl, session-disjoint split
loop/bench/eot_build.py      corpus -> labelled decision moments (~25 min)
loop/bench/eot_cer.py        PHASE 0 go/no-go
loop/bench/eot_eval.py       level 1 -- cut-off vs miss, swept, as EER
loop/bench/eot_replay.py     level 2 -- realised latency through the state machine
loop/bench/eot_transfer.py   meetings -> corpus_tts, by utterance type
loop/bench/eot_prosody.py    pre-pausal prosody probe (E26/E27/E28)
loop/impl/endpoint_llm.py    the resident LLM as an is_finished decision function
```

Design: `git show 4734fc5:docs/specs/2026-08-27-thai-endpointer-training-design.md`
Plan: `git show 4734fc5:docs/plans/2026-08-27-thai-endpointer-instrument.md` (tasks 1–13 executed; 14 not written)
