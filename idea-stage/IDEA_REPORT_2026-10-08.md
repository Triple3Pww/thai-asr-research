# Idea Report — a Thai ASR architecture designed from zero (2026-10-08)

**Direction** (his words): *"try start from 0 and invent the whole new architecture of ASR … research as
much as possible across all the field that you think might be useful, then analyse it and report me the
possibility."* Pipeline run `/aris:research-pipeline`, Stage 1 (idea discovery) + one measurement pilot.
**Stages 2–3 (training, review loop) were not run** — the ask was a possibility report; see §7.

> Point-in-time. Every percentage below about OUR errors is measured on this box and reproducible with
> `adapter/probe_error_anatomy.py`. Every number about OTHER people's systems is their claim, unverified here.

---

## 0. The answer in five lines

1. **"From zero" in weights is not possible here; "from zero" in design is.** A new encoder trained from
   random weights on the ~150 h of fresh Thai on disk cannot reach a 115M model trained on thousands of
   hours (`F28`: the corpora that would help are already in the incumbents' training data). So the design
   keeps a strong frozen ear and redesigns everything after it.
2. **We measured where the best real-time model (NECTEC 115M) still fails before inventing anything.** The
   ear is better than the score says: **~23% of its errors are the same loanword written in Latin vs Thai
   script, ~20% of the pure-Thai errors are the right sound spelled wrong, and 29% of all errors sit
   inside names that are only 5.7% of the text.** Tone-only errors are just 8%.
3. **So the invented architecture separates the jobs:** an **Ear** that reports *what was said as sound*
   (Thai syllables: onset · rhyme · tone), a **Memory** that looks up names by sound, and a **Pen** that
   turns sound into spelling and chooses the script — and the Pen can be trained on **text alone**, which
   Thai has in unlimited supply while Thai audio is the bottleneck.
4. **Possibility: realistic, not guaranteed.** The parts it targets are ~40–50% of the current error; a
   plausible capture is a 15–25% relative CER drop (0.068 → ~0.052–0.058 canon). The two-stage idea itself is
   published for Mandarin, Polish/German and Vietnamese-English; **the Thai version with script choice and
   a sound-keyed name memory is not**, as far as one Consensus search shows.
5. **The cheapest first step needs no GPU:** a script-blind, sound-aware metric. Up to a quarter of the
   "error" we have been optimising against may be spelling convention (same lesson as `F1`, `F17`).

---

## 1. What was searched (Consensus first, as directed)

Ten Consensus searches + one web search, chosen to cover fields an ASR redesign could borrow from:

| field | what it offers | the number that matters |
|---|---|---|
| **Hearing biology** — cochlea models (CoNNear, CARFAC, gammatone + hair cell) | ear-like front ends | gains are **in noise**, on small-vocabulary tasks; none beats a learned front end on large-vocabulary clean speech [b1–b3] |
| **Neuroscience** — predictive coding | the brain predicts the next sound top-down and only sends the *surprise* up; priors *sharpen* the sensory signal [n1, n2] | ASR equivalent = inject context early in the encoder: −60% rare-word error on LibriSpeech [r8] |
| **Tonal phonetics** | explicit pitch/tone models | Mandarin: a *perfect* tone oracle = 30–32% CER cut [t1]; Thai with tone features: 0.4–6.3% WER [t2] |
| **Thai phonology** | syllable = onset + rhyme + tone, a closed set | onset–rhyme units beat monophones by 26–30% on Thai [s1] |
| **Syllable tokenisers** (Sylber, SyllableLM) | speech at ~5 tokens/s instead of 50–100 | 30× less training compute, 4× faster inference for speech LMs [s2, s3] |
| **LLM decoders** (SLAM-ASR, Speech-LLaMA, SALM) | world knowledge in the decoder | strong, but large and autoregressive (slow) [l1, l2] |
| **Retrieval / memory** (kNN-CTC, BR-ASR, speech-enriched memory) | look names up instead of storing them in weights | 200k-entry bias lists at 20 ms/query, −45% biased-WER [r1]; −33–39% on rare entities with RAG correction [r2] |
| **Non-autoregressive refiners** (Mask-CTC, Align-Refine, Edit-Flow 2026) | fix a fast draft in 2–3 parallel steps | Mask-CTC 17.9 → 12.1% WER on WSJ at RTF 0.07 [a1]; Edit-Flow: 2 edit steps suffice [a2] |
| **Analysis-by-synthesis / speech chain** | speak each hypothesis back with TTS and compare | old idea (Stevens & Halle), revived in cognitive science [n3]; machine speech chain works for semi-supervised training [c1] |
| **Test-time adaptation** (SUTA, speaker codes) | adapt per utterance / per speaker without labels | −20% WER from 1 minute of a speaker's audio [ta1] |
| **Two-stage sound → spelling** (P2G, pinyin) | decouple hearing from writing | pinyin two-stage −11% over 11 dialects [p1]; LLM-P2G −3.6/−6.9% Polish/German [p2]; P2G decoder pretraining on 4B tokens of text helps out-of-domain [p3]; Vietnamese–English code-switch two-stage SOTA [p4] |

The 2026 open-model releases (Cohere Transcribe, Nemotron 3.5 ASR) are still Conformer encoder +
decoder — **no architectural break in the mainstream** [w1]; they don't list Thai either.

---

## 2. The pilot: where NECTEC's errors actually are

`adapter/probe_error_anatomy.py` on `results/nectec_fc_greedy_hyps.jsonl` (682 real utts, 71,559 ref
chars). Own Levenshtein after `canon()` + SARA AM fold + punctuation strip on both sides; Latin-script
entity spans in the reference replaced by their annotated Thai form (the benchmark's scorer accepts
either, plain Levenshtein would not). Own-scorer CER **0.0778** (the benchmark scorer says 0.0676 — shares,
not absolutes, are the point). Raw output: `results/anatomy_nectec_metrics.json`.

| slice of the 5,566 character edits | share | what it means for design |
|---|---|---|
| **Latin vs Thai script for the same loanword** (`superg`/`ซูเปอร์จี`, `wifi`/`ไวไฟ`, `maria`/`มาเรีย`) | **23.1%** | not a hearing error. The references themselves mix scripts, so part of this is a metric artifact |
| **Right sound, wrong spelling** — errors that vanish when both sides go to a sound key (royin romanisation, pure-Thai utts) | **21.5%** (crude consonant-class key: 17.9%) | a writing problem: fixable from text, without audio |
| **Tone mark only** | **8.3%** | an explicit pitch branch can fix at most this. Mandarin's 30% oracle does not transfer: modern encoders already hear tone |
| **Inside named-entity spans** | **29.0%** of errors in **5.7%** of the text (5× density) | names need a memory, not more weights (`F24`, `F34`, `F36`) |
| **Particles / fillers** (นะครับ, เนี่ย, อะ) | 4.5% | small; transcription-convention, not hearing |
| **Edits in bursts of ≥4 chars** | **61.1%** | most error is whole words/names lost or replaced, not single-character slips |
| **5-best oracle** (perfect re-ranker over NECTEC's own beam) | CER 0.0778 → 0.0742, **−4.6%** | the beam does not contain the answer. Re-ranking one model's beam is capped at ~5% (`F29`: diversity is the ceiling) |

**The reading:** NECTEC's ear is better than its score. Something like **40–50% of what we have been
calling error is the Pen** (script + spelling), **~30% is names**, and **~5% is what better search over its own
beam could ever buy**. A new architecture should therefore spend its novelty on *writing* and *memory*,
not on a bigger ear or a smarter beam.

---

## 3. Ideas generated, and how each one fared

Scored against the measured error budget above, not against intuition. "Ceiling" = the share of
NECTEC's current error the idea is even aimed at.

| # | idea | from field | ceiling (measured) | evidence elsewhere | cost on this box | verdict |
|---|---|---|---|---|---|---|
| I1 | **Ear–Memory–Pen**: syllable-phonetic Ear + sound-keyed name Memory + text-trained Pen (§4) | P2G, Thai phonology, retrieval, predictive coding | **~45–60%** (script + spelling + names, overlapping) | pinyin −11% [p1], LLM-P2G −3.6/−6.9% [p2], RAG names −33–39% [r2] | medium: heads on frozen encoder (cached features exist), Pen = text training | ⭐ **RECOMMENDED** |
| I2 | **Script-blind, sound-aware metric** | — (measurement hygiene, like `F1`/`F17`) | re-prices up to ~23% of "error" | `F17`: a third of the benchmark error was convention | none (CPU) | ⭐ **DO FIRST** — needed to even measure I1 fairly |
| I3 | Explicit **tone/pitch branch** | tonal phonetics | **8.3%** | Mandarin oracle 30% [t1]; Thai real gains 0.4–6.3% [t2] | low | ❌ ceiling too small on its own; fold into I1's Ear as one input |
| I4 | **Analysis-by-synthesis verifier** (TTS each candidate, compare) | cognitive science, speech chain | **≤4.6%** (beam oracle) | IBM patent; no modern WER number found | high latency (TTS per candidate) | ❌ capped by the beam; the answer is not in the list |
| I5 | **Predictive-coding feedback loop** (LM prediction re-enters encoder) | neuroscience | names + context words | early context injection −60% rare-word [r8] | high (retrain encoder) | ⏸ later: realise it cheaply as I1's Memory feeding the Ear |
| I6 | **Mask-CTC / Edit-Flow parallel refiner** | NAR decoding | the 61% burst errors | −32% vs CTC on WSJ [a1] | medium; needs fresh paired data (`F28`) | ⏸ a possible Pen implementation, not a separate bet |
| I7 | **Syllable-rate (5 Hz) tokens** | speech LMs | speed, not accuracy | 30× compute, 4× speed [s2] | medium | ⏸ makes an LLM Pen affordable; adopt inside I1 if the Pen is an LM |
| I8 | **Cochlear / spiking front end** | biology, neuromorphic | noise robustness only | gains in noise on digits/commands [b2, b3] | high | ❌ the benchmark's errors are not noise errors |
| I9 | **Per-speaker test-time adaptation** | TTA | conversational speakers | −20% from 1 min of audio [ta1] | latency + stability risk | ⏸ product feature for the voice loop, orthogonal to I1 |
| I10 | **Brand-new encoder from random init** | — | all | — | needs ~10³–10⁴ h fresh Thai; we have ~150 h | ❌ infeasible here (`F28`) |

---

## 4. The invented architecture: Ear – Memory – Pen

```
audio ─► EAR (frozen NECTEC FastConformer, 115M)
          ├─ RNN-T head ............ draft text (what it does today)
          ├─ CTC head (exists, F36) . frame posteriors for names
          └─ NEW syllable head ..... Thai sound per syllable, factored:
                                     onset (≈21+clusters) · rhyme · tone (5) · [pitch features in]
                 │
                 ▼
        MEMORY: names looked up BY SOUND, not by spelling
          • catalog of Thai places/brands/people + their transliterations, keyed by syllable sound
          • optional user list (today's bias@N) is just a small catalog
          • retrieval like BR-ASR (200k entries, ~20 ms) + the existing numba spotter (kws_fast, F37)
                 │
                 ▼
        PEN: sound + draft + retrieved names ─► final spelling AND script
          • trained on TEXT ONLY: take any Thai sentence → G2P → corrupt the sound
            (simulated Ear mistakes) → learn to write the original sentence back
          • decides ไวไฟ vs wifi, เค้า vs เขา, ๆ vs repeat, digits vs words — every convention
            question that `F1`/`F2`/`F17`/`F32` found by accident, made a learned, swappable policy
          • non-autoregressive (edit-based, I6) to stay real-time; LLM-sized only if latency allows
```

**Why this split is the right one for Thai specifically**

- **Thai writing is the hard part, not Thai hearing.** 44 consonant letters for 21 initial sounds,
  silent letters (์), and a reference convention that flips between Latin and Thai for loanwords. The
  pilot says this is ~40–50% of the measured error. An end-to-end model has to learn spelling *from paired
  audio*, the scarcest thing we have; the Pen learns it from text, the most plentiful.
- **Convention becomes a setting, not a training accident.** `F32` showed fine-tuning on a corpus teaches
  its *typist*. With a separate Pen, the typist is one small model you can swap per customer — the Ear
  never sees label conventions at all.
- **Names stop living in weights.** The memory grows by adding rows, not by retraining (the repo's own
  `F36`/`F37` line, generalised from "a 50-name list" to "a catalog").
- **It keeps everything that already works.** NECTEC, its 42.9 ms one-pass splice, the CTC head and the
  cached encoder features are all reused. Nothing is thrown away to try this.

**Where it can fail (stated before any run)**

1. **Cascade loss.** If the Ear's syllable head is wrong, the Pen cannot recover the word. Mitigation from
   the literature: pass N-best/lattices, not one string (top-K marginalised training [p2]); keep the RNN-T
   draft as a second input so the Pen never has *less* information than today.
2. **Thai G2P quality bounds the labels.** The best published Thai G2P is ~10% word error [g1]. The syllable
   head's training targets come from G2P, so its errors become label noise.
3. **The script choice may be unlearnable** — if the benchmark references pick Latin vs Thai arbitrarily, no
   Pen can predict them, and the right move is the metric (I2), not a model.
4. **Latency.** A generative LLM Pen would cost the ~370 ms the picker cost (`F29`). The real-time version
   must be the edit-based Pen.

---

## 5. Novelty

Consensus search "two-stage phoneme/pinyin intermediate then text-only trained P2G": **the split is
published** — Mandarin pinyin two-stage [p1], LLM-P2G for Polish/German [p2], P2G pretraining [p3],
Vietnamese–English code-switching two-stage [p4], POWSM unifying ASR/PR/G2P/P2G [p5]. **Not found:** a Thai
instance; script choice (Latin vs Thai) treated as the Pen's job; a name memory keyed by Thai syllable
sound; and an error budget measured before the design. **Verdict: incremental-but-real** — the
contribution is the Thai measurement + the composition, not the two-stage idea. ⚠️ One search is not a
novelty check; run `/novelty-check` before claiming anything in writing.

---

## 6. Possibility, honestly

| question | answer |
|---|---|
| Can it beat NECTEC on the benchmark? | **Plausible.** Aimed at ~45–60% of the error. If it captures a third of that: **−15–25% relative CER**. No measurement on this box yet. |
| Can it stay real-time? | **Yes if the Pen is edit-based** (Ear + Memory already run at 42.9 ms, `F37`). No if the Pen is a generative LLM. |
| Does it need data we don't have? | **No — that is its point.** Ear heads train on cached features of corpora already here; the Pen trains on Thai text. |
| What would kill it early? | (a) script-blind metric shows the 23% was mostly the benchmark's inconsistency → Pen's job shrinks; (b) syllable head on frozen encoder can't reach a low syllable error → cascade loss dominates. |
| Biggest unknown | The Pen's real gain on *real* Ear output (vs simulated noise). Literature gains range from −3.6% to −11%; the high end assumes a strong first stage. |

---

## 6b. Review status — NOT cross-model reviewed

The Gate-1 reviewer (`gemini-review` MCP) returned `Connection closed`, the same failure as the
2026-10-06 run, so this report is **unaccepted**. Self-review in its place, against my own claims:

1. **I4 was rejected on one model's beam.** The 4.6% cap holds for NECTEC's own 5-best; a verifier over
   the *multi-model* pool (`F27`, `F29`) has a higher oracle. The rejection stands for latency, not for ceiling.
2. **The Pen needs a control I did not name above: plain text-only error correction of the RNN-T draft**
   (the repo's `pilot_ger.py` line). If a corrector that never sees the syllables does as well, the Ear's
   new head is not earning its place. E3 must run both.
3. **The slices overlap and two are upper bounds** (script errors include name errors; royin hides vowel
   length). "45–60% targeted" is not additive capture — the −15–25% is a guess, not an estimate.

## 7. Next steps, ranked (Stage 2 plan — not started)

| step | what | GPU | decides |
|---|---|---|---|
| **E0** | Add a **script-blind + sound-key CER** to `eval_canon.py`; re-score all existing systems (like `F17`) | none | how much of the 23% is real; may reorder the leaderboard |
| **E1** | **Thai P2G ambiguity**: run a Thai G2P (with tone) over the text banks + LOTUSDIS labels; count how often one sound sequence has several spellings | none | the Pen's ceiling |
| **E2** | **Syllable head** on the frozen NECTEC encoder (reuse `ctc_head.py` + `ctchead_cache*`; labels from G2P) | ~1 h | the Ear's syllable error rate → cascade risk |
| **E3** | **Pen v0**: small edit-based model trained on G2P-noised Thai text; input = syllables + RNN-T draft. **Control: the same Pen fed the draft only** | ~2–4 h | the first real end-to-end number, and whether the syllable head matters |
| **E4** | **Memory**: sound-keyed catalog retrieval replacing the bias list | ~1 h | names without a user list |

---

## References (Consensus, exact URLs)

- [b1] [A convolutional neural-network model of human cochlear mechanics and filter tuning for real-time applications](https://consensus.app/papers/details/fcbf466ee9385459b4c3e11ac1020660/?utm_source=claude_code) (Baby et al., 2020, Nature Machine Intelligence)
- [b2] [Robust Cochlear-Model-Based Speech Recognition](https://consensus.app/papers/details/3ddb2513b4145340861d7164957e9f98/?utm_source=claude_code) (Russo et al., 2019)
- [b3] [Deep Spiking Neural Network model for time-variant signals classification](https://consensus.app/papers/details/79ed5dce78195fc2b1070a4903e109c9/?utm_source=claude_code) (Dominguez-Morales et al., 2018)
- [n1] [Evidence of a predictive coding hierarchy in the human brain listening to speech](https://consensus.app/papers/details/b112f5479a7a54efae831f9f874785e8/?utm_source=claude_code) (Caucheteux et al., 2023, Nature Human Behaviour)
- [n2] [Sensory sharpening and semantic prediction errors unify competing models of predictive processing in human speech comprehension](https://consensus.app/papers/details/778a90a556095cea9364a84993aacb63/?utm_source=claude_code) (Schneider et al., 2025, PLOS Biology)
- [n3] [Feedforward and feedback in speech perception: Revisiting analysis by synthesis](https://consensus.app/papers/details/4f5da873c5905123928be55cb372a89a/?utm_source=claude_code) (Poeppel et al., 2011)
- [t1] [Improved tone modeling for Mandarin broadcast news speech recognition](https://consensus.app/papers/details/5a505844e69c5c5e96503241226646a5/?utm_source=claude_code) (Lei et al., 2006)
- [t2] [Incorporating tone features to convolutional neural network to improve Mandarin/Thai speech recognition](https://consensus.app/papers/details/bb9d7697000d507499df5c82df2bc22c/?utm_source=claude_code) (Hu et al., 2014)
- [s1] [An acoustic study of syllable rhymes: a basis for Thai continuous speech recognition system](https://consensus.app/papers/details/d9ba682dbb165ce39b47999b34df11ae/?utm_source=claude_code) (Maneenoi, 2003)
- [s2] [SyllableLM: Learning Coarse Semantic Units for Speech Language Models](https://consensus.app/papers/details/3bcaa85805a159af976a033946970498/?utm_source=claude_code) (Baade et al., 2024)
- [s3] [Sylber: Syllabic Embedding Representation of Speech from Raw Audio](https://consensus.app/papers/details/088e14283648572a874c46bad0e0e533/?utm_source=claude_code) (Cho et al., 2024)
- [l1] [An Embarrassingly Simple Approach for LLM with Strong ASR Capacity](https://consensus.app/papers/details/ea8ec3ea806d5ee391341e03a15179d0/?utm_source=claude_code) (Ma et al., 2024)
- [l2] [SALM: Speech-Augmented Language Model with in-Context Learning](https://consensus.app/papers/details/2f818fb6a2c6543591292a3a0ab4a271/?utm_source=claude_code) (Chen et al., 2023)
- [r1] [BR-ASR: Efficient and Scalable Bias Retrieval Framework for Contextual Biasing ASR in Speech LLM](https://consensus.app/papers/details/a18c266d4957519886be8cdd0464f1db/?utm_source=claude_code) (Gong et al., 2025)
- [r2] [Retrieval Augmented Correction of Named Entity Speech Recognition Errors](https://consensus.app/papers/details/f7db29731d405c6f83bc271e1cf02e12/?utm_source=claude_code) (Pusateri et al., 2024)
- [r3] [KNN-CTC: Enhancing ASR via Retrieval of CTC Pseudo Labels](https://consensus.app/papers/details/f671d1e615c6552f98ce7932bab3ac55/?utm_source=claude_code) (Zhou et al., 2023)
- [r4] [Speech-enriched Memory for Inference-time Adaptation of ASR Models to Word Dictionaries](https://consensus.app/papers/details/33091a2b4fe7569c9354da2dabb3ec4b/?utm_source=claude_code) (Mittal et al., 2023)
- [r8] [Improving Neural Biasing for Contextual Speech Recognition by Early Context Injection and Text Perturbation](https://consensus.app/papers/details/dd121196ec3c5c45be922e2eba969023/?utm_source=claude_code) (Huang et al., 2024)
- [a1] [Mask CTC: Non-Autoregressive End-to-End ASR with CTC and Mask Predict](https://consensus.app/papers/details/abda76cda7f6514391425085df300b6e/?utm_source=claude_code) (Higuchi et al., 2020)
- [a2] [CTC-Seeded Token Edit Refinement for Non-Autoregressive Speech Recognition](https://consensus.app/papers/details/f251e512db57573a88cba11a6a4e5224/?utm_source=claude_code) (Huang et al., 2026)
- [c1] [Machine Speech Chain](https://consensus.app/papers/details/2a1883320bdb55de94622601fa15f625/?utm_source=claude_code) (Tjandra et al., 2020)
- [ta1] [Robust Unsupervised Adaptation of a Speech Recogniser Using Entropy Minimisation and Speaker Codes](https://consensus.app/papers/details/ad03381c95515945902da08d44362fa3/?utm_source=claude_code) (van Dalen et al., 2025)
- [p1] [Toward Unified Chinese Multi-Dialectal Speech Recognition via Pinyin Intermediate Representation](https://consensus.app/papers/details/39cf17454d7c59c3a97d28ad447002a4/?utm_source=claude_code) (Cheng et al., 2026)
- [p2] [LLM-based phoneme-to-grapheme for phoneme-based speech recognition](https://consensus.app/papers/details/c589139969d75d08828dbbcf933a5199/?utm_source=claude_code) (Ma et al., 2025)
- [p3] [Phoneme-to-Grapheme Conversion Based Large-Scale Pre-Training for End-to-End ASR](https://consensus.app/papers/details/978515c44cf55cf0ad6da8b2b3319d29/?utm_source=claude_code) (Masumura et al., 2020)
- [p4] [TSPC: A Two-Stage Phoneme-Centric Architecture for code-switching Vietnamese-English Speech Recognition](https://consensus.app/papers/details/8896d79f8cc7525fb5f62ba816e7c8d6/?utm_source=claude_code) (Nguyen et al., 2025)
- [p5] [POWSM: A Phonetic Open Whisper-Style Speech Foundation Model](https://consensus.app/papers/details/86dbb2752fd15ce9b149f8fab16ff3a2/?utm_source=claude_code) (Li et al., 2025)
- [g1] [Efficient two-stage processing for joint sequence model-based Thai grapheme-to-phoneme conversion](https://consensus.app/papers/details/098db698a82351eab5b05bb8ac7cfc74/?utm_source=claude_code) (Rugchatjaroen et al., 2019)
- [w1] [Best open source speech to text models](https://www.gladia.io/blog/best-open-source-speech-to-text-models); [NVIDIA Nemotron speech FAQ](https://perspectives.nvidia.com/nemotron-speech/task/faq/what-is-the-best-open-source-streaming-speech-recognition-model-for-transcribing/)

---

## 8. RESULTS (same day, after "do it") — the design is falsified; see `F39`

| step | stop condition (pre-stated) | result | verdict |
|---|---|---|---|
| E0 script-blind + sound metric | — | script-blind −~10% for ALL systems; sound metric reorders the field (A2 hears 11% better than NECTEC at a CER tie) | ✅ useful diagnostic |
| E1 sound→spelling ambiguity | Pen ceiling | most-frequent spelling per TONED sound wrong on only **1.25%** of words | 🔴 Pen has little to do |
| E2 syllable Ear | Ear must beat G2P(draft) | **0.164 vs 0.115** — worse | 🔴 stop met |
| E3 Pen v0 (+ Ear arm) | must beat draft under cross-fit | 0 utterances changed; every threshold worse | 🔴 no gain |
| E4 name Memory | coverage | Thai Wikipedia covers 30% of missed names; needs a retrieval shortlist not on this box | ⏸ ceiling only |

**Why §2 misled:** the "~20% right sound, wrong spelling" slice came from sound keys without tone or vowel
length. Those are exactly what NECTEC gets wrong — they are hearing errors. §4's premise ("Thai writing is
the hard part") is false for this model on this benchmark.

## 9. The loop that followed (2026-10-08 → 10) — `F40`–`F46`

After the design failed, he asked to keep looping until success, then set the target at **CER 0.060** (≈ −13.7% vs
NECTEC) within ≤500 ms, proven on a **locked test half** (sessions split, commit `494d951`; dev = 339 utts).

| round | idea | result |
|---|---|---|
| `F40` | six real-time no-list ideas (catalog lists, session memory, TTA, RNN-T ensemble, NECTEC-family ROVER, ear/pen merge) | all lost |
| `F41` | NECTEC-drafted speculative Pathumma-L + typhoon-L + ROVER | 481 ms, −6.2% full set; TVSpeech tie; dev −3.1% |
| `F42` | ROVER + LM tie-break; sentence-level product of experts | worse / −2.3% dev |
| `F43`/`F44` | trained judges on LOTUSDIS, GigaSpeech2, pooled without ear identity | ±0 / +1.5% / ±0 dev |
| ⭐ `F45` | **NECTEC fine-tuned on fresh GigaSpeech2 (+ conv-LOTUSDIS), α .5–.75** | **dev −5.8 / −6.8%; TVSpeech −12.1 / −9.5%** |
| `F46` | + food orders; elderly speech | worse; not fresh |

**Where it stands:** best real-time model −6.8% on the dev half, about half the target. The lesson of the whole line: the
gap was in the EAR, and only fresh, convention-matched human audio moved it; no combiner of the existing ears did.
The test half is still unscored.
