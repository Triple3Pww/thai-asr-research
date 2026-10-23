# The two text banks for the adapter experiment

Built 2026-09-12 with the Antigravity CLI (`agy`, Gemini 3.5 Flash) via `make_text_banks.py`,
audited with `check_text_banks.py`, then **relabelled into Typhoon's own transcription convention**
by `fix_label_convention.py` — see the finding below, which cost 34 rows and would otherwise have
cost GPU hours and a wrong result.

They are the input to `colab/gen_colab.py`, which speaks them with VoxCPM. Why two banks, and why
they differ, is the adapter design §5.3 (`git show 4734fc5:docs/specs/2026-09-11-synthetic-adapter-asr-design.md`): phase 1 trains the
**adapter** with the decoder frozen, phase 2 trains the **decoder** with the adapter frozen.
Feeding either bank to the wrong phase defeats the mechanism.

| bank | rows | purpose |
|---|---|---|
| `texts/phase1_known.json` | **440** | words the decoder ALREADY knows, so the only remaining source of loss is TTS-ness |
| `texts/phase2_rare.json` | **245** | the rare words themselves — the vocabulary no Thai corpus contains |

`*.raw.json` beside each is the untouched `agy` output, kept so the relabelling is reversible.

## Every row carries two text fields, and they are not the same thing

| field | content |
|---|---|
| `text` | what VoxCPM **speaks** — spaced, numbers as Thai words, SARA AM composed (ordinary written Thai) |
| `label_text` | what the ASR must **emit** — **unspaced**, numbers as Thai words, SARA AM **decomposed** |

`text` keeps its spaces because it is fed to a speech model, where spacing helps VoxCPM phrase a
digit sequence as separate digits rather than as one word. `label_text` matches Typhoon on all
**three** measured axes — spacing, numbers, and SARA AM encoding. The third is in `_thai.py` and is
the nastiest of them, because the two encodings render identically and Unicode NFC does not bridge
them.

Phase 2 additionally carries `entity` — the rare token alone, so §8's rare-word accuracy metric
has something to score.

---

## 🔴 The finding that reshaped both banks

Measured 2026-09-12 against **53 real Thai turns** in `live/_out/turns-*.jsonl`. The live panel's
path is mic → Silero → Typhoon (:8007) → llama-server → VoxCPM, so the `asked` field is
`typhoon-asr-realtime`'s own output:

| | Typhoon | the banks as generated |
|---|---|---|
| ASCII digits | **0 / 53** | **279 / 279** labels |
| Thai number words | 7 transcripts | — |
| any space | **1 / 53** (mean 0.02) | mean **4.64** per label |

The decisive row is a phone number read aloud — exactly where a digit-writing ASR would emit
digits:

```
ชื่อพอร์ตศูนย์แปดสามศูนย์หนึ่งแปดเก้าห้าสามเจ็ดครับ
```

It wrote words. Dates too: `วันที่ยี่สิบเอ็ด`, `วันที่ยี่สิบ`.

**Why it mattered.** Phase 2 trains the **decoder**. Training it on digit labels teaches it to
change convention, which raises **general CER** — the harm term in §8's adoption gate — and would
have read as a *leak* when it was really a relabelling. The experiment would have produced a
confident wrong answer.

**Three consequences, all applied:**

1. `label_text` is now unspaced Thai words in **both** banks. 🔴 Phase 1 needed this too, and it is
   easy to miss because that bank contains no numbers at all: phase 1 trains the adapter through a
   **frozen** decoder, so a target that disagrees with the decoder's spacing convention makes the
   loss dominated by an error the adapter cannot fix — the exact failure the two-phase split exists
   to prevent. 344 spaces removed across 440 rows.
2. **Entities reduced to their Thai core.** `ควอนตัมชาร์จ 160` now scores as `ควอนตัมชาร์จ`; the
   model number is a normalisation artifact, not a rare word. 30 entities affected.
3. **The 69 `postcode` rows lost their purpose.** With Typhoon already writing number words there
   is no digit-normalisation gap to teach, and a 5-digit `entity` can never appear in a correct
   label, so §8 could not score them at all. **35 were salvaged** — their sentence also names a
   district, so the entity was re-pointed at that place name — and **34 dropped.**

⚠️ **So this bank makes no claim about numeric normalisation.** That was the one lesson the
postcode rows carried, and the incumbent already has it. What survives is genuine out-of-vocabulary
**vocabulary**, which is the side of the line synthetic data actually wins on.

## 🔴 The third axis, found while measuring arm 1 — and it was worth 26.6% of CER

Thai SARA AM, the vowel in `บำรุง`, has two encodings that **render identically**: `U+0E33`
composed, and `U+0E4D U+0E32` decomposed. Unicode gives `U+0E33` **no canonical decomposition**, so
`unicodedata.normalize` is a no-op — measured, NFC changed **0 of 150** clips.

| | decomposed | composed |
|---|---|---|
| Typhoon's hypotheses | **57** | 0 |
| Common Voice references | 0 | **57** |

**47 of 150 clips (31%) carried the penalty.** Folding it moves mean CER **0.0706 → 0.0518** and
median **0.0405 → 0.0000** — the median clip becomes a *perfect* transcript — and perfect
transcripts 57 → 85.

**Why it was a trap rather than an inaccuracy.** The banks used the composed form exclusively (987
occurrences). Train the decoder on those labels, score with an unfolded metric, and the arm posts a
~26% general-CER *improvement that is purely a spelling change* — and §8's gate adopts it for a
reason with nothing to do with synthetic audio. Exactly the same shape as the digit problem, found
the same way, in the same hour.

Fixed on both sides, which is the only safe arrangement: scoring folds SARA AM on **both** operands,
and labels are written decomposed so fine-tuning does not silently change the model's output
spelling.

## Arm 1 — the baseline these banks will be judged against

`arm1_baseline.py`, 150 Common Voice Thai clips through shipped `typhoon-asr-realtime`:

| CER mode | mean | median | perfect |
|---|---|---|---|
| raw (whitespace only) | 0.0723 | 0.0430 | 55/150 |
| + punctuation | 0.0706 | 0.0405 | 57/150 |
| **+ SARA AM folded — quote this** | **0.0518** | **0.0000** | **85/150** |

15.0 min of audio, CPU, RTF 0.009. Full rows in `_out/arm1_typhoon_shipped.json`.

### ✅ And the rare-word half — arm 1 is COMPLETE

`arm1_rareword.py`, on `wayu-ai/thai-contextasr-bench` with **the benchmark's own scorer**,
condition `none`, **682 real-split utterances** (439 gold entity mentions):

| metric | value | 95% CI |
|---|---|---|
| CER overall | 0.1125 | [0.1022, 0.1239] |
| **U-CER** — outside entity spans, the harm term | **0.1057** | — |
| **B-CER** — inside entity spans | **0.2618** | [0.2330, 0.2927] |
| **entity recall** | **0.328** | [0.2751, 0.3802] |

RTF 0.007 on CPU. **Later arms must raise recall without U-CER rising above 0.1057.**

⭐⭐ **The thesis, quantified.** Rescoring the benchmark's six published baselines on the *same* 682
utterances: recall spans 0.253–0.435, and **B-CER is 2.01–3.04× U-CER for every model tested** —
RNNT, CTC and encoder-decoder alike. Entity spans are universally two to three times harder, and
**the field's best model still misses 57% of named entities** unbiased. Closing our 0.328 to 0.435
only *matches* the best; beating it would be new. Full table: `_out/arm1_rareword_comparison.json`.

⚠️ Ours is 6th of 7, but it is a ~115M-class RNNT at RTF 0.007 on **CPU** against ~1.5B Whispers on
GPU. Not size-matched.

🔴 Four constraints, all in the result JSON: **real split only** (the other 1,380 utts are TTS, and
evaluating a synthetic-data method on synthetic audio is circular); **condition `none` only** (our
model takes no bias list); the authors **declare split A may be contaminated** and prescribe
context-vs-no-context deltas we cannot form, so this is valid arm-vs-arm and is *not* a claim about
Typhoon's true ability; and **CC-BY-NC-4.0 — evaluation and thesis only, never training data or
product.**

## The other flaw, which does NOT need fixing

A place name can be paired with the wrong postal code. Province prefixes are reliable —
spot-checked Chiang Rai 57, Phatthalung 93, Suphanburi 72, Phayao 56, Nakhon Ratchasima 30000, all
correct — but district-level codes are not: one row gave บางใหญ่ as 11120, which is Pak Kret
(Bang Yai is 11140).

**It does not affect training.** The audio says `หนึ่งหนึ่งหนึ่งสองศูนย์` and the label says the
same, so the pairing is *internally consistent*. Geography is irrelevant to what the ASR learns.
⚠️ It *would* break any use that treats these rows as fact — a knowledge base, or an eval needing
a correct address. `check_text_banks.py` cannot catch it and says so; it tests the two-digit prefix
range only.

---

## phase1_known.json — 440 sentences

440/440 unique · 0 digits · 0 Latin · 38–84 chars · 40 each across 11 call-centre themes ·
exactly **one** repeated 18-char opening in the whole bank, so it is not templates.

No digits anywhere, deliberately: a number is a normalisation lesson and normalisation is phase
2's business — except that, per the finding above, it turns out not to be anyone's business here.

## phase2_rare.json — 245 sentences, 245 distinct entities

66–124 chars (median 93) · 0 digits in `text` or `label_text` · 0 Latin · `entity` present in
`label_text` for **245/245**.

| kind | rows | what it stresses |
|---|---|---|
| `place` | 155 | an unseen syllable sequence (120 original + 35 salvaged from postcode rows) |
| `product` | 90 | a foreign word forced through Thai phonotactics |

`make_text_banks.py` dedupes on **entity** as well as sentence — 300 sentences teaching 120 words
is a worse bank than 300 teaching 300. It rejected 21 rows during generation for reusing a postal
code.

---

## Reproducing

```bash
python adapter/make_text_banks.py --phase 1 --batch 40     # 12 themes
python adapter/make_text_banks.py --phase 2 --batch 30     # 10 specs
python adapter/fix_label_convention.py --apply             # REQUIRED after generating
python adapter/check_text_banks.py adapter/texts/phase2_rare.json --samples 12
```

🔴 `fix_label_convention.py` is not optional polish — a freshly generated bank is in the **wrong**
convention, because the prompt asks for digits in the WRITTEN field. The fix runs after, rather
than the prompt being changed, so the raw model output stays inspectable in `*.raw.json`.

🔴 **`id` values must stay stable.** `gen_colab.py` seeds every generation from the clip id, so
renumbering a bank regenerates the whole corpus rather than extending it. Dropped rows leave gaps
in the id sequence — that is intended. To add rows, append.

One `agy` call per batch, and no partial output on timeout, so `ASK_TIMEOUT` is deliberately
generous (420 s). The first phase-1 run lost a batch of 40 outright at 180 s.

## What this buys, in clips

`--per-text` is the speaker-diversity knob and the one the §7 ratio sweep turns. Voices are
assigned so a text's N clips are N **different** speakers — guaranteed by a coprime stride, not
left to chance.

| bank | `--per-text` | clips | audio | generation at local RTF 0.29 |
|---|---|---|---|---|
| phase 1 | 8 | 3,520 | ~6.4 h | ~1.8 h |
| phase 2 | 8 | 1,960 | ~4.0 h | ~1.2 h |

~10 h of audio, ~1.2 GB — inside the design's "20–30 hours is enough, do not plan a big run"
discipline, with room to raise `--per-text` for the ratio sweep.

## Still missing before any of this trains

- the **rare-word evaluation corpus** (`wayu-ai/thai-contextasr-bench`, 2,062 utts) — not pulled
- **Thai telephone-channel room impulse responses** — gate on arm 3, the arm that sets the bar the
  whole architecture has to beat (§2)
- **arm 1, the untrained baseline** — non-negotiable and runs first (S5)

Generating audio does not unblock any of those three.

🔄 **Superseded (2026-10-08).** The three items above were closed long ago — the benchmark was pulled and
arm 1 ran in September (`RESEARCH-PROGRESS.md` §1); the impulse-response arm was never run. Kept as written.

## The banks as ASR training data — 2026-10-07 (`F35`)

The first time the synthetic audio trained an ASR rather than a picker: NECTEC fine-tuned on convention-
matched LOTUSDIS **+ 4,201 phase-1/phase-2 clips** (`lot3`, `E:/Claude/datasets/synth/manifest_nectec_train.jsonl`).
Two conversions were needed and are in that manifest: SARA AM **re-composed** (NECTEC writes composed, the
banks store decomposed — the opposite of the Typhoon rule above), and 13 clips of 6 entities that occur in
the benchmark references dropped. Result: synthetic held-out (60 unseen texts) CER 0.039 → 0.011 (−72%),
real benchmark flat (0.0695 → 0.0695 at α .5), entity recall **down** (0.367 → 0.335 at α 1). The model
learned the VoxCPM voice and the entity *types*, not Thai names in real mouths. 🔴 **Label convention is
per base model:** decomposed for Typhoon, composed for NECTEC — check the base model's own output first.

## The evaluation harness added 2026-10-08 → 10 (`F38`–`F46`)

The text banks above are one part of `adapter/`; this is the part built for the "from zero → CER 0.060" line.
Every script states its venv in its docstring (ASR = `E:/asr-spike/.venv`, Thai text = `E:/thainlp-spike/.venv`,
Whisper/CTranslate2 = `E:/fw-spike/.venv`).

| need | script | why it exists |
|---|---|---|
| one honest number per arm | `official_utt.py` | per-utterance errors from the benchmark's own `score_utterance` (Thai-or-Latin entity forms), summing exactly to the official CER |
| dev/test discipline | `paired_vs.py BASE ARM --half dev` / `--half test --final` | paired bootstrap (10k) on official errors; the test half refuses without `--final` and logs each use to `results/test_evaluations.log` |
| the split itself | `results/split_dev_test.json` | 34 sessions → dev 339 / test 343, committed `494d951` before any tuning |
| hearing vs spelling | `eval_soundblind.py` | script-blind CER + toned-syllable error rate; separates a better ear from a different typist |
| where the errors are | `probe_error_anatomy.py` | error share by character class, script, entity span, burst length |
| fast large Whispers | `spec_whisper.py` | NECTEC's draft verified in parallel by a large Whisper, KV-cached, lossless (264–297 ms vs ~1,030) |
| combining systems | `pilot_rover.py --add`, `rover_manifest.py`, `pool_search_dev.py`, `poe_rescore.py`, `judge.py` | ROVER on any set, dev-only pool search, sentence product of experts, trained judges |
| new training data | `gs2_convention.py`, `night_gs2*.sh` | rewrite a corpus into NECTEC's convention (incl. dropping loanword-script conflicts), leak-guard, fine-tune → interpolate → score |

🔴 Conventions that carry over from the banks: labels in the BASE model's convention (`F32`, `F45`), SARA AM folded on
both sides, number words not digits, and the 20-character leak guard against all 2,062 benchmark references.
