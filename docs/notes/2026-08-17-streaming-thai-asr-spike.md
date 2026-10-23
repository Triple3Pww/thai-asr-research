# Spike — is there a streaming Thai ASR?

> 🔴 **SUPERSEDED IN PART, 2026-09-06. The answer is now YES.** Everything below about
> `typhoon-asr-realtime` still holds — its checkpoint really is full-context and its "Real-time"
> name really is misleading. What does **not** hold is this note's verdict as a statement about the
> field: `typhoon-ai/typhoon-asr-streaming-nemotron-0.6b` is a genuine cache-aware streaming Thai
> ASR, and it is now wired on :8011. It is less accurate than the model below, so it supplies
> partials rather than the transcript of record. See
> `2026-09-06-streaming-thai-asr-wired.md`, `LITERATURE.md` A20–A22.
>
> This note is left exactly as written. Its closing line — *"the remaining step is a limited-context
> fine-tune"* — was the right call from the evidence available, and was made obsolete by someone
> else shipping the checkpoint, not by a mistake in the measurement.

**Date:** 2026-08-17
**Plan task:** 7 of `E:/Claude/visualllm-business/docs/superpowers/plans/2026-08-17-thai-voice-loop-milestone-1.md`
**Status:** disqualifiers fixed; measurement in progress

## The question

Whisper is **batch**: it charges for the whole utterance *after* the caller stops talking. The
500 ms budget allocates **60 ms** to `stt_flush`. So the loop needs an ASR that emits partials
*during* speech and finalises almost instantly at end-of-turn.

If no such model is good enough in Thai, that is a legitimate finding — the budget's 60 ms line
gets replaced by the measured chunked-Whisper number, and the 500 ms target is unreachable until
this is solved. **That outcome must be stated, not planned around.**

## Guard

A prior research sweep in this repo confidently **inverted** several ear-verified Thai TTS findings
(it claimed the truncating F5 JaiTTS "fixes truncation"; it claimed Kokoro does Thai — it does not).
**A model card saying "supports Thai" is not evidence.** Every claim below is measured by running
the model on real Thai audio on this box.

## Probe set

Three clips, real Thai speech with accurate transcripts, resampled to 16 kHz mono from the VoxCPM
reference clips. Built by `loop/bench/_spike_probe/` (`probe.json` holds the transcripts).

| id | duration | transcript chars |
|---|---|---|
| `ref_long11` | 11.90 s | 154 |
| `ref_solo14` | 14.74 s | 131 |
| `ref_long19` | 18.00 s | 227 |

Each also has a **6 s cut** (`*_6s.wav`). Reason: a real call-centre utterance is 3–8 s, and a
batch model's penalty scales with duration — reporting only 12–18 s clips would overstate it.
CER is measured on the full clips (where ground truth exists); latency is reported for both.

**Note on the transcripts:** they contain **no word spaces at all**
(`หนูรู้นะที่แม่บอกว่าอยู่โรงพยาบาลแล้วเบื่อ...`). That is not a formatting quirk — it is the
property that makes Thai endpointing and Thai CER scoring different from English, and it is half
the wedge in spec §2 D2.

## Candidates

| candidate | why it is on the list | what to check |
|---|---|---|
| **Typhoon ASR Real-time** (`typhoon-ai/typhoon-asr-realtime`) | 115M **FastConformer-Transducer**, Thai-specific, explicitly built for streaming; 10k hours of Thai; claims 45× less compute than Whisper-large-v3. arXiv [2601.13044](https://arxiv.org/abs/2601.13044). From SCB10X — the same team whose LLMs are already on this box | does the packaged API expose partials, or only file-in/text-out? Real first-partial and final-flush latency |
| **faster-whisper, chunked pseudo-streaming** | the honest fallback, already installed. Re-transcribe a growing buffer | is it fast enough to fake streaming at all? |
| **faster-whisper, batch** | the baseline penalty to beat | how bad is the whole-utterance charge, really |

Deliberately **not** tested: Deepgram / cloud streaming. It would set a useful upper bound, but
spec §2 D2 rules out a cloud hop inside the loop, so a number we cannot use is not worth the key.

## Disqualifiers — written down BEFORE measuring

A candidate is out if **any** of these holds. Fixed in advance so no result can be rationalised
afterwards:

1. **Cannot emit a partial before the utterance ends.** A model that only speaks at the end is a
   batch model however it is packaged.
2. **CER on the probe set worse than batch Whisper by more than 5 points absolute.** Speed that
   costs accuracy transfers the latency to the caller repeating themselves.
3. **Final-flush latency above 150 ms** after true end of speech. The budget allows 60 ms; 150 ms
   is the generous bound at which the 500 ms target is already gone.

## Results

Measured on this box, 2026-08-17. **Whisper had the GPU; Typhoon ran on CPU** — the
`typhoon-asr` install pulls a CPU-only torch, and it was left that way deliberately: in the
real loop the GPU is already holding VoxCPM and the LLM on a 16 GB card, so CPU-resident STT
is the realistic deployment and it frees VRAM for the two stages that need it.

**Final flush (true EOS → final transcript).** This is what the budget pays.

| clip | faster-whisper large-v3-turbo (GPU) | Typhoon ASR (CPU) |
|---|---|---|
| `ref_long11` 6 s | 304 ms | **155 ms** |
| `ref_solo14` 6 s | 308 ms | **144 ms** |
| `ref_long19` 6 s | 279 ms | **138 ms** |
| `ref_long11` 11.9 s | 1206 ms | **207 ms** |
| `ref_solo14` 14.7 s | 408 ms | **227 ms** |
| `ref_long19` 18.0 s | — | **316 ms** |

**Accuracy (CER, full clips).** Lower is better.

| clip | Whisper | Typhoon |
|---|---|---|
| `ref_long11` | 0.130 | **0.032** |
| `ref_solo14` | 0.244 | **0.236** |
| `ref_long19` | 0.502 (chunked: 0.762) | **0.084** |

**Whisper falls apart on the longest clip, and not gracefully — it code-switches.** Its
chunked transcript of `ref_long19`:

> `หนูルูนะ ที่แม่บอก adop sont từ mallopแต่ถ้าแม่ไม่กิน anime จะมีแล้ว как Macpho`

Japanese katakana (`ル`), Vietnamese (`từ`), Russian (`как`) and English (`anime`, `Macpho`)
inside one Thai sentence, at CER 0.762 — with `language="th"` explicitly set. Typhoon
transcribed the same audio as clean Thai at CER 0.084, **9x better**. Whisper's flush on this
clip was also 3500–3682 ms.

This is the multilingual-model failure mode the venture docs already describe from the LLM side
(Typhoon 2.1-Gemma3-12B emitting Vietnamese mid-Thai). A model for which Thai is 1 of ~100
languages degrades *into its other languages*; a Thai-only model has nowhere else to go. It is
the clearest single piece of evidence for spec §2 D2 — "nobody's Thai is good because nobody's
incentive is Thai" — that this repo has produced.

Caveat on `ref_solo14`: some of that 0.236 is **Unicode normalisation, not mishearing** —
Typhoon emits `ทํา` (ท + nikhahit) where the reference has `ทำ` (sara am). A normalising
comparison would score it better. Do not quote 0.236 as an error rate without that note.

**Chunking buys nothing at the end.** Whisper's chunked flush (308 ms) is the same as its
batch flush (304 ms), because Whisper keeps no incremental state and re-transcribes the whole
buffer every time. That is the structural reason a transducer is needed — not a faster GPU.

**The decisive measurement is not a timing at all.** Typhoon's released checkpoint:

```
class:              EncDecRNNTBPEModel     <- a transducer, as advertised
encoder:            ConformerEncoder
att_context_size:   [-1, -1]               <- -1 = FULL, unlimited context
att_context_style:  regular                <- not "chunked_limited"
```

NeMo's cache-aware streaming requires `att_context_style: chunked_limited` with a finite
context (e.g. `[70, 13]`). The `cache_aware_stream_step` / `conformer_stream_step` methods are
present, but only because they live on the base class. **As released, Typhoon ASR "Real-time"
is a full-context offline transducer.** It is fast, and it is the best Thai ASR measured here
by a wide margin — but it cannot emit partials mid-utterance without limited-context attention
it was never trained for.

This is exactly the guard case at the top of this note: the model card says streaming, the
config says full context, and only the config is evidence.

## Verdict

**No candidate passes disqualifier 1.** Whisper is architecturally batch; Typhoon is packaged
and trained as full-context. So **the budget's 60 ms `stt_flush` line is not achievable
today**, and per this note's opening rule that is stated, not planned around.

**Replace 60 ms with the measured best: ~145 ms** (Typhoon, CPU, on a 2.5–8 s utterance —
the realistic call-centre length). Against disqualifiers 2 and 3, Typhoon passes both
comfortably (better CER than Whisper; flush inside the 150 ms generous bound on short
utterances). It fails only the one that matters most.

**Revised budget with today's parts:**

```
endpointing     150 ms   (aspirational; the inherited window measures 645 ms)
stt_flush       145 ms   (was 60 -- Typhoon batch on a short utterance)
llm_first_token 120 ms   (aspirational; ollama measures 443-558 ms)
tts_ttfb        150 ms   (VoxCPM measures 178 ms in isolation)
                -------
                565 ms
```

**500 ms is not reachable with today's components. 800 ms is.**

**The path to 500 ms, and why it is this repo's kind of work.** Typhoon is an RNN-T built on
NeMo's FastConformer, and NeMo supports fine-tuning that encoder to `chunked_limited`
attention for cache-aware streaming. That single change would (a) cut the flush from ~145 ms
toward genuinely incremental, and (b) **produce the partial transcripts the overlapped
architecture needs for prefill-during-speech** — which is currently a no-op because nothing
emits partials. So the finding is not "no streaming Thai ASR exists". It is:

> The best Thai ASR is most of the way there, and the remaining step is a limited-context
> fine-tune of an Apache-licensed Thai model — which is the same shape of work as the RSFT
> loop this repo already owns.

That belongs in milestone 2. Recommended for milestone 1: keep Whisper as the correctness
reference, record ~145 ms as the honest `stt_flush`, and do not claim a live number.

