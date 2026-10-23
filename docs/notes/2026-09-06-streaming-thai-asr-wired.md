# Streaming Thai ASR — found, measured, wired, and it does not fit

**Date:** 2026-09-06
**Supersedes (in part):** `2026-08-17-streaming-thai-asr-spike.md`
**Ledger rows:** `LITERATURE.md` A20, A21, A22 · **Summary:** `HISTORY.md` 2026-09-06

---

## 0. How it was found, which is the part worth keeping

I asserted the repo's standing finding — "no streaming Thai ASR exists" — and, to be careful,
re-fetched the `typhoon-asr-realtime` model card to check for a newer revision. It was unchanged,
so I reported the gap still open and proposed a limited-context fine-tune as a research project.

His entire reply was a URL: `typhoon-ai/typhoon-asr-streaming-nemotron-0.6b`.

**A targeted web search and a model-card fetch both missed a released checkpoint from a lab this
repo already depends on.** That is the second time (Instadesk, 2026-08-27) that his knowledge of
the Thai landscape beat a search. The generalised rule now lives in the
`feedback-ask-him-for-the-vendor-list` memory: **before asserting an absence in Thai speech AI,
say what was checked and ask him.** A repo doc restating an absence is not evidence it still holds —
which is exactly what `CLAUDE.md` had become.

## 1. The model

| | |
|---|---|
| class | `EncDecRNNTBPEModelWithPrompt` — cache-aware FastConformer-Transducer |
| size / base | 0.6B, on NVIDIA Nemotron 3.5 ASR (40 language-locales) |
| licence | OpenMDW-1.1 |
| prompt | **required** — `set_inference_prompt("th-TH")` + `decoding.set_strip_lang_tags(True)` |
| look-ahead | `att_context_size [56, R]`, R ∈ {3, 6, 13} → 240 / 480 / 1040 ms |
| environment | NeMo **3.0.0 already in `E:/asr-spike/.venv` has the class** — no upgrade needed |

Two facts that shape everything else: `conformer_stream_step` **does** apply the language prompt
(checked in source — `_apply_prompt_to_encoded` is called), but NeMo's own
`transcribe_simulate_cache_aware_streaming` **refuses this model class**
(`NotImplementedError`). So there was no reference streaming implementation to copy.

## 2. The three feeding bugs — all silent, all plausible-looking

These cost the afternoon. Every one produced fluent Thai rather than an error, which is why the
control (the model's own offline transcript of the same clip) was the only thing that caught them.

| # | bug | symptom | fix |
|---|---|---|---|
| 1 | `CacheAwareStreamingAudioBuffer.append_audio()` defaults to `stream_id=-1`, which appends a **new batch row** — it is for batching files, not extending a stream | 75 one-frame streams; **the encoder never ran once**; empty transcript, RTF 0.056 | `stream_id=0` after the first, or drive the preprocessor yourself |
| 2 | running the mel preprocessor on each arriving 160 ms slice independently | an edge artefact per slice; transcript **froze at ~8 s** and was wrong | preprocess the whole stream, append only new frames, hold back the last 4 (they sit against the padded edge) |
| 3 | 🔴 **the one that mattered** — the buffer's `__iter__` advances `buffer_idx += shift_size` **whether or not that many real frames exist** | a short chunk feeds zero-padding **and skips the frames arriving next**: fluent, grammatical Thai **missing most of the words** (29 chars vs 154) | step the encoder only when a full chunk of real frames is available |

Bug 3 moved CER from **0.903 → 0.117** on `ref_long11`. It is the most convincing wrong output
produced in this repo: nothing about the text suggests a plumbing fault.

**The lesson, and it is the repo's existing rule in a new place:** a streaming ASR must be judged
against *its own offline transcript of the same audio*, not against the reference. Against the
reference, bug 3 looked like "the model is mediocre".

## 3. Results

Three `_spike_probe` clips, CER against the same references, margin 4 frames, 160 ms feed.

| | long11 | solo14 | long19 | mean |
|---|---|---|---|---|
| `typhoon-asr-realtime` 115M CPU offline *(2026-08-17 window)* | 0.032 | 0.236 | 0.084 | **0.117** |
| this model offline | 0.123 | 0.183 | 0.141 | **0.149** |
| this model streaming, 240 ms | 0.117 | 0.229 | 0.229 | **0.192** |
| this model streaming, 480 ms | — | — | — | **0.193** |
| this model streaming, 1040 ms | — | — | — | **0.195** |

- 🔴 **The first row is cross-window** and the box drifts. Direction only; re-bracket before quoting.
- ✅ **0.149 vs 0.192 is same-window, same model** — so *"streaming costs ~0.04 CER"* is a real
  measurement here, and it independently confirms `LITERATURE.md` A10.
- ⭐ **Look-ahead is flat** — 0.192 / 0.193 / 0.195. It is not a quality knob on this evidence.
  Take 240 ms. **R=0 crashes** (`cannot reshape tensor of 0 elements`, zero-size cache).
- Speed: **~34 ms per encoder step** against a 320 ms chunk (GPU), **29 ms p50 / 40 ms p95** HTTP
  round-trip per 160 ms chunk, first partial at **~700–950 ms** of audio through the live panel.

## 4. 🔴 It does not fit on this box — and VoxCPM dies without saying so

| | needs | free with the loop up | outcome |
|---|---|---|---|
| GPU | **2.9 GB** VRAM (10609 → 7684 MiB when stopped) | 1.7 GB | VoxCPM needs ~6.9 GB, had 5.6 → **silent death** |
| CPU | ~4.5 GB RAM (second NeMo process) | ~5.0 GB | `import nemo` → **`MemoryError`** |

**The failure mode deserves its own paragraph.** VoxCPM did not raise, did not log, and printed no
traceback. It reached `Running on device: cuda, dtype: bfloat16` and **exited with code 0**. The
only downstream symptom was `ConnectionError` on :8005. Restarting it with :8011 stopped, nothing
else changed: up in ~110 s. **On this box, an exit-0 death during model load means VRAM — check
`nvidia-smi` before anything else.**

The full loop sits at **14.6 of 16.3 GB VRAM**, and VoxCPM alone holds **8.9 GB of system RAM**
against ~5 GB free of 31.8. There is no headroom of either kind. `live/run-thai.ps1` therefore
starts :8011 only under **`-Stream`**.

**The route to using it live is to free VRAM**, and the 9b → 4b LLM swap is the obvious lever —
wanted anyway, since `clause_cut` is 100% decode and 4b decodes ~1.7× faster. It remains a
Thai-quality decision, not a latency one.

## 5. What was built

- `asr/server_streaming.py` — :8011, stdlib HTTP (same discipline as :8007 and VoxCPM), session-keyed
  `POST /stream?sid=&final=`, `POST /stream/reset`, plus a one-shot `/asr` for parity. Handler wraps
  every request: an exception answers with a named error instead of dropping the socket.
- `asr/run-streaming.ps1` — ASCII, PS 5.1. Deliberately does **not** set `HF_HOME` (`asr/run.ps1`
  repoints it, which would force a second 1.6 GB download).
- `live/thai.py` — streaming client on a bounded queue drained by one worker, so the mic thread
  never waits on the network (29 ms p50 but 750 ms max observed). Drops rather than blocks.
- `live/thai.html` — the partial renders **dimmed**, replaced by the committed transcript when it
  lands. They are different models and must never look alike on screen.

Verified: with :8011 down the panel completes turns normally, `partial` empty, error named.

## 6. Two bugs found in the server, both fixed

- **Finalising a session with zero audio killed the request handler** — the mel preprocessor raises
  on a zero-length signal, and the client saw a bare `RemoteDisconnected` with nothing in the log.
  It is reached normally: the panel resets a session, then the endpointer's finalise arrives for a
  session that no longer exists. Nothing to transcribe is a legitimate answer.
- **Any handler exception dropped the socket silently.** Now returns `500` with the exception type
  and message, and clears the session. Verified: a 1-byte body returns
  `ValueError: buffer size must be a multiple of element size` and the server stays up.

## 7. What is now unblocked

`feed()` can return a real `PartialTranscript` for the first time in this repo's life — it has
returned `None` on every frame since day one, which is why prefill-during-speech has always been a
no-op. Spending that is the next work: prefill during speech, and `endpoint_semantic` reading a
live transcript instead of waiting for a finished one.

**It is display-only today**, and step 1 is freeing the VRAM to run it at all.

## 8. Repro

```powershell
.\asr\run-streaming.ps1                    # :8011, GPU, 240 ms look-ahead
.\live\run-thai.ps1 -Stream                # the panel WITHOUT the full loop
```

Throwaway probes used to get here (feeding matrix, HTTP client, panel driver) were scratchpad-only
and are not kept; §2's table is what they established.
