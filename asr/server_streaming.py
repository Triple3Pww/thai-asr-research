"""Typhoon ASR Streaming Nemotron 0.6B (Thai) as an out-of-process HTTP server, on the GPU.

    POST /stream?sid=<n>&final=0   body = raw PCM16 mono  ->  {"text": "<partial so far>"}
    POST /stream?sid=<n>&final=1   body = raw PCM16 mono  ->  {"text": "<final>", "final": true}
    POST /asr?sr=16000             body = raw PCM16 mono  ->  {"text": ..., "infer_ms": ...}
    GET  /health                                          ->  {"ok": true, "model": ...}

WHY THIS EXISTS, AND WHY IT DOES NOT REPLACE :8007.

`asr/server.py` (:8007, typhoon-asr-realtime, CPU) is FULL-CONTEXT -- it cannot say anything
until the caller stops. That made prefill-during-speech a no-op for the whole life of this
repo. This model is genuinely cache-aware streaming (`EncDecRNNTBPEModelWithPrompt`,
att_context_size [56,3]), so it emits a growing transcript WHILE the caller is still talking.

It is not a replacement, because it is LESS ACCURATE. Measured here 2026-09-06 on the three
`loop/bench/_spike_probe` clips, character error rate against the same references:

    typhoon-asr-realtime  115M  CPU  offline    0.032 / 0.236 / 0.084   mean 0.117
    this model            0.6B  GPU  offline    0.123 / 0.183 / 0.141   mean 0.149
    this model            0.6B  GPU  streaming  0.117 / 0.229 / 0.229   mean 0.192  (240 ms)

    (the incumbent's column is from the 2026-08-17 spike, a DIFFERENT measurement window --
     re-bracket both in one window before quoting this as a verdict. See the drift warning
     in CLAUDE.md.)

So the intended pairing is BOTH: this server's partials drive prefill and the semantic
endpointer while the caller speaks; :8007 still produces the final transcript that has to be
right. Do not delete :8007.

🔴 IT DOES NOT CURRENTLY FIT ALONGSIDE THE FULL LOOP. Measured 2026-09-06, and it cost an
hour of confusing symptoms, so it is the first thing in this file after the numbers:

    GPU   this server holds 2.9 GB (10609 -> 7684 MiB when stopped). The full loop
          (llama-server 9b + VoxCPM) sits at 14.6 GB of 16.3. VoxCPM needs ~6.9 GB and had
          5.6. It DIED SILENTLY during model load -- exit code 0, no traceback, no message,
          :8005 just never appeared. Restarting it with this server stopped: up in ~110 s.
    CPU   the fallback this note used to recommend, and it is ALSO blocked. With the loop up
          there is ~5 GB of system RAM free (VoxCPM alone holds 8.9 GB) and a second NeMo
          process needs ~4.5 GB: `import nemo` raises MemoryError before the checkpoint is
          touched. Neither device has room.

So `live/run-thai.ps1` does NOT start this by default -- it is `-Stream`, opt-in. Run it to
see the partial WITHOUT the full loop, or free VRAM first (the 9b -> 4b LLM swap is the
obvious candidate and is wanted for other reasons anyway). Do not "just turn it on" with
-StartAll: the failure mode is a TTS server that never starts and no error to read.

WHY THE GPU AT ALL. Measured ~34 ms per encoder step against a 320 ms chunk -- roughly 10x
real time, with headroom, and 29 ms p50 round-trip over HTTP. CPU throughput is still
unmeasured, because the box has never had the RAM free to try it.

LATENCY SETTING. att_context_size [56, R] where R*80 ms is the look-ahead the model waits for.
Supported: R = 3 / 6 / 13 -> 240 / 480 / 1040 ms. R=0 crashes (zero-size cache). Measured mean
CER 0.192 / 0.193 / 0.195 -- essentially flat -- so the cheapest is taken. R is not a quality
knob on this evidence; do not raise it hoping for accuracy.

THE FEEDING RULE THAT COST AN AFTERNOON. Two non-obvious things break this quietly:

  1. `CacheAwareStreamingAudioBuffer.append_audio()` defaults to `stream_id=-1`, which creates
     a NEW batch row per call -- it is built for batching files, not for appending to one
     stream. Pass stream_id=0 after the first, or run the preprocessor yourself (we do).
  2. Its `__iter__` advances `buffer_idx += shift_size` whether or not that many real frames
     exist. Feeding it a short chunk therefore pads with zeros AND skips the frames that
     arrive next. Symptom: a fluent-looking transcript missing most of the words. We step the
     encoder only when a full chunk of real frames is available (`_drain`).

Also: mel features are computed over the WHOLE stream so far and only the new frames are
appended, because preprocessing each arriving slice independently injects an edge artefact
every slice (that version froze at ~8 s). `_EDGE_MARGIN` holds back the last few frames,
which are still against the padded edge, until the next round.

RUN (ASR venv -- NOT the VoxCPM one):

    E:/asr-spike/.venv/Scripts/python.exe asr/server_streaming.py     # port 8011
    (or:  ./asr/run-streaming.ps1 )

Boot is ~25-40 s: NeMo import, checkpoint restore, warmup. Socket opens after warmup.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

MODEL_REPO = os.environ.get("ASR_STREAM_REPO", "typhoon-ai/typhoon-asr-streaming-nemotron-0.6b")
MODEL_FILE = os.environ.get("ASR_STREAM_FILE", "typhoon-asr-streaming-nemotron-0.6b.nemo")
PORT = int(os.environ.get("ASR_STREAM_PORT", "8011") or "8011")
DEVICE = os.environ.get("ASR_STREAM_DEVICE", "cuda")
LANG = os.environ.get("ASR_STREAM_LANG", "th-TH")
RIGHT_CTX = int(os.environ.get("ASR_STREAM_RIGHT", "3") or "3")   # 3/6/13 -> 240/480/1040 ms
SR = 16000

#: Trailing mel frames held back each round -- they sit against the preprocessor's padded
#: edge and would be frozen into the buffer wrong. They are appended next round instead.
_EDGE_MARGIN = 4

_MODEL = None
_LOCK = threading.Lock()      # one caller at a time, same contract as :8007 and VoxCPM


def _load() -> None:
    global _MODEL
    import torch
    from huggingface_hub import hf_hub_download
    import nemo.collections.asr as nemo_asr

    path = hf_hub_download(MODEL_REPO, MODEL_FILE)
    print(f"[asr-stream] loading {MODEL_FILE} on {DEVICE} ...", flush=True)
    t0 = time.perf_counter()
    m = nemo_asr.models.ASRModel.restore_from(path, map_location=DEVICE)
    m.eval()
    m.set_inference_prompt(LANG)              # prompt-conditioned: without this it is not Thai
    m.decoding.set_strip_lang_tags(True)
    m.encoder.setup_streaming_params(att_context_size=[56, RIGHT_CTX])
    _MODEL = m
    print(f"[asr-stream] att_context_size={m.encoder.att_context_size} "
          f"(~{RIGHT_CTX * 80} ms look-ahead)", flush=True)

    s = _Session()                            # warm up before the socket opens
    s.feed(np.zeros(SR, dtype=np.float32))
    s.finish()
    print(f"[asr-stream] ready in {time.perf_counter() - t0:.1f}s (port {PORT})", flush=True)


class _Session:
    """One caller's streaming state: raw audio, mel buffer, encoder cache, decoder hypotheses.

    Held across requests so the encoder never re-reads audio it has already consumed -- that
    is the whole point of cache-aware streaming, and the reason this is not just chunked
    re-transcription.
    """

    def __init__(self) -> None:
        import torch
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )
        self._torch = torch
        self.buf = CacheAwareStreamingAudioBuffer(model=_MODEL, online_normalization=False)
        self.cache = _MODEL.encoder.get_initial_cache_state(batch_size=1)
        self.prev_hyp = None
        self.pred_out = None
        self.audio = np.zeros(0, dtype=np.float32)
        self.appended = 0
        self.text = ""
        self.enc_ms = 0.0

    def feed(self, y: np.ndarray, final: bool = False) -> str:
        torch = self._torch
        self.audio = np.concatenate([self.audio, y]) if self.audio.size else y
        # 🔴 The preprocessor raises on a zero-length signal, and that used to kill the whole
        # request handler -- the client saw a bare RemoteDisconnected with nothing in the log.
        # It is reached normally: the panel resets a session, then the endpointer's finalise
        # arrives for a session that no longer exists and builds an empty one. Nothing to
        # transcribe is a legitimate answer, not an error.
        if self.audio.size == 0:
            return self.text
        with torch.inference_mode():
            proc, _ = self.buf.preprocess_audio(self.audio.copy())
        end = proc.size(-1) if final else max(0, proc.size(-1) - _EDGE_MARGIN)
        if end > self.appended:
            self.buf.append_processed_signal(
                proc[:, :, self.appended:end],
                stream_id=(0 if self.appended else -1),
            )
            self.appended = end
            self._drain(final)
        return self.text

    def _drain(self, final: bool) -> None:
        torch = self._torch
        cfg = _MODEL.encoder.streaming_cfg
        t0 = time.perf_counter()
        while True:
            cs = cfg.chunk_size
            cs = (cs[0] if self.buf.buffer_idx == 0 else cs[1]) if isinstance(cs, list) else cs
            avail = int(self.buf.streams_length[0]) - self.buf.buffer_idx
            # A short chunk would zero-pad AND skip the frames arriving next. Wait instead.
            if avail <= 0 or (avail < cs and not final):
                break
            chunk, lens = next(iter(self.buf))
            with torch.inference_mode():
                (self.pred_out, texts, c0, c1, c2, self.prev_hyp) = _MODEL.conformer_stream_step(
                    processed_signal=chunk,
                    processed_signal_length=lens,
                    cache_last_channel=self.cache[0],
                    cache_last_time=self.cache[1],
                    cache_last_channel_len=self.cache[2],
                    keep_all_outputs=final,
                    previous_hypotheses=self.prev_hyp,
                    previous_pred_out=self.pred_out,
                    drop_extra_pre_encoded=None,
                    return_transcription=True,
                )
            self.cache = (c0, c1, c2)
            if texts:
                t = texts[0]
                self.text = t.text if hasattr(t, "text") else str(t)
        self.enc_ms = (time.perf_counter() - t0) * 1000.0

    def finish(self) -> str:
        return self.feed(np.zeros(0, dtype=np.float32), final=True)


_SESSIONS: dict[str, _Session] = {}


def _pcm_to_float(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def _offline(y: np.ndarray) -> str:
    import torch
    with torch.inference_mode():
        out = _MODEL.transcribe(audio=[y], batch_size=1, target_lang=LANG, verbose=False)
    if not out:
        return ""
    return str(getattr(out[0], "text", out[0])).strip()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a) -> None:
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/health":
            self._json(200, {"ok": _MODEL is not None, "model": MODEL_REPO,
                             "sample_rate": SR, "device": DEVICE,
                             "right_context_ms": RIGHT_CTX * 80, "streaming": True})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        # An exception here would propagate out of the handler and close the socket with no
        # response at all -- the caller sees RemoteDisconnected and the cause is invisible.
        # Answer with the error instead, and drop the session so the next turn starts clean.
        try:
            self._post()
        except Exception as e:                                  # noqa: BLE001
            sid = parse_qs(urlparse(self.path).query).get("sid", ["0"])[0]
            with _LOCK:
                _SESSIONS.pop(sid, None)
            self._json(500, {"error": f"{type(e).__name__}: {e}", "text": ""})

    def _post(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))

        if u.path == "/stream":
            sid = q.get("sid", ["0"])[0]
            final = q.get("final", ["0"])[0] in ("1", "true", "yes")
            t0 = time.perf_counter()
            with _LOCK:
                s = _SESSIONS.get(sid)
                if s is None:
                    s = _SESSIONS[sid] = _Session()
                text = s.feed(_pcm_to_float(raw), final=final)
                enc_ms = s.enc_ms
                if final:
                    _SESSIONS.pop(sid, None)
            self._json(200, {"text": text, "final": final,
                             "enc_ms": round(enc_ms, 1),
                             "infer_ms": round((time.perf_counter() - t0) * 1000.0, 1)})
            return

        if u.path == "/stream/reset":
            with _LOCK:
                _SESSIONS.pop(q.get("sid", ["0"])[0], None)
            self._json(200, {"ok": True})
            return

        if u.path == "/asr":
            sr = int(q.get("sr", [str(SR)])[0])
            if sr != SR:
                self._json(400, {"error": f"expected sr={SR}, got {sr}"})
                return
            y = _pcm_to_float(raw)
            t0 = time.perf_counter()
            with _LOCK:
                text = _offline(y)
            self._json(200, {"text": text,
                             "infer_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                             "audio_s": round(len(y) / SR, 3)})
            return

        self._json(404, {"error": "not found"})


def main() -> None:
    _load()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[asr-stream] listening on http://127.0.0.1:{PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
