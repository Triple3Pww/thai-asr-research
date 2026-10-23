"""Thai ASR as an out-of-process HTTP server (Typhoon ASR Real-time, or since 2026-10-05 by
default NECTEC's Pathumma FastConformer -- see run.ps1 `-Model`).

2026-10-05: `nectec/Pathumma-stt-th-fastconformer-rnnt-large` (CC-BY-4.0, same FastConformer-RNNT
family, ~same latency) scored CER 0.0676 on the 682 real thai-contextasr-bench utterances against
typhoon-whisper-large-v3's 0.0735 and typhoon-asr-realtime's 0.0945 (canon normaliser; beam 5,
113 ms/turn). Its lead over large-v3 is NOT significant (paired bootstrap P=0.084), and it is worse
on named entities. It writes digits (77/682 clips) and composed SARA AM, where Typhoon never does.
Record: E:/Claude/asr-research/idea-stage/IDEA_REPORT.md.

(The text below was written for the Typhoon model and still describes how this server works.)

    POST /asr?sr=16000   body = raw PCM16 mono   ->   {"text": "...", "infer_ms": 123.4}
    GET  /health                                 ->   {"ok": true, "model": "..."}

WHY OUT OF PROCESS, AND WHY THE CPU.

NeMo pulls a CPU-only torch and a large dependency tree; installing it beside VoxCPM risks a
torch downgrade that breaks the live TTS, so it lives in its own venv (`E:/asr-spike/.venv`)
and talks over HTTP -- exactly like the VoxCPM server, and for the same reason.

The CPU is not a compromise, it is the point. Measured 2026-08-17: Typhoon on CPU finalises a
6 s Thai utterance in 138-155 ms against faster-whisper-large-v3-turbo's 279-308 ms ON THE GPU,
at a third of the character error rate (0.032 vs 0.130 on `ref_long11`). It also frees the ~2 GB
of VRAM Whisper was holding on a 16 GB card that already carries the LLM and VoxCPM.

STDLIB ONLY -- no FastAPI, no uvicorn. Not minimalism for its own sake: this venv exists to hold
NeMo, and pip-installing a web stack into it to serve one endpoint is how a working environment
stops working. `ThreadingHTTPServer` with a lock is the same one-turn-at-a-time contract
voxcpm/app.py enforces.

WHAT THIS IS NOT. Typhoon's released checkpoint is FULL-CONTEXT (`att_context_size [-1,-1]`)
despite the "Real-time" name, so it cannot emit partials mid-utterance -- see
`E:/Claude/asr-research/docs/notes/2026-08-17-streaming-thai-asr-spike.md`. This server transcribes a
finished utterance. It is the fastest honest STT available today, not the streaming one the
overlapped loop eventually needs.

RUN (from the repo root, in the ASR venv -- NOT the VoxCPM one):

    E:/asr-spike/.venv/Scripts/python.exe asr/server.py          # port 8007
    (or just:  ./asr/run.ps1 )

Boot is ~30-60 s: NeMo import, checkpoint load, and one warmup transcription before the socket
opens -- so the first real request never pays the lazy-init cost. Same discipline as VoxCPM.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

MODEL_NAME = os.environ.get("ASR_MODEL", "typhoon-ai/typhoon-asr-realtime")
PORT = int(os.environ.get("ASR_PORT", "8007") or "8007")
SR = 16000

# ---------------------------------------------------------------- decoding knobs
# The checkpoint ships `strategy: greedy_batch` with no language model. That is a CHOICE, not a
# property of the model, and it is the cheapest accuracy lever in the repo -- NeMo 3.0.0 already
# contains batched beam search, GPU n-gram fusion and a context-biasing boosting tree, all
# reachable from the same decoding config (verified in `rnnt_decoding.py`: `greedy_batch` and
# `malsd_batch` both accept `fusion_models`).
#
# These are env vars rather than code so `adapter/arm1_*.py` can be re-run UNCHANGED per arm --
# the harness must not move when the thing under test moves. /health reports what is actually
# live, and the arm scripts record that payload, so no result file can be mistaken for another.
DEVICE = os.environ.get("ASR_DEVICE", "cpu")          # cpu is the deployed choice; see docstring
STRATEGY = os.environ.get("ASR_STRATEGY", "greedy_batch")
BEAM_SIZE = int(os.environ.get("ASR_BEAM_SIZE", "4") or "4")
NGRAM_LM = os.environ.get("ASR_NGRAM_LM", "") or None          # .nemo / ARPA / .kenlm path
NGRAM_ALPHA = float(os.environ.get("ASR_NGRAM_ALPHA", "0.0") or "0.0")
BOOST_FILE = os.environ.get("ASR_BOOST_FILE", "") or None      # one phrase per line, UTF-8
BOOST_ALPHA = float(os.environ.get("ASR_BOOST_ALPHA", "0.0") or "0.0")
BOOST_SCORE = float(os.environ.get("ASR_BOOST_SCORE", "1.0") or "1.0")
# 🔴 CUDA graphs and a fused n-gram LM do not mix on this box. NeMo's GPU LM prints
# "Triton is disabled. Version without Triton is not compatible with Cuda graphs" -- Triton has no
# Windows wheel -- and `malsd_batch` + an LM then dies inside capture with
# `RuntimeError: Capture must end on the same stream it began on`. So graphs default OFF whenever
# a fusion model is loaded, and ASR_CUDA_GRAPHS=1 forces them back on for anyone who wants to
# re-test after a Triton build exists. Graphs stay ON for plain greedy/beam, where they are free.
_CG = os.environ.get("ASR_CUDA_GRAPHS")
CUDA_GRAPHS = (_CG not in ("0", "false", "False")) if _CG is not None else not (
    os.environ.get("ASR_NGRAM_LM") or os.environ.get("ASR_BOOST_FILE"))

_MODEL = None
_DECODING = {}             # what is actually live, echoed by /health
_LOCK = threading.Lock()   # one utterance at a time; the loop is single-call by design


def _apply_decoding(m) -> dict:
    """Rebuild the model's decoding from the env knobs. Returns what was actually applied.

    Everything here is a config value NeMo already honours; nothing is patched. The one
    non-obvious part is WHERE the fusion models hang: `rnnt_decoding.py` reads
    `ngram_lm_model` / `boosting_tree` off `decoding.greedy` for the greedy strategies and off
    `decoding.beam` for every beam strategy, so the same knob has to be written to a different
    sub-key depending on the strategy. Writing it to the wrong one fails SILENTLY -- the model
    decodes fine, just without the LM -- which is exactly the kind of null result that would be
    reported as "n-gram fusion does not help Thai".
    """
    from omegaconf import OmegaConf, open_dict

    cfg = OmegaConf.structured(m.cfg.decoding)
    with open_dict(cfg):
        cfg.strategy = STRATEGY
        sub = "greedy" if STRATEGY in ("greedy", "greedy_batch") else "beam"
        if sub == "greedy":
            cfg.greedy.use_cuda_graph_decoder = CUDA_GRAPHS
        else:
            cfg.beam.allow_cuda_graphs = CUDA_GRAPHS
        if sub == "beam":
            cfg.beam.beam_size = BEAM_SIZE
            # 🔴 The checkpoint ships `beam.return_best_hypothesis: false`, so a beam decoder
            # hands back the whole N-best LIST per utterance instead of one hypothesis. Nothing
            # errors -- `_transcribe` just reads `.text` off a list and gets "". Measured: CER
            # 1.0000 on all 150 clips, which reads exactly like "beam search destroys Thai".
            cfg.beam.return_best_hypothesis = True
        if NGRAM_LM:
            cfg[sub].ngram_lm_model = NGRAM_LM
            cfg[sub].ngram_lm_alpha = NGRAM_ALPHA
        if BOOST_FILE:
            cfg[sub].boosting_tree = {"key_phrases_file": BOOST_FILE,
                                      "context_score": BOOST_SCORE}
            cfg[sub].boosting_tree_alpha = BOOST_ALPHA
    m.change_decoding_strategy(cfg)

    applied = {"strategy": STRATEGY, "device": DEVICE, "cuda_graphs": CUDA_GRAPHS}
    if STRATEGY not in ("greedy", "greedy_batch"):
        applied["beam_size"] = BEAM_SIZE
    if NGRAM_LM:
        applied["ngram_lm"] = NGRAM_LM
        applied["ngram_alpha"] = NGRAM_ALPHA
    if BOOST_FILE:
        applied["boost_file"] = BOOST_FILE
        applied["boost_alpha"] = BOOST_ALPHA
        applied["boost_score"] = BOOST_SCORE
    return applied


def _load() -> None:
    global _MODEL, _DECODING
    import nemo.collections.asr as nemo_asr

    print(f"[asr] loading {MODEL_NAME} on {DEVICE} ...", flush=True)
    t0 = time.perf_counter()
    if MODEL_NAME.endswith(".nemo"):
        # A local checkpoint. nectec/Pathumma-stt-th-fastconformer-rnnt-large ships one .nemo
        # whose file name does not match its repo name, so `from_pretrained` cannot fetch it;
        # run.ps1 points here at the downloaded file instead.
        m = nemo_asr.models.ASRModel.restore_from(MODEL_NAME, map_location=DEVICE)
    else:
        m = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME, map_location=DEVICE)
    m.eval()
    shipped = str(getattr(m.cfg.decoding, "strategy", "?"))
    _DECODING = _apply_decoding(m)
    _DECODING["shipped_strategy"] = shipped
    print(f"[asr] decoding: {_DECODING}", flush=True)
    _MODEL = m
    # Warm up before the socket opens. The first transcription pays lazy CUDA-free kernel
    # setup and dataloader construction; a caller must never wear that.
    _transcribe(np.zeros(SR, dtype=np.float32))
    print(f"[asr] ready in {time.perf_counter() - t0:.1f}s (port {PORT})", flush=True)


def _transcribe(y: np.ndarray) -> str:
    """NeMo 3.0 takes a float32 array directly -- no temp wav.

    The task-7 spike wrote a wav per call and flagged that overhead as unrepresentative.
    Removing it did NOT make this faster: measured here at 158-169 ms on the same 6 s clips
    where the spike measured 138-155 ms, with a League client and a wallpaper engine on the
    box. Writing 192 KB to an SSD was never the cost, and the prediction that it was is
    corrected rather than quietly dropped. Accuracy is identical to the spike (CER 0.032 /
    0.236 / 0.084 on the three probe clips), which is the useful thing this cross-check
    established: two code paths, same transcripts.
    """
    out = _MODEL.transcribe([y], verbose=False, batch_size=1)
    if not out:
        return ""
    first = out[0]
    if isinstance(first, (list, tuple)):       # N-best came back despite the flag above
        if not first:
            return ""
        first = first[0]
    if isinstance(first, str):
        return first.strip()
    text = str(getattr(first, "text", "")).strip()
    if not text:
        # Never let an empty string leave here quietly: it scores as CER 1.0 and looks like a
        # catastrophic accuracy result rather than a plumbing bug.
        print(f"[asr] 🔴 empty transcript from {type(first).__name__}", flush=True)
    return text


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a) -> None:      # keep the console readable; timings go to the client
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/health":
            self._json(200, {"ok": _MODEL is not None, "model": MODEL_NAME, "sample_rate": SR,
                             "decoding": _DECODING})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        u = urlparse(self.path)
        if u.path != "/asr":
            self._json(404, {"error": "not found"})
            return

        q = parse_qs(u.query)
        sr = int(q.get("sr", [str(SR)])[0])
        if sr != SR:
            # Resampling here would hide a bug in the caller and the model would quietly get
            # chipmunks. Whoever built the frames knows the rate; make them say it.
            self._json(400, {"error": f"expected {SR} Hz PCM16, got {sr}"})
            return

        n = int(self.headers.get("Content-Length") or 0)
        pcm = self.rfile.read(n) if n else b""
        if len(pcm) < 2:
            self._json(400, {"error": "empty body"})
            return

        y = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        t0 = time.perf_counter()
        with _LOCK:
            text = _transcribe(y)
        self._json(200, {"text": text,
                         "infer_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                         "audio_s": round(len(y) / SR, 3)})


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Thai on a cp1252 console
    _load()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
