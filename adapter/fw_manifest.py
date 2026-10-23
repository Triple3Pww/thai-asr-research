#!/usr/bin/env python3
"""Transcribe a NeMo manifest with a Whisper model under CTranslate2 -> {id, text} jsonl.

    E:/fw-spike/.venv/Scripts/python.exe adapter/fw_manifest.py \\
        --model E:/fw-spike/models/typhoon-whisper-large-v3-ct2 \\
        --manifest E:/asr-bench/typhoon-eval/manifest_tvspeech.jsonl --out results/tvspeech_tw_hyps.jsonl

WHY (2026-10-06). The bar has to be scored on every extra test set by the SAME scorer as the RNNTs
(`eval_manifest.py --hyps`), so the Whisper side only emits text. Runs in E:/fw-spike/.venv (CLAUDE.md:
nothing added to the ASR venv). Greedy (beam 1), language th, no timestamps, no conditioning on
previous text, PCM16 audio as on disk; clips here are <= 30 s, one window each.
"""

from __future__ import annotations

import argparse
import json
import os
import site
from pathlib import Path

for sp in site.getsitepackages():                       # CUDA 12 DLLs shipped as pip wheels
    for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
        d = Path(sp) / sub
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beam", type=int, default=1)
    # 2026-10-06: WITHOUT timestamps a ~29 s Thai clip hits Whisper's 448-token decoder cap (~1 token
    # per Thai char) and the rest is silently dropped -- TVSpeech CER 0.2004, 14.3% of it deletion runs
    # >= 20 chars. With timestamps the decoder seeks past the last finished segment and continues.
    ap.add_argument("--timestamps", action="store_true")
    # ...and that diagnosis was WRONG (refs need <= 398 tokens). Measured: all 570 outputs stop at
    # <= 225 tokens = n_text_ctx // 2, the reference decoder's default sample_len, which faster-whisper
    # copies. Thai needs ~1.2 chars/token, so a 30 s clip is cut short. Allow the whole context.
    ap.add_argument("--max-new-tokens", type=int, default=440)
    # ...and lifting max_new_tokens changed nothing (TVSpeech 0.2036, 170 clips still cut): the model
    # generates Thai in fragment tokens, so the 448-token decoder budget runs out at ~224 canonical
    # tokens whatever the setting. The fair fix is shorter windows: VAD chunks of <= N seconds.
    ap.add_argument("--vad-max-s", type=float, default=0, help="VAD-chunk the audio, chunks <= this many s (0 = off)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="file of ids to restrict to")
    args = ap.parse_args()
    from faster_whisper import WhisperModel
    m = WhisperModel(args.model, device="cuda", compute_type="float16")
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    if args.ids:
        keep = set(open(args.ids, encoding="utf-8").read().split())
        rows = [r for r in rows if r["id"] in keep]
    if args.limit:
        rows = rows[: args.limit]
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(rows):
            if args.vad_max_s > 0:
                # faster-whisper's own vad_filter RE-JOINS its chunks and still decodes 30 s windows
                # (tested: 20 truncated clips 0.323 -> 0.316). Cut here, decode each piece on its own.
                from faster_whisper.audio import decode_audio
                from faster_whisper.vad import VadOptions, get_speech_timestamps
                wav = decode_audio(r["audio_filepath"], sampling_rate=16000)
                ts = get_speech_timestamps(wav, VadOptions(max_speech_duration_s=args.vad_max_s,
                                                           min_silence_duration_ms=300))
                parts = []
                for t in ts or [{"start": 0, "end": len(wav)}]:
                    sg, _ = m.transcribe(wav[t["start"]: t["end"]], language="th", task="transcribe",
                                         beam_size=args.beam, without_timestamps=True,
                                         condition_on_previous_text=False, vad_filter=False,
                                         max_new_tokens=args.max_new_tokens)
                    parts.append("".join(x.text for x in sg).strip())
                f.write(json.dumps({"id": r["id"], "text": " ".join(parts).strip()}, ensure_ascii=False) + "\n")
                continue
            segs, _ = m.transcribe(r["audio_filepath"], language="th", task="transcribe", beam_size=args.beam,
                                   without_timestamps=not args.timestamps, condition_on_previous_text=False, vad_filter=args.vad_max_s > 0,
                                   vad_parameters={"max_speech_duration_s": args.vad_max_s} if args.vad_max_s > 0 else None,
                                   max_new_tokens=args.max_new_tokens)
            f.write(json.dumps({"id": r["id"], "text": "".join(s.text for s in segs).strip()}, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(i, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
