#!/usr/bin/env python3
"""BATCHED Whisper greedy transcription of a NeMo manifest under CTranslate2 -> {id, text} jsonl (resumable).

    E:/fw-spike/.venv/Scripts/python.exe adapter/fw_batch.py --model E:/fw-spike/models/nectec_Pathumma-whisper-th-large-v3-ct2 \\
        --manifest M.jsonl --out O.jsonl [--bs 16] [--device cuda]

WHY (2026-10-11, loop). `fw_manifest.py` decodes one clip at a time: Pathumma-L took 4 h 10 min for shard 0's
21.5k hard clips (~86 clips/min), which makes a label-verification pass over more GigaSpeech2-train shards
too slow for a 24 h loop. Here B clips (all <= 30 s, one window each, as fw_manifest without --vad-max-s)
share one encoder/decoder call through `ctranslate2.models.Whisper.generate`, same prompt as
fw_manifest (<|startoftranscript|><|th|><|transcribe|><|notimestamps|>), greedy, no previous-text
conditioning. Same venv (E:/fw-spike/.venv); nothing added to the ASR venv.
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
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import ctranslate2
    import numpy as np
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    wm = WhisperModel(args.model, device=args.device, compute_type="float16" if args.device == "cuda" else "int8")
    fe, tok = wm.feature_extractor, wm.hf_tokenizer
    prompt = [tok.token_to_id(t) for t in ("<|startoftranscript|>", "<|th|>", "<|transcribe|>", "<|notimestamps|>")]
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8") if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    n_frames = fe.nb_max_frames
    with open(args.out, "a", encoding="utf-8", newline="\n") as f:
        for i in range(0, len(todo), args.bs):
            chunk = todo[i: i + args.bs]
            feats = []
            for r in chunk:
                wav = decode_audio(r["audio_filepath"], sampling_rate=16000)
                mel = fe(wav)[:, :n_frames]
                if mel.shape[1] < n_frames:
                    mel = np.pad(mel, ((0, 0), (0, n_frames - mel.shape[1])))
                feats.append(mel)
            sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(np.stack(feats).astype(np.float32)))
            res = wm.model.generate(sv, [prompt] * len(chunk), beam_size=1, max_length=448,
                                    suppress_blank=True, return_scores=False)
            for r, o in zip(chunk, res):
                ids = [t for t in o.sequences_ids[0] if t < tok.token_to_id("<|endoftext|>")]
                f.write(json.dumps({"id": r["id"], "text": tok.decode(ids).strip()}, ensure_ascii=False) + "\n")
            f.flush()
            if (i // args.bs) % 50 == 0:
                print(i + len(chunk), "/", len(todo), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
