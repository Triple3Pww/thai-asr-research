#!/usr/bin/env python3
"""N-best lists from a Whisper model under CTranslate2 -- the input an LLM picker would see.

    E:/fw-spike/.venv/Scripts/python.exe adapter/nbest_fw.py --model E:/fw-spike/models/typhoon-whisper-turbo-ct2 --n 5 --tag nbest_turbo5

His idea (2026-10-04): *"is the asr can return the possibility, what if we pass like top
possibility and pass it to llm to pick the right context?"* -- N-best rescoring. One ASR, its top-N
sentences with scores, an LLM picks. Before building the picker, measure its CEILING: if the right
sentence is rarely in the top N, no picker can help. `nbest_oracle.py` scores that.

Writes results/<tag>_nbest.jsonl: one row per utterance, `cands` = [{"text", "score"}...] best first,
where score is CTranslate2's length-normalised log-probability. Latency of the beam call is recorded
(batch 1, warm-up excluded) because beam search costs time and the loop is latency-first.
"""

from __future__ import annotations

import argparse
import json
import os
import site
import statistics
import time
from pathlib import Path

for sp in site.getsitepackages():
    for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
        d = Path(sp) / sub
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]

import ctranslate2  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from faster_whisper.audio import pad_or_trim  # noqa: E402
from faster_whisper.feature_extractor import FeatureExtractor  # noqa: E402
from transformers import WhisperTokenizer  # noqa: E402

BENCH = Path("E:/asr-bench/thai-contextasr-bench")
RES = Path(__file__).resolve().parent.parent / "results"


def load_rows(args, outp: Path):
    """The 682 real benchmark utterances, or any {utt_id, wav} manifest (e.g. Porjai, for picker
    training). Output is streamed per row, and a manifest run resumes from what is already written."""
    if args.manifest:
        rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    else:
        rows = sorted((json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")),
                      key=lambda r: r["utt_id"])
        rows = [dict(r, wav=str(BENCH / r["wav"])) for r in rows if r["split"] == "real"]
    done = set()
    if args.manifest and outp.exists():
        done = {json.loads(l)["utt_id"] for l in outp.open(encoding="utf-8") if l.strip()}
    return rows, done, outp.open("a" if done else "w", encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--manifest", default=None, help="any {utt_id, wav} jsonl (default: the 682 real utts)")
    args = ap.parse_args()

    m = ctranslate2.models.Whisper(args.model, device="cuda", compute_type="float16")
    tok = WhisperTokenizer.from_pretrained(args.model)
    fe = FeatureExtractor(feature_size=m.n_mels)
    prompt = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|th|>", "<|transcribe|>", "<|notimestamps|>"])

    rows, done, fout = load_rows(args, RES / f"{args.tag}_nbest.jsonl")
    lat, out = [], []
    for i, r in enumerate(rows):
        if r["utt_id"] in done:
            continue
        y, sr = sf.read(r["wav"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = (np.clip(np.round(y * 32767), -32768, 32767) / 32768).astype(np.float32)
        t0 = time.perf_counter()
        feats = pad_or_trim(fe(y))
        sv = ctranslate2.StorageView.from_array(np.ascontiguousarray(feats[None].astype(np.float32)))
        res = m.generate(sv, [prompt], beam_size=args.n, num_hypotheses=args.n, return_scores=True,
                         max_length=448, length_penalty=1.0)[0]
        dt = time.perf_counter() - t0
        cands = [{"text": tok.decode(ids, skip_special_tokens=True).strip(), "score": round(float(s), 4)}
                 for ids, s in zip(res.sequences_ids, res.scores)]
        out.append({"utt_id": r["utt_id"], "cands": cands})
        fout.write(json.dumps(out[-1], ensure_ascii=False) + "\n"); fout.flush()
        if i >= 5:
            lat.append(dt)
    fout.close()
    lat.sort()
    run = {"model": args.model, "n": args.n, "utts": len(rows), "manifest": args.manifest,
           "latency_ms_p50": round(1000 * statistics.median(lat), 1) if lat else None,
           "latency_ms_p90": round(1000 * lat[int(0.9 * len(lat))], 1) if lat else None}
    (RES / f"{args.tag}_run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(json.dumps(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
