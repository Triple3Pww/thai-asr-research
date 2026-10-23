#!/usr/bin/env python3
"""Loop iteration 7: NECTEC drafts, a large Whisper verifies -- lossless speculative decoding across two ASRs.

    PYTHONUTF8=1 python adapter/spec_whisper.py --model nectec/Pathumma-whisper-th-large-v3 --limit 50 --check
    PYTHONUTF8=1 python adapter/spec_whisper.py --model nectec/Pathumma-whisper-th-large-v3 --tag spec_patL

WHY (2026-10-08 loop). A ROVER of NECTEC + A2 + turbo + Pathumma-whisper-large + typhoon-whisper-large beats
NECTEC by 6.1% (P .02) with our own transcripts -- but each large Whisper costs ~1,030 ms/turn (CTranslate2),
and int8 only reaches 952 ms. The cost is the 32-layer decoder run once PER TOKEN. NECTEC's transcript is
available after 43 ms and agrees with Whisper on most tokens, so use it as the draft:

  1. encode the audio once (Whisper encoder, fp16)
  2. draft = Whisper-tokenisation of NECTEC's transcript + <|endoftext|>
  3. ONE decoder pass over prompt + accepted + draft (teacher forcing); accept the longest draft prefix whose
     tokens equal Whisper's own argmax, then append Whisper's argmax at the first mismatch
  4. re-sync: find where Whisper's text so far ends inside NECTEC's text, re-tokenise the rest as the new draft
  5. repeat until <|endoftext|>

Every emitted token is Whisper's greedy argmax given the same prefix, so the result is Whisper's greedy output
(numerics aside) -- the draft only decides how many tokens each pass verifies. `--check` runs plain greedy
`generate` beside it and reports token-exact agreement and both latencies.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from ctc_head import pcm16  # noqa: E402

BENCH = Path("E:/asr-bench/thai-contextasr-bench")


class Spec:
    def __init__(self, name, dev="cuda"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.proc = WhisperProcessor.from_pretrained(name)
        self.m = WhisperForConditionalGeneration.from_pretrained(name, dtype=torch.float16).to(dev).eval()
        self.tok = self.proc.tokenizer
        self.dev = dev
        t = self.tok
        self.prompt = [t.convert_tokens_to_ids(x) for x in
                       ("<|startoftranscript|>", "<|th|>", "<|transcribe|>", "<|notimestamps|>")]
        self.eot = t.eos_token_id
        self.max_len = 440

    @torch.no_grad()
    def features(self, y):
        f = self.proc(y, sampling_rate=16000, return_tensors="pt").input_features
        return f.to(self.dev, torch.float16)

    @torch.no_grad()
    def encode(self, feats):
        return self.m.model.encoder(feats).last_hidden_state

    @torch.no_grad()
    def greedy(self, feats):
        out = self.m.generate(feats, language="th", task="transcribe", do_sample=False, num_beams=1,
                              max_new_tokens=self.max_len)
        ids = [i for i in out[0].tolist() if i < self.eot]
        return self.tok.decode(ids, skip_special_tokens=True).strip()

    def _draft_from(self, emitted_text: str, draft_text: str) -> list[int]:
        """Rest of NECTEC's text after the point where Whisper's emitted text ends (longest suffix match)."""
        if not emitted_text:
            rest = draft_text
        else:
            rest = None
            for L in range(min(12, len(emitted_text)), 1, -1):
                k = draft_text.find(emitted_text[-L:])
                if k >= 0:
                    rest = draft_text[k + L:]
                    break
            if rest is None:
                return []
        return self.tok.encode(rest, add_special_tokens=False) + [self.eot]

    @torch.no_grad()
    def spec_cached(self, enc, draft_text: str):
        """`spec` with a KV cache: a pass feeds only tokens the cache has not seen; on a mismatch the cache
        is cropped back to the accepted prefix. Same decisions, fewer FLOPs."""
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache
        dec = self.m.model.decoder
        head = self.m.proj_out
        cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
        done = 0                                       # tokens of prompt+out already in the cache
        out: list[int] = []
        draft = self.tok.encode(draft_text, add_special_tokens=False) + [self.eot]
        passes = 0
        while len(out) < self.max_len:
            draft = draft[: max(0, 447 - len(self.prompt) - len(out))]   # Whisper has 448 decoder positions
            seq = self.prompt + out + draft
            new = seq[done:]
            h = dec(input_ids=torch.tensor([new], device=self.dev), encoder_hidden_states=enc,
                    past_key_values=cache, use_cache=True).last_hidden_state
            first = len(self.prompt) + len(out) - 1 - done     # index in `new` predicting draft[0]
            pred = head(h[0, first:]).argmax(-1).tolist()
            passes += 1
            k = 0
            while k < len(draft) and pred[k] == draft[k]:
                k += 1
            out += draft[:k]
            done = len(self.prompt) + len(out)          # cache now valid up to here (draft[k:] rejected)
            cache.crop(done)
            out.append(pred[k])
            if out[-1] == self.eot:
                out = out[:-1]
                break
            draft = self._draft_from(self.tok.decode(out, skip_special_tokens=True), draft_text)
        return self.tok.decode(out, skip_special_tokens=True).strip(), passes

    @torch.no_grad()
    def spec(self, enc, draft_text: str):
        dec = self.m.model.decoder
        head = self.m.proj_out
        out: list[int] = []
        draft = self.tok.encode(draft_text, add_special_tokens=False) + [self.eot]
        passes = 0
        while len(out) < self.max_len:
            draft = draft[: max(0, 447 - len(self.prompt) - len(out))]
            seq = self.prompt + out + draft
            x = torch.tensor([seq], device=self.dev)
            h = dec(input_ids=x, encoder_hidden_states=enc).last_hidden_state
            pred = head(h[0, len(self.prompt) + len(out) - 1:]).argmax(-1).tolist()   # predictions for positions of draft
            passes += 1
            k = 0
            while k < len(draft) and pred[k] == draft[k]:
                k += 1
            out += draft[:k]
            # pred has len(draft)+1 entries: at a mismatch take Whisper's token there; if the whole draft
            # (possibly empty -- re-sync failed) was accepted, take Whisper's next token after it. Either way
            # this pass emits >= 1 token, so an empty draft degrades to plain greedy, never to a stop.
            out.append(pred[k])
            if out[-1] == self.eot:
                out = out[:-1]
                break
            emitted = self.tok.decode(out, skip_special_tokens=True)
            draft = self._draft_from(emitted, draft_text)
        return self.tok.decode(out, skip_special_tokens=True).strip(), passes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", default=str(ROOT / "results/nectec_fc_greedy_hyps.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-cache", action="store_true", help="re-run the whole prefix every pass (first version)")
    a = ap.parse_args()
    S = Spec(a.model)
    D = {json.loads(l)["utt_id"]: json.loads(l)["text"] for l in open(a.draft, encoding="utf-8")}
    rows = [json.loads(l) for l in (BENCH / "data/utterances.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["split"] == "real"][: a.limit or None]
    res, t_spec, t_greedy, same, npass = {}, [], [], 0, []
    for i, r in enumerate(rows):
        y, _ = sf.read(BENCH / r["wav"], dtype="float32")
        y = pcm16(y.mean(axis=1) if y.ndim > 1 else y)
        feats = S.features(y)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        enc = S.encode(feats)
        text, p = (S.spec if a.no_cache else S.spec_cached)(enc, D[r["utt_id"]])
        torch.cuda.synchronize(); t1 = time.perf_counter()
        if i >= 3:
            t_spec.append(1000 * (t1 - t0)); npass.append(p)
        res[r["utt_id"]] = text
        if a.check:
            torch.cuda.synchronize(); t2 = time.perf_counter()
            g = S.greedy(feats)
            torch.cuda.synchronize(); t3 = time.perf_counter()
            if i >= 3:
                t_greedy.append(1000 * (t3 - t2))
            same += g.strip() == text.strip()
            if g != text and i < 40:
                print("DIFF", r["utt_id"], "\n  spec  :", text[:90], "\n  greedy:", g[:90], flush=True)
        if i % 100 == 0:
            print(i, len(rows), flush=True)
    rep = {"model": a.model, "n": len(res), "spec_ms_p50": float(np.median(t_spec)),
           "spec_ms_p90": float(np.percentile(t_spec, 90)), "passes_p50": float(np.median(npass))}
    if a.check:
        rep.update({"greedy_ms_p50": float(np.median(t_greedy)), "identical_to_greedy": same})
    print(rep)
    if a.tag:
        with (ROOT / "results" / f"{a.tag}_hyps.jsonl").open("w", encoding="utf-8", newline="\n") as g:
            for u, t in res.items():
                g.write(json.dumps({"utt_id": u, "condition": "none", "text": t}, ensure_ascii=False) + "\n")
        (ROOT / "results" / f"{a.tag}_run.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
