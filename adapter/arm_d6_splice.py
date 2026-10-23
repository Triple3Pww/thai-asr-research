#!/usr/bin/env python3
"""ARM D6 -- the two-model splice: our RNN-T writes the transcript, Typhoon's CTC spots entities.

    # their released pipeline, unmodified, on our harness (the claim under test):
    py adapter/arm_d6_splice.py --mode ctc  --condition bias@50
    py adapter/arm_d6_splice.py --mode ctc  --condition dis@50     # equal-size control
    py adapter/arm_d6_splice.py --mode ctc  --condition none       # no spotting at all
    # ours: the RNN-T transcript with the same spots spliced in
    py adapter/arm_d6_splice.py --mode rnnt --condition bias@50

WHY TWO MODELS. `A26`'s spotter needs a per-frame `[T, V]` CTC posterior matrix: it scores a
keyword by Viterbi over every SPM segmentation of it, normalised against a per-frame filler. A
transducer emits a `[T, U, V]` lattice conditioned on its own prefix -- there is nothing shaped
like a per-frame posterior to read. Hence two models over the SAME frozen encoder:
`typhoon-asr-realtime-nemo-ctc` declares `base_model: typhoon-asr-realtime`, and all three
tokenizers here hash to md5 88e04338795f42f0492c953e461ac785, so spliced pieces land in one
vocabulary. Frames align too: both subsample x8 at 10 ms, i.e. 0.08 s per frame.

WHY `--mode ctc` RUNS FIRST, AND WHY IT IS NOT THE CONTRIBUTION. It is their code path end to
end (their spotter, their splice-into-CTC-greedy), so it measures **their published claim on our
harness** -- `A26` reports entity recall .158 -> .852 with boosting. Ours is a different starting
point: this repo's unbiased recall is already **.328**, so "their number" cannot be read off their
card. `--mode rnnt` is the contribution: keep the better transcript (RNN-T CER beats CTC) and take
only the spots from the CTC head.

🔴 THE CONTROL THE GATE NOW REQUIRES. `E:/Claude/asr-research/RESEARCH-FINDINGS.md` F8d: the benchmark's B-CER/U-CER split
moves with bias-list SIZE independently of the model, so an arm may only be compared against a
**same-size** condition. Run `dis@N` beside every `bias@N` and quote that difference.

🔴 A LOADING DEFECT, worked around here rather than in their file. Their `KeywordSpotter` calls
`ASRModel.restore_from`, which (a) is abstract in NeMo 3.0.0 and (b) chokes on this NeMo-2.5.0
checkpoint with `ConfigAttributeError: Missing key key_phrase_items_list` -- `BoostingTreeModelConfig`
grew a field after the checkpoint was exported. Fixed by restoring the concrete
`EncDecCTCModelBPE` with a patched config (`model_config.patched.yaml`, generated once by
`_patch_ctc_config()`), leaving their released code untouched so it stays quotable.

⚠️ MEASURED BEFORE BUILDING: the spotter separates Thai entities cleanly (gold -0.64 / 0.0 against
decoys -2.4 to -4.8, their recommended tau is -1.5) but scores **Latin-spelled** gold entities at
-16 to -20, indistinguishable from an unrelated decoy -- a Thai speaker says the Thai pronunciation,
and a Latin string has no acoustic path through a Thai-character vocabulary. **108 of 439 gold
mentions (25%) are spelled in Latin**, and so are ~25% of every bias list. Those are unspottable by
construction, which bounds this arm before it runs. Transliterating them is a separate, legitimate
arm -- and it must be applied to gold and distractors alike, or it is an oracle leak (only gold
entities carry `form_th` in the annotations).
"""

from __future__ import annotations

import os
import argparse
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

BENCH = Path("E:/asr-bench/thai-contextasr-bench")
CTC_DIR = Path("E:/asr-bench/nemo-ctc")
CTC_NEMO = CTC_DIR / "nemo-ctc.nemo"
PATCHED = CTC_DIR / "model_config.patched.yaml"
HERE = Path(__file__).resolve().parent
# Results dir -- `results/` at the repo root, beside this `adapter/` package.
# ADAPTER_OUT still overrides it, for a scratch run that must not touch the record.
OUT = Path(os.environ.get("ADAPTER_OUT") or (HERE.parent / "results"))
RNNT_MODEL = "typhoon-ai/typhoon-asr-realtime"
SR = 16000
FRAME_S = 0.08


def pcm16_roundtrip(y):
    """float32 -> int16 -> float32, as the HTTP path does. See arm_d4_boosting.py."""
    import numpy as np

    pcm = (np.clip(y, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def _patch_ctc_config() -> Path:
    """Add the field NeMo 3.0.0 expects and the 2.5.0 checkpoint lacks. Idempotent."""
    from omegaconf import OmegaConf

    if PATCHED.exists():
        return PATCHED
    cfg = OmegaConf.load(CTC_DIR / "model_config.yaml")
    for sub in ("greedy", "beam"):
        d = cfg.get("decoding", {}).get(sub, None)
        if d is not None and "boosting_tree" in d and "key_phrase_items_list" not in d.boosting_tree:
            d.boosting_tree.key_phrase_items_list = None
    OmegaConf.save(cfg, PATCHED)
    return PATCHED


class Spotter:
    """Their `KeywordSpotter`, with only the constructor replaced."""

    def __init__(self, device: str = "cuda"):
        sys.path.insert(0, str(CTC_DIR))
        from nemo.collections.asr.models import EncDecCTCModelBPE

        self.m = EncDecCTCModelBPE.restore_from(
            str(CTC_NEMO), override_config_path=str(_patch_ctc_config()),
            map_location=device).eval()
        self.device = device
        sp = self.m.tokenizer.tokenizer
        self.sp = sp
        self.piece_ids: dict[str, tuple] = {}
        for i in range(sp.get_piece_size()):
            p = sp.id_to_piece(i).lstrip("\u2581")       # no-leading-space, as they do
            self.piece_ids[p] = self.piece_ids.get(p, ()) + (i,)
        self.blank = sp.get_piece_size()

    def logprobs(self, wav):
        import torch

        with torch.no_grad():
            x = torch.as_tensor(wav, dtype=torch.float32, device=self.device)[None]
            ln = torch.tensor([x.shape[1]], device=self.device)
            out = self.m.forward(input_signal=x, input_signal_length=ln)
            lp, enc_len = out[0], out[1]
            return lp[0, : int(enc_len[0])].float().cpu().numpy()

    def normalize_kw(self, kw: str) -> str:
        return "".join(self.sp.encode(kw, out_type=str)).replace("\u2581", "")

    def greedy_emit(self, lp):
        """CTC greedy collapse -> [(frame, piece)], their word_boost.py main loop."""
        ids = lp.argmax(axis=1)
        emit, prev = [], -1
        for t, i in enumerate(ids):
            if i != prev and i != self.blank:
                pc = self.sp.id_to_piece(int(i))
                if not (pc.startswith("<") and pc.endswith(">")):
                    emit.append((t, pc))
            prev = i
        return emit

    def spots(self, lp, keywords: list[str], tau: float, max_gap: int):
        """Accepted, non-overlapping keyword spans, highest score first."""
        from kws_spot import build_lattice, spot
        if getattr(self, "fast", False):
            # 2026-10-07: numba port, bit-identical on 3,503 real pairs (`kws_fast.py --check`), 16x faster
            from kws_fast import spot_fast as spot  # noqa: F811

        cands = []
        for kw in keywords:
            k = self.normalize_kw(kw.replace(" ", ""))
            if not k:
                continue
            edges, ok, _ = build_lattice(k, self.piece_ids)
            if not ok:
                continue
            sc, t0, t1 = spot(lp, edges, len(k), self.blank, max_gap=max_gap)
            if sc >= tau:
                cands.append((sc, t0, t1, kw))
        taken: list[tuple] = []
        for sc, s_, e_, kw in sorted(cands, reverse=True):
            if all(e_ < ts or s_ > te for _, ts, te, _ in taken):
                taken.append((sc, s_, e_, kw))
        return sorted(taken, key=lambda x: x[1])


class HeadSpotter(Spotter):
    """2026-10-07: spots from a CTC head on NECTEC's OWN frozen encoder (`ctc_head.py`), not Typhoon's
    second model. Shares the RNN-T's model object, so one model is loaded; scoring, lattice and
    acceptance are the inherited Typhoon code, unchanged -- only the posterior's source differs."""

    def __init__(self, nemo_model, head_path: str, device: str = "cuda"):
        import torch

        sys.path.insert(0, str(CTC_DIR))          # their kws_spot.py lives beside their model

        self.m = nemo_model
        self.device = device
        ck = torch.load(head_path, map_location=device)
        D, V, H = ck["d"], ck["vocab"], ck.get("hidden", 0)
        self.head = (torch.nn.Sequential(torch.nn.Linear(D, H), torch.nn.GELU(), torch.nn.Linear(H, V + 1))
                     if H else torch.nn.Linear(D, V + 1)).to(device)
        self.head.load_state_dict(ck["state"])
        self.head.eval()
        sp = self.m.tokenizer.tokenizer
        self.sp = sp
        self.piece_ids = {}
        for i in range(sp.get_piece_size()):
            pc = sp.id_to_piece(i).lstrip("▁")
            self.piece_ids[pc] = self.piece_ids.get(pc, ()) + (i,)
        self.blank = V

    def logprobs(self, wav):
        import torch

        # 2026-10-07: NeMo 3.0.0 `transcribe()` returns with model.training False but ENCODER.training
        # True. Called directly after it, the encoder ran with dropout and UPDATED BatchNorm statistics
        # every utterance -- the first head run drifted NECTEC's own transcript in 481/682 utts. Force eval.
        self.m.preprocessor.eval(); self.m.encoder.eval()
        with torch.no_grad():
            x = torch.as_tensor(wav, dtype=torch.float32, device=self.device)[None]
            ln = torch.tensor([x.shape[1]], device=self.device)
            f, fl = self.m.preprocessor(input_signal=x, length=ln)
            enc, el = self.m.encoder(audio_signal=f, length=fl)
            lp = torch.log_softmax(self.head(enc[0, :, : int(el[0])].T.float()), -1)
            return lp.cpu().numpy()


def splice(emit: list[tuple], taken: list[tuple]) -> str:
    """Their splice: walk the frame-tagged pieces, replacing each spotted span."""
    pieces, k = [], 0
    for _, s_, e_, kw in taken:
        while k < len(emit) and emit[k][0] < s_:
            pieces.append(emit[k][1]); k += 1
        pieces.append("\u2581" + kw)
        while k < len(emit) and emit[k][0] <= e_:
            k += 1
    pieces += [p for _, p in emit[k:]]
    return "".join(pieces).replace("\u2581", " ").strip()


class RnntTranscriber:
    """Our model, with per-token frame indices so spots can be spliced into ITS text."""

    def __init__(self, device: str = "cuda", model: str = RNNT_MODEL):
        import nemo.collections.asr as nemo_asr
        from omegaconf import OmegaConf, open_dict

        # 2026-10-06: any RNN-T on the same 0.08 s frame grid works -- the splice is by frame index and
        # inserts the keyword as TEXT, so the transcriber need not share the spotter's encoder or
        # tokenizer. NECTEC's FastConformer: window_stride 0.01 x subsampling 8 = 0.08 s (checked).
        self.m = (nemo_asr.models.ASRModel.restore_from(model, map_location=device) if model.endswith(".nemo")
                  else nemo_asr.models.ASRModel.from_pretrained(model, map_location=device))
        self.m.eval()
        cfg = OmegaConf.structured(self.m.cfg.decoding)
        with open_dict(cfg):
            cfg.strategy = "greedy_batch"
            cfg.compute_timestamps = True          # needed to place a spot in OUR text
        self.m.change_decoding_strategy(cfg)

    def transcribe(self, y):
        out = self.m.transcribe([y], verbose=False, batch_size=1, timestamps=True)
        h = out[0] if out else None
        if isinstance(h, (list, tuple)):
            h = h[0] if h else None
        text = (h if isinstance(h, str) else str(getattr(h, "text", ""))).strip()
        return text, h


def rnnt_splice(text: str, hyp, taken: list[tuple], pad_start: int = 0, pad_end: int = 0) -> str:
    """Replace, in OUR transcript, the character range covered by each accepted spot.

    The transducer's `timestamp["char"]` (NeMo's per-token entries) gives each emitted token a
    start frame in the SAME frame grid as the CTC head -- both heads sit on one encoder at x8
    subsampling. Tokens whose start frame lies inside a spot's [t0, t1] are the ones the spot
    is claiming; they get replaced by the keyword. If timestamps are unavailable we fall back to
    appending nothing and say so, rather than guessing a position.
    """
    ts = getattr(hyp, "timestamp", None) or {}
    toks = ts.get("char") or ts.get("token") or []
    if not toks or not taken:
        return text
    out, used = [], set()
    for i, tk in enumerate(toks):
        f = tk.get("start_offset", tk.get("start", None))
        piece = tk.get("char", tk.get("token", ""))
        if isinstance(piece, list):
            piece = "".join(piece)
        hit = None
        if f is not None:
            for j, (_, s_, e_, kw) in enumerate(taken):
                # 2026-10-07: pads, in frames. With 0/0 a token that starts just past the spot's last
                # frame survives beside the inserted name ("...ธนสาร ศาสตร์", "...ประเสริฐ ริฐ") --
                # the reason recall rose +97 on NECTEC while B-CER did not move.
                if s_ - pad_start <= f <= e_ + pad_end:
                    hit = j
                    break
        if hit is None:
            out.append(piece)
        elif hit not in used:
            used.add(hit)
            out.append(" " + taken[hit][3] + " ")
    for j, (_, _, _, kw) in enumerate(taken):     # spots that matched no token: append
        if j not in used:
            out.append(" " + kw)
    return "".join(out).replace("\u2581", " ").strip()


def load_lists(condition: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if condition == "none":
        return out
    with gzip.open(BENCH / "data/bias_lists.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("condition") == condition:
                out[d["utt_id"]] = list(d.get("entries") or [])
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ctc", choices=("ctc", "rnnt"))
    ap.add_argument("--condition", default="bias@50")
    ap.add_argument("--split", default="real", choices=("real", "synth"))
    ap.add_argument("--tau", type=float, default=-1.5, help="their recommended default")
    ap.add_argument("--max-gap", type=int, default=6)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rnnt-model", default=RNNT_MODEL, help="HF id or .nemo for --mode rnnt")
    ap.add_argument("--spotter", default="typhoon", choices=("typhoon", "head"),
                    help="typhoon = their 115M CTC model (2 encoders); head = ctc_head.py on NECTEC (1 encoder)")
    ap.add_argument("--fast-spot", action="store_true", help="numba spotter (kws_fast.py), same answers")
    ap.add_argument("--head", default="", help="checkpoints/ctchead_<tag>.pt for --spotter head")
    ap.add_argument("--pad-start", type=int, default=0, help="frames before a spot whose tokens it also claims")
    ap.add_argument("--pad-end", type=int, default=0, help="frames after a spot whose tokens it also claims")
    ap.add_argument("--lists-file", default="", help="2026-10-08: JSON {utt_id: [entries]} built WITHOUT the "
                    "benchmark (e.g. catalog retrieval from the draft); output is labelled condition 'none' "
                    "because the system was given no list")
    args = ap.parse_args()

    import soundfile as sf

    utts = [json.loads(l) for l in
            (BENCH / "data/utterances.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [u for u in utts if u.get("split") == args.split]
    if args.sample and args.sample < len(rows):
        import random

        random.Random(20260912).shuffle(rows)
        rows = sorted(rows[: args.sample], key=lambda r: r["utt_id"])
    if args.lists_file:
        lists = json.loads(Path(args.lists_file).read_text(encoding="utf-8"))
        args.condition = "none"
    else:
        lists = load_lists(args.condition)
    tag = args.tag or f"d6_{args.mode}_{args.split}_{args.condition.replace('@','')}_tau{args.tau:g}"
    print(f"mode {args.mode} | split {args.split} | {len(rows)} utts | condition {args.condition} "
          f"| tau {args.tau}")

    rn = RnntTranscriber(args.device, args.rnnt_model) if args.mode == "rnnt" else None
    sp = HeadSpotter(rn.m, args.head, args.device) if args.spotter == "head" else Spotter(args.device)
    sp.fast = args.fast_spot

    OUT.mkdir(parents=True, exist_ok=True)
    hyps = OUT / f"{tag}_hyps.jsonl"
    audio_s = 0.0
    n_spot = 0
    t0 = time.perf_counter()
    with hyps.open("w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(rows, 1):
            y, sr = sf.read(BENCH / r["wav"], dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            audio_s += len(y) / SR
            y = pcm16_roundtrip(y)
            lp = sp.logprobs(y)
            kws = lists.get(r["utt_id"], [])
            taken = sp.spots(lp, kws, args.tau, args.max_gap) if kws else []
            n_spot += len(taken)
            if args.mode == "ctc":
                text = splice(sp.greedy_emit(lp), taken)
            else:
                base, hyp = rn.transcribe(y)
                text = rnnt_splice(base, hyp, taken, args.pad_start, args.pad_end)
            f.write(json.dumps({"utt_id": r["utt_id"], "condition": args.condition,
                                "text": text}, ensure_ascii=False) + "\n")
            if i % 100 == 0 or i == len(rows):
                print(f"  [{i}/{len(rows)}] {time.perf_counter()-t0:.0f}s  spots so far {n_spot}",
                      flush=True)
    el = time.perf_counter() - t0
    (OUT / f"{tag}_run.json").write_text(json.dumps(
        {"tag": tag, "mode": args.mode, "condition": args.condition, "split": args.split,
         "n_utts": len(rows), "tau": args.tau, "max_gap": args.max_gap,
         "spots_accepted": n_spot, "audio_h": round(audio_s / 3600, 3),
         "rtf": round(el / audio_s, 4), "align_backend": "dp"},
        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"  -> {hyps}  ({audio_s/3600:.2f} h, RTF {el/audio_s:.3f}, {n_spot} spots accepted)")

    metrics = OUT / f"{tag}_metrics.json"
    p = subprocess.run([sys.executable, str(HERE / "_score.py"), "--align", "dp",
                        "--hyps", str(hyps), "--out", str(metrics)],
                       capture_output=True, text=True, encoding="utf-8")
    print((p.stdout or "").strip()[-300:] or "(no output)")
    if p.returncode != 0:
        print((p.stderr or "")[-1500:], file=sys.stderr)
        return p.returncode
    m = json.loads(metrics.read_text(encoding="utf-8")).get(args.condition, {})
    if m:
        g = m["counts"]["gold"]
        print(f"  CER {m['cer']:.5f}  U-CER {m['u_cer']:.5f}  B-CER {m['b_cer']:.5f}  "
              f"recall {m['entity_recall']:.5f} ({g[0]}/{g[1]})")
        print("  🔴 compare against the SAME-SIZE condition (dis@N), not against `none`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
