#!/usr/bin/env python3
"""Fine-tune typhoon-asr-realtime (115M FastConformer-RNNT) on teacher labels -> the real-time student.

    python adapter/finetune_rt.py --train E:/Claude/datasets/yodas2_th/train_rover8.jsonl --tag s1
    python adapter/finetune_rt.py ... --max-steps 50 --tag smoke           # does it run at all

The direction (2026-10-03): *"make the a model that beat typhoon best score that can be use in
realtime."* typhoon-whisper-turbo is 1.2 s per turn here and still loses to the bar; the 115M RNNT
the voice loop already serves is ~0.1 s. So the student is that model, and the teacher is the
all-8 ROVER that beat typhoon-whisper-large-v3 (`idea-stage/IDEA_REPORT.md`).

Choices, each with its why:
  * LR 5e-5 constant after 200 warmup, not the shipped 1e-3/5000-warmup recipe: this is a few-hour
    adaptation of a model trained on ~11k h. A pre-training LR would erase what it knows, and
    there is no replay data (Typhoon's corpus is not ours).
  * Validation = a held-out slice of the TEACHER labels (by video, so no speaker/recording leaks
    across), never the benchmark: the benchmark is touched once, at the end, by `eval_rt_local.py`.
  * bf16-mixed, batch 16, durations 2-20 s, 1 GPU. Writes `checkpoints/rt_student_<tag>.nemo`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parent.parent


def split_by_video(train: Path, frac: float, seed: int) -> tuple[Path, Path]:
    rows = [json.loads(l) for l in train.open(encoding="utf-8") if l.strip()]
    vids = sorted({r["id"].rsplit("-", 3)[0] for r in rows})
    random.Random(seed).shuffle(vids)
    held = set(vids[: max(1, int(len(vids) * frac))])
    tr, va = train.with_suffix(".tr.jsonl"), train.with_suffix(".va.jsonl")
    with tr.open("w", encoding="utf-8", newline="\n") as a, va.open("w", encoding="utf-8", newline="\n") as b:
        for r in rows:
            (b if r["id"].rsplit("-", 3)[0] in held else a).write(json.dumps(r, ensure_ascii=False) + "\n")
    return tr, va


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--loss", default="fast", choices=("fast", "default", "pytorch"))
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--vram-frac", type=float, default=0.9,
                    help="cap torch's CUDA cache at this fraction of VRAM (0 = no cap)")
    # 2026-10-06: the student can be any FastConformer-RNNT, e.g. NECTEC's .nemo (restore_from)
    ap.add_argument("--base", default="typhoon-ai/typhoon-asr-realtime")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--val", default=None, help="explicit validation manifest; skips split_by_video")
    args = ap.parse_args()
    # 2026-10-11: on Windows (WDDM sysmem fallback) cudaMalloc past full VRAM silently succeeds in SYSTEM RAM
    # instead of raising OOM, so torch's caching allocator never frees its cache and grows into RAM until the
    # commit limit dies (gs2conf: private 16.7 -> 33 GB, at bs 8 and bs 4 alike). A cap makes the allocator
    # hit its own limit first, free cached blocks and retry.
    import torch
    if args.vram_frac and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.vram_frac)
    import sys
    if not sys.flags.utf8_mode:
        # NeMo opens manifests with open() and no encoding; on Windows that is cp1252, which dies
        # on the first Thai byte. UTF-8 mode is the only fix that reaches inside NeMo.
        raise SystemExit("run with PYTHONUTF8=1 (or python -X utf8): NeMo reads manifests as cp1252")

    import nemo.collections.asr as nemo_asr
    pl.seed_everything(args.seed)
    if args.val:
        tr, va = Path(args.train), Path(args.val)
    else:
        tr, va = split_by_video(Path(args.train), args.val_frac, 2026)
    m = (nemo_asr.models.ASRModel.restore_from(args.base, map_location="cpu") if args.base.endswith(".nemo")
         else nemo_asr.models.ASRModel.from_pretrained(args.base, map_location="cpu"))

    with open_dict(m.cfg):
        for key, path, shuf in (("train_ds", tr, True), ("validation_ds", va, False)):
            c = m.cfg[key] if key in m.cfg else OmegaConf.create({})
            c.manifest_filepath = str(path)
            c.sample_rate = 16000
            c.batch_size = args.bs
            c.shuffle = shuf
            c.num_workers = 0                      # Windows: worker processes re-import NeMo, slowly
            # 2026-10-06: pin_memory=True leaked ~25 MB/step (30.9 GB private after ~1.2k steps, commit
            # limit hit, 10x slowdown): torch's pinned host cache keeps a block per variable batch shape.
            c.pin_memory = False
            c.max_duration = 20.0
            c.min_duration = 1.0
            c.trim_silence = False
            c.use_start_end_token = False
            for k in ("is_tarred", "tarred_audio_filepaths", "use_lhotse"):
                if k in c:
                    c[k] = False if k != "tarred_audio_filepaths" else None
            m.cfg[key] = c
        m.cfg.optim.lr = args.lr
        m.cfg.optim.sched = OmegaConf.create({"name": "CosineAnnealing", "warmup_steps": 200,
                                              "min_lr": args.lr / 10})
        m.cfg.spec_augment.freq_masks = m.cfg.spec_augment.get("freq_masks", 2)
    # the in-training WER log decodes with CUDA graphs per new shape; plain greedy, nothing cached
    with open_dict(m.cfg.decoding):
        m.cfg.decoding.strategy = "greedy_batch"
        m.cfg.decoding.greedy.use_cuda_graph_decoder = False
    m.change_decoding_strategy(m.cfg.decoding)
    m.setup_training_data(m.cfg.train_ds)
    m.setup_validation_data(m.cfg.validation_ds)

    if args.loss == "fast":
        # `_rnnt_fast.py`: vectorised over u, one Python step per frame; selftested against NeMo's
        # reference loss to 2e-6. The pure-PyTorch NeMo loss below is the same objective at ~100x
        # the cost (100 s per 16 x 20 s batch).
        import sys as _s
        _s.path.insert(0, str(Path(__file__).resolve().parent))
        from _rnnt_fast import FastRNNTLoss
        m.loss._loss = FastRNNTLoss(blank=m.joint.num_classes_with_blank - 1)
        m.loss._force_float32 = True
        if getattr(m.joint, "fuse_loss_wer", False):
            m.joint.set_loss(m.loss)
    elif args.loss != "default":
        # The shipped loss is `warprnnt_numba`, and numba 0.67 in this venv fails its typing pass
        # ("Signature mismatch: 2 argument types given, but function takes 1 arguments") on the
        # first training step. The pure-PyTorch RNNT loss computes the same objective, slower.
        from nemo.collections.asr.losses.rnnt import RNNTLoss
        m.loss = RNNTLoss(num_classes=m.joint.num_classes_with_blank - 1, loss_name=args.loss,
                          reduction=m.cfg.get("rnnt_reduction", "mean_batch"))
        if getattr(m.joint, "fuse_loss_wer", False):
            m.joint.set_loss(m.loss)

    if args.freeze_encoder:
        m.encoder.freeze()

    class EpochSave(pl.Callback):
        # 2026-10-06: a memory reap killed a run at step 759 with nothing on disk -- save every epoch
        def on_train_epoch_end(self, trainer, pl_module):
            p = ROOT / "checkpoints" / f"rt_student_{args.tag}_ep{trainer.current_epoch}.nemo"
            p.parent.mkdir(exist_ok=True)
            pl_module.save_to(str(p))
            print(f"-> {p}", flush=True)

    trainer = pl.Trainer(callbacks=[EpochSave()], accelerator="gpu", devices=1, precision="bf16-mixed",
                         max_epochs=args.epochs, max_steps=args.max_steps,
                         log_every_n_steps=25, val_check_interval=0.5, num_sanity_val_steps=0,
                         enable_checkpointing=False, logger=False, gradient_clip_val=1.0,
                         accumulate_grad_batches=args.accum)
    m.set_trainer(trainer)
    m.setup_optimization(m.cfg.optim)
    trainer.fit(m)

    out = ROOT / "checkpoints" / f"rt_student_{args.tag}.nemo"
    out.parent.mkdir(exist_ok=True)
    m.save_to(str(out))
    (ROOT / "results" / f"finetune_rt_{args.tag}_run.json").write_text(json.dumps({
        "base": args.base, "seed": args.seed, "train": str(tr), "val": str(va), "lr": args.lr, "epochs": args.epochs,
        "max_steps": args.max_steps, "bs": args.bs, "accum": args.accum, "vram_frac": args.vram_frac, "loss": args.loss, "freeze_encoder": args.freeze_encoder,
        "checkpoint": str(out), "metrics": {k: float(v) for k, v in trainer.callback_metrics.items()}},
        indent=1), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
