"""fetch_thaiser_shards.py -- pull the rest of THAI-SER so their own protocol can be run.

WHY. `F18`/`F19`/`F22` are all measured on 3 of 29 shards: 1,256 filtered clips, 22 actors,
**12 studio sessions**. The corpus authors evaluate with 8-fold cross-validation over studio
sessions (`tann9949/thaiser-experiments`, `experiment/data/thaiser.py`), 5 sessions held out per
fold. Checked against our 12 sessions (`studio001-003, 034-038, 071-074`): **seven of their eight
folds contain zero test clips here.** Their protocol is simply not runnable on this subset.

\u2b50 Studio-disjoint IS speaker-disjoint in this corpus -- verified, every one of our 22 actors
appears in exactly one session -- so their folds and our actor-holdout are equivalent in kind.
What the full corpus buys is the ability to run THEIR folds, on THEIR mic, at THEIR agreement
threshold, and get a number that is comparable rather than merely similar.

It also fixes the thinnest thing under `F18`: 426 training clips from SEVEN actors, with every
rung balanced on dev-side early stopping.

COST. 29 shards, 12.7 GB total; 3 are already here. Licence **CC-BY-SA-4.0, ungated** -- no token
needed, and unlike `thai-contextasr-bench` (CC-BY-NC) this one permits more than evaluation.
\u26a0 Share-alike on model weights is legally murky; that is a question for a human, not this script.

\U0001f534 DISK IS THE REAL CONSTRAINT. D: had 24.8 GB free. Parquet 12.7 GB + a two-mic decoded cache
(~7 GB) leaves ~6 GB. `--verify-only` checks what is present without downloading; `--prune` drops
a shard's parquet once it has been cached, for the tighter path.

Usage:
    $py = "D:/sensevoice-gpu/.venv/Scripts/python.exe"
    & $py adapter/fetch_thaiser_shards.py --verify-only
    & $py adapter/fetch_thaiser_shards.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = "airesearch/thai-ser"
DEST = Path("D:/thai-ser")
N_SHARDS = 29
# The three already here were pulled under a flat name; keep that convention so the existing
# cache, and every number measured from it, still resolve.
LOCAL = "train-{:05d}.parquet"
REMOTE = "data/train-{:05d}-of-00029.parquet"


def present() -> dict[int, int]:
    out = {}
    for i in range(N_SHARDS):
        p = DEST / LOCAL.format(i)
        if p.exists() and p.stat().st_size > 0:
            out[i] = p.stat().st_size
    return out


def free_gb(drive: str = "D:") -> float:
    return shutil.disk_usage(drive + "\\").free / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=4.0,
                    help="abort before a shard that would take free space under this")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new shards")
    a = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    have = present()
    missing = [i for i in range(N_SHARDS) if i not in have]
    print(f"[state] {len(have)}/{N_SHARDS} shards present "
          f"({sum(have.values())/1e9:.2f} GB) | missing {len(missing)}")
    print(f"[state] D: free {free_gb():.1f} GB")
    if have:
        print(f"[state] have: {sorted(have)}")
    if a.verify_only or not missing:
        return 0

    from huggingface_hub import hf_hub_download

    t0, done = time.time(), 0
    for i in missing:
        if free_gb() < a.min_free_gb:
            print(f"\U0001f534 stopping: D: free {free_gb():.1f} GB is under --min-free-gb "
                  f"{a.min_free_gb}. {len(missing)-done} shards not fetched.", flush=True)
            break
        dst = DEST / LOCAL.format(i)
        try:
            src = hf_hub_download(repo_id=REPO, filename=REMOTE.format(i),
                                  repo_type="dataset")
        except Exception as e:
            print(f"  !! shard {i} failed: {type(e).__name__}: {e}", flush=True)
            continue
        # Copy out of the HF cache rather than symlink: this box has symlinks disabled (the
        # hub warns about it), and the cache would otherwise hold a second full copy anyway.
        shutil.copyfile(src, dst)
        try:
            os.remove(src)                      # reclaim the cache copy immediately
        except OSError:
            pass
        done += 1
        print(f"  [{done}/{len(missing)}] shard {i:02d} -> {dst.name} "
              f"({dst.stat().st_size/1e6:.0f} MB) | D: free {free_gb():.1f} GB | "
              f"{time.time()-t0:.0f}s", flush=True)
        if a.limit and done >= a.limit:
            break

    have = present()
    print(f"\n[done] {len(have)}/{N_SHARDS} shards, {sum(have.values())/1e9:.2f} GB, "
          f"D: free {free_gb():.1f} GB")
    print("[next] rebuild the cache over the new shards:")
    print("       adapter/finetune_emotion2vec.py --cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
