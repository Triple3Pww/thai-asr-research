#!/usr/bin/env bash
# 2026-10-10, toward CER 0.05 by architecture: the FUSE pilot (`idea-stage/IDEA_REPORT_2026-10-10.md` §4).
# Three arms on the same cached features, same seed/steps, one GPU job at a time:
#   f1 = fusion block, Whisper on; c1 = same block, Whisper zeroed; c0 = no block, decoder/joint only.
# Each arm then transcribes the benchmark from cache; scored on the DEV half only (test is untouched).
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=E:/asr-spike/.venv/Scripts/python.exe
mkdir -p logs
# probes 2026-10-10: decoder/joint updates HURT (c0 0.0759 -> 0.082 on GS2 val) and fusion lr 3e-4 diverged;
# fusion-only at 5e-5 reached 0.0625 in 1,200 steps. So: decoder frozen, lr 5e-5. c0 is then NECTEC itself.
for spec in "f2|--lr 5e-5 --dec-lr 0" "c2|--lr 5e-5 --dec-lr 0 --no-whisper"; do
  tag=${spec%%|*}; extra=${spec#*|}
  echo "=== $(date +%H:%M) train $tag $extra"
  [ -f checkpoints/fuse_$tag.pt ] || $PY adapter/fuse_ears.py train --tag $tag $extra > logs/fuse_$tag.log 2>&1
  tail -2 logs/fuse_$tag.log
  [ -f checkpoints/fuse_$tag.pt ] && $PY adapter/fuse_ears.py eval --tag $tag 2>&1 | tail -1
  [ -f results/fuse_${tag}_bench_hyps.jsonl ] && $PY adapter/paired_vs.py results/nectec_fc_greedy_hyps.jsonl results/fuse_${tag}_bench_hyps.jsonl --half dev 2>&1 | tail -4
done
echo "=== $(date +%H:%M) END"
