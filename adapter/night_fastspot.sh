#!/usr/bin/env bash
# Fast spotter, end to end: must reproduce the slow head run byte-for-byte, then time the one-pass pipeline.
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1
PY=E:/asr-spike/.venv/Scripts/python.exe; NEC=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
while ! grep -q "=== .* done" logs/night_a2head.log 2>/dev/null; do sleep 30; done
echo "=== $(date +%H:%M) real bias@50 with --fast-spot"
$PY adapter/arm_d6_splice.py --mode rnnt --rnnt-model $NEC --spotter head --head checkpoints/ctchead_h3.pt --fast-spot \
  --split real --condition bias@50 --tau -1.5 --pad-start 1 --pad-end 2 --tag ctchead_h3_real_bias50_tau-1.5_fast 2>&1 | grep -E "CER|accepted|Error|Traceback" | tail -3
cmp results/ctchead_h3_real_bias50_tau-1.5_hyps.jsonl results/ctchead_h3_real_bias50_tau-1.5_fast_hyps.jsonl && echo "IDENTICAL to the slow run"
echo "=== $(date +%H:%M) latency with --fast-spot"
$PY adapter/latency_onepass.py --head checkpoints/ctchead_h3.pt --tau -1.5 --fast-spot 2>&1 | tail -1
echo "=== $(date +%H:%M) done"
