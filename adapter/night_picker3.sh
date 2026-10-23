#!/usr/bin/env bash
# Overnight 2026-10-05: 3-source picker (turbo + typhoon-asr-realtime + NECTEC) trained on the
# SYNTHETIC corpus -- the one labelled audio none of the three models has heard (his choice:
# "try synthetic"). Strictly sequential: one model on the GPU at a time (the box has been reaped
# for RAM three times this week). Each step is resumable or cheap to redo. Log: results/night_picker3.log
set -u
cd /e/Claude/asr-research
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=E:/asr-spike/.venv/Scripts/python.exe
FW=/e/fw-spike/.venv/Scripts/python.exe
D=E:/Claude/datasets/synth
NEC=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
step() { echo "=== $(date '+%H:%M:%S') $*"; }

for part in train heldout; do
  M=$D/manifest_$part.jsonl
  step "turbo top-5 on synth $part"
  $FW -u adapter/nbest_fw.py --model E:/fw-spike/models/typhoon-whisper-turbo-ct2 --n 5 --tag synth_${part}_turbo5 --manifest $M 2>&1 | grep -E "^\{|Error|Traceback"
  step "realtime top-5 on synth $part"
  $PY -u adapter/nbest_rt.py --n 5 --tag synth_${part}_rt5 --manifest $M 2>&1 | grep -E "^\{|Error|Traceback"
  step "nectec top-5 on synth $part"
  $PY -u adapter/nbest_rt.py --model $NEC --n 5 --tag synth_${part}_nec5 --manifest $M 2>&1 | grep -E "^\{|Error|Traceback"
  step "picker examples synth $part"
  $PY -u adapter/picker_data.py --turbo results/synth_${part}_turbo5_nbest.jsonl --rt results/synth_${part}_rt5_nbest.jsonl \
      --nectec results/synth_${part}_nec5_nbest.jsonl --manifest $M --out $D/picker3_$part.jsonl 2>&1 | grep -E "^\{|Error|Traceback"
done

step "benchmark 3-source prompts (no labels)"
$PY -u adapter/picker_data.py --turbo results/nbest_turbo5_nbest.jsonl --rt results/nbest_rt5_nbest.jsonl \
    --nectec results/nbest_nectec5_nbest.jsonl --out results/picker3_bench_prompts.jsonl 2>&1 | grep -E "^\{|Error|Traceback"

step "train picker p3 (Qwen3-0.6B) on synthetic"
$PY -u adapter/train_picker.py --train $D/picker3_train.jsonl --heldout $D/picker3_heldout.jsonl \
    --heldout-manifest $D/manifest_heldout.jsonl --tag p3 --epochs 3 > results/train_picker_p3.log 2>&1
grep -E "before|epoch|^\{|Traceback" results/train_picker_p3.log | tail -6

step "predict benchmark with p3"
$PY -u adapter/train_picker.py --predict results/picker3_bench_prompts.jsonl --ckpt checkpoints/picker_p3 --tag p3 2>&1 | grep -E "^\{|Traceback"

step "score"
$PY adapter/eval_canon.py results/picker_p3_bench_hyps.jsonl results/picker_p1_bench_hyps.jsonl results/nbest_nectec5_top1_hyps.jsonl results/nbest_turbo5_top1_hyps.jsonl 2>&1 | grep -E "system|picker|nbest|^tw "
step "done"
