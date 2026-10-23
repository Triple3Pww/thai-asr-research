#!/usr/bin/env bash
# 2026-10-05 early morning: does REAL read speech in the picker's training fix its under-trust of
# NECTEC on read speech? p3 (synthetic only) is best of all on conversation but loses to NECTEC on
# read sentences. Porjai is real read speech with human labels -- but NECTEC memorised it (top-1 CER
# 0.0154), which may teach "always trust NECTEC". This tests that empirically instead of assuming:
# p4 = synthetic + Porjai, same model and recipe, scored on the benchmark next to p3.
# Learning curve that motivated it: synthetic half -> full = 0.0751 -> 0.0727 (only -3.2%).
set -u
cd /e/Claude/asr-research
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=E:/asr-spike/.venv/Scripts/python.exe
D=E:/Claude/datasets
NEC=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
step() { echo "=== $(date '+%H:%M:%S') $*"; }

step "nectec top-5 on Porjai train (16k)"
$PY -u adapter/nbest_rt.py --model $NEC --n 5 --tag porjai_tr_nec5 --manifest $D/porjai/manifest_train.jsonl 2>&1 | grep -E "^\{|Error|Traceback"

step "3-source picker examples on Porjai train"
$PY -u adapter/picker_data.py --turbo results/porjai_tr_turbo5_nbest.jsonl --rt results/porjai_tr_rt5_nbest.jsonl \
    --nectec results/porjai_tr_nec5_nbest.jsonl --manifest $D/porjai/manifest_train.jsonl \
    --out $D/porjai/picker3_train.jsonl 2>&1 | grep -E "^\{|Error|Traceback"

step "mix synthetic + Porjai"
cat $D/synth/picker3_train.jsonl $D/porjai/picker3_train.jsonl > $D/picker3_mix_train.jsonl
wc -l $D/picker3_mix_train.jsonl

step "train picker p4 (synthetic + Porjai), held-out = synthetic held-out"
$PY -u adapter/train_picker.py --train $D/picker3_mix_train.jsonl --heldout $D/synth/picker3_heldout.jsonl \
    --heldout-manifest $D/synth/manifest_heldout.jsonl --tag p4 --epochs 2 > results/train_picker_p4.log 2>&1
grep -E "epoch|^\{|Traceback" results/train_picker_p4.log | tail -4

step "predict benchmark with p4"
$PY -u adapter/train_picker.py --predict results/picker3_bench_prompts.jsonl --ckpt checkpoints/picker_p4 --tag p4 2>&1 | grep -E "^\{|Traceback"

step "score"
$PY adapter/eval_canon.py results/picker_p4_bench_hyps.jsonl results/picker_p3_bench_hyps.jsonl results/nbest_nectec5_top1_hyps.jsonl 2>&1 | grep -E "system|picker|nbest|^tw "
step "done"
