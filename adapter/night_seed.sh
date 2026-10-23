#!/usr/bin/env bash
# lot2 recipe, second seed: 1 epoch conv-LOTUSDIS, alpha 0.75 (the val choice), scored on the benchmark and TVSpeech.
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1
PY=E:/asr-spike/.venv/Scripts/python.exe; L=E:/Claude/datasets/lotusdis_lav
BASE=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
echo "=== $(date +%H:%M) train lot2s2"
$PY adapter/finetune_rt.py --base $BASE --train $L/manifest_train_conv.jsonl --val $L/manifest_val_conv.jsonl \
   --tag lot2s2 --epochs 1 --lr 5e-5 --seed 7 > logs/ft_lot2s2.log 2>&1
$PY adapter/avg_nemo.py --base $BASE --tuned checkpoints/rt_student_lot2s2.nemo --alpha 0.75 --out checkpoints/lot2s2_a75.nemo
$PY adapter/eval_manifest.py --manifest $L/manifest_val_conv.jsonl --models lot2s2_a75=checkpoints/lot2s2_a75.nemo --tag lot2s2_lotval
$PY adapter/eval_rt_local.py --model checkpoints/lot2s2_a75.nemo --tag lot2s2_a75_bench
$PY adapter/eval_manifest.py --manifest E:/asr-bench/typhoon-eval/manifest_tvspeech.jsonl --models lot2s2_a75=checkpoints/lot2s2_a75.nemo --tag deepseek_tvspeech
echo "=== $(date +%H:%M) done"
