#!/usr/bin/env bash
# Evaluate a NECTEC fine-tune and its interpolations with the base -- ONE job at a time.
#   bash adapter/eval_chain_lot.sh lot1 checkpoints/rt_student_lot1.nemo
# 2026-10-06: two GPU jobs side by side filled RAM and both were reaped; everything here is serial.
# alpha is chosen on LOTUSDIS VAL (speaker-held-out), never on a test set (reviewer, Gemini thread e6800f81).
set -u
cd "$(dirname "$0")/.."
TAG=$1; TUNED=$2
PY=E:/asr-spike/.venv/Scripts/python.exe
BASE=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
LOT=E:/Claude/datasets/lotusdis_lav
SUF=${SUF:-}   # "_conv" -> score val/test against the convention-matched labels (lotusdis_convention.py)
export PYTHONUTF8=1
SPECS="base=$BASE"
for a in 25 50 75; do
  out=checkpoints/${TAG}_a$a.nemo
  [ -f $out ] || $PY adapter/avg_nemo.py --base $BASE --tuned $TUNED --alpha 0.$a --out $out
  SPECS="$SPECS a$a=$out"
done
SPECS="$SPECS a100=$TUNED"
$PY adapter/eval_manifest.py --manifest $LOT/manifest_val$SUF.jsonl  --models $SPECS --tag ${TAG}_lotval
$PY adapter/eval_manifest.py --manifest $LOT/manifest_test$SUF.jsonl --models $SPECS --tag ${TAG}_lottest
for s in $SPECS; do n=${s%%=*}; p=${s#*=}; [ $n = base ] && continue
  $PY adapter/eval_rt_local.py --model $p --tag ${TAG}_${n}_bench
done
$PY adapter/eval_canon.py results/nectec_fc_greedy_hyps.jsonl results/${TAG}_a*_bench_hyps.jsonl > results/${TAG}_bench_canon.txt 2>&1
cat results/${TAG}_bench_canon.txt
