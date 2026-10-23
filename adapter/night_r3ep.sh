#!/usr/bin/env bash
# Loop iteration 10 (2026-10-11): 1 -> 2 epochs was a real -3% (two seeds); does a 3rd epoch add more?
# Same data and recipe as r2ep (gs2lot mix, LR 5e-5, seed 2026), bs 8 x accum 2 under the --vram-frac .9 cap.
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=E:/asr-spike/.venv/Scripts/python.exe; G=E:/Claude/datasets/gigaspeech2_th; C=checkpoints
BASE=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
score() { m=$1; t=$2; $PY adapter/eval_rt_local.py --model $m --tag ${t}_bench 2>&1 | tail -1
  echo "== $t"; $PY adapter/paired_vs.py results/nectec_fc_greedy_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "official_cer|p_arm"
  $PY adapter/paired_vs.py results/r2ep_a75_bench_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "rel_change|p_arm"
  $PY adapter/dev_by_source.py results/${t}_bench_hyps.jsonl 2>&1 | grep bench; }
echo "=== $(date +%H:%M) train r3ep"
[ -f $C/rt_student_r3ep.nemo ] || $PY adapter/finetune_rt.py --base $BASE --train $G/manifest_train_conv_plus_lot.jsonl --val $G/manifest_val_conv.jsonl --tag r3ep --epochs 3 --lr 5e-5 --bs 8 --accum 2 > logs/ft_r3ep.log 2>&1
for a in 60 75 90; do out=$C/r3ep_a$a.nemo
  [ -f $out ] || $PY adapter/avg_nemo.py --base $BASE --tuned $C/rt_student_r3ep.nemo --alpha 0.$a --out $out > /dev/null 2>&1
  score $out r3ep_a$a; done
echo "=== $(date +%H:%M) END r3ep"
