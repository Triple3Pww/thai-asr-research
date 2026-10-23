#!/usr/bin/env bash
# Loop iteration 15 (2026-10-11): the 2-epoch recipe is the only lever that worked; is LR 5e-5 its best?
# r2ep recipe at LR 1e-4 and 3e-5 (gs2lot mix, 2 epochs, bs 8 x 2, --vram-frac .85), alpha .75.
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=E:/asr-spike/.venv/Scripts/python.exe; G=E:/Claude/datasets/gigaspeech2_th; C=checkpoints
BASE=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
score() { m=$1; t=$2; $PY adapter/eval_rt_local.py --model $m --tag ${t}_bench 2>&1 | tail -1
  echo "== $t"; $PY adapter/paired_vs.py results/nectec_fc_greedy_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "official_cer|p_arm"
  $PY adapter/paired_vs.py results/r2ep_a75_bench_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "rel_change|p_arm"
  $PY adapter/dev_by_source.py results/${t}_bench_hyps.jsonl 2>&1 | grep bench; }
for v in "lr3 3e-5"; do set -- $v; t=r2$1
  echo "=== $(date +%H:%M) train $t (LR $2)"
  [ -f $C/rt_student_$t.nemo ] || $PY adapter/finetune_rt.py --base $BASE --train $G/manifest_train_conv_plus_lot.jsonl --val $G/manifest_val_conv.jsonl --tag $t --epochs 2 --lr $2 --bs 8 --accum 2 --vram-frac 0.8 > logs/ft_$t.log 2>&1
  out=$C/${t}_a75.nemo
  [ -f $out ] || $PY adapter/avg_nemo.py --base $BASE --tuned $C/rt_student_$t.nemo --alpha 0.75 --out $out > /dev/null 2>&1
  score $out ${t}_a75
done
echo "=== $(date +%H:%M) END lr3only"
