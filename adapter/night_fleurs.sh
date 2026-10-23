#!/usr/bin/env bash
# 2026-10-11 loop iteration 5: gs2lot mix + FLEURS-th TRAIN (benchmark convention, Latin names kept, leak-guarded).
# Then the recipe sweep (iteration 4). Dev half only; test untouched. One GPU job at a time.
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=E:/asr-spike/.venv/Scripts/python.exe; G=E:/Claude/datasets/gigaspeech2_th; F=E:/Claude/datasets/fleurs_th
BASE=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo; C=checkpoints
while ps -ef | grep -v grep | grep -q night_soup3; do sleep 60; done
echo "=== $(date +%H:%M) fleurs prep + freshness"
[ -f results/fleurs_train_fresh.json ] || $PY adapter/fleurs_prep.py 2>&1 | grep -E "^(kept|nectec|gs2lot)"
cat $G/manifest_train_conv_plus_lot.jsonl $F/manifest_train_conv.jsonl > $F/manifest_gs2lot_fleurs.jsonl
[ -f $C/rt_student_gs2lotfl.nemo ] || $PY adapter/finetune_rt.py --base $BASE --train $F/manifest_gs2lot_fleurs.jsonl --val $G/manifest_val_conv.jsonl --tag gs2lotfl --epochs 1 --lr 5e-5 --bs 8 --accum 2 > logs/ft_gs2lotfl.log 2>&1
echo "=== $(date +%H:%M) trained gs2lotfl"; tail -c 300 logs/ft_gs2lotfl.log | tail -1
for a in 50 75 100; do
  m=$C/gs2lotfl_a$a.nemo; [ $a = 100 ] && m=$C/rt_student_gs2lotfl.nemo
  [ -f $m ] || $PY adapter/avg_nemo.py --base $BASE --tuned $C/rt_student_gs2lotfl.nemo --alpha 0.$a --out $m > /dev/null 2>&1
  $PY adapter/eval_rt_local.py --model $m --tag gs2lotfl_a${a}_bench 2>&1 | tail -1
  echo "== gs2lotfl_a$a"; $PY adapter/paired_vs.py results/nectec_fc_greedy_hyps.jsonl results/gs2lotfl_a${a}_bench_hyps.jsonl --half dev 2>&1 | grep -E "official_cer|p_arm"
  $PY adapter/paired_vs.py results/gs2lot_a75_bench_hyps.jsonl results/gs2lotfl_a${a}_bench_hyps.jsonl --half dev 2>&1 | grep -E "rel_change|p_arm"
  $PY adapter/dev_by_source.py results/gs2lotfl_a${a}_bench_hyps.jsonl 2>&1 | grep bench
done
echo "=== $(date +%H:%M) END fleurs"
bash adapter/night_recipe.sh
