#!/usr/bin/env bash
# 2026-10-11 loop iteration 4: is gs2lot under-trained or mis-tuned? Same data (F45 gs2lot mix), vary the recipe:
# (bs 8 x accum 2: batch 16 overflowed VRAM into 19 GB of system RAM on 2026-10-11, F46)
#   r2ep = 2 epochs; rlr1 = LR 1e-4; rseed = seed 7 (then a uniform soup with gs2lot = a seed soup).
# Each tuned model alpha .5/.75 vs the base; dev half only, test untouched. One GPU job at a time.
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=E:/asr-spike/.venv/Scripts/python.exe; G=E:/Claude/datasets/gigaspeech2_th
BASE=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo; C=checkpoints
TR=$G/manifest_train_conv_plus_lot.jsonl
train() { tag=$1; shift; [ -f $C/rt_student_$tag.nemo ] || $PY adapter/finetune_rt.py --base $BASE --train $TR --val $G/manifest_val_conv.jsonl --tag $tag "$@" > logs/ft_$tag.log 2>&1; echo "=== $(date +%H:%M) trained $tag"; tail -1 logs/ft_$tag.log; }
score() { m=$1; t=$2; $PY adapter/eval_rt_local.py --model $m --tag ${t}_bench 2>&1 | tail -1
  echo "== $t"; $PY adapter/paired_vs.py results/nectec_fc_greedy_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "official_cer|p_arm"
  $PY adapter/paired_vs.py results/gs2lot_a75_bench_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "rel_change|p_arm"; }
blend() { tag=$1; for a in 50 75; do out=$C/${tag}_a$a.nemo; [ -f $out ] || $PY adapter/avg_nemo.py --base $BASE --tuned $2 --alpha 0.$a --out $out > /dev/null 2>&1; score $out ${tag}_a$a; done; }
train r2ep --epochs 2 --lr 5e-5 --bs 8 --accum 2; blend r2ep $C/rt_student_r2ep.nemo
train rlr1 --epochs 1 --lr 1e-4 --bs 8 --accum 2; blend rlr1 $C/rt_student_rlr1.nemo
train rseed --epochs 1 --lr 5e-5 --seed 7 --bs 8 --accum 2; blend rseed $C/rt_student_rseed.nemo
[ -f $C/seedsoup.nemo ] || $PY adapter/avg_nemo.py --base $C/rt_student_gs2lot.nemo --tuned $C/rt_student_rseed.nemo --alpha 0.5 --out $C/seedsoup.nemo > /dev/null 2>&1
blend seedsoup $C/seedsoup.nemo
echo "=== $(date +%H:%M) END"
