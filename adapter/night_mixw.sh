#!/usr/bin/env bash
# Loop iteration 14 (2026-10-11): every set ADDED to the gs2lot mix cost the conversational a half (iterations 2,5,6,11).
# Re-weight the mix itself instead: GS2 x2, then conv-LOTUSDIS x2, r2ep recipe (2 epochs, bs 8 x 2, --vram-frac .8: at .9 commit fell to 0.7 GB).
# Confound, stated: duplicating one half also adds optimizer steps on it (r3ep showed a 3rd epoch over-fits).
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=E:/asr-spike/.venv/Scripts/python.exe; G=E:/Claude/datasets/gigaspeech2_th; C=checkpoints
BASE=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
score() { m=$1; t=$2; $PY adapter/eval_rt_local.py --model $m --tag ${t}_bench 2>&1 | tail -1
  echo "== $t"; $PY adapter/paired_vs.py results/nectec_fc_greedy_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "official_cer|p_arm"
  $PY adapter/paired_vs.py results/r2ep_a75_bench_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "rel_change|p_arm"
  $PY adapter/dev_by_source.py results/${t}_bench_hyps.jsonl 2>&1 | grep bench; }
M=$G/manifest_train_conv_plus_lot.jsonl
grep -vi lotus $M > $G/mix_gs2only.tmp; grep -i lotus $M > $G/mix_lotonly.tmp
cat $M $G/mix_gs2only.tmp > $G/manifest_mix_gs2x2.jsonl
cat $M $G/mix_lotonly.tmp > $G/manifest_mix_lotx2.jsonl
for v in gs2x2 lotx2; do
  echo "=== $(date +%H:%M) train r2$v: $(wc -l < $G/manifest_mix_$v.jsonl) clips"
  [ -f $C/rt_student_r2$v.nemo ] || $PY adapter/finetune_rt.py --base $BASE --train $G/manifest_mix_$v.jsonl --val $G/manifest_val_conv.jsonl --tag r2$v --epochs 2 --lr 5e-5 --bs 8 --accum 2 --vram-frac 0.8 > logs/ft_r2$v.log 2>&1
  out=$C/r2${v}_a75.nemo
  [ -f $out ] || $PY adapter/avg_nemo.py --base $BASE --tuned $C/rt_student_r2$v.nemo --alpha 0.75 --out $out > /dev/null 2>&1
  score $out r2${v}_a75
done
echo "=== $(date +%H:%M) END mixw"
