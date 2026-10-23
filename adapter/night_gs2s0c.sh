#!/usr/bin/env bash
# 2026-10-10 loop iteration 2: gs2lot mix + 4.2 h of shard-0 hard clips where Pathumma-L CONFIRMS the label (CER<=.02, beats NECTEC).
# 2026-10-09, toward CER 0.060: NECTEC + 1 epoch of convention-matched GigaSpeech2-th (fresh to NECTEC, `F44`),
# the lot2 (A2) recipe. Interpolate with the base, score each alpha on GS2 held-out videos AND transcribe the
# benchmark so `paired_vs.py --half dev` can compare -- the test half is not touched here. One GPU job at a time.
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=E:/asr-spike/.venv/Scripts/python.exe; G=E:/Claude/datasets/gigaspeech2_th
BASE=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
mkdir -p logs
echo "=== $(date +%H:%M) train gs2s0c"
[ -f checkpoints/rt_student_gs2s0c.nemo ] || $PY adapter/finetune_rt.py --base $BASE --train E:/Claude/datasets/gs2_train/manifest_gs2lot_s0c.jsonl \
   --val $G/manifest_val_conv.jsonl --tag gs2s0c --epochs 1 --lr 5e-5 --bs 8 --accum 2 > logs/ft_gs2s0c.log 2>&1
tail -3 logs/ft_gs2s0c.log
SPECS="base=$BASE"
for a in 25 50 75; do
  out=checkpoints/gs2s0c_a$a.nemo
  [ -f $out ] || $PY adapter/avg_nemo.py --base $BASE --tuned checkpoints/rt_student_gs2s0c.nemo --alpha 0.$a --out $out
  SPECS="$SPECS a$a=$out"
done
SPECS="$SPECS a100=checkpoints/rt_student_gs2s0c.nemo"
echo "=== $(date +%H:%M) GS2 held-out videos"
$PY adapter/eval_manifest.py --manifest $G/manifest_val_conv.jsonl --models $SPECS --tag gs2s0c_gs2val 2>&1 | grep -E "^(base|a[0-9]+) "
for a in 25 50 75 100; do
  m=checkpoints/gs2s0c_a$a.nemo; [ $a = 100 ] && m=checkpoints/rt_student_gs2s0c.nemo
  echo "=== $(date +%H:%M) benchmark a$a"
  $PY adapter/eval_rt_local.py --model $m --tag gs2s0c_a${a}_bench 2>&1 | tail -1
done
for a in 25 50 75 100; do echo "dev a$a vs NECTEC:"; $PY adapter/paired_vs.py results/nectec_fc_greedy_hyps.jsonl results/gs2s0c_a${a}_bench_hyps.jsonl --half dev 2>&1 | grep -E "official_cer|p_arm"; echo " vs gs2lot_a75:"; $PY adapter/paired_vs.py results/gs2lot_a75_bench_hyps.jsonl results/gs2s0c_a${a}_bench_hyps.jsonl --half dev 2>&1 | grep -E "rel_change|p_arm"; done
echo "=== $(date +%H:%M) END"
