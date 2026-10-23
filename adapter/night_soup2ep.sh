#!/usr/bin/env bash
# Loop iteration 4c (2026-10-11): two-seed soup of the 2-epoch tunes (r2ep + r2eps7), blended with the base at
# alpha .75 / .9. Waits for night_gs2conf.sh to finish so only one GPU job runs at a time.
cd /e/Claude/asr-research
export PYTHONUTF8=1
PY=E:/asr-spike/.venv/Scripts/python.exe
C=checkpoints
BASE=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
until grep -q "END gs2conf" logs/night_gs2conf.log; do sleep 300; done
score() { m=$1; t=$2; $PY adapter/eval_rt_local.py --model $m --tag ${t}_bench 2>&1 | tail -1
  echo "== $t"; $PY adapter/paired_vs.py results/nectec_fc_greedy_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "official_cer|p_arm"
  $PY adapter/paired_vs.py results/r2ep_a75_bench_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "rel_change|p_arm"
  $PY adapter/dev_by_source.py results/${t}_bench_hyps.jsonl 2>&1 | tail -1; }
echo "=== $(date +%H:%M) soup r2ep + r2eps7"
[ -f $C/soup2ep.nemo ] || $PY adapter/avg_nemo.py --base $C/rt_student_r2ep.nemo --tuned $C/rt_student_r2eps7.nemo --alpha 0.5 --out $C/soup2ep.nemo > /dev/null 2>&1
for a in 75 90; do out=$C/soup2ep_a$a.nemo; [ -f $out ] || $PY adapter/avg_nemo.py --base $BASE --tuned $C/soup2ep.nemo --alpha 0.$a --out $out > /dev/null 2>&1; score $out soup2ep_a$a; done
echo "=== $(date +%H:%M) done"
