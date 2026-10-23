#!/usr/bin/env bash
# 2026-10-11 loop iteration 3: uniform soups of the sibling GS2 fine-tunes (same NECTEC init), then blended with
# the base at alpha .75 / .6. No training. Dev half only; test untouched. One GPU job at a time.
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=E:/asr-spike/.venv/Scripts/python.exe; BASE=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
C=checkpoints
[ -f $C/soup2_lot_s0c.nemo ] || $PY adapter/avg_nemo.py --base $C/rt_student_gs2lot.nemo --tuned $C/rt_student_gs2s0c.nemo --alpha 0.5 --out $C/soup2_lot_s0c.nemo
[ -f $C/soup3_lot_s0c_a.nemo ] || $PY adapter/avg_nemo.py --base $C/soup2_lot_s0c.nemo --tuned $C/rt_student_gs2a.nemo --alpha 0.3333 --out $C/soup3_lot_s0c_a.nemo
for s in soup2_lot_s0c soup3_lot_s0c_a; do for a in 60 75; do
  out=$C/${s}_a$a.nemo
  [ -f $out ] || $PY adapter/avg_nemo.py --base $BASE --tuned $C/$s.nemo --alpha 0.$a --out $out
  $PY adapter/eval_rt_local.py --model $out --tag ${s}_a${a}_bench 2>&1 | tail -1
  echo "== ${s}_a$a"; $PY adapter/paired_vs.py results/nectec_fc_greedy_hyps.jsonl results/${s}_a${a}_bench_hyps.jsonl --half dev 2>&1 | grep -E "official_cer|p_arm"
  $PY adapter/paired_vs.py results/gs2lot_a75_bench_hyps.jsonl results/${s}_a${a}_bench_hyps.jsonl --half dev 2>&1 | grep -E "rel_change|p_arm"
done; done
echo "=== END"
