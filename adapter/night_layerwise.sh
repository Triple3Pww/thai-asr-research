#!/usr/bin/env bash
# Loop iteration 9 (2026-10-11): r2ep with separate encoder / decoder+joint interpolation alphas.
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=E:/asr-spike/.venv/Scripts/python.exe; C=checkpoints
BASE=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo; T=$C/rt_student_r2ep.nemo
score() { m=$1; t=$2; $PY adapter/eval_rt_local.py --model $m --tag ${t}_bench 2>&1 | tail -1
  echo "== $t"; $PY adapter/paired_vs.py results/nectec_fc_greedy_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "official_cer|p_arm"
  $PY adapter/paired_vs.py results/r2ep_a75_bench_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "rel_change|p_arm"
  $PY adapter/dev_by_source.py results/${t}_bench_hyps.jsonl 2>&1 | grep bench; }
for ed in "90 50" "90 75" "60 100" "75 100" "100 50"; do set -- $ed
  t=r2ep_e$1d$2; out=$C/$t.nemo; echo "=== $(date +%H:%M) $t"
  [ -f $out ] || $PY adapter/avg_nemo.py --base $BASE --tuned $T --alpha $(awk "BEGIN{print $1/100}") --alpha-dec $(awk "BEGIN{print $2/100}") --out $out > /dev/null 2>&1
  score $out $t
done
echo "=== $(date +%H:%M) END layerwise"
