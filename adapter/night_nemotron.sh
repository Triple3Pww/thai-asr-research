#!/usr/bin/env bash
# Loop iteration 13 (2026-10-11): typhoon-asr-streaming-nemotron-0.6b (641M FastConformer-RNNT, 5.6x NECTEC,
# released 2026-09-15, never scored here) on the benchmark, offline full-file decode, two right contexts.
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=E:/asr-spike/.venv/Scripts/python.exe; M=E:/asr-spike/models/typhoon-nemotron/typhoon-asr-streaming-nemotron-0.6b.nemo
score() { t=$1; shift; $PY adapter/eval_rt_local.py --model $M --tag ${t}_bench --prompt th-TH "$@" 2>&1 | tail -1
  echo "== $t"; $PY adapter/paired_vs.py results/nectec_fc_greedy_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "official_cer|p_arm"
  $PY adapter/paired_vs.py results/r2ep_a75_bench_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "rel_change|p_arm"
  $PY adapter/dev_by_source.py results/${t}_bench_hyps.jsonl 2>&1 | grep bench; }
echo "=== $(date +%H:%M) nemotron ctx 56,13"; score nemo06_c13 --att-context 56,13
echo "=== $(date +%H:%M) nemotron ctx 56,6"; score nemo06_c6 --att-context 56,6
echo "=== $(date +%H:%M) END nemotron"
