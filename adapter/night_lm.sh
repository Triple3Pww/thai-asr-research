#!/usr/bin/env bash
# Loop iteration 7 (2026-10-11): beam search + shallow fusion of a token 4-gram over 2 M GigaSpeech2-train
# transcripts (adapter/build_lm_arpa.py) on the best model r2ep_a75. Control first: the same beam, no LM.
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=E:/asr-spike/.venv/Scripts/python.exe; M=checkpoints/r2ep_a75.nemo; LM=lm/nec_gs2_2m_4g.arpa
score() { t=$1; shift; $PY adapter/eval_rt_local.py --model $M --tag ${t}_bench "$@" 2>&1 | tail -1
  echo "== $t"; $PY adapter/paired_vs.py results/nectec_fc_greedy_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "official_cer|p_arm"
  $PY adapter/paired_vs.py results/r2ep_a75_bench_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "rel_change|p_arm"
  $PY adapter/dev_by_source.py results/${t}_bench_hyps.jsonl 2>&1 | grep bench; }
echo "=== $(date +%H:%M) build LM"
[ -f $LM ] || $PY adapter/build_lm_arpa.py --n-utts 2000000 --order 4 --out $LM 2>&1 | grep -E "^(text|tokens|order|wrote|\{)"
[ -f results/r2ep_beam4_bench_hyps.jsonl ] || { echo "=== $(date +%H:%M) beam 4, no LM (control)"
score r2ep_beam4 --strategy malsd_batch --beam 4; }
for a in 0.2 0.4 0.6; do echo "=== $(date +%H:%M) beam 4 + LM alpha $a"; score r2ep_b4lm${a#0.} --strategy malsd_batch --beam 4 --lm $LM --lm-alpha $a; done
echo "=== $(date +%H:%M) END lm"
