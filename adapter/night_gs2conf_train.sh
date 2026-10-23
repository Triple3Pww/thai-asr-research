#!/usr/bin/env bash
# 2026-10-11: tail of night_gs2conf.sh (training + scoring only), relaunched at bs 4 x accum 4 after the bs-8 run spilled VRAM into system RAM.
# 2026-10-11 loop iteration 6 (rewritten 02:30 after r2ep = 2 epochs became the best): first REPLICATE r2ep (seed 7) and
# extend its alpha, then scale the Pathumma-L-confirmed GS2-train set (iteration 2 rule) over shards 1-4,
# then gs2lot mix + confirmed s0..s4 -> train (bs 8 x accum 2), alpha sweep, dev + a/c breakdown. Test untouched.
# Waits for the recipe sweep; one GPU job at a time.
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=E:/asr-spike/.venv/Scripts/python.exe; FW=E:/fw-spike/.venv/Scripts/python.exe
G=E:/Claude/datasets/gigaspeech2_th; D=E:/Claude/datasets/gs2_train; C=checkpoints
BASE=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
PATL=E:/fw-spike/models/nectec_Pathumma-whisper-th-large-v3-ct2
score() { m=$1; t=$2; $PY adapter/eval_rt_local.py --model $m --tag ${t}_bench 2>&1 | tail -1
  echo "== $t"; $PY adapter/paired_vs.py results/nectec_fc_greedy_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "official_cer|p_arm"
  $PY adapter/paired_vs.py results/gs2lot_a75_bench_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "rel_change|p_arm"
  $PY adapter/paired_vs.py results/r2ep_a75_bench_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "rel_change|p_arm"
  $PY adapter/dev_by_source.py results/${t}_bench_hyps.jsonl 2>&1 | grep bench; }
echo "=== $(date +%H:%M) RESTART train gs2conf at bs 4 x accum 4 (bs 8 overflowed VRAM into RAM, killed 09:42)"
[ -f $C/rt_student_gs2conf.nemo ] || $PY adapter/finetune_rt.py --base $BASE --train $D/manifest_gs2lot_conf04.jsonl --val $G/manifest_val_conv.jsonl --tag gs2conf --epochs 2 --lr 5e-5 --bs 4 --accum 4 > logs/ft_gs2conf.log 2>&1
for a in 50 75 100; do
  m=$C/gs2conf_a$a.nemo; [ $a = 100 ] && m=$C/rt_student_gs2conf.nemo
  [ -f $m ] || $PY adapter/avg_nemo.py --base $BASE --tuned $C/rt_student_gs2conf.nemo --alpha 0.$a --out $m > /dev/null 2>&1
  score $m gs2conf_a$a
done
echo "=== $(date +%H:%M) END gs2conf"
