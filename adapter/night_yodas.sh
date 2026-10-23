#!/usr/bin/env bash
# Loop iteration 11 (2026-10-11): YODAS2-th uploader captions (human, Latin names kept), two-ear confirmed,
# added to the gs2lot mix with the r2ep recipe (2 epochs, bs 8 x 2, --vram-frac .9).
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=E:/asr-spike/.venv/Scripts/python.exe; FW=E:/fw-spike/.venv/Scripts/python.exe
G=E:/Claude/datasets/gigaspeech2_th; Y=E:/Claude/datasets/yodas2_th; C=checkpoints
BASE=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
PATL=E:/fw-spike/models/nectec_Pathumma-whisper-th-large-v3-ct2
score() { m=$1; t=$2; $PY adapter/eval_rt_local.py --model $m --tag ${t}_bench 2>&1 | tail -1
  echo "== $t"; $PY adapter/paired_vs.py results/nectec_fc_greedy_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "official_cer|p_arm"
  $PY adapter/paired_vs.py results/r2ep_a75_bench_hyps.jsonl results/${t}_bench_hyps.jsonl --half dev 2>&1 | grep -E "rel_change|p_arm"
  $PY adapter/dev_by_source.py results/${t}_bench_hyps.jsonl 2>&1 | grep bench; }
echo "=== $(date +%H:%M) prep + NECTEC pass"
$PY adapter/yodas_prep.py --stage prep 2>&1 | grep -E "^(kept|\{|  NECTEC)"
echo "=== $(date +%H:%M) Pathumma-L on the hard clips"
$FW adapter/fw_batch.py --model $PATL --manifest $Y/manifest_yodas_hard.jsonl --out $Y/patL_yodas_hard.jsonl --bs 16 2>&1 | tail -1
$PY adapter/yodas_prep.py --stage confirm
cat $G/manifest_train_conv_plus_lot.jsonl $Y/manifest_yodas_conf.jsonl > $Y/manifest_gs2lot_yodas.jsonl
echo "=== $(date +%H:%M) train yod2ep: $(wc -l < $Y/manifest_gs2lot_yodas.jsonl) clips"
[ -f $C/rt_student_yod2ep.nemo ] || $PY adapter/finetune_rt.py --base $BASE --train $Y/manifest_gs2lot_yodas.jsonl --val $G/manifest_val_conv.jsonl --tag yod2ep --epochs 2 --lr 5e-5 --bs 8 --accum 2 > logs/ft_yod2ep.log 2>&1
for a in 60 75 90; do out=$C/yod2ep_a$a.nemo
  [ -f $out ] || $PY adapter/avg_nemo.py --base $BASE --tuned $C/rt_student_yod2ep.nemo --alpha 0.$a --out $out > /dev/null 2>&1
  score $out yod2ep_a$a; done
echo "=== $(date +%H:%M) END yodas"
