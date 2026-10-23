#!/usr/bin/env bash
# One-pass name splice: CTC head on NECTEC's frozen encoder. tau chosen on the SYNTH split only (the head's
# score scale is not Typhoon's), pads fixed at 1:2 (F34); then real dis@50 + bias@50 once; then latency.
#   bash adapter/night_head.sh checkpoints/ctchead_h2.pt
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1
PY=E:/asr-spike/.venv/Scripts/python.exe
NEC=${RNNT:-E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo}   # RNNT=<.nemo> for another transcriber
HEAD=$1; H=$(basename $HEAD .pt)
for t in -0.5 -1 -1.5 -2 -3 -4; do
  echo "=== $(date +%H:%M) synth bias@50 tau $t"
  $PY adapter/arm_d6_splice.py --mode rnnt --rnnt-model $NEC --spotter head --head $HEAD --split synth \
     --condition bias@50 --tau $t --pad-start 1 --pad-end 2 --tag ${H}_synth_bias50_tau$t 2>&1 | grep -E "CER|spots accepted|Error|Traceback" | tail -3
done
tau=$($PY - "$H" <<'PYX'
import json,glob,re,sys
best=None
for f in glob.glob(f"results/{sys.argv[1]}_synth_bias50_tau*_metrics.json"):
    m=json.load(open(f,encoding="utf-8"))["bias@50"]; t=re.search(r"_tau(-?[\d.]+)_metrics",f).group(1)
    if best is None or m["cer"]<best[0]: best=(m["cer"],t)
print(best[1])
PYX
)
echo "=== chosen on synth: tau $tau"
for c in dis@50 bias@50; do
  echo "=== $(date +%H:%M) real $c tau $tau"
  $PY adapter/arm_d6_splice.py --mode rnnt --rnnt-model $NEC --spotter head --head $HEAD --split real \
     --condition $c --tau $tau --pad-start 1 --pad-end 2 --tag ${H}_real_${c/@/}_tau$tau 2>&1 | grep -E "CER|spots accepted|Error|Traceback" | tail -3
done
echo "=== $(date +%H:%M) latency"
$PY adapter/latency_onepass.py --head $HEAD --tau $tau 2>&1 | tail -1
echo "=== $(date +%H:%M) done"
