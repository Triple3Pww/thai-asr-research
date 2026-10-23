#!/usr/bin/env bash
# E8b: D6-on-NECTEC splice boundary pads. Chosen on the SYNTH split only, then one real bias@50 + dis@50.
# Waits for the TVSpeech bar (night_tvbar2.sh) -- never two GPU jobs at once.
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1
PY=E:/asr-spike/.venv/Scripts/python.exe
NEC=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
while ! grep -q "=== .* done" logs/night_tvbar2.log; do sleep 30; done
for pp in 0:0 0:1 0:2 0:3 1:1 1:2 2:2; do ps=${pp%%:*}; pe=${pp##*:}
  echo "=== $(date +%H:%M) synth bias@50 pad $ps:$pe"
  $PY adapter/arm_d6_splice.py --mode rnnt --rnnt-model $NEC --split synth --condition bias@50 --tau -1.5 \
     --pad-start $ps --pad-end $pe --tag nec_d6pad_synth_bias50_p${ps}_${pe} 2>&1 | grep -E "CER|recall" | tail -1
done
best=$($PY - <<'PYX'
import json,glob,re
best=None
for f in glob.glob("results/nec_d6pad_synth_bias50_p*_metrics.json"):
    d=json.load(open(f,encoding="utf-8")); m=d.get("bias@50",d)
    k=re.search(r"_p(\d+)_(\d+)_metrics",f).groups()
    if best is None or m["cer"]<best[0]: best=(m["cer"],k)
print(best[1][0],best[1][1])
PYX
)
set -- $best; echo "=== chosen on synth: pad $1:$2"
for c in dis@50 bias@50; do
  echo "=== $(date +%H:%M) real $c pad $1:$2"
  $PY adapter/arm_d6_splice.py --mode rnnt --rnnt-model $NEC --split real --condition $c --tau -1.5 \
     --pad-start $1 --pad-end $2 --tag nec_d6pad_real_${c/@/}_p$1_$2 2>&1 | grep -E "CER|recall" | tail -1
done
echo "=== $(date +%H:%M) done"
