#!/usr/bin/env bash
# Decode-time contextual biasing on FROZEN NECTEC (reviewer thread 3f875f07, 2026-10-06): the one route
# left after every weight-space pilot failed. Waits for night_deepseek.sh -- never two GPU jobs at once.
# Protocol from arm_d4_boosting.py: control first, sweep on SYNTH only, one held-out run on REAL,
# bias@N always beside dis@N (CLAUDE.md: the scorer credits longer lists for free).
set -u
cd "$(dirname "$0")/.."
PY=E:/asr-spike/.venv/Scripts/python.exe
NEC=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
export PYTHONUTF8=1
while ! grep -q "=== .* done" logs/night_deepseek.log; do sleep 30; done
echo "=== $(date +%H:%M) control none"
$PY adapter/arm_d4_boosting.py --model $NEC --split real --condition none --tag nec_d4_real_none
echo "=== $(date +%H:%M) synth sweep bias@50"
$PY adapter/arm_d4_boosting.py --model $NEC --split synth --condition bias@50 --sweep 0.5:1,1:1,1:3,2:3,2:5,4:5 --tag "" 2>&1 | tail -15
echo "=== $(date +%H:%M) done"

# D6 on NECTEC: Typhoon's CTC spotter (separate 115M pass) + NECTEC's transcript, spliced by frame.
# tau -1.5 = their recommended default, the value D6 was reported at -- no tuning on real.
while pgrep -f arm_d4_boosting >/dev/null 2>&1 || tasklist | grep -q "arm_d4"; do sleep 30; done
for c in none dis@50 bias@50; do
  echo "=== $(date +%H:%M) d6 nectec $c"
  $PY adapter/arm_d6_splice.py --mode rnnt --rnnt-model $NEC --condition $c --tau -1.5 --tag nec_d6_rnnt_real_${c/@/}_tau-1.5 2>&1 | tail -4
done
echo "=== $(date +%H:%M) d6 done"
