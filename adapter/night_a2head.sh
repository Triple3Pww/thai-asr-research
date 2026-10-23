#!/usr/bin/env bash
# A2 + names: the conv-LOTUSDIS checkpoint (lot2 seed 1, alpha .75 -- the VAL choice) writes the transcript,
# and a CTC head trained on ITS frozen encoder spots the names. Same recipe as ctchead_h3 (MLP 1024, 30 ep).
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1
PY=E:/asr-spike/.venv/Scripts/python.exe
A2=checkpoints/lot2_a75.nemo
echo "=== $(date +%H:%M) cache A2 encoder features"
$PY adapter/ctc_head.py cache --base $A2 2>&1 | grep -E "\->|Traceback|Error"
echo "=== $(date +%H:%M) train head on A2 encoder"
$PY adapter/ctc_head.py train --base $A2 --tag a2h3 --hidden 1024 --epochs 30 2>&1 | grep -E "best|Traceback|Error"
echo "=== $(date +%H:%M) splice eval"
RNNT=$A2 bash adapter/night_head.sh checkpoints/ctchead_a2h3.pt
