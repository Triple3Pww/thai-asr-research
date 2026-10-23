#!/usr/bin/env bash
# 2026-10-12: our best real-time models vs the published Thai models on the PUBLIC test sets Typhoon/NECTEC report
# (TVSpeech, FLEURS-th test), one scorer (eval_manifest: gate_lotusdis.norm, pooled CER), one window.
# Overlap with our training text checked first: no test sentence leaked (LOOP row 16).
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=E:/asr-spike/.venv/Scripts/python.exe; FW=E:/fw-spike/.venv/Scripts/python.exe
BASE=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
TYP=C:/Users/MARU/.cache/huggingface/hub/models--typhoon-ai--typhoon-asr-realtime/snapshots/2c58a30ba9a3bf92d095a5df91bec6996f04c3a1/typhoon-asr-realtime.nemo
TV=E:/asr-bench/typhoon-eval/manifest_tvspeech.jsonl; FL=E:/Claude/datasets/fleurs_th/manifest_test_eval.jsonl
M="nectec=$BASE r2ep_a75=checkpoints/r2ep_a75.nemo gs2a_a50=checkpoints/gs2a_a50.nemo typhoon_rt=$TYP"
echo "=== $(date +%H:%M) whisper on FLEURS test"
for w in "twL typhoon-whisper-large-v3-ct2" "patL nectec_Pathumma-whisper-th-large-v3-ct2"; do set -- $w
  # clips <= 28 s: batched CTranslate2 (one 30 s window holds them); the 13 longer ones: VAD chunks <= 15 s
  o=results/fleurs_test_$1_hyps.jsonl
  [ -f $o ] && [ $(wc -l < $o) -ge 1021 ] && continue
  $FW adapter/fw_batch.py --model E:/fw-spike/models/$2 --manifest E:/Claude/datasets/fleurs_th/manifest_test_short.jsonl --out results/fleurs_test_$1_short.jsonl > logs/pub_fw_$1.log 2>&1
  $FW adapter/fw_manifest.py --model E:/fw-spike/models/$2 --manifest $FL --ids E:/Claude/datasets/fleurs_th/test_long_ids.txt --out results/fleurs_test_$1_long.jsonl --vad-max-s 15 >> logs/pub_fw_$1.log 2>&1
  cat results/fleurs_test_$1_short.jsonl results/fleurs_test_$1_long.jsonl > $o
done
echo "=== $(date +%H:%M) TVSpeech"
$PY adapter/eval_manifest.py --manifest $TV --tag pub_tvspeech --models $M \
  --hyps twL=results/tvspeech_tw_vad15_hyps.jsonl patL=results/tvspeech_patL_vad15_hyps.jsonl 2>&1 | grep -E "^[a-zA-Z0-9_]+ 0\.[0-9]+$"
echo "=== $(date +%H:%M) FLEURS test"
$PY adapter/eval_manifest.py --manifest $FL --tag pub_fleurs --models $M \
  --hyps twL=results/fleurs_test_twL_hyps.jsonl patL=results/fleurs_test_patL_hyps.jsonl 2>&1 | grep -E "^[a-zA-Z0-9_]+ 0\.[0-9]+$"
echo "=== $(date +%H:%M) END pubsets"
