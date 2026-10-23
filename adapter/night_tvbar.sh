#!/usr/bin/env bash
# Re-run the TVSpeech bar WITH timestamps after night_bias.sh -- never two GPU jobs at once.
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1
while ! grep -q "=== .* d6 done" logs/night_bias.log; do sleep 30; done
echo "=== $(date +%H:%M) tw TVSpeech with timestamps"
E:/fw-spike/.venv/Scripts/python.exe adapter/fw_manifest.py --timestamps --model E:/fw-spike/models/typhoon-whisper-large-v3-ct2 \
  --manifest E:/asr-bench/typhoon-eval/manifest_tvspeech.jsonl --out results/tvspeech_tw_ts_hyps.jsonl
E:/asr-spike/.venv/Scripts/python.exe adapter/eval_manifest.py --manifest E:/asr-bench/typhoon-eval/manifest_tvspeech.jsonl \
  --hyps tw_notimestamps=results/tvspeech_tw_hyps.jsonl tw_ts=results/tvspeech_tw_ts_hyps.jsonl \
  base=results/deepseek_tvspeech_base_hyps.jsonl lot2_a50=results/deepseek_tvspeech_lot2_a50_hyps.jsonl \
  lot2_a75=results/deepseek_tvspeech_lot2_a75_hyps.jsonl --tag deepseek_tvspeech_bar
echo "=== $(date +%H:%M) done"
