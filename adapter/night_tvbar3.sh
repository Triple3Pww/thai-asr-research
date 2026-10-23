#!/usr/bin/env bash
# TVSpeech bar, final: audio cut at speech pauses into <= 15 s pieces, each decoded alone (fw_manifest --vad-max-s).
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1
echo "=== $(date +%H:%M) tw TVSpeech vad-cut 15 s"
E:/fw-spike/.venv/Scripts/python.exe adapter/fw_manifest.py --vad-max-s 15 --model E:/fw-spike/models/typhoon-whisper-large-v3-ct2 \
  --manifest E:/asr-bench/typhoon-eval/manifest_tvspeech.jsonl --out results/tvspeech_tw_vad15_hyps.jsonl
E:/asr-spike/.venv/Scripts/python.exe adapter/eval_manifest.py --manifest E:/asr-bench/typhoon-eval/manifest_tvspeech.jsonl \
  --hyps tw_cap224=results/tvspeech_tw_hyps.jsonl tw_vad15=results/tvspeech_tw_vad15_hyps.jsonl \
  base=results/deepseek_tvspeech_base_hyps.jsonl lot2_a50=results/deepseek_tvspeech_lot2_a50_hyps.jsonl \
  lot2_a75=results/deepseek_tvspeech_lot2_a75_hyps.jsonl lot3_a50=results/deepseek_tvspeech_lot3_a50_hyps.jsonl --tag deepseek_tvspeech_bar
echo "=== $(date +%H:%M) done"
