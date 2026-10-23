#!/usr/bin/env bash
# TVSpeech bar, third attempt: full decoder budget (max_new_tokens 440). See fw_manifest.py notes.
set -u; cd "$(dirname "$0")/.."; export PYTHONUTF8=1
echo "=== $(date +%H:%M) tw TVSpeech max_new_tokens=440"
E:/fw-spike/.venv/Scripts/python.exe adapter/fw_manifest.py --max-new-tokens 440 --model E:/fw-spike/models/typhoon-whisper-large-v3-ct2 \
  --manifest E:/asr-bench/typhoon-eval/manifest_tvspeech.jsonl --out results/tvspeech_tw_full_hyps.jsonl
E:/asr-spike/.venv/Scripts/python.exe adapter/eval_manifest.py --manifest E:/asr-bench/typhoon-eval/manifest_tvspeech.jsonl \
  --hyps tw_cap224=results/tvspeech_tw_hyps.jsonl tw_full=results/tvspeech_tw_full_hyps.jsonl \
  base=results/deepseek_tvspeech_base_hyps.jsonl lot2_a50=results/deepseek_tvspeech_lot2_a50_hyps.jsonl \
  lot2_a75=results/deepseek_tvspeech_lot2_a75_hyps.jsonl lot3_a50=results/deepseek_tvspeech_lot3_a50_hyps.jsonl --tag deepseek_tvspeech_bar
echo "=== $(date +%H:%M) done"
