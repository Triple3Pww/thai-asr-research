#!/usr/bin/env bash
# The rest of the 2026-10-06 "DeepSeek for Thai ASR" pilots, strictly serial, detached (nohup) so a
# Claude Code shell reap cannot kill it. One GPU job at a time -- two in parallel filled RAM earlier.
#   nohup bash adapter/night_deepseek.sh > logs/night_deepseek.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=E:/asr-spike/.venv/Scripts/python.exe
FW=E:/fw-spike/.venv/Scripts/python.exe
BASE=E:/asr-spike/models/pathumma-stt-th-fastconformer-rnnt-large.nemo
L=E:/Claude/datasets/lotusdis_lav
TV=E:/asr-bench/typhoon-eval/manifest_tvspeech.jsonl
export PYTHONUTF8=1
log() { echo "=== $(date +%H:%M:%S) $*"; }

# lot2 was stopped in epoch 2 by a host-memory leak (pin_memory, fixed in finetune_rt.py); its
# finished epoch-1 checkpoint IS lot2 -- reported as 1 epoch, not 2.
cp -n checkpoints/rt_student_lot2_ep0.nemo checkpoints/rt_student_lot2.nemo
log "eval lot2 (1 epoch)";   SUF=_conv bash adapter/eval_chain_lot.sh lot2 checkpoints/rt_student_lot2.nemo

log "train lot3 = conv LOTUSDIS + synthetic entities"
$PY adapter/finetune_rt.py --base $BASE --train $L/manifest_train_conv_plus_synth.jsonl \
    --val $L/manifest_val_conv.jsonl --tag lot3 --epochs 2 --lr 5e-5 > logs/ft_lot3.log 2>&1
log "eval lot3";   SUF=_conv bash adapter/eval_chain_lot.sh lot3 checkpoints/rt_student_lot3.nemo

log "synthetic held-out (entity audio the tunes never saw, 60 held-out texts)"
$PY - <<'EOF'
import json
with open("E:/Claude/datasets/synth/manifest_nectec_heldout.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for l in open("E:/Claude/datasets/synth/manifest_heldout.jsonl", encoding="utf-8"):
        r = json.loads(l)
        f.write(json.dumps({"audio_filepath": r["wav"], "duration": r["dur"], "text": r["text"].replace("\u0e4d\u0e32", "\u0e33"), "id": r["utt_id"]}, ensure_ascii=False) + "\n")
EOF
SPECS="base=$BASE"; for t in lot2 lot3; do for a in a25 a50 a75; do SPECS="$SPECS ${t}_$a=checkpoints/${t}_$a.nemo"; done; SPECS="$SPECS ${t}_a100=checkpoints/rt_student_$t.nemo"; done
$PY adapter/eval_manifest.py --manifest E:/Claude/datasets/synth/manifest_nectec_heldout.jsonl --models $SPECS --tag deepseek_synthheld

log "bar on TVSpeech (typhoon-whisper-large-v3, CTranslate2, greedy)"
[ "$(wc -l < results/tvspeech_tw_hyps.jsonl 2>/dev/null || echo 0)" -ge 570 ] || \
  $FW adapter/fw_manifest.py --model E:/fw-spike/models/typhoon-whisper-large-v3-ct2 --manifest $TV --out results/tvspeech_tw_hyps.jsonl
log "RNNTs on TVSpeech + bar, one scorer"
$PY adapter/eval_manifest.py --manifest $TV --hyps tw=results/tvspeech_tw_hyps.jsonl --models $SPECS --tag deepseek_tvspeech
log "done"
