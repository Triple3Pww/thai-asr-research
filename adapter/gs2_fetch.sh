#!/usr/bin/env bash
# 2026-10-11 loop: fetch + extract GigaSpeech2-th TRAIN shards (speechcolab/gigaspeech2, CC-BY-4.0, HF login), delete the tarball.
# usage: bash adapter/gs2_fetch.sh 3 4
set -u; D=E:/Claude/datasets/gs2_train
for s in "$@"; do
  [ -d $D/audio/$s ] && { echo "shard $s already extracted"; continue; }
  E:/asr-spike/.venv/Scripts/python.exe -c "from huggingface_hub import hf_hub_download as h; print(h('speechcolab/gigaspeech2','data/th/train/$s.tar.gz',repo_type='dataset',local_dir='$D'))" || continue
  tar --force-local -xzf $D/data/th/train/$s.tar.gz -C $D/audio && rm -f $D/data/th/train/$s.tar.gz && echo "shard $s extracted: $(ls $D/audio/$s | wc -l) videos"
done
