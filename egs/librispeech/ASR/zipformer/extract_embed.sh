#!/bin/bash
set -e

#config
epoch=30
avg=10
exp_dir="/mgData2/yangb/icefall/egs/librispeech/ASR/zipformer/exp_zipformer_60M_960h_offline"
bpe_model="/mgData2/yangb/icefall/egs/librispeech/ASR/data/lang_bpe_500/bpe.model"
cut_file="/data_a100/userhome/yangb/data/fbank/librispeech_cuts_test-other.jsonl.gz"
embed_file="/data_a100/userhome/yangb/data/fbank/librispeech_embed_test-other.npy"

. /mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/audio-discretizer/utils/parse_options.sh
echo "epoch: ${epoch}"
echo "avg: ${avg}"
echo "exp_dir: ${exp_dir}"
echo "bpe_model: ${bpe_model}"
echo "cut_file: ${cut_file}"
echo "embed_file: ${embed_file}"

python /mgData2/yangb/icefall/egs/librispeech/ASR/zipformer/extract_embed.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir $exp_dir \
    --bpe-model $bpe_model \
    --cut-file $cut_file \
    --embed-file $embed_file
