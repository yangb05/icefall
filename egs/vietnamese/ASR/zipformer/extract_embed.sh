#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1
#config
epoch=30
avg=15
exp_dir="/mgData2/yangb/icefall/egs/vietnamese/ASR/zipformer/exp_zipformer_60M_2200h_offline_no_perturb"
encoder_embed="epoch-30-avg-15-encoder-embed.pt"
bpe_model="/mgData2/yangb/icefall/egs/vietnamese/ASR/data/lang_bpe_10000/bpe.model"
cut_file="/mgData2/yangb/icefall/egs/vietnamese/ASR/data/fbank/vietnamese_cuts_train_no_perturb.jsonl.gz"
embed_file="/data_a100/userhome/yangb/data/fbank/vietnamese_embed_train_no_perturb.npy"

. /mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/audio-discretizer/utils/parse_options.sh
echo "epoch: ${epoch}"
echo "avg: ${avg}"
echo "exp_dir: ${exp_dir}"
echo "bpe_model: ${bpe_model}"
echo "cut_file: ${cut_file}"
echo "embed_file: ${embed_file}"

python /mgData2/yangb/icefall/egs/vietnamese/ASR/zipformer/extract_embed.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir $exp_dir \
    --encoder-embed $encoder_embed \
    --bpe-model $bpe_model \
    --cut-file $cut_file \
    --embed-file $embed_file
