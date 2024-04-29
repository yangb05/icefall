export CUDA_VISIBLE_DEVICES=1
for epoch in 10; do
  for avg in 3 5 8; do
    ./zipformer/decode.py \
      --epoch $epoch \
      --avg $avg \
      --exp-dir /data_a100/userhome/yangb/data/checkpoints/vietnamese_ASR/exp_zipformer_60M_yodas_vi000_offline_reproduce \
      --bpe-model data/lang_bpe_10000/bpe.model \
      --max-duration 1500 \
      --decoding-method greedy_search
  done
done