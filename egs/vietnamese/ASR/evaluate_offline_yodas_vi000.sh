export CUDA_VISIBLE_DEVICES=1
for epoch in 30 35 40 45 50; do
  for avg in 10 15 20 25 30; do
    ./zipformer/decode.py \
      --epoch $epoch \
      --avg $avg \
      --exp-dir ./zipformer/exp_zipformer_60M_yodas_vi000_offline \
      --bpe-model data/lang_bpe_10000/bpe.model \
      --max-duration 1500 \
      --decoding-method greedy_search
  done
done