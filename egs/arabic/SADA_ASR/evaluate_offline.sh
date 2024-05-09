export CUDA_VISIBLE_DEVICES=1
for epoch in 20 30 40 50; do
  for avg in 10 15 20 25; do
    ./zipformer/decode.py \
      --epoch $epoch \
      --avg $avg \
      --exp-dir ./zipformer/exp_offline_500 \
      --bpe-model data/lang_bpe_500/bpe.model \
      --max-duration 1500 \
      --decoding-method greedy_search
  done
done