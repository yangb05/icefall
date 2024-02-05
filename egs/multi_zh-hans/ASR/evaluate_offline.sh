export CUDA_VISIBLE_DEVICES=0
for epoch in 20; do
  for avg in 1; do
    ./zipformer/decode.py \
      --epoch $epoch \
      --avg $avg \
      --use-averaged-model False \
      --exp-dir ./zipformer/exp_offline \
      --bpe-model data/lang_bpe_2000_offline/bpe.model \
      --max-duration 1500 \
      --decoding-method greedy_search
  done
done