export CUDA_VISIBLE_DEVICES=1
for epoch in 15 20 25 30; do
  for avg in 5 10 15 20; do
    ./zipformer/decode.py \
      --epoch $epoch \
      --avg $avg \
      --exp-dir zipformer/exp_zipformer_60M_2200h_offline_no_perturb \
      --bpe-model data/lang_bpe_10000/bpe.model \
      --max-duration 1500 \
      --decoding-method greedy_search
  done
done