export CUDA_VISIBLE_DEVICES=0
for epoch in 20 30 40 50; do
  for avg in 10 15 20 25 30; do
    ./zipformer/decode.py \
      --epoch $epoch \
      --avg $avg \
      --exp-dir zipformer/exp_zipformer_60M_284h_offline_no_perturb \
      --bpe-model data/lang_bpe_2000/bpe.model \
      --max-duration 1500 \
      --decoding-method greedy_search
  done
done