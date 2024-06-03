export CUDA_VISIBLE_DEVICES=1
for epoch in 20 30 40 50; do
  for avg in 10 15 20 30 40; do
    ./zipformer/decode.py \
      --epoch $epoch \
      --avg $avg \
      --exp-dir zipformer/exp_zipformer_60M_yodas_vi000_offline_vocab_2000 \
      --bpe-model data/lang_bpe_2000/bpe.model \
      --max-duration 2000 \
      --decoding-method greedy_search
  done
done