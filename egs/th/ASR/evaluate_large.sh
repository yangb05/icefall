export CUDA_VISIBLE_DEVICES=0
for epoch in 20 30 40 50; do
  for avg in 10 15 20 30; do
    ./zipformer/decode.py \
      --epoch $epoch \
      --avg $avg \
      --exp-dir ./zipformer/exp_zipformer_150M_464h_offline_lrepoch_22 \
      --bpe-model data/lang_bpe_2000/bpe.model \
      --num-encoder-layers 2,2,4,5,4,2 \
      --feedforward-dim 512,768,1536,2048,1536,768 \
      --encoder-dim 192,256,512,768,512,256 \
      --encoder-unmasked-dim 192,192,256,320,256,192 \
      --max-duration 5000 \
      --decoding-method greedy_search
  done
done