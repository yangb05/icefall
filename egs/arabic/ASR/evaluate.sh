export CUDA_VISIBLE_DEVICES=3
for epoch in 15 20 25 30; do
  for avg in 5 10 15 20; do
    ./zipformer/streaming_decode.py \
      --epoch $epoch \
      --avg $avg \
      --causal 1 \
      --chunk-size 16 \
      --left-context-frames 128 \
      --exp-dir ./zipformer/exp_lr_epoch_3.5_fp16 \
      --bpe-model data/lang_bpe_10000/bpe.model \
      --decoding-method greedy_search \
      --num-decode-streams 2000
  done
done