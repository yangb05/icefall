export CUDA_VISIBLE_DEVICES=1
for epoch in 26 28 30; do
  for avg in 6 8 10; do
    ./zipformer/streaming_decode.py \
      --epoch $epoch \
      --avg $avg \
      --causal 1 \
      --chunk-size 16 \
      --left-context-frames 128 \
      --exp-dir ./zipformer/exp_lr_epoch_3.5_fp16_yodas_vi000 \
      --bpe-model data/lang_bpe_10000/bpe.model \
      --decoding-method greedy_search \
      --num-decode-streams 2000
  done
done