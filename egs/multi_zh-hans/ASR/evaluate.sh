export CUDA_VISIBLE_DEVICES=1
for epoch in 30 28 25 22 20 18; do
  for avg in 15 10 8 5 2; do
    ./zipformer/streaming_decode.py \
        --epoch $epoch \
        --avg $avg \
        --causal 1 \
        --chunk-size 16 \
        --left-context-frames 128 \
        --exp-dir ./zipformer/exp_lr_epoch_3.5_fp16_bpe_offline \
        --bpe-model data/lang_bpe_2000_offline/bpe.model \
        --decoding-method greedy_search \
        --num-decode-streams 2000
  done
done