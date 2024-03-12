export CUDA_VISIBLE_DEVICES=0
for epoch in 20; do
  for avg in 1; do
    ./zipformer/streaming_decode.py \
      --num-encoder-layers 2,2,5,6,5,2 \
      --feedforward-dim 512,768,2048,2560,2048,768 \
      --encoder-dim 192,256,768,1280,768,256 \
      --epoch $epoch \
      --avg $avg \
      --causal 1 \
      --chunk-size 16 \
      --left-context-frames 128 \
      --exp-dir zipformer/exp_scale_XL_lr_epoch_3.5_fp16 \
      --bpe-model data/lang_bpe_2000/bpe.model \
      --decoding-method greedy_search \
      --num-decode-streams 2000
  done
done