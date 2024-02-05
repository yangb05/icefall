export CUDA_VISIBLE_DEVICES=0
for epoch in 21 20 19 18 17 16; do
  for avg in 1 2 3 4 5; do
    ./zipformer/streaming_decode.py \
      --num-encoder-layers 2,2,4,5,4,2 \
      --feedforward-dim 512,768,1536,2048,1536,768 \
      --encoder-dim 192,256,512,768,512,256 \
      --epoch $epoch \
      --avg $avg \
      --causal 1 \
      --chunk-size 16 \
      --left-context-frames 128 \
      --exp-dir zipformer/exp_scale_L_lr_epoch_3.5_fp16 \
      --bpe-model data/lang_bpe_2000/bpe.model \
      --decoding-method greedy_search \
      --num-decode-streams 2000
  done
done