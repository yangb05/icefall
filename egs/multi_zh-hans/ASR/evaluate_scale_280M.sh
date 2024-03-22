export CUDA_VISIBLE_DEVICES=0
for epoch in 13 15 18 20; do
  for avg in 1 3 5 7; do
    ./zipformer/streaming_decode.py \
      --num-encoder-layers 2,2,4,5,4,2 \
      --feedforward-dim 512,1024,2048,3072,2048,1024 \
      --encoder-dim 192,384,768,1024,768,384 \
      --encoder-unmasked-dim 192,256,320,512,320,256 \
      --epoch $epoch \
      --avg $avg \
      --causal 1 \
      --chunk-size 16 \
      --left-context-frames 128 \
      --exp-dir zipformer/exp_scale_280M_lr_epoch_3.5_fp16 \
      --bpe-model data/lang_bpe_2000/bpe.model \
      --decoding-method greedy_search \
      --num-decode-streams 2000
  done
done
