export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"

./zipformer/train.py \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,1024,2048,3072,2048,1024 \
  --encoder-dim 192,384,768,1024,768,384 \
  --encoder-unmasked-dim 192,256,320,512,320,256 \
  --world-size 6 \
  --use-fp16 1 \
  --causal 1 \
  --num-epochs 50 \
  --exp-dir zipformer/exp_scale_280M_lr_epoch_3.5_fp16 \
  --bpe-model data/lang_bpe_2000/bpe.model \
  --keep-last-k 1 \
  --max-duration 2000 \
  --num-workers 20 \
  --master-port 12355
