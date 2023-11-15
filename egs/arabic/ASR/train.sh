export CUDA_VISIBLE_DEVICES="4,5,6,7"

./zipformer/train.py \
  --world-size 4 \
  --use-fp16 1 \
  --causal 1 \
  --exp-dir zipformer/exp_lr_epoch_3.5_fp16 \
  --bpe-model data/lang_bpe_10000/bpe.model \
  --keep-last-k 5 \
  --max-duration 1500 \
  --num-workers 20
  
  