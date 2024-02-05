export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"

./zipformer/train.py \
  --world-size 6 \
  --use-fp16 1 \
  --causal 1 \
  --exp-dir zipformer/exp_lr_epoch_3.5_fp16_bpe_offline \
  --bpe-model data/lang_bpe_2000_offline/bpe.model \
  --start-epoch 5 \
  --keep-last-k 1 \
  --max-duration 3000 \
  --num-workers 20
  
  