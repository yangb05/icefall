./zipformer/train.py \
  --world-size 8 \
  --use-fp16 1 \
  --causal 1 \
  --exp-dir zipformer/exp_lr_epoch_3.5_fp16 \
  --bpe-model data/lang_bpe_5000/bpe.model \
  --keep-last-k 1 \
  --max-duration 300 \
  --num-workers 20
