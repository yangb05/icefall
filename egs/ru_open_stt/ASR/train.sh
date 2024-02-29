export CUDA_VISIBLE_DEVICES="1"

./zipformer/train.py \
  --world-size 1 \
  --use-fp16 1 \
  --causal 1 \
  --exp-dir zipformer/exp_lr_epoch_3.5_fp16 \
  --bpe-model data/lang_bpe_5000/bpe.model \
  --keep-last-k 1 \
  --max-duration 3000 \
  --num-workers 20