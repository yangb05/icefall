export CUDA_VISIBLE_DEVICES="1"

./zipformer/train.py \
  --world-size 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_offline_500 \
  --bpe-model data/lang_bpe_500/bpe.model \
  --keep-last-k 1 \
  --num-epochs 50 \
  --max-duration 1500 \
  --num-workers 20 \
  --master-port 12355
  
  