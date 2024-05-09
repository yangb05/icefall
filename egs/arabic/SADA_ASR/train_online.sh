export CUDA_VISIBLE_DEVICES="0"

./zipformer/train.py \
  --world-size 1 \
  --use-fp16 1 \
  --causal 1 \
  --exp-dir zipformer/exp_online \
  --bpe-model data/lang_bpe_500/bpe.model \
  --keep-last-k 1 \
  --max-duration 1500 \
  --num-workers 20 \
  --master-port 12355
  
  