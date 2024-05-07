export CUDA_VISIBLE_DEVICES="1,2,3"

./zipformer/train.py \
  --world-size 3 \
  --use-fp16 1 \
  --causal 1 \
  --exp-dir zipformer/exp_online \
  --bpe-model data/lang_bpe_10000/bpe.model \
  --keep-last-k 1 \
  --max-duration 1500 \
  --num-workers 20
  
  