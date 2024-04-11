export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"

./zipformer/train.py \
  --world-size 6 \
  --use-fp16 1 \
  --causal 1 \
  --exp-dir zipformer/exp_zzy_yodas_vi000 \
  --bpe-model data/lang_bpe_10000/bpe.model \
  --keep-last-k 1 \
  --max-duration 1500 \
  --num-workers 20
