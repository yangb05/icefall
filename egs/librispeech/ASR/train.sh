export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"

./zipformer/train.py \
  --world-size 6 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_zipformer_60M_960h_offline \
  --bpe-model data/lang_bpe_500/bpe.model \
  --keep-last-k 1 \
  --max-duration 1500 \
  --num-workers 20
  
  