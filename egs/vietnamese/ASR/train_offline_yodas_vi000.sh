export CUDA_VISIBLE_DEVICES="3,4,5,6,7"

./zipformer/train.py \
  --world-size 5 \
  --use-fp16 1 \
  --num-epochs 50 \
  --lr-epochs 14 \
  --lr-batches 11200 \
  --exp-dir zipformer/exp_zipformer_60M_yodas_vi000_offline_vocab_2000 \
  --bpe-model data/lang_bpe_2000/bpe.model \
  --keep-last-k 1 \
  --max-duration 1500 \
  --num-workers 20
  
  