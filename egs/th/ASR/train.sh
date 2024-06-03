export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"

./zipformer/train.py \
  --world-size 6 \
  --use-fp16 1 \
  --lr-epochs 22 \
  --lr-batches 2200 \
  --num-epochs 50 \
  --exp-dir zipformer/exp_zipformer_60M_464h_offline_lrepoch_22 \
  --bpe-model data/lang_bpe_2000/bpe.model \
  --keep-last-k 1 \
  --max-duration 1500 \
  --num-workers 20
  
  