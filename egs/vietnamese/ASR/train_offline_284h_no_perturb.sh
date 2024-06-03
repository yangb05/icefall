export CUDA_VISIBLE_DEVICES="0,2,3,5,6"

./zipformer/train.py \
  --world-size 5 \
  --num-epochs 50 \
  --lr-epochs 35 \
  --lr-batches 5000 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_zipformer_60M_284h_offline_no_perturb \
  --bpe-model data/lang_bpe_2000/bpe.model \
  --keep-last-k 1 \
  --max-duration 2000 \
  --master-port 12355 \
  --num-workers 20
  
  