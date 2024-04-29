export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"

./zipformer/train.py \
  --world-size 6 \
  --use-fp16 1 \
  --exp-dir /data_a100/userhome/yangb/data/checkpoints/vietnamese_ASR/exp_zipformer_60M_yodas_vi000_offline_reproduce \
  --bpe-model data/lang_bpe_10000/bpe.model \
  --keep-last-k 1 \
  --max-duration 1500 \
  --num-epochs 50 \
  --num-workers 20
  
  