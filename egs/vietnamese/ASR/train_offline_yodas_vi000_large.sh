export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./zipformer/train.py \
  --world-size 8 \
  --use-fp16 1 \
  --num-epochs 50 \
  --lr-epochs 14 \
  --lr-batches 11200 \
  --exp-dir zipformer/exp_zipformer_150M_yodas_vi000_offline_vocab_2000 \
  --bpe-model data/lang_bpe_2000/bpe.model \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --keep-last-k 1 \
  --max-duration 1500 \
  --num-workers 20
  
  