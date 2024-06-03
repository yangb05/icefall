export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"

./zipformer/train.py \
  --world-size 6 \
  --use-fp16 1 \
  --lr-epochs 22 \
  --lr-batches 2200 \
  --num-epochs 50 \
  --exp-dir zipformer/exp_zipformer_150M_464h_offline_lrepoch_22 \
  --bpe-model data/lang_bpe_2000/bpe.model \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --keep-last-k 1 \
  --max-duration 1500 \
  --num-workers 20
  
  