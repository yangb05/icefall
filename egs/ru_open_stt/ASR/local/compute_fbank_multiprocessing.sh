export CUDA_VISIBLE_DEVICES=0
start=$1
stop=$2
python3 ./local/compute_fbank_ru_open_stt_splits.py \
  --num-workers 20 \
  --batch-duration 600 \
  --start $start \
  --stop $stop \
  --num-splits 1000
