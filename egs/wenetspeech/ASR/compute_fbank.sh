start=$1
stop=$2
python3 ./local/compute_fbank_wenetspeech_splits.py \
    --training-subset L \
    --num-workers 20 \
    --num-splits 1000 \
    --batch-duration 600 \
    --start $start \
    --stop $stop
    