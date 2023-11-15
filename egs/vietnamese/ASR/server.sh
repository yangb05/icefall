python3 /mgData2/yangb/sherpa-onnx/python-api-examples/streaming_server.py \
  --encoder ./zipformer/exp_lr_epoch_5.5_fp16/encoder-epoch-25-avg-15-chunk-16-left-128.onnx \
  --decoder ./zipformer/exp_lr_epoch_5.5_fp16/decoder-epoch-25-avg-15-chunk-16-left-128.onnx \
  --joiner ./zipformer/exp_lr_epoch_5.5_fp16/joiner-epoch-25-avg-15-chunk-16-left-128.onnx \
  --tokens ./data/lang_bpe_10000/tokens.txt \
  --doc-root /mgData2/yangb/sherpa-onnx/python-api-examples/web \
  --port 40040