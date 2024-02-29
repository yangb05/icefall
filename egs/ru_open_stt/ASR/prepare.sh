#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=15
stage=-1
stop_stage=11

# Split trainset to this number of pieces
# This is to avoid OOM during feature extraction.
num_splits=1000

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/ru_open_stt
#      You can find untar, wavedata inside it.

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# vocab size for sentence piece models.
# It will generate data/lang_bbpe_xxx,
# data/lang_bbpe_yyy if the array contains xxx, yyy
vocab_sizes=(
  # 2000
  # 1000
  5000
)

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "stage 0: Download data"

  # If you have pre-downloaded it to /path/to/ru_open_stt,
  # you can create a symlink
  #
  #   ln -sfv /path/to/ru_open_stt $dl_dir/ru_open_stt
  #
  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  #
  #   ln -sfv /path/to/musan $dl_dir/musan
  #
  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare ru_open_stt manifest"
  # We assume that you have downloaded the ru_open_stt corpus
  # to $dl_dir/ru_open_stt
  if [ ! -f data/manifests/.aishell_manifests.done ]; then
    mkdir -p data/manifests
    lhotse prepare ru-open-stt $dl_dir/ru_open_stt data/manifests
    touch data/manifests/.ru_open_stt_manifests.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  if [ ! -f data/manifests/.musan_manifests.done ]; then
    ln -svf /mgData2/yangb/icefall/egs/multi_zh-hans/ASR/data/manifests/musan* data/manifests/
    touch data/manifests/.musan_manifests.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Preprocess ru_open_stt manifest"
  if [ ! -f /data_a100/userhome/yangb/data/fbank/.preprocess_complete ]; then
    python3 ./local/preprocess_ru_open_stt.py --perturb-speed True
    touch /data_a100/userhome/yangb/data/fbank/.preprocess_complete
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for ru_open_stt DEV and TEST"
  python3 ./local/compute_fbank_ru_open_stt_dev_test.py
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Split Train into ${num_splits} pieces"
  split_dir=/data_a100/userhome/yangb/data/fbank/train_split_${num_splits}
  if [ ! -f $split_dir/.split_completed ]; then
    lhotse split $num_splits /data_a100/userhome/yangb/data/fbank/ru_open_stt_cuts_train_raw.jsonl.gz $split_dir
    touch $split_dir/.split_completed
  fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Compute features for Train"
  python3 ./local/compute_fbank_ru_open_stt_splits.py \
    --num-workers 20 \
    --batch-duration 600 \
    --start 0 \
    --stop 100 \
    --num-splits $num_splits
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Combine features for Train"
  if [ ! -f /data_a100/userhome/yangb/data/fbank/ru_open_stt_cuts_train.jsonl.gz ]; then
    pieces=$(find /data_a100/userhome/yangb/data/fbank/train_split_1000 -name "ru_open_stt_cuts_train.*.jsonl.gz")
    lhotse combine $pieces /data_a100/userhome/yangb/data/fbank/ru_open_stt_cuts_train.jsonl.gz
  fi
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Compute fbank for musan"
  if [ ! -f data/fbank/.msuan.done ]; then
    mkdir -p data/fbank
    ln -svf /mgData2/yangb/icefall/egs/multi_zh-hans/ASR/data/fbank/musan* data/fbank/
    touch data/fbank/.msuan.done
  fi
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Prepare BPE based lang"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    mkdir -p $lang_dir

    if [ ! -f $lang_dir/transcript_words.txt ]; then
      log "Generate data for BPE training"
      file=$(
        find "data/manifests/ru_open_stt_supervisions_train.jsonl.gz"
      )
      gunzip -c ${file} | awk -F '"' '{print $18}' > $lang_dir/transcript_words.txt

      # Ensure space only appears once
      sed -i 's/\t/ /g' $lang_dir/transcript_words.txt
      sed -i 's/[ ][ ]*/ /g' $lang_dir/transcript_words.txt
      sed -i "s/\xe2\x80\x8b//g" $lang_dir/transcript_words.txt
      sed -i "s/\xe2\x80\x8c//g" $lang_dir/transcript_words.txt
    fi
 
    if [ ! -f $lang_dir/words.txt ]; then
      cat $lang_dir/transcript_words.txt | sed 's/ /\n/g' \
        | sort -u | sed '/^$/d' > $lang_dir/words.txt
      (echo '!SIL'; echo '<SPOKEN_NOISE>'; echo '<UNK>'; ) |
        cat - $lang_dir/words.txt | sort | uniq | awk '
        BEGIN {
          print "<eps> 0";
        }
        {
          if ($1 == "<s>") {
            print "<s> is in the vocabulary!" | "cat 1>&2"
            exit 1;
          }
          if ($1 == "</s>") {
            print "</s> is in the vocabulary!" | "cat 1>&2"
            exit 1;
          }
          printf("%s %d\n", $1, NR);
        }
        END {
          printf("#0 %d\n", NR+1);
          printf("<s> %d\n", NR+2);
          printf("</s> %d\n", NR+3);
        }' > $lang_dir/words || exit 1;
      mv $lang_dir/words $lang_dir/words.txt
    fi
 
    if [ ! -f $lang_dir/bpe.model ]; then
      ./local/train_bpe_model.py \
        --lang-dir $lang_dir \
        --vocab-size $vocab_size \
        --transcript $lang_dir/transcript_words.txt
    fi
  
    if [ ! -f $lang_dir/L_disambig.pt ]; then
      ./local/prepare_lang_bpe.py --lang-dir $lang_dir

      log "Validating $lang_dir/lexicon.txt"
      ./local/validate_bpe_lexicon.py \
        --lexicon $lang_dir/lexicon.txt \
        --bpe-model $lang_dir/bpe.model
    fi

    if [ ! -f $lang_dir/L.fst ]; then
      log "Converting L.pt to L.fst"
      ./shared/convert-k2-to-openfst.py \
        --olabels aux_labels \
        $lang_dir/L.pt \
        $lang_dir/L.fst
    fi

    if [ ! -f $lang_dir/L_disambig.fst ]; then
      log "Converting L_disambig.pt to L_disambig.fst"
      ./shared/convert-k2-to-openfst.py \
        --olabels aux_labels \
        $lang_dir/L_disambig.pt \
        $lang_dir/L_disambig.fst
    fi
  done
fi