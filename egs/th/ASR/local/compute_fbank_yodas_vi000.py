#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This file computes fbank features of the vietnamese dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
from filter_cuts import filter_cuts
from lhotse import CutSet, KaldifeatFbank, KaldifeatFbankConfig
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import str2bool

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bpe-model",
        type=str,
        help="""Path to the bpe.model. If not None, we will remove short and
        long utterances before extracting features""",
    )

    parser.add_argument(
        "--perturb-speed",
        type=str2bool,
        default=True,
        help="""Perturb speed with factor 0.9 and 1.1 on train subset.""",
    )

    return parser.parse_args()


def compute_fbank_yodas_vi000(
    bpe_model: Optional[str] = None,
    perturb_speed: Optional[bool] = True,
):
    src_dir = Path("/data_a100/userhome/yangb/data/yodas/manifests")
    output_dir = Path("/data_a100/userhome/yangb/data/yodas/fbank")
    # number of workers in dataloader
    num_workers = 20
    # number of seconds in a batch
    batch_duration = 600
    subsets = ("vi000",)
    if bpe_model:
        logging.info(f"Loading {bpe_model}")
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))

    logging.info(f"device: {device}")
    logging.info("Loading manifest")
    prefix = "yodas"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=subsets,
        output_dir=src_dir,
        suffix=suffix,
        prefix=prefix,
    )
    assert manifests is not None

    assert len(manifests) == len(subsets), (
        len(manifests),
        len(subsets),
        list(manifests.keys()),
        subsets,
    )
    
    for partition in subsets:
        cuts_path = output_dir / f"{prefix}_cuts_{partition}.{suffix}"
        if cuts_path.is_file():
            logging.info(f"{cuts_path} exists - skipping")
            continue
        cut_set = CutSet.from_manifests(
            recordings=manifests[partition]["recordings"],
            supervisions=manifests[partition]["supervisions"],
        )
        print(cut_set.describe())
        if bpe_model:
            cut_set = filter_cuts(cut_set, sp)
        if perturb_speed:
            logging.info(f"Doing speed perturb")
            cut_set = (
                cut_set
                + cut_set.perturb_speed(0.9)
                + cut_set.perturb_speed(1.1)
            )
        logging.info(f"Computing features for {partition}")
        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{output_dir}/{prefix}_feats_{partition}",
            num_workers=num_workers,
            batch_duration=batch_duration,
            overwrite=True,
        )
        logging.info(f"Saving to {cuts_path}")
        cut_set.to_file(cuts_path)
        logging.info(f"Saved to {cuts_path}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    compute_fbank_yodas_vi000(
        bpe_model=args.bpe_model,
        perturb_speed=args.perturb_speed
    )
