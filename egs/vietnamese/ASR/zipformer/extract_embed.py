#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Yifan Yang)
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


import argparse
from pathlib import Path


import torch
from tqdm import tqdm
import numpy as np
from train import add_model_arguments, get_model, get_params
import sentencepiece as spm
from icefall.checkpoint import average_checkpoints_with_averaged_model, find_checkpoints
from lhotse import CutSet


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=15,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=5,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="/data_a100/userhome/yangb/data/checkpoints/vietnamese_ASR/exp_zipformer_60M_2200h_offline",
        help="The experiment dir",
    )

    parser.add_argument(
        "--encoder-embed",
        type=str,
        default="epoch-30-avg-15-encoder-embed.pt",
        help="The experiment dir",
    )
    
    parser.add_argument(
        "--bpe-model",
        type=str,
        default="/mgData2/yangb/icefall/egs/vietnamese/ASR/data/lang_bpe_10000/bpe.model",
        help="Path to the bpe model",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )
    
    parser.add_argument(
        "--cut-file",
        type=str,
        default="/mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/data/fbank/musan_cuts.jsonl.gz",
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )
    
    parser.add_argument(
        "--embed-file",
        type=str,
        default="/mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/data/fbank/musan_embed.npy",
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device",
    )
    
    add_model_arguments(parser)

    return parser


@torch.no_grad()
def load_model(params):
    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    print("Script started")
    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    params.blank_id = sp.piece_to_id("<blk>")
    params.blank_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    print("About to create model")
    model = get_model(params)

    if params.iter > 0:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
            : params.avg + 1
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
            )
        elif len(filenames) < params.avg + 1:
            raise ValueError(
                f"Not enough checkpoints ({len(filenames)}) found for"
                f" --iter {params.iter}, --avg {params.avg}"
            )
        filename_start = filenames[-1]
        filename_end = filenames[0]
        print(
            "Calculating the averaged model over iteration checkpoints"
            f" from {filename_start} (excluded) to {filename_end}"
        )
        model.load_state_dict(
            average_checkpoints_with_averaged_model(
                filename_start=filename_start,
                filename_end=filename_end,
            )
        )
    else:
        assert params.avg > 0, params.avg
        start = params.epoch - params.avg
        assert start >= 1, start
        filename_start = f"{params.exp_dir}/epoch-{start}.pt"
        filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
        print(
            f"Calculating the averaged model over epoch range from "
            f"{start} (excluded) to {params.epoch}"
        )
        model.load_state_dict(
            average_checkpoints_with_averaged_model(
                filename_start=filename_start,
                filename_end=filename_end,
            )
        )
    model.eval()
    if not (params.exp_dir / params.encoder_embed).exists():
        torch.save({"model_state_dict": model.encoder_embed.state_dict()}, params.exp_dir / params.encoder_embed)
    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")
    print("Loading model Done!")
    return model
        

def run():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    params = get_params()
    params.update(vars(args))
    cut_embeds = {}
    model = load_model(params)
    device = torch.device(params.device)
    print(f"Device: {device}")
    model.to(params.device)
    model.eval()
    # get cutset
    cutset = CutSet.from_file(args.cut_file)
    # get embeds
    batch = []
    for cut in tqdm(cutset):
        fbank = cut.load_features()
        x = torch.tensor(fbank).unsqueeze(0).to(device)
        x_lens = torch.tensor(x.size(1)).unsqueeze(0).to(device)
        embed = model.encoder_embed(x, x_lens)[0].squeeze(0).detach().cpu().numpy()
        cut_embeds[cut.id] = embed
    # save embeds
    np.save(params.embed_file, cut_embeds)    


if __name__ == "__main__":
    run()
