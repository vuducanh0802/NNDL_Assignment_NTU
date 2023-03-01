#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
    cal_gcmvn_stats,
)
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from fairseq.data.audio.audio_utils import get_waveform, convert_waveform
from examples.joint_embedding_pretraining.dataprep.mustc_dataset import MUSTC
from examples.joint_embedding_pretraining.dataprep.covost_dataset import CoVoST
from examples.joint_embedding_pretraining.dataprep.europarl_dataset import EUROPARL

# general processing script to handle ST/ASR datasets.
# Each dataset should be written as a separate torch dataclass (and imported into this script) that extracts:
# waveform, sample_rate, source sentence, target sentence, speaker_id, utterance id
# and return an iterable that can be accessed with __getitem__
# process function here will iterate through the above dataset and return:
# 1) a zip file containing the data
# 2) a manifest tsv file with paths to data location, target text etc., 1 utterance per line
# 3) a config yaml file containing data root, augmentation, path to vocab model etc.
# 4) sentencepiece vocab model if specified

# TO DO:
# waveform volume norm - 1) whether to do or not 2) to carry out here or in augmentations code
# test on covost
# test on librispeech


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


def process(args, SPLITS, root, cur_root, audio_root):
    """Process dataset to extract data and zip, generate manifest and config files"""
    extract_features(args, SPLITS, root, cur_root, audio_root)
    train_text = build_manifest(args, SPLITS, root, cur_root, audio_root, create_zipfile=True)
    if args.learn_vocab:
        spm_filename_prefix = learn_vocab(args, train_text)
        generate_config(args, spm_filename_prefix)
    else:
        generate_config(args)
    shutil.rmtree(audio_root) # Clean up


def generate_manifest(args, SPLITS, root, cur_root, audio_root):
    """Generate manifest and config files without (re)extracting data"""
    train_text = build_manifest(args, SPLITS, root, cur_root, audio_root, create_zipfile=False)
    if args.learn_vocab:
        spm_filename_prefix = learn_vocab(args, train_text)
        generate_config(args, spm_filename_prefix)
    else:
        generate_config(args)


def extract_features(args, SPLITS, root, cur_root, audio_root):
    for split in SPLITS:
        # Init specified dataset with split
        print(f"Fetching split {split}...")
        if args.dataset == "mustc":
            dataset = MUSTC(root.as_posix(), args.lang, split)
        elif args.dataset == "covost":
            dataset = CoVoST(root.as_posix(), split, "en", args.lang)
        elif args.dataset == "europarl":
            dataset = EUROPARL(root.as_posix(), args.lang, split)
        # Extract audio/features, save as numpy array
        if args.use_audio_input:
            print(f"Extracting audio into {audio_root}")
            for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                _wavform, _ = convert_waveform(
                    waveform, sample_rate, to_mono=True,
                    to_sample_rate=args.sr
                )
                sf.write(
                    (audio_root / f"{utt_id}.flac").as_posix(),
                    _wavform.T.numpy(), args.sr
                )
        else:
            print(f"Extracting log mel filter bank features into {audio_root}")
            gcmvn_feature_list = []
            if split == 'train' and args.cmvn_type == "global":
                print("And estimating cepstral mean and variance stats...")

            for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                features = extract_fbank_features(
                    waveform, sample_rate, audio_root / f"{utt_id}.npy"
                )
                if split == 'train' and args.cmvn_type == "global":
                    if len(gcmvn_feature_list) < args.gcmvn_max_num:
                        gcmvn_feature_list.append(features)

            if split == 'train' and args.cmvn_type == "global":
                # Estimate and save cmv
                stats = cal_gcmvn_stats(gcmvn_feature_list)
                with open(cur_root / "gcmvn.npz", "wb") as f:
                    np.savez(f, mean=stats["mean"], std=stats["std"])


def build_manifest(args, SPLITS, root, cur_root, audio_root, create_zipfile=False):
    if args.use_audio_input:
        #assuming frame length 10ms, convert frames to samples
        args.min_n_frames = args.min_n_frames*args.sr/100
        args.max_n_frames = args.max_n_frames*args.sr/100

    # Pack features into ZIP
    zip_path = cur_root / f"{audio_root.name}.zip"
    if create_zipfile:
        print("ZIPing audios/features...")
        create_zip(audio_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(
        zip_path,
        is_audio=args.use_audio_input,
    )
    print(audio_paths)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    for split in SPLITS:
        is_train_split = split.startswith("train")
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        if args.dataset == "mustc":
            dataset = MUSTC(root.as_posix(), args.lang, split)
        elif args.dataset == "covost":
            dataset = CoVoST(root.as_posix(), split, "en", args.lang)
        elif args.dataset == "europarl":
            dataset = EUROPARL(root.as_posix(), args.lang, split)

        for _, _, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["audio"].append(audio_paths[utt_id])
            manifest["n_frames"].append(audio_lengths[utt_id])
            manifest["tgt_text"].append(
                src_utt if args.task == "asr" else tgt_utt
            )
            manifest["speaker"].append(speaker_id)
        if is_train_split:
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=True,
                                min_n_frames=args.min_n_frames, max_n_frames=args.max_n_frames)
        save_df_to_tsv(df, cur_root / f"{split}_{args.task}.tsv")
    return train_text


def learn_vocab(args, train_text):
    #train sentencepiece model
    root = Path(args.data_root).absolute()
    cur_root = root / f"en-{args.lang}"
    v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{args.task}"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            cur_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
            )
    return spm_filename_prefix


def generate_config(args, spm_filename_prefix=""):
    # Generate config YAML
    root = Path(args.data_root).absolute()
    cur_root = root / f"en-{args.lang}"
    if args.use_audio_input:
        gen_config_yaml(
            cur_root,
            spm_filename=spm_filename_prefix + ".model",
            yaml_filename=f"config_{args.task}.yaml",
            specaugment_policy=None,
            extra={"use_audio_input": True}
        )
    else:
        gen_config_yaml(
            cur_root,
            spm_filename=spm_filename_prefix + ".model",
            yaml_filename=f"config_{args.task}.yaml",
            specaugment_policy="lb",
            cmvn_type=args.cmvn_type,
            gcmvn_path=(
                cur_root / "gcmvn.npz" if args.cmvn_type == "global"
                else None
            ),
        )


def main():
    parser = argparse.ArgumentParser()
    #task args
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--dataset", type=str, choices=["mustc", "covost", "europarl"])
    parser.add_argument("--task","-t", type=str, choices=["asr", "st"])
    parser.add_argument("--lang", "-l", default=None, type=str, help="specify a target language")
    parser.add_argument("--use-audio-input", action="store_true", help="if true, process raw audio instead of filterbank features")
    parser.add_argument("--learn-vocab", action="store_true", help="if true, train sentencepiece vocab model")
    parser.add_argument("--only-manifest", action="store_true", help="if true, only generate manifest file without extracting data")
    #audio args
    parser.add_argument("--max-n-frames", default=3000, type=int, help="max no of frames in an input (1 frame default 10ms)")
    parser.add_argument("--min_n_frames", default=5, type=int, help="min no of frames in an input (1 frame default 10ms)")
    parser.add_argument("--sr", default=16000, type=int, help="sample rate for audio input, will resample to this if different from native rate")
    #vocab args
    parser.add_argument("--vocab-type", default="unigram", type=str, choices=["bpe", "unigram", "char"]),
    parser.add_argument("--vocab-size", default=8000, type=int)
    #parser.add_argument("--input-manifest", "-m", default=None, type=str, help="path to manifest for building vocab model")
    #augmentation args
    parser.add_argument("--cmvn-type", default="utterance", choices=["global", "utterance"],
                        help="The type of cepstral mean and variance normalization")
    parser.add_argument("--gcmvn-max-num", default=150000, type=int,
                        help="Maximum number of sentences to use to estimate global mean and variance")

    args = parser.parse_args()

    # Set up data dirs
    root = Path(args.data_root).absolute()

    if args.dataset == "mustc":
        SPLITS = MUSTC.SPLITS
        cur_root = root / f"en-{args.lang}"
    elif args.dataset == "covost":
        SPLITS = CoVoST.SPLITS
        cur_root = root / f"en-{args.lang}"
    elif args.dataset == "europarl":
        SPLITS = EUROPARL.SPLITS
        cur_root = root / "en" / f"{args.lang}"
    assert cur_root.is_dir(), f"{cur_root.as_posix()} does not exist. Skipped."
    audio_root = cur_root / ("flac" if args.use_audio_input else "fbank80")
    print("audio_root", audio_root)
    audio_root.mkdir(parents=True, exist_ok=True)
    assert audio_root.exists()

    if args.only_manifest:
        generate_manifest(args, SPLITS, root, cur_root, audio_root)
    else:
        process(args, SPLITS, root, cur_root, audio_root)


if __name__ == "__main__":
    main()
