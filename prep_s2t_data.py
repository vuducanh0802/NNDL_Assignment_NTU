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
import json
from tqdm import tqdm

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

from fairseq.data.audio.audio_utils import get_waveform, convert_waveform
from examples.joint_embedding_pretraining.dataprep.mustc_dataset import MUSTC
from examples.joint_embedding_pretraining.dataprep.covost_dataset import CoVoST
from examples.joint_embedding_pretraining.dataprep.europarl_dataset import EUROPARL
from examples.joint_embedding_pretraining.dataprep.iwsltcorpus_dataset import IWSLTCORPUS
from examples.joint_embedding_pretraining.dataprep.librispeech_dataset import LIBRISPEECHST


# general processing script to handle ST/ASR datasets.
# Each dataset should be written as a separate torch dataclass (and imported into this script) that extracts:
# waveform, sample_rate, source sentence, target sentence, speaker_id, utterance id
# and return an iterable that can be accessed with __getitem__
# process function here will iterate through the above dataset and return:
# 1) a zip file containing the data
# 2) a manifest tsv file with paths to data location, target text etc., 1 utterance per line
# 3) a config yaml file containing data root, augmentation, path to vocab model etc.
# 4) sentencepiece vocab model if specified

# WHAT TO DO WHEN ADDING NEW DATASET
# 1) write a new dataset class as per above
# 2) import new class into this script
# 3) extend get_dataset_split function with class init
# 4) define cur_root in main() according to dataset folder structure
# 5) add dataset in parser arg --dataset

# TO DO:
# waveform volume norm - 1) whether to do or not 2) to carry out here or in augmentations code
# test on covost



log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


def get_dataset_split(split, dataset_name, root, *args, **kwargs):
    """extend this function to initialize new dataset class"""
    if dataset_name == "mustc":
        dataset = MUSTC(root.as_posix(), kwargs['lang'], split)
    elif dataset_name == "covost":
        dataset = CoVoST(root.as_posix(), split, "en", kwargs['lang'])
    elif dataset_name == "europarl":
        dataset = EUROPARL(root.as_posix(), kwargs['lang'], split)
    elif dataset_name == "iwsltcorpus":
        dataset = IWSLTCORPUS(root.as_posix(), kwargs['lang'], split)
    elif dataset_name == 'librispeech':
        dataset = LIBRISPEECHST(root.as_posix(), split)
    return dataset


def process(args, SPLITS, root, cur_root, audio_root):
    """Process dataset to extract data and zip, generate manifest and config files"""
    for split in SPLITS:
        print(f"Fetching split {split}...")
        # Init specified dataset with split
        dataset = get_dataset_split(split, args.dataset, root, lang=args.lang)
        extract_features(args, split, dataset, cur_root, audio_root)

    train_text = []
    audio_paths, audio_lengths = create_zip_folder(args, cur_root, audio_root, create_zipfile=True)
    for split in SPLITS:
        print(f"Fetching split {split}...")
        # Init specified dataset with split
        dataset = get_dataset_split(split, args.dataset, root, lang=args.lang)
        train_text_p = build_manifest(args, split, dataset, cur_root, audio_paths, audio_lengths)
        if len(train_text_p) > 0:
            train_text.extend(train_text_p)

    if args.learn_vocab:
        spm_filename_prefix = learn_vocab(args, train_text)
        generate_config(args, cur_root, spm_filename_prefix)
    else:
        generate_config(args, cur_root)
    shutil.rmtree(audio_root) # Clean up


def generate_manifest(args, SPLITS, root, cur_root, audio_root):
    """Generate manifest and config files without (re)extracting data"""
    train_text = []
    audio_paths, audio_lengths = create_zip_folder(args, cur_root, audio_root, create_zipfile=False)
    for split in SPLITS:
        print(f"Fetching split {split}...")
        # Init specified dataset with split
        dataset = get_dataset_split(split, args.dataset, root, lang=args.lang)
        train_text_p = build_manifest(args, split, dataset, cur_root, audio_paths, audio_lengths)
        if len(train_text_p) > 0:
            train_text.extend(train_text_p)

    if args.learn_vocab:
        spm_filename_prefix = learn_vocab(args, train_text)
        generate_config(args, cur_root, spm_filename_prefix)
    else:
        generate_config(args, cur_root)


def extract_features(args, split, dataset, cur_root, audio_root):
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


def create_zip_folder(args, cur_root, audio_root, create_zipfile=False):
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
    return audio_paths, audio_lengths


def build_manifest(args, split, dataset, cur_root, audio_paths, audio_lengths):
    if args.use_audio_input:
        #assuming frame length 10ms, convert frames to samples
        min_n_frames = args.min_n_frames*args.sr/100
        max_n_frames = args.max_n_frames*args.sr/100

    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    src_utt_mt = []
    tgt_utt_mt = []
    is_train_split = split.startswith("train")
    manifest = {c: [] for c in MANIFEST_COLUMNS}

    for _, _, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
        manifest["id"].append(utt_id)
        manifest["audio"].append(audio_paths[utt_id])
        manifest["n_frames"].append(audio_lengths[utt_id])
        manifest["tgt_text"].append(
            src_utt if args.task == "asr" else tgt_utt
        )
        manifest["speaker"].append(speaker_id)
        if args.get_mt_pair:
            src_utt_mt.append(src_utt)
            tgt_utt_mt.append(tgt_utt)
    if is_train_split:
        train_text.extend(manifest["tgt_text"])
    df = pd.DataFrame.from_dict(manifest)
    df = filter_manifest_df(df, is_train_split=is_train_split, #this will only filter max/min length on train set,
                            min_n_frames=min_n_frames, max_n_frames=max_n_frames) # change if filtering on valid/test as well
    save_df_to_tsv(df, cur_root / f"{split}_{args.task}.tsv")

    if args.get_mt_pair:
        assert len(src_utt_mt) == len(tgt_utt_mt)
        write_lines_to_file(src_utt_mt, cur_root/"MT", f"{split}", ".en")
        write_lines_to_file(tgt_utt_mt, cur_root/"MT", f"{split}", f".{args.lang}")
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


def generate_config(args, cur_root, spm_filename_prefix=""):
    # Generate config YAML
    if args.use_audio_input:
        trans_cfg = {}

        if args.wave_aug is not None:  # training feature pipeline
            trans_cfg["_train"] = args.wave_aug.split(",")

        if args.use_feat_transform:
            trans_cfg["*"] = ["fbank", f"{args.cmvn_type}_cmvn"]  # common features
            trans_cfg["_train"] += ["fbank", f"{args.cmvn_type}_cmvn"]  # training features

        if args.timefreq_aug is not None:
            trans_cfg["_train"] += args.timefreq_aug.split(",")

        feat_pipe_cfg = {"use_audio_input": True}
        if trans_cfg:
            feat_pipe_cfg.update({"transforms": trans_cfg})
            if args.aug_args is not None:
                feat_pipe_cfg.update(args.aug_args)

        gen_config_yaml(
            cur_root,
            spm_filename=spm_filename_prefix + ".model",
            yaml_filename=f"config_{args.task}.yaml",
            specaugment_policy=None,
            extra=feat_pipe_cfg
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


def write_lines_to_file(lines,outdir,filename,outname):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = f"{outdir}/{filename}{outname}"
    print(f"saving: {outfile}")
    open(outfile, 'w', encoding='utf-8').writelines(l.strip()+'\n' for l in lines)


def main():
    parser = argparse.ArgumentParser()
    #task args
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--dataset", type=str, choices=["mustc", "covost", "europarl", "iwsltcorpus", "librispeech"])
    parser.add_argument("--task","-t", type=str, choices=["asr", "st"])
    parser.add_argument("--lang", "-l", default=None, type=str, help="specify a target language")
    parser.add_argument("--use-audio-input", action="store_true", help="if true, process raw audio instead of filterbank features")
    parser.add_argument("--learn-vocab", action="store_true", help="if true, train sentencepiece vocab model")
    parser.add_argument("--only-manifest", action="store_true", help="if true, only generate manifest file without extracting data")
    parser.add_argument("--get-mt-pair", action="store_true", help="if true, generate text files containing aligned src_utt and tgt_utt for MT, \
        works for ST data with both transcriptions and translations present")
    #audio args
    parser.add_argument("--max-n-frames", default=3000, type=int, help="max no of frames in an input (1 frame default 10ms ~ 480000 samples@ 16kHz)")
    parser.add_argument("--min_n_frames", default=5, type=int, help="min no of frames in an input (1 frame default 10ms ~ 800 samples @ 16kHz)")
    parser.add_argument("--sr", default=16000, type=int, help="sample rate for audio input, will resample to this if different from native rate")
    parser.add_argument("--use-feat-transform", action="store_true", help="use fbank transformation on-the-fly (use with --use-audio-input=True)")
    #vocab args
    parser.add_argument("--vocab-type", default="unigram", type=str, choices=["bpe", "unigram", "char"]),
    parser.add_argument("--vocab-size", default=8000, type=int)
    #parser.add_argument("--input-manifest", "-m", default=None, type=str, help="path to manifest for building vocab model")
    #augmentation args
    parser.add_argument("--cmvn-type", default="utterance", choices=["global", "utterance"],
                        help="The type of cepstral mean and variance normalization")
    parser.add_argument("--gcmvn-max-num", default=150000, type=int,
                        help="Maximum number of sentences to use to estimate global mean and variance")
    parser.add_argument("--wave-aug", type=str, default=None, help="waveform feature augmentations")
    parser.add_argument("--timefreq-aug", type=str, default=None, help="feature augmentations in frequency domain")
    parser.add_argument("--aug-args", type=json.loads, default="null", help="override default arguments for augmentations in --wave-aug and --timefreq-aug")

    args = parser.parse_args()

    # Set up data dirs
    root = Path(args.data_root).absolute()

    #set up dataset specific variables here
    if args.dataset == "mustc":
        SPLITS = MUSTC.SPLITS
        cur_root = root / f"en-{args.lang}"
    elif args.dataset == "covost":
        SPLITS = CoVoST.SPLITS
        cur_root = root / f"en-{args.lang}"
    elif args.dataset == "europarl":
        SPLITS = EUROPARL.SPLITS
        cur_root = root / "en" / f"{args.lang}"
    elif args.dataset == "iwsltcorpus":
        SPLITS = IWSLTCORPUS.SPLITS
        cur_root = root
    elif args.dataset == "librispeech":
        SPLITS = LIBRISPEECHST.SPLITS
        cur_root = root

    assert cur_root.is_dir(), f"{cur_root.as_posix()} does not exist. Skipped."
    audio_root = cur_root / ("flac" if args.use_audio_input else "fbank80")
    audio_root.mkdir(parents=True, exist_ok=True)

    if args.only_manifest:
        # Extract audio/features, save as numpy array
        generate_manifest(args, SPLITS, root, cur_root, audio_root)
    else:
        process(args, SPLITS, root, cur_root, audio_root)


if __name__ == "__main__":
    main()
