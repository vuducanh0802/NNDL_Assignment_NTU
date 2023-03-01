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


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["train", "dev", "tst-COMMON", "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru", "zh", "jp"]

    def __init__(self, root: str, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = Path(root) / f"en-{lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the MuST-C YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", lang]:
            with open(txt_root / f"{split}.{_lang}", encoding="utf-8") as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(
            self, n: int
    ) -> Tuple[torch.Tensor, int, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, \
            utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute()
    max_n_frames = 3000
    min_n_frames = 5

    for lang in MUSTC.LANGUAGES:
        cur_root = root / f"en-{lang}"
        if not cur_root.is_dir():
            print(f"{cur_root.as_posix()} does not exist. Skipped.")
            continue
        # Extract features
        audio_root = cur_root / ("flac" if args.use_audio_input else "fbank80")
        audio_root.mkdir(exist_ok=True)

        for split in MUSTC.SPLITS:
            print(f"Fetching split {split}...")
            dataset = MUSTC(root.as_posix(), lang, split)
            if args.use_audio_input:
                print("Converting audios...")
                tgt_sample_rate = 16_000
                min_n_frames = min_n_frames*tgt_sample_rate/100
                max_n_frames = max_n_frames*tgt_sample_rate/100
                for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                    _wavform, _ = convert_waveform(
                        waveform, sample_rate, to_mono=True,
                        to_sample_rate=tgt_sample_rate
                    )
                    sf.write(
                        (audio_root / f"{utt_id}.flac").as_posix(),
                        _wavform.T.numpy(), tgt_sample_rate
                    )
            else:
                print("Extracting log mel filter bank features...")
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

        # Pack features into ZIP
        zip_path = cur_root / f"{audio_root.name}.zip"
        print("ZIPing audios/features...")
        create_zip(audio_root, zip_path)
        print("Fetching ZIP manifest...")
        audio_paths, audio_lengths = get_zip_manifest(
            zip_path,
            is_audio=args.use_audio_input,
        )
        # Generate TSV manifest
        print("Generating manifest...")
        train_text = []
        for split in MUSTC.SPLITS:
            is_train_split = split.startswith("train")
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            dataset = MUSTC(args.data_root, lang, split)
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
            df = filter_manifest_df(df, is_train_split=is_train_split,
                                    min_n_frames=min_n_frames, max_n_frames=max_n_frames)
            save_df_to_tsv(df, cur_root / f"{split}_{args.task}.tsv")
        # Generate vocab
        if not args.no_vocab:
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
        else:
            spm_filename_prefix = f""
        # Generate config YAML
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
        # Clean up
        shutil.rmtree(audio_root)


def process_joint(args):
    cur_root = Path(args.data_root)
    assert all(
        (cur_root / f"en-{lang}").is_dir() for lang in MUSTC.LANGUAGES
    ), "do not have downloaded data available for all 8 languages"
    # Generate vocab
    vocab_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size_str}_{args.task}"
    with NamedTemporaryFile(mode="w") as f:
        for lang in MUSTC.LANGUAGES:
            tsv_path = cur_root / f"en-{lang}" / f"train_{args.task}.tsv"
            df = load_df_from_tsv(tsv_path)
            for t in df["tgt_text"]:
                f.write(t + "\n")
        special_symbols = None
        if args.task == 'st':
            special_symbols = [f'<lang:{lang}>' for lang in MUSTC.LANGUAGES]
        gen_vocab(
            Path(f.name),
            cur_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
            special_symbols=special_symbols
        )
    # Generate config YAML
    gen_config_yaml(
        cur_root,
        spm_filename=spm_filename_prefix + ".model",
        yaml_filename=f"config_{args.task}.yaml",
        specaugment_policy="ld",
        prepend_tgt_lang_tag=(args.task == "st"),
    )
    # Make symbolic links to manifests
    for lang in MUSTC.LANGUAGES:
        for split in MUSTC.SPLITS:
            src_path = cur_root / f"en-{lang}" / f"{split}_{args.task}.tsv"
            desc_path = cur_root / f"{split}_{lang}_{args.task}.tsv"
            if not desc_path.is_symlink():
                os.symlink(src_path, desc_path)


def rebuild_manifest(args):
    # only rebuild the manifest files without extracting the filter banks and generating vocab
    # use for example when want to change the filenames in the manifest files (like when shifting dataset around)
    root = Path(args.data_root).absolute()
    max_n_frames = 3000
    min_n_frames = 5
    tgt_sample_rate = 16_000
    if args.use_audio_input:
        min_n_frames = min_n_frames*tgt_sample_rate/100
        max_n_frames = max_n_frames*tgt_sample_rate/100

    for lang in MUSTC.LANGUAGES:
        cur_root = root / f"en-{lang}"
        if args.lang is not None: #only run specific language if specified
            if lang != args.lang:
                continue
        if not cur_root.is_dir():
            print(f"{cur_root.as_posix()} does not exist. Skipped.")
            continue
        audio_root = cur_root / ("flac" if args.use_audio_input else "fbank80")

        # Pack features into ZIP
        zip_path = cur_root / f"{audio_root.name}.zip"
        print("Fetching ZIP manifest...")
        audio_paths, audio_lengths = get_zip_manifest(
            zip_path,
            is_audio=args.use_audio_input,
        )
        # Generate TSV manifest
        print("Generating manifest...")

        train_text = []
        for split in MUSTC.SPLITS:
            is_train_split = split.startswith("train")
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            dataset = MUSTC(args.data_root, lang, split)
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
            df = filter_manifest_df(df, is_train_split=is_train_split,
                                    min_n_frames=min_n_frames, max_n_frames=max_n_frames)
            save_df_to_tsv(df, cur_root / f"{split}_{args.task}.tsv")

        if not args.no_vocab:
            build_vocab(args, train_text)


def build_vocab(args, train_text=None):
    root = Path(args.data_root).absolute()
    cur_root = root / f"en-{args.lang}"
    v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{args.task}"

    if train_text is None:
        train_text = load_csvfile(args.input_manifest, 'tgt_text')

    with NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False) as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            cur_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
        )
        f.close()
        os.unlink(f.name)


def load_csvfile(filename, column=None, delimiter='\t'):
    """opens text file and return sentences in a list. Specify a column header if required."""
    import csv
    with open(filename, encoding="utf-8") as f:
        reader = csv.DictReader(
                    f,
                    delimiter=delimiter,
                    quotechar='"', #None,
                    doublequote=False,
                    lineterminator="\n",
                    #quoting=csv.QUOTE_NONE,
        )
        if column is not None:
            lines = [row[column] for row in reader]
        else:
            lines = [row for row in reader]
    return lines

def write_lines_to_file(lines,outdir,filename,outname):
    filter_out_empty_lines = lines #[l for l in lines if len(l)>0]
    print(f"outlines:{len(filter_out_empty_lines)}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = f"{outdir}/{filename}{outname}"
    print(f"saving: {outfile}")
    open(outfile, 'w', encoding='utf-8').writelines(l.strip()+'\n' for l in filter_out_empty_lines)

def main():
    parser = argparse.ArgumentParser()
    #task args
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--task", type=str, choices=["asr", "st"])
    parser.add_argument("--joint", action="store_true", help="")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--lang", "-l", default=None, type=str, help="Specify a MuST-C target language")
    parser.add_argument("--use-audio-input", action="store_true")
    parser.add_argument("--no-vocab", action="store_true", help="Skip training sentencepiece vocab model")
    #vocab args
    parser.add_argument("--vocab-type", default="unigram", type=str,
                        choices=["bpe", "unigram", "char"]),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--input-manifest", "-m", default=None, type=str,
                        help="Path to manifest for building vocab model")
    #augmentation args
    parser.add_argument("--cmvn-type", default="utterance", choices=["global", "utterance"],
                        help="The type of cepstral mean and variance normalization")
    parser.add_argument("--gcmvn-max-num", default=150000, type=int,
                        help="Maximum number of sentences to use to estimate global mean and variance")

    args = parser.parse_args()

    if args.joint:
        process_joint(args)
    elif args.rebuild_manifest:
        rebuild_manifest(args)
    elif args.input_manifest is not None:
        build_vocab(args)
    else:
        process(args)


if __name__ == "__main__":
    main()
