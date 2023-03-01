#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path
import shutil
from itertools import groupby
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from fairseq.data.audio.audio_utils import get_waveform, convert_waveform


log = logging.getLogger(__name__)



class EUROPARL(Dataset):
    """
    Create a Dataset for Europarl-ST. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    https://www.mllp.upv.es/europarl-st/
    """

    SPLITS = ["dev", "test", "train", "train-noisy"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pl", "pt", "ro"]

    def __init__(self, root: str, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        wav_root = Path(root) / "en" / "audios" / "wav"
        txt_root = Path(root) / "en" / f"{lang}" / split
        if wav_root.exists() and any(wav_root.iterdir()):
            assert wav_root.is_dir() and txt_root.is_dir()
        else:
            #convert m4a audio files to wav
            wav_root.mkdir(parents=True,exist_ok=True)
            try:
                from pydub import AudioSegment
                audio_paths = list(wav_root.parent.glob('*.m4a'))
                for file in tqdm(audio_paths):
                    track = AudioSegment.from_file(file, format='m4a')
                    wav_basename = file.with_suffix('.wav').name
                    wav_path = wav_root / wav_basename
                    file_handle = track.export(wav_path, format='wav')
                assert len(audio_paths) == len(list(wav_root.glob('*.wav')))
            except:
                print("Please install pydub to handle wav file conversion")
        # Load audio segments
        with open(txt_root / "segments.lst", encoding="utf-8") as f:
            segments_list = [line.split() for line in f]
        segments = []
        for seg in segments_list:
            segment_a = {}
            segment_a["audio"] = seg[0]
            segment_a["offset"] = float(seg[1])
            segment_a["duration"] = float(seg[2]) - float(seg[1])
            segments.append(segment_a)
        # Load source and target utterances
        for _lang in ["en", lang]:
            with open(txt_root / f"segments.{_lang}", encoding="utf-8") as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Load speakers id
        with open(txt_root / "speakers.lst", encoding="utf-8") as f:
            speakers = [r.strip() for r in f]
        with open(txt_root / "speeches.lst", encoding="utf-8") as f:
            speeches = [r.strip() for r in f]
        speeches_and_speakers = list(zip(speeches, speakers))
        # Gather info
        self.data = []
        for wav_filename, datapoint in groupby(segments, lambda x: x['audio']):
            wav_path = wav_root / f"{wav_filename}.wav"
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(datapoint, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(segment["offset"] * sample_rate)
                n_frames = int(segment["duration"] * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                for entry in speeches_and_speakers:
                    if segment['audio'] == entry[0]:
                        speaker_id = entry[1]
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
                        speaker_id,
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


def load_textfile(filename):
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
        #lines = [l for l in (line.strip() for line in f) if l] #skips empty line
    return lines

