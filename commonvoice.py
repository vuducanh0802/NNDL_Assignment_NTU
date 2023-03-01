import logging
import os
from pathlib import Path
import shutil
from itertools import groupby
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf

import torch
from torch.utils.data import Dataset

from fairseq.data.audio.audio_utils import get_waveform, convert_waveform


log = logging.getLogger(__name__)


class COMMONVOICEST(Dataset):
	SPLITS = ["train", "dev", "tst-COMMON", "tst-HE"]
	LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru", "zh", "ja"]

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
		
		
		
