#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
import shutil
from typing import Optional, Tuple

import pandas as pd
import torchaudio
from examples.speech_to_text.data_utils import load_df_from_tsv

from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive


log = logging.getLogger(__name__)



class CoVoST(Dataset):
    """Create a Dataset for CoVoST (https://github.com/facebookresearch/covost).

    Args:
        root (str): root path to the dataset and generated manifests/features
        source_language (str): source (audio) language
        target_language (str, optional): target (text) language,
        None for no translation (default: None)
        version (int, optional): CoVoST version. (default: 2)
        download (bool, optional): Whether to download the dataset if it is not
        found at root path. (default: ``False``).
    """

    COVOST_URL_TEMPLATE = (
        "https://dl.fbaipublicfiles.com/covost/"
        "covost_v2.{src_lang}_{tgt_lang}.tsv.tar.gz"
    )

    VERSIONS = {2}
    SPLITS = ["train", "dev", "test"]

    XX_EN_LANGUAGES = {
        1: ["fr", "de", "nl", "ru", "es", "it", "tr", "fa", "sv-SE", "mn", "zh-CN"],
        2: [
            "fr",
            "de",
            "es",
            "ca",
            "it",
            "ru",
            "zh-CN",
            "pt",
            "fa",
            "et",
            "mn",
            "nl",
            "tr",
            "ar",
            "sv-SE",
            "lv",
            "sl",
            "ta",
            "ja",
            "id",
            "cy",
        ],
    }
    EN_XX_LANGUAGES = {
        1: [],
        2: [
            "de",
            "tr",
            "fa",
            "sv-SE",
            "mn",
            "zh-CN",
            "cy",
            "ca",
            "sl",
            "et",
            "id",
            "ar",
            "ta",
            "lv",
            "ja",
        ],
    }

    def __init__(
        self,
        root: str,
        split: str,
        source_language: str,
        target_language: Optional[str] = None,
        version: int = 2,
    ) -> None:
        assert version in self.VERSIONS and split in self.SPLITS
        assert source_language is not None
        self.no_translation = target_language is None
        if not self.no_translation:
            assert "en" in {source_language, target_language}
            if source_language == "en":
                assert target_language in self.EN_XX_LANGUAGES[version]
            else:
                assert source_language in self.XX_EN_LANGUAGES[version]
        else:
            # Hack here so that we can get "split" column from CoVoST TSV.
            # Note that we use CoVoST train split for ASR which is an extension
            # to Common Voice train split.
            target_language = "de" if source_language == "en" else "en"

        self.root: Path = Path(root)

        cv_tsv_path = self.root / "validated.tsv"
        assert cv_tsv_path.is_file()

        covost_url = self.COVOST_URL_TEMPLATE.format(
            src_lang=source_language, tgt_lang=target_language
        )
        covost_archive = self.root / Path(covost_url).name
        if not covost_archive.is_file():
            download_url(covost_url, self.root.as_posix(), hash_value=None)
        extract_archive(covost_archive.as_posix())

        cv_tsv = load_df_from_tsv(cv_tsv_path)
        covost_tsv = load_df_from_tsv(
            self.root / Path(covost_url).name.replace(".tar.gz", "")
        )
        df = pd.merge(
            left=cv_tsv[["path", "sentence", "client_id"]],
            right=covost_tsv[["path", "translation", "split"]],
            how="inner",
            on="path",
        )
        if split == "train":
            df = df[(df["split"] == split) | (df["split"] == f"{split}_covost")]
        else:
            df = df[df["split"] == split]
        data = df.to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]
        self.data = []
        for e in data:
            try:
                path = self.root / "clips" / e["path"]
                _ = torchaudio.info(path.as_posix())
                self.data.append(e)
            except RuntimeError:
                pass

    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, str, Optional[str], str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, sentence, translation, speaker_id,
            sample_id)``
        """
        data = self.data[n]
        path = self.root / "clips" / data["path"]
        waveform, sample_rate = torchaudio.load(path)
        sentence = data["sentence"]
        translation = None if self.no_translation else data["translation"]
        speaker_id = data["client_id"]
        _id = data["path"].replace(".mp3", "")
        return waveform, sample_rate, sentence, translation, speaker_id, _id

    def __len__(self) -> int:
        return len(self.data)
