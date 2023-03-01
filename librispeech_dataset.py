from typing import Tuple

from torch import Tensor
from torchaudio.datasets.utils import _load_waveform
from torchaudio.datasets.librispeech import LIBRISPEECH

class LIBRISPEECHST(LIBRISPEECH):
    """
    Create a Dataset for LibriSpeech. Each item is a tuple of the form:
    waveform, sample_rate, utterance, utterance, speaker_id,
    utterance_id
    """

    SPLITS = [
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    ]

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Transcript
            str:
                Transcript
            str:
                Speaker ID
            str:
                Utterance ID
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])

        transcript = metadata[2]
        speaker_id = str(metadata[3])
        utt_id = f"{speaker_id}-{str(metadata[4])}-{metadata[5]:04d}"
        return (waveform,) + (metadata[1], transcript, transcript, speaker_id, utt_id)
