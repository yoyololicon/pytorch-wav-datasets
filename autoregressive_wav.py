from .random_wav import RandomWAVDataset
from torchaudio.transforms import MuLawEncoding


class AutoregressiveWAVDataset(RandomWAVDataset):
    """
    Wave-file-only Dataset.
    """

    def __init__(self,
                 data_dir: str,
                 size: int,
                 segment: int,
                 deterministic: bool = True):
        super().__init__(data_dir, size, segment + 1, deterministic)

    def __getitem__(self, index):
        x = super().__getitem__(index)
        return x[:-1], x[1:]
