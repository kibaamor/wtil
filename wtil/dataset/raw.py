import glob
import logging
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

ProcessFn = Callable[[str], Sequence[torch.Tensor]]


class RawFileDataset(Dataset):
    def __init__(self, filename: str, seq_len: int, process_fn: ProcessFn):
        self.filename = filename
        self.seq_len = seq_len
        self.dataset = TensorDataset(*process_fn(filename))

    def __len__(self):
        return max(len(self.dataset) - self.seq_len, 0)

    def __getitem__(self, index: np.ndarray):
        if index.ndim == 0:
            return self.dataset[index : index + self.seq_len]

        assert index.ndim == 1
        batch = [self.dataset[i : i + self.seq_len] for i in index]
        features = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        return features, labels

    def to_dataloader(self, batch_size: int) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=True)

    @staticmethod
    def make_dataloader(filename: str, seq_len: int, batch_size: int, process_fn: ProcessFn):
        dataset = RawFileDataset(filename, seq_len, process_fn)
        return dataset.to_dataloader(batch_size=batch_size)


def by_pass(x: Any) -> Any:
    return x


class RawDirDataset(Dataset):
    def __init__(self, path_glob: str, seq_len: int, process_fn: ProcessFn):
        self.file_list = glob.glob(path_glob)
        if len(self.file_list) == 0:
            logging.critical(f"Invalid path_glob: {path_glob}")
        self.seq_len = seq_len
        self.process_fn = process_fn

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index: np.ndarray) -> RawFileDataset:
        return RawFileDataset(self.file_list[index], self.seq_len, self.process_fn)

    def to_dataloader(self, num_workers: int = 1) -> DataLoader:
        return DataLoader(self, shuffle=True, num_workers=num_workers, collate_fn=by_pass)

    @staticmethod
    def make_dataloader(path_glob: int, seq_len: int, process_fn: ProcessFn, num_workers: int = 1):
        dataset = RawDirDataset(path_glob, seq_len, process_fn)
        return dataset.to_dataloader(num_workers=num_workers)
