import glob
import logging
from typing import Callable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

ProcessFn = Callable[[str], Sequence[torch.Tensor]]


class FileDataset(Dataset):
    def __init__(self, filename: str, seq_len: int, process_fn: ProcessFn):
        self.filename = filename
        self.seq_len = seq_len
        self.dataset = TensorDataset(*process_fn(filename))

    def __len__(self):
        return max(len(self.dataset) - self.seq_len, 0)

    def __getitem__(self, index):
        return self.dataset[index : index + self.seq_len]

    @staticmethod
    def make_dataloader(filename, seq_len, batch_size, process_fn: ProcessFn):
        dataset = FileDataset(filename, seq_len, process_fn)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        dataloader.filename = filename
        return dataloader


def by_pass(x):
    return x


class DirDataset(Dataset):
    def __init__(self, path_glob: str, seq_len: int, batch_size: int, process_fn: ProcessFn):
        self.file_list = glob.glob(path_glob)
        if len(self.file_list) == 0:
            logging.critical(f"Invalid path_glob: {path_glob}")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.process_fn = process_fn

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return FileDataset.make_dataloader(self.file_list[index], self.seq_len, self.batch_size, self.process_fn)

    @staticmethod
    def make_dataloader(path_glob, seq_len, batch_size, process_fn: ProcessFn, num_workers):

        dataset = DirDataset(path_glob, seq_len, batch_size, process_fn)
        dataloader = DataLoader(dataset, shuffle=True, num_workers=num_workers, collate_fn=by_pass)
        return dataloader
