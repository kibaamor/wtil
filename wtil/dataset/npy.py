import glob
import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


class NpyFileDataset(Dataset):
    def __init__(self, filename: str, seq_len: int):
        self.filename = filename
        self.seq_len = seq_len

        data = np.load(filename, allow_pickle=True)
        obs_data = data.item()["obs"]
        act_data = data.item()["act"]

        self.obs_keys = list(obs_data.keys())
        self.obs_values = TensorDataset(*[torch.from_numpy(obs) for obs in obs_data.values()])
        self.act_keys = list(act_data.keys())
        self.act_values = TensorDataset(*[torch.from_numpy(act) for act in act_data.values()])

        self.batch_len = len(obs_data[self.obs_keys[0]])

    def __len__(self):
        return max(self.batch_len - self.seq_len, 0)

    def __getitem__(self, index: np.ndarray) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if index.ndim == 0:
            obs_value_batch = self.obs_values[index : index + self.seq_len]
            act_value_batch = self.act_values[index : index + self.seq_len]
        else:
            assert index.ndim == 1
            obs_value_list = [self.obs_values[i : i + self.seq_len] for i in index]
            act_value_list = [self.act_values[i : i + self.seq_len] for i in index]

            obs_value_batch = torch.stack([v[0] for v in obs_value_list])
            act_value_batch = torch.stack([v[0] for v in act_value_list])

        obs_batch = dict(zip(self.obs_keys, obs_value_batch))
        act_batch = dict(zip(self.act_keys, act_value_batch))
        return obs_batch, act_batch

    def to_dataloader(self, batch_size: int) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=True)

    @staticmethod
    def make_dataloader(filename: str, seq_len: int, batch_size: int):
        dataset = NpyFileDataset(filename, seq_len)
        return dataset.to_dataloader(batch_size=batch_size)


def by_pass(x: Any) -> Any:
    return x


class NpyDirDataset(Dataset):
    def __init__(self, path_glob: str, seq_len: int):
        self.file_list = glob.glob(path_glob)
        if len(self.file_list) == 0:
            logging.critical(f"Invalid path_glob: {path_glob}")
        self.seq_len = seq_len

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index: np.ndarray) -> NpyFileDataset:
        return NpyFileDataset(self.file_list[index], self.seq_len)

    def to_dataloader(self, num_workers: int = 1) -> DataLoader:
        return DataLoader(self, shuffle=True, num_workers=num_workers, collate_fn=by_pass)

    @staticmethod
    def make_dataloader(path_glob: int, seq_len: int, num_workers: int = 1):
        dataset = NpyDirDataset(path_glob, seq_len)
        return dataset.to_dataloader(num_workers=num_workers)
