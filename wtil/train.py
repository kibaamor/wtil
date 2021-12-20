#!/usr/bin/env python3
import logging
import os
import pathlib
from typing import Any, Callable

import torch
import tqdm
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

from wtil.loss import calc_loss
from wtil.model import Model
from wtil.preprocess.act import ACT_N
from wtil.preprocess.obs import OBS_N
from wtil.preprocess.process import gen_process_fn
from wtil.utils.dataset import DirDataset, FileDataset
from wtil.utils.logger import config_logger

USE_CUDA = True  # torch.cuda.is_available()
USE_DEVICE = "cuda" if USE_CUDA else "cpu"


def preprocess_data(
    data_dir=pathlib.Path(__file__).parent / "data",
    seq_len=8,
    device=USE_DEVICE,
):
    return DirDataset(
        path_glob=f"{data_dir}/*.txt",
        seq_len=seq_len,
        process_fn=gen_process_fn(device=device),
    )


def train(
    global_step: int,
    writer: SummaryWriter,
    model: nn.Module,
    criterion: Callable[[Any, Any], dict],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    num_epochs: int,
    filename: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
) -> int:

    with tqdm.tqdm(total=num_epochs, desc=f"{filename}") as t:
        for _ in range(num_epochs):

            model.train(True)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for features, labels in train_dataloader:
                optimizer.zero_grad()
                loss = sum(criterion(model(features), labels).values())
                loss.backward()
                optimizer.step()

            model.train(False)
            with torch.no_grad():
                train_loss = criterion(model(train_dataset[:][0]), train_dataset[:][1])
                for k, v in train_loss.items():
                    writer.add_scalar(f"train_loss/{k}", v, global_step=global_step)

                test_loss = criterion(model(test_dataset[:][0]), test_dataset[:][1])
                for k, v in test_loss.items():
                    writer.add_scalar(f"test_loss/{k}", v, global_step=global_step)

            t.set_postfix(
                train_loss=torch.mean(torch.tensor(list(train_loss.values()))).item(),
                test_loss=torch.mean(torch.tensor(list(test_loss.values()))).item(),
            )
            t.update()

            global_step += 1

    return global_step


def cross_valid(
    global_step: int,
    writer: SummaryWriter,
    k_fold: int,
    model: nn.Module,
    criterion: Callable[[Any, Any], dict],
    optimizer: torch.optim.Optimizer,
    dir_dataset: DirDataset,
    batch_size: int,
    num_epochs: int,
) -> int:

    dir_dataloader = dir_dataset.to_dataloader()
    for file_dataset_batch in dir_dataloader:

        file_dataset: FileDataset = file_dataset_batch[0]
        logging.info(f"processing file: {file_dataset.filename}")
        kf = KFold(n_splits=k_fold, shuffle=True)
        for train_indices, test_indices in kf.split(file_dataset):

            filename = os.path.basename(file_dataset.filename)
            train_dataset = Subset(file_dataset, train_indices)
            test_dataset = Subset(file_dataset, test_indices)
            global_step = train(
                global_step,
                writer,
                model,
                criterion,
                optimizer,
                batch_size,
                num_epochs,
                filename,
                train_dataset,
                test_dataset,
            )

    return global_step


def main():

    config_logger(False, False, None)

    global_step = 0
    writer = SummaryWriter("./tb")
    k_fold = 10
    model = Model(OBS_N, 512, ACT_N).to(USE_DEVICE)
    criterion = calc_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dir_dataset = preprocess_data()
    batch_size = 16
    num_epochs = 1000
    iters = 10

    for i in range(iters):

        logging.info(f"iterator: {i}")
        global_step = cross_valid(
            global_step=global_step,
            writer=writer,
            k_fold=k_fold,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dir_dataset=dir_dataset,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
