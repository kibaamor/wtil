#!/usr/bin/env python3
import glob
import logging
import os
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict

import numpy as np
import torch
import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

from wtil.dataset.npy import NpyDirDataset, NpyFileDataset
from wtil.loss import calc_loss
from wtil.model import Model
from wtil.process.mask import get_control_dir_mask_array, get_move_dir_mask_array
from wtil.utils.logger import config_logger

USE_CUDA = torch.cuda.is_available()
USE_DEVICE = "cuda" if USE_CUDA else "cpu"
BEST_ACCURACY = 0.0

WORKING_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = str(WORKING_DIR / "data_processed")
CHECKPOINTS_DIR = str(WORKING_DIR / "checkpoints")
TENSORBOARD_DIR = str(WORKING_DIR / "tensorboard")


def process_data(data_dir=DATA_DIR, seq_len=8):
    return NpyDirDataset(path_glob=f"{data_dir}/*.npy", seq_len=seq_len)


def to_device(data: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in data.items()}


def test(
    global_step: int,
    writer: SummaryWriter,
    prefix: str,
    dataset: Dataset,
    model: Model,
    dir_mask: Dict[str, torch.Tensor],
    criterion: Callable[[Any, Any, Any], dict],
    batch_size: int,
) -> float:

    losses = defaultdict(list)
    accuracies = defaultdict(list)

    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for features, labels in dataloader:
            features = to_device(features, USE_DEVICE)
            labels = to_device(labels, USE_DEVICE)

            loss, accuracy = criterion(model(features), labels, dir_mask)
            for k, v in loss.items():
                losses[k].append(v.item())
            for k, v in accuracy.items():
                accuracies[k].append(v)

    for k, v in losses.items():
        writer.add_scalar(f"{prefix}_loss/{k}", np.mean(v), global_step=global_step)
    for k, v in accuracies.items():
        writer.add_scalar(f"{prefix}_accuracy/{k}", np.mean(v), global_step=global_step)

    avg_loss = np.array(list(losses.values())).mean()
    avg_accuracy = np.array(list(accuracies.values())).mean()

    return avg_loss, avg_accuracy


def train(
    global_step: int,
    writer: SummaryWriter,
    model: Model,
    dir_mask: Dict[str, torch.Tensor],
    criterion: Callable[[Any, Any, Any], dict],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    num_epochs: int,
    filename: str,
    train_dataset: Dataset,
) -> int:

    with tqdm.tqdm(total=num_epochs, desc=f"{filename}") as t:
        for _ in range(num_epochs):
            model.train(True)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for features, labels in train_dataloader:
                features = to_device(features, USE_DEVICE)
                labels = to_device(labels, USE_DEVICE)

                optimizer.zero_grad()
                loss, _ = criterion(model(features), labels, dir_mask)
                loss = sum(loss.values())
                loss.backward()
                optimizer.step()

            model.log(writer, global_step)

            model.train(False)
            loss, accuracy = test(
                global_step=global_step,
                writer=writer,
                prefix="train",
                dataset=train_dataset,
                model=model,
                dir_mask=dir_mask,
                criterion=criterion,
                batch_size=batch_size,
            )

            t.set_postfix(loss=loss, accuracy=accuracy)
            t.update()

            global_step += 1

    return global_step


def cross_valid(
    global_step: int,
    writer: SummaryWriter,
    k_fold: int,
    model: Model,
    dir_mask: Dict[str, torch.Tensor],
    criterion: Callable[[Any, Any], dict],
    optimizer: torch.optim.Optimizer,
    dir_dataset: NpyDirDataset,
    batch_size: int,
    num_epochs: int,
) -> int:

    global BEST_ACCURACY

    dir_dataloader = dir_dataset.to_dataloader()
    for file_dataset_batch in dir_dataloader:

        file_dataset: NpyFileDataset = file_dataset_batch[0]
        logging.info(f"processing file: {file_dataset.filename}")
        kf = KFold(n_splits=k_fold, shuffle=True)
        for train_indices, test_indices in kf.split(file_dataset):

            filename = os.path.basename(file_dataset.filename)
            train_dataset = Subset(file_dataset, train_indices)
            test_dataset = Subset(file_dataset, test_indices)
            global_step = train(
                global_step=global_step,
                writer=writer,
                model=model,
                dir_mask=dir_mask,
                criterion=criterion,
                optimizer=optimizer,
                batch_size=batch_size,
                num_epochs=num_epochs,
                filename=filename,
                train_dataset=train_dataset,
            )

            _, test_accuracy = test(
                global_step=global_step,
                writer=writer,
                prefix="test",
                dataset=test_dataset,
                model=model,
                dir_mask=dir_mask,
                criterion=criterion,
                batch_size=batch_size,
            )

            logging.info(f"test accuracy: {test_accuracy:0.5f}, best accuracy:{BEST_ACCURACY:0.5f}")
            if test_accuracy > BEST_ACCURACY:
                BEST_ACCURACY = test_accuracy
                model.to("cpu")
                torch.save(model.state_dict(), f"{CHECKPOINTS_DIR}/wtil_{test_accuracy:.5f}.pth")
                model.to(USE_DEVICE)

    return global_step


def get_best_checkpoint() -> str:

    global BEST_ACCURACY

    filename_list = glob.glob(f"{CHECKPOINTS_DIR}/wtil_*.pth")
    logging.debug(f"checkpoints: {filename_list}")

    if len(filename_list) == 0:
        return ""

    score_list = []
    for filename in filename_list:
        score_str = filename.split("_")[1][: -len(".pth")]
        score_val = float(score_str)
        score_list.append((score_val, score_str))
    best_score = max(score_list, key=lambda v: v[0])

    BEST_ACCURACY = best_score[0]
    return f"{CHECKPOINTS_DIR}/wtil_{best_score[1]}.pth"


def main():

    config_logger(False, False, None)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    model = Model()

    pth = get_best_checkpoint()
    if len(pth) > 0:
        logging.info(f"load model from file: {pth}")
        model.load_state_dict(torch.load(pth))

    model.to(USE_DEVICE)

    batch_size = 256
    seq_len = 16
    k_fold = 10
    dir_dataset = process_data(seq_len=seq_len)
    num_epochs = 10
    iters = 100000
    lr = 1e-3

    global_step = 0
    writer = SummaryWriter(TENSORBOARD_DIR)
    criterion = calc_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    move_dir_mask = torch.from_numpy(get_move_dir_mask_array()).type(torch.long).to(USE_DEVICE)
    control_dir_mask = torch.from_numpy(get_control_dir_mask_array()).type(torch.long).to(USE_DEVICE)
    dir_mask = dict(
        move_dir=move_dir_mask,
        control_dir=control_dir_mask,
    )

    for i in range(iters):

        logging.info(f"iterator: {i}")
        global_step = cross_valid(
            global_step=global_step,
            writer=writer,
            k_fold=k_fold,
            model=model,
            dir_mask=dir_mask,
            criterion=criterion,
            optimizer=optimizer,
            dir_dataset=dir_dataset,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
