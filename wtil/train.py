#!/usr/bin/env python3
import logging
import os
import pathlib

import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

from wtil.net import Model
from wtil.utils.dataset import DirDataset
from wtil.utils.logger import config_logger
from wtil.utils.process import gen_process_fn
from wtil.utils.process_act import ACT_N, ACTION_NUM, DIRECTION_NUM
from wtil.utils.process_obs import OBS_N

USE_CUDA = False  # torch.cuda.is_available()
USE_DEVICE = "cuda" if USE_CUDA else "cpu"


def train(writer, dir_dataloader, model, loss_func, optimizer, epochs):
    model.train()
    step = 0
    for _ in range(epochs):
        for file_dataloader in dir_dataloader:
            for file_data in file_dataloader:
                logging.info(f"processing file: {file_data.filename}")
                with tqdm.tqdm(total=len(file_data), desc=f"{os.path.basename(file_data.filename)}") as t:
                    for batch in file_data:
                        assert len(batch) == 2
                        # print("batch", type(batch), len(batch), batch[0].shape, batch[1].shape)

                        obs_batch, act_batch = batch[0], batch[1]
                        predict_act_batch = model(obs_batch)

                        action_loss, moving_dir_loss, control_dir_loss = loss_func(predict_act_batch, act_batch)
                        loss = action_loss + moving_dir_loss + control_dir_loss
                        loss.backward()
                        optimizer.step()

                        writer.add_scalar("loss/action", action_loss.item(), global_step=step)
                        writer.add_scalar("loss/moving", moving_dir_loss.item(), global_step=step)
                        writer.add_scalar("loss/control", control_dir_loss.item(), global_step=step)
                        t.set_postfix(
                            action=action_loss.item(),
                            moving=moving_dir_loss.item(),
                            control=control_dir_loss.item(),
                        )
                        t.update()

                        step += 1


def get_dir_dataloader() -> DirDataset:
    data_dir = pathlib.Path(__file__).parent / "data"

    path_glob = f"{data_dir}/*.txt"
    seq_len = 8
    batch_size = 16
    process_fn = gen_process_fn(device=USE_DEVICE)
    num_workers = 1

    dir_dataloader = DirDataset.make_dataloader(path_glob, seq_len, batch_size, process_fn, num_workers)
    return dir_dataloader


def calc_loss(predict, target):
    predict_action = predict[:, :, :ACTION_NUM].reshape(-1, ACTION_NUM)
    target_action = target[:, :, :1].reshape(-1).type(torch.long)
    action_loss = F.cross_entropy(predict_action, target_action)

    predict_moving_dir = predict[:, :, ACTION_NUM : ACTION_NUM + DIRECTION_NUM].reshape(-1)
    target_moving_dir = target[:, :, 1 : 1 + DIRECTION_NUM].reshape(-1)
    moving_dir_loss = F.mse_loss(predict_moving_dir, target_moving_dir)

    predict_control_dir = predict[:, :, ACTION_NUM + DIRECTION_NUM :].reshape(-1)
    target_control_dir = target[:, :, 1 + DIRECTION_NUM :].reshape(-1)
    control_dir_loss = F.mse_loss(predict_control_dir, target_control_dir)

    return action_loss, moving_dir_loss, control_dir_loss


def main():
    config_logger(False, False, None)

    dir_dataloader = get_dir_dataloader()

    hidden_n = 512
    lr = 1e-4

    model = Model(OBS_N, hidden_n, ACT_N).to(USE_DEVICE)
    loss_func = calc_loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    writer = SummaryWriter("./tb")
    epochs = 10
    train(writer, dir_dataloader, model, loss_func, optimizer, epochs)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
