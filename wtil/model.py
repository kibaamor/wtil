from typing import Callable, List, Optional

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from wtil.process.act import ACT_N, ACTION_NUM, DIRECTION_NUM
from wtil.process.obs import DEPTH_MAP_SHAPE, ENCODE_DATA_LENGTH, ENCODE_OPPO_DATA_LENGTH


def mlp(
    sizes: List[int],
    activation: Optional[Callable[[], nn.Module]] = None,
    output_activation: Optional[Callable[[], nn.Module]] = None,
) -> nn.Module:

    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        act = activation if i < len(sizes) - 2 else output_activation
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


def conv2d(
    filters: List[int],
    kernel_size: int,
    activation: Optional[Callable[[], nn.Module]] = None,
    pool_size: Optional[int] = None,
    with_flatten: bool = True,
):
    layers = []
    for i in range(len(filters) - 1):
        layers.append(nn.Conv2d(filters[i], filters[i + 1], kernel_size=kernel_size, padding="same"))
        if activation:
            layers.append(activation())
        if False:
            layers.append(nn.Conv2d(filters[i + 1], filters[i + 1], kernel_size=kernel_size, padding="same"))
            if activation:
                layers.append(activation())
        if pool_size is not None:
            layers.append(nn.MaxPool2d(kernel_size=pool_size))
    if with_flatten:
        layers.append(nn.Flatten())
    return nn.Sequential(*layers)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.data_encoder = mlp(
            sizes=[ENCODE_DATA_LENGTH, 128],
            activation=nn.ReLU,
        )
        self.oppo_data_encoder = mlp(
            sizes=[ENCODE_OPPO_DATA_LENGTH, 128],
            activation=nn.ReLU,
        )

        depth_map_channel_num = DEPTH_MAP_SHAPE[0]
        self.depth_map_encoder = conv2d(
            filters=[depth_map_channel_num, depth_map_channel_num * 2, depth_map_channel_num * 4],
            kernel_size=3,
            activation=nn.ReLU,
            pool_size=2,
            with_flatten=True,
        )
        self.core_encoder = nn.GRU(
            input_size=6416,
            hidden_size=1024,
            num_layers=3,
            dropout=0.2,
            batch_first=True,
        )
        self.fc = mlp([1024, 512, ACT_N], activation=nn.ReLU)

    def log(self, writer: SummaryWriter, global_step: int):
        for name, param in self.named_parameters():
            writer.add_histogram(tag=name + "_grad", values=param.grad, global_step=global_step)
            writer.add_histogram(tag=name + "_data", values=param.data, global_step=global_step)

    def forward(self, obs_batch):
        data = obs_batch["data"].type(torch.float)
        oppo_data = obs_batch["oppo_data"].type(torch.float)
        depth_map = obs_batch["depth_map"].type(torch.float)

        encoded_data = self.data_encoder(data)
        encoded_oppo_data = self.oppo_data_encoder(oppo_data)

        shape = depth_map.shape
        depth_map = depth_map.reshape((shape[0] * shape[1],) + shape[2:])
        encoded_depth_map = self.depth_map_encoder(depth_map)
        encoded_depth_map = encoded_depth_map.reshape(shape[0], shape[1], -1)

        core_input = torch.cat([encoded_data, encoded_oppo_data, encoded_depth_map], dim=2)
        encoded_core = self.core_encoder(core_input)
        hidden_state = encoded_core[0][:, -1:, :].reshape(shape[0], -1)

        encoded_fc = self.fc(hidden_state)

        action_mask = obs_batch["action_mask"][:, -1:, :].reshape(shape[0], -1)
        encoded_action = encoded_fc[:, :ACTION_NUM]
        masked_action = encoded_action * action_mask
        action_probs = torch.softmax(masked_action, dim=-1)

        dir_list = torch.tanh(encoded_fc[:, ACTION_NUM:])

        move_dir = dir_list[:, :DIRECTION_NUM]
        control_dir = dir_list[:, DIRECTION_NUM:]

        move_dir_norm = move_dir.norm(dim=-1, keepdim=True)
        control_dir_norm = control_dir.norm(dim=-1, keepdim=True)

        normalized_move_dir = move_dir / move_dir_norm
        normalized_control_dir = control_dir / control_dir_norm

        return dict(
            action_probs=action_probs,
            move_dir=normalized_move_dir,
            control_dir=normalized_control_dir,
        )
