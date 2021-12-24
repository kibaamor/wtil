from typing import Callable, List, Optional

import torch
from torch import nn

from wtil.preprocess.act import ACT_N, ACTION_NUM
from wtil.preprocess.obs import DEPTH_MAP_CHANNEL, DEPTH_MAP_DATA_SIZE, DEPTH_MAP_SHAPE, OBS_N


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
            sizes=[OBS_N - DEPTH_MAP_DATA_SIZE, 512],
            activation=nn.ReLU,
        )
        self.depth_map_encoder = conv2d(
            filters=[DEPTH_MAP_CHANNEL, DEPTH_MAP_CHANNEL * 2, DEPTH_MAP_CHANNEL * 4],
            kernel_size=3,
            activation=nn.ReLU,
            pool_size=2,
            with_flatten=True,
        )
        self.core_encoder = nn.LSTM(
            input_size=6672,
            hidden_size=1024,
            num_layers=3,
            dropout=0.2,
            batch_first=True,
        )
        self.fc = mlp([1024, 512, ACT_N], activation=nn.ReLU)

    def forward(self, obs):
        input_shape = obs.shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        data = obs[:, :, :-DEPTH_MAP_DATA_SIZE]
        encoded_data = self.data_encoder(data)

        depth_map = obs[:, :, OBS_N - DEPTH_MAP_DATA_SIZE :].reshape((batch_size * seq_len,) + DEPTH_MAP_SHAPE)
        encoded_depth_map = self.depth_map_encoder(depth_map).reshape(batch_size, seq_len, -1)

        core_input = torch.cat([encoded_data, encoded_depth_map], dim=2)
        encoded_core = self.core_encoder(core_input)
        hidden_state = encoded_core[0][:, -1:, :].reshape(batch_size, -1)

        encoded_fc = self.fc(hidden_state)

        act = torch.softmax(encoded_fc[:, :ACTION_NUM], dim=-1)
        dir = torch.tanh(encoded_fc[:, ACTION_NUM:])
        ret = torch.cat([act, dir], dim=-1)
        return ret
