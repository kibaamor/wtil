import torch
from torch import nn

from wtil.preprocess.act import ACTION_NUM


class Model(nn.Module):
    def __init__(self, obs_n: int, hidden_n: int, act_n: int, device: str = "auto"):
        super().__init__()
        self.fc1 = nn.Linear(obs_n, hidden_n)
        self.fc2 = nn.Linear(hidden_n, act_n)

    def forward(self, obs):
        hidden = torch.relu(self.fc1(obs))
        hidden = self.fc2(hidden)
        act = torch.softmax(hidden[:, :, :ACTION_NUM], dim=-1)
        dir = torch.tanh(hidden[:, :, ACTION_NUM:])
        ret = torch.cat([act, dir], dim=-1)
        return ret
