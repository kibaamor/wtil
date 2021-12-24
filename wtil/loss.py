import torch
import torch.nn.functional as F

from wtil.preprocess.act import ACTION_NUM, DIRECTION_NUM


def calc_loss(predict: torch.Tensor, target: torch.Tensor) -> dict:
    batch_size = target.shape[0]
    target = target[:, -1:, :].reshape(batch_size, -1)

    predict_action = predict[:, :ACTION_NUM]
    target_action = target[:, :1].reshape(-1).type(torch.long)
    action_loss = F.cross_entropy(predict_action, target_action)

    predict_moving_dir = predict[:, ACTION_NUM : ACTION_NUM + DIRECTION_NUM]
    target_moving_dir = target[:, 1 : 1 + DIRECTION_NUM]
    moving_dir_loss = F.mse_loss(predict_moving_dir, target_moving_dir)

    predict_control_dir = predict[:, ACTION_NUM + DIRECTION_NUM :]
    target_control_dir = target[:, 1 + DIRECTION_NUM :]
    control_dir_loss = F.mse_loss(predict_control_dir, target_control_dir)

    return dict(action=action_loss, moving_dir=moving_dir_loss, control_dir=control_dir_loss)
