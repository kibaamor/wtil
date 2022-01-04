from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def calc_action(predict_action_probs: torch.Tensor, target_action_id: torch.Tensor) -> Tuple[torch.Tensor, float]:
    target_action = target_action_id[:, -1, :].reshape(-1)
    action_loss = F.cross_entropy(predict_action_probs, target_action)

    predict_action = predict_action_probs.argmax(-1)
    action_matched = (predict_action == target_action).type(torch.float).sum()
    action_accuracy = (action_matched / predict_action.numel()).item()

    return action_loss, action_accuracy


def calc_dir(predict_dir: torch.Tensor, target_dir: torch.Tensor, dir_mask: torch.Tensor) -> Tuple[torch.Tensor, float]:
    target_dir = target_dir[:, -1:, :].reshape(target_dir.shape[0], -1)

    return 0, 0


def calc_loss(
    predict: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    dir_mask: Dict[str, torch.Tensor],
) -> dict:

    action_loss, action_accuracy = calc_action(predict["action_probs"], target["action_id"])
    move_dir_loss, move_dir_accuracy = calc_dir(predict["move_dir"], target["move_dir"], dir_mask["move_dir"])
    control_dir_loss, control_dir_accuracy = calc_dir(
        predict["control_dir"], target["control_dir"], dir_mask["control_dir"]
    )

    loss = dict(action=action_loss, move_dir=move_dir_loss, control_dir=control_dir_loss)
    accuracy = dict(action=action_accuracy, move_dir=move_dir_accuracy, control_dir=control_dir_accuracy)

    return loss, accuracy
