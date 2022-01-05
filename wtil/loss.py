from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def calc_action(
    predict_action_probs: torch.Tensor,
    target_action_id: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    target_action = target_action_id[:, -1, :].reshape(-1)
    action_loss = F.cross_entropy(predict_action_probs, target_action)

    # predict_action = predict_action_probs.argmax(-1)
    predict_action = predict_action_probs.multinomial(num_samples=1).squeeze()
    action_matched = (predict_action == target_action).type(torch.float).sum()
    action_accuracy = (action_matched / predict_action.numel()).item()

    return predict_action.detach().type(torch.long), action_loss, action_accuracy


def calc_dir(
    predict_action: torch.Tensor,
    predict_dir: torch.Tensor,
    target_dir: torch.Tensor,
    dir_mask: torch.Tensor,
) -> Tuple[torch.Tensor, float]:

    target_dir = target_dir[:, -1:, :].reshape(target_dir.shape[0], -1)
    mask = dir_mask[predict_action]

    dot = torch.sum(predict_dir * target_dir, dim=-1)
    ratio = torch.rad2deg(torch.acos(dot)) / 360.0
    masked_ratio = ratio * mask
    dir_loss = masked_ratio.mean()

    delta_ratio = 0.02  # 360 * 0.02 = 7.2 degree
    dir_matched = torch.where(ratio < delta_ratio, 1, 0)
    total_count = mask.sum()
    matched_count = torch.sum(dir_matched * mask)
    dir_accuracy = 1.0
    if total_count > 0:
        dir_accuracy = (matched_count * 1.0 / total_count).item()

    return dir_loss, dir_accuracy


def calc_loss(
    predict: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    dir_mask: Dict[str, torch.Tensor],
) -> dict:

    predict_action, action_loss, action_accuracy = calc_action(predict["action_probs"], target["action_id"])
    move_dir_loss, move_dir_accuracy = calc_dir(
        predict_action,
        predict["move_dir"],
        target["move_dir"],
        dir_mask["move_dir"],
    )
    control_dir_loss, control_dir_accuracy = calc_dir(
        predict_action,
        predict["control_dir"],
        target["control_dir"],
        dir_mask["control_dir"],
    )

    loss = dict(action=action_loss, move_dir=move_dir_loss, control_dir=control_dir_loss)
    accuracy = dict(action=action_accuracy, move_dir=move_dir_accuracy, control_dir=control_dir_accuracy)

    return loss, accuracy
