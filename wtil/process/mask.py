import numpy as np

from wtil.process.act import ACTION_NUM


def get_move_dir_mask(action_id: int) -> int:
    db = [1, 5, 17, 19, 20]
    return 1 if action_id in db else 0


def get_control_dir_mask(action_id: int) -> int:
    db = [2, 6, 7]
    return 1 if action_id in db else 0


def get_move_dir_mask_array() -> np.ndarray:
    mask = np.zeros(ACTION_NUM, dtype=int)
    for i in range(ACTION_NUM):
        mask[i] = get_move_dir_mask(i)
    return mask


def get_control_dir_mask_array() -> np.ndarray:
    mask = np.zeros(ACTION_NUM, dtype=int)
    for i in range(ACTION_NUM):
        mask[i] = get_control_dir_mask(i)
    return mask
