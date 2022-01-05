from typing import Dict, List, Sequence, Tuple

import numpy as np
from asyncenv.api.wt.wt_pb2 import ActionData, RLAIData, RLVector3D
from numpy import int32

from wtil.process.utils import unit_vector

ACTION_NUM = 22
DIRECTION_NUM = 3
ACT_N = 28


def encode_act(prev_obs: Tuple[RLAIData, List[RLAIData]], act: ActionData) -> Dict[str, np.ndarray]:
    prev_self_obs = prev_obs[0] if prev_obs is not None else None

    encoded_action_id = encode_action_id(act.ActionID)
    encoded_move_direction = encode_move_direction(prev_self_obs, act)
    encoded_control_direction = encode_control_direction(prev_self_obs, act)

    return dict(
        action_id=encoded_action_id,
        move_dir=encoded_move_direction,
        control_dir=encoded_control_direction,
    )


def decode_act(prev_obs: Tuple[RLAIData, List[RLAIData]], data: Dict[str, np.ndarray]) -> ActionData:
    prev_self_obs = prev_obs[0] if prev_obs is not None else None

    action_id = decode_action_id(data["action_probs"])
    move_direction = decode_move_direction(prev_self_obs, data["move_dir"])
    control_direction = decode_control_direction(prev_self_obs, data["control_dir"])
    return ActionData(
        ActionID=[action_id],
        EpisodeId=prev_self_obs.EpisodeId,
        MoveDirection=move_direction,
        ControlDirection=control_direction,
    )


def encode_action_id(action_ids: Sequence[int32]) -> np.ndarray:
    action_id = action_ids[0] if len(action_ids) > 0 else 0
    assert 0 <= action_id < ACTION_NUM
    return np.array([action_id])


def decode_action_id(data: List[float]) -> int32:
    assert len(data) == ACTION_NUM
    action_id = np.random.choice(ACTION_NUM, 1, p=data).item()
    return int32(action_id)


def encode_move_direction(prev_self_obs: RLAIData, act: ActionData) -> np.ndarray:
    dir = act.MoveDirection
    return unit_vector(np.array([dir.X, dir.Y, dir.Z]))


def decode_move_direction(prev_self_obs: RLAIData, data: np.ndarray) -> RLVector3D:
    return RLVector3D(X=data[0], Y=data[1], Z=data[2])


def encode_control_direction(prev_self_obs: RLAIData, act: ActionData) -> np.ndarray:
    dir = act.ControlDirection
    return unit_vector(np.array([dir.X, dir.Y, dir.Z]))


def decode_control_direction(prev_self_obs: RLAIData, data: np.ndarray) -> RLVector3D:
    return RLVector3D(X=data[0], Y=data[1], Z=data[2])
