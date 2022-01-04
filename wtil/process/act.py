from typing import Dict, List, Sequence, Tuple

import numpy as np
from asyncenv.api.wt.wt_pb2 import ActionData, RLAIData, RLVector3D
from numpy import int32

from wtil.process.utils import decode_vector3d, encode_vector3d

ACTION_NUM = 22
DIRECTION_NUM = 4
ACT_N = 30


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


def decode_act(prev_obs: Tuple[RLAIData, List[RLAIData]], data: List[float]) -> ActionData:
    assert len(data) == ACT_N

    prev_self_obs = prev_obs[0] if prev_obs is not None else None

    action_id = decode_action_id(data[:ACTION_NUM])
    move_direction = decode_move_direction(prev_self_obs, data[ACTION_NUM : ACTION_NUM + DIRECTION_NUM])
    control_direction = decode_control_direction(prev_self_obs, data[ACTION_NUM + DIRECTION_NUM :])
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
    if prev_self_obs is None:
        return np.zeros(4)
    return encode_vector3d(prev_self_obs.MovingState.MovingVelocity, act.MoveDirection)


def decode_move_direction(prev_self_obs: RLAIData, data: np.ndarray) -> RLVector3D:
    if prev_self_obs is None:
        return RLVector3D()
    return decode_vector3d(prev_self_obs.MovingState.MovingVelocity, data)


def encode_control_direction(prev_self_obs: RLAIData, act: ActionData) -> np.ndarray:
    if prev_self_obs is None:
        return np.zeros(4)
    return encode_vector3d(prev_self_obs.CurrControlDirection, act.ControlDirection)


def decode_control_direction(prev_self_obs: RLAIData, data: np.ndarray) -> RLVector3D:
    if prev_self_obs is None:
        return RLVector3D()
    return decode_vector3d(prev_self_obs.CurrControlDirection, data)
