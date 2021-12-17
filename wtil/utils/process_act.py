from typing import Any, List, Sequence, Tuple

from numpy import int32

from wtil.api.wt_pb2 import ActionData, RLAIData
from wtil.utils.process_utils import encode_vector3d_ratio

ACTION_NUM = 22
DIRECTION_NUM = 2
ACT_N = 26


def encode_act(obs_list: List[Tuple[RLAIData, List[RLAIData]]], act_list: List[ActionData], index: int) -> Any:
    prev_obs = obs_list[index - 1][0] if index > 0 else None
    act = act_list[index]

    encoded_action_id = encode_action_id(act.ActionID)
    encoded_move_direction = encode_move_direction(prev_obs, act)
    encoded_control_direction = encode_control_direction(prev_obs, act)
    encoded_act = encoded_action_id + encoded_move_direction + encoded_control_direction

    assert len(encoded_act) == (ACT_N - ACTION_NUM + 1)
    return encoded_act


def encode_action_id(action_ids: Sequence[int32]) -> List[float]:
    """1"""
    action_id = action_ids[0] if len(action_ids) > 0 else 0
    assert action_id < ACTION_NUM
    return [action_id]  # np.eye(MAX_ACTION_ID + 1)[action_id].tolist()


def encode_move_direction(prev_obs: RLAIData, act: ActionData) -> List[float]:
    """2"""
    if prev_obs is None:
        return [0.0] * 2
    return encode_vector3d_ratio(prev_obs.MovingState.MovingVelocity, act.MoveDirection)


def encode_control_direction(prev_obs: RLAIData, act: ActionData) -> List[float]:
    """2"""
    if prev_obs is None:
        return [0.0] * 2
    return encode_vector3d_ratio(prev_obs.CurrControlDirection, act.ControlDirection)
