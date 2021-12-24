import re
from typing import Any, List, Sequence, Tuple

import numpy as np

from wtil.api.wt_pb2 import (
    ActionData,
    ERLCharacterActionState,
    ERLCharacterMovingState,
    ERLVirtualDepthMapPointHitObjectType,
    NearestProjectileActor,
    ObservationData,
    RLActionState,
    RLAIData,
    RLChargeAttackTime,
    RLMovingState,
    RLRangedWeapon,
    VirtualDepthMapSimple,
)
from wtil.preprocess.utils import encode_bool, encode_onehot, encode_ratio, encode_vector3d

MaxHealth = 1000
MaxHealthPotion = 10.0
MaxStamina = 400.0

MAX_BURST_COUNT = {0: 0, -1: 1, 1: 2, 3: 3}
MAX_AMMO_LEFT = 10000
MOVING_STATE = {
    ERLCharacterMovingState.CHAR_MOVE_STATE_WALKING: 0,
    ERLCharacterMovingState.CHAR_MOVE_STATE_RUNNING: 1,
    ERLCharacterMovingState.CHAR_MOVE_STATE_DODGING: 2,
    ERLCharacterMovingState.CHAR_MOVE_STATE_SLIDING: 3,
    ERLCharacterMovingState.CHAR_MOVE_STATE_NOT_MOVING: 4,
    ERLCharacterMovingState.CHAR_MOVE_STATE_WEAK: 5,
    ERLCharacterMovingState.CHAR_MOVE_STATE_DODGESHOOT: 6,
    ERLCharacterMovingState.CHAR_MOVE_STATE_STAGGER: 7,
}
ACTION_STATE = {
    ERLCharacterActionState.RANGED_ATTAK: 0,
    ERLCharacterActionState.SPECIAL_RANGED_ATTACK: 1,
    ERLCharacterActionState.DEFENDING: 2,
    ERLCharacterActionState.AIM: 3,
    ERLCharacterActionState.MELEE_SPECIAL_1: 4,
    ERLCharacterActionState.MELEE_SPECIAL_2: 5,
    ERLCharacterActionState.RELOAD: 6,
    ERLCharacterActionState.USE_DRAGON_HEART: 7,
    ERLCharacterActionState.MELEE_SPECIAL_2_CHARGING: 8,
    ERLCharacterActionState.SPECIAL_RANGED_ATTACK_CHARGING: 9,
}
MAX_ACTION_STATE_LENGTH = 4
ACTION_ID_NUM = 22
MAX_VALID_ACTION_LENGTH = 4

DEPTH_MAP_CHANNEL = 7  # number of ERLVirtualDepthMapPointHitObjectType + 1(distances)
DEPTH_MAP_HEIGHT = 45
DEPTH_MAP_WIDTH = 80
DEPTH_MAP_SHAPE = (DEPTH_MAP_CHANNEL, DEPTH_MAP_HEIGHT, DEPTH_MAP_WIDTH)
DEPTH_MAP_DATA_SIZE = DEPTH_MAP_CHANNEL * DEPTH_MAP_HEIGHT * DEPTH_MAP_WIDTH

DATA_N = 25 + 6 + MaxHealth + int(MaxStamina / 10) + MaxHealthPotion + 3 + 3 + 3 + 11 + 75 + 88
OBS_N = int(DATA_N * 2) + DEPTH_MAP_DATA_SIZE


def split_obs(obs: ObservationData) -> Tuple[RLAIData, List[RLAIData]]:
    for i in range(len(obs.AIData)):
        if obs.AIData[i].EpisodeId == obs.EpisodeId:
            return obs.AIData[i], obs.AIData[:i] + obs.AIData[i:]


def process_obs(obs_list: List[ObservationData]) -> List[Tuple[RLAIData, List[RLAIData]]]:
    return [split_obs(obs) for obs in obs_list]


def encode_obs(obs_list: List[Tuple[RLAIData, List[RLAIData]]], act_list: List[ActionData], index: int) -> Any:
    data, oppo_data_list = obs_list[index]
    oppo_data = oppo_data_list[0] if len(oppo_data_list) > 0 else RLAIData()
    encoded_data = encode_data(data) + encode_data(oppo_data)
    assert len(encoded_data) == int(DATA_N * 2)

    encoded_depth_map = encode_depth_map(data)
    assert encoded_depth_map.shape == DEPTH_MAP_SHAPE

    return encoded_data + encoded_depth_map.reshape(-1).tolist()


def encode_data(data: RLAIData) -> List[float]:
    assert MaxHealth == data.MaxHealth
    encoded_weapon = encode_weapon(data.HeldWeapon)
    encoded_projectile = encode_projectile(data.ProjectileActor)
    encoded_health = encode_ratio(data.CurrHealth / data.MaxHealth, data.MaxHealth)
    encoded_stamina = encode_ratio(data.CurrStamina / MaxStamina, int(MaxStamina / 10))
    encoded_health_potion = encode_ratio(data.HealthPotion / MaxHealthPotion, MaxHealthPotion)
    encoded_location = encode_vector3d(data.CurrLocation)
    encoded_face_dir = encode_vector3d(data.CurrFaceDirection)
    encoded_control_dir = encode_vector3d(data.CurrControlDirection)
    encoded_moving_state = encode_moving_state(data.MovingState)
    encoded_action_state = encode_action_state(data.ActionState)
    encoded_valid_actions = encode_valid_actions(data.ValidActions)

    return (
        encoded_weapon
        + encoded_projectile
        + encoded_health
        + encoded_stamina
        + encoded_health_potion
        + encoded_location
        + encoded_face_dir
        + encoded_control_dir
        + encoded_moving_state
        + encoded_action_state
        + encoded_valid_actions
    )


def encode_weapon(weapon: RLRangedWeapon) -> List[float]:
    """25"""
    if weapon.WeaponID != -1:
        accumulate_shoot = encode_bool(weapon.SupportAccumlateShoot)  # 1

        assert len(MAX_BURST_COUNT) == 4
        max_burst_count = encode_onehot(weapon.MaxBurstCount, MAX_BURST_COUNT)  # 4

        ammo_loaded_state = encode_ratio(weapon.AmmoLoaded / weapon.MaxAmmoLoad, 10)  # 10
        ammo_left_state = encode_ratio(weapon.TotalAmmoLeft / MAX_AMMO_LEFT, 10)  # 10
        return accumulate_shoot + max_burst_count + ammo_loaded_state + ammo_left_state

    return [0.0] * 25


def encode_moving_state(moving_state: RLMovingState) -> List[float]:
    """11"""
    assert len(MOVING_STATE) == 8
    encoded_cur_state = encode_onehot(moving_state.CurrState, MOVING_STATE)
    encoded_moving_velocity = encode_vector3d(moving_state.MovingVelocity)
    return encoded_cur_state + encoded_moving_velocity


def encode_action_state(action_state: RLActionState) -> List[float]:
    """75"""
    encoded_cur_state_list = encode_action_state_cur_state_list(action_state.CurrStateList)
    encoded_ranged_attack = encode_charge_attack_time(action_state.RangedAttack)
    encoded_special_ranged_attack = encode_charge_attack_time(action_state.SpecialRangedAttack)
    encoded_melee_ranged_attack2 = encode_charge_attack_time(action_state.MeleeSpecialAttack2)
    encoded_normal_defend_success = encode_bool(action_state.NormalShootingDefendSuccess)
    encoded_perfect_defend_success = encode_bool(action_state.PerfectDefendSuccess)
    return (
        encoded_cur_state_list
        + encoded_ranged_attack
        + encoded_special_ranged_attack
        + encoded_melee_ranged_attack2
        + encoded_normal_defend_success
        + encoded_perfect_defend_success
    )


def encode_action_state_cur_state_list(state_list) -> List[float]:
    """40"""
    encoded = []

    assert len(ACTION_STATE) == 10
    for state in state_list:
        encoded.extend(encode_onehot(state, ACTION_STATE))

    assert MAX_ACTION_STATE_LENGTH == 4
    encoded.extend([0.0] * len(ACTION_STATE) * max(0, MAX_ACTION_STATE_LENGTH - len(state_list)))

    return encoded


def encode_charge_attack_time(cat: RLChargeAttackTime) -> List[float]:
    """11"""
    if cat.MaxTime > 0:
        can_fire = encode_bool(cat.CurrTime > cat.MinTime)
        return can_fire + encode_ratio(cat.CurrTime / cat.MaxTime, 10)
    return [0.0] * 11


def encode_valid_actions(valid_actions: Sequence[int]) -> List[float]:
    """88"""
    encoded = []
    for action in valid_actions:
        assert ACTION_ID_NUM == 22
        encoded.extend(np.eye(ACTION_ID_NUM)[action].tolist())
    encoded.extend([0.0] * ACTION_ID_NUM * max(0, MAX_VALID_ACTION_LENGTH - len(valid_actions)))
    return encoded


def encode_projectile(projectile: NearestProjectileActor) -> List[float]:
    """6"""
    return encode_vector3d(projectile.Location) + encode_vector3d(projectile.Velocity)


def encode_depth_map(data: RLAIData) -> np.ndarray:
    depth_map: VirtualDepthMapSimple = data.DepthMap

    assert depth_map.ScreenHeight == DEPTH_MAP_HEIGHT
    assert depth_map.ScreenWidth == DEPTH_MAP_WIDTH

    data = np.array(re.split(",|\|", depth_map.ScreenPixelString), dtype=np.int)
    hits = data[::2].reshape(DEPTH_MAP_HEIGHT, DEPTH_MAP_WIDTH)
    distances = data[1::2].reshape(DEPTH_MAP_HEIGHT, DEPTH_MAP_WIDTH)
    max_distance = 5000.0

    enemy = np.where(hits == int(ERLVirtualDepthMapPointHitObjectType.OBJECT_ENEMY), 1.0, 0.0)
    ally = np.where(hits == int(ERLVirtualDepthMapPointHitObjectType.OBJECT_ALLY), 1.0, 0.0)
    neutral = np.where(hits == int(ERLVirtualDepthMapPointHitObjectType.OBJECT_NEUTRAl), 1.0, 0.0)
    obstacle = np.where(hits == int(ERLVirtualDepthMapPointHitObjectType.OBJECT_OBSTACLE), 1.0, 0.0)
    obstacle_destructible = np.where(
        hits == int(ERLVirtualDepthMapPointHitObjectType.OBJECT_OBSTACLE_DESTRUCTIBLE), 1.0, 0.0
    )
    none = np.where(hits == int(ERLVirtualDepthMapPointHitObjectType.OBJECT_NONE), 1.0, 0.0)
    dist = distances / max_distance

    return np.stack([enemy, ally, neutral, obstacle, obstacle_destructible, none, dist])
