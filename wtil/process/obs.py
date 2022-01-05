import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
from asyncenv.api.wt.wt_pb2 import (
    ERLCharacterActionState,
    ERLCharacterMovingState,
    ERLVirtualDepthMapPointHitObjectType,
    NearestProjectileActor,
    ObservationData,
    RLActionState,
    RLActionStateItem,
    RLAIData,
    RLChargeAttackTime,
    RLMovingState,
    RLRangedWeapon,
    VirtualDepthMapSimple,
)

from wtil.process.utils import encode_bool, encode_onehot, encode_ratio, encode_vector3d

MAX_ENCODED_HEALTH_LENGTH = 100
MAX_ENCODED_STAMINA_LENGTH = 100
MAX_ENCODED_HEALTH_POTION_LENGTH = 100
MAX_ENCODED_AMMO_LENGTH = 10
MAX_ACTION_STATE_LENGTH = 4
MAX_ENCODED_TIME_LENGTH = 10

MAX_HEALTH_POTION = 10.0
MAX_STAMINA = 400.0
MAX_AMMO_LEFT = 10000
ACTION_ID_NUM = 22

MAX_BURST_COUNT = {0: 0, -1: 1, 1: 2, 3: 3}
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

ENCODE_DATA_LENGTH = 426
ENCODE_OPPO_DATA_LENGTH = 426
DEPTH_MAP_SHAPE = (7, 45, 80)
ACTION_MASK_LENGTH = ACTION_ID_NUM


def process_obs(obs: ObservationData) -> Tuple[RLAIData, List[RLAIData]]:
    for i in range(len(obs.AIData)):
        if obs.AIData[i].EpisodeId == obs.EpisodeId:
            return obs.AIData[i], obs.AIData[:i] + obs.AIData[i:]


def encode_obs(obs: Tuple[RLAIData, List[RLAIData]]) -> Dict[str, np.ndarray]:
    data, oppo_data_list = obs
    oppo_data = oppo_data_list[0] if len(oppo_data_list) > 0 else RLAIData()

    encoded_data = encode_data(data)
    encoded_oppo_data = encode_data(oppo_data)
    encoded_depth_map = encode_depth_map(data)
    encoded_action_mask = gen_action_mask(data)

    assert ENCODE_DATA_LENGTH == len(encoded_data)
    assert ENCODE_OPPO_DATA_LENGTH == len(encoded_oppo_data)
    assert DEPTH_MAP_SHAPE == encoded_depth_map.shape
    assert ACTION_MASK_LENGTH == len(encoded_action_mask)

    return dict(
        data=encoded_data,
        oppo_data=encoded_oppo_data,
        depth_map=encoded_depth_map,
        action_mask=encoded_action_mask,
    )


def encode_data(data: RLAIData) -> np.ndarray:
    encoded_weapon = encode_weapon(data.HeldWeapon)
    encoded_projectile = encode_projectile(data.ProjectileActor)
    encoded_health = encode_ratio(data.CurrHealth / data.MaxHealth, MAX_ENCODED_HEALTH_LENGTH)
    encoded_stamina = encode_ratio(data.CurrStamina / MAX_STAMINA, MAX_ENCODED_STAMINA_LENGTH)
    encoded_health_potion = encode_ratio(data.HealthPotion / MAX_HEALTH_POTION, MAX_ENCODED_HEALTH_POTION_LENGTH)
    encoded_location = encode_vector3d(data.CurrLocation)
    encoded_face_dir = encode_vector3d(data.CurrFaceDirection)
    encoded_control_dir = encode_vector3d(data.CurrControlDirection)
    encoded_moving_state = encode_moving_state(data.MovingState)
    encoded_action_state = encode_action_state(data.ActionState)

    return np.concatenate(
        [
            encoded_weapon,
            encoded_projectile,
            encoded_health,
            encoded_stamina,
            encoded_health_potion,
            encoded_location,
            encoded_face_dir,
            encoded_control_dir,
            encoded_moving_state,
            encoded_action_state,
        ]
    )


def encode_weapon(weapon: RLRangedWeapon) -> np.ndarray:
    MAX_SIZE = 1 + len(MAX_BURST_COUNT) + MAX_ENCODED_AMMO_LENGTH + MAX_ENCODED_AMMO_LENGTH

    if weapon.WeaponID == -1:
        return np.zeros(MAX_SIZE)

    accumulate_shoot = encode_bool(weapon.SupportAccumlateShoot)

    assert len(MAX_BURST_COUNT) == 4
    max_burst_count = encode_onehot(weapon.MaxBurstCount, MAX_BURST_COUNT)
    ammo_loaded_state = encode_ratio(weapon.AmmoLoaded / weapon.MaxAmmoLoad, MAX_ENCODED_AMMO_LENGTH)
    ammo_left_state = encode_ratio(weapon.TotalAmmoLeft / MAX_AMMO_LEFT, MAX_ENCODED_AMMO_LENGTH)

    return np.concatenate(
        [
            accumulate_shoot,
            max_burst_count,
            ammo_loaded_state,
            ammo_left_state,
        ]
    )


def encode_moving_state(moving_state: RLMovingState) -> np.ndarray:
    assert len(MOVING_STATE) == 8
    encoded_cur_state = encode_onehot(moving_state.CurrState, MOVING_STATE)
    encoded_moving_velocity = encode_vector3d(moving_state.MovingVelocity)
    return np.concatenate(
        [
            encoded_cur_state,
            encoded_moving_velocity,
        ]
    )


def encode_action_state(action_state: RLActionState) -> np.ndarray:
    encoded_cur_state_list = encode_action_state_cur_state_list(action_state.CurrStateList)
    encoded_ranged_attack = encode_charge_attack_time(action_state.RangedAttack)
    encoded_special_ranged_attack = encode_charge_attack_time(action_state.SpecialRangedAttack)
    encoded_melee_ranged_attack2 = encode_charge_attack_time(action_state.MeleeSpecialAttack2)
    encoded_normal_defend_success = encode_bool(action_state.NormalShootingDefendSuccess)
    encoded_perfect_defend_success = encode_bool(action_state.PerfectDefendSuccess)
    return np.concatenate(
        [
            encoded_cur_state_list,
            encoded_ranged_attack,
            encoded_special_ranged_attack,
            encoded_melee_ranged_attack2,
            encoded_normal_defend_success,
            encoded_perfect_defend_success,
        ]
    )


def encode_action_state_cur_state_list(state_list: Sequence[RLActionStateItem]) -> np.ndarray:
    MAX_SIZE = len(ACTION_STATE) * MAX_ACTION_STATE_LENGTH

    if len(state_list) == 0:
        return np.zeros(MAX_SIZE)

    state_list = [item.State for item in state_list]
    state_list.sort()

    encoded_state_list = [encode_onehot(state, ACTION_STATE) for state in state_list[:MAX_ACTION_STATE_LENGTH]]
    encoded = np.concatenate(encoded_state_list)

    if len(encoded) < MAX_SIZE:
        encoded = np.pad(encoded, (0, MAX_SIZE - len(encoded)))

    return encoded


def encode_charge_attack_time(cat: RLChargeAttackTime) -> np.ndarray:
    MAX_SIZE = 1 + MAX_ENCODED_TIME_LENGTH

    if cat.MaxTime <= 0:
        return np.zeros(MAX_SIZE)

    encoded_can_fire = encode_bool(cat.CurrTime > cat.MinTime)
    encoded_fire_ratio = encode_ratio(cat.CurrTime / cat.MaxTime, MAX_ENCODED_TIME_LENGTH)
    return np.concatenate(
        [
            encoded_can_fire,
            encoded_fire_ratio,
        ]
    )


def encode_projectile(projectile: NearestProjectileActor) -> np.ndarray:
    encoded_location = encode_vector3d(projectile.Location)
    encoded_velocity = encode_vector3d(projectile.Velocity)
    return np.concatenate(
        [
            encoded_location,
            encoded_velocity,
        ]
    )


def encode_depth_map(data: RLAIData) -> np.ndarray:
    depth_map: VirtualDepthMapSimple = data.DepthMap
    height = depth_map.ScreenHeight
    width = depth_map.ScreenWidth

    data = np.array(re.split(",|\|", depth_map.ScreenPixelString), dtype=np.int)  # noqa: W605
    hits = data[::2].reshape(height, width)
    distances = data[1::2].reshape(height, width)

    enemy = np.where(hits == int(ERLVirtualDepthMapPointHitObjectType.OBJECT_ENEMY), 1.0, 0.0)
    ally = np.where(hits == int(ERLVirtualDepthMapPointHitObjectType.OBJECT_ALLY), 1.0, 0.0)
    neutral = np.where(hits == int(ERLVirtualDepthMapPointHitObjectType.OBJECT_NEUTRAl), 1.0, 0.0)
    obstacle = np.where(hits == int(ERLVirtualDepthMapPointHitObjectType.OBJECT_OBSTACLE), 1.0, 0.0)
    obstacle_destructible = np.where(
        hits == int(ERLVirtualDepthMapPointHitObjectType.OBJECT_OBSTACLE_DESTRUCTIBLE), 1.0, 0.0
    )
    none = np.where(hits == int(ERLVirtualDepthMapPointHitObjectType.OBJECT_NONE), 1.0, 0.0)
    dist = distances / depth_map.MaxViewDistance

    return np.stack([enemy, ally, neutral, obstacle, obstacle_destructible, none, dist])


def gen_action_mask(data: RLAIData) -> np.ndarray:
    mask = np.zeros(ACTION_ID_NUM, dtype=int)

    for action in data.ValidActions:
        assert 0 <= action < ACTION_ID_NUM
        mask[action] = 1

    return mask
