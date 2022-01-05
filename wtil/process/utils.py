from typing import Any, Dict

import numpy as np
from asyncenv.api.wt.wt_pb2 import RLVector3D


def encode_ratio(ratio: float, length: int) -> np.ndarray:
    encoded = np.zeros(length)
    encoded[: int(ratio * length)] = 1.0
    return encoded


def encode_vector3d(v: RLVector3D) -> np.ndarray:
    return np.array([v.X, v.Y, v.Z])


def encode_bool(v: bool) -> np.ndarray:
    return np.array([1.0 if v else 0.0])


def encode_onehot(v: Any, d: Dict[Any, int]) -> np.ndarray:
    return np.eye(len(d))[d[v]]


def unit_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if not np.isclose(norm, 0):
        vec = vec / norm
    return vec
