from typing import Any, Dict, List

import numpy as np

from wtil.api.wt_pb2 import RLVector3D


def encode_ratio(ratio: float, length: int) -> List[float]:
    """length"""
    n = int(min(ratio, 1.0) * length)
    return [1.0] * n + [0.0] * int(length - n)


def encode_vector3d(v: RLVector3D) -> List[float]:
    """3"""
    return [v.X, v.Y, v.Z]


def encode_bool(v: bool) -> List[float]:
    """1"""
    return [1.0 if v else 0.0]


def encode_onehot(v: Any, d: Dict[Any, int]) -> List[float]:
    """len(d)"""
    return np.eye(len(d))[d[v]].tolist()


def calc_vector2d_ratio(src: np.ndarray, dst: np.ndarray) -> float:
    assert src.size == 2
    assert dst.size == 2
    return -np.arctan2(np.cross(src, dst), np.dot(src, dst)) / np.pi


def encode_vector3d_ratio(src: RLVector3D, dst: RLVector3D) -> List[float]:
    """2"""
    horizonSrc = np.array([src.X, src.Y])
    horizonDst = np.array([dst.X, dst.Y])
    verticalSrc = np.array([src.X, src.Z])
    verticalDst = np.array([dst.X, dst.Z])
    return [calc_vector2d_ratio(horizonSrc, horizonDst), calc_vector2d_ratio(verticalSrc, verticalDst)]
