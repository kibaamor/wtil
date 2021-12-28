from typing import Any, Dict, List

import numpy as np
from asyncenv.api.wt.wt_pb2 import RLVector3D
from pyrr import Quaternion, Vector3


def encode_ratio(ratio: float, length: int) -> List[float]:
    """length"""
    n = int(min(ratio, 1.0) * length)
    return [1.0] * n + [0.0] * int(length - n)


def vector3d_to_list(v: RLVector3D) -> List[float]:
    """3"""
    return [v.X, v.Y, v.Z]


def encode_bool(v: bool) -> List[float]:
    """1"""
    return [1.0 if v else 0.0]


def encode_onehot(v: Any, d: Dict[Any, int]) -> List[float]:
    """len(d)"""
    return np.eye(len(d))[d[v]].tolist()


def rotation_to(src: Vector3, dst: Vector3) -> Quaternion:
    """Get a quaternion representing the rotation from src to dst.

    Args:
        src (Vector3): source direction.
        dst (Vector3): destination direction.

    Returns:
        Quaternion: the rotation from src to dst.
    """

    src = src.normalized
    dst = dst.normalized
    dot = src | dst
    if dot < -0.999999:
        tmp = Vector3([1, 0, 0]) ^ src
        if tmp.length < 0.000001:
            tmp = Vector3([0, 1, 0]) ^ src
        tmp = tmp.normalized
        return Quaternion(tmp.tolist() + [np.pi]).normalized
    elif dot > 0.999909:
        return Quaternion([0, 0, 0, 1])
    else:
        return Quaternion((src ^ dst).tolist() + [1 + dot]).normalized


def encode_vector3d(src: RLVector3D, dst: RLVector3D) -> List[float]:
    """4"""
    src = Vector3([src.X, src.Y, src.Z])
    dst = Vector3([dst.X, dst.Y, dst.Z])

    if np.isclose(src.length, 0) or np.isclose(dst.length, 0):
        quat = Quaternion([0, 0, 0, 1])
    else:
        quat = rotation_to(src, dst)
    return quat.tolist()


def decode_vector3d(src: RLVector3D, data: List[float]) -> RLVector3D:
    assert len(data) == 4
    src = Vector3([src.X, src.Y, src.Z])

    if np.isclose(src.length, 0):
        dst = Vector3([0, 0, 0])
    else:
        src = src.normalized
        quat = Quaternion(data).normalized
        dst = quat * src
    return RLVector3D(X=dst.x, Y=dst.y, Z=dst.z)
