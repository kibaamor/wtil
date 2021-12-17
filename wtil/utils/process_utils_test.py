import numpy as np
from assertpy import assert_that

from wtil.utils.process_utils import calc_vector2d_ratio


def test_calc_vector2d_ratio():
    top = np.array([0, 1])
    right_top = np.array([2, 2])
    right = np.array([3, 0])
    right_bottom = np.array([4, -4])
    bottom = np.array([0, -5])
    left_bottom = np.array([-6, -6])
    left = np.array([-7, 0])
    left_top = np.array([-8, 8])

    dirs = [top, right_top, right, right_bottom, bottom, left_bottom, left, left_top]
    for i in range(-4, 5):
        for si in range(len(dirs)):
            src = dirs[si]
            dst = dirs[(si + i) % len(dirs)]
            desc = f"si:{si}, i:{i}, src:{src}, dst:{dst}"
            target = i / 4.0
            if i % 4 == 0:
                target = -np.abs(target)
            assert_that(calc_vector2d_ratio(src, dst), description=desc).is_close_to(
                target, tolerance=np.finfo(float).eps
            )
