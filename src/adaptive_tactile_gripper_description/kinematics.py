"""曲柄滑块机构的解析位置与闭合雅可比。"""

from __future__ import annotations

import math

CRANK_RADIUS_M = 0.025
LINK_LENGTH_M = 0.04353553390593274
OFFSET_M = 0.017677669529663688
THETA0_RAD = math.pi / 4.0
POSITION_MIN_RAD = 0.0
POSITION_MAX_RAD = 1.7
PROTOCOL_RANGE_RAD = (0.0, 1.7)


def _rod_projection(q_rad: float) -> tuple[float, float, float]:
    """返回曲柄角、连杆的侧向偏移和沿导轨的投影长度。"""
    alpha = THETA0_RAD + q_rad
    offset = CRANK_RADIUS_M * math.sin(alpha) - OFFSET_M
    return alpha, offset, math.sqrt(LINK_LENGTH_M**2 - offset**2)


def single_finger_closure(q_rad: float) -> float:
    """返回相对全开位置的单指闭合位移，单位为米。"""
    alpha, _, projection = _rod_projection(q_rad)
    return CRANK_RADIUS_M * (math.cos(THETA0_RAD) - math.cos(alpha)) + LINK_LENGTH_M - projection


SINGLE_FINGER_STROKE_M = single_finger_closure(POSITION_MAX_RAD)
TOTAL_STROKE_M = 2 * SINGLE_FINGER_STROKE_M


def drive_angle_for_closure(closure_m: float) -> float:
    """由单指闭合位移求机构范围内的曲柄角，单位为弧度。"""
    if not 0.0 <= closure_m <= SINGLE_FINGER_STROKE_M:
        raise ValueError(f"单指闭合位移超出机构范围：{closure_m} m")
    if closure_m == 0.0:
        return POSITION_MIN_RAD
    if closure_m == SINGLE_FINGER_STROKE_M:
        return POSITION_MAX_RAD
    lo, hi = POSITION_MIN_RAD, POSITION_MAX_RAD
    for _ in range(52):
        mid = (lo + hi) / 2
        if single_finger_closure(mid) < closure_m:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def get_joint_positions(q_drive_rad: float, *, q_backlash_rad: float = 0.0) -> dict[str, float]:
    """返回同一构型的 MJCF 关节位置，键名不含命名空间。

    URDF 的驱动关节取两角之和，左右销轴和滑块可直接使用返回值。
    滑块单位为米，其余关节单位为弧度。默认回差角为零。
    """
    q = q_drive_rad + q_backlash_rad
    if not POSITION_MIN_RAD <= q <= POSITION_MAX_RAD:
        raise ValueError(f"实际曲柄角超出机构范围：{q} rad")
    _, offset, projection = _rod_projection(q)
    pin = q + math.atan2(offset, projection)
    slide = single_finger_closure(q)
    return {
        "gripper_drive": q_drive_rad,
        "gripper_backlash": q_backlash_rad,
        "left_link_crank_pin": pin,
        "right_link_crank_pin": pin,
        "left_finger_slide": slide,
        "right_finger_slide": slide,
    }


def get_closure_jacobian(q_rad: float) -> float:
    """返回双指总闭合位移对实际曲柄角的导数，单位为米每弧度。"""
    alpha, offset, projection = _rod_projection(q_rad)
    return 2 * CRANK_RADIUS_M * (math.sin(alpha) + offset * math.cos(alpha) / projection)
