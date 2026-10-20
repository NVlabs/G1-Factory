# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_arena.utils.pose import PoseRange


def pose_range_from_quat(
    position_xyz_min: tuple[float, float, float],
    position_xyz_max: tuple[float, float, float],
    rotation_xyzw: tuple[float, float, float, float],
    yaw_jitter: float | tuple[float, float] = 0.0,
    roll_jitter: float | tuple[float, float] = 0.0,
    pitch_jitter: float | tuple[float, float] = 0.0,
) -> PoseRange:
    """Create an Arena pose range around a quaternion orientation.

    Angular jitter values are offsets in radians. A scalar produces a
    symmetric range around the center angle, while a tuple specifies the
    minimum and maximum offsets.
    """
    quaternion = torch.tensor(rotation_xyzw).unsqueeze(0)
    roll, pitch, yaw = (angle.item() for angle in euler_xyz_from_quat(quaternion))

    def offset_range(center: float, jitter: float | tuple[float, float]) -> tuple[float, float]:
        if isinstance(jitter, (list, tuple)):
            return center + jitter[0], center + jitter[1]
        return center - jitter, center + jitter

    roll_range = offset_range(roll, roll_jitter)
    pitch_range = offset_range(pitch, pitch_jitter)
    yaw_range = offset_range(yaw, yaw_jitter)

    return PoseRange(
        position_xyz_min=position_xyz_min,
        position_xyz_max=position_xyz_max,
        rpy_min=(roll_range[0], pitch_range[0], yaw_range[0]),
        rpy_max=(roll_range[1], pitch_range[1], yaw_range[1]),
    )
