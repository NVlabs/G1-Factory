# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.g1.g1 import G1WBCJointEmbodiment as ArenaG1WBCJointEmbodiment
from isaaclab_arena.embodiments.g1.g1 import G1WBCPinkEmbodiment as ArenaG1WBCPinkEmbodiment
from isaaclab_arena.utils.pose import Pose

from g1_factory.utils.paths import asset_path

_G1_CAMERA_OFFSET = Pose(
    position_xyz=(0.085, 0.0, 0.53),
    rotation_xyzw=(-0.6272113751, 0.6272113751, -0.3265055756, 0.3265055756),
)


def _apply_factory_overrides(
    embodiment: ArenaG1WBCJointEmbodiment | ArenaG1WBCPinkEmbodiment,
    camera_offset: Pose | None,
    use_tiled_camera: bool,
) -> None:
    """Apply the robot and camera settings used by the G1 Factory tasks."""
    robot = embodiment.scene_config.robot
    robot.spawn.usd_path = asset_path("g1_robot/g1_29dof_with_hand_rev_1_0.usd")
    robot.spawn.articulation_props.enabled_self_collisions = True
    robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)

    camera = embodiment.camera_config.robot_head_cam
    embodiment.camera_config.set_use_tiled_camera(use_tiled_camera)
    if camera_offset is not None:
        camera.offset.pos = camera_offset.position_xyz
        camera.offset.rot = camera_offset.rotation_xyzw
    camera.spawn.focal_length = 15.0
    camera.spawn.horizontal_aperture = 2 * camera.spawn.focal_length * math.tan(math.radians(132.5 / 2))
    camera.spawn.vertical_aperture = None
    camera.spawn.clipping_range = (0.1, 50.0)


@register_asset
class G1WBCJointEmbodiment(ArenaG1WBCJointEmbodiment):
    """Embodiment for the G1 robot with WBC policy and direct joint upperbody control."""

    name = "g1_factory_wbc_joint"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        lock_waist: bool = False,
        camera_offset: Pose | None = _G1_CAMERA_OFFSET,
        use_tiled_camera: bool = True,
    ) -> None:
        super().__init__(
            enable_cameras=enable_cameras,
            initial_pose=initial_pose,
            lock_waist=lock_waist,
        )
        _apply_factory_overrides(self, camera_offset, use_tiled_camera)


@register_asset
class G1WBCPinkEmbodiment(ArenaG1WBCPinkEmbodiment):
    """Embodiment for the G1 robot with WBC policy and PINK IK upperbody control."""

    name = "g1_factory_wbc_pink"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        lock_waist: bool = False,
        camera_offset: Pose | None = _G1_CAMERA_OFFSET,
        use_tiled_camera: bool = False,
    ) -> None:
        super().__init__(
            enable_cameras=enable_cameras,
            initial_pose=initial_pose,
            lock_waist=lock_waist,
        )
        _apply_factory_overrides(self, camera_offset, use_tiled_camera)
