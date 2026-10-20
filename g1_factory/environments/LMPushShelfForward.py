# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

from g1_factory.environments.base import G1FactoryEnvironmentBase


class LMPushShelfForward(G1FactoryEnvironmentBase):

    name: str = "LMPushShelfForward"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        from g1_factory.tasks.g1_locomanip_pnp_task import G1LocomanipPnPTask
        from g1_factory.utils.pose import pose_range_from_quat

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -0.015), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        assets = []

        TARGET_X = 1.5
        TARGET_Y = 0.0

        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        background.set_initial_pose(Pose(position_xyz=(2.0, 0.0, background.get_object_cfg()[1].init_state.pos[2])))
        assets.append(background)

        object = self.asset_registry.get_asset_by_name(args_cli.object)()
        object.set_initial_pose(
            pose_range_from_quat(
                position_xyz_min=(TARGET_X - 0.3, TARGET_Y - 0.3, -0.8),
                position_xyz_max=(TARGET_X + 0.3, TARGET_Y + 0.3, -0.8),
                rotation_xyzw=(0.0, 0.0, -0.70711, 0.70711),
                yaw_jitter=0.06,
            )
        )
        assets.append(object)

        target_zone = self.asset_registry.get_asset_by_name("locomanip_target_zone")()
        target_zone.set_initial_pose(
            pose_range_from_quat(
                position_xyz_min=(TARGET_X * 2.87 - 0.1, TARGET_Y - 0.1, -0.8),
                position_xyz_max=(TARGET_X * 2.87 + 0.1, TARGET_Y + 0.1, -0.8),
                rotation_xyzw=(0, 0, 0, 1),
                yaw_jitter=0.02,
            )
        )
        assets.append(target_zone)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        scene = Scene(assets=assets)

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=G1LocomanipPnPTask(
                object,
                target_zone,
                drop_height=None,
                max_x_separation=0.3,
                max_y_separation=0.3,
                max_z_separation=0.01,
                upright_threshold=0.9,
                upright_z_axis_up=True,
                gripper_near_threshold=1.1,
                require_static=True,
            ),
            teleop_device=teleop_device,
        )

        isaaclab_arena_environment.task.robot_pose_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="locomanip_mobile_cart", help="Object to push")
        parser.add_argument(
            "--background",
            type=str,
            default="factory_room",
            help="Background scene (factory_room, ground_plane, kitchen, etc.)",
        )
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--embodiment", type=str, default="g1_factory_wbc_pink")
