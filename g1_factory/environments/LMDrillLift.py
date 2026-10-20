# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

from g1_factory.environments.base import G1FactoryEnvironmentBase


class LMDrillLift(G1FactoryEnvironmentBase):

    name: str = "LMDrillLift"

    def get_env(self, args_cli: argparse.Namespace):
        import math

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        from g1_factory.tasks.g1_locomanip_lift_task import G1LocomanipLiftTask
        from g1_factory.terms.events import set_relative_initial_pose
        from g1_factory.utils.pose import pose_range_from_quat

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -0.015), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        assets = []

        TABLE_X = 2
        TABLE_Y = 0.0

        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        assets.append(background)

        if not args_cli.table or args_cli.table.lower() == "none":
            raise ValueError(f"{self.name} requires a table asset.")
        table = self.asset_registry.get_asset_by_name(args_cli.table)()
        table.set_initial_pose(
            pose_range_from_quat(
                position_xyz_min=(TABLE_X - 0.16, TABLE_Y - 0.16, -0.8),
                position_xyz_max=(TABLE_X + 0.16, TABLE_Y + 0.16, -0.8),
                rotation_xyzw=(0.0, 0.0, 1.0, 0.0),
                yaw_jitter=0.10,
            )
        )
        assets.append(table)

        object = self.asset_registry.get_asset_by_name(args_cli.object)()
        set_relative_initial_pose(
            obj=object,
            reference=table,
            offset_range={"x": (-0.38, -0.10), "y": (-0.40, 0.40), "z": (0.80145, 0.80145)},
            rotation_xyzw=(0.0, 0.0, -0.70711, 0.70711),
            yaw_jitter=0.4,
        )
        assets.append(object)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        scene = Scene(assets=assets)

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=G1LocomanipLiftTask(
                object,
                lift_height=args_cli.lift_height,
                require_grasped=True,
            ),
            teleop_device=teleop_device,
        )

        isaaclab_arena_environment.task.robot_pose_range = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.pi / 18, math.pi / 18),
        }

        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="locomanip_power_drill", help="Object to lift")
        parser.add_argument(
            "--background",
            type=str,
            default="factory_room",
            help="Background scene (factory_room, ground_plane, kitchen, etc.)",
        )
        parser.add_argument("--table", type=str, default="locomanip_white_table", help="Required table asset")
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--embodiment", type=str, default="g1_factory_wbc_pink")
        parser.add_argument(
            "--lift_height",
            type=float,
            default=0.16,
            help="Minimum world-frame object root Z for success (meters)",
        )
