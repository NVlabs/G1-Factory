# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

from g1_factory.environments.base import G1FactoryEnvironmentBase


class LMDrillPnP(G1FactoryEnvironmentBase):

    name: str = "LMDrillPnP"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        from g1_factory.tasks.g1_locomanip_pnp_task import G1LocomanipPnPTask
        from g1_factory.terms.events import set_relative_initial_pose
        from g1_factory.utils.pose import pose_range_from_quat

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -0.015), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        assets = []

        TABLE_X = 1.2
        TABLE_Y = 0.0

        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        assets.append(background)

        table_right = self.asset_registry.get_asset_by_name(args_cli.table_right)()
        table_right.name = "table_right"
        table_right.prim_path = "{ENV_REGEX_NS}/table_right"
        table_right.scale = (0.8, 1.0, 1.0)
        table_right.object_cfg = table_right._init_object_cfg()
        table_right.set_initial_pose(
            pose_range_from_quat(
                position_xyz_min=(TABLE_X - 0.12, TABLE_Y - 0.12, -0.8),
                position_xyz_max=(TABLE_X + 0.12, TABLE_Y + 0.12, -0.8),
                rotation_xyzw=(0.0, 0.0, 1.0, 0.0),
                yaw_jitter=0.12,
            )
        )
        assets.append(table_right)

        object = self.asset_registry.get_asset_by_name(args_cli.object)()
        set_relative_initial_pose(
            obj=object,
            reference=table_right,
            offset_range={"x": (-0.30, -0.14), "y": (-0.30, 0.30), "z": (0.8, 0.8)},
            rotation_xyzw=(0.0, 0.0, -0.70711, 0.70711),
            yaw_jitter=0.3,
        )
        assets.append(object)

        table_left = self.asset_registry.get_asset_by_name(args_cli.table_left)()
        table_left.name = "table_left"
        table_left.prim_path = "{ENV_REGEX_NS}/table_left"
        table_left.scale = (0.8, 1.0, 1.0)
        table_left.object_cfg = table_left._init_object_cfg()
        table_left.set_initial_pose(
            pose_range_from_quat(
                position_xyz_min=(TABLE_Y - 0.12, TABLE_X - 0.12, -0.8),
                position_xyz_max=(TABLE_Y + 0.12, TABLE_X + 0.12, -0.8),
                rotation_xyzw=(0, 0, -0.70711, 0.70711),
                yaw_jitter=0.12,
            )
        )
        assets.append(table_left)

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
                table_left,
                max_x_separation=0.3,
                max_y_separation=0.27,
                max_z_separation=0.81,
                gripper_far_threshold=0.1,
                upright_threshold=0.8,
                upright_z_axis_up=True,
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
        parser.add_argument(
            "--object", type=str, default="locomanip_power_drill", help="Object to transfer between tables"
        )
        parser.add_argument(
            "--background",
            type=str,
            default="factory_room",
            help="Background scene (factory_room, ground_plane, kitchen, etc.)",
        )
        parser.add_argument("--table_right", type=str, default="locomanip_white_table", help="Source table asset")
        parser.add_argument("--table_left", type=str, default="locomanip_white_table", help="Destination table asset")
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--embodiment", type=str, default="g1_factory_wbc_pink")
