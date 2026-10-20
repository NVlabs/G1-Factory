# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

from g1_factory.utils.paths import asset_path


@register_asset
class FactoryBackground(LibraryBackground):
    """Factory scene for the G1 Factory benchmark."""

    name = "factory_room"
    tags = ["background"]
    usd_path = asset_path("factory/gear_factory.usdz")
    initial_pose = Pose(position_xyz=(0, 0, -0.800825570380548))
    object_min_z = -0.2
