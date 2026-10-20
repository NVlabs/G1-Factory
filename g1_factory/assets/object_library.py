# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

from g1_factory.utils.paths import asset_path


@register_asset
class LocomanipWhiteTable(LibraryObject):
    """Worktable for loco-manipulation tasks."""

    name = "locomanip_white_table"
    tags = ["object"]
    usd_path = asset_path("factory_ergo_table/factory_ergo_table_01.usd")


@register_asset
class LocomanipPowerDrill(LibraryObject):
    """Power drill for loco-manipulation tasks."""

    name = "locomanip_power_drill"
    tags = ["object"]
    usd_path = asset_path("power_drill_01/sm_powerdrill_b01.usd")


@register_asset
class LocomanipPowerDrillHolderTask(LibraryObject):
    """Power drill for holder tasks."""

    name = "locomanip_power_drill_holder_task"
    tags = ["object"]
    usd_path = asset_path("power_drill_02/sm_powerdrill_b01.usd")


@register_asset
class LocomanipIndustrialShelf(LibraryObject):
    """Industrial storage shelf."""

    name = "locomanip_industrial_shelf"
    tags = ["object"]
    usd_path = asset_path("industrial_steel_shelving/openindustrialsteelshelving_a12.usd")


@register_asset
class LocomanipCardBox(LibraryObject):
    """Cardboard box for loco-manipulation tasks."""

    name = "locomanip_cardbox"
    tags = ["object"]
    usd_path = asset_path("cardboard_box/cardbox_a1_01.usd")
    scale = (0.8, 0.8, 0.8)


@register_asset
class LocomanipLongBox(LibraryObject):
    """Long cardboard box for loco-manipulation tasks."""

    name = "locomanip_longbox"
    tags = ["object"]
    usd_path = asset_path("long_box/longbox_a08.usd")


@register_asset
class LocomanipPowerDrillHolder(LibraryObject):
    """Power drill holder."""

    name = "locomanip_powerdrill_holder"
    tags = ["object"]
    usd_path = asset_path("power_drill_holder/powerdrill_holder.usd")


@register_asset
class LocomanipMobileCart(LibraryObject):
    """Articulated mobile shelving cart."""

    name = "locomanip_mobile_cart"
    tags = ["object"]
    usd_path = asset_path("mobile_shelving_cart/sm_mobileshelvingcart_a01.usd")
    object_type: ObjectType = ObjectType.ARTICULATION

    spawn_cfg_addon = {
        "articulation_props": sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
        ),
    }


@register_asset
class LocomanipTargetZone(LibraryObject):
    """Target zone."""

    name = "locomanip_target_zone"
    tags = ["object"]
    usd_path = asset_path("target_zone/target_zone.usd")


@register_asset
class LocomanipButton(LibraryObject, Pressable):
    """Pressable button mounted on the control box."""

    name = "locomanip_button"
    tags = ["object", "pressable"]
    usd_path = asset_path("control_box/button.usd")
    object_type: ObjectType = ObjectType.ARTICULATION

    pressable_joint_name = "button_joint"
    pressedness_threshold = 0.5

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(
            instance_name=instance_name,
            prim_path=prim_path,
            initial_pose=initial_pose,
            pressable_joint_name=self.pressable_joint_name,
            pressedness_threshold=self.pressedness_threshold,
        )


@register_asset
class LocomanipControlBox(LibraryObject):
    """Industrial control station."""

    name = "locomanip_control_box"
    tags = ["object"]
    usd_path = asset_path("control_box/control_box.usd")
