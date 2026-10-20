# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.assets.register import register_retargeter
from isaaclab_arena.assets.retargeter_library import (
    G1WbcPinkIsaacTeleopRetargeter as ArenaG1WbcPinkIsaacTeleopRetargeter,
)

from g1_factory.embodiments.g1 import G1WBCPinkEmbodiment


@register_retargeter
class G1WbcPinkIsaacTeleopRetargeter(ArenaG1WbcPinkIsaacTeleopRetargeter):
    """Isaac Teleop pipeline builder for the G1 Factory WBC Pink embodiment."""

    embodiment = G1WBCPinkEmbodiment.name
