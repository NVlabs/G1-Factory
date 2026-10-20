# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class G1FactoryEnvironmentBase(ExampleEnvironmentBase):
    """Base class that registers external G1 Factory components."""

    def __init__(self) -> None:
        super().__init__()

        import g1_factory.assets.background_library  # noqa: F401
        import g1_factory.assets.object_library  # noqa: F401
        import g1_factory.embodiments.g1  # noqa: F401
        import g1_factory.teleop.retargeter  # noqa: F401
