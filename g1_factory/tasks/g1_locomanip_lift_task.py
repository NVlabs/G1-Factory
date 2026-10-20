# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from collections.abc import Callable
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena.utils.pose import PoseRange

from g1_factory.terms.terminations import (
    is_grasped,
    is_gripper_far,
    is_in_contact,
    is_static,
    is_upright,
    object_lifted,
)

_DEFAULT_POSE_RANGE = {
    "x": (-0.0, 0.0),
    "y": (-0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}


class G1LocomanipLiftTask(TaskBase):
    """Lift an object above an absolute world-frame root Z threshold.

    ``lift_height`` sets the threshold. When set, ``gripper_far_threshold``
    requires at least one finger within that distance of the object root.
    """

    def __init__(
        self,
        object: Asset,
        lift_height: float = 0.15,
        drop_height: float | None = None,
        episode_length_s: float | None = None,
        pose_range: dict[str, tuple[float, float]] | None = None,
        task_description: str | None = None,
        robot_pose_range: dict[str, tuple[float, float]] | None = None,
        require_static: bool = False,
        upright_threshold: float | None = None,
        upright_z_axis_up: bool = False,
        gripper_far_threshold: float | None = None,
        require_grasped: bool = False,
        not_in_contact_with: Asset | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.object = object
        self.lift_height = lift_height
        self.drop_height = drop_height
        self.pose_range = pose_range if pose_range is not None else _DEFAULT_POSE_RANGE
        self.task_description = task_description
        self.robot_pose_range = robot_pose_range
        self.require_static = require_static
        self.upright_threshold = upright_threshold
        self.upright_z_axis_up = upright_z_axis_up
        self.gripper_far_threshold = gripper_far_threshold
        self.require_grasped = require_grasped
        self.not_in_contact_with = not_in_contact_with

        self._contact_sensor_name: str | None = None
        self._contact_sensor_cfg = None
        if (
            not_in_contact_with is not None
            and isinstance(object, ObjectBase)
            and isinstance(not_in_contact_with, ObjectBase)
        ):
            self._contact_sensor_name = f"contact_sensor_{object.name}"
            self._contact_sensor_cfg = object.get_contact_sensor_cfg(contact_against_object=not_in_contact_with)

    def get_scene_cfg(self):
        if self._contact_sensor_cfg is not None:
            SceneCfg = make_configclass(
                "SceneCfg",
                [(self._contact_sensor_name, type(self._contact_sensor_cfg), self._contact_sensor_cfg)],
            )
            return SceneCfg()
        return None

    def get_termination_cfg(self):
        return TerminationsCfg(
            object=self.object,
            lift_height=self.lift_height,
            drop_height=self.drop_height,
            require_static=self.require_static,
            upright_threshold=self.upright_threshold,
            upright_z_axis_up=self.upright_z_axis_up,
            gripper_far_threshold=self.gripper_far_threshold,
            require_grasped=self.require_grasped,
            not_in_contact_sensor_name=self._contact_sensor_name,
        )

    def get_events_cfg(self):
        return EventCfg(object=self.object, pose_range=self.pose_range, robot_pose_range=self.robot_pose_range)

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("Function not implemented yet.")

    def get_metrics(self):
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.object,
            offset=np.array([-1.3, 1.7, 1.5]),
        )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = MISSING
    object_dropping: TerminationTermCfg | None = None
    success: TerminationTermCfg = MISSING

    def __init__(
        self,
        object: Asset,
        lift_height: float = 0.15,
        drop_height: float | None = -0.05,
        require_static: bool = False,
        upright_threshold: float | None = None,
        upright_z_axis_up: bool = False,
        gripper_far_threshold: float | None = None,
        require_grasped: bool = False,
        not_in_contact_sensor_name: str | None = None,
    ):
        self.time_out = TerminationTermCfg(func=mdp_isaac_lab.time_out)
        self.object_dropping = None

        if drop_height is not None:
            self.object_dropping = TerminationTermCfg(
                func=mdp_isaac_lab.root_height_below_minimum,
                params={"minimum_height": drop_height, "asset_cfg": SceneEntityCfg(object.name)},
            )

        extra_checks: list[tuple[Callable, dict]] = []
        if require_static:
            extra_checks.append((
                is_static,
                {
                    "asset_cfg": SceneEntityCfg(object.name),
                },
            ))
        if upright_threshold is not None:
            extra_checks.append((
                is_upright,
                {
                    "asset_cfg": SceneEntityCfg(object.name),
                    "threshold": upright_threshold,
                    "z_axis_up": upright_z_axis_up,
                },
            ))

        if gripper_far_threshold is not None:
            gripper_distance_threshold = gripper_far_threshold

            def _gripper_near(env):
                return ~is_gripper_far(
                    env,
                    object_cfg=SceneEntityCfg(object.name),
                    threshold=gripper_distance_threshold,
                )

            extra_checks.append((_gripper_near, {}))

        if require_grasped:
            extra_checks.append((
                is_grasped,
                {
                    "object_cfg": SceneEntityCfg(object.name),
                },
            ))

        if not_in_contact_sensor_name is not None:
            excluded_contact_sensor_name = not_in_contact_sensor_name

            def _not_in_contact(env):
                return ~is_in_contact(
                    env,
                    contact_sensor_cfg=SceneEntityCfg(excluded_contact_sensor_name),
                )

            extra_checks.append((_not_in_contact, {}))

        lift_check_params = {"minimum_height": lift_height, "asset_cfg": SceneEntityCfg(object.name)}

        if extra_checks:
            success_checks = extra_checks

            def _success(env):
                result = object_lifted(env, **lift_check_params)
                for check_fn, check_params in success_checks:
                    result = result & check_fn(env, **check_params)
                return result

            self.success = TerminationTermCfg(func=_success, params={})
        else:
            self.success = TerminationTermCfg(func=object_lifted, params=lift_check_params)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all: EventTermCfg = MISSING
    reset_object_position: EventTermCfg | None = None
    reset_robot_pose: EventTermCfg | None = None

    def __init__(
        self,
        object: Asset,
        pose_range: dict[str, tuple[float, float]] | None = None,
        robot_pose_range: dict[str, tuple[float, float]] | None = None,
    ):
        if pose_range is None:
            pose_range = _DEFAULT_POSE_RANGE
        if robot_pose_range is None:
            robot_pose_range = _DEFAULT_POSE_RANGE

        self.reset_all = EventTermCfg(func=mdp_isaac_lab.reset_scene_to_default, mode="reset")
        self.reset_object_position = None

        is_physics_object = isinstance(object, ObjectBase) and object.object_type in (
            ObjectType.RIGID,
            ObjectType.ARTICULATION,
        )
        object_event_cfg = object.get_event_cfg()[1] if isinstance(object, ObjectBase) else None
        # Preserve asset-level randomization; otherwise apply the task pose range.
        has_own_randomization = isinstance(object, ObjectBase) and (
            isinstance(object.get_initial_pose(), PoseRange)
            or (object_event_cfg is not None and object_event_cfg.func is not set_object_pose)
        )
        if is_physics_object and not has_own_randomization:
            self.reset_object_position = EventTermCfg(
                func=mdp_isaac_lab.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": pose_range,
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg(object.name),
                },
            )

        self.reset_robot_pose = EventTermCfg(
            func=mdp_isaac_lab.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": robot_pose_range,
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
