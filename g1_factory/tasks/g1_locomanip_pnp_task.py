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
from isaaclab_arena.tasks.predicates.spatial import objects_in_proximity
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena.utils.pose import Pose, PoseRange

from g1_factory.terms.events import reset_all_distractors_uniform
from g1_factory.terms.terminations import (
    is_gripper_far,
    is_in_contact,
    is_static,
    is_upright,
    object_near_fixed_position,
)

_DEFAULT_POSE_RANGE = {
    "x": (-0.0, 0.0),
    "y": (-0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}


def _is_physics_object(obj: Asset) -> bool:
    return isinstance(obj, ObjectBase) and obj.object_type in (ObjectType.RIGID, ObjectType.ARTICULATION)


class G1LocomanipPnPTask(TaskBase):
    """G1 pick-and-place task with configurable success criteria and reset randomization."""

    def __init__(
        self,
        object: Asset,
        destination: Asset,
        drop_height: float | None = -0.6,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        robot_pose_range: dict[str, tuple[float, float]] | None = None,
        max_x_separation: float = 0.1,
        max_y_separation: float = 0.1,
        max_z_separation: float = 0.859,
        distractors: list | None = None,
        require_static: bool = False,
        require_contact: bool = False,
        upright_threshold: float | None = None,
        upright_z_axis_up: bool = False,
        gripper_far_threshold: float | None = None,
        gripper_near_threshold: float | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.object = object
        self.destination = destination
        self.drop_height = drop_height
        self.task_description = task_description
        self.robot_pose_range = robot_pose_range
        self.max_x_separation = max_x_separation
        self.max_y_separation = max_y_separation
        self.max_z_separation = max_z_separation
        self.distractors = distractors if distractors is not None else []
        self.require_static = require_static
        self.require_contact = require_contact
        self.upright_threshold = upright_threshold
        self.upright_z_axis_up = upright_z_axis_up
        self.gripper_far_threshold = gripper_far_threshold
        self.gripper_near_threshold = gripper_near_threshold
        self._contact_sensor_name: str | None = None
        self._contact_sensor_cfg = None
        if require_contact and _is_physics_object(object) and isinstance(destination, ObjectBase):
            self._contact_sensor_name = f"contact_sensor_{object.name}"
            self._contact_sensor_cfg = object.get_contact_sensor_cfg(contact_against_object=destination)

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
            destination=self.destination,
            drop_height=self.drop_height,
            max_x_separation=self.max_x_separation,
            max_y_separation=self.max_y_separation,
            max_z_separation=self.max_z_separation,
            require_static=self.require_static,
            contact_sensor_name=self._contact_sensor_name,
            upright_threshold=self.upright_threshold,
            upright_z_axis_up=self.upright_z_axis_up,
            gripper_far_threshold=self.gripper_far_threshold,
            gripper_near_threshold=self.gripper_near_threshold,
        )

    def get_events_cfg(self):
        return EventCfg(
            object=self.object,
            distractors=self.distractors,
            robot_pose_range=self.robot_pose_range,
        )

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
        destination: Asset,
        drop_height: float | None = -0.05,
        max_x_separation: float = 0.1,
        max_y_separation: float = 0.1,
        max_z_separation: float = 0.859,
        require_static: bool = False,
        contact_sensor_name: str | None = None,
        upright_threshold: float | None = None,
        upright_z_axis_up: bool = False,
        gripper_far_threshold: float | None = None,
        gripper_near_threshold: float | None = None,
    ):
        self.object = object
        self.destination = destination
        self.time_out = TerminationTermCfg(func=mdp_isaac_lab.time_out)
        self.object_dropping = None
        if drop_height is not None:
            if _is_physics_object(object):
                self.object_dropping = TerminationTermCfg(
                    func=mdp_isaac_lab.root_height_below_minimum,
                    params={"minimum_height": drop_height, "asset_cfg": SceneEntityCfg(object.name)},
                )

        extra_checks: list[tuple[Callable, dict]] = []
        if _is_physics_object(object):
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
        if contact_sensor_name is not None:
            extra_checks.append((
                is_in_contact,
                {
                    "contact_sensor_cfg": SceneEntityCfg(contact_sensor_name),
                },
            ))

        if gripper_far_threshold is not None:
            extra_checks.append((
                is_gripper_far,
                {
                    "object_cfg": SceneEntityCfg(object.name),
                    "threshold": gripper_far_threshold,
                },
            ))

        if gripper_near_threshold is not None:
            gripper_near_distance_threshold = gripper_near_threshold

            def _gripper_near(env):
                return ~is_gripper_far(
                    env,
                    object_cfg=SceneEntityCfg(object.name),
                    threshold=gripper_near_distance_threshold,
                )

            extra_checks.append((_gripper_near, {}))

        placement_check_fn = None
        placement_check_params: dict = {}

        if _is_physics_object(object) and _is_physics_object(destination):
            placement_check_fn = objects_in_proximity
            placement_check_params = {
                "object_cfg": SceneEntityCfg(self.object.name),
                "target_object_cfg": SceneEntityCfg(self.destination.name),
                "max_x_separation": max_x_separation,
                "max_y_separation": max_y_separation,
                "max_z_separation": max_z_separation,
            }
        elif _is_physics_object(object) and isinstance(destination, ObjectBase):
            dest_initial = destination.get_initial_pose()
            dest_pose = dest_initial.get_midpoint() if isinstance(dest_initial, PoseRange) else dest_initial
            if isinstance(dest_pose, Pose):
                placement_check_fn = object_near_fixed_position
                placement_check_params = {
                    "object_cfg": SceneEntityCfg(self.object.name),
                    "target_position": dest_pose.position_xyz,
                    "max_x_separation": max_x_separation,
                    "max_y_separation": max_y_separation,
                    "max_z_separation": max_z_separation,
                }

        if placement_check_fn is not None and extra_checks:
            success_checks = extra_checks

            def _success(env):
                result = placement_check_fn(env, **placement_check_params)
                for check_fn, check_params in success_checks:
                    result = result & check_fn(env, **check_params)
                return result

            self.success = TerminationTermCfg(func=_success, params={})
        elif placement_check_fn is not None:
            self.success = TerminationTermCfg(func=placement_check_fn, params=placement_check_params)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all: EventTermCfg = MISSING
    reset_object_position: EventTermCfg | None = None
    reset_distractors_position: EventTermCfg | None = None
    reset_robot_pose: EventTermCfg | None = None

    def __init__(
        self,
        object: Asset,
        distractors: list | None = None,
        pose_range: dict[str, tuple[float, float]] | None = None,
        robot_pose_range: dict[str, tuple[float, float]] | None = None,
    ):
        self.reset_all = EventTermCfg(func=mdp_isaac_lab.reset_scene_to_default, mode="reset")
        self.reset_object_position = None
        self.reset_distractors_position = None
        if pose_range is None:
            pose_range = _DEFAULT_POSE_RANGE
        if robot_pose_range is None:
            robot_pose_range = _DEFAULT_POSE_RANGE

        object_event_cfg = object.get_event_cfg()[1] if isinstance(object, ObjectBase) else None
        has_own_randomization = isinstance(object, ObjectBase) and (
            isinstance(object.get_initial_pose(), PoseRange)
            or (object_event_cfg is not None and object_event_cfg.func is not set_object_pose)
        )
        if _is_physics_object(object) and not has_own_randomization:
            self.reset_object_position = EventTermCfg(
                func=mdp_isaac_lab.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": pose_range,
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg(object.name),
                },
            )

        if distractors is not None and len(distractors) > 0:
            distractor_names = [d.name for d in distractors]
            self.reset_distractors_position = EventTermCfg(
                func=reset_all_distractors_uniform,
                mode="reset",
                params={
                    "distractor_names": distractor_names,
                    "pose_range": pose_range,
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
