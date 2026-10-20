# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.tasks.press_button_task import PressButtonTask as ArenaPressButtonTask
from isaaclab_arena.tasks.press_button_task import PressEventCfg as ArenaPressEventCfg
from isaaclab_arena.tasks.press_button_task import TerminationsCfg

from g1_factory.terms.terminations import is_gripper_far


class PressButtonTask(ArenaPressButtonTask):
    """Button-press task with robot pose randomization and optional finger proximity.

    When set, ``gripper_far_threshold`` requires at least one finger to be
    within that distance of the button root when the button is pressed.
    """

    def __init__(
        self,
        pressable_object: Pressable,
        pressedness_threshold: float | None = None,
        reset_pressedness: float | None = None,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        robot_pose_range: dict[str, tuple[float, float]] | None = None,
        gripper_far_threshold: float | None = None,
    ):
        super().__init__(
            pressable_object=pressable_object,
            pressedness_threshold=pressedness_threshold,
            reset_pressedness=reset_pressedness,
            episode_length_s=episode_length_s,
            task_description=task_description,
        )
        self.task_description = task_description
        self.robot_pose_range = robot_pose_range
        self.gripper_far_threshold = gripper_far_threshold

    def get_termination_cfg(self):
        press_params = {}
        if self.pressedness_threshold is not None:
            press_params["pressedness_threshold"] = self.pressedness_threshold

        if self.gripper_far_threshold is not None:
            is_pressed_fn = self.pressable_object.is_pressed
            is_pressed_params = press_params
            pressable_object_name = self.pressable_object.name
            gripper_distance_threshold = self.gripper_far_threshold

            def _success(env):
                pressed = is_pressed_fn(env, **is_pressed_params)
                gripper_near = ~is_gripper_far(
                    env,
                    object_cfg=SceneEntityCfg(pressable_object_name),
                    threshold=gripper_distance_threshold,
                )
                return pressed & gripper_near

            success = TerminationTermCfg(func=_success, params={})
        else:
            success = TerminationTermCfg(
                func=self.pressable_object.is_pressed,
                params=press_params,
            )
        return TerminationsCfg(success=success)

    def get_events_cfg(self):
        return PressEventCfg(
            self.pressable_object,
            reset_pressedness=self.reset_pressedness,
            robot_pose_range=self.robot_pose_range,
        )


@configclass
class PressEventCfg(ArenaPressEventCfg):
    """Configuration for button and robot reset events."""

    reset_robot_pose: EventTermCfg | None = None

    def __init__(
        self,
        pressable_object: Pressable,
        reset_pressedness: float | None,
        robot_pose_range: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__(pressable_object=pressable_object, reset_pressedness=reset_pressedness)
        self.reset_robot_pose = None
        if robot_pose_range is not None:
            self.reset_robot_pose = EventTermCfg(
                func=mdp_isaac_lab.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": robot_pose_range,
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            )
