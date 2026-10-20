# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Adapted for G1 Factory from Isaac Lab Arena.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor


G1_FINGER_BODY_NAMES: list[str] = [
    "left_hand_index_0_link",
    "left_hand_index_1_link",
    "left_hand_middle_0_link",
    "left_hand_middle_1_link",
    "left_hand_thumb_0_link",
    "left_hand_thumb_1_link",
    "left_hand_thumb_2_link",
    "right_hand_index_0_link",
    "right_hand_index_1_link",
    "right_hand_middle_0_link",
    "right_hand_middle_1_link",
    "right_hand_thumb_0_link",
    "right_hand_thumb_1_link",
    "right_hand_thumb_2_link",
]


def object_near_fixed_position(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_position: tuple[float, float, float],
    max_x_separation: float,
    max_y_separation: float,
    max_z_separation: float,
) -> torch.Tensor:
    """Determine if an object is within the specified proximity of a fixed position.

    The target is relative to the environment origin; thresholds apply per axis.

    Returns:
        A boolean tensor of shape (num_envs,).
    """
    object: RigidObject = env.scene[object_cfg.name]
    object_pos = object.data.root_pos_w.torch - env.scene.env_origins

    target_pos = torch.tensor(target_position, device=env.device, dtype=torch.float32).unsqueeze(0)

    x_separation = torch.abs(object_pos[:, 0] - target_pos[:, 0])
    y_separation = torch.abs(object_pos[:, 1] - target_pos[:, 1])
    z_separation = torch.abs(object_pos[:, 2] - target_pos[:, 2])
    done = x_separation < max_x_separation
    done = torch.logical_and(done, y_separation < max_y_separation)
    done = torch.logical_and(done, z_separation < max_z_separation)

    return done


def object_lifted(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check whether the object's root height is above the minimum height.

    Args:
        env: The environment.
        minimum_height: Minimum root height in the world frame (m).
        asset_cfg: The asset configuration for the object.

    Returns:
        A boolean tensor of shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w.torch[:, 2] > minimum_height


def is_upright(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    threshold: float = 0.01,
    z_axis_up: bool = False,
) -> torch.Tensor:
    """Check whether an object is in its upright orientation.

    Uses the dot product of the object's local z-axis with world up.
    With ``z_axis_up=True``, it must be at least ``threshold``; otherwise,
    its absolute value must be at most ``threshold``.

    Args:
        env: The environment.
        asset_cfg: The asset configuration for the object.
        threshold: Use a value near zero for a horizontal local z-axis,
            or near one for an upward local z-axis.
        z_axis_up: If True, the object's local z-axis points up when upright.

    Returns:
        A boolean tensor of shape (num_envs,) indicating upright status.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w.torch
    z_dot_up = 1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)
    if z_axis_up:
        return z_dot_up >= threshold
    return torch.abs(z_dot_up) <= threshold


def is_static(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    lin_vel_threshold: float = 0.15,
    ang_vel_threshold: float = 0.15,
) -> torch.Tensor:
    """Check whether an object's linear and angular speeds are within their limits.

    Args:
        env: The environment.
        asset_cfg: The asset configuration for the object.
        lin_vel_threshold: Maximum linear velocity norm (m/s).
        ang_vel_threshold: Maximum angular velocity norm (rad/s).

    Returns:
        A boolean tensor of shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_norm = torch.norm(asset.data.root_lin_vel_w.torch, dim=-1)
    ang_vel_norm = torch.norm(asset.data.root_ang_vel_w.torch, dim=-1)
    return (lin_vel_norm <= lin_vel_threshold) & (ang_vel_norm <= ang_vel_threshold)


def is_in_contact(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    force_threshold: float = 0.95,
) -> torch.Tensor:
    """Check whether a contact sensor detects a force above the threshold.

    Uses filtered forces when available, otherwise net body forces.
    Configure a sensor filter to restrict detection to a target body.

    Args:
        env: The environment.
        contact_sensor_cfg: The contact sensor configuration.
        force_threshold: Contact force norm threshold (N).

    Returns:
        A boolean tensor of shape (num_envs,).
    """
    sensor: ContactSensor = env.unwrapped.scene[contact_sensor_cfg.name]
    fm = sensor.data.force_matrix_w
    if fm is not None:
        force_norm = torch.norm(fm.torch, dim=-1).reshape(env.unwrapped.num_envs, -1)
        return force_norm.max(dim=-1).values > force_threshold
    force_norm = torch.norm(sensor.data.net_forces_w.torch, dim=-1).reshape(env.unwrapped.num_envs, -1)
    return force_norm.max(dim=-1).values > force_threshold


def _gripper_obj_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    ee_body_name: str,
) -> torch.Tensor:
    """Compute the distance from an end-effector body to the object's root.

    Returns:
        Distance tensor of shape (num_envs,), in meters.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee_body_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_pos = robot.data.body_pos_w.torch[:, ee_body_idx, :]
    obj_pos = obj.data.root_pos_w.torch
    return torch.norm(ee_pos - obj_pos, dim=-1)


def is_gripper_far(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_names: list[str] | str | None = None,
    threshold: float = 0.4,
) -> torch.Tensor:
    """Check whether every specified end-effector exceeds the object distance threshold.

    Args:
        env: The environment.
        object_cfg: The object configuration.
        robot_cfg: The robot configuration.
        ee_body_names: End-effector body names; defaults to ``G1_FINGER_BODY_NAMES``.
        threshold: Distance threshold from the object's root (m).

    Returns:
        A boolean tensor of shape (num_envs,).
    """
    if ee_body_names is None:
        ee_body_names = G1_FINGER_BODY_NAMES
    elif isinstance(ee_body_names, str):
        ee_body_names = [ee_body_names]
    all_far = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for name in ee_body_names:
        dist = _gripper_obj_distance(env, object_cfg, robot_cfg, name)
        all_far = all_far & (dist > threshold)
    return all_far


G1_FINGER_JOINT_REGEX = ".*_hand_.*_joint"


def _fingers_closed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_joint_regex: str = G1_FINGER_JOINT_REGEX,
    min_deviation: float = 0.1,
    min_fingers_closed: int = 3,
) -> torch.Tensor:
    """Check whether enough finger joints have moved from their default positions.

    Args:
        env: The environment.
        robot_cfg: The robot configuration.
        finger_joint_regex: Regex matching finger joint names.
        min_deviation: Joint displacement threshold from the default position (rad).
        min_fingers_closed: Minimum number of joints exceeding the threshold.

    Returns:
        A boolean tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    joint_ids, _ = robot.find_joints(finger_joint_regex)
    pos = robot.data.joint_pos.torch[:, joint_ids]
    default_pos = robot.data.default_joint_pos.torch[:, joint_ids]
    deviation = torch.abs(pos - default_pos)
    closed_count = (deviation > min_deviation).sum(dim=-1)
    return closed_count >= min_fingers_closed


def is_grasped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_body_names: list[str] | str | None = None,
    proximity_threshold: float = 0.18,
    contact_sensor_cfgs: list[SceneEntityCfg] | SceneEntityCfg | None = None,
    contact_force_threshold: float = 1.0,
    require_closed_fingers: bool = True,
    finger_joint_regex: str = G1_FINGER_JOINT_REGEX,
    min_deviation: float = 0.1,
    min_fingers_closed: int = 3,
) -> torch.Tensor:
    """Estimate grasp status from finger proximity and optional contact and closure checks.

    At least one finger must be near the object's root. When sensors are provided,
    its configured sensor must also detect contact. Closure uses all joints
    matching ``finger_joint_regex``, so proximity and closure may involve different hands.

    Args:
        env: The environment.
        object_cfg: The object configuration.
        robot_cfg: The robot configuration.
        ee_body_names: End-effector body names; defaults to ``G1_FINGER_BODY_NAMES``.
        proximity_threshold: Maximum distance from a finger to the object's root (m).
        contact_sensor_cfgs: Optional shared sensor or a list with one sensor
            per entry in ``ee_body_names``, in the same order.
        contact_force_threshold: Contact force norm threshold (N).
        require_closed_fingers: Require the minimum number of closed finger joints.
        finger_joint_regex: Regex selecting the joints used to estimate closure.
        min_deviation: Joint displacement threshold from the default position (rad).
        min_fingers_closed: Minimum number of joints exceeding the threshold.

    Returns:
        A boolean tensor of shape (num_envs,).
    """
    if ee_body_names is None:
        ee_body_names = G1_FINGER_BODY_NAMES
    elif isinstance(ee_body_names, str):
        ee_body_names = [ee_body_names]
    if isinstance(contact_sensor_cfgs, SceneEntityCfg):
        contact_sensor_cfgs = [contact_sensor_cfgs] * len(ee_body_names)

    any_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for i, name in enumerate(ee_body_names):
        dist = _gripper_obj_distance(env, object_cfg, robot_cfg, name)
        gripper_close = dist <= proximity_threshold
        if contact_sensor_cfgs is not None:
            finger_contact = is_in_contact(
                env,
                contact_sensor_cfg=contact_sensor_cfgs[i],
                force_threshold=contact_force_threshold,
            )
            gripper_close = gripper_close & finger_contact
        any_grasped = any_grasped | gripper_close

    if require_closed_fingers:
        fingers_are_closed = _fingers_closed(
            env,
            robot_cfg=robot_cfg,
            finger_joint_regex=finger_joint_regex,
            min_deviation=min_deviation,
            min_fingers_closed=min_fingers_closed,
        )
        any_grasped = any_grasped & fingers_are_closed
    return any_grasped
