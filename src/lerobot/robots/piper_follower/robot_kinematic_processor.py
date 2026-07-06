#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kinematic processor helpers for the Piper follower arm.

URDF joint names : joint1 ~ joint6  (no underscore, as defined in the URDF)
piper_sdk keys   : joint_1.pos ~ joint_6.pos  (with underscore)

The mapping is handled by PIPER_URDF_JOINT_NAMES / PIPER_SDK_MOTOR_NAMES below.
End-effector frame in the URDF: ``link6``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import RobotAction, RobotActionProcessorStep, TransitionKey
from lerobot.utils.rotation import Rotation

if TYPE_CHECKING:
    from lerobot.model.kinematics import RobotKinematics

# URDF joint order (matches placo joint index)
PIPER_URDF_JOINT_NAMES: list[str] = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]

# piper_sdk / lerobot motor keys (without .pos suffix)
PIPER_SDK_MOTOR_NAMES: list[str] = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]

PIPER_EE_FRAME = "link6"
PIPER_URDF_RELPATH = "assets/piper_description/urdf/piper_no_gripper_description.urdf"
PIPERX_URDF_RELPATH = "assets/piper_x_description/urdf/piper_x_description_no_gripper.urdf"
PIPER_SDK_TO_URDF_SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)

PIPER_DIFF_IK_DAMPING = 0.05
PIPER_DIFF_IK_MAX_JOINT_STEP_DEG = 8.0
PIPER_DIFF_IK_JACOBIAN_EPS_RAD = 1e-3


def piper_sdk_joints_to_urdf_joints(joints_deg: np.ndarray) -> np.ndarray:
    """Convert Piper SDK joint signs to the URDF convention used by placo."""
    joints = np.asarray(joints_deg, dtype=float).copy()
    joints[: len(PIPER_SDK_TO_URDF_SIGN)] *= PIPER_SDK_TO_URDF_SIGN
    return joints


def piper_urdf_joints_to_sdk_joints(joints_deg: np.ndarray) -> np.ndarray:
    """Convert Piper URDF joint signs back to the SDK command convention."""
    return piper_sdk_joints_to_urdf_joints(joints_deg)


def make_piper_kinematics(urdf_relpath: str = PIPER_URDF_RELPATH) -> "RobotKinematics":
    """Build a RobotKinematics instance for the Piper arm.

    Args:
        urdf_relpath: Package-relative path to the URDF (no-gripper variant).

    Returns:
        Configured RobotKinematics using placo.
    """
    from lerobot.model.kinematics import RobotKinematics

    urdf_path_obj = resources.files("lerobot").joinpath(urdf_relpath)
    if not urdf_path_obj.is_file():
        raise FileNotFoundError(
            f"Piper URDF not found at '{urdf_relpath}'. "
            "Run `git lfs pull --include='src/lerobot/assets/piper_description/**'` and retry."
        )
    return RobotKinematics(
        urdf_path=str(urdf_path_obj),
        target_frame_name=PIPER_EE_FRAME,
        joint_names=PIPER_URDF_JOINT_NAMES,
    )


def make_piperx_kinematics() -> "RobotKinematics":
    return make_piper_kinematics(urdf_relpath=PIPERX_URDF_RELPATH)


def fk_position(kinematics: "RobotKinematics", q_urdf_deg: np.ndarray) -> np.ndarray:
    return np.asarray(kinematics.forward_kinematics(q_urdf_deg)[:3, 3], dtype=float).copy()


def numerical_position_jacobian(
    kinematics: "RobotKinematics", q_urdf_deg: np.ndarray, eps_rad: float = PIPER_DIFF_IK_JACOBIAN_EPS_RAD
) -> np.ndarray:
    """Finite-difference position Jacobian in metres/radian."""
    base_pos = fk_position(kinematics, q_urdf_deg)
    jacobian = np.zeros((3, len(q_urdf_deg)), dtype=float)
    eps_deg = np.rad2deg(eps_rad)

    for i in range(len(q_urdf_deg)):
        q_perturbed = q_urdf_deg.copy()
        q_perturbed[i] += eps_deg
        jacobian[:, i] = (fk_position(kinematics, q_perturbed) - base_pos) / eps_rad

    return jacobian


def solve_dls_position_step(
    jacobian: np.ndarray,
    delta_pos: np.ndarray,
    damping: float = PIPER_DIFF_IK_DAMPING,
) -> np.ndarray:
    """Damped least-squares differential IK step in radians."""
    jj_t = jacobian @ jacobian.T
    return jacobian.T @ np.linalg.solve(
        jj_t + (damping**2) * np.eye(jj_t.shape[0], dtype=float),
        delta_pos,
    )


@dataclass
class PiperInverseKinematicsEEToJoints(RobotActionProcessorStep):
    """Piper TCP IK step using placo solver with SDK/URDF joint sign conversion built in."""

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True
    max_joint_step_deg: float = PIPER_DIFF_IK_MAX_JOINT_STEP_DEG
    position_weight: float = 1.0
    orientation_weight: float = 0.01

    def action(self, action: RobotAction) -> RobotAction:
        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Joints observation is required for computing Piper kinematics")

        arm_motor_names = [name for name in self.motor_names if name != "gripper"]
        q_sdk = np.array([float(observation[f"{name}.pos"]) for name in arm_motor_names], dtype=float)
        q_urdf = piper_sdk_joints_to_urdf_joints(q_sdk)

        if self.q_curr is None:
            self.q_curr = q_urdf

        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        q_init = q_urdf if self.initial_guess_current_joints else self.q_curr
        q_target_urdf = self.kinematics.inverse_kinematics(
            current_joint_pos=q_init,
            desired_ee_pose=t_des,
            position_weight=self.position_weight,
            orientation_weight=self.orientation_weight,
        )

        # Safety clip: limit per-step joint movement
        dq = np.clip(q_target_urdf - self.q_curr, -self.max_joint_step_deg, self.max_joint_step_deg)
        q_target_urdf = self.q_curr + dq

        self.q_curr = q_target_urdf
        q_target_sdk = piper_urdf_joints_to_sdk_joints(q_target_urdf)

        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target_sdk[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION].pop(f"ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        self.q_curr = None
