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

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PiperForwardKinematicsPose:
    x_mm: float
    y_mm: float
    z_mm: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x_mm, self.y_mm, self.z_mm, self.roll_deg, self.pitch_deg, self.yaw_deg],
            dtype=float,
        )


class PiperForwardKinematics:
    """Forward kinematics helper that matches the Piper SDK DH convention."""

    def __init__(self, dh_is_offset: bool = True):
        self._pi = math.pi
        self._rad_to_deg = 180.0 / self._pi

        self._a = np.array([0.0, 0.0, 285.03, -21.98, 0.0, 0.0], dtype=float)
        self._alpha = np.array(
            [0.0, -self._pi / 2.0, 0.0, self._pi / 2.0, -self._pi / 2.0, self._pi / 2.0],
            dtype=float,
        )
        self._theta = np.array(
            [0.0, -self._pi * 174.22 / 180.0, -100.78 / 180.0 * self._pi, 0.0, 0.0, 0.0],
            dtype=float,
        )
        self._d = np.array([123.0, 0.0, 0.0, 250.75, 0.0, 91.0], dtype=float)
        self.init_pose = PiperForwardKinematicsPose(55.0, 0.0, 205.0, 0.0, 85.0, 0.0)

        if dh_is_offset:
            self._theta = np.array(
                [0.0, -self._pi * 172.22 / 180.0, -102.78 / 180.0 * self._pi, 0.0, 0.0, 0.0],
                dtype=float,
            )
            self.init_pose = PiperForwardKinematicsPose(56.128, 0.0, 213.266, 0.0, 85.0, 0.0)

    def _link_transformation(self, alpha: float, a: float, theta: float, d: float) -> np.ndarray:
        calpha = math.cos(alpha)
        salpha = math.sin(alpha)
        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        return np.array(
            [
                [ctheta, -stheta, 0.0, a],
                [stheta * calpha, ctheta * calpha, -salpha, -salpha * d],
                [stheta * salpha, ctheta * salpha, calpha, calpha * d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _matrix_to_pose(self, transform: np.ndarray) -> PiperForwardKinematicsPose:
        x_mm = float(transform[0, 3])
        y_mm = float(transform[1, 3])
        z_mm = float(transform[2, 3])

        if transform[2, 0] < -1.0 + 1e-4:
            pitch_deg = self._pi / 2.0 * self._rad_to_deg
            yaw_deg = 0.0
            roll_deg = math.atan2(transform[0, 1], transform[1, 1]) * self._rad_to_deg
        elif transform[2, 0] > 1.0 - 1e-4:
            pitch_deg = -self._pi / 2.0 * self._rad_to_deg
            yaw_deg = 0.0
            roll_deg = -math.atan2(transform[0, 1], transform[1, 1]) * self._rad_to_deg
        else:
            pitch_rad = math.atan2(-transform[2, 0], math.sqrt(transform[0, 0] ** 2 + transform[1, 0] ** 2))
            pitch_deg = pitch_rad * self._rad_to_deg
            yaw_deg = (
                math.atan2(transform[1, 0] / math.cos(pitch_rad), transform[0, 0] / math.cos(pitch_rad))
                * self._rad_to_deg
            )
            roll_deg = (
                math.atan2(transform[2, 1] / math.cos(pitch_rad), transform[2, 2] / math.cos(pitch_rad))
                * self._rad_to_deg
            )

        return PiperForwardKinematicsPose(x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg)

    def calculate_link_transforms(self, joints_rad: np.ndarray) -> list[np.ndarray]:
        joints_rad = np.asarray(joints_rad, dtype=float)
        if joints_rad.shape != (6,):
            raise ValueError(f"Expected 6 joint angles in radians, got shape {joints_rad.shape}")

        transforms: list[np.ndarray] = []
        current = np.eye(4, dtype=float)
        for i in range(6):
            current = current @ self._link_transformation(
                self._alpha[i],
                self._a[i],
                joints_rad[i] + self._theta[i],
                self._d[i],
            )
            transforms.append(current.copy())
        return transforms

    def calculate_link_poses(self, joints_rad: np.ndarray) -> list[PiperForwardKinematicsPose]:
        return [self._matrix_to_pose(transform) for transform in self.calculate_link_transforms(joints_rad)]

    def calculate_end_effector_transform(self, joints_rad: np.ndarray) -> np.ndarray:
        return self.calculate_link_transforms(joints_rad)[-1]

    def calculate_end_effector_pose(self, joints_rad: np.ndarray) -> PiperForwardKinematicsPose:
        return self.calculate_link_poses(joints_rad)[-1]
