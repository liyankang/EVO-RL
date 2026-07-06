# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from pathlib import Path
import tempfile

import numpy as np

class RobotKinematics:
    """Robot kinematics using placo library for forward and inverse kinematics."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ):
        """
        Initialize placo-based kinematics solver.

        Args:
            urdf_path (str): Path to the robot URDF file
            target_frame_name (str): Name of the end-effector frame in the URDF
            joint_names (list[str] | None): List of joint names to use for the kinematics solver
        """
        try:
            import placo  # type: ignore[import-not-found] # C++ library with Python bindings, no type stubs available. TODO: Create stub file or request upstream typing support.
        except ImportError as e:
            raise ImportError(
                "placo is required for RobotKinematics. "
                "Please install the optional dependencies of `kinematics` in the package."
            ) from e

        urdf_file = Path(urdf_path).expanduser().resolve()
        package_root = str(urdf_file.parent.parent)

        try:
            self.robot = placo.RobotWrapper(str(urdf_file), package_root)
        except Exception as error:
            if self._supports_package_root(error):
                raise

            try:
                # Older placo builds only accept the URDF path and resolve package:// internally.
                self.robot = placo.RobotWrapper(str(urdf_file))
            except ValueError as fallback_error:
                if "package://meshes/" not in str(fallback_error):
                    raise
                rewritten_urdf = self._rewrite_package_mesh_uris(urdf_file)
                self.robot = placo.RobotWrapper(str(rewritten_urdf))
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # Fix the base

        self.target_frame_name = target_frame_name

        # Set joint names
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names

        # Initialize frame task for IK
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

    @staticmethod
    def _supports_package_root(error: Exception) -> bool:
        """Return True when the constructor failure is not just a signature mismatch."""
        error_text = str(error)
        return (
            "did not match C++ signature" not in error_text
            and "ArgumentError" not in error.__class__.__name__
        )

    @staticmethod
    def _rewrite_package_mesh_uris(urdf_file: Path) -> Path:
        urdf_text = urdf_file.read_text(encoding="utf-8")
        rewritten = urdf_text.replace("package://meshes/", "../meshes/")
        if rewritten == urdf_text:
            raise ValueError(f"URDF '{urdf_file}' does not contain package://meshes/ URIs to rewrite.")

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=urdf_file.suffix,
            prefix=f"{urdf_file.stem}_rewritten_",
            dir=str(urdf_file.parent),
            delete=False,
            encoding="utf-8",
        ) as rewritten_file:
            rewritten_file.write(rewritten)
            return Path(rewritten_file.name)

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for given joint configuration given the target frame name in the constructor.

        Args:
            joint_pos_deg: Joint positions in degrees (numpy array)

        Returns:
            4x4 transformation matrix of the end-effector pose
        """

        # Convert degrees to radians
        joint_pos_rad = np.deg2rad(joint_pos_deg[: len(self.joint_names)])

        # Update joint positions in placo robot
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])

        # Update kinematics
        self.robot.update_kinematics()

        # Get the transformation matrix
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
        max_iters: int = 1,
        pos_tol: float = 1e-3,
        max_step_deg: float = None,
    ) -> np.ndarray:
        """
        Compute inverse kinematics using placo solver (single step by default).

        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_weight: Weight for position constraint in IK
            orientation_weight: Weight for orientation constraint in IK
            max_iters: Number of solver iterations (default 1 for single-step use)
            pos_tol: Position convergence threshold in metres (only used when max_iters > 1)
            max_step_deg: Per-iteration joint clamp in degrees (None = no clamp)

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """
        self.tip_frame.T_world_frame = desired_ee_pose
        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)

        q_deg = current_joint_pos[: len(self.joint_names)].copy().astype(float)

        for _ in range(max_iters):
            q_rad = np.deg2rad(q_deg)
            for i, joint_name in enumerate(self.joint_names):
                self.robot.set_joint(joint_name, q_rad[i])

            self.solver.solve(True)
            self.robot.update_kinematics()

            q_new_deg = np.rad2deg([self.robot.get_joint(n) for n in self.joint_names])
            if max_step_deg is not None:
                q_new_deg = q_deg + np.clip(q_new_deg - q_deg, -max_step_deg, max_step_deg)
            q_deg = q_new_deg

            if max_iters > 1:
                fk = self.forward_kinematics(q_deg)
                if np.linalg.norm(fk[:3, 3] - desired_ee_pose[:3, 3]) < pos_tol:
                    break

        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = q_deg
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        return q_deg
