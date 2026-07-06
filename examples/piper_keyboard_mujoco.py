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

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import mujoco
import mujoco.viewer
import numpy as np

from lerobot.robots.piper_follower.robot_kinematic_processor import (
    PIPER_EE_FRAME,
    PIPER_SDK_MOTOR_NAMES,
    PIPER_URDF_RELPATH,
    PIPER_URDF_JOINT_NAMES,
    make_piper_kinematics,
    piper_sdk_joints_to_urdf_joints,
    piper_urdf_joints_to_sdk_joints,
)
from lerobot.utils.rotation import Rotation

if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
    raise RuntimeError("No DISPLAY detected. Run this example from a local desktop session.")

try:
    from pynput import keyboard
except Exception as exc:
    raise RuntimeError("pynput is required for keyboard teleop in this example.") from exc

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

EE_BOUNDS_MIN = np.array([-0.6, -0.6, 0.0], dtype=float)
EE_BOUNDS_MAX = np.array([0.6, 0.6, 0.8], dtype=float)
MAX_EE_STEP_M = 0.05


def _rewrite_package_mesh_uris(urdf_file: Path) -> Path:
    urdf_text = urdf_file.read_text(encoding="utf-8")
    rewritten = urdf_text.replace("package://meshes/", "../meshes/")
    if rewritten == urdf_text:
        return urdf_file

    temp_dir = Path(tempfile.mkdtemp(prefix="piper_mujoco_"))
    urdf_dir = temp_dir / "urdf"
    mesh_dir = temp_dir / "meshes"
    urdf_dir.mkdir()

    source_mesh_dir = urdf_file.parent.parent / "meshes"
    shutil.copytree(source_mesh_dir, mesh_dir)
    for mesh_file in mesh_dir.iterdir():
        if mesh_file.is_file():
            shutil.copy2(mesh_file, urdf_dir / mesh_file.name)

    rewritten_path = urdf_dir / urdf_file.name
    rewritten_path.write_text(rewritten, encoding="utf-8")
    return rewritten_path


def _make_mujoco_model() -> mujoco.MjModel:
    urdf_path = resources.files("lerobot").joinpath(PIPER_URDF_RELPATH)
    urdf_path = _rewrite_package_mesh_uris(Path(str(urdf_path)))
    print(f"Loading MuJoCo URDF from: {urdf_path}")
    return mujoco.MjModel.from_xml_path(str(urdf_path))


def _body_rotation_matrix(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)


def _body_transform(data: mujoco.MjData, body_id: int) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _body_rotation_matrix(data, body_id)
    transform[:3, 3] = np.asarray(data.xpos[body_id], dtype=float)
    return transform


def _make_joint_qpos_adrs(model: mujoco.MjModel) -> np.ndarray:
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in PIPER_URDF_JOINT_NAMES]
    if any(joint_id < 0 for joint_id in joint_ids):
        missing = [name for name, joint_id in zip(PIPER_URDF_JOINT_NAMES, joint_ids, strict=True) if joint_id < 0]
        raise RuntimeError(f"Missing Piper joints in MuJoCo model: {missing}")
    return np.array([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=int)


@dataclass
class KeyboardTargetState:
    position: np.ndarray
    rotation: np.ndarray
    pos_step: float
    step_scale: float


class KeyboardTCPController:
    def __init__(self, initial_state: KeyboardTargetState):
        self.state = initial_state
        self._pressed: set[str] = set()
        self._consumed: set[str] = set()
        self._listener: keyboard.Listener | None = None

    def connect(self) -> None:
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    @property
    def is_connected(self) -> bool:
        return self._listener is not None and self._listener.is_alive()

    def _on_press(self, key):
        if key == keyboard.Key.esc:
            self.disconnect()
            return False
        if hasattr(key, "char") and key.char is not None:
            self._pressed.add(key.char.lower())

    def _on_release(self, key) -> None:
        if hasattr(key, "char") and key.char is not None:
            char = key.char.lower()
            self._pressed.discard(char)
            self._consumed.discard(char)

    def _get_new_keys(self) -> set[str]:
        new_keys = self._pressed - self._consumed
        self._consumed |= new_keys
        return new_keys

    def get_action(self) -> tuple[dict[str, float], bool]:
        moved = False
        for key in self._get_new_keys():
            if key == "w":
                self.state.position[0] += self.state.pos_step
                moved = True
            elif key == "s":
                self.state.position[0] -= self.state.pos_step
                moved = True
            elif key == "a":
                self.state.position[1] += self.state.pos_step
                moved = True
            elif key == "d":
                self.state.position[1] -= self.state.pos_step
                moved = True
            elif key == "q":
                self.state.position[2] += self.state.pos_step
                moved = True
            elif key == "e":
                self.state.position[2] -= self.state.pos_step
                moved = True
            elif key in ("+", "="):
                self.state.pos_step *= self.state.step_scale
                log.info("Step -> %.4f m", self.state.pos_step)
            elif key == "-":
                self.state.pos_step /= self.state.step_scale
                log.info("Step -> %.4f m", self.state.pos_step)

        return (
            {
                "ee.x": float(self.state.position[0]),
                "ee.y": float(self.state.position[1]),
                "ee.z": float(self.state.position[2]),
            },
            moved,
        )


def _current_sim_observation(data: mujoco.MjData, joint_qpos_adrs: np.ndarray) -> dict[str, float]:
    q_urdf_deg = np.rad2deg(np.asarray(data.qpos[joint_qpos_adrs], dtype=float))
    q_sdk_deg = piper_urdf_joints_to_sdk_joints(q_urdf_deg)
    observation = {f"{name}.pos": float(value) for name, value in zip(PIPER_SDK_MOTOR_NAMES, q_sdk_deg, strict=True)}
    observation["gripper.pos"] = 0.0
    return observation


def _clip_target_position(target_pos: np.ndarray, last_pos: np.ndarray | None) -> np.ndarray:
    clipped = np.clip(np.asarray(target_pos, dtype=float), EE_BOUNDS_MIN, EE_BOUNDS_MAX)
    if last_pos is None:
        return clipped

    delta = clipped - last_pos
    norm = float(np.linalg.norm(delta))
    if norm > MAX_EE_STEP_M and norm > 0.0:
        clipped = last_pos + delta * (MAX_EE_STEP_M / norm)
    return clipped


def _solve_ik(
    kinematics,
    q_guess_urdf_deg: np.ndarray,
    action: dict[str, float],
    last_safe_pos: np.ndarray | None,
    target_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    target_pos = np.array([action["ee.x"], action["ee.y"], action["ee.z"]], dtype=float)
    target_pos = _clip_target_position(target_pos, last_safe_pos)

    target_transform = np.eye(4, dtype=float)
    target_transform[:3, :3] = target_rotation
    target_transform[:3, 3] = target_pos

    q_target_urdf_deg = kinematics.inverse_kinematics(q_guess_urdf_deg, target_transform)
    final_transform = kinematics.forward_kinematics(q_target_urdf_deg)
    err_mm = float(np.linalg.norm(final_transform[:3, 3] - target_pos) * 1000.0)
    return np.asarray(q_target_urdf_deg, dtype=float), target_pos, err_mm


def _set_sim_joints_from_sdk(
    model: mujoco.MjModel, data: mujoco.MjData, joint_qpos_adrs: np.ndarray, sdk_joint_targets_deg: np.ndarray
) -> None:
    q_target_urdf_deg = piper_sdk_joints_to_urdf_joints(sdk_joint_targets_deg)
    data.qpos[joint_qpos_adrs] = np.deg2rad(q_target_urdf_deg)
    mujoco.mj_forward(model, data)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo simulation for Piper keyboard TCP teleoperation.")
    parser.add_argument("--fps", type=int, default=30, help="Simulation/control loop frequency.")
    parser.add_argument(
        "--init-joints-sdk",
        type=float,
        nargs=6,
        default=(0.315, -1.109, 3.636, 0.757, 19.965, -0.994),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Initial Piper SDK joint angles in degrees.",
    )
    parser.add_argument("--pos-step", type=float, default=0.01, help="Keyboard TCP translation step in meters.")
    parser.add_argument("--step-scale", type=float, default=1.5, help="Multiplier for +/- step-size controls.")
    parser.add_argument(
        "--init-duration",
        type=float,
        default=3.0,
        help="Seconds used to smoothly move from the current pose to the fixed initial joints.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.0,
        help="How often to print TCP and joint state summaries in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model = _make_mujoco_model()
    data = mujoco.MjData(model)
    kinematics = make_piper_kinematics()
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PIPER_EE_FRAME)
    if ee_body_id < 0:
        raise RuntimeError(f"Could not find end-effector body '{PIPER_EE_FRAME}' in MuJoCo model.")
    joint_qpos_adrs = _make_joint_qpos_adrs(model)

    init_sdk_deg = np.asarray(args.init_joints_sdk, dtype=float)
    init_urdf_deg = piper_sdk_joints_to_urdf_joints(init_sdk_deg)

    current_urdf_deg = np.rad2deg(np.asarray(data.qpos[joint_qpos_adrs], dtype=float))
    n_steps = max(int(args.init_duration * args.fps), 1)

    print(f"\nMoving to initial joints over {args.init_duration:.1f}s ...")
    print(f"Target: {', '.join(f'{v:.1f}' for v in init_sdk_deg)} deg")
    for step_idx in range(1, n_steps + 1):
        alpha = step_idx / n_steps
        q_interp_urdf_deg = current_urdf_deg + (init_urdf_deg - current_urdf_deg) * alpha
        data.qpos[joint_qpos_adrs] = np.deg2rad(q_interp_urdf_deg)
        mujoco.mj_forward(model, data)
        time.sleep(1.0 / args.fps)
    print("Initial pose reached.")

    init_tcp_transform = _body_transform(data, ee_body_id)
    init_rotation = init_tcp_transform[:3, :3].astype(float).copy()
    init_rotvec = Rotation.from_matrix(init_rotation).as_rotvec()
    controller = KeyboardTCPController(
        KeyboardTargetState(
            position=init_tcp_transform[:3, 3].astype(float).copy(),
            rotation=init_rotation,
            pos_step=args.pos_step,
            step_scale=args.step_scale,
        )
    )

    controller.connect()
    if not controller.is_connected:
        raise RuntimeError("Keyboard listener did not start. Run this example in a local desktop session.")

    print(f"TCP:    x={controller.state.position[0]:.4f}  y={controller.state.position[1]:.4f}  z={controller.state.position[2]:.4f}")
    print(f"rotvec: wx={init_rotvec[0]:.4f}  wy={init_rotvec[1]:.4f}  wz={init_rotvec[2]:.4f}")
    print(f"Step: {controller.state.pos_step:.4f} m")
    print("\nControls: W/S=+/-x  A/D=+/-y  Q/E=+/-z  +/-=step  ESC=quit\n")

    q_guess_urdf_deg = init_urdf_deg.copy()
    last_safe_pos = controller.state.position.copy()
    last_joint_cmd = init_sdk_deg.copy()
    last_print = 0.0

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and controller.is_connected:
                loop_start = time.perf_counter()

                observation = _current_sim_observation(data, joint_qpos_adrs)
                del observation
                ee_action, moved = controller.get_action()
                if moved:
                    q_guess_urdf_deg, last_safe_pos, err_mm = _solve_ik(
                        kinematics=kinematics,
                        q_guess_urdf_deg=q_guess_urdf_deg,
                        action=ee_action,
                        last_safe_pos=last_safe_pos,
                        target_rotation=controller.state.rotation,
                    )
                    last_joint_cmd = piper_urdf_joints_to_sdk_joints(q_guess_urdf_deg)
                    _set_sim_joints_from_sdk(model, data, joint_qpos_adrs, last_joint_cmd)

                    tcp_xyz = last_safe_pos
                    joints_str = "  ".join(
                        f"{PIPER_SDK_MOTOR_NAMES[i]}:{last_joint_cmd[i]:+7.2f}" for i in range(len(PIPER_SDK_MOTOR_NAMES))
                    )
                    print(
                        f"\ntcp:[{tcp_xyz[0]:+.4f} {tcp_xyz[1]:+.4f} {tcp_xyz[2]:+.4f}]"
                        f"  ik_err:{err_mm:.1f}mm\n"
                        f"  joints(sdk): {joints_str}",
                        flush=True,
                    )
                else:
                    _set_sim_joints_from_sdk(model, data, joint_qpos_adrs, last_joint_cmd)

                now = time.monotonic()
                if args.print_interval > 0.0 and now - last_print >= args.print_interval:
                    tcp_transform = _body_transform(data, ee_body_id)
                    tcp_xyz = tcp_transform[:3, 3]
                    print(
                        "TCP "
                        f"x={tcp_xyz[0]:+.3f} y={tcp_xyz[1]:+.3f} z={tcp_xyz[2]:+.3f}"
                    )
                    print("SDK joints deg", np.array2string(last_joint_cmd, precision=2, suppress_small=True))
                    last_print = now

                viewer.sync()
                dt = time.perf_counter() - loop_start
                time.sleep(max(1.0 / args.fps - dt, 0.0))
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
