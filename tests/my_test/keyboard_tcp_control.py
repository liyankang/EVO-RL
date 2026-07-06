#!/usr/bin/env python
"""
Standalone keyboard TCP control for Piper arm.

On startup, moves to fixed initial joint angles (INIT_JOINTS),
then computes TCP via FK. Keyboard increments from there using
position-only differential IK.

Usage:
    python tests/my_tests/keyboard_tcp_control.py

Keys:
    W/S  = +/- X
    A/D  = +/- Y
    Q/E  = +/- Z (up/down)
    +/-  = increase/decrease step size
    ESC  = quit
"""

import logging
import sys
import time

import numpy as np

from lerobot.robots.piper_follower import PiperFollower, PiperFollowerConfig
from lerobot.robots.piper_follower.robot_kinematic_processor import (
    PIPER_SDK_MOTOR_NAMES,
    make_piper_kinematics,
    piper_sdk_joints_to_urdf_joints,
    piper_urdf_joints_to_sdk_joints,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PORT = "can0"
ROBOT_ID = "my_piper_follower"
FPS = 60
POS_STEP = 0.01
STEP_SCALE = 5
DLS_DAMPING = 0.05
MAX_JOINT_STEP_DEG = 8.0
JACOBIAN_EPS_RAD = 1e-3

# Fixed initial joint angles (SDK convention, degrees)
INIT_JOINTS = (0.315, -1.109, 3.636, 0.757, 19.965, -0.994)

# EE safety bounds (metres)
EE_MIN = np.array([-0.6, -0.6, 0.0])
EE_MAX = np.array([0.6, 0.6, 0.8])

try:
    from pynput import keyboard as kb
except ImportError:
    print("pynput is required: pip install pynput")
    sys.exit(1)


class KeyState:
    """Thread-safe keyboard state tracker. One press = one trigger."""

    def __init__(self):
        self._pressed: set[str] = set()
        self._consumed: set[str] = set()
        self._lock = __import__("threading").Lock()
        self._quit = False

    def on_press(self, key):
        with self._lock:
            if key == kb.Key.esc:
                self._quit = True
                return False
            if hasattr(key, "char") and key.char is not None:
                self._pressed.add(key.char)

    def on_release(self, key):
        with self._lock:
            if hasattr(key, "char") and key.char is not None:
                self._pressed.discard(key.char)
                self._consumed.discard(key.char)

    def get_new_keys(self) -> set[str]:
        with self._lock:
            new_keys = self._pressed - self._consumed
            self._consumed |= new_keys
            return new_keys

    @property
    def quit(self):
        return self._quit


def fk_position(kinematics, q_urdf_deg: np.ndarray) -> np.ndarray:
    return kinematics.forward_kinematics(q_urdf_deg)[:3, 3].copy()


def numerical_position_jacobian(
    kinematics, q_urdf_deg: np.ndarray, eps_rad: float = JACOBIAN_EPS_RAD
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
    damping: float = DLS_DAMPING,
) -> np.ndarray:
    """Damped least-squares differential IK step in radians."""
    jj_t = jacobian @ jacobian.T
    return jacobian.T @ np.linalg.solve(
        jj_t + (damping**2) * np.eye(jj_t.shape[0], dtype=float),
        delta_pos,
    )


def clip_position(pos: np.ndarray) -> np.ndarray:
    return np.clip(pos, EE_MIN, EE_MAX)


def main():
    robot_cfg = PiperFollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        require_calibration=False,
        enable_on_connect=True,
        sync_gripper=True,
    )
    robot = PiperFollower(robot_cfg)
    robot.connect(calibrate=False)

    try:
        kinematics = make_piper_kinematics()

        q_target_sdk = np.array(INIT_JOINTS, dtype=float)
        action = {}
        for i, name in enumerate(PIPER_SDK_MOTOR_NAMES):
            action[f"{name}.pos"] = float(q_target_sdk[i])
        robot.send_action(action)
        time.sleep(0.5)

        q_current_urdf = piper_sdk_joints_to_urdf_joints(q_target_sdk)
        tcp_pos = fk_position(kinematics, q_current_urdf)
        pos_step = POS_STEP

        print(f"TCP:    x={tcp_pos[0]:.4f}  y={tcp_pos[1]:.4f}  z={tcp_pos[2]:.4f}")
        print(f"Step: {pos_step:.4f} m")
        print("Mode: differential IK (position only)")
        print("\nControls: W/S=+/-x  A/D=+/-y  Q/E=+/-z  +/-=step  ESC=quit\n")

        ks = KeyState()
        listener = kb.Listener(on_press=ks.on_press, on_release=ks.on_release)
        listener.start()

        try:
            while not ks.quit:
                t0 = time.perf_counter()
                keys = ks.get_new_keys()
                delta_pos = np.zeros(3, dtype=float)

                for k in keys:
                    k = k.lower()
                    if k == "w":
                        delta_pos[0] += pos_step
                    elif k == "s":
                        delta_pos[0] -= pos_step
                    elif k == "a":
                        delta_pos[1] += pos_step
                    elif k == "d":
                        delta_pos[1] -= pos_step
                    elif k == "q":
                        delta_pos[2] += pos_step
                    elif k == "e":
                        delta_pos[2] -= pos_step
                    elif k in ("+", "="):
                        pos_step *= STEP_SCALE
                        log.info(f"Step -> {pos_step:.4f} m")
                    elif k == "-":
                        pos_step /= STEP_SCALE
                        log.info(f"Step -> {pos_step:.4f} m")

                if np.any(delta_pos):
                    target_pos = clip_position(tcp_pos + delta_pos)
                    delta_pos = target_pos - tcp_pos

                    jacobian = numerical_position_jacobian(kinematics, q_current_urdf)
                    dq_rad = solve_dls_position_step(jacobian, delta_pos)
                    dq_deg = np.rad2deg(dq_rad)
                    dq_deg = np.clip(dq_deg, -MAX_JOINT_STEP_DEG, MAX_JOINT_STEP_DEG)

                    q_candidate_urdf = q_current_urdf + dq_deg
                    candidate_pos = fk_position(kinematics, q_candidate_urdf)
                    achieved_delta = candidate_pos - tcp_pos
                    err_mm = np.linalg.norm(candidate_pos - target_pos) * 1000

                    q_current_urdf = q_candidate_urdf
                    tcp_pos = candidate_pos
                    q_target_sdk = piper_urdf_joints_to_sdk_joints(q_current_urdf)

                    action = {}
                    for i, name in enumerate(PIPER_SDK_MOTOR_NAMES):
                        action[f"{name}.pos"] = float(q_target_sdk[i])
                    robot.send_action(action)

                    joints_str = "  ".join(
                        f"{PIPER_SDK_MOTOR_NAMES[i]}:{q_target_sdk[i]:+7.2f}" for i in range(6)
                    )
                    print(
                        f"\ntcp:[{tcp_pos[0]:+.4f} {tcp_pos[1]:+.4f} {tcp_pos[2]:+.4f}]"
                        f"  target:[{target_pos[0]:+.4f} {target_pos[1]:+.4f} {target_pos[2]:+.4f}]"
                        f"  err:{err_mm:.1f}mm\n"
                        f"  dpos:[{achieved_delta[0]:+.4f} {achieved_delta[1]:+.4f} {achieved_delta[2]:+.4f}]"
                        f"  joints(sdk): {joints_str}",
                        flush=True,
                    )

                dt = time.perf_counter() - t0
                time.sleep(max(1.0 / FPS - dt, 0.0))

            print("\n\nESC pressed - stopping.")

        except KeyboardInterrupt:
            print("\n\nCtrl+C - stopping.")

        finally:
            listener.stop()

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
