#!/usr/bin/env python
"""
MuJoCo simulation version of keyboard TCP control for Piper arm.
Uses position-only differential IK for smoother keyboard teleoperation.

Requirements:
    pip install mujoco pynput scipy numpy

Usage:
    python tests/my_tests/keyboard_tcp_mujoco.py

Keys:
    W/S  = +/- X
    A/D  = +/- Y
    Q/E  = +/- Z (up/down)
    +/-  = increase/decrease step size
    ESC  = quit
"""

import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from lerobot.robots.piper_follower.robot_kinematic_processor import (
    PIPER_SDK_MOTOR_NAMES,
    make_piper_kinematics,
    piper_sdk_joints_to_urdf_joints,
    piper_urdf_joints_to_sdk_joints,
)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────
FPS = 60
POS_STEP = 0.01  # metres per key press (1cm)
STEP_SCALE = 5
DLS_DAMPING = 0.05
MAX_JOINT_STEP_DEG = 8.0
JACOBIAN_EPS_RAD = 1e-3

# Fixed initial joint angles (SDK convention, degrees)
INIT_JOINTS = (0.315, -1.109, 3.636, 0.757, 19.965, -0.994)

# EE safety bounds (metres)
EE_MIN = np.array([-0.6, -0.6, 0.0])
EE_MAX = np.array([0.6, 0.6, 0.8])

# MuJoCo joint names (must match URDF)
MJ_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
# ────────────────────────────────────────────────────────────────────

try:
    import mujoco
    import mujoco.viewer  # MuJoCo 3.x 需要显式导入
except ImportError:
    print("mujoco is required: pip install mujoco")
    sys.exit(1)


try:
    from pynput import keyboard as kb
except ImportError:
    print("pynput is required: pip install pynput")
    sys.exit(1)





# ── Keyboard ────────────────────────────────────────────────────────

class KeyState:
    """Thread-safe keyboard state tracker. One press = one trigger."""

    def __init__(self):
        self._pressed: set = set()
        self._consumed: set = set()
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

    def get_new_keys(self) -> set:
        with self._lock:
            new_keys = self._pressed - self._consumed
            self._consumed |= new_keys
            return new_keys

    @property
    def quit(self):
        return self._quit


# ── MuJoCo helpers ──────────────────────────────────────────────────

def set_joint_qpos(model, data, q_rad):
    for i, jname in enumerate(MJ_JOINT_NAMES):
        jid = model.joint(jname).id
        data.qpos[model.jnt_qposadr[jid]] = q_rad[i]


def rewrite_urdf_for_mujoco(urdf_path: str) -> str:
    """Rewrite package mesh URIs into relative paths that MuJoCo can resolve."""
    urdf_file = Path(urdf_path).expanduser().resolve()
    urdf_text = urdf_file.read_text(encoding="utf-8")
    rewritten = urdf_text.replace("package://meshes/", "../meshes/")
    if rewritten == urdf_text:
        return str(urdf_file)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=urdf_file.suffix,
        prefix=f"{urdf_file.stem}_mujoco_",
        dir=str(urdf_file.parent),
        delete=False,
        encoding="utf-8",
    ) as rewritten_file:
        rewritten_file.write(rewritten)
        return rewritten_file.name


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


# ── Main ────────────────────────────────────────────────────────────

def main():
    # ── Kinematics (uses RobotKinematics internally) ────────────
    print("Loading kinematics ...")
    kinematics = make_piper_kinematics()

    # ── MuJoCo model ────────────────────────────────────────────
    print("Building MuJoCo model ...")
    from importlib import resources
    urdf_path_obj = resources.files("lerobot").joinpath("assets/piper_description/urdf/piper_no_gripper_description.urdf")
    urdf_path = rewrite_urdf_for_mujoco(str(urdf_path_obj))
    model = mujoco.MjModel.from_xml_path(urdf_path)
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)

    # ── Set initial joints ──────────────────────────────────────
    q_sdk = np.array(INIT_JOINTS, dtype=float)
    q_urdf_deg = piper_sdk_joints_to_urdf_joints(q_sdk)
    q_init_rad = np.deg2rad(q_urdf_deg)

    set_joint_qpos(model, data, q_init_rad)
    mujoco.mj_forward(model, data)

    # FK for initial TCP position
    tcp_pos = fk_position(kinematics, q_urdf_deg)
    q_current_urdf = q_urdf_deg.copy()

    print(f"\nInitial TCP:    x={tcp_pos[0]:.4f}  y={tcp_pos[1]:.4f}  z={tcp_pos[2]:.4f}")
    print(f"Step: {POS_STEP:.4f} m")
    print("\nControls: W/S=+/-x  A/D=+/-y  Q/E=+/-z  +/-=step  ESC=quit")
    print("Mode: differential IK (position only, gravity off)\n")

    # ── Keyboard ────────────────────────────────────────────────
    ks = KeyState()
    listener = kb.Listener(on_press=ks.on_press, on_release=ks.on_release)
    listener.start()

    # ── Viewer ──────────────────────────────────────────────────
    viewer = mujoco.viewer.launch_passive(model, data)

    try:
        pos_step = POS_STEP
        while not ks.quit and viewer.is_running():
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
                target_pos = np.clip(tcp_pos + delta_pos, EE_MIN, EE_MAX)
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
                q_target_rad = np.deg2rad(q_current_urdf)

                # Update MuJoCo
                set_joint_qpos(model, data, q_target_rad)

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

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()

            dt = time.perf_counter() - t0
            time.sleep(max(1.0 / FPS - dt, 0.0))

        print("\n\nESC pressed — stopping.")

    except KeyboardInterrupt:
        print("\n\nCtrl+C — stopping.")

    finally:
        listener.stop()
        viewer.close()


if __name__ == "__main__":
    main()
