#!/usr/bin/env python
"""
Minimal local validation for the formal keyboard_piper_tcp pipeline.

This script does not require pytest. It verifies three things:

1. KeyboardPiperTCPTeleop can be instantiated and emits ee.* actions.
2. The formal Piper IK processor can consume ee.* actions.
3. Position-only updates prefer the differential IK path instead of the
   full inverse_kinematics fallback.

Usage:
    python tests/my_test/validate_keyboard_piper_tcp_pipeline.py
"""

from __future__ import annotations

import numpy as np

from lerobot.robots.piper_follower.robot_kinematic_processor import (
    PIPER_SDK_MOTOR_NAMES,
    PiperInverseKinematicsEEToJoints,
)
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardPiperTCPTeleopConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardPiperTCPTeleop


class FakeKinematics:
    def __init__(self):
        self.inverse_calls = 0

    def forward_kinematics(self, q_current):
        transform = np.eye(4)
        # Simple linear mapping for smoke validation only.
        transform[:3, 3] = np.asarray(q_current[:3], dtype=float) * 0.01
        return transform

    def inverse_kinematics(self, q_current, target):
        del q_current, target
        self.inverse_calls += 1
        raise RuntimeError("Unexpected full IK fallback during position-only differential IK validation.")


def main() -> None:
    teleop = KeyboardPiperTCPTeleop(KeyboardPiperTCPTeleopConfig())
    action = teleop.get_action()
    print("Teleop action keys:", sorted(action))

    kinematics = FakeKinematics()
    step = PiperInverseKinematicsEEToJoints(
        kinematics=kinematics,
        motor_names=PIPER_SDK_MOTOR_NAMES + ["gripper"],
    )
    step.transition = {
        "observation": {f"joint_{i}.pos": 0.0 for i in range(1, 7)},
    }

    ee_action = {
        "ee.x": 0.01,
        "ee.y": 0.0,
        "ee.z": 0.0,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "ee.gripper_pos": 1.0,
    }
    out = step.action(ee_action)

    print("Differential IK fallback calls:", kinematics.inverse_calls)
    print("Formal pipeline output keys:", sorted(out))

    if kinematics.inverse_calls != 0:
        raise SystemExit("Validation failed: expected differential IK path, but full IK fallback was used.")

    expected = {f"{name}.pos" for name in PIPER_SDK_MOTOR_NAMES} | {"gripper.pos"}
    if set(out) != expected:
        raise SystemExit(f"Validation failed: unexpected output keys {sorted(out)}")

    print("Validation passed.")


if __name__ == "__main__":
    main()
