#!/usr/bin/env python3

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

"""
Phone WebXR teleoperation for Piper robot arm.

Integrates teleop_service (Flask + SocketIO + placo IK) with the lerobot project.

Usage:
    python examples/phone_to_piper/teleoperate.py
    python examples/phone_to_piper/teleoperate.py --can-port can0
    python examples/phone_to_piper/teleoperate.py --port 5000 --freq 30
"""

import argparse
import os
import sys
import time

# Add project paths to sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_SRC_ROOT = os.path.join(_REPO_ROOT, 'src')
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

# Add teleop_service to path for imports
TELEOP_SERVICE_DIR = None
for _candidate in [
    "./teleop_service",
    "../teleop_service",
]:
    _p = os.path.abspath(_candidate)
    if os.path.isdir(_p):
        TELEOP_SERVICE_DIR = os.path.dirname(_p)
        if _p not in sys.path:
            sys.path.insert(0, _p)
        break

if TELEOP_SERVICE_DIR is None:
    raise ImportError("Cannot find teleop_service directory")

from real_robot import PiperRealRobot, PiperIKBridge
from server import TeleopServer
from lerobot.utils.robot_utils import precise_sleep

FPS = 30


def main():
    parser = argparse.ArgumentParser(description="Phone WebXR teleoperation for Piper")
    parser.add_argument("--can-port", default="can0", help="CAN port (default: can0)")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000)")
    parser.add_argument("--freq", type=int, default=FPS, help=f"Control frequency Hz (default: {FPS})")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip calibration")
    parser.add_argument("--speed-ratio", type=int, default=50, help="Motor speed 0~100 (default: 50)")
    parser.add_argument("--message-timeout-ms", type=float, default=1000.0, help="Message timeout ms")
    args = parser.parse_args()

    # --- Initialize robot (CAN + placo IK + calibration) ---
    robot = PiperRealRobot(
        can_port=args.can_port,
        require_calibration=not args.no_calibrate,
        speed_ratio=args.speed_ratio,
    )

    # --- IK bridge: EE pose → joint angles (placo) ---
    ik_bridge = PiperIKBridge(robot)

    # --- TeleopServer: Flask + SocketIO + TeleopController ---
    # We use serve_web_only() to run Flask in background.
    # The main thread runs the teleop loop (synchronous, like lerobot teleop_loop).
    server = TeleopServer(
        get_obs_fn=robot.get_obs,
        on_action_fn=None,  # We handle actions in the main loop
        use_ik=False,  # We use our own placo IK, not MuJoCo IK
        arm_type="piper",
        port=args.port,
        control_freq=float(args.freq),
        message_timeout_ms=args.message_timeout_ms,
    )

    # Start Flask + listener thread (no control loop thread)
    server.serve_web_only()

    print("=" * 50)
    print("Phone → Piper 遥操作")
    print(f"CAN: {args.can_port}  |  FPS: {args.freq}  |  IK: placo")
    print("手机触摸屏幕开始控制，松开停止")
    print("=" * 50)

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Get robot observation (joints → FK → EE pose)
            obs = robot.get_obs()

            # 2. TeleopController: phone pose → smoothed EE target
            ee_action = server.controller.step(obs)
            if ee_action is None:
                # No phone connected yet
                precise_sleep(max(1.0 / args.freq - (time.perf_counter() - t0), 0.0))
                continue

            # 3. IK: EE pose → joint angles (URDF deg)
            action = ik_bridge.compute_action(
                ee_action["arm_pos"],
                ee_action["arm_quat"],
                obs,
            )
            action["gripper_pos"] = ee_action["gripper_pos"]

            # 4. Send to robot (handles safe clip + EMA smoothing + URDF→SDK)
            robot.send_action(action)

            # 5. Sleep to maintain frequency
            precise_sleep(max(1.0 / args.freq - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        server.stop()
        robot.close()


if __name__ == "__main__":
    main()
