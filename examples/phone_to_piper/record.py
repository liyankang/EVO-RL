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
Record dataset with phone WebXR teleoperation for Piper robot arm.

Collects (observation, action) pairs in lerobot dataset format.
- Observation: joint angles (URDF deg) + gripper
- Action: joint angles (URDF deg) + gripper

Usage:
    python examples/phone_to_piper/record.py
    python examples/phone_to_piper/record.py --num-episodes 10 --episode-time 60
"""

import argparse
import os
import sys
import time

import numpy as np

# Add project paths to sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_SRC_ROOT = os.path.join(_REPO_ROOT, 'src')
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

# Add teleop_service to path
for _candidate in ["./teleop_service", "../teleop_service"]:
    _p = os.path.abspath(_candidate)
    if os.path.isdir(_p):
        if os.path.dirname(_p) not in sys.path:
            sys.path.insert(0, os.path.dirname(_p))
        if _p not in sys.path:
            sys.path.insert(0, _p)
        break

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.robot_utils import precise_sleep

from real_robot import PiperRealRobot, PiperIKBridge
from server import TeleopServer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 15
TASK_DESCRIPTION = "Phone teleop Piper manipulation"
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"

# Joint names for dataset features (URDF deg)
JOINT_KEYS = [f"joint_{i}.pos" for i in range(1, 7)]
GRIPPER_KEY = "gripper.pos"
ALL_KEYS = JOINT_KEYS + [GRIPPER_KEY]


def obs_to_dict(obs: dict) -> dict:
    """Convert PiperRealRobot.get_obs() numpy arrays to flat dict of scalars."""
    d = {}
    q = obs["joint_angles"]  # np.array (6,) URDF deg
    for i, key in enumerate(JOINT_KEYS):
        d[key] = float(q[i])
    d[GRIPPER_KEY] = float(obs["gripper_pos"][0]) if hasattr(obs["gripper_pos"], "__len__") else float(obs["gripper_pos"])
    return d


def action_to_dict(action: dict) -> dict:
    """Convert IK bridge output to flat dict of scalars."""
    d = {}
    q = action["joint_angles"]  # np.array (6,) URDF deg
    for i, key in enumerate(JOINT_KEYS):
        d[key] = float(q[i])
    gripper = action.get("gripper_pos", [0.0])
    d[GRIPPER_KEY] = float(gripper[0]) if hasattr(gripper, "__len__") else float(gripper)
    return d


def main():
    parser = argparse.ArgumentParser(description="Record dataset with phone teleop for Piper")
    parser.add_argument("--can-port", default="can0", help="CAN port (default: can0)")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--freq", type=int, default=FPS, help=f"Control frequency (default: {FPS})")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip calibration")
    parser.add_argument("--speed-ratio", type=int, default=50, help="Motor speed 0~100")
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES, help="Number of episodes")
    parser.add_argument("--episode-time", type=float, default=EPISODE_TIME_SEC, help="Episode duration (s)")
    parser.add_argument("--reset-time", type=float, default=RESET_TIME_SEC, help="Reset duration (s)")
    parser.add_argument("--repo-id", default=HF_REPO_ID, help="HuggingFace dataset repo ID")
    parser.add_argument("--task", default=TASK_DESCRIPTION, help="Task description")
    parser.add_argument("--message-timeout-ms", type=float, default=1000.0, help="Message timeout ms")
    args = parser.parse_args()

    # --- Initialize robot ---
    robot = PiperRealRobot(
        can_port=args.can_port,
        require_calibration=not args.no_calibrate,
        speed_ratio=args.speed_ratio,
    )
    ik_bridge = PiperIKBridge(robot)

    # --- Initialize TeleopServer (Flask in background) ---
    server = TeleopServer(
        get_obs_fn=robot.get_obs,
        on_action_fn=None,
        use_ik=False,
        arm_type="piper",
        port=args.port,
        control_freq=float(args.freq),
        message_timeout_ms=args.message_timeout_ms,
    )
    server.serve_web_only()

    # --- Create dataset ---
    # Hardware features: joint angles (URDF deg) + gripper
    hw_features = {key: float for key in ALL_KEYS}
    action_features = hw_to_dataset_features(hw_features, ACTION, use_video=False)
    obs_features = hw_to_dataset_features(hw_features, OBS_STR, use_video=False)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.freq,
        features=dataset_features,
        robot_type="piper_follower",
        use_videos=False,
    )

    # --- Keyboard listener for episode control ---
    listener, events = init_keyboard_listener()

    print("=" * 50)
    print("Phone → Piper 数据采集")
    print(f"CAN: {args.can_port}  |  FPS: {args.freq}")
    print(f"Episodes: {args.num_episodes}  |  Duration: {args.episode_time}s")
    print(f"Repo: {args.repo_id}")
    print("按 N 开始新 episode，按 R 重录，按 Q 退出")
    print("=" * 50)

    def record_episode(duration: float):
        """Run one recording episode."""
        step_count = 0
        t_start = time.time()

        while time.time() - t_start < duration:
            if events["stop_recording"]:
                break

            t0 = time.perf_counter()

            # 1. Get observation
            obs = robot.get_obs()
            obs_dict = obs_to_dict(obs)

            # 2. Get EE target from TeleopController
            ee_action = server.controller.step(obs)

            if ee_action is not None:
                # 3. IK → joint angles
                action = ik_bridge.compute_action(
                    ee_action["arm_pos"],
                    ee_action["arm_quat"],
                    obs,
                )
                action["gripper_pos"] = ee_action["gripper_pos"]
            else:
                # No phone input yet, hold current position
                action = {
                    "joint_angles": obs["joint_angles"].copy(),
                    "gripper_pos": obs["gripper_pos"].copy() if hasattr(obs["gripper_pos"], "copy") else np.array([obs["gripper_pos"]]),
                }

            # 4. Send to robot
            robot.send_action(action)

            # 5. Save to dataset buffer
            action_dict = action_to_dict(action)
            obs_frame = build_dataset_frame(dataset.features, obs_dict, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, action_dict, prefix=ACTION)
            frame = {**obs_frame, **action_frame, "task": args.task}
            dataset.add_frame(frame)

            step_count += 1
            precise_sleep(max(1.0 / args.freq - (time.perf_counter() - t0), 0.0))

        return step_count

    try:
        episode_idx = 0
        while episode_idx < args.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {episode_idx + 1} of {args.num_episodes}")

            # Record episode
            n_steps = record_episode(args.episode_time)
            print(f"  Episode {episode_idx + 1}: {n_steps} frames")

            # Save episode
            dataset.save_episode()
            episode_idx += 1

            # Reset if needed
            if not events["stop_recording"] and episode_idx < args.num_episodes:
                if events.get("rerecord_episode"):
                    log_say("Re-recording episode")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    episode_idx -= 1
                    continue

                log_say("Reset environment")
                robot.reset()
                # Brief reset teleop
                record_episode(args.reset_time)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        log_say("Stop recording")
        server.stop()
        robot.close()
        listener.stop()

        dataset.finalize()
        print(f"\nDataset saved: {args.repo_id}")
        # dataset.push_to_hub()  # Uncomment to upload


if __name__ == "__main__":
    main()
