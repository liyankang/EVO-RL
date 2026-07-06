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
Evaluate a trained policy on Piper robot arm with optional phone teleop intervention.

The policy outputs joint angles (URDF deg) matching the record.py action space.
Phone teleop can be enabled for human-in-the-loop intervention (HIL mode).

Usage:
    python examples/phone_to_piper/evaluate.py \\
        --policy-path <hf_username>/<model_repo_id> \\
        --dataset-path <hf_username>/<dataset_repo_id>

    # With phone teleop for intervention:
    python examples/phone_to_piper/evaluate.py \\
        --policy-path <hf_username>/<model_repo_id> \\
        --dataset-path <hf_username>/<dataset_repo_id> \\
        --enable-phone-teleop
"""

import argparse
import sys
import os
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
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.robot_utils import precise_sleep

from real_robot import PiperRealRobot, PiperIKBridge
from server import TeleopServer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "Phone teleop Piper manipulation"
HF_MODEL_ID = "<hf_username>/<model_repo_id>"
HF_DATASET_ID = "<hf_username>/<dataset_repo_id>"

JOINT_KEYS = [f"joint_{i}.pos" for i in range(1, 7)]
GRIPPER_KEY = "gripper.pos"
ALL_KEYS = JOINT_KEYS + [GRIPPER_KEY]


def dict_to_action_dict(d: dict) -> dict:
    """Convert flat dict to numpy arrays for robot."""
    q = np.array([d[key] for key in JOINT_KEYS])
    gripper = np.array([d.get(GRIPPER_KEY, 0.0)])
    return {"joint_angles": q, "gripper_pos": gripper}


def obs_to_dict(obs: dict) -> dict:
    """Convert PiperRealRobot.get_obs() to flat dict."""
    d = {}
    q = obs["joint_angles"]
    for i, key in enumerate(JOINT_KEYS):
        d[key] = float(q[i])
    d[GRIPPER_KEY] = float(obs["gripper_pos"][0]) if hasattr(obs["gripper_pos"], "__len__") else float(obs["gripper_pos"])
    return d


def main():
    parser = argparse.ArgumentParser(description="Evaluate policy on Piper")
    parser.add_argument("--can-port", default="can0", help="CAN port")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--freq", type=int, default=FPS, help="Control frequency")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip calibration")
    parser.add_argument("--speed-ratio", type=int, default=50, help="Motor speed 0~100")
    parser.add_argument("--policy-path", default=HF_MODEL_ID, help="HF model repo ID")
    parser.add_argument("--dataset-path", default=HF_DATASET_ID, help="HF dataset repo ID")
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--episode-time", type=float, default=EPISODE_TIME_SEC)
    parser.add_argument("--enable-phone-teleop", action="store_true",
                        help="Enable phone teleop for human-in-the-loop intervention")
    parser.add_argument("--message-timeout-ms", type=float, default=1000.0)
    args = parser.parse_args()

    # --- Initialize robot ---
    robot = PiperRealRobot(
        can_port=args.can_port,
        require_calibration=not args.no_calibrate,
        speed_ratio=args.speed_ratio,
    )

    # --- Load policy ---
    # NOTE: Import policy class based on your training config
    # Example: from lerobot.policies.act.modeling_act import ACTPolicy
    # policy = ACTPolicy.from_pretrained(args.policy_path)
    raise NotImplementedError(
        "Please uncomment and configure the policy import above based on your training setup. "
        "Supported policies: ACTPolicy, DiffusionPolicy, VQBetPolicy, etc."
    )

    # --- Build policy pre/post processors ---
    # dataset = LeRobotDataset.create(...)  # or load existing
    # preprocessor, postprocessor = make_pre_post_processors(
    #     policy_cfg=policy.config,
    #     pretrained_path=args.policy_path,
    #     dataset_stats=dataset.meta.stats,
    # )

    # --- Optional: phone teleop for intervention ---
    server = None
    ik_bridge = None
    if args.enable_phone_teleop:
        ik_bridge = PiperIKBridge(robot)
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
        print("Phone teleop enabled for intervention")

    # --- Keyboard listener ---
    listener, events = init_keyboard_listener()

    print("=" * 50)
    print("Piper Policy Evaluation")
    print(f"Policy: {args.policy_path}")
    print(f"Episodes: {args.num_episodes}  |  Duration: {args.episode_time}s")
    print("=" * 50)

    try:
        for episode_idx in range(args.num_episodes):
            if events["stop_recording"]:
                break

            log_say(f"Evaluating episode {episode_idx + 1} of {args.num_episodes}")

            # Reset policy
            # policy.reset()

            t_start = time.time()
            while time.time() - t_start < args.episode_time:
                if events["stop_recording"]:
                    break

                t0 = time.perf_counter()

                # 1. Get observation
                obs = robot.get_obs()
                obs_dict = obs_to_dict(obs)

                # 2. Policy inference
                # policy_action = policy.select_action(obs_dict)
                # robot_action = postprocessor(policy_action)
                raise NotImplementedError("Policy inference not configured")

                # 3. Check for phone intervention
                use_phone = False
                if server is not None:
                    ee_action = server.controller.step(obs)
                    if ee_action is not None:
                        use_phone = True
                        phone_action = ik_bridge.compute_action(
                            ee_action["arm_pos"],
                            ee_action["arm_quat"],
                            obs,
                        )
                        phone_action["gripper_pos"] = ee_action["gripper_pos"]
                        robot.send_action(phone_action)

                # 4. Send policy action (if no phone intervention)
                if not use_phone:
                    # action = dict_to_action_dict(robot_action)
                    # robot.send_action(action)
                    pass

                precise_sleep(max(1.0 / args.freq - (time.perf_counter() - t0), 0.0))

            print(f"  Episode {episode_idx + 1} done")

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        log_say("Stop evaluation")
        if server:
            server.stop()
        robot.close()
        listener.stop()


if __name__ == "__main__":
    main()
