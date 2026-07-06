#!/usr/bin/env python3
"""
手机 WebXR 遥操作服务进程（独立运行）

职责：
  1. 连接 Piper 机械臂（CAN 总线）
  2. 启动 Flask/SocketIO 服务接收手机 WebXR 位姿
  3. 用 placo IK 将 EE 位姿转换为关节角，直接驱动机械臂
  4. 通过 ZMQ PUB socket 广播实际下发的关节角，供 recording_loop 写数据集

recording_loop 的分工：
  - Policy 运行时（未干预）：recording_loop 自己调 robot.send_action() 发指令
  - 手机干预时：phone_server 驱动机械臂，recording_loop 从 ZMQ 读关节角只写数据集
    （recording_loop 在干预期间跳过自己的 robot.send_action()）

用法:
    # 终端1: 启动手机服务（连真机）
    python teleop_service/phone_server.py --can-port can0 --no-calibrate

    # 终端2: 启动推理+录制
    lerobot-human-inloop-record \\
        --robot.type=piper_follower --teleop.type=phone_piper_tcp ...
"""

import argparse
import json
import sys
import os
import threading

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

PHONE_SERVER_ZMQ_PORT = 5556
PHONE_SERVER_WEB_PORT = 5000


def main():
    parser = argparse.ArgumentParser(description="手机 WebXR → Piper IK → CAN + ZMQ")
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--web-port", type=int, default=PHONE_SERVER_WEB_PORT)
    parser.add_argument("--zmq-port", type=int, default=PHONE_SERVER_ZMQ_PORT)
    parser.add_argument("--freq", type=float, default=30.0)
    parser.add_argument("--speed-ratio", type=int, default=50)
    parser.add_argument("--message-timeout-ms", type=float, default=500.0)
    args = parser.parse_args()

    import zmq
    zmq_ctx = zmq.Context()
    pub = zmq_ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{args.zmq_port}")
    print(f"[phone_server] ZMQ PUB on tcp://*:{args.zmq_port}")

    from real_robot import PiperRealRobot, PiperIKBridge
    from server import TeleopServer

    print("[phone_server] 连接 Piper...")
    robot = PiperRealRobot(
        can_port=args.can_port,
        require_calibration=not args.no_calibrate,
        speed_ratio=args.speed_ratio,
    )
    ik_bridge = PiperIKBridge(robot)

    _current_obs: dict = {}
    _obs_lock = threading.Lock()

    def custom_get_obs():
        obs = robot.get_obs()
        with _obs_lock:
            _current_obs.clear()
            _current_obs.update(obs)
        # Publish current joint state so recording_loop can track robot position
        _publish(obs["joint_angles"], obs["gripper_pos"], msg_type="obs")
        return obs

    def custom_on_action(ee_action: dict):
        with _obs_lock:
            obs = dict(_current_obs)
        if not obs:
            return

        gripper_pos = ee_action["gripper_pos"]
        pos_delta = np.linalg.norm(ee_action["arm_pos"] - obs["arm_pos"])
        quat_dot = float(np.abs(np.dot(ee_action["arm_quat"], obs["arm_quat"])))
        arm_skip = pos_delta < 0.002 and quat_dot > 0.9999

        if arm_skip:
            action = {
                "joint_angles": obs["joint_angles"],
                "gripper_pos": gripper_pos,
                "q_curr_urdf_deg": obs["joint_angles"],
            }
        else:
            action = ik_bridge.compute_action(
                ee_action["arm_pos"], ee_action["arm_quat"], obs
            )
            action["gripper_pos"] = gripper_pos

        # Publish first so recording_loop can stop its own send_action before we hit CAN.
        _publish(action["joint_angles"], gripper_pos, msg_type="action")

        # Drive the robot
        robot.send_action(action)

    def _publish(q_urdf: np.ndarray, gripper_pos, msg_type: str):
        from lerobot.robots.piper_follower.robot_kinematic_processor import (
            PIPER_SDK_MOTOR_NAMES,
            piper_urdf_joints_to_sdk_joints,
        )
        q_sdk = piper_urdf_joints_to_sdk_joints(q_urdf)
        gripper_val = float(np.asarray(gripper_pos).flat[0])
        payload = {name: float(q_sdk[i]) for i, name in enumerate(PIPER_SDK_MOTOR_NAMES)}
        payload["gripper"] = gripper_val
        payload["_type"] = msg_type
        pub.send_string(json.dumps(payload))

    def on_state(state: str):
        print(f"[phone_server] state: {state}")
        if state == "reset_env":
            robot.reset()

    server = TeleopServer(
        get_obs_fn=custom_get_obs,
        on_action_fn=custom_on_action,
        on_state_fn=on_state,
        host="0.0.0.0",
        port=args.web_port,
        control_freq=args.freq,
        message_timeout_ms=args.message_timeout_ms,
        use_ik=False,
    )

    print(f"[phone_server] Flask on http://0.0.0.0:{args.web_port}")
    print("[phone_server] 等待手机连接...")

    try:
        server.run(block=True)
    except KeyboardInterrupt:
        print("\n[phone_server] 中断")
    finally:
        server.stop()
        robot.close()
        pub.close()
        zmq_ctx.term()


if __name__ == "__main__":
    main()
