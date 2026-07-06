#!/usr/bin/env python3
"""
手动记录 home 关节角

用法：
    python teleop_service/record_home.py --port can0 --no-calibrate

把机械臂手动推到目标位置，按 Enter 记录当前关节角。
"""

import argparse
import sys
import os
import time
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(REPO_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig
from lerobot.robots.piper_follower.piper_follower import PiperFollower
from lerobot.robots.piper_follower.robot_kinematic_processor import (
    PIPER_SDK_MOTOR_NAMES,
    make_piper_kinematics,
    piper_sdk_joints_to_urdf_joints,
)
from scipy.spatial.transform import Rotation as R


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='can0')
    parser.add_argument('--no-calibrate', action='store_true')
    args = parser.parse_args()

    config = PiperFollowerConfig(
        port=args.port,
        require_calibration=not args.no_calibrate,
        sync_gripper=True,
        enable_on_connect=True,
        cameras={},
    )
    robot = PiperFollower(config)
    robot.connect(calibrate=not args.no_calibrate)
    print('已连接，每秒刷新一次关节角。调好位置后按 Enter 记录。\n')

    kin = make_piper_kinematics()

    try:
        import threading

        stop_flag = [False]

        def display_loop():
            while not stop_flag[0]:
                obs = robot._read_raw_observation()
                sdk = np.array([obs[f'{n}.pos'] for n in PIPER_SDK_MOTOR_NAMES])
                urdf = piper_sdk_joints_to_urdf_joints(sdk)
                T = kin.forward_kinematics(urdf)
                pos = T[:3, 3]
                quat = R.from_matrix(T[:3, :3]).as_quat()
                euler = np.rad2deg(R.from_matrix(T[:3, :3]).as_euler('xyz'))
                print(f'\r关节(SDK deg): {np.round(sdk, 2)}  '
                      f'FK pos: {np.round(pos, 3)}  '
                      f'欧拉xyz(deg): {np.round(euler, 1)}    ', end='', flush=True)
                time.sleep(0.5)

        t = threading.Thread(target=display_loop, daemon=True)
        t.start()

        input()  # 按 Enter
        stop_flag[0] = True
        time.sleep(0.1)

        obs = robot._read_raw_observation()
        sdk = np.array([obs[f'{n}.pos'] for n in PIPER_SDK_MOTOR_NAMES])
        urdf = piper_sdk_joints_to_urdf_joints(sdk)
        T = kin.forward_kinematics(urdf)
        pos = T[:3, 3]
        quat = R.from_matrix(T[:3, :3]).as_quat()
        euler = np.rad2deg(R.from_matrix(T[:3, :3]).as_euler('xyz'))
        gripper = obs.get('gripper.pos', 0.0)

        print('\n\n========== 记录结果 ==========')
        print(f'SDK 关节角 (deg): {np.round(sdk, 3).tolist()}')
        print(f'URDF 关节角 (deg): {np.round(urdf, 3).tolist()}')
        print(f'FK 末端位置 (m): {np.round(pos, 4).tolist()}')
        print(f'FK 末端四元数 [x,y,z,w]: {np.round(quat, 4).tolist()}')
        print(f'FK 末端欧拉角 xyz (deg): {np.round(euler, 2).tolist()}')
        print(f'夹爪: {gripper:.3f}')
        print()
        print('复制到 real_robot.py 的 home_joints_sdk_deg:')
        print(f'self.home_joints_sdk_deg = np.array([')
        for i, v in enumerate(sdk):
            name = PIPER_SDK_MOTOR_NAMES[i]
            print(f'    {v:.3f},  # {name}')
        print(f'])')
        print('==============================')

    finally:
        robot.disconnect()


if __name__ == '__main__':
    main()
