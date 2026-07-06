#!/usr/bin/env python3
"""
IK 验证脚本：让机器人移动到偏移目标，验证末端实际到达位置是否符合预期。

流程：
  1. 读取当前关节角 → FK 得到当前末端位置
  2. 用户指定偏移（或默认往前 3cm）→ IK 求解目标关节角
  3. 机器人缓慢移动到目标关节角
  4. 再次读取关节角 → FK 得到实际到达位置
  5. 打印目标 vs 实际误差

用法：
    python teleop_service/test_ik.py --port can0 --no-calibrate
    python teleop_service/test_ik.py --port can0 --no-calibrate --dx 0.03 --dz 0.02
    python teleop_service/test_ik.py --port can0 --no-calibrate --return   # 执行后回到初始位置
"""

import argparse
import sys
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

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
    piper_urdf_joints_to_sdk_joints,
)


# ---------- IK（数值雅可比 DLS） ----------
DAMPING = 0.05
EPS_DEG = 0.1


def fk_pos(kin, q_urdf_deg):
    return kin.forward_kinematics(q_urdf_deg)[:3, 3].copy()


def fk_full(kin, q_urdf_deg):
    return kin.forward_kinematics(q_urdf_deg)


def num_jacobian(kin, q_deg):
    base = fk_pos(kin, q_deg)
    J = np.zeros((3, len(q_deg)))
    eps_rad = np.deg2rad(EPS_DEG)
    for i in range(len(q_deg)):
        q2 = q_deg.copy()
        q2[i] += EPS_DEG
        J[:, i] = (fk_pos(kin, q2) - base) / eps_rad
    return J


def ik(kin, target_pos, q_guess_urdf_deg, max_iters=50, pos_tol=1e-3, max_step_deg=5.0):
    q = q_guess_urdf_deg.copy()
    for it in range(max_iters):
        err = target_pos - fk_pos(kin, q)
        if np.linalg.norm(err) < pos_tol:
            break
        J = num_jacobian(kin, q)
        dq_rad = J.T @ np.linalg.solve(J @ J.T + DAMPING ** 2 * np.eye(3), err)
        q = q + np.clip(np.rad2deg(dq_rad), -max_step_deg, max_step_deg)
    final_err = np.linalg.norm(target_pos - fk_pos(kin, q))
    return q, final_err


# ---------- 慢速分步发送 ----------
def slow_move(robot, q_curr_sdk, q_target_sdk, step_deg=2.0, dt=0.1):
    """从 q_curr_sdk 分步插值到 q_target_sdk，每步 step_deg，间隔 dt 秒。"""
    max_diff = np.abs(q_target_sdk - q_curr_sdk).max()
    n_steps = max(1, int(np.ceil(max_diff / step_deg)))
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        q_interp = q_curr_sdk + alpha * (q_target_sdk - q_curr_sdk)
        action = {f'{name}.pos': float(q_interp[j]) for j, name in enumerate(PIPER_SDK_MOTOR_NAMES)}
        robot.send_action(action)
        time.sleep(dt)


def read_joints_sdk(robot):
    obs = robot._read_raw_observation()
    return np.array([obs[f'{name}.pos'] for name in PIPER_SDK_MOTOR_NAMES])


def print_pose(label, pos, quat_xyzw=None):
    print(f'  {label}: pos = [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]', end='')
    if quat_xyzw is not None:
        print(f'  quat = [{quat_xyzw[0]:+.3f}, {quat_xyzw[1]:+.3f}, {quat_xyzw[2]:+.3f}, {quat_xyzw[3]:+.3f}]', end='')
    print()


def main():
    parser = argparse.ArgumentParser(description='Piper IK 验证测试')
    parser.add_argument('--port', default='can0')
    parser.add_argument('--no-calibrate', action='store_true')
    parser.add_argument('--dx', type=float, default=0.03, help='x 方向偏移 (m)，默认 +3cm（往前）')
    parser.add_argument('--dy', type=float, default=0.0,  help='y 方向偏移 (m)')
    parser.add_argument('--dz', type=float, default=0.0,  help='z 方向偏移 (m)')
    parser.add_argument('--step-deg', type=float, default=2.0, help='慢速移动每步角度 (deg)')
    parser.add_argument('--return', dest='do_return', action='store_true',
                        help='执行完毕后返回初始位置')
    args = parser.parse_args()

    offset = np.array([args.dx, args.dy, args.dz])

    print('=' * 55)
    print('Piper IK 验证测试')
    print('=' * 55)

    # 初始化运动学
    kin = make_piper_kinematics()

    # 连接机器人
    print(f'\n连接 Piper (CAN: {args.port})...')
    config = PiperFollowerConfig(
        id='piper',
        port=args.port,
        require_calibration=not args.no_calibrate,
        sync_gripper=True,
        enable_on_connect=True,
        cameras={},
    )
    robot = PiperFollower(config)
    robot.connect(calibrate=not args.no_calibrate)
    print('已连接\n')

    try:
        # ===== Step 1: 读当前位置 =====
        print('─' * 40)
        print('Step 1: 读取当前关节角 → FK')
        time.sleep(0.3)
        q_init_sdk = read_joints_sdk(robot)
        q_init_urdf = piper_sdk_joints_to_urdf_joints(q_init_sdk)
        T_init = fk_full(kin, q_init_urdf)
        pos_init = T_init[:3, 3]
        quat_init = R.from_matrix(T_init[:3, :3]).as_quat()

        print(f'  初始关节角 SDK (deg): {np.round(q_init_sdk, 3)}')
        print(f'  初始关节角URDF (deg): {np.round(q_init_urdf, 3)}')
        print_pose('初始末端位置', pos_init, quat_init)

        # ===== Step 2: 计算目标，求 IK =====
        print()
        print('─' * 40)
        print(f'Step 2: 目标偏移 = {offset} m → IK 求解')
        pos_target = pos_init + offset
        print(f'  目标末端位置: [{pos_target[0]:+.4f}, {pos_target[1]:+.4f}, {pos_target[2]:+.4f}]')

        q_target_urdf, ik_err = ik(kin, pos_target, q_init_urdf)
        q_target_sdk = piper_urdf_joints_to_sdk_joints(q_target_urdf)
        pos_ik_verify = fk_pos(kin, q_target_urdf)

        print(f'  IK 收敛误差:    {ik_err * 1000:.2f} mm')
        print(f'  FK(IK结果) pos: [{pos_ik_verify[0]:+.4f}, {pos_ik_verify[1]:+.4f}, {pos_ik_verify[2]:+.4f}]')
        print(f'  目标关节角 SDK: {np.round(q_target_sdk, 3)}')
        print(f'  关节角变化量:   {np.round(q_target_sdk - q_init_sdk, 3)}')

        if ik_err > 5e-3:
            print(f'\n  警告：IK 误差 {ik_err*1000:.1f}mm 较大，目标位置可能超出工作空间。继续执行？[y/N] ', end='')
            if input().strip().lower() != 'y':
                print('已取消')
                return

        # ===== Step 3: 执行移动 =====
        print()
        print('─' * 40)
        print('Step 3: 缓慢移动到目标关节角...')
        slow_move(robot, q_init_sdk, q_target_sdk, step_deg=args.step_deg)
        print('  移动完成，等待稳定 1s...')
        time.sleep(1.0)

        # ===== Step 4: 读取实际到达位置 =====
        print()
        print('─' * 40)
        print('Step 4: 读取实际到达关节角 → FK')
        q_actual_sdk = read_joints_sdk(robot)
        q_actual_urdf = piper_sdk_joints_to_urdf_joints(q_actual_sdk)
        T_actual = fk_full(kin, q_actual_urdf)
        pos_actual = T_actual[:3, 3]
        quat_actual = R.from_matrix(T_actual[:3, :3]).as_quat()

        print(f'  实际关节角 SDK: {np.round(q_actual_sdk, 3)}')
        print_pose('实际末端位置', pos_actual, quat_actual)

        # ===== Step 5: 汇总误差 =====
        print()
        print('─' * 40)
        print('结果汇总:')
        print(f'  目标位置:   [{pos_target[0]:+.4f}, {pos_target[1]:+.4f}, {pos_target[2]:+.4f}]')
        print(f'  实际位置:   [{pos_actual[0]:+.4f}, {pos_actual[1]:+.4f}, {pos_actual[2]:+.4f}]')
        err_vec = pos_actual - pos_target
        err_norm = np.linalg.norm(err_vec)
        print(f'  误差向量:   [{err_vec[0]:+.4f}, {err_vec[1]:+.4f}, {err_vec[2]:+.4f}]')
        print(f'  误差大小:   {err_norm * 1000:.2f} mm')

        if err_norm < 0.005:
            print('  ✓ IK 正确（误差 < 5mm）')
        elif err_norm < 0.015:
            print('  ⚠ IK 基本正确（误差 5~15mm，可能有机械间隙或速度未到位）')
        else:
            print('  ✗ IK 误差较大（> 15mm），需排查')

        # ===== 可选：返回初始位置 =====
        if args.do_return:
            print()
            print('─' * 40)
            print('返回初始位置...')
            time.sleep(0.5)
            slow_move(robot, q_actual_sdk, q_init_sdk, step_deg=args.step_deg)
            print('  已返回初始位置')

    finally:
        robot.disconnect()
        print('\n已断开连接')


if __name__ == '__main__':
    main()
