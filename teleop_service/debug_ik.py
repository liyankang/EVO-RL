#!/usr/bin/env python3
"""
IK/FK 诊断脚本

测试流程：
1. 连接 Piper
2. 移动到指定初始位置
3. 读取实际关节角
4. FK 计算末端位姿
5. IK 反求关节角
6. 对比误差

使用示例：
    # 使用默认 home 姿态
    python teleop_service/debug_ik.py --port can0
    
    # 指定关节角 (deg)
    python teleop_service/debug_ik.py --port can0 --joints 0 -45 90 0 45 0
    
    # 测试多个位置
    python teleop_service/debug_ik.py --port can0 --test-multiple
"""

import argparse
import sys
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# 确保项目路径可用
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


def test_ik_fk(kinematics, q_urdf_deg, label=""):
    """测试单个位置的 IK/FK 一致性"""
    print(f"\n{'='*60}")
    print(f"测试位置: {label}")
    print(f"{'='*60}")
    
    # FK: URDF deg → 末端位姿
    transform = kinematics.forward_kinematics(q_urdf_deg)
    pos = transform[:3, 3]
    rot_mat = transform[:3, :3]
    quat_xyzw = R.from_matrix(rot_mat).as_quat()
    
    print(f"\n[FK 结果]")
    print(f"  位置: x={pos[0]:+.4f}  y={pos[1]:+.4f}  z={pos[2]:+.4f} m")
    print(f"  姿态: quat=[{quat_xyzw[0]:+.4f}, {quat_xyzw[1]:+.4f}, {quat_xyzw[2]:+.4f}, {quat_xyzw[3]:+.4f}]")
    
    # IK: 末端位姿 → URDF deg
    q_ik_urdf_deg = kinematics.inverse_kinematics(
        current_joint_pos=q_urdf_deg,
        desired_ee_pose=transform,
    )
    
    # 计算误差
    joint_error_deg = q_ik_urdf_deg - q_urdf_deg
    
    print(f"\n[IK 结果]")
    print(f"  原始关节角 (URDF deg): {q_urdf_deg}")
    print(f"  IK 求解结果 (URDF deg): {q_ik_urdf_deg}")
    print(f"  关节角误差 (deg): {joint_error_deg}")
    print(f"  最大误差: {np.max(np.abs(joint_error_deg)):.4f} deg")
    
    # FK(IK(pos)) 验证
    transform2 = kinematics.forward_kinematics(q_ik_urdf_deg)
    pos2 = transform2[:3, 3]
    pos_error = np.linalg.norm(pos2 - pos)
    
    print(f"\n[FK(IK(pos)) 验证]")
    print(f"  原始位置: {pos}")
    print(f"  FK(IK) 位置: {pos2}")
    print(f"  位置误差: {pos_error:.6f} m = {pos_error*1000:.3f} mm")
    
    # 判断是否成功
    if pos_error < 0.001:  # 1mm
        print(f"\n✓ IK/FK 一致性良好")
        return True
    else:
        print(f"\n✗ IK/FK 误差过大！")
        return False


def main():
    parser = argparse.ArgumentParser(description='IK/FK 诊断工具')
    parser.add_argument('--port', default='can0', help='CAN 端口')
    parser.add_argument('--no-calibrate', action='store_true', help='跳过校准')
    parser.add_argument('--joints', type=float, nargs=6, metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'),
                        help='指定初始关节角 (SDK deg)')
    parser.add_argument('--test-multiple', action='store_true', help='测试多个典型位置')
    
    args = parser.parse_args()
    
    print('='*60)
    print('IK/FK 诊断工具')
    print('='*60)
    
    # 初始化运动学
    print('\n[1] 初始化 Piper 运动学 (placo)...')
    kinematics = make_piper_kinematics()
    print('✓ 运动学初始化完成')
    
    # 连接机器人
    print(f'\n[2] 连接 Piper (CAN: {args.port})...')
    config = PiperFollowerConfig(
        port=args.port,
        require_calibration=not args.no_calibrate,
        sync_gripper=True,
        enable_on_connect=True,
        cameras={},
    )
    robot = PiperFollower(config)
    robot.connect(calibrate=not args.no_calibrate)
    print('✓ Piper 已连接')
    
    try:
        # 读取当前位置
        print('\n[3] 读取当前位置...')
        raw_obs = robot._read_raw_observation()
        current_joints_sdk_deg = np.array([
            raw_obs[f'{name}.pos'] for name in PIPER_SDK_MOTOR_NAMES
        ])
        print(f'当前关节角 (SDK deg): {current_joints_sdk_deg}')
        
        # 测试当前位置
        q_urdf_deg = piper_sdk_joints_to_urdf_joints(current_joints_sdk_deg)
        test_ik_fk(kinematics, q_urdf_deg, label="当前位置")
        
        # 移动到指定位置
        if args.joints is not None:
            target_joints_sdk_deg = np.array(args.joints)
            print(f'\n[4] 移动到指定位置 (SDK deg): {target_joints_sdk_deg}')
            
            # 发送目标关节角
            action = {}
            for i, name in enumerate(PIPER_SDK_MOTOR_NAMES):
                action[f'{name}.pos'] = float(target_joints_sdk_deg[i])
            robot.send_action(action)
            
            print('等待 2 秒...')
            time.sleep(2.0)
            
            # 重新读取
            raw_obs = robot._read_raw_observation()
            actual_joints_sdk_deg = np.array([
                raw_obs[f'{name}.pos'] for name in PIPER_SDK_MOTOR_NAMES
            ])
            print(f'实际关节角 (SDK deg): {actual_joints_sdk_deg}')
            
            q_urdf_deg = piper_sdk_joints_to_urdf_joints(actual_joints_sdk_deg)
            test_ik_fk(kinematics, q_urdf_deg, label="指定位置")
        
        # 测试多个典型位置
        if args.test_multiple:
            print('\n[5] 测试多个典型位置...')
            
            test_positions = [
                ("home 姿态", np.array([0, -45, 90, 0, 45, 0])),
                ("前伸", np.array([0, -30, 60, 0, 30, 0])),
                ("左侧", np.array([30, -45, 90, 0, 45, 0])),
                ("右侧", np.array([-30, -45, 90, 0, 45, 0])),
                ("高位", np.array([0, -60, 120, 0, 60, 0])),
                ("低位", np.array([0, -20, 40, 0, 20, 0])),
            ]
            
            results = []
            for label, joints_sdk_deg in test_positions:
                q_urdf_deg = piper_sdk_joints_to_urdf_joints(joints_sdk_deg)
                success = test_ik_fk(kinematics, q_urdf_deg, label=label)
                results.append((label, success))
            
            print(f'\n{"="*60}')
            print('测试结果汇总')
            print(f'{"="*60}')
            for label, success in results:
                status = "✓ 通过" if success else "✗ 失败"
                print(f'  {label:15s}: {status}')
        
        print(f'\n{"="*60}')
        print('诊断完成')
        print(f'{"="*60}')
        
    finally:
        print('\n断开连接...')
        robot.disconnect()
        print('✓ Piper 已断开')


if __name__ == '__main__':
    main()
