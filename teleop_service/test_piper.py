#!/usr/bin/env python3
"""
Piper 真机基础测试脚本

测试内容：
1. 连接 PiperFollower
2. 读取关节状态
3. 发送简单动作
4. 断开连接

使用示例：
    python teleop_service/test_piper.py --port can0
    python teleop_service/test_piper.py --port can0 --no-calibrate
"""

import argparse
import sys
import os
import time
import numpy as np

# 确保项目路径可用
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(REPO_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig
from lerobot.robots.piper_follower.piper_follower import PiperFollower
from lerobot.utils.piper_sdk import PIPER_JOINT_NAMES


def main():
    parser = argparse.ArgumentParser(description='Piper 真机基础测试')
    parser.add_argument('--port', default='can0', help='CAN 端口 (默认: can0)')
    parser.add_argument('--no-calibrate', action='store_true', help='跳过校准')
    parser.add_argument('--test-action', action='store_true', help='测试发送动作')
    parser.add_argument('--joint', type=int, default=1, choices=range(1,7), help='测试关节编号 1~6 (默认: 1)')
    parser.add_argument('--delta', type=float, default=10.0, help='关节角变化量 (deg, 默认: 10)')
    
    args = parser.parse_args()
    
    print('=' * 50)
    print('Piper 真机基础测试')
    print('=' * 50)
    
    # 1. 创建配置并连接
    print(f'\n[1] 连接 Piper (CAN: {args.port})...')
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
        # 2. 读取当前状态
        print('\n[2] 读取当前关节状态...')
        obs = robot._read_raw_observation()
        print('当前关节角 (deg):')
        for name in PIPER_JOINT_NAMES:
            print(f'  {name}: {obs[f"{name}.pos"]:+8.3f}')
        print(f'  gripper: {obs.get("gripper.pos", 0.0):+8.3f}')
        
        # 3. 测试发送动作（可选）
        if args.test_action:
            print('\n[3] 测试发送动作...')
            print(f'将关节 {args.joint} 移动 +{args.delta} 度...')

            # 读取当前关节角
            current_joints = {
                f'{name}.pos': obs[f'{name}.pos']
                for name in PIPER_JOINT_NAMES
            }

            # 修改指定关节
            joint_key = f'joint_{args.joint}.pos'
            target_joints = current_joints.copy()
            target_joints[joint_key] += args.delta
            print(f'  命令: {joint_key}: {current_joints[joint_key]:+.3f} → {target_joints[joint_key]:+.3f}')

            # 发送动作
            robot.send_action(target_joints)
            print('✓ 动作已发送')

            # 等待执行
            print('等待 2 秒...')
            time.sleep(2.0)

            # 重新读取状态
            obs2 = robot._read_raw_observation()
            print('移动后关节角 (deg):')
            for name in PIPER_JOINT_NAMES:
                delta = obs2[f'{name}.pos'] - obs[f'{name}.pos']
                marker = ' <-- 目标关节' if name == f'joint_{args.joint}' else ''
                print(f'  {name}: {obs[f"{name}.pos"]:+8.3f} → {obs2[f"{name}.pos"]:+8.3f}  (Δ={delta:+.3f}){marker}')

            # 恢复到原始位置
            print('\n恢复到原始位置...')
            robot.send_action(current_joints)
            time.sleep(1.0)
            print('✓ 已恢复')
        else:
            print('\n[3] 跳过动作测试（使用 --test-action 启用）')
        
        print('\n' + '=' * 50)
        print('测试完成！')
        print('=' * 50)
        
    finally:
        # 4. 断开连接
        print('\n[4] 断开连接...')
        robot.disconnect()
        print('✓ Piper 已断开')


if __name__ == '__main__':
    main()
