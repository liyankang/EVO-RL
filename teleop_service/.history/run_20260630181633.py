#!/usr/bin/env python3
"""
遥操作服务入口文件

使用示例：

1. 基本使用（需要自己实现机器人接口）：

    from teleop_service.run import run_teleop_service
    
    def get_obs():
        # 返回机器人当前状态
        return {
            'base_pose': ...,      # np.array [x, y, theta]
            'arm_pos': ...,        # np.array [x, y, z]
            'arm_quat': ...,       # np.array [x, y, z, w]
            'gripper_pos': ...,    # np.array [pos]
        }
    
    def on_action(action):
        # 执行机器人动作
        robot.step(action)
    
    def on_state(state):
        # 处理状态事件
        if state == 'reset_env':
            robot.reset()
    
    run_teleop_service(get_obs, on_action, on_state)

2. 连接真实机器人（使用 arm_server 和 base_server）：

    python run_teleop.py --real

3. 使用仿真环境：

    python run_teleop.py --sim
"""

import argparse
import sys
import time
import threading
import numpy as np


def run_teleop_service(
    get_obs_fn,
    on_action_fn=None,
    on_state_fn=None,
    host='0.0.0.0',
    port=5000,
    control_freq=10.0,
    use_ik=False,
    arm_type='piper',
    ee_offset=0.0,
    model_path=None,
):
    """
    启动遥操作服务
    
    Args:
        get_obs_fn: 获取机器人观测的函数
            返回 dict: {'base_pose': ..., 'arm_pos': ..., 'arm_quat': ..., 'gripper_pos': ...}
        on_action_fn: 接收 action 的回调（可选）
            当 use_ik=False 时:
                action: {'base_pose': ..., 'arm_pos': ..., 'arm_quat': ..., 'gripper_pos': ...}
            当 use_ik=True 时:
                action: {'base_pose': ..., 'joint_angles': ..., 'gripper_pos': ...}
        on_state_fn: 接收状态事件的回调（可选）
            state: 'episode_started' | 'episode_ended' | 'reset_env'
        host: 服务器地址
        port: 服务器端口
        control_freq: 控制频率 (Hz)
        use_ik: 是否启用 IK 求解（将末端位姿转换为关节角）
        arm_type: 机械臂类型（仅支持 'piper'）
        ee_offset: 末端执行器偏移（夹爪长度），单位 m
        model_path: MuJoCo 模型文件路径（可选，默认自动查找）
    """
    from teleop_service.server import TeleopServer
    
    server = TeleopServer(
        get_obs_fn=get_obs_fn,
        on_action_fn=on_action_fn,
        on_state_fn=on_state_fn,
        host=host,
        port=port,
        control_freq=control_freq,
        use_ik=use_ik,
        arm_type=arm_type,
        ee_offset=ee_offset,
        model_path=model_path,
    )
    
    server.run(block=True)


# =============================================================================
# 以下为示例机器人接口实现
# =============================================================================

class MockRobot:
    """模拟机器人（用于测试）"""
    
    def __init__(self):
        self.base_pose = np.zeros(3)
        self.arm_pos = np.array([0.5, 0.0, 0.4])
        self.arm_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.gripper_pos = np.array([0.0])
    
    def get_obs(self):
        return {
            'base_pose': self.base_pose.copy(),
            'arm_pos': self.arm_pos.copy(),
            'arm_quat': self.arm_quat.copy(),
            'gripper_pos': self.gripper_pos.copy(),
        }
    
    def step(self, action):
        print(f"Action: base={action['base_pose'][:2]}, arm={action['arm_pos']}, gripper={action['gripper_pos']}")
        # 更新模拟状态
        self.base_pose = action['base_pose']
        self.arm_pos = action['arm_pos']
        self.arm_quat = action['arm_quat']
        self.gripper_pos = action['gripper_pos']
    
    def reset(self):
        print("Robot reset")
        self.base_pose = np.zeros(3)
        self.arm_pos = np.array([0.5, 0.0, 0.4])
        self.arm_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.gripper_pos = np.array([0.0])


class RealRobot:
    """真实机器人（通过 RPC 连接 arm_server 和 base_server）"""
    
    def __init__(self):
        # 导入项目依赖
        sys.path.insert(0, '/'.join(__file__.split('/')[:-2]))  # 添加项目根目录到路径
        
        from constants import BASE_RPC_HOST, BASE_RPC_PORT, ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
        from arm_server import ArmManager
        from base_server import BaseManager
        
        # 连接底盘 RPC 服务器
        base_manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            base_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception('无法连接到底盘 RPC 服务器，请确保 base_server.py 正在运行') from e
        
        # 连接机械臂 RPC 服务器
        arm_manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            arm_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception('无法连接到机械臂 RPC 服务器，请确保 arm_server.py 正在运行') from e
        
        self.base = base_manager.Base(max_vel=(0.5, 0.5, 1.57), max_accel=(0.5, 0.5, 1.57))
        self.arm = arm_manager.Arm()
    
    def get_obs(self):
        obs = {}
        obs.update(self.base.get_state())
        obs.update(self.arm.get_state())
        return obs
    
    def step(self, action):
        self.base.execute_action(action)
        self.arm.execute_action(action)
    
    def reset(self):
        print('重置底盘...')
        self.base.reset()
        print('重置机械臂...')
        self.arm.reset()
        print('重置完成')
    
    def close(self):
        self.base.close()
        self.arm.close()


def main():
    parser = argparse.ArgumentParser(description='手机 WebXR 遥操作服务')
    parser.add_argument('--sim', action='store_true', help='使用模拟机器人（无可视化）')
    parser.add_argument('--sim-mj', action='store_true', help='使用 MuJoCo 仿真（Piper 可视化）')
    parser.add_argument('--real', action='store_true', help='使用真实机器人')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--freq', type=float, default=10.0, help='控制频率 (Hz)')
    parser.add_argument('--ik', action='store_true', help='启用 IK 求解（输出关节角而非末端位姿）')
    parser.add_argument('--arm-type', default='piper', choices=['piper'], help='机械臂类型')
    parser.add_argument('--ee-offset', type=float, default=0.0, help='末端执行器偏移（夹爪长度），单位 m')
    parser.add_argument('--model-path', default=None, help='MuJoCo 模型文件路径')
    
    args = parser.parse_args()
    
    # MuJoCo 仿真模式（特殊处理：viewer 必须在主线程）
    if args.sim_mj:
        from teleop_service.sim_env import PiperSim
        from teleop_service.server import TeleopServer
        
        print('使用 MuJoCo 仿真（Piper）')
        sim = PiperSim(model_path=args.model_path)
        
        # 遥操作服务（Flask 在后台线程运行）
        server = TeleopServer(
            get_obs_fn=sim.get_obs,
            on_action_fn=lambda action: sim.send_command(action),
            on_state_fn=lambda state: sim.reset() if state == 'reset_env' else None,
            host=args.host,
            port=args.port,
            control_freq=args.freq,
            use_ik=True,  # MuJoCo 需要 IK 将末端位姿转为关节角
            arm_type='piper',  # PiperSim 只支持 Piper
            ee_offset=args.ee_offset,
            model_path=args.model_path,
        )
        
        # Flask 在后台线程
        server.run(block=False)
        
        # MuJoCo viewer 在主线程（阻塞）
        sim.run()
        return
    
    # 创建机器人实例
    if args.real:
        print('使用真实机器人')
        robot = RealRobot()
    else:
        print('使用模拟机器人（添加 --real 参数使用真实机器人，--sim-mj 使用 MuJoCo 可视化）')
        robot = MockRobot()
    
    # 定义回调函数
    def get_obs():
        return robot.get_obs()
    
    def on_action(action):
        robot.step(action)
    
    def on_state(state):
        if state == 'episode_started':
            print('Episode 开始')
        elif state == 'episode_ended':
            print('Episode 结束')
        elif state == 'reset_env':
            print('重置环境')
            robot.reset()
    
    # 启动服务
    try:
        run_teleop_service(
            get_obs_fn=get_obs,
            on_action_fn=on_action,
            on_state_fn=on_state,
            host=args.host,
            port=args.port,
            control_freq=args.freq,
            use_ik=args.ik,
            arm_type=args.arm_type,
            ee_offset=args.ee_offset,
            model_path=args.model_path,
        )
    finally:
        if hasattr(robot, 'close'):
            robot.close()


if __name__ == '__main__':
    main()
