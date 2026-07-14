#!/usr/bin/env python3
"""
真机遥操作测试脚本 - 通过 WebXR 手机控制 Piper 真机

流程：
1. 连接 PiperFollower（CAN 总线）
2. 启动 TeleopServer（Flask + SocketIO）
3. 手机 WebXR → 末端位姿 → IK(placo) → 关节角(deg) → PiperFollower
4. PiperFollower 关节角(deg) → FK(placo) → 末端位姿 → TeleopController

使用示例：
    python teleop_service/real_robot.py
    python teleop_service/real_robot.py --port can0
    python teleop_service/real_robot.py --port can0 --no-calibrate
    python teleop_service/real_robot.py --port can0 --server-port 8080
"""

import argparse
import sys
import os
import threading
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

# teleop_service 内的模块
from server import TeleopServer


class PiperRealRobot:
    """
    Piper 真机封装器

    将 PiperFollower（CAN 控制）包装为 TeleopServer 所需的接口：
    - get_obs(): 读取关节角(deg) → FK(placo) → 末端位姿 → obs dict
    - send_action(action): 接收关节角 → 发送到 PiperFollower
    - reset(): 重置到 home 姿态

    使用 Piper 自带运动学库（placo + URDF），不依赖 MuJoCo，
    关节角全程使用 deg 单位，无 SDK↔URDF 手动转换问题。
    """

    def __init__(
        self,
        can_port: str = 'can0',
        require_calibration: bool = True,
        speed_ratio: int = 30,
    ):
        # 初始化 Piper 运动学（基于 URDF + placo）
        print('初始化 Piper 运动学求解器 (placo)...')
        self.kinematics = make_piper_kinematics()

        # 创建 PiperFollower 配置
        config = PiperFollowerConfig(
            id='piper',
            port=can_port,
            require_calibration=require_calibration,
            sync_gripper=True,
            enable_on_connect=True,
            cameras={},  # 遥操作不需要摄像头
            speed_ratio=speed_ratio,
        )

        # 创建并连接机器人
        print(f'正在连接 Piper (CAN: {can_port})...')
        self.robot = PiperFollower(config)
        self.robot.connect(calibrate=require_calibration)
        print('Piper 已连接')

        # 读取校准中性点（raw deg），用于将原始观测转换到校准坐标系
        # send_action 的 _offset_to_target 做 target = home_deg + offset
        # 所以观测必须减去 home_deg 才能与指令在同一空间
        self._neutral_sdk_deg = np.zeros(6)
        if self.robot.is_calibrated:
            for i, name in enumerate(PIPER_SDK_MOTOR_NAMES):
                cal_key = f'{name}.pos'
                if cal_key in self.robot.calibration:
                    cal = self.robot.calibration[cal_key]
                    self._neutral_sdk_deg[i] = float(cal.homing_offset) / float(config.calibration_scale)
            print(f'校准中性点 raw SDK (deg): {np.round(self._neutral_sdk_deg, 2)}')
        else:
            print('警告: 未校准，中性点全为 0')

        # 读取当前关节角作为 home 姿态 (SDK deg, calibrated space)
        raw_obs = self.robot._read_raw_observation()
        # 用户指定的初始位置 (SDK deg) — 手臂自然前伸，末端在底座前上方
        # j2=90, j3=-90: FK pos ≈ [0.377, 0, 0.457]
        self.home_joints_sdk_deg = np.array([
             7.676,   # joint_1
            83.899,   # joint_2
           -26.100,   # joint_3
           -12.000,   # joint_4
             8.120,   # joint_5  有效范围 [-45, 45]，取中点
           -36.701,   # joint_6
        ])

        
        self.home_gripper_deg = 0.600
        print(f'Home 关节角 SDK (deg): {self.home_joints_sdk_deg}')

        # 当前 IK 猜测 (URDF deg)，用于增量 IK
        self.q_curr_urdf_deg = piper_sdk_joints_to_urdf_joints(self.home_joints_sdk_deg)

        # 指令平滑：上一次实际下发的关节角（URDF deg），用于 EMA 防止抖动
        self._cmd_urdf_deg = self.q_curr_urdf_deg.copy()
        self._cmd_smoothing = 0.45  # EMA 系数，越小越平滑，越大越灵敏

        # 调试计数器
        self._obs_step = 0
        self._action_step = 0

        # 线程锁
        self._lock = threading.Lock()

        # 启动时自动 reset 到 home
        self.reset()

    def _read_joints_sdk_deg(self):
        """读取当前关节角 (SDK deg, calibrated space)

        _read_raw_observation 返回原始电机角度，但 send_action 的
        _offset_to_target 会加上校准中性点。为保持一致，这里减去中性点。
        """
        with self._lock:
            raw_obs = self.robot._read_raw_observation()
        joints_sdk_deg = np.array([
            raw_obs[f'{name}.pos'] - self._neutral_sdk_deg[i]
            for i, name in enumerate(PIPER_SDK_MOTOR_NAMES)
        ])
        gripper_raw = raw_obs.get('gripper.pos', 0.0)
        gripper_pos = gripper_raw
        # 夹爪中性点可能不同，用单独的 gripper neutral
        if self.robot.is_calibrated and 'gripper.pos' in self.robot.calibration:
            gcal = self.robot.calibration['gripper.pos']
            g_neutral = float(gcal.homing_offset) / float(self.robot.config.calibration_scale)
            gripper_pos = gripper_raw - g_neutral
        return joints_sdk_deg, gripper_pos

    def _fk(self, q_urdf_deg):
        """
        正运动学：URDF 关节角 (deg) → 末端位姿

        Returns:
            pos: [x, y, z]
            quat: [x, y, z, w] (TeleopController 需要的格式)
        """
        transform = self.kinematics.forward_kinematics(q_urdf_deg)
        pos = transform[:3, 3].copy()
        rot_mat = transform[:3, :3]
        quat_xyzw = R.from_matrix(rot_mat).as_quat()  # [x, y, z, w]
        return pos, quat_xyzw

    # URDF 关节限位 (deg)
    JOINT_LO = np.array([-150.0,   0.0, -170.0, -100.0, -45.0, -120.0])
    JOINT_HI = np.array([ 150.0, 180.0,    0.0,  100.0,  45.0,  120.0])

    def _ik(self, target_pos, target_quat, q_guess_urdf_deg, max_iters=80, pos_tol=1e-3, rot_tol=1e-2, max_step_deg=5.0):
        """
        6-DOF 逆运动学：末端位姿（位置+姿态） → URDF 关节角 (deg)

        DLS IK + 零空间投影。位置和姿态误差分别加权，
        位置权重高保证末端到位，姿态权重低防止 j4/j6 过度补偿。
        """
        DAMPING_POS = 0.05
        DAMPING_ROT = 0.1    # 姿态阻尼，比位置大但不过度抑制
        EPS_DEG = 0.1
        NS_GAIN = 0.05
        ROT_WEIGHT = 0.7     # 姿态权重，0.7 平衡位置跟踪和姿态响应
        # 期望中间值（URDF deg），趋向自然前伸姿态
        Q_MID = np.array([0.0, 90.0, -90.0, 0.0, 0.0, 0.0])

        def _fk_full(q_deg):
            T = self.kinematics.forward_kinematics(q_deg)
            return T[:3, 3].copy(), T[:3, :3].copy()

        target_rot = R.from_quat(target_quat)
        q = np.clip(q_guess_urdf_deg.copy(), self.JOINT_LO, self.JOINT_HI)

        for _ in range(max_iters):
            pos_cur, rot_cur = _fk_full(q)
            rot_cur_r = R.from_matrix(rot_cur)

            # 位置误差
            pos_err = target_pos - pos_cur
            # 姿态误差 (angle-axis)
            rot_err = (target_rot * rot_cur_r.inv()).as_rotvec()

            # 加权误差：位置直接跟踪，姿态降权
            err = np.concatenate([pos_err, ROT_WEIGHT * rot_err])

            pos_ok = np.linalg.norm(pos_err) < pos_tol
            rot_ok = np.linalg.norm(rot_err) < rot_tol
            if pos_ok and rot_ok:
                break

            # 数值雅可比 6×6
            J = np.zeros((6, len(q)))
            eps_rad = np.deg2rad(EPS_DEG)
            for i in range(len(q)):
                q2 = q.copy()
                q2[i] += EPS_DEG
                p2, r2 = _fk_full(q2)
                r2_r = R.from_matrix(r2)
                J[:3, i] = (p2 - pos_cur) / eps_rad
                J[3:, i] = (r2_r * rot_cur_r.inv()).as_rotvec() / eps_rad

            # 加权雅可比：姿态行乘权重
            J[3:] *= ROT_WEIGHT

            # DLS 伪逆，位置和姿态用不同阻尼
            W = np.diag([1, 1, 1, DAMPING_ROT**2, DAMPING_ROT**2, DAMPING_ROT**2])
            JT = J.T
            J_pinv = JT @ np.linalg.solve(J @ JT + W, np.eye(6))

            # 主任务
            dq_main_rad = J_pinv @ err
            # 零空间：趋向中间值
            null_proj = np.eye(len(q)) - J_pinv @ J
            dq_null_rad = null_proj @ (NS_GAIN * np.deg2rad(Q_MID - q))

            dq_rad = dq_main_rad + dq_null_rad
            q = q + np.clip(np.rad2deg(dq_rad), -max_step_deg, max_step_deg)
            q = np.clip(q, self.JOINT_LO, self.JOINT_HI)

        return q

    def get_obs(self):
        """
        获取机器人当前观测（供 TeleopController 使用）

        Returns:
            dict: {
                'base_pose': [0, 0, 0],
                'arm_pos': [x, y, z],
                'arm_quat': [x, y, z, w],
                'gripper_pos': [pos],
                'joint_angles': q_urdf_deg (6-dim URDF deg，用于 IK 初始猜测),
            }
        """
        joints_sdk_deg, gripper_pos = self._read_joints_sdk_deg()

        # SDK → URDF
        q_urdf_deg = piper_sdk_joints_to_urdf_joints(joints_sdk_deg)

        # 同步实际关节角，send_action 安全限幅用真实状态做回退基准
        self.q_curr_urdf_deg = q_urdf_deg.copy()

        # FK 计算末端位姿
        arm_pos, arm_quat = self._fk(q_urdf_deg)

        self._obs_step += 1
        if self._obs_step % 10 == 1:
            print(f'[get_obs#{self._obs_step}] '
                  f'joints(deg)={np.round(q_urdf_deg, 2)} '
                  f'FK pos={np.round(arm_pos, 4)} '
                  f'quat={np.round(arm_quat, 4)} '
                  f'gripper={gripper_pos:.3f}')

        return {
            'base_pose': np.zeros(3),
            'arm_pos': arm_pos,
            'arm_quat': arm_quat,
            'gripper_pos': np.array([gripper_pos]),
            'joint_angles': q_urdf_deg,
        }

    def send_action(self, action):
        """
        发送动作指令到 Piper

        action 格式（placo IK 输出）:
            {
                'base_pose': ...,
                'joint_angles': q_target (6-dim URDF deg),
                'gripper_pos': [pos],
                'q_curr_urdf_deg': 当前实际关节角 (URDF deg，可选，用于安全截断基准)
            }
        """
        q_urdf_deg = action['joint_angles']

        # 安全限制：以实际关节角为基准，避免突变
        max_step_deg = 30.0
        q_base = action.get('q_curr_urdf_deg', self.q_curr_urdf_deg)
        dq = np.clip(q_urdf_deg - q_base, -max_step_deg, max_step_deg)
        q_safe_urdf_deg = q_base + dq

        # EMA 平滑：对下发指令做指数移动平均，消除高频抖动
        self._cmd_urdf_deg += self._cmd_smoothing * (q_safe_urdf_deg - self._cmd_urdf_deg)
        q_smooth_urdf_deg = self._cmd_urdf_deg

        # URDF → SDK
        joints_sdk_deg = piper_urdf_joints_to_sdk_joints(q_smooth_urdf_deg)

        self._action_step += 1
        if self._action_step % 10 == 1:
            gripper_val = action.get('gripper_pos', [0])[0]
            print(f'[send_action#{self._action_step}] '
                  f'IK out(deg)={np.round(q_urdf_deg, 2)} '
                  f'safe(deg)={np.round(q_safe_urdf_deg, 2)} '
                  f'smooth(deg)={np.round(q_smooth_urdf_deg, 2)} '
                  f'SDK(deg)={np.round(joints_sdk_deg, 2)} '
                  f'gripper={gripper_val:.3f}')

        # 构建 PiperFollower action dict
        piper_action = {}
        for i, name in enumerate(PIPER_SDK_MOTOR_NAMES):
            piper_action[f'{name}.pos'] = float(joints_sdk_deg[i])

        # 夹爪
        if 'gripper_pos' in action:
            piper_action['gripper.pos'] = float(action['gripper_pos'][0])

        with self._lock:
            self.robot.send_action(piper_action)

    def reset(self):
        """重置到 home 姿态（慢速分步移动）"""
        print('重置到 home 姿态...')
        q_home_sdk = self.home_joints_sdk_deg.copy()
        q_home_urdf = piper_sdk_joints_to_urdf_joints(q_home_sdk)

        # 先保持当前位置，等机械臂停稳（遥操作结束后可能还在运动）
        joints_sdk_deg, gripper_pos = self._read_joints_sdk_deg()
        hold_action = {f'{name}.pos': float(joints_sdk_deg[i]) for i, name in enumerate(PIPER_SDK_MOTOR_NAMES)}
        hold_action['gripper.pos'] = gripper_pos
        with self._lock:
            self.robot.send_action(hold_action)
        time.sleep(0.8)

        # 重新读取稳定后的关节角，作为插值起点
        joints_sdk_deg, _ = self._read_joints_sdk_deg()
        q_curr_urdf = piper_sdk_joints_to_urdf_joints(joints_sdk_deg)
        print(f'  插值起点(deg): {np.round(q_curr_urdf, 2)}')

        max_step = 3.0  # 每步最大 3 度，慢速回 home
        steps = int(np.abs(q_home_urdf - q_curr_urdf).max() / max_step) + 1
        for i in range(1, steps + 1):
            alpha = i / steps
            q_interp_urdf = q_curr_urdf + alpha * (q_home_urdf - q_curr_urdf)
            q_interp_sdk = piper_urdf_joints_to_sdk_joints(q_interp_urdf)
            piper_action = {f'{name}.pos': float(q_interp_sdk[j]) for j, name in enumerate(PIPER_SDK_MOTOR_NAMES)}
            piper_action['gripper.pos'] = float(self.home_gripper_deg)
            with self._lock:
                self.robot.send_action(piper_action)
            time.sleep(0.1)

        # 等待机械臂到位
        time.sleep(1.0)

        # 读取实际到位后的关节角，作为后续 IK 的基准
        joints_sdk_deg, _ = self._read_joints_sdk_deg()
        self.q_curr_urdf_deg = piper_sdk_joints_to_urdf_joints(joints_sdk_deg)
        self._cmd_urdf_deg = self.q_curr_urdf_deg.copy()  # 同步平滑状态
        print(f'重置完成，实际关节角(deg): {np.round(self.q_curr_urdf_deg, 2)}')

    def close(self):
        """断开连接"""
        print('断开 Piper 连接...')
        self.robot.disconnect()
        print('Piper 已断开')


class PiperIKBridge:
    """
    IK 桥接器

    将 TeleopServer 的 MuJoCo-based IK 替换为 Piper placo IK。
    TeleopServer 传入 use_ik=True 时，其内部 IK 输出的是 MuJoCo qpos (rad)，
    但我们需要的是 Piper placo 的 URDF deg。

    因此我们不用 TeleopServer 的内置 IK，而是自己处理：
    - TeleopServer 的 controller.step() 返回 EE pose target
    - 我们用 placo IK 转换为关节角
    - 通过 send_action 发送

    这个类包装了 TeleopServer 的 IK 行为。
    """

    def __init__(self, robot: PiperRealRobot):
        self.robot = robot
        self._step = 0

    def compute_action(self, ee_target_pos, ee_target_quat, obs):
        """
        从 EE 目标位姿计算关节角

        Args:
            ee_target_pos: [x, y, z] 目标位置
            ee_target_quat: [x, y, z, w] 目标姿态
            obs: 当前观测（包含 joint_angles 用于 IK 初始猜测）

        Returns:
            action dict with 'joint_angles' in URDF deg
        """
        self._step += 1
        q_guess = obs['joint_angles']  # URDF deg from get_obs
        q_target = self.robot._ik(ee_target_pos, ee_target_quat, q_guess)

        if self._step % 10 == 1:
            print(f'[IK bridge#{self._step}] '
                  f'EE target pos={np.round(ee_target_pos, 4)} '
                  f'quat={np.round(ee_target_quat, 4)}')
            print(f'[IK bridge#{self._step}] '
                  f'guess(deg)={np.round(q_guess, 2)} '
                  f'-> result(deg)={np.round(q_target, 2)}')

        return {
            'base_pose': np.zeros(3),
            'joint_angles': q_target,
            'gripper_pos': obs['gripper_pos'],
            'q_curr_urdf_deg': q_guess,  # 实际关节角，用于 send_action 安全截断基准
        }


def main():
    parser = argparse.ArgumentParser(description='Piper 真机 WebXR 遥操作')
    parser.add_argument('--port', default='can0', help='CAN 端口 (默认: can0)')
    parser.add_argument('--no-calibrate', action='store_true',
                        help='跳过校准（使用已有校准文件）')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    parser.add_argument('--server-port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--freq', type=float, default=60.0, help='控制频率 (Hz)')
    parser.add_argument('--message-timeout-ms', type=float, default=250.0,
                        help='消息超时时间 (ms)')
    parser.add_argument('--speed-ratio', type=int, default=70,
                        help='机械臂运动速度 0~100 (默认: 30)')

    args = parser.parse_args()

    # 创建真机封装器（使用 placo 运动学，不需要 MuJoCo）
    robot = PiperRealRobot(
        can_port=args.port,
        require_calibration=not args.no_calibrate,
        speed_ratio=args.speed_ratio,
    )

    # IK 桥接器
    ik_bridge = PiperIKBridge(robot)

    # 状态回调
    def on_state(state):
        if state == 'episode_started':
            print('Episode 开始')
        elif state == 'episode_ended':
            print('Episode 结束')
        elif state == 'reset_env':
            print('重置环境')
            robot.reset()

    # 自定义 action 回调：使用 placo IK 而非 MuJoCo IK
    # TeleopServer 的 use_ik=False 模式会直接传 EE pose，
    # 我们在 on_action 中自己做 IK
    _current_obs = {}

    def custom_get_obs():
        nonlocal _current_obs
        _current_obs = robot.get_obs()
        return _current_obs

    def custom_on_action(ee_action):
        """接收 EE pose，用 placo IK 转换为关节角"""
        action = ik_bridge.compute_action(
            ee_action['arm_pos'],
            ee_action['arm_quat'],
            _current_obs,
        )
        # 把 gripper 也带上
        action['gripper_pos'] = ee_action['gripper_pos']
        robot.send_action(action)

    # 创建遥操作服务（use_ik=False，我们自己处理 IK）
    server = TeleopServer(
        get_obs_fn=custom_get_obs,
        on_action_fn=custom_on_action,
        on_state_fn=on_state,
        host=args.host,
        port=args.server_port,
        control_freq=args.freq,
        message_timeout_ms=args.message_timeout_ms,
        use_ik=False,  # 不用 MuJoCo IK，用 placo
        arm_type='piper',
    )

    print('\n' + '=' * 50)
    print('Piper 真机遥操作服务 (placo IK)')
    print(f'CAN 端口: {args.port}')
    print(f'控制频率: {args.freq} Hz')
    print('IK: placo (URDF)')
    print('=' * 50 + '\n')

    try:
        server.run(block=True)
    except KeyboardInterrupt:
        print('\n用户中断')
    finally:
        server.stop()
        robot.close()


if __name__ == '__main__':
    main()
