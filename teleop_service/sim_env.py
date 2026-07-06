# PiperSim - Piper 机械臂 MuJoCo 仿真环境
#
# 在 teleop_service 中集成 MuJoCo 可视化仿真，
# 接收遥操作指令（关节角或末端位姿），驱动 Piper 仿真模型。

import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np


class PiperSim:
    """
    Piper 机械臂 MuJoCo 仿真环境

    功能：
    - 加载 Piper MuJoCo 模型并显示可视化窗口
    - 通过命令队列接收关节角目标（由 IK 求解器输出）
    - 使用 position actuator（PD 控制器）跟踪目标关节角
    - 通过 FK 计算末端位姿，作为观测返回给遥操作控制器

    用法：
        sim = PiperSim()
        # 在后台线程中运行 Flask 服务后，调用：
        sim.run()  # 阻塞，显示 MuJoCo 可视化窗口
    """

    def __init__(self, model_path=None):
        # 模型路径
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), 'models', 'scene.xml'
            )

        # 加载 MuJoCo 模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 启用重力补偿
        self.model.body_gravcomp[:] = 1.0

        # 命令队列（从控制线程 → 物理线程）
        self.command_queue = []
        self.command_lock = threading.Lock()
        self.reset_requested = False

        # 观测状态（由物理线程更新，控制线程读取）
        self.obs_state = {
            'arm_pos': np.zeros(3),
            'arm_quat': np.array([0.0, 0.0, 0.0, 1.0]),
            'gripper_pos': np.zeros(1),
            'joint_angles': self.data.qpos.copy(),
        }
        self.obs_lock = threading.Lock()

        # 末端 site
        self.site_id = self.model.site('ee_site').id
        self.site_quat = np.empty(4)

        # 关节目标（retract 姿态）
        self.arm_joint_ids = [
            self.model.joint(f'joint{i}').id for i in range(1, 7)
        ]
        self.arm_joint_qposadr = [
            self.model.jnt_qposadr[jid] for jid in self.arm_joint_ids
        ]
        self.gripper_joint_ids = [
            self.model.joint('joint7').id,
            self.model.joint('joint8').id,
        ]
        self.gripper_joint_qposadr = [
            self.model.jnt_qposadr[jid] for jid in self.gripper_joint_ids
        ]
        self.target_qpos = self.data.qpos.copy()
        self.filtered_target_qpos = self.data.qpos.copy()
        self.joint_smoothing = np.array([0.18, 0.18, 0.18, 0.08, 0.12, 0.12])
        self.max_joint_step = np.deg2rad(
            np.array([1.0, 1.0, 1.0, 0.35, 0.7, 0.7])
        )

        # 重置到 home 姿态
        self._reset_data()

        # 初始化观测
        mujoco.mj_forward(self.model, self.data)
        self._update_obs()

    def _reset_data(self):
        """在仿真线程内重置 MuJoCo data。"""
        mujoco.mj_resetData(self.model, self.data)
        if self.model.key('home').qpos is not None:
            self.data.qpos[:] = self.model.key('home').qpos
        self.data.ctrl[:] = self.data.qpos[:6]
        self.target_qpos[:] = self.data.qpos
        self.filtered_target_qpos[:] = self.data.qpos
        mujoco.mj_forward(self.model, self.data)
        self._update_obs()

    def reset(self):
        """请求重置到 home 姿态。可从 Flask/SocketIO 线程安全调用。"""
        with self.command_lock:
            self.reset_requested = True

    def send_command(self, action):
        """
        发送控制指令

        Args:
            action: dict
                - 'joint_angles': np.array [8]，关节角目标（含夹爪）
                - 'gripper_pos': np.array [1]，夹爪开合度（可选）
        """
        with self.command_lock:
            if 'joint_angles' in action:
                ja = action['joint_angles']
                # 臂关节目标
                for i, qadr in enumerate(self.arm_joint_qposadr):
                    self.target_qpos[qadr] = ja[qadr]
                # 夹爪目标
                if len(ja) > 6:
                    for qadr in self.gripper_joint_qposadr:
                        self.target_qpos[qadr] = ja[qadr]

            if 'gripper_pos' in action:
                gp = action['gripper_pos'][0]
                for qadr in self.gripper_joint_qposadr:
                    self.target_qpos[qadr] = -gp * 0.038

    def _update_obs(self):
        """更新末端位姿观测（在物理线程中调用）"""
        site_pos = self.data.site(self.site_id).xpos.copy()
        site_mat = self.data.site(self.site_id).xmat
        mujoco.mju_mat2Quat(self.site_quat, site_mat)
        arm_quat = np.array([
            self.site_quat[1], self.site_quat[2],
            self.site_quat[3], self.site_quat[0],
        ])  # (w,x,y,z) → (x,y,z,w)

        with self.obs_lock:
            self.obs_state['arm_pos'] = site_pos
            self.obs_state['arm_quat'] = arm_quat
            # 夹爪开合度：joint range [0, -0.038]
            gripper_qpos = self.data.qpos[self.gripper_joint_qposadr[0]]
            self.obs_state['gripper_pos'] = np.array(
                [abs(gripper_qpos) / 0.038]
            )
            self.obs_state['joint_angles'] = self.data.qpos.copy()

    def _apply_target_qpos(self):
        """Move displayed joints toward targets with a per-frame rate limit."""
        qpos = self.data.qpos.copy()
        self.filtered_target_qpos[:6] += (
            self.joint_smoothing *
            (self.target_qpos[:6] - self.filtered_target_qpos[:6])
        )
        self.filtered_target_qpos[self.gripper_joint_qposadr] = self.target_qpos[
            self.gripper_joint_qposadr
        ]
        arm_delta = self.filtered_target_qpos[:6] - qpos[:6]
        qpos[:6] += np.clip(
            arm_delta, -self.max_joint_step, self.max_joint_step
        )
        qpos[self.gripper_joint_qposadr] = self.filtered_target_qpos[
            self.gripper_joint_qposadr
        ]
        self.data.qpos[:] = qpos

    def get_obs(self):
        """获取当前观测（在控制线程中调用）"""
        with self.obs_lock:
            return {
                'base_pose': np.zeros(3),
                'arm_pos': self.obs_state['arm_pos'].copy(),
                'arm_quat': self.obs_state['arm_quat'].copy(),
                'gripper_pos': self.obs_state['gripper_pos'].copy(),
                'joint_angles': self.obs_state['joint_angles'].copy(),
            }

    def run(self):
        """
        运行仿真（阻塞，必须在主线程调用以显示 viewer）

        该窗口用于遥操作可视化，采用运动学更新而不是物理积分，避免
        Piper mesh/inertia 参数导致 MuJoCo 数值发散。
        """
        print('\n' + '=' * 50)
        print('Piper MuJoCo 仿真已启动')
        print('关闭 viewer 窗口退出仿真')
        print('=' * 50 + '\n')

        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            while viewer.is_running():
                with self.command_lock:
                    if self.reset_requested:
                        self._reset_data()
                        self.reset_requested = False

                    if self.command_queue:
                        commands = self.command_queue[:]
                        self.command_queue.clear()
                    else:
                        commands = []

                for cmd in commands:
                    self.send_command(cmd)

                with self.command_lock:
                    self._apply_target_qpos()

                mujoco.mj_forward(self.model, self.data)
                self._update_obs()
                viewer.sync()
                time.sleep(self.model.opt.timestep)

    def run_headless(self, duration=60.0):
        """
        无窗口运行仿真（用于测试）

        Args:
            duration: 运行时长（秒）
        """
        print(f'Piper 仿真（无窗口）运行 {duration} 秒...')
        start = time.time()

        while time.time() - start < duration:
            with self.command_lock:
                if self.reset_requested:
                    self._reset_data()
                    self.reset_requested = False

                self._apply_target_qpos()

            mujoco.mj_forward(self.model, self.data)
            self._update_obs()

            time.sleep(self.model.opt.timestep)

        print('仿真结束')
