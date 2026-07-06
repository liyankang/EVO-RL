# TeleopController - 将 WebXR 位姿转换为机器人控制指令
#
# 核心功能：
# - 坐标系转换：WebXR → 机器人坐标系
# - 增量控制：基于参考位姿的相对增量
# - 双设备支持：主设备控制臂/底盘，副设备只能控制底盘

import math
import numpy as np
from scipy.spatial.transform import Rotation as R

# iPhone 摄像头相对于设备中心的偏移
DEVICE_CAMERA_OFFSET = np.array([0.0, 0.02, -0.04])

TWO_PI = 2 * math.pi
ARM_ORIENTATION_GAIN = 0.5   # 手机旋转 → EE 目标旋转的缩放系数，<1 表示手机转更多才触发等量末端旋转
ARM_POS_SMOOTHING = 0.5    # 位置 EMA 系数 (0~1)，越大越灵敏，越小越平滑。10Hz 下 0.25 约 0.4s 追上 70%
ARM_ROT_SMOOTHING = 0.5      # 旋转 EMA 系数，同上
GRIPPER_SMOOTHING = 1.0     # 夹爪 EMA 系数


def convert_webxr_pose(pos, quat):
    """
    将 WebXR 坐标系转换为机器人坐标系。
    
    WebXR: +x 右, +y 上, +z 后
    Robot: +x 前, +y 左, +z 上
    
    Args:
        pos: dict with keys 'x', 'y', 'z'
        quat: dict with keys 'x', 'y', 'z', 'w'
    
    Returns:
        pos: np.array [x, y, z] in robot frame
        rot: scipy.spatial.transform.Rotation
    """
    # 坐标轴映射
    pos = np.array([-pos['z'], -pos['x'], pos['y']], dtype=np.float64)
    rot = R.from_quat([-quat['z'], -quat['x'], quat['y'], quat['w']])
    
    # 补偿摄像头偏移（绕设备中心旋转而非摄像头）
    pos = pos + rot.apply(DEVICE_CAMERA_OFFSET)
    
    return pos, rot


class TeleopController:
    """
    遥操作控制器（对齐 policies.py 原始实现）
    
    将手机 WebXR 位姿转换为机器人目标位姿（底盘 + 机械臂 + 夹爪）。
    支持主/副双设备，主设备控制臂或底盘，副设备只能控制底盘。
    """
    
    def __init__(self):
        # 设备 ID 管理
        self.primary_device_id = None    # 主设备：控制臂或底盘
        self.secondary_device_id = None  # 副设备：只能控制底盘
        self.enabled_counts = {}
        
        # 底盘当前位姿
        self.base_pose = None
        
        # 目标位姿（初始化标记）
        self.targets_initialized = False
        self.base_target_pose = None
        self.arm_target_pos = None
        self.arm_target_rot = None
        self.gripper_target_pos = None
        self.arm_command_pos = None
        self.arm_command_rot = None
        self.gripper_command_pos = None
        
        # WebXR 参考位姿（触摸开始时记录）
        self.base_xr_ref_pos = None
        self.base_xr_ref_rot_inv = None
        self.arm_xr_ref_pos = None
        self.arm_xr_ref_rot_inv = None
        
        # 机器人参考位姿（触摸开始时记录）
        self.base_ref_pose = None
        self.arm_ref_pos = None
        self.arm_ref_rot = None
        self.arm_ref_base_pose = None  # 副设备控制底盘时的补偿
        self.gripper_ref_pos = None
    
    def process_message(self, data):
        """
        处理来自前端的消息。
        
        Args:
            data: dict，包含以下字段：
                - device_id: str
                - teleop_mode: 'arm' | 'base' (可选)
                - position: dict {x, y, z}
                - orientation: dict {x, y, z, w}
                - gripper_delta: float (arm 模式)
        """
        if not self.targets_initialized:
            return
        
        # Use device ID to disambiguate between primary and secondary devices
        device_id = data['device_id']
        
        # Update enabled count for the device that sent this message
        self.enabled_counts[device_id] = (
            self.enabled_counts.get(device_id, 0) + 1 
            if 'teleop_mode' in data else 0
        )
        
        # Assign primary and secondary devices
        if self.enabled_counts[device_id] > 2:
            if self.primary_device_id is None and device_id != self.secondary_device_id:
                # Note: We skip the first 2 steps because WebXR pose updates have higher latency than touch events
                self.primary_device_id = device_id
            elif self.secondary_device_id is None and device_id != self.primary_device_id:
                self.secondary_device_id = device_id
        elif self.enabled_counts[device_id] == 0:
            if device_id == self.primary_device_id:
                self.primary_device_id = None  # Primary device no longer enabled
                self.base_xr_ref_pos = None
                self.arm_xr_ref_pos = None
            elif device_id == self.secondary_device_id:
                self.secondary_device_id = None
                self.base_xr_ref_pos = None
        
        # Teleop is enabled
        if self.primary_device_id is not None and 'teleop_mode' in data:
            pos, rot = convert_webxr_pose(data['position'], data['orientation'])
            
            # Base movement
            if data['teleop_mode'] == 'base' or device_id == self.secondary_device_id:
                # Store reference poses
                if self.base_xr_ref_pos is None:
                    self.base_ref_pose = self.base_pose.copy()
                    self.base_xr_ref_pos = pos[:2]
                    self.base_xr_ref_rot_inv = rot.inv()
                
                # Position
                self.base_target_pose[:2] = self.base_ref_pose[:2] + (pos[:2] - self.base_xr_ref_pos)
                
                # Orientation
                base_fwd_vec_rotated = (rot * self.base_xr_ref_rot_inv).apply([1.0, 0.0, 0.0])
                base_target_theta = self.base_ref_pose[2] + math.atan2(
                    base_fwd_vec_rotated[1], base_fwd_vec_rotated[0]
                )
                self.base_target_pose[2] += (
                    (base_target_theta - self.base_target_pose[2] + math.pi) % TWO_PI - math.pi
                )
            
            # Arm movement
            elif data['teleop_mode'] == 'arm':
                # Store reference poses
                if self.arm_xr_ref_pos is None:
                    self.arm_xr_ref_pos = pos
                    self.arm_xr_ref_rot_inv = rot.inv()
                    self.arm_ref_pos = self.arm_target_pos.copy()
                    self.arm_ref_rot = self.arm_target_rot
                    self.arm_ref_base_pose = self.base_pose.copy()
                    self.gripper_ref_pos = self.gripper_target_pos
                
                # Rotations around z-axis to go between global frame (base) and local frame (arm)
                z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * self.base_pose[2])
                z_rot_inv = z_rot.inv()
                ref_z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * self.arm_ref_base_pose[2])
                
                # Position
                pos_diff = pos - self.arm_xr_ref_pos
                pos_diff += ref_z_rot.apply(self.arm_ref_pos) - z_rot.apply(self.arm_ref_pos)
                pos_diff[:2] += self.arm_ref_base_pose[:2] - self.base_pose[:2]
                self.arm_target_pos = self.arm_ref_pos + z_rot_inv.apply(pos_diff)
                
                # Orientation
                xr_delta_rot = z_rot_inv * (rot * self.arm_xr_ref_rot_inv) * ref_z_rot
                xr_delta_rot = R.from_rotvec(
                    ARM_ORIENTATION_GAIN * xr_delta_rot.as_rotvec()
                )
                self.arm_target_rot = xr_delta_rot * self.arm_ref_rot
                
                # Gripper position
                self.gripper_target_pos = np.clip(self.gripper_ref_pos + data['gripper_delta'], 0.0, 1.0)
        
        # Teleop is disabled
        elif self.primary_device_id is None:
            # Update target pose in case base is pushed while teleop is disabled
            self.base_target_pose = self.base_pose
    
    def step(self, obs):
        """
        根据当前观测更新目标位姿并返回 action。
        
        Args:
            obs: dict，包含：
                - base_pose: np.array [x, y, theta]
                - arm_pos: np.array [x, y, z]
                - arm_quat: np.array [x, y, z, w]
                - gripper_pos: np.array [pos]
        
        Returns:
            action: dict 或 None（未启用时）
                - base_pose: np.array [x, y, theta]
                - arm_pos: np.array [x, y, z]
                - arm_quat: np.array [x, y, z, w]
                - gripper_pos: np.array [pos]
        """
        # Update robot state
        self.base_pose = obs['base_pose']

        # Initialize targets
        if not self.targets_initialized:
            self.base_target_pose = obs['base_pose']
            self.arm_target_pos = obs['arm_pos'].copy()
            self.arm_target_rot = R.from_quat(obs['arm_quat'])
            self.gripper_target_pos = obs['gripper_pos'].copy()
            self.arm_command_pos = self.arm_target_pos.copy()
            self.arm_command_rot = self.arm_target_rot
            self.gripper_command_pos = self.gripper_target_pos.copy()
            self.targets_initialized = True

        # While phone is not touching, keep arm_target_pos in sync with the
        # actual robot EE so the first touch starts from the current position.
        if self.arm_xr_ref_pos is None:
            self.arm_target_pos = obs['arm_pos'].copy()
            self.arm_target_rot = R.from_quat(obs['arm_quat'])
            self.gripper_target_pos = obs['gripper_pos'].copy()
            self.arm_command_pos = self.arm_target_pos.copy()
            self.arm_command_rot = self.arm_target_rot
            self.gripper_command_pos = self.gripper_target_pos.copy()
        
        # Return no action if teleop is not enabled
        if self.primary_device_id is None:
            return None
        
        # Smooth the command sent to IK to reduce WebXR jitter and IK jumps.
        self.arm_command_pos += (
            ARM_POS_SMOOTHING * (self.arm_target_pos - self.arm_command_pos)
        )
        rot_delta = self.arm_command_rot.inv() * self.arm_target_rot
        self.arm_command_rot = (
            self.arm_command_rot *
            R.from_rotvec(ARM_ROT_SMOOTHING * rot_delta.as_rotvec())
        )
        self.gripper_command_pos += (
            GRIPPER_SMOOTHING *
            (self.gripper_target_pos - self.gripper_command_pos)
        )

        # Get most recent teleop command
        arm_quat = self.arm_command_rot.as_quat()
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)
        
        action = {
            'base_pose': self.base_target_pose.copy(),
            'arm_pos': self.arm_command_pos.copy(),
            'arm_quat': arm_quat,
            'gripper_pos': self.gripper_command_pos.copy(),
        }
        
        return action
    
    def reset(self):
        """重置控制器状态"""
        self.__init__()
