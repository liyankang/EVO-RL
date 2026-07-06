# IK/FK 求解器 - 基于 MuJoCo
#
# 支持多种机械臂：
# - Kinova Gen3 (7-DOF)
# - Piper (6-DOF + 夹爪)
#
# 提供：
# - IK：末端位姿 → 关节角（阻尼最小二乘法 + nullspace 优化）
# - FK：关节角 → 末端位姿

import os
import mujoco
import numpy as np

DAMPING_COEFF = 1e-4  # 增大阻尼，减少抖动
MAX_ANGLE_CHANGE = np.deg2rad(45)

# 预设机械臂配置
ARM_CONFIGS = {
    'piper': {
        'model_path': None,
        'site_name': 'ee_site',
        'arm_joint_names': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
        'ee_offset_default': 0.0,
        'ee_site_offset': None,  # site 位置已在模型中定义
    },
}


def _find_model_path():
    """自动查找 Piper 模型文件"""
    candidates = [
        os.path.join(os.path.dirname(__file__), 'models', 'robot.xml'),
        os.path.join(os.path.dirname(__file__), '..', 'Piper', 'robot.xml'),
    ]
    
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        f'找不到 Piper 模型文件。请指定 model_path 参数。\n'
        f'搜索路径: {candidates}'
    )


class IKSolver:
    """
    机械臂逆运动学求解器
    
    使用阻尼最小二乘法（Damped Least Squares）+ nullspace 优化，
    基于 MuJoCo 进行运动学计算。
    
    支持多种机械臂，通过 arm_type 参数指定：
    - 'gen3': Kinova Gen3 (7-DOF)
    - 'piper': Piper (6-DOF)
    
    用法：
        # Gen3
        solver = IKSolver(arm_type='gen3')
        qpos = solver.solve(target_pos, target_quat, current_qpos)
        
        # Piper
        solver = IKSolver(arm_type='piper')
        qpos = solver.solve(target_pos, target_quat, current_qpos)
    """
    
    def __init__(self, arm_type='piper', ee_offset=None, model_path=None, site_name=None):
        """
        Args:
            arm_type: 机械臂类型（仅支持 'piper'）
            ee_offset: 末端执行器偏移（夹爪长度），单位 m。None 使用默认值
            model_path: MuJoCo 模型文件路径，默认自动查找
            site_name: 末端 site 名称，默认根据 arm_type 自动选择
        """
        self.arm_type = arm_type
        config = ARM_CONFIGS.get(arm_type, {})
        
        # 确定模型路径
        if model_path is None:
            model_path = _find_model_path()
        
        # 确定 site 名称
        if site_name is None:
            site_name = config.get('site_name', 'pinch_site')
        self.site_name = site_name
        
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.model.body_gravcomp[:] = 1.0
        
        # 缓存引用
        self.qpos0 = self.model.key('home').qpos.copy()
        self.site_id = self.model.site(site_name).id
        self.site_pos = self.data.site(self.site_id).xpos
        self.site_mat = self.data.site(self.site_id).xmat
        
        # 设置末端偏移（如果有）
        if ee_offset is not None:
            ee_site_offset = config.get('ee_site_offset')
            if ee_site_offset is not None:
                offset = np.array(ee_site_offset)
                offset[2] -= ee_offset
                self.model.site(self.site_id).pos = offset
        
        # 获取臂关节索引（用于 IK 只优化臂关节）
        arm_joint_names = config.get('arm_joint_names', [])
        if arm_joint_names:
            self.arm_joint_ids = []
            self.arm_joint_qposadr = []
            self.arm_joint_dofadr = []
            for name in arm_joint_names:
                try:
                    jid = self.model.joint(name).id
                    self.arm_joint_ids.append(jid)
                    self.arm_joint_qposadr.append(self.model.jnt_qposadr[jid])
                    self.arm_joint_dofadr.append(self.model.jnt_dofadr[jid])
                except KeyError:
                    pass
            self.nv_arm = len(self.arm_joint_ids)
        else:
            # 默认使用所有关节
            self.arm_joint_ids = list(range(self.model.njnt))
            self.arm_joint_qposadr = list(range(self.model.nv))
            self.arm_joint_dofadr = list(range(self.model.nv))
            self.nv_arm = self.model.nv
        
        # 预分配数组
        self.err = np.empty(6)
        self.err_pos, self.err_rot = self.err[:3], self.err[3:]
        self.site_quat = np.empty(4)
        self.site_quat_inv = np.empty(4)
        self.err_quat = np.empty(4)
        self.jac = np.empty((6, self.model.nv))
        self.jac_pos, self.jac_rot = self.jac[:3], self.jac[3:]
        self.damping = DAMPING_COEFF * np.eye(6)
        
        # 臂关节雅可比提取矩阵
        self.arm_jac = np.empty((6, self.nv_arm))
        self.eye_arm = np.eye(self.nv_arm)
    
    def solve(self, pos, quat, curr_qpos, max_iters=20, err_thresh=1e-4):
        """
        逆运动学求解：末端位姿 → 关节角
        
        Args:
            pos: 目标位置 [x, y, z]
            quat: 目标姿态 [x, y, z, w]
            curr_qpos: 当前关节角（完整 qpos，包含夹爪等）
            max_iters: 最大迭代次数
            err_thresh: 误差阈值
        
        Returns:
            qpos: 目标关节角（与 curr_qpos 维度相同）
        """
        quat = quat[[3, 0, 1, 2]]  # (x,y,z,w) -> (w,x,y,z)
        
        # 设置初始关节配置
        self.data.qpos[:] = curr_qpos
        
        for _ in range(max_iters):
            # 更新运动学
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            
            # 平移误差
            self.err_pos[:] = pos - self.site_pos
            
            # 旋转误差
            mujoco.mju_mat2Quat(self.site_quat, self.site_mat)
            mujoco.mju_negQuat(self.site_quat_inv, self.site_quat)
            mujoco.mju_mulQuat(self.err_quat, quat, self.site_quat_inv)
            mujoco.mju_quat2Vel(self.err_rot, self.err_quat, 1.0)
            
            # 检查是否到达目标
            if np.linalg.norm(self.err) < err_thresh:
                break
            
            # 计算完整雅可比
            mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, self.site_id)
            
            # 提取臂关节对应的雅可比列
            for i, dofadr in enumerate(self.arm_joint_dofadr):
                self.arm_jac[:, i] = self.jac[:, dofadr]
            
            # 计算臂关节更新量
            arm_update = self.arm_jac.T @ np.linalg.solve(
                self.arm_jac @ self.arm_jac.T + self.damping, self.err
            )
            
            # Nullspace 优化（趋向 retract 姿态，仅臂关节）
            qpos0_err = np.zeros(self.nv_arm)
            for i, qadr in enumerate(self.arm_joint_qposadr):
                qpos0_err[i] = np.mod(
                    self.qpos0[qadr] - self.data.qpos[qadr] + np.pi, 2 * np.pi
                ) - np.pi
            
            nullspace_proj = (
                self.eye_arm - 
                (self.arm_jac.T @ np.linalg.pinv(self.arm_jac @ self.arm_jac.T + self.damping)) @ self.arm_jac
            )
            arm_update += nullspace_proj @ qpos0_err
            
            # 限制最大角度变化
            update_max = np.abs(arm_update).max()
            if update_max > MAX_ANGLE_CHANGE:
                arm_update *= MAX_ANGLE_CHANGE / update_max
            
            # 应用更新（仅臂关节）
            for i, qadr in enumerate(self.arm_joint_qposadr):
                self.data.qpos[qadr] += arm_update[i]
        
        return self.data.qpos.copy()


class FKSolver:
    """
    机械臂正运动学求解器
    
    基于 MuJoCo 计算给定关节角时的末端位姿。
    
    用法：
        solver = FKSolver(arm_type='piper')
        pos, quat = solver.solve(qpos)
    """
    
    def __init__(self, arm_type='piper', ee_offset=None, model_path=None, site_name=None):
        """
        Args:
            arm_type: 机械臂类型（仅支持 'piper'）
            ee_offset: 末端执行器偏移（夹爪长度），单位 m
            model_path: MuJoCo 模型文件路径，默认自动查找
            site_name: 末端 site 名称
        """
        config = ARM_CONFIGS.get(arm_type, {})
        
        if model_path is None:
            model_path = _find_model_path()
        if site_name is None:
            site_name = config.get('site_name', 'pinch_site')
        
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.model.body_gravcomp[:] = 1.0
        
        # 缓存引用
        self.site_id = self.model.site(site_name).id
        self.site_pos = self.data.site(self.site_id).xpos
        self.site_mat = self.data.site(self.site_id).xmat
        
        # 设置末端偏移
        if ee_offset is not None:
            ee_site_offset = config.get('ee_site_offset')
            if ee_site_offset is not None:
                offset = np.array(ee_site_offset)
                offset[2] -= ee_offset
                self.model.site(self.site_id).pos = offset
        
        # 预分配
        self.quat = np.empty(4)
    
    def solve(self, qpos):
        """
        正运动学求解：关节角 → 末端位姿
        
        Args:
            qpos: 关节角（完整 qpos）
        
        Returns:
            pos: 末端位置 [x, y, z]
            quat: 末端姿态 [x, y, z, w]
        """
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        
        pos = self.site_pos.copy()
        mujoco.mju_mat2Quat(self.quat, self.site_mat)
        # 转换回 (x, y, z, w) 格式
        quat = np.array([self.quat[1], self.quat[2], self.quat[3], self.quat[0]])
        
        return pos, quat


class KinematicsSolver:
    """
    运动学求解器（IK + FK 合一）
    
    用法：
        # Gen3
        kin = KinematicsSolver(arm_type='gen3')
        
        # Piper
        kin = KinematicsSolver(arm_type='piper')
        
        qpos = kin.ik(target_pos, target_quat, current_qpos)
        pos, quat = kin.fk(qpos)
    """
    
    def __init__(self, arm_type='piper', ee_offset=None, model_path=None, site_name=None):
        self.ik_solver = IKSolver(
            arm_type=arm_type, ee_offset=ee_offset,
            model_path=model_path, site_name=site_name
        )
        self.fk_solver = FKSolver(
            arm_type=arm_type, ee_offset=ee_offset,
            model_path=model_path, site_name=site_name
        )
    
    def ik(self, pos, quat, curr_qpos, **kwargs):
        """逆运动学：末端位姿 → 关节角"""
        return self.ik_solver.solve(pos, quat, curr_qpos, **kwargs)
    
    def fk(self, qpos):
        """正运动学：关节角 → 末端位姿"""
        return self.fk_solver.solve(qpos)


if __name__ == '__main__':
    import sys
    arm_type = sys.argv[1] if len(sys.argv) > 1 else 'piper'
    
    print(f'Testing IK solver with arm_type={arm_type}')
    kin = KinematicsSolver(arm_type=arm_type)
    
    # 使用 retract 姿态作为初始值
    retract_qpos = kin.ik_solver.qpos0.copy()
    
    # 先用 FK 计算当前末端位姿
    pos, quat = kin.fk(retract_qpos)
    print(f'FK(retract) pos: {pos}')
    print(f'FK(retract) quat: {quat}')
    
    # 再用 IK 反求关节角
    qpos = kin.ik(pos, quat, retract_qpos)
    print(f'IK result qpos (deg): {np.rad2deg(qpos).round()}')
    
    # 验证 FK(IK(pos, quat)) == pos
    pos2, quat2 = kin.fk(qpos)
    print(f'FK(IK) pos: {pos2}')
    print(f'FK(IK) quat: {quat2}')
    print(f'Position error: {np.linalg.norm(pos - pos2):.6f}')
