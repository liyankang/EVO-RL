# TeleopServer - Flask + SocketIO 遥操作服务
#
# 提供 Web 界面和 SocketIO 通信，接收手机 WebXR 数据并转换为机器人控制指令。

import logging
import socket
import threading
import time
from queue import Queue
from typing import Callable, Optional

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from teleop_controller import TeleopController


class TeleopServer:
    """
    遥操作服务
    
    功能：
    - 运行 Flask Web 服务器，提供 WebXR 前端页面
    - 通过 SocketIO 接收手机位姿数据
    - 使用 TeleopController 将位姿转换为机器人 action
    - 可选启用 IK 求解，将末端位姿转换为关节角
    - 通过回调函数将 action 传递给外部机器人控制代码
    
    使用示例（末端位姿模式）：
        def on_action(action):
            # action: {'base_pose': ..., 'arm_pos': ..., 'arm_quat': ..., 'gripper_pos': ...}
            robot.step(action)
        
        server = TeleopServer(
            get_obs_fn=lambda: robot.get_obs(),
            on_action_fn=on_action,
        )
        server.run()
    
    使用示例（关节角模式，启用 IK）：
        def on_joint_action(action):
            # action: {'base_pose': ..., 'joint_angles': ..., 'gripper_pos': ...}
            robot.send_joint_command(action['joint_angles'], action['gripper_pos'])
        
        server = TeleopServer(
            get_obs_fn=lambda: robot.get_obs(),
            on_action_fn=on_joint_action,
            use_ik=True,
            ee_offset=0.12,
        )
        server.run()
    """
    
    def __init__(
        self,
        get_obs_fn: Callable[[], dict],
        on_action_fn: Optional[Callable[[dict], None]] = None,
        on_state_fn: Optional[Callable[[str], None]] = None,
        host: str = '0.0.0.0',
        port: int = 5000,
        control_freq: float = 10.0,
        message_timeout_ms: float = 250.0,
        use_ik: bool = False,
        arm_type: str = 'piper',
        ee_offset: float = 0.0,
        model_path: Optional[str] = None,
    ):
        """
        Args:
            get_obs_fn: 获取机器人当前观测的函数
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
            message_timeout_ms: 消息超时时间 (ms)，超过此时间的消息将被丢弃
            use_ik: 是否启用 IK 求解（将末端位姿转换为关节角）
            arm_type: 机械臂类型（仅支持 'piper'）
            ee_offset: 末端执行器偏移（夹爪长度），单位 m
            model_path: MuJoCo 模型文件路径（可选，默认自动查找）
        """
        self.get_obs_fn = get_obs_fn
        self.on_action_fn = on_action_fn
        self.on_state_fn = on_state_fn
        self.host = host
        self.port = port
        self.control_period = 1.0 / control_freq
        self.message_timeout_ms = message_timeout_ms
        self.use_ik = use_ik
        
        # IK 求解器（可选）
        self.ik_solver = None
        if use_ik:
            from teleop_service.ik_solver import IKSolver
            self.ik_solver = IKSolver(
                arm_type=arm_type, ee_offset=ee_offset, model_path=model_path
            )
            print(f'IK 求解器已启用 (arm_type={arm_type}, ee_offset={ee_offset})')
        
        # Flask 应用
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, async_mode='threading')  # threading 模式支持 adhoc SSL
        self.message_queue = Queue()
        
        # 遥操作控制器
        self.controller = TeleopController()
        
        # Episode 状态管理（对齐 main.py 逻辑）
        # 状态流转: None -> 'episode_started' -> 'episode_ended' -> 'reset_env' -> 循环
        self.teleop_state = None
        self.episode_ended = False
        
        # 运行状态
        self._running = False
        
        # 设置路由
        self._setup_routes()
        
        # 降低 Flask 日志级别
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    def _setup_routes(self):
        """设置 Flask 路由和 SocketIO 事件处理"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.socketio.on('message')
        def handle_message(data):
            # 回显时间戳（用于 RTT 计算）
            emit('echo', data['timestamp'])
            # 将消息放入队列
            self.message_queue.put(data)
    
    def _listener_loop(self):
        """后台线程：从消息队列读取并处理消息"""
        while self._running:
            if not self.message_queue.empty():
                data = self.message_queue.get()
                
                # 处理状态事件
                if 'state_update' in data:
                    self.teleop_state = data['state_update']
                    if self.on_state_fn is not None:
                        self.on_state_fn(self.teleop_state)
                    
                    # 重置事件时重置控制器
                    if self.teleop_state == 'reset_env':
                        self.controller.reset()
                        self.episode_ended = False
                
                # 处理遥操作消息（丢弃过期消息）
                elif 1000 * time.time() - data['timestamp'] < self.message_timeout_ms:
                    self.controller.process_message(data)
            
            time.sleep(0.001)
    
    def _control_loop(self):
        """主控制循环（对齐 main.py 逻辑）
        
        状态流转:
        1. 持续调用 step() 初始化控制器（targets_initialized）
        2. 等待 'episode_started'（用户在网页点击「开始遥操作」）
        3. 执行控制循环：get_obs → controller.step → on_action
        4. 收到 'episode_ended' → 停止执行动作（允许遥操作调整）
        5. 收到 'reset_env' → 重置环境，回到步骤 2
        """
        while self._running:
            step_start = time.time()
            
            # 获取当前观测
            obs = self.get_obs_fn()
            
            # 始终调用 step() 初始化控制器（使 targets_initialized = True）
            action = self.controller.step(obs)
            
            # Episode 生命周期管理
            # 检查 episode 是否刚结束
            if self.teleop_state == 'episode_ended' and not self.episode_ended:
                self.episode_ended = True
                print('Episode 结束，可继续遥操作调整。点击「重置环境」开始下一个 Episode')
            
            # 执行 action（仅在 episode 进行中）
            if self.teleop_state == 'episode_started' and not self.episode_ended:
                if action is not None and self.on_action_fn is not None:
                    if self.use_ik and self.ik_solver is not None:
                        # IK 模式：将末端位姿转换为关节角
                        curr_qpos = obs.get('joint_angles', None)
                        if curr_qpos is None:
                            curr_qpos = self.ik_solver.qpos0.copy()
                        
                        joint_angles = self.ik_solver.solve(
                            action['arm_pos'], action['arm_quat'], curr_qpos
                        )
                        
                        ik_action = {
                            'base_pose': action['base_pose'],
                            'joint_angles': joint_angles,
                            'gripper_pos': action['gripper_pos'],
                        }
                        self.on_action_fn(ik_action)
                    else:
                        # 末端位姿模式
                        self.on_action_fn(action)
            
            # 保持控制频率
            step_end = time.time()
            sleep_time = self.control_period - (step_end - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_ip_address(self) -> str:
        """获取本机 IP 地址"""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(('8.8.8.8', 1))
            address = s.getsockname()[0]
        except Exception:
            address = '127.0.0.1'
        finally:
            s.close()
        return address
    
    def run(self, block: bool = True):
        """
        启动遥操作服务
        
        Args:
            block: 是否阻塞运行（默认 True）
        """
        self._running = True
        
        # 启动消息监听线程
        listener_thread = threading.Thread(target=self._listener_loop, daemon=True)
        listener_thread.start()
        
        # 启动控制循环线程
        control_thread = threading.Thread(target=self._control_loop, daemon=True)
        control_thread.start()
        
        # 打印访问地址
        ip = self.get_ip_address()
        print(f'\n{"="*50}')
        print(f'遥操作服务已启动')
        print(f'手机访问地址: http://{ip}:{self.port}')
        print(f'注意: WebXR 需要 HTTPS，如果手机显示"XR 不可用"，请配置 SSL 证书')
        print(f'\n操作流程:')
        print(f'1. 手机打开上述地址')
        print(f'2. 点击「开始遥操作」→ AR/VR 会话启动')
        print(f'3. 移动手机 → 机械臂响应')
        print(f'4. 点击按钮 → 结束/重置')
        print(f'{"="*50}\n')
        
        if block:
            # 阻塞运行 Flask 服务器
            try:
                self.socketio.run(self.app, host=self.host, port=self.port)
            finally:
                self._running = False
        else:
            # 非阻塞运行
            web_thread = threading.Thread(
                target=lambda: self.socketio.run(self.app, host=self.host, port=self.port),
                daemon=True
            )
            web_thread.start()
            return self
    
    def stop(self):
        """停止服务"""
        self._running = False
        print('遥操作服务已停止')
