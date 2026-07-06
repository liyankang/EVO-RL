# Piper 真机遥操作使用指南

## 文件说明

- `real_robot.py` - Piper 真机 WebXR 遥操作主脚本
- `test_piper.py` - Piper 真机基础测试脚本
- `run.py` - 统一入口（支持多种模式）

## 快速开始

### 1. 基础测试（推荐先执行）

测试 Piper 连接和基本功能：

```bash
# 基本测试（读取状态）
python teleop_service/test_piper.py --port can0

# 跳过校准（使用已有校准文件）
python teleop_service/test_piper.py --port can0 --no-calibrate

# 测试发送动作
python teleop_service/test_piper.py --port can0 --test-action
```

### 2. WebXR 遥操作

启动完整的 WebXR 遥操作服务：

```bash
# 方式 1：使用独立脚本
python teleop_service/real_robot.py --port can0

# 方式 2：使用统一入口
python teleop_service/run.py --real-piper --can-port can0

# 跳过校准
python teleop_service/real_robot.py --port can0 --no-calibrate

# 自定义服务器端口
python teleop_service/real_robot.py --port can0 --server-port 8080
```

启动后，手机访问显示的 IP 地址（如 `http://192.168.1.100:5000`），点击「开始遥操作」即可控制。

### 3. 其他模式

```bash
# MuJoCo 仿真（可视化）
python teleop_service/run.py --sim-mj

# 模拟机器人（无硬件）
python teleop_service/run.py --sim

# RPC 模式（旧版）
python teleop_service/run.py --real
```

## 参数说明

### real_robot.py 参数

- `--port` - CAN 端口（默认: can0）
- `--no-calibrate` - 跳过校准，使用已有校准文件
- `--host` - 服务器地址（默认: 0.0.0.0）
- `--server-port` - 服务器端口（默认: 5000）
- `--freq` - 控制频率 Hz（默认: 10.0）
- `--ee-offset` - 末端执行器偏移，单位 m（默认: 0.0）
- `--model-path` - MuJoCo 模型文件路径（可选）
- `--message-timeout-ms` - 消息超时时间 ms（默认: 250.0）

### run.py 额外参数

- `--real-piper` - 使用 Piper 真机（CAN 总线）
- `--can-port` - CAN 端口（默认: can0）
- `--no-calibrate` - 跳过校准

## 工作流程

1. **连接机器人** - 通过 CAN 总线连接 PiperFollower
2. **初始化 IK** - 加载 MuJoCo 模型用于逆运动学求解
3. **启动服务** - Flask + SocketIO 服务器提供 WebXR 界面
4. **遥操作循环**：
   - 手机 WebXR → 末端位姿
   - IK 求解 → 关节角
   - 发送到 Piper
   - 读取关节角 → FK → 末端位姿
   - 更新 TeleopController

## 注意事项

- 首次使用需要校准机器人（不使用 `--no-calibrate`）
- 确保 CAN 总线已正确配置（`can0` 或其他端口）
- WebXR 需要 HTTPS 才能在手机上正常工作
- 控制频率建议 10-20 Hz，过高可能导致延迟

## 故障排查

### 无法连接 CAN

```bash
# 检查 CAN 接口
ip link show can0

# 如果未启动，手动启动
sudo ip link set can0 up type can bitrate 1000000
```

### 校准问题

如果校准文件丢失或损坏，重新运行校准：

```bash
python teleop_service/real_robot.py --port can0
# 不使用 --no-calibrate，会提示校准
```

### WebXR 不可用

手机浏览器显示「XR 不可用」时，需要配置 HTTPS：

1. 使用 SSL 证书
2. 或使用 ngrok 等工具创建 HTTPS 隧道
3. 或在 Flask 中启用 SSL

## 架构说明

```
手机 WebXR
    ↓
TeleopServer (Flask + SocketIO)
    ↓
TeleopController (位姿转换)
    ↓
IKSolver (MuJoCo IK)
    ↓
PiperRealRobot (封装器)
    ↓
PiperFollower (CAN 控制)
    ↓
Piper 机械臂
```

反向链路（状态反馈）：

```
Piper 机械臂
    ↓
PiperFollower (读取关节角)
    ↓
PiperRealRobot (FK 计算末端位姿)
    ↓
TeleopController (更新参考位姿)
    ↓
TeleopServer (发送给前端)
```


  joint_1:   +1.369
  joint_2:  +47.896
  joint_3:  -45.975
  joint_4:   +3.697
  joint_5:  +54.464
  joint_6:   -0.516
  gripper:   +2.400
