# 视频轨迹截断工具使用说明

## 功能概述

这个工具提供了一个Web界面，可以用来截断机器人轨迹数据集中的视频和对应的轨迹数据。改进版本支持自动加载数据集并直观地选择要截断的episodes。

### 主要功能

- 📁 **数据集自动加载**: 输入数据集目录，自动解析所有episodes信息
- 🎯 **直观选择**: 网格显示所有episodes，点击选择要截断的轨迹
- 📷 **多相机支持**: 支持primary和wrist相机的视频预览和切换
- 🎬 **视频预览**: 在浏览器中直接预览视频文件
- 🎯 **精确选择**: 通过滑块精确选择截断位置（按帧）
- 🔄 **同步处理**: 自动处理该episode的所有相机视频和轨迹数据
- 📊 **帧信息显示**: 显示当前帧号、总帧数、时间戳
- 📸 **帧预览**: 可以预览截断位置的图像
- 💾 **自动备份**: 自动备份所有相关文件
- 🔧 **完整处理**: 同时截断所有视频、轨迹数据和更新元数据
- 📈 **进度跟踪**: 实时显示处理进度和状态
- 🔄 **灵活截断**: 支持选择丢弃前面或后面部分

## 安装依赖

```bash
pip install -r video_truncator_requirements.txt
```

## 使用方法

### 启动服务器

```bash /home/user/.cache/huggingface/lerobot/lerobot_v3/data_toiletButton_0703
├── data
│   └── chunk-000
├── meta
│   ├── episodes
│   ├── info.json
│   ├── stats.json
│   └── tasks.parquet
├── meta_bak_before_filter
│   ├── episodes
│   ├── info.json
│   ├── stats.json
│   └── tasks.parquet
└── videos
    ├── observation.images.primary
    └── observation.images.wrist


python video_truncator_server.py
```

默认配置：
- 地址: `http://0.0.0.0:5000`
- 可以通过浏览器访问 `http://localhost:5000`

### 自定义启动参数

```bash
python video_truncator_server.py --host 127.0.0.1 --port 8080 --debug
```

参数说明：
- `--host`: 服务器地址（默认: 0.0.0.0）
- `--port`: 服务器端口（默认: 5000）
- `--debug`: 启用调试模式

## 使用界面

### 1. 加载数据集

- **数据集根目录**: 输入您的数据集路径，例如 `/media/Data/huggingface/lerobot/lerobot_v3/data_toiletButton_0713`
- 点击"加载数据集"按钮
- 工具会自动验证数据集结构并显示基本信息：
  - 总Episodes数量
  - 总帧数
  - 帧率
  - 机器人类型

### 2. 选择Episode

- 在Episode网格中浏览所有可用的轨迹
- 每个Episode卡片显示：
  - Episode编号
  - 视频状态（Primary/Wrist相机是否存在）
  - 轨迹长度（帧数）
  - 时长
- 点击选择要截断的Episode

### 3. 预览视频

- 选择相机视角（Primary Camera 或 Wrist Camera）
- 视频会自动加载并播放
- 使用视频播放器控件预览视频内容

### 4. 选择截断位置

- 通过滑块选择要截断的帧位置
- 实时显示当前帧号和时间戳
- 点击"📸 预览当前帧"按钮查看截断位置的图像

### 5. 执行截断

- 点击"✂️ 执行截断"按钮
- 等待处理完成
- 查看处理结果和备份位置

## 数据集结构

工具期望的LeRobot v2.1数据集结构：

```
dataset_root/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet
│       ├── file-001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.primary/
│       │   ├── file-000.mp4
│       │   ├── file-001.mp4
│       │   └── ...
│       └── observation.images.wrist/
│           ├── file-000.mp4
│           ├── file-001.mp4
│           └── ...
└── meta/
    ├── episodes/
    │   └── chunk-000/
    │       └── file-000.parquet
    ├── info.json
    ├── stats.json
    └── tasks.parquet
```

## 处理流程

1. **数据集验证**: 检查数据集目录结构是否完整
2. **Episode解析**: 读取parquet文件，解析所有episodes信息
3. **文件备份**: 在数据集根目录创建备份文件夹，备份所有相关文件
4. **视频截断**: 自动检测并截断该episode的所有相机视频到指定帧数
5. **轨迹截断**: 截断对应的parquet轨迹数据，重新计算索引
6. **元数据更新**: 更新info.json中的总帧数统计

## 备份机制

每次截断操作都会自动创建备份：

```
dataset_root/backup_truncate_YYYYMMDD_HHMMSS/
├── file-000.parquet
├── file-001.parquet
├── observation_images_primary_file-XXX.mp4
├── observation_images_wrist_file-XXX.mp4
├── info.json
├── tasks.parquet
└── episodes/
    └── chunk-000/
        └── file-000.parquet
```

如果出现问题，可以从备份中恢复原始数据。

## API接口

### POST /load_dataset

加载数据集信息

**请求体**:
```json
{
  "dataset_root": "/path/to/dataset"
}
```

**响应**:
```json
{
  "success": true,
  "dataset_info": {
    "total_episodes": 198,
    "total_frames": 46431,
    "fps": 30,
    "robot_type": "piper_follower",
    "cameras": ["observation.images.primary", "observation.images.wrist"]
  },
  "episodes": [
    {
      "episode_index": 0,
      "length": 234,
      "primary_video_exists": true,
      "wrist_video_exists": true,
      "primary_video_path": "/path/to/video.mp4",
      "wrist_video_path": "/path/to/video.mp4",
      "start_timestamp": 0.0,
      "end_timestamp": 7.8
    }
  ]
}
```

### POST /truncate

执行截断操作

**请求体**:
```json
{
  "dataset_root": "/path/to/dataset",
  "episode_index": 80,
  "truncate_frame": 100,
  "discard_mode": "after"
}
```

参数说明：
- `dataset_root`: 数据集根目录路径
- `episode_index`: 要截断的episode索引
- `truncate_frame`: 要保留的帧数
- `discard_mode`: 截断模式（可选，默认为 "after"）
  - `"after"`: 丢弃后面，保留前面 - 从开头截断到指定帧数
  - `"before"`: 丢弃前面，保留后面 - 从指定帧数开始截断到结尾

**响应**:
```json
{
  "success": true,
  "backup_dir": "/path/to/backup",
  "processed_videos": {
    "observation.images.primary": {
      "original_frames": 200,
      "truncated_frames": 100,
      "path": "/path/to/video.mp4"
    },
    "observation.images.wrist": {
      "original_frames": 200,
      "truncated_frames": 100,
      "path": "/path/to/video.mp4"
    }
  },
  "successful_videos_count": 2,
  "failed_videos_count": 0,
  "trajectory_frames": 100,
  "episode_number": 80
}
```

### GET /status

获取处理状态

**响应**:
```json
{
  "is_processing": false,
  "progress": 100,
  "message": "截断完成！",
  "error": null
}
```

### GET /health

健康检查

**响应**:
```json
{
  "status": "healthy"
}
```

## 注意事项

1. **文件权限**: 确保对数据集目录有读写权限
2. **磁盘空间**: 确保有足够的磁盘空间进行备份
3. **视频格式**: 目前支持MP4格式
4. **数据一致性**: 确保视频和轨迹数据的帧数一致
5. **备份重要**: 重要数据建议先手动备份
6. **索引重算**: 截断后会重新计算所有数据的全局索引

## 支持的数据集格式

- LeRobot v3 数据集格式
- 支持多相机数据
- 支持chunked数据存储
- 自动识别文件命名模式

## 故障排除

### 数据集无法加载

- 检查数据集路径是否正确
- 确认数据集目录结构完整
- 检查是否有读取权限

### 视频无法加载

- 检查视频文件是否存在
- 确认视频文件格式为MP4
- 检查浏览器是否支持视频播放

### 截断失败

- 检查数据集目录结构是否正确
- 确认所有必需文件都存在
- 查看服务器日志了解详细错误信息

### 轨迹数据不匹配

- 确认视频和轨迹数据来自同一个episode
- 检查原始数据的帧数是否一致

## 开发和调试

### 启用调试模式

```bash
python video_truncator_server.py --debug
```

调试模式会提供：
- 详细的错误信息
- 自动重载功能
- 详细的日志输出

### 查看日志

服务器运行时会输出详细的日志信息，包括：
- 数据集加载过程
- 处理进度
- 文件操作
- 错误信息

## 技术栈

- **后端**: Flask + Python
- **前端**: HTML + CSS + JavaScript
- **视频处理**: OpenCV
- **数据处理**: Pandas + PyArrow
- **数据格式**: Parquet

## 更新日志

### v2.1 (当前版本)

- ✨ 新增灵活截断模式：支持选择丢弃前面或后面部分
- ✨ 新增 `discard_mode` API 参数控制截断方向
- ✨ 改进视频和轨迹数据同步截断逻辑
- 🐛 优化FFmpeg参数，提高截断精度

### v2.0

- ✨ 新增数据集自动加载功能
- ✨ 新增Episode网格选择界面
- ✨ 新增多相机支持（Primary/Wrist）
- ✨ 改进用户界面和交互体验
- ✨ 优化数据集结构适配
- 🐛 修复轨迹数据处理逻辑
- 🐛 修复索引重算问题

### v1.0 (初始版本)

- 基础视频截断功能
- 手动输入视频路径
- 单一相机支持

## 许可证

本工具遵循与主项目相同的许可证。

## 联系方式

如有问题或建议，请联系项目维护者。