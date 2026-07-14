# 视频截断系统架构文档

## 概述

视频截断系统现已拆分为两个独立模块，实现核心功能与Web服务的解耦，便于后续维护和扩展。

## 模块架构

### 1. 核心处理模块 (`video_processor.py`)

**职责**：视频和轨迹数据的核心处理逻辑

**主要组件**：
- `DatasetLoader`：数据集加载和验证
- `VideoTruncator`：视频和轨迹截断处理
- `restore_latest_backup()`：备份恢复功能
- 辅助函数：JSON/JSONL读写、统计信息处理等

**功能特点**：
- ✅ 独立的数据处理逻辑
- ✅ 可作为Python库直接导入使用
- ✅ 完整的单元测试支持
- ✅ 支持命令行工具调用

### 2. Web服务模块 (`video_truncator_server.py`)

**职责**：HTTP API接口和Web界面服务

**主要功能**：
- RESTful API接口
- 视频文件流式传输
- 处理状态管理
- 健康检查接口

**API端点**：
- `GET /` - 主页界面
- `POST /load_dataset` - 加载数据集
- `POST /get_video_info` - 获取视频信息
- `POST /truncate` - 执行截断操作
- `POST /undo` - 回退操作
- `GET /status` - 获取处理状态
- `GET /video/<path>` - 视频文件服务
- `GET /health` - 健康检查

## 优势

### 1. 功能解耦
- **核心逻辑独立**：`video_processor.py`不依赖Flask等Web框架
- **Web服务简洁**：`video_truncator_server.py`只负责HTTP接口
- **模块化设计**：每个模块职责清晰，便于单独测试

### 2. 灵活性提升
- **多用途核心**：核心模块可用于命令行工具、批处理脚本等
- **独立部署**：可分别部署和扩展各个模块
- **接口适配**：可轻松创建其他接口（如gRPC、CLI等）

### 3. 维护性改进
- **代码组织**：相关功能集中管理
- **测试友好**：核心逻辑可独立进行单元测试
- **版本控制**：模块可独立版本管理

## 使用方式

### 1. 作为Web服务使用

```bash
# 启动Web服务器
python video_truncator_server.py --host 0.0.0.0 --port 5000

# 访问界面
# 浏览器打开 http://localhost:5000
```

### 2. 作为Python库使用

```python
from video_processor import DatasetLoader, VideoTruncator, restore_latest_backup

# 加载数据集
loader = DatasetLoader("/path/to/dataset")
episodes_info = loader.get_episodes_list()

# 执行截断
truncator = VideoTruncator(
    dataset_root="/path/to/dataset",
    episode_index=0,
    keep_start_frame=100,
    keep_end_frame=500
)
result = truncator.truncate()

# 恢复备份
restore_latest_backup("/path/to/dataset")
```

### 3. 作为命令行工具使用

```bash
# 可以创建简单的CLI工具
python -c "
from video_processor import VideoTruncator
truncator = VideoTruncator('/dataset', 0, keep_start_frame=100, keep_end_frame=500)
result = truncator.truncate()
print(result)
"
```

## 文件结构

```
evo_rl/
├── video_processor.py              # 核心处理模块
├── video_truncator_server.py       # Web服务模块
├── video_truncator.html           # Web界面
├── video_truncator_requirements.txt  # 依赖项
├── VIDEO_FIX_DOCUMENTATION.md      # 修复文档
└── MODULE_ARCHITECTURE.md         # 架构文档（本文件）
```

## 依赖关系

### 核心模块依赖
- `numpy` - 数值计算
- `pandas` - 数据处理
- `opencv-python` - 视频处理
- `pyarrow` - Parquet文件支持
- `av` - 高级视频处理（可选）

### Web服务模块依赖
- `flask` - Web框架
- `flask-cors` - 跨域支持（可选）

## 开发指南

### 添加新的核心功能

1. 在`video_processor.py`中添加新功能
2. 编写单元测试
3. 如需Web接口，在`video_truncator_server.py`中添加API端点

### 扩展Web接口

1. 在`video_truncator_server.py`中添加新路由
2. 调用核心模块的功能
3. 添加适当的错误处理和日志记录

### 性能优化

1. 核心算法优化：修改`video_processor.py`
2. 并发处理：在Web服务模块中添加异步支持
3. 缓存策略：在适当位置添加缓存机制

## 测试建议

### 核心模块测试
```python
import unittest
from video_processor import DatasetLoader, VideoTruncator

class TestVideoProcessor(unittest.TestCase):
    def test_dataset_loading(self):
        loader = DatasetLoader("/test/dataset")
        result = loader.get_episodes_list()
        self.assertTrue(result['success'])
    
    def test_video_truncation(self):
        truncator = VideoTruncator("/test/dataset", 0, keep_start_frame=0, keep_end_frame=100)
        result = truncator.truncate()
        self.assertTrue(result['success'])
```

### Web服务测试
```bash
# 测试健康检查
curl http://localhost:5000/health

# 测试数据集加载
curl -X POST http://localhost:5000/load_dataset \
  -H "Content-Type: application/json" \
  -d '{"dataset_root": "/path/to/dataset"}'
```

## 故障排除

### 模块导入错误
```bash
# 确保在正确的目录中运行
cd /home/user/code_1/evo_rl
python video_truncator_server.py
```

### 依赖缺失
```bash
# 安装所有依赖
pip install -r video_truncator_requirements.txt
```

### 端口冲突
```bash
# 使用不同端口启动
python video_truncator_server.py --port 5001
```

## 扩展建议

### 1. 添加批处理模式
```python
# 批量处理多个episodes
for episode_index in range(10):
    truncator = VideoTruncator(dataset_root, episode_index, **params)
    truncator.truncate()
```

### 2. 添加进度回调
```python
def progress_callback(progress, message):
    print(f"[{progress}%] {message}")

truncator = VideoTruncator(dataset_root, episode_index, **params)
truncator.truncate(progress_callback=progress_callback)
```

### 3. 添加异步处理
```python
import asyncio
from video_processor import VideoTruncator

async def async_truncate():
    loop = asyncio.get_event_loop()
    truncator = VideoTruncator(dataset_root, episode_index, **params)
    result = await loop.run_in_executor(None, truncator.truncate)
    return result
```

## 总结

通过将视频截断系统拆分为两个独立模块，我们实现了：

- ✅ **清晰的职责分离**：核心逻辑与Web服务解耦
- ✅ **更好的可维护性**：模块化设计便于维护和扩展
- ✅ **增强的灵活性**：支持多种使用方式和部署场景
- ✅ **改进的测试性**：核心功能可独立测试
- ✅ **更强的扩展性**：便于添加新功能和接口

这种架构设计为后续的功能扩展和性能优化提供了良好的基础。