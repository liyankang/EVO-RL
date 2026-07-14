#!/usr/bin/env python3
"""视频与轨迹截断处理核心模块，支持 LeRobot v2.1 数据格式。

此模块已被重构为多个子模块，提供更好的代码组织：
- file_utils.py: 文件操作工具函数
- dataset_loader.py: 数据集加载和验证
- video_utils.py: 视频处理工具函数
- backup_manager.py: 备份管理功能
- video_truncator.py: 核心截断逻辑
"""

from pathlib import Path
from typing import Dict, Any

# 导入主要类和函数以保持向后兼容
from .dataset_loader import DatasetLoader
from .video_truncator import VideoTruncator
from .backup_manager import restore_latest_backup
from .file_utils import (
    _json_default,
    _read_json,
    _write_json,
    _read_jsonl,
    _write_jsonl,
    _flatten_dict,
    _serialize_stats_tree
)
from video_utils import (
    write_video_frame_window,
    get_truncation_window
)
from action_smoother import ActionSmoother

__all__ = [
    'DatasetLoader',
    'VideoTruncator',
    'restore_latest_backup',
    '_json_default',
    '_read_json',
    '_write_json',
    '_read_jsonl',
    '_write_jsonl',
    '_flatten_dict',
    '_serialize_stats_tree',
    'write_video_frame_window',
    'get_truncation_window',
]

__version__ = '2.0.0'

# 保持原有模块结构兼容性的便捷函数
def create_truncator(dataset_root: Path, episode_index: int, **kwargs) -> VideoTruncator:
    """创建VideoTruncator实例的便捷函数。
    
    Args:
        dataset_root: 数据集根目录
        episode_index: 要截断的episode索引
        **kwargs: 传递给VideoTruncator的额外参数
        
    Returns:
        VideoTruncator实例
    """
    return VideoTruncator(dataset_root, episode_index, **kwargs)


def create_loader(dataset_root: Path) -> DatasetLoader:
    """创建DatasetLoader实例的便捷函数。
    
    Args:
        dataset_root: 数据集根目录
        
    Returns:
        DatasetLoader实例
    """
    return DatasetLoader(dataset_root)