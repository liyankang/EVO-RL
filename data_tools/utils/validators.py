"""数据验证模块，验证数据集结构和路径"""

from pathlib import Path
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def is_valid_dataset_path(path: Path) -> bool:
    """检查路径是否是有效的数据集路径
    
    Args:
        path: 要检查的路径
        
    Returns:
        是否是有效的数据集路径
    """
    try:
        path = Path(path)
        
        # 必须是目录
        if not path.is_dir():
            return False
        
        # 必须包含必要的子目录
        required_dirs = ['data', 'meta']
        for dir_name in required_dirs:
            if not (path / dir_name).exists():
                return False
        
        # 检查是否有元数据文件
        meta_dir = path / 'meta'
        required_files = ['info.json']
        for file_name in required_files:
            if not (meta_dir / file_name).exists():
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"验证数据集路径时出错: {e}")
        return False


def validate_dataset_structure(dataset_root: Path) -> Tuple[bool, List[str]]:
    """验证数据集目录结构是否完整
    
    Args:
        dataset_root: 数据集根目录
        
    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []
    
    # 检查根目录
    if not dataset_root.exists():
        errors.append(f"数据集根目录不存在: {dataset_root}")
        return False, errors
    
    if not dataset_root.is_dir():
        errors.append(f"数据集根路径不是目录: {dataset_root}")
        return False, errors
    
    # 检查必需的子目录
    required_subdirs = {
        'data': '数据目录',
        'meta': '元数据目录',
        'videos': '视频目录'
    }
    
    for subdir_name, description in required_subdirs.items():
        subdir_path = dataset_root / subdir_name
        if not subdir_path.exists():
            errors.append(f"{description}不存在: {subdir_path}")
        elif not subdir_path.is_dir():
            errors.append(f"{description}不是有效的目录: {subdir_path}")
    
    # 检查 data 目录下的结构
    data_dir = dataset_root / 'data'
    if data_dir.exists():
        # 检查是否有 chunk-000 目录或直接的 parquet 文件
        chunk_dir = data_dir / 'chunk-000'
        if chunk_dir.exists():
            parquet_files = list(chunk_dir.glob('*.parquet'))
            if not parquet_files:
                errors.append(f"data/chunk-000 目录中没有找到 parquet 文件")
        else:
            # 检查 data 目录下是否有 parquet 文件
            parquet_files = list(data_dir.glob('*.parquet'))
            if not parquet_files:
                errors.append(f"data 目录中没有找到 parquet 文件")
    
    # 检查 meta 目录下的文件
    meta_dir = dataset_root / 'meta'
    if meta_dir.exists():
        required_files = ['info.json']
        for file_name in required_files:
            file_path = meta_dir / file_name
            if not file_path.exists():
                errors.append(f"元数据文件不存在: {file_path}")
            elif not file_path.is_file():
                errors.append(f"元数据文件不是有效的文件: {file_path}")
    
    return len(errors) == 0, errors


def validate_episode_data(dataset_root: Path, episode_index: int) -> Tuple[bool, List[str]]:
    """验证特定 episode 的数据是否完整
    
    Args:
        dataset_root: 数据集根目录
        episode_index: episode 索引
        
    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []
    
    # 检查轨迹数据文件
    data_dir = dataset_root / 'data' / 'chunk-000'
    if data_dir.exists():
        trajectory_file = data_dir / f"episode_{episode_index:06d}.parquet"
        if not trajectory_file.exists():
            errors.append(f"轨迹数据文件不存在: {trajectory_file}")
    else:
        errors.append(f"数据目录不存在: {data_dir}")
    
    # 检查视频文件（至少一个相机）
    video_dir = dataset_root / 'videos' / 'chunk-000'
    if video_dir.exists():
        camera_dirs = [d for d in video_dir.iterdir() if d.is_dir()]
        if not camera_dirs:
            errors.append(f"视频目录中没有相机子目录: {video_dir}")
        else:
            # 至少有一个相机的视频文件存在
            video_found = False
            for camera_dir in camera_dirs:
                video_file = camera_dir / f"episode_{episode_index:06d}.mp4"
                if video_file.exists():
                    video_found = True
                    break
            
            if not video_found:
                errors.append(f"没有找到 episode {episode_index} 的视频文件")
    else:
        errors.append(f"视频目录不存在: {video_dir}")
    
    return len(errors) == 0, errors


def get_video_cameras(dataset_root: Path) -> List[str]:
    """获取数据集中所有相机的名称
    
    Args:
        dataset_root: 数据集根目录
        
    Returns:
        相机名称列表
    """
    cameras = []
    
    video_dir = dataset_root / 'videos' / 'chunk-000'
    if video_dir.exists() and video_dir.is_dir():
        for item in video_dir.iterdir():
            if item.is_dir():
                cameras.append(item.name)
    
    return sorted(cameras)