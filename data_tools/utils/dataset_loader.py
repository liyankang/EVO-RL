"""数据集加载器模块，用于加载和验证数据集元数据。"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import cv2

from .file_utils import _read_json

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load dataset metadata for the truncation UI."""

    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root)
        self.data_dir = self.dataset_root / "data" / "chunk-000"
        self.video_dir = self.dataset_root / "videos" / "chunk-000"
        self.meta_dir = self.dataset_root / "meta"

    def validate_dataset(self) -> Tuple[bool, List[str]]:
        """验证数据集目录结构是否完整。"""
        errors = []

        if not self.dataset_root.exists():
            errors.append(f"Dataset root does not exist: {self.dataset_root}")
            return False, errors
        if not self.data_dir.exists():
            errors.append(f"Data directory does not exist: {self.data_dir}")
        if not self.video_dir.exists():
            errors.append(f"Video directory does not exist: {self.video_dir}")
        if not self.meta_dir.exists():
            errors.append(f"Meta directory does not exist: {self.meta_dir}")

        return len(errors) == 0, errors

    def get_episodes_list(self) -> Dict[str, Any]:
        """获取数据集中所有episode的列表和元信息。"""
        try:
            info_path = self.meta_dir / "info.json"
            if not info_path.exists():
                raise FileNotFoundError(f"info.json does not exist: {info_path}")

            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)

            parquet_files = sorted(self.data_dir.glob("episode_*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No trajectory data files found: {self.data_dir}/episode_*.parquet")

            episodes = []
            for pq_file in parquet_files:
                episode_index = int(pq_file.stem.replace('episode_', ''))
                episode_data = pd.read_parquet(pq_file)
                episode_length = len(episode_data)
                start_timestamp = float(episode_data['timestamp'].iloc[0]) if 'timestamp' in episode_data.columns else 0.0
                end_timestamp = float(episode_data['timestamp'].iloc[-1]) if 'timestamp' in episode_data.columns else 0.0

                primary_video_path = self.video_dir / "observation.images.primary" / f"episode_{episode_index:06d}.mp4"
                wrist_video_path = self.video_dir / "observation.images.wrist" / f"episode_{episode_index:06d}.mp4"

                episodes.append({
                    'episode_index': episode_index,
                    'length': episode_length,
                    'primary_video_exists': primary_video_path.exists(),
                    'wrist_video_exists': wrist_video_path.exists(),
                    'primary_video_path': str(primary_video_path) if primary_video_path.exists() else None,
                    'wrist_video_path': str(wrist_video_path) if wrist_video_path.exists() else None,
                    'start_timestamp': start_timestamp,
                    'end_timestamp': end_timestamp,
                })

            episodes.sort(key=lambda item: item['episode_index'])
            return {
                'success': True,
                'dataset_info': {
                    'total_episodes': info.get('total_episodes', len(episodes)),
                    'total_frames': info.get('total_frames', 0),
                    'fps': info.get('fps', 30),
                    'robot_type': info.get('robot_type', 'unknown'),
                    'cameras': list(info.get('features', {}).keys()),
                },
                'episodes': episodes,
            }
        except Exception as e:
            logger.error(f"Failed to get episode list: {e}")
            return {'success': False, 'error': str(e)}

    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """获取视频文件的基本信息。"""
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                return {'success': False, 'error': f'Video file does not exist: {video_path}'}

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {'success': False, 'error': f'Unable to open video file: {video_path}'}

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            return {
                'success': True,
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
            }
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return {'success': False, 'error': str(e)}