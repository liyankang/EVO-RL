"""视频截断核心类，处理轨迹和视频数据的截断操作。"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import cv2

try:
    from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
    from lerobot.datasets.utils import write_stats
except Exception:  # pragma: no cover - keep the standalone web tool usable.
    aggregate_stats = None
    compute_episode_stats = None
    write_stats = None

from file_utils import _read_json, _write_json, _read_jsonl, _write_jsonl, _flatten_dict, _serialize_stats_tree
from video_utils import write_video_frame_window, get_truncation_window
from backup_manager import BackupManager

logger = logging.getLogger(__name__)


class VideoTruncator:
    """Truncate videos, trajectory parquet, and metadata for a dataset episode."""

    def __init__(
        self,
        dataset_root: Path,
        episode_index: int,
        truncate_frame: Optional[int] = None,
        discard_mode: str = 'after',
        keep_start_frame: Optional[int] = None,
        keep_end_frame: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.episode_index = int(episode_index)
        self.truncate_frame = int(truncate_frame) if truncate_frame is not None else None
        self.discard_mode = discard_mode
        self.keep_start_frame = int(keep_start_frame) if keep_start_frame is not None else None
        self.keep_end_frame = int(keep_end_frame) if keep_end_frame is not None else None
        # 新增：基于时间戳的截断参数
        self.start_time = float(start_time) if start_time is not None else None
        self.end_time = float(end_time) if end_time is not None else None
        self.data_dir = self.dataset_root / "data" / "chunk-000"
        self.video_dir = self.dataset_root / "videos" / "chunk-000"
        self.meta_dir = self.dataset_root / "meta"
        self.backup_manager = BackupManager(self.dataset_root, self.episode_index)

    def _data_parquet_files(self) -> List[Path]:
        """获取所有数据parquet文件。"""
        files = sorted(self.data_dir.glob("episode_*.parquet"))
        if files:
            return files
        return sorted((self.dataset_root / "data").glob("chunk-*/*.parquet"))

    def _metadata_parquet_files(self, subdir: str) -> List[Path]:
        """获取指定子目录的元数据parquet文件。"""
        return sorted((self.meta_dir / subdir).glob("chunk-*/*.parquet"))

    def _load_info(self) -> Dict:
        """加载info.json文件。"""
        return _read_json(self.meta_dir / "info.json", {}) or {}

    def _find_episode_data_file(self) -> Tuple[Path, Optional[np.ndarray]]:
        """查找包含指定episode的parquet文件。"""
        target_file = self.data_dir / f"episode_{self.episode_index:06d}.parquet"
        if target_file.exists():
            return target_file, None

        for parquet_path in self._data_parquet_files():
            df = pd.read_parquet(parquet_path)
            if "episode_index" not in df.columns:
                continue
            mask = df["episode_index"] == self.episode_index
            if mask.any():
                return parquet_path, mask

        raise FileNotFoundError(f"Trajectory parquet for episode {self.episode_index} was not found")

    def _episode_stats_from_dataframe(self, df: pd.DataFrame, info: Dict) -> Dict:
        """从DataFrame计算episode统计信息。"""
        if compute_episode_stats is None:
            return {}

        features = info.get("features", {})
        episode_data = {}
        episode_features = {}
        for key, feature in features.items():
            if key not in df.columns:
                continue
            if feature.get("dtype") in {"image", "video", "string"}:
                continue

            values = df[key].to_numpy()
            try:
                if len(values) and isinstance(values[0], (list, tuple, np.ndarray)):
                    values = np.stack(values)
                else:
                    values = values.astype(np.float64, copy=False)
            except Exception:
                continue

            episode_data[key] = values
            episode_features[key] = feature

        if not episode_data:
            return {}
        return compute_episode_stats(episode_data, episode_features)

    def _reindex_all_data_parquets(self) -> int:
        """截断后重排全局index，并按episode重算frame_index。"""
        total_frames = 0
        for parquet_path in self._data_parquet_files():
            df = pd.read_parquet(parquet_path)
            changed = False
            if "index" in df.columns:
                df["index"] = range(total_frames, total_frames + len(df))
                changed = True
            if "frame_index" in df.columns:
                if "episode_index" in df.columns:
                    df["frame_index"] = df.groupby("episode_index").cumcount()
                else:
                    df["frame_index"] = range(len(df))
                changed = True
            total_frames += len(df)
            if changed:
                df.to_parquet(parquet_path, index=False)
        return total_frames

    def _all_episode_lengths(self) -> Dict[int, int]:
        """获取所有episode的长度。"""
        lengths = {}
        for parquet_path in self._data_parquet_files():
            df = pd.read_parquet(parquet_path)
            if "episode_index" in df.columns:
                for ep_idx, group in df.groupby("episode_index", sort=False):
                    lengths[int(ep_idx)] = int(len(group))
            else:
                try:
                    ep_idx = int(parquet_path.stem.replace("episode_", ""))
                except ValueError:
                    continue
                lengths[ep_idx] = int(len(df))
        return lengths

    def _compute_all_episode_stats(self, info: Dict) -> Dict[int, Dict]:
        """计算所有episode的统计信息。"""
        if compute_episode_stats is None:
            return {}

        all_stats = {}
        for parquet_path in self._data_parquet_files():
            df = pd.read_parquet(parquet_path)
            if "episode_index" in df.columns:
                for ep_idx, group in df.groupby("episode_index", sort=False):
                    all_stats[int(ep_idx)] = self._episode_stats_from_dataframe(group.reset_index(drop=True), info)
            else:
                try:
                    ep_idx = int(parquet_path.stem.replace("episode_", ""))
                except ValueError:
                    continue
                all_stats[ep_idx] = self._episode_stats_from_dataframe(df.reset_index(drop=True), info)
        return all_stats

    def _write_aggregated_stats_json(self, all_episode_stats: Dict):
        """写入聚合统计信息到stats.json文件。"""
        if not all_episode_stats or aggregate_stats is None or write_stats is None:
            return

        non_empty_stats = [stats for stats in all_episode_stats.values() if stats]
        if not non_empty_stats:
            return

        aggregated_stats = aggregate_stats(non_empty_stats)
        write_stats(aggregated_stats, self.dataset_root)

    def _update_legacy_episodes_stats_jsonl(self, episode_stats: Dict, new_length: int, start_ts: Optional[float], end_ts: Optional[float]):
        """更新旧版episodes_stats.jsonl文件。"""
        # LeRobot 2.1 期望 episodes_stats.jsonl 使用嵌套的 stats 字段，而不是平铺 stats/... 键。
        episodes_stats_jsonl_path = self.meta_dir / "episodes_stats.jsonl"
        if not episodes_stats_jsonl_path.exists():
            return

        rows = _read_jsonl(episodes_stats_jsonl_path)
        updated = False
        for row in rows:
            if row.get("episode_index") != self.episode_index:
                continue

            # 清理旧的平铺格式键
            for key in list(row.keys()):
                if key.startswith("stats/"):
                    row.pop(key)

            # 更新统计信息 - 只包含非视频数据的统计
            # 视频数据在 episodes_stats.jsonl 中不应该出现统计信息
            clean_stats = {}
            for key, value in episode_stats.items():
                # 排除视频/图像特征的统计信息，因为它们的计数可能与轨迹数据不一致
                if not any(video_key in key for video_key in ['observation.images', 'observation.video']):
                    clean_stats[key] = value

            row["stats"] = _serialize_stats_tree(clean_stats)
            if "length" in row:
                row["length"] = new_length
            if "num_frames" in row:
                row["num_frames"] = new_length
            if start_ts is not None and "start_timestamp" in row:
                row["start_timestamp"] = start_ts
            if end_ts is not None and "end_timestamp" in row:
                row["end_timestamp"] = end_ts
            updated = True
            break

        if updated:
            _write_jsonl(episodes_stats_jsonl_path, rows)

    def _update_episode_metadata_parquet(self, subdir: str, episode_index: int, new_length: int, start_ts: Optional[float], end_ts: Optional[float], flat_stats: Dict):
        """更新episode元数据parquet文件。"""
        for parquet_path in self._metadata_parquet_files(subdir):
            df = pd.read_parquet(parquet_path)
            if "episode_index" not in df.columns:
                continue

            mask = df["episode_index"] == episode_index
            if not mask.any():
                continue

            if "length" in df.columns:
                df.loc[mask, "length"] = new_length
            if "num_frames" in df.columns:
                df.loc[mask, "num_frames"] = new_length
            if start_ts is not None and "start_timestamp" in df.columns:
                df.loc[mask, "start_timestamp"] = start_ts
            if end_ts is not None and "end_timestamp" in df.columns:
                df.loc[mask, "end_timestamp"] = end_ts

            for key, value in flat_stats.items():
                df.loc[mask, key] = [value] * int(mask.sum())

            df.to_parquet(parquet_path, index=False)

    def validate_paths(self) -> List[str]:
        """验证数据集路径是否完整。"""
        errors = []

        if not self.dataset_root.exists():
            errors.append(f"Dataset root does not exist: {self.dataset_root}")

        parquet_files = self._data_parquet_files()
        if not parquet_files:
            errors.append(f"No trajectory parquet files found under {self.dataset_root / 'data'}")
        else:
            try:
                self._find_episode_data_file()
            except FileNotFoundError as exc:
                errors.append(str(exc))

        has_video = False
        if self.video_dir.exists():
            for camera_dir in self.video_dir.iterdir():
                if camera_dir.is_dir() and (camera_dir / f"episode_{self.episode_index:06d}.mp4").exists():
                    has_video = True
                    break

        if not has_video:
            errors.append(f"No video file found for episode {self.episode_index}")

        if not self.meta_dir.exists():
            errors.append(f"Meta directory does not exist: {self.meta_dir}")

        return errors
    
    def validate_video_trajectory_alignment(self) -> Dict[str, Any]:
        """验证视频和轨迹数据的对应关系。"""
        result = {
            'valid': True,
            'trajectory_info': {},
            'video_info': {},
            'alignment_check': {},
            'errors': []
        }
        
        try:
            # 获取轨迹数据信息
            target_file, episode_mask = self._find_episode_data_file()
            file_df = pd.read_parquet(target_file)
            episode_data = file_df if episode_mask is None else file_df.loc[episode_mask]
            trajectory_frames = len(episode_data)
            
            result['trajectory_info'] = {
                'frames': trajectory_frames,
                'has_timestamp': 'timestamp' in episode_data.columns,
                'time_range': None
            }
            
            if 'timestamp' in episode_data.columns:
                timestamps = episode_data['timestamp'].to_numpy()
                result['trajectory_info']['time_range'] = [
                    float(timestamps[0]),
                    float(timestamps[-1])
                ]
            
            # 获取视频信息
            if not self.video_dir.exists():
                result['errors'].append("视频目录不存在")
                result['valid'] = False
                return result
            
            video_check_results = {}
            for camera_subdir in self.video_dir.iterdir():
                if not camera_subdir.is_dir():
                    continue
                
                camera_name = camera_subdir.name
                video_path = camera_subdir / f"episode_{self.episode_index:06d}.mp4"
                
                if not video_path.exists():
                    continue
                
                # 获取视频帧数
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    result['errors'].append(f"无法打开视频: {camera_name}")
                    continue
                
                video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = video_frames / fps if fps > 0 else 0
                cap.release()
                
                # 检查帧数是否匹配
                frames_match = video_frames == trajectory_frames
                
                video_check_results[camera_name] = {
                    'frames': video_frames,
                    'fps': fps,
                    'duration': duration,
                    'frames_match': bool(frames_match),
                    'frame_difference': abs(video_frames - trajectory_frames)
                }
                
                if not frames_match:
                    result['valid'] = False
                    result['errors'].append(
                        f"{camera_name}: 视频帧数({video_frames})与轨迹帧数({trajectory_frames})不匹配，差异{abs(video_frames - trajectory_frames)}帧"
                    )
            
            result['video_info'] = video_check_results
            result['alignment_check'] = {
                'total_cameras': len(video_check_results),
                'matching_cameras': sum(1 for v in video_check_results.values() if v['frames_match']),
                'all_match': bool(all(v['frames_match'] for v in video_check_results.values()))
            }
            
            # 如果所有相机都匹配，检查时间戳一致性
            if result['alignment_check']['all_match'] and 'timestamp' in episode_data.columns:
                # 检查时间戳间隔是否合理
                timestamps = episode_data['timestamp'].to_numpy()
                if len(timestamps) > 1:
                    time_intervals = np.diff(timestamps)
                    avg_interval = np.mean(time_intervals)
                    min_interval = np.min(time_intervals)
                    max_interval = np.max(time_intervals)
                    
                    # 假设视频帧率为30fps，时间戳间隔应该约为0.033秒
                    expected_interval = 1.0 / 30.0
                    interval_variation = abs(avg_interval - expected_interval)
                    
                    result['alignment_check']['timestamp_analysis'] = {
                        'avg_interval': float(avg_interval),
                        'min_interval': float(min_interval),
                        'max_interval': float(max_interval),
                        'expected_interval': expected_interval,
                        'interval_variation': float(interval_variation),
                        'is_consistent': bool(interval_variation < 0.01)  # 允许10ms的误差
                    }
                    
                    if not result['alignment_check']['timestamp_analysis']['is_consistent']:
                        result['valid'] = False
                        result['errors'].append(
                            f"时间戳间隔不一致: 平均间隔{avg_interval:.3f}s，期望约{expected_interval:.3f}s"
                        )
        
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"验证过程出错: {str(e)}")
        
        return result

    def _trajectory_lengths(self) -> Dict[str, int]:
        """计算轨迹截断后的长度。"""
        target_file, episode_mask = self._find_episode_data_file()
        file_df = pd.read_parquet(target_file)
        episode_data = file_df if episode_mask is None else file_df.loc[episode_mask]
        original_length = len(episode_data)
        if original_length == 0:
            raise ValueError(f"Episode {self.episode_index} contains no frames")

        # 如果是基于时间戳的截断，需要提供时间戳数据
        timestamps = None
        if self.start_time is not None or self.end_time is not None:
            if "timestamp" not in episode_data.columns:
                raise ValueError("Timestamp column not found in trajectory data for time-based truncation")
            timestamps = episode_data["timestamp"].to_numpy()

        start_frame, end_frame = get_truncation_window(
            original_length, timestamps, self.start_time, self.end_time,
            self.keep_start_frame, self.keep_end_frame, self.truncate_frame, self.discard_mode
        )
        
        return {
            "original_length": original_length,
            "new_length": end_frame - start_frame,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }

    def _prepare_all_videos(self, expected_original_length: int, expected_new_length: int, trajectory_timestamps: Optional[np.ndarray] = None) -> Dict[str, Dict]:
        """准备所有视频的截断。"""
        if not self.video_dir.exists():
            logger.warning(f"Video directory does not exist: {self.video_dir}")
            return {}

        processed_videos = {}
        for camera_subdir in self.video_dir.iterdir():
            if not camera_subdir.is_dir():
                continue

            camera_name = camera_subdir.name
            video_path = camera_subdir / f"episode_{self.episode_index:06d}.mp4"
            if not video_path.exists():
                logger.info(f"Skipping missing video: {camera_name}/{video_path.name}")
                continue

            try:
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                if fps <= 0 or width <= 0 or height <= 0:
                    raise ValueError(
                        f"Invalid video metadata for {camera_name}: fps={fps}, width={width}, height={height}"
                    )

                if total_frames != expected_original_length:
                    raise ValueError(
                        f"Video/action length mismatch before truncation for {camera_name}: "
                        f"video has {total_frames} frames, trajectory has {expected_original_length} action steps"
                    )

                # 计算视频的时间戳（假设视频帧与轨迹帧一一对应）
                video_timestamps = trajectory_timestamps if trajectory_timestamps is not None else None
                
                temp_output = video_path.with_suffix('.temp.mp4')
                start_frame, end_frame = get_truncation_window(
                    total_frames, video_timestamps, self.start_time, self.end_time,
                    self.keep_start_frame, self.keep_end_frame, self.truncate_frame, self.discard_mode
                )
                
                written_frames = write_video_frame_window(
                    video_path, temp_output, start_frame, end_frame, fps, width, height
                )
                if written_frames != expected_new_length:
                    raise RuntimeError(
                        f"Frame-accurate video copy failed for {camera_name}: "
                        f"wrote {written_frames}, expected {expected_new_length}"
                    )
                if not temp_output.exists():
                    raise RuntimeError("Temporary output video was not created")

                cap_verify = cv2.VideoCapture(str(temp_output))
                truncated_frames = int(cap_verify.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_verify.release()
                if truncated_frames != expected_new_length:
                    raise RuntimeError(
                        f"Temporary video/action length mismatch for {camera_name}: "
                        f"video has {truncated_frames} frames, trajectory will have {expected_new_length} action steps"
                    )

                processed_videos[camera_name] = {
                    'original_frames': total_frames,
                    'truncated_frames': truncated_frames,
                    'path': str(video_path),
                    'temp_path': str(temp_output),
                }
            except Exception as e:
                logger.error(f"Failed to truncate video {camera_name}: {e}")
                processed_videos[camera_name] = {'error': str(e)}

        return processed_videos

    def _commit_prepared_videos(self, processed_videos: Dict):
        """提交准备好的视频文件。"""
        for camera_name, video_info in processed_videos.items():
            if "error" in video_info:
                continue

            temp_path = Path(video_info.pop("temp_path"))
            video_path = Path(video_info["path"])
            if not temp_path.exists():
                raise RuntimeError(f"Temporary video for {camera_name} disappeared: {temp_path}")
            shutil.move(str(temp_path), str(video_path))

    def _cleanup_prepared_videos(self, processed_videos: Dict):
        """清理临时视频文件。"""
        for video_info in processed_videos.values():
            temp_path = video_info.get("temp_path")
            if temp_path:
                temp_path = Path(temp_path)
                if temp_path.exists():
                    temp_path.unlink()

    def truncate_trajectory(self) -> Dict[str, Any]:
        """截断轨迹数据。"""
        info = self._load_info()
        target_file, episode_mask = self._find_episode_data_file()
        file_df = pd.read_parquet(target_file)

        if episode_mask is None:
            episode_data = file_df.copy()
            episode_positions = None
        else:
            episode_data = file_df.loc[episode_mask].copy()
            episode_positions = np.flatnonzero(episode_mask.to_numpy())

        original_length = len(episode_data)
        if original_length == 0:
            raise ValueError(f"Episode {self.episode_index} contains no frames")

        # 为基于时间戳的截断提供时间戳数据
        timestamps = None
        if self.start_time is not None or self.end_time is not None:
            if "timestamp" not in episode_data.columns:
                raise ValueError("Timestamp column not found in trajectory data for time-based truncation")
            timestamps = episode_data["timestamp"].to_numpy()

        start_frame, end_frame = get_truncation_window(
            original_length, timestamps, self.start_time, self.end_time,
            self.keep_start_frame, self.keep_end_frame, self.truncate_frame, self.discard_mode
        )
        
        if start_frame == 0 and end_frame >= original_length:
            total_frames = self._reindex_all_data_parquets()
            return {
                "new_length": original_length,
                "original_length": original_length,
                "start_frame": 0,
                "end_frame": original_length,
                "start_timestamp": float(episode_data["timestamp"].iloc[0]) if "timestamp" in episode_data.columns else None,
                "end_timestamp": float(episode_data["timestamp"].iloc[-1]) if "timestamp" in episode_data.columns else None,
                "total_frames": total_frames,
                "episode_stats": self._episode_stats_from_dataframe(episode_data.reset_index(drop=True), info),
            }

        has_timestamp = "timestamp" in episode_data.columns
        timestamp_diff = 1.0 / 30.0
        if has_timestamp and original_length > 1:
            diffs = episode_data["timestamp"].diff().dropna()
            if not diffs.empty:
                timestamp_diff = float(diffs.iloc[0])

        truncated_episode_data = episode_data.iloc[start_frame:end_frame].copy()

        truncated_episode_data = truncated_episode_data.reset_index(drop=True)
        if "frame_index" in truncated_episode_data.columns:
            truncated_episode_data["frame_index"] = range(len(truncated_episode_data))
        if has_timestamp:
            start_timestamp = 0.0
            truncated_episode_data["timestamp"] = [
                start_timestamp + i * timestamp_diff for i in range(len(truncated_episode_data))
            ]
        else:
            start_timestamp = None

        if episode_mask is None:
            updated_file_df = truncated_episode_data
        else:
            start_pos = int(episode_positions[0])
            end_pos = int(episode_positions[-1]) + 1
            before_df = file_df.iloc[:start_pos].copy()
            after_df = file_df.iloc[end_pos:].copy()
            updated_file_df = pd.concat([before_df, truncated_episode_data, after_df], ignore_index=True)

        updated_file_df.to_parquet(target_file, index=False)
        total_frames = self._reindex_all_data_parquets()

        episode_stats = self._episode_stats_from_dataframe(truncated_episode_data, info)
        end_timestamp = (
            float(truncated_episode_data["timestamp"].iloc[-1]) if "timestamp" in truncated_episode_data.columns else None
        )
        return {
            "new_length": len(truncated_episode_data),
            "original_length": original_length,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "total_frames": total_frames,
            "episode_stats": episode_stats,
        }

    def update_meta_files(self, truncate_result: Dict):
        """更新元数据文件。"""
        # 统一同步 legacy json/jsonl 与 v3 parquet 元数据，避免截断后不同文件各自为政。
        new_length = int(truncate_result["new_length"])
        start_ts = truncate_result.get("start_timestamp")
        end_ts = truncate_result.get("end_timestamp")
        info = self._load_info()
        episode_lengths = self._all_episode_lengths()
        total_frames = int(truncate_result.get("total_frames") or sum(episode_lengths.values()))
        total_episodes = len(episode_lengths)

        info_path = self.meta_dir / "info.json"
        if info_path.exists():
            info["total_frames"] = total_frames
            info["total_episodes"] = total_episodes
            _write_json(info_path, info)

        episodes_jsonl_path = self.meta_dir / "episodes.jsonl"
        if episodes_jsonl_path.exists():
            rows = _read_jsonl(episodes_jsonl_path)
            updated = False
            for row in rows:
                if row.get("episode_index") != self.episode_index:
                    continue
                row["length"] = new_length
                row["num_frames"] = new_length
                if start_ts is not None:
                    row["start_timestamp"] = start_ts
                if end_ts is not None:
                    row["end_timestamp"] = end_ts
                updated = True
                break
            if updated:
                _write_jsonl(episodes_jsonl_path, rows)

        all_episode_stats = self._compute_all_episode_stats(info)
        episode_stats = all_episode_stats.get(self.episode_index, truncate_result.get("episode_stats", {}))
        flat_stats = _flatten_dict({"stats": episode_stats})
        self._update_legacy_episodes_stats_jsonl(episode_stats, new_length, start_ts, end_ts)
        self._update_episode_metadata_parquet("episodes", self.episode_index, new_length, start_ts, end_ts, flat_stats)
        self._update_episode_metadata_parquet(
            "episodes_stats", self.episode_index, new_length, start_ts, end_ts, flat_stats
        )

        stats_json_path = self.meta_dir / "stats.json"
        if stats_json_path.exists():
            self._write_aggregated_stats_json(all_episode_stats)

    def truncate(self, validate_alignment: bool = True) -> Dict[str, Any]:
        """执行截断操作的主函数。"""
        # 记录时间戳截断信息
        time_info = ""
        if self.start_time is not None and self.end_time is not None:
            time_info = f" time_range=[{self.start_time}s, {self.end_time}s]"
        elif self.truncate_frame is not None:
            time_info = f" truncate_frame={self.truncate_frame}"
        
        logger.info(
            "Starting truncation: episode=%s%s keep_start=%s keep_end=%s",
            self.episode_index,
            time_info,
            self.keep_start_frame,
            self.keep_end_frame,
        )

        errors = self.validate_paths()
        if errors:
            raise ValueError("\n".join(errors))

        # 自动验证视频轨迹对应关系
        if validate_alignment:
            logger.info("Validating video-trajectory alignment...")
            alignment_result = self.validate_video_trajectory_alignment()
            
            if not alignment_result['valid']:
                # 记录警告但继续执行
                logger.warning(
                    f"Video-trajectory alignment issues found: {alignment_result['errors']}"
                )
                logger.warning("Proceeding with truncation despite alignment issues")
            else:
                logger.info("Video-trajectory alignment validated successfully")

        processed_videos = {}
        trajectory_plan = self._trajectory_lengths()
        
        # 获取轨迹时间戳用于视频处理
        trajectory_timestamps = None
        if self.start_time is not None or self.end_time is not None:
            target_file, episode_mask = self._find_episode_data_file()
            file_df = pd.read_parquet(target_file)
            episode_data = file_df if episode_mask is None else file_df.loc[episode_mask]
            if "timestamp" in episode_data.columns:
                trajectory_timestamps = episode_data["timestamp"].to_numpy()

        try:
            processed_videos = self._prepare_all_videos(
                expected_original_length=trajectory_plan["original_length"],
                expected_new_length=trajectory_plan["new_length"],
                trajectory_timestamps=trajectory_timestamps
            )
            failed_videos = {k: v for k, v in processed_videos.items() if 'error' in v}
            if failed_videos:
                details = "; ".join(f"{camera}: {info['error']}" for camera, info in failed_videos.items())
                raise RuntimeError(f"Video/action synchronization check failed: {details}")

            backup_dir = self.backup_manager.create_backup(
                processed_videos, self._data_parquet_files(), self.meta_dir
            )

            trajectory_result = self.truncate_trajectory()
            if trajectory_result["new_length"] != trajectory_plan["new_length"]:
                raise RuntimeError(
                    "Trajectory truncation length changed unexpectedly: "
                    f"planned {trajectory_plan['new_length']}, got {trajectory_result['new_length']}"
                )

            self._commit_prepared_videos(processed_videos)

            self.update_meta_files(trajectory_result)
            
            # 调用LeRobotDataset.recompute_stats()来更新数据集统计信息
            try:
                from lerobot.datasets import LeRobotDataset
                logger.info("Calling dataset.recompute_stats() to update statistics...")
                dataset = LeRobotDataset(self.dataset_root)
                dataset.recompute_stats()
                logger.info("Successfully recomputed dataset statistics")
            except ImportError:
                logger.warning("LeRobotDataset not available, skipping recompute_stats()")
            except Exception as stats_error:
                logger.warning(f"Failed to recompute stats: {stats_error}")

            successful_videos = {k: v for k, v in processed_videos.items() if 'error' not in v}

            logger.info(
                "Truncation finished: videos=%s trajectory_frames=%s",
                len(successful_videos),
                trajectory_result['new_length'],
            )
            return {
                'success': True,
                'processed_videos': processed_videos,
                'successful_videos_count': len(successful_videos),
                'failed_videos_count': 0,
                'trajectory_frames': trajectory_result['new_length'],
                'trajectory_result': trajectory_result,
                'backup_dir': str(backup_dir),
                'episode_number': self.episode_index,
                'time_range': [self.start_time, self.end_time] if self.start_time is not None else None,
            }
        except Exception as e:
            self._cleanup_prepared_videos(processed_videos)
            logger.error(f"Truncation failed: {e}")
            raise