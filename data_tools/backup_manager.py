"""备份管理模块，用于创建和管理截断操作的备份。"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from file_utils import _read_json, _write_json


class BackupManager:
    """管理视频截断操作的备份和恢复。"""

    def __init__(self, dataset_root: Path, episode_index: int):
        self.dataset_root = Path(dataset_root)
        self.episode_index = episode_index
        self.backup_root = self.dataset_root / ".video_truncator_backups"
        self.undo_manifest_path = self.dataset_root / ".video_truncator_undo.json"

    def _relative_to_dataset(self, path: Path) -> str:
        """获取相对于数据集根目录的路径。"""
        return Path(path).resolve().relative_to(self.dataset_root.resolve()).as_posix()

    def _copy_into_backup(self, backup_dir: Path, path: Path) -> str:
        """将文件复制到备份目录中。"""
        rel_path = self._relative_to_dataset(path)
        dst = backup_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        return rel_path

    def create_backup(self, processed_videos: Dict, parquet_files: List[Path], meta_dir: Path) -> Path:
        """创建备份快照。
        
        Args:
            processed_videos: 处理后的视频信息字典
            parquet_files: 需要备份的parquet文件列表
            meta_dir: 元数据目录
            
        Returns:
            备份目录路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backup_root / f"{timestamp}_episode_{self.episode_index:06d}"
        backup_dir.mkdir(parents=True, exist_ok=False)

        files = []
        # 备份parquet文件
        for parquet_path in parquet_files:
            files.append(self._copy_into_backup(backup_dir, parquet_path))

        # 备份视频文件
        for video_info in processed_videos.values():
            video_path = Path(video_info.get("path", ""))
            if video_path.exists():
                files.append(self._copy_into_backup(backup_dir, video_path))

        # 备份元数据目录
        if meta_dir.exists():
            meta_dst = backup_dir / "meta"
            shutil.copytree(meta_dir, meta_dst)

        # 创建备份清单
        manifest = {
            "backup_dir": str(backup_dir),
            "dataset_root": str(self.dataset_root),
            "episode_index": self.episode_index,
            "created_at": timestamp,
            "files": sorted(set(files)),
            "meta_dir": "meta" if meta_dir.exists() else None,
        }
        _write_json(backup_dir / "manifest.json", manifest)
        _write_json(self.undo_manifest_path, manifest)
        return backup_dir

    def restore_latest_backup(self) -> Dict[str, Any]:
        """从最新备份恢复数据。
        
        Returns:
            恢复操作的结果信息
        """
        manifest_path = self.undo_manifest_path
        manifest = _read_json(manifest_path)
        if not manifest:
            raise FileNotFoundError("没有可回退的上一步操作")

        backup_dir = Path(manifest["backup_dir"])
        if not backup_dir.exists():
            raise FileNotFoundError(f"回退快照不存在: {backup_dir}")

        meta_backup = backup_dir / "meta"
        meta_target = self.dataset_root / "meta"
        if meta_backup.exists():
            shutil.rmtree(meta_target)
        shutil.copytree(meta_backup, meta_target)

        for rel_path in manifest.get("files", []):
            src = backup_dir / rel_path
            dst = self.dataset_root / rel_path
            if not src.exists():
                raise FileNotFoundError(f"快照文件缺失: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        manifest_path.unlink(missing_ok=True)
        return {
            "success": True,
            "restored_from": str(backup_dir),
            "episode_index": manifest.get("episode_index"),
            "restored_files": len(manifest.get("files", [])),
        }


def restore_latest_backup(dataset_root: Path) -> Dict[str, Any]:
    """从最新备份恢复数据（便捷函数）。
    
    Args:
        dataset_root: 数据集根目录
        
    Returns:
        恢复操作的结果信息
    """
    # 这里使用默认的episode_index，因为从备份中可以获取
    backup_manager = BackupManager(dataset_root, episode_index=0)
    return backup_manager.restore_latest_backup()