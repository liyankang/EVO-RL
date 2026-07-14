#!/usr/bin/env python3
"""
动作数据平滑模块，支持对数据集的 action 列进行平滑处理。

用法:
    from action_smoother import ActionSmoother
    
    smoother = ActionSmoother(
        dataset_root="/path/to/dataset",
        method="savgol",
        window=7,
        polyorder=2
    )
    result = smoother.smooth_dataset(dry_run=True)
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


class ActionSmoother:
    """对数据集的 action 列进行平滑处理"""

    def __init__(
        self,
        dataset_root: Path,
        method: str = 'savgol',
        window: int = 7,
        polyorder: int = 2,
        sigma: float = 2.0,
        cols: Optional[List[str]] = None
    ):
        self.dataset_root = Path(dataset_root)
        self.data_dir = self.dataset_root / "data"
        self.method = method
        self.window = window
        self.polyorder = polyorder
        self.sigma = sigma
        self.cols = cols or ["action"]

    def _smooth_episode(
        self, 
        arr: np.ndarray, 
        method: str, 
        window: int, 
        polyorder: int, 
        sigma: float
    ) -> np.ndarray:
        """
        对单个 episode 的动作序列做平滑。

        Args:
            arr: shape (T, D)，T 是帧数，D 是关节数
            method: 平滑方法 ('savgol' 或 'gaussian')
            window: 滤波窗口帧数
            polyorder: savgol 多项式阶数
            sigma: gaussian 标准差

        Returns:
            平滑后的 (T, D) 数组
        """
        if len(arr) < 4:
            return arr.copy()

        out = arr.copy().astype(np.float64)
        for d in range(arr.shape[1]):  # 逐关节维度滤波
            if method == "savgol":
                # 窗口不能超过序列长度，且必须是奇数
                w = min(window, len(arr))
                if w % 2 == 0:
                    w -= 1
                if w < 3:
                    continue
                # polyorder 必须小于 window
                out[:, d] = savgol_filter(
                    arr[:, d], 
                    window_length=w, 
                    polyorder=min(polyorder, w - 1)
                )
            elif method == "gaussian":
                out[:, d] = gaussian_filter1d(arr[:, d], sigma=sigma)
        return out

    def _analyze_statistics(
        self, 
        original: np.ndarray, 
        smoothed: np.ndarray, 
        episode_ids: np.ndarray
    ) -> Dict[str, Any]:
        """
        分析平滑前后的跳变幅度对比。

        Args:
            original: 原始数据
            smoothed: 平滑后数据
            episode_ids: episode 索引数组

        Returns:
            统计信息字典
        """
        joint_names = [f"j{i+1}" for i in range(6)] + ["gripper"]
        n_dim = original.shape[1]

        stats = {
            "original": {},
            "smoothed": {},
            "improvement": {}
        }

        for label, arr in [("original", original), ("smoothed", smoothed)]:
            # mask: True 表示相邻两帧属于同一 episode
            mask = episode_ids[:-1] == episode_ids[1:]
            # diff[i] = |arr[i+1] - arr[i]|，再用 mask 过滤掉跨 episode 边界
            diffs = np.abs(np.diff(arr, axis=0))[mask]

            joint_stats = {}
            for i in range(min(n_dim, len(joint_names))):
                d = diffs[:, i]
                joint_stats[joint_names[i]] = {
                    "p95": float(np.percentile(d, 95)),
                    "p99": float(np.percentile(d, 99)),
                    "max": float(d.max()),
                    "mean": float(d.mean()),
                    "std": float(d.std())
                }

            # 任意关节跳变超过阈值的帧比例
            overall = diffs.max(axis=1)
            joint_stats["overall"] = {
                "gt_2deg": float((overall > 2).mean() * 100),
                "gt_5deg": float((overall > 5).mean() * 100),
                "gt_10deg": float((overall > 10).mean() * 100)
            }

            stats[label] = joint_stats

        # 计算改进百分比（只对有p95指标的关节计算）
        if "action" in self.cols or not self.cols:
            for joint in joint_names + ["overall"]:
                if joint in stats["original"] and joint in stats["smoothed"]:
                    orig = stats["original"][joint]
                    smooth = stats["smoothed"][joint]
                    
                    # 只对有p95等指标的关节计算改进
                    if "p95" in orig and "p95" in smooth:
                        stats["improvement"][joint] = {
                            "p95": ((orig["p95"] - smooth["p95"]) / orig["p95"] * 100) if orig["p95"] > 0 else 0,
                            "p99": ((orig["p99"] - smooth["p99"]) / orig["p99"] * 100) if orig["p99"] > 0 else 0,
                            "max": ((orig["max"] - smooth["max"]) / orig["max"] * 100) if orig["max"] > 0 else 0
                        }

        return stats

    def _process_file(
        self, 
        parquet_path: Path, 
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        处理单个 parquet 文件，对指定列逐 episode 平滑。

        Args:
            parquet_path: parquet 文件路径
            dry_run: 是否为预览模式

        Returns:
            处理结果字典
        """
        logger.info(f"处理文件: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        if "episode_index" not in df.columns:
            return {
                "error": f"文件 {parquet_path} 缺少 episode_index 列"
            }

        episode_ids = df["episode_index"].values
        episodes = np.unique(episode_ids)

        result = {
            "file": str(parquet_path),
            "episodes": int(len(episodes)),
            "total_frames": int(len(df)),
            "processed_cols": [],
            "statistics": {},
            "backup_created": False
        }

        originals = {}  # 保留原始数组用于统计对比
        smoothed_cols = {}  # 平滑后的数组

        for col in self.cols:
            if col not in df.columns:
                logger.warning(f"跳过 {col}（列不存在）")
                continue

            # 把 DataFrame 里每行存的 numpy array 堆叠成 (N, D) 矩阵
            arr = np.stack(df[col].values)
            originals[col] = arr.copy()
            out = arr.copy().astype(np.float64)

            # 逐 episode 平滑，保证不同 episode 之间不互相影响
            for ep in episodes:
                ep_idx = np.where(episode_ids == ep)[0]
                out[ep_idx] = self._smooth_episode(
                    arr[ep_idx], 
                    self.method, 
                    self.window, 
                    self.polyorder, 
                    self.sigma
                )

            smoothed_cols[col] = out
            result["processed_cols"].append(col)

            # 打印平滑前后对比统计（仅针对 action 列）
            if col == "action":
                result["statistics"] = self._analyze_statistics(
                    originals["action"], 
                    smoothed_cols["action"], 
                    episode_ids
                )

        if dry_run:
            logger.info("[dry-run] 不写文件，退出")
            result["dry_run"] = True
            return result

        # 备份原文件
        backup = parquet_path.with_suffix(".parquet.bak")
        if not backup.exists():
            shutil.copy2(parquet_path, backup)
            logger.info(f"备份 → {backup.name}")
            result["backup_created"] = True
        else:
            logger.info(f"备份已存在，跳过备份")

        # 将平滑后的数组写回 DataFrame
        for col, out in smoothed_cols.items():
            df[col] = [row for row in out]

        df.to_parquet(parquet_path, index=False)
        logger.info(f"已写入 {parquet_path}")

        result["dry_run"] = False
        result["success"] = True

        return result

    def smooth_dataset(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        对整个数据集进行平滑处理。

        Args:
            dry_run: 是否为预览模式

        Returns:
            处理结果字典
        """
        parquet_files = sorted(self.data_dir.rglob("*.parquet"))
        if not parquet_files:
            return {
                "success": False,
                "error": f"未找到 parquet 文件：{self.data_dir}"
            }

        logger.info(f"找到 {len(parquet_files)} 个 parquet 文件")
        logger.info(
            f"方法: {self.method}  window={self.window}  "
            f"polyorder={self.polyorder}  sigma={self.sigma}"
        )
        logger.info(f"平滑列: {self.cols}")

        results = []
        for f in parquet_files:
            try:
                result = self._process_file(f, dry_run)
                results.append(result)
            except Exception as e:
                logger.error(f"处理文件 {f} 时出错: {e}")
                results.append({
                    "file": str(f),
                    "error": str(e),
                    "success": False
                })

        # 汇总统计
        total_frames = sum(r.get("total_frames", 0) for r in results)
        total_episodes = sum(r.get("episodes", 0) for r in results)
        successful_files = sum(1 for r in results if r.get("success"))
        failed_files = sum(1 for r in results if r.get("error"))

        # 合并统计信息（如果有 action 列的统计）
        combined_stats = None
        action_results = [r for r in results if "statistics" in r and r["statistics"]]
        if action_results:
            # 取第一个文件的统计作为示例
            combined_stats = action_results[0]["statistics"]

        return {
            "success": failed_files == 0,
            "total_files": len(parquet_files),
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_frames": total_frames,
            "total_episodes": total_episodes,
            "dry_run": dry_run,
            "method": self.method,
            "parameters": {
                "window": self.window,
                "polyorder": self.polyorder,
                "sigma": self.sigma,
                "cols": self.cols
            },
            "statistics": combined_stats,
            "file_results": results
        }


def create_smoother(dataset_root: str, **kwargs) -> ActionSmoother:
    """创建 ActionSmoother 实例的便捷函数。"""
    return ActionSmoother(dataset_root, **kwargs)