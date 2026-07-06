#!/usr/bin/env python3
"""
离线重算数据集 stats.json，不依赖网络/LeRobotDataset。
直接读 data parquet，计算 action / observation.state 的 min/max/mean/std/quantile。

用法:
    python scripts/recompute_stats.py \
        --dataset /media/Data/huggingface/lerobot/lerobot_v3/data_toiletButton_0703
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def compute_stats(arr: np.ndarray) -> dict:
    """计算一个 (N, D) 数组的统计量，返回 lerobot stats 格式。"""
    return {
        "min":  arr.min(axis=0).tolist(),
        "max":  arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std":  arr.std(axis=0).tolist(),
        "q01":  np.percentile(arr, 1,  axis=0).tolist(),
        "q99":  np.percentile(arr, 99, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="数据集根目录")
    args = parser.parse_args()

    root = Path(args.dataset)
    stats_path = root / "meta" / "stats.json"

    # 读所有 data parquet
    parquets = sorted((root / "data").rglob("*.parquet"))
    print(f"读取 {len(parquets)} 个 parquet 文件...")
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    print(f"总帧数: {len(df)}")

    # 读现有 stats.json（保留其他字段，只更新 action/observation.state）
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        shutil.copy2(stats_path, str(stats_path) + ".bak")
        print(f"备份 stats.json → stats.json.bak")
    else:
        stats = {}

    # 计算各列统计
    target_cols = ["action", "observation.state"]
    for col in target_cols:
        if col not in df.columns:
            print(f"  跳过 {col}（列不存在）")
            continue
        arr = np.stack(df[col].values).astype(np.float64)
        stats[col] = compute_stats(arr)
        print(f"  {col}: shape={arr.shape}  mean={np.array(stats[col]['mean']).round(3).tolist()}")

    # 也更新 complementary_info.* 的连续值列
    for col in df.columns:
        if not col.startswith("complementary_info"):
            continue
        try:
            arr = np.array(df[col].tolist(), dtype=np.float64).reshape(len(df), -1)
            if arr.shape[1] == 0:
                continue
            stats[col] = compute_stats(arr)
            print(f"  {col}: shape={arr.shape}")
        except Exception:
            pass

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ stats.json 已更新: {stats_path}")


if __name__ == "__main__":
    main()
