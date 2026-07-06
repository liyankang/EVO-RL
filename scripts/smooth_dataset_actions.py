#!/usr/bin/env python3
"""
对 lerobot 数据集的 action 列做逐 episode 平滑，然后原地覆盖 parquet 文件。

用法:
    python scripts/smooth_dataset_actions.py \
        --dataset /media/Data/huggingface/lerobot/lerobot_v3/data_toiletButton_0703 \
        --method savgol \
        --window 7 \
        --polyorder 2 \
        --dry-run        # 先预览效果，不写文件

    python scripts/smooth_dataset_actions.py \
        --dataset /media/Data/huggingface/lerobot/lerobot_v3/data_toiletButton_0703 \
        --method savgol \
        --window 7 \
        --polyorder 2

方法说明:
    savgol  : Savitzky-Golay 滤波，保留轨迹形状、去除高频抖动（推荐）
    gaussian: 高斯平滑，更激进，适合噪声很大的数据

参数说明:
    --window    : 滤波窗口帧数（奇数），越大越平滑，响应越慢。推荐 5~11。
    --polyorder : savgol 多项式阶数，越小越平滑。推荐 2~3。
    --sigma     : gaussian 标准差（帧数），越大越平滑。推荐 1.5~3。
    --cols      : 要平滑的列名，默认只平滑 action。
                  也可以加 observation.state: --cols action observation.state
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


def smooth_episode(arr: np.ndarray, method: str, window: int, polyorder: int, sigma: float) -> np.ndarray:
    """
    对单个 episode 的动作序列做平滑。

    arr: shape (T, D)，T 是帧数，D 是关节数（7 = 6关节 + 夹爪）
    返回: 平滑后的 (T, D)，dtype float64

    每个关节维度独立做一维滤波，episode 之间不互相影响。
    短于 4 帧的 episode 直接返回原值（太短无法滤波）。
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
            # polyorder 必须小于 window，否则 savgol 会报错
            out[:, d] = savgol_filter(arr[:, d], window_length=w, polyorder=min(polyorder, w - 1))
        elif method == "gaussian":
            # sigma 越大，平滑越强，但会让轨迹"圆润"地偏离原始路径
            out[:, d] = gaussian_filter1d(arr[:, d], sigma=sigma)
    return out


def analyze(original: np.ndarray, smoothed: np.ndarray, episode_ids: np.ndarray) -> None:
    """
    打印平滑前后的跳变幅度对比。

    只统计同一 episode 内的相邻帧差，跨 episode 边界的差不算
    （episode 切换时关节角本来就会跳，那是正常的）。
    """
    joint_names = [f"j{i+1}" for i in range(6)] + ["gripper"]
    n_dim = original.shape[1]

    for label, arr in [("原始", original), ("平滑后", smoothed)]:
        # mask: True 表示相邻两帧属于同一 episode
        mask = episode_ids[:-1] == episode_ids[1:]
        # diff[i] = |arr[i+1] - arr[i]|，再用 mask 过滤掉跨 episode 边界
        diffs = np.abs(np.diff(arr, axis=0))[mask]

        print(f"\n  [{label}] 相邻帧跳变幅度 (p95 / p99 / max)，单位：度")
        for i in range(min(n_dim, len(joint_names))):
            d = diffs[:, i]
            print(f"    {joint_names[i]:8s}: p95={np.percentile(d, 95):.3f}  p99={np.percentile(d, 99):.3f}  max={d.max():.3f}")

        # 任意关节跳变超过阈值的帧比例（反映整体平滑效果）
        overall = diffs.max(axis=1)
        print(f"    {'任意关节':8s}: >2°={(overall > 2).mean() * 100:.1f}%  >5°={(overall > 5).mean() * 100:.1f}%")


def process_file(parquet_path: Path, cols: list[str], method: str,
                 window: int, polyorder: int, sigma: float, dry_run: bool) -> None:
    """
    读取一个 parquet 文件，对指定列逐 episode 平滑，写回原路径。

    流程：
      1. 读 parquet → DataFrame
      2. 对每个 episode 的帧单独做滤波（不跨 episode 混合）
      3. dry-run 时只打印统计，不写文件
      4. 正式运行时先备份原文件（.bak），再写回平滑结果
    """
    print(f"\n处理: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    episode_ids = df["episode_index"].values
    episodes = np.unique(episode_ids)
    print(f"  episodes: {len(episodes)},  总帧数: {len(df)}")

    originals = {}      # 保留原始数组用于统计对比
    smoothed_cols = {}  # 平滑后的数组

    for col in cols:
        if col not in df.columns:
            print(f"  跳过 {col}（列不存在）")
            continue

        # 把 DataFrame 里每行存的 numpy array 堆叠成 (N, D) 矩阵
        arr = np.stack(df[col].values)
        originals[col] = arr.copy()
        out = arr.copy().astype(np.float64)

        # 逐 episode 平滑，保证不同 episode 之间不互相影响
        for ep in episodes:
            ep_idx = np.where(episode_ids == ep)[0]
            out[ep_idx] = smooth_episode(arr[ep_idx], method, window, polyorder, sigma)

        smoothed_cols[col] = out

    # 打印平滑前后对比统计（仅针对 action 列）
    if "action" in originals:
        print("\n  统计对比（action）:")
        analyze(originals["action"], smoothed_cols["action"], episode_ids)

    if dry_run:
        print("\n  [dry-run] 不写文件，退出。")
        return

    # 备份原文件，防止写坏后无法恢复
    backup = parquet_path.with_suffix(".parquet.bak")
    if not backup.exists():
        shutil.copy2(parquet_path, backup)
        print(f"  备份 → {backup.name}")
    else:
        print(f"  备份已存在，跳过备份。")

    # 将平滑后的数组写回 DataFrame，再存为 parquet
    # 注意：parquet 里每行是一个 numpy array，需要用列表推导式拆开
    for col, out in smoothed_cols.items():
        df[col] = [row for row in out]

    df.to_parquet(parquet_path, index=False)
    print(f"  已写入 {parquet_path}")


def main():
    parser = argparse.ArgumentParser(description="平滑 lerobot 数据集 action 序列")
    parser.add_argument("--dataset", required=True, help="数据集根目录（包含 data/ 子目录）")
    parser.add_argument("--method", default="savgol", choices=["savgol", "gaussian"],
                        help="平滑方法：savgol（推荐）或 gaussian")
    parser.add_argument("--window", type=int, default=7,
                        help="savgol 窗口大小（奇数帧数），越大越平滑")
    parser.add_argument("--polyorder", type=int, default=2,
                        help="savgol 多项式阶数，越小越平滑")
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="gaussian sigma（帧数），越大越平滑")
    parser.add_argument("--cols", nargs="+", default=["action"],
                        help="要平滑的列名，默认只平滑 action")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印统计，不写文件")
    args = parser.parse_args()

    # 遍历 data/ 目录下所有 parquet 文件（支持多 chunk）
    data_dir = Path(args.dataset) / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"未找到 parquet 文件：{data_dir}")
        return

    print(f"找到 {len(parquet_files)} 个 parquet 文件")
    print(f"方法: {args.method}  window={args.window}  polyorder={args.polyorder}  sigma={args.sigma}")
    print(f"平滑列: {args.cols}")
    if args.dry_run:
        print("模式: dry-run（只统计，不写文件）")

    for f in parquet_files:
        process_file(f, args.cols, args.method, args.window, args.polyorder, args.sigma, args.dry_run)

    print("\n完成。")


if __name__ == "__main__":
    main()
