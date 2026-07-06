#!/usr/bin/env python3
"""
删除数据集中的指定 episode(s)，后续 episode 重编号，同步更新所有关联文件。

用法:
    # 手动指定删除
    python delete_ep21.py --root /path/to/dataset --episodes 21
    python delete_ep21.py --root /path/to/dataset --episodes 21 5 12

    # 自动过滤异常长度轨迹（IQR 方法）
    python delete_ep21.py --root /path/to/dataset --auto-filter
    python delete_ep21.py --root /path/to/dataset --auto-filter --iqr-factor 2.0
    python delete_ep21.py --root /path/to/dataset --auto-filter --min-length 100 --max-length 400

    # 预览（不实际删除）
    python delete_ep21.py --root /path/to/dataset --auto-filter --dry-run

功能:
    1. 备份 meta/ 目录
    2. 删除 parquet + 视频文件（支持 episode_*.parquet 和 chunk/file-*.parquet 两种格式）
    3. 后续 episode 全部重编号（文件重命名 + parquet 内数据更新）
    4. 重建全局 index
    5. 更新 episodes.jsonl / episodes_stats.jsonl / info.json / episodes parquet
    6. 最终校验 data / videos / meta 一致性
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def backup_meta(root):
    """备份 meta/ 目录"""
    meta_dir = f"{root}/meta"
    backup_dir = f"{root}/meta_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if os.path.isdir(meta_dir):
        shutil.copytree(meta_dir, backup_dir)
        print(f"[备份] meta/ → {backup_dir}")
    return backup_dir


def get_all_episodes(root):
    """从 data/ 目录获取所有存在的 episode 列表"""
    data_files = sorted(glob.glob(f"{root}/data/**/episode_*.parquet", recursive=True))
    episodes = set()
    for f in data_files:
        base = os.path.basename(f)
        ep = int(base.replace("episode_", "").replace(".parquet", ""))
        episodes.add(ep)
    return sorted(episodes)


def get_video_cameras(root):
    """获取所有视频相机目录"""
    vid_pattern = f"{root}/videos/**/episode_*.mp4"
    vid_files = sorted(glob.glob(vid_pattern, recursive=True))
    cameras = set()
    for f in vid_files:
        rel = os.path.relpath(f, root)
        parts = rel.split(os.sep)
        cameras.add(parts[1])  # 如 observation.images.primary
    return sorted(cameras)


def delete_episode_files(root, ep):
    """删除指定 episode 的数据文件"""
    # 删除 parquet
    for f in glob.glob(f"{root}/data/**/episode_{ep:06d}.parquet", recursive=True):
        os.remove(f)
        print(f"  [删除] {os.path.relpath(f, root)}")

    # 删除视频
    for f in glob.glob(f"{root}/videos/**/episode_{ep:06d}.mp4", recursive=True):
        os.remove(f)
        print(f"  [删除] {os.path.relpath(f, root)}")


def renumber_episode_file(root, old_ep, new_ep):
    """重命名 episode 数据文件"""
    # 重命名 parquet
    for old_path in glob.glob(
        f"{root}/data/**/episode_{old_ep:06d}.parquet", recursive=True
    ):
        new_path = old_path.replace(
            f"episode_{old_ep:06d}.parquet", f"episode_{new_ep:06d}.parquet"
        )
        os.rename(old_path, new_path)

    # 重命名视频
    for old_path in glob.glob(
        f"{root}/videos/**/episode_{old_ep:06d}.mp4", recursive=True
    ):
        new_path = old_path.replace(
            f"episode_{old_ep:06d}.mp4", f"episode_{new_ep:06d}.mp4"
        )
        os.rename(old_path, new_path)


def patch_parquet_episode_index(root, old_ep, new_ep, starting_index):
    """
    修改 parquet 内的 episode_index 和重建全局 index。

    返回该 episode 的帧数和新版本的起始 index 偏移。
    """
    pattern = f"{root}/data/**/episode_{new_ep:06d}.parquet"
    parquet_files = glob.glob(pattern, recursive=True)
    if not parquet_files:
        print(f"  [警告] 找不到 parquet: episode_{new_ep:06d}.parquet")
        return 0, starting_index

    df = pd.read_parquet(parquet_files[0])
    n_frames = len(df)

    # 更新 episode_index
    df["episode_index"] = new_ep

    # 重建帧索引
    df["frame_index"] = range(n_frames)

    # 重建全局 index
    df["index"] = range(starting_index, starting_index + n_frames)

    # 写回
    df.to_parquet(parquet_files[0], index=False)
    return n_frames, starting_index + n_frames


def load_jsonl(path):
    """加载 JSONL 文件，不存在则返回空列表"""
    if not os.path.exists(path):
        print(f"  [提示] {os.path.basename(path)} 不存在，自动创建")
        return None
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(path, data):
    """写入 JSONL 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_parquet_by_ep(root, ep):
    """通过 episode 编号读取 parquet 文件"""
    files = glob.glob(f"{root}/data/**/episode_{ep:06d}.parquet", recursive=True)
    if not files:
        raise FileNotFoundError(f"找不到 parquet: episode_{ep:06d}.parquet")
    return pd.read_parquet(files[0], engine="pyarrow")


def update_episodes_jsonl(root, ep_to_old_new):
    """更新 episodes.jsonl：自动从 parquet 重建"""
    path = f"{root}/meta/episodes.jsonl"
    remaining_eps = get_all_episodes(root)

    episodes = load_jsonl(path)
    if episodes is None or len(episodes) != len(remaining_eps):
        # 不存在或数量不匹配，从 parquet 自动重建
        print("  [自动] 从 data/ 重建 episodes.jsonl")
        tasks = guess_tasks(root)
        episodes = []
        for ep in sorted(remaining_eps):
            df = read_parquet_by_ep(root, ep)
            length = len(df)
            episodes.append({
                "episode_index": ep,
                "tasks": tasks,
                "length": length,
            })
    else:
        # 删除被删的 episode
        deleted_eps = {e for e, (old, new) in ep_to_old_new.items() if old is None}
        episodes = [e for e in episodes if e["episode_index"] not in deleted_eps]
        # 重编号
        for ep in episodes:
            old_idx = ep["episode_index"]
            if old_idx in ep_to_old_new:
                new_idx = ep_to_old_new[old_idx][1]
                if new_idx is not None and new_idx != old_idx:
                    ep["episode_index"] = new_idx
        # 刷新长度
        for ep in episodes:
            df = read_parquet_by_ep(root, ep["episode_index"])
            ep["length"] = len(df)

    episodes.sort(key=lambda x: x["episode_index"])
    write_jsonl(path, episodes)
    print(f"[更新] episodes.jsonl: {len(episodes)} 条")


def compute_stats_from_parquet(root, ep):
    """从 parquet 自动计算该 episode 的统计信息（min/max/mean/std）"""
    df = read_parquet_by_ep(root, ep)
    drop_cols = ["episode_index", "frame_index", "index", "timestamp"]
    num_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype in ["float32", "float64", "int32", "int64"]]

    stats = {}
    for col in num_cols:
        s = df[col]
        stats[col] = {
            "min": s.min().tolist() if hasattr(s.min(), "__iter__") else [s.min()],
            "max": s.max().tolist() if hasattr(s.max(), "__iter__") else [s.max()],
            "mean": s.mean().tolist() if hasattr(s.mean(), "__iter__") else [s.mean()],
            "std": s.std().tolist() if hasattr(s.std(), "__iter__") else [s.std()],
        }
    return stats


def guess_tasks(root):
    """从 tasks.jsonl 猜测任务名，没有则用默认"""
    tasks_path = f"{root}/meta/tasks.jsonl"
    if os.path.exists(tasks_path):
        tasks_data = load_jsonl(tasks_path)
        if tasks_data:
            return [tasks_data[0]["task"]]
    return ["unknown"]


def update_episodes_stats_jsonl(root, ep_to_old_new):
    """更新 episodes_stats.jsonl，检查不到则自动从 parquet 计算"""
    path = f"{root}/meta/episodes_stats.jsonl"
    remaining_eps = get_all_episodes(root)

    stats_existing = load_jsonl(path)
    need_rebuild = (
        stats_existing is None
        or len(stats_existing) != len(remaining_eps)
    )

    if need_rebuild:
        # 不存在或数量不匹配，从 parquet 自动计算
        print("  [自动] 从 data/ 重建 episodes_stats.jsonl")
        stats = []
        for ep in sorted(remaining_eps):
            ep_stats = compute_stats_from_parquet(root, ep)
            stats.append({"episode_index": ep, "stats": ep_stats})
    else:
        stats = stats_existing
        # 删除被删的 episode
        deleted_eps = {e for e, (old, new) in ep_to_old_new.items() if old is None}
        stats = [s for s in stats if s["episode_index"] not in deleted_eps]
        # 重编号
        for s in stats:
            old_idx = s["episode_index"]
            if old_idx in ep_to_old_new:
                new_idx = ep_to_old_new[old_idx][1]
                if new_idx is not None and new_idx != old_idx:
                    s["episode_index"] = new_idx

    stats.sort(key=lambda x: x["episode_index"])
    write_jsonl(path, stats)
    print(f"[更新] episodes_stats.jsonl: {len(stats)} 条")


def update_info_json(root):
    """重新统计并更新 info.json，不存在则自动创建"""
    path = f"{root}/meta/info.json"

    data_files = sorted(
        glob.glob(f"{root}/data/**/episode_*.parquet", recursive=True)
    )
    n_eps = len(data_files)
    total_frames = 0
    for f in data_files:
        df = pd.read_parquet(f)
        total_frames += len(df)

    # 统计视频
    vid_files = glob.glob(f"{root}/videos/**/episode_*.mp4", recursive=True)
    total_videos = len(vid_files)

    if os.path.exists(path):
        with open(path) as f:
            info = json.load(f)
    else:
        print("  [自动] info.json 不存在，自动创建")
        info = {
            "codebase_version": "v2.1",
            "robot_type": "unknown",
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 1,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": 30,
            "splits": {"train": "0:0"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        }

    info["total_episodes"] = n_eps
    info["total_frames"] = total_frames
    info["total_videos"] = total_videos
    info["splits"]["train"] = f"0:{n_eps}"

    with open(path, "w") as f:
        json.dump(info, f, indent=2)

    print(
        f"[更新] info.json: total_episodes={n_eps}, "
        f"total_frames={total_frames}, total_videos={total_videos}"
    )
    return n_eps, total_frames, total_videos


def validate(root, expected_n_eps):
    """校验 data / videos / meta 一致性"""
    print("\n===== 最终校验 =====")
    errors = []

    # 1. data 文件
    data_files = sorted(
        glob.glob(f"{root}/data/**/episode_*.parquet", recursive=True)
    )
    data_eps = set()
    for f in data_files:
        base = os.path.basename(f)
        ep = int(base.replace("episode_", "").replace(".parquet", ""))
        data_eps.add(ep)

    expected_eps = set(range(expected_n_eps))
    missing_in_data = expected_eps - data_eps
    extra_in_data = data_eps - expected_eps
    if missing_in_data:
        errors.append(f"[错误] data/ 缺少 episode: {sorted(missing_in_data)}")
    if extra_in_data:
        errors.append(f"[错误] data/ 多余 episode: {sorted(extra_in_data)}")

    # 2. 视频文件
    for camera in get_video_cameras(root):
        vid_files = sorted(
            glob.glob(f"{root}/videos/**/{camera}/episode_*.mp4", recursive=True)
        )
        vid_eps = set()
        for f in vid_files:
            base = os.path.basename(f)
            ep = int(base.replace("episode_", "").replace(".mp4", ""))
            vid_eps.add(ep)

        missing_vid = expected_eps - vid_eps
        extra_vid = vid_eps - expected_eps
        if missing_vid:
            errors.append(
                f"[错误] {camera} 缺少 episode 视频: {sorted(missing_vid)}"
            )
        if extra_vid:
            errors.append(
                f"[错误] {camera} 多余 episode 视频: {sorted(extra_vid)}"
            )
        print(f"  [校验] {camera}: {len(vid_files)} 个视频 ✓" if not (missing_vid or extra_vid) else f"  [校验] {camera}: {len(vid_files)} 个视频 ✗")

    # 3. meta 文件（不存在也视为正常）
    meta_eps = set()
    meta_path = f"{root}/meta/episodes.jsonl"
    meta_data = load_jsonl(meta_path)
    if meta_data:
        meta_eps = {e["episode_index"] for e in meta_data}
        missing_meta = expected_eps - meta_eps
        extra_meta = meta_eps - expected_eps
        if missing_meta:
            errors.append(f"[警告] episodes.jsonl 缺少: {sorted(missing_meta)}")
        if extra_meta:
            errors.append(f"[警告] episodes.jsonl 多余: {sorted(extra_meta)}")
        print(f"  [校验] episodes.jsonl: {len(meta_data)} 条 ✓" if not (missing_meta or extra_meta) else f"  [校验] episodes.jsonl: {len(meta_data)} 条 ✗")
    else:
        print(f"  [校验] episodes.jsonl: 不存在（跳过）")

    # 4. 全局 index 连续性
    all_df = []
    for f in sorted(data_files):
        df = pd.read_parquet(f)
        all_df.append(df)

    if all_df:
        combined = pd.concat(all_df, ignore_index=True)
        expected_index = list(range(len(combined)))
        actual_index = combined["index"].tolist()
        if actual_index != expected_index:
            mismatches = sum(1 for a, b in zip(actual_index, expected_index) if a != b)
            errors.append(
                f"[错误] 全局 index 不连续: {mismatches}/{len(combined)} 处不匹配"
            )
        else:
            print(f"  [校验] 全局 index: {len(combined)} 条连续 ✓")

        # 5. episode_index 一致性
        actual_eps = combined["episode_index"].unique()
        missing_ep_index = expected_eps - set(actual_eps)
        if missing_ep_index:
            errors.append(
                f"[错误] parquet 内缺少 episode_index: {sorted(missing_ep_index)}"
            )
        else:
            print(f"  [校验] episode_index 字段: {len(actual_eps)} 个可对齐 ✓")

    if errors:
        print("\n❌ 校验发现错误:")
        for err in errors:
            print(f"  {err}")
        return False
    else:
        print("\n✅ 全部校验通过!")
        return True


def get_episode_lengths(root: str) -> dict[int, int]:
    """返回 {episode_index: length}，兼容两种格式：
    - episode_*.parquet（旧格式，每条一个文件）
    - data/chunk-*/file-*.parquet（新格式，多条共用一个文件）
    """
    lengths: dict[int, int] = {}

    # 旧格式
    for f in glob.glob(f"{root}/data/**/episode_*.parquet", recursive=True):
        ep = int(os.path.basename(f).replace("episode_", "").replace(".parquet", ""))
        df = pd.read_parquet(f, columns=["episode_index"])
        lengths[ep] = len(df)

    if lengths:
        return lengths

    # 新格式：chunk/file parquet，从 episodes parquet 读长度
    ep_files = sorted(glob.glob(f"{root}/meta/episodes/**/file-*.parquet", recursive=True))
    if ep_files:
        eps_df = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
        if "episode_index" in eps_df.columns and "length" in eps_df.columns:
            for _, row in eps_df.iterrows():
                lengths[int(row["episode_index"])] = int(row["length"])
            return lengths

    # 新格式回退：从 data parquet 直接统计
    for f in sorted(glob.glob(f"{root}/data/**/file-*.parquet", recursive=True)):
        df = pd.read_parquet(f, columns=["episode_index"])
        for ep, grp in df.groupby("episode_index"):
            lengths[int(ep)] = lengths.get(int(ep), 0) + len(grp)

    return lengths


def auto_detect_outliers(
    lengths: dict[int, int],
    iqr_factor: float,
    min_length: int | None,
    max_length: int | None,
) -> list[int]:
    """用 IQR 方法检测过长/过短轨迹，也支持手动指定绝对上下限。"""
    arr = np.array(list(lengths.values()), dtype=float)
    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr = q75 - q25
    iqr_lower = q25 - iqr_factor * iqr
    iqr_upper = q75 + iqr_factor * iqr

    lower = max(iqr_lower, min_length) if min_length is not None else iqr_lower
    upper = min(iqr_upper, max_length) if max_length is not None else iqr_upper

    print(f"长度统计: min={arr.min():.0f}  Q25={q25:.0f}  Q75={q75:.0f}  max={arr.max():.0f}")
    print(f"过滤阈值: [{lower:.0f}, {upper:.0f}]  (IQR×{iqr_factor} + 手动限制)")

    outliers = [ep for ep, length in lengths.items() if not (lower <= length <= upper)]
    return sorted(outliers)


def delete_episodes_chunk_format(root: str, to_delete: set[int], dry_run: bool) -> None:
    """删除 chunk/file parquet 格式数据集中指定 episode 的帧，并重映射 episode_index。"""
    data_files = sorted(glob.glob(f"{root}/data/**/file-*.parquet", recursive=True))
    ep_files   = sorted(glob.glob(f"{root}/meta/episodes/**/file-*.parquet", recursive=True))

    # 读取当前 episodes 表，确定映射
    eps_df = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
    surviving = eps_df[~eps_df["episode_index"].isin(to_delete)].copy()
    old_indices = sorted(surviving["episode_index"].tolist())
    idx_map = {old: new for new, old in enumerate(old_indices)}

    print(f"  删除 episodes: {sorted(to_delete)}")
    print(f"  剩余: {len(surviving)} 条")

    if dry_run:
        print("  [dry-run] 不写文件")
        return

    # data parquet
    for fpath in data_files:
        df = pd.read_parquet(fpath)
        before = len(df)
        df = df[~df["episode_index"].isin(to_delete)].copy()
        df["episode_index"] = df["episode_index"].map(idx_map)
        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        df["index"] = df.index
        df.to_parquet(fpath, index=False)
        print(f"  data/{os.path.basename(fpath)}: 删除 {before - len(df)} 帧，剩余 {len(df)}")

    # episodes parquet
    surviving["episode_index"] = surviving["episode_index"].map(idx_map)
    surviving = surviving.reset_index(drop=True)
    for fpath in ep_files:
        surviving.to_parquet(fpath, index=False)
        print(f"  episodes/{os.path.basename(fpath)}: 剩余 {len(surviving)} 条")

    # info.json
    info_path = os.path.join(root, "meta", "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        info["total_episodes"] = len(surviving)
        info["total_frames"]   = int(surviving["length"].sum())
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"  info.json: total_episodes={info['total_episodes']}, total_frames={info['total_frames']}")


def validate_v3(root: str) -> bool:
    """校验 V3 格式数据集一致性。"""
    print("\n===== 最终校验 =====")
    errors = []

    data_files = sorted(glob.glob(f"{root}/data/**/file-*.parquet", recursive=True))
    ep_files   = sorted(glob.glob(f"{root}/meta/episodes/**/file-*.parquet", recursive=True))
    info_path  = os.path.join(root, "meta", "info.json")

    if not data_files:
        errors.append("[错误] 找不到 data/**/file-*.parquet")
    if not ep_files:
        errors.append("[错误] 找不到 meta/episodes/**/file-*.parquet")

    if data_files and ep_files:
        df_all = pd.concat([pd.read_parquet(p) for p in data_files], ignore_index=True)
        eps_df = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)

        data_eps  = set(df_all["episode_index"].unique())
        meta_eps  = set(eps_df["episode_index"].unique())

        if data_eps != meta_eps:
            errors.append(f"[错误] data 与 episodes parquet episode_index 不一致: "
                          f"only_data={data_eps - meta_eps}, only_meta={meta_eps - data_eps}")
        else:
            print(f"  [校验] episode_index 一致: {len(data_eps)} 条 ✓")

        expected_idx = list(range(len(df_all)))
        if df_all["index"].tolist() != expected_idx:
            errors.append("[错误] 全局 index 不连续")
        else:
            print(f"  [校验] 全局 index: {len(df_all)} 条连续 ✓")

        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            if info.get("total_episodes") != len(meta_eps):
                errors.append(f"[错误] info.json total_episodes={info['total_episodes']} 与实际 {len(meta_eps)} 不符")
            if info.get("total_frames") != len(df_all):
                errors.append(f"[错误] info.json total_frames={info['total_frames']} 与实际 {len(df_all)} 不符")
            if not errors:
                print(f"  [校验] info.json 一致 ✓")

    if errors:
        print("\n❌ 校验发现错误:")
        for e in errors:
            print(f"  {e}")
        return False
    print("\n✅ 全部校验通过!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="删除 episode(s)（V3 chunk/file 格式），后续重编号，同步更新所有关联文件"
    )
    parser.add_argument("--root", required=True, help="数据集根目录路径")
    parser.add_argument("--episodes", type=int, nargs="+",
                        help="手动指定要删除的 episode 编号（与 --auto-filter 二选一）")
    parser.add_argument("--auto-filter", action="store_true",
                        help="自动检测并删除异常长度轨迹（IQR 方法）")
    parser.add_argument("--iqr-factor", type=float, default=1.5,
                        help="IQR 倍数，越小越激进，默认 1.5")
    parser.add_argument("--min-length", type=int, default=None,
                        help="强制最短保留帧数（低于此值必删）")
    parser.add_argument("--max-length", type=int, default=None,
                        help="强制最长保留帧数（高于此值必删）")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览将删除哪些 episode，不实际修改文件")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(os.path.join(root, "meta")):
        print(f"[错误] 不是有效的数据集目录: {root}/meta 不存在")
        sys.exit(1)

    # ── 确定要删除的 episode ──────────────────────────────────────────────────
    if args.auto_filter:
        lengths = get_episode_lengths(root)
        if not lengths:
            print("[错误] 找不到 episode 数据")
            sys.exit(1)
        eps_to_delete = set(auto_detect_outliers(
            lengths, args.iqr_factor, args.min_length, args.max_length
        ))
        # 如果同时传了 --episodes，合并
        if args.episodes:
            eps_to_delete |= set(args.episodes)
    elif args.episodes:
        eps_to_delete = set(args.episodes)
    else:
        parser.error("请指定 --episodes 或 --auto-filter")

    eps_to_delete = sorted(eps_to_delete)
    print(f"\n数据集: {root}")
    print(f"将删除 {len(eps_to_delete)} 条 episodes: {eps_to_delete}")

    if not eps_to_delete:
        print("没有需要删除的 episode，退出。")
        return

    if args.dry_run:
        print("\n[dry-run] 以上为预览，不修改任何文件。")
        return

    # ── 备份 meta ────────────────────────────────────────────────────────────
    backup_meta(root)

    # ── 执行删除（V3 格式）───────────────────────────────────────────────────
    print("\n===== 删除与重编号 =====")
    delete_episodes_chunk_format(root, set(eps_to_delete), dry_run=False)

    # ── 校验 ─────────────────────────────────────────────────────────────────
    validate_v3(root)

    ep_files = sorted(glob.glob(f"{root}/meta/episodes/**/file-*.parquet", recursive=True))
    remaining = len(pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True))
    print(f"\n===== 完成! =====")
    print(f"删除 {len(eps_to_delete)} 条，剩余 {remaining} 条")
    print(f"\n后续步骤（需手动执行）:")
    print(f"  1. 重算归一化统计:  python -m lerobot.datasets.v30.augment_dataset_quantile_stats --repo-id=<id> --root={os.path.dirname(root)} --overwrite")
    print(f"  2. 重算 indicator:  lerobot-value-infer --dataset.repo_id=<id> --dataset.root={os.path.dirname(root)} ...")


if __name__ == "__main__":
    main()
