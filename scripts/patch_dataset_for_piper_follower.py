#!/usr/bin/env python3
"""
Patch a v2→v3 converted Piper dataset to be compatible with the current
piper_follower + HIL record system.

Changes:
  1. info.json:  robot_type piper→piper_follower, shape 10→7, rename joints, add HIL fields
  2. stats.json: truncate action/observation.state stats from 10→7
  3. data parquet: truncate action & observation.state columns from (10,)→(7,)
  4. episodes parquet: truncate per-episode stats from 10→7, add episode_success="success"

Usage:
  python scripts/patch_dataset_for_piper_follower.py \
    --root /media/Data/huggingface/lerobot/lerobot_v3/data_toiletButton_0626 \
    --dry-run   # preview without writing
"""

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ── Mapping ──────────────────────────────────────────────────────────────────
OLD_JOINT_NAMES = [
    "main_joint_1", "main_joint_2", "main_joint_3",
    "main_joint_4", "main_joint_5", "main_joint_6",
    "main_gripper",
]
NEW_JOINT_NAMES = [
    "joint_1.pos", "joint_2.pos", "joint_3.pos",
    "joint_4.pos", "joint_5.pos", "joint_6.pos",
    "gripper.pos",
]
OLD_DIM = 10
NEW_DIM = 7  # 默认值，可通过命令行 --new-dim 覆盖


# ── 1. Patch info.json ──────────────────────────────────────────────────────
def patch_info_json(root: Path, dry_run: bool):
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    print(f"[1/4] Patching {info_path}")

    # robot_type
    old_type = info.get("robot_type")
    info["robot_type"] = "piper_follower"
    print(f"  robot_type: {old_type} → piper_follower")

    # action & observation.state
    for key in ("action", "observation.state"):
        feat = info["features"][key]
        old_shape = feat["shape"]
        old_names = feat["names"]
        feat["shape"] = [NEW_DIM]
        feat["names"] = NEW_JOINT_NAMES
        print(f"  {key}.shape: {old_shape} → [{NEW_DIM}]")
        print(f"  {key}.names: {old_names} → {NEW_JOINT_NAMES}")

    # Add HIL complementary_info fields
    hil_fields = {
        "complementary_info.policy_action": {
            "dtype": "float32",
            "shape": [NEW_DIM],
            "names": NEW_JOINT_NAMES,
        },
        "complementary_info.is_intervention": {
            "dtype": "float32",
            "shape": [1],
            "names": ["is_intervention"],
        },
        "complementary_info.state": {
            "dtype": "float32",
            "shape": [1],
            "names": ["state"],
        },
        "complementary_info.collector_policy_id": {
            "dtype": "string",
            "shape": [1],
            "names": ["collector_policy_id"],
        },
    }
    for name, schema in hil_fields.items():
        if name not in info["features"]:
            info["features"][name] = schema
            print(f"  + added {name}")
        else:
            print(f"  = {name} already exists, skipped")

    if not dry_run:
        shutil.copy2(info_path, str(info_path) + ".bak")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"  ✓ Saved (backup: {info_path}.bak)")
    else:
        print(f"  [dry-run] Would save {info_path}")


# ── 2. Patch stats.json ─────────────────────────────────────────────────────
def patch_stats_json(root: Path, dry_run: bool):
    stats_path = root / "meta" / "stats.json"
    if not stats_path.exists():
        print(f"[2/4] Skipping {stats_path} (not found)")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    print(f"[2/4] Patching {stats_path}")

    for key in ("action", "observation.state"):
        if key not in stats:
            continue
        for stat_name, values in stats[key].items():
            if isinstance(values, list) and len(values) == OLD_DIM:
                stats[key][stat_name] = values[:NEW_DIM]
                print(f"  {key}.{stat_name}: len {OLD_DIM} → {NEW_DIM}")

    if not dry_run:
        shutil.copy2(stats_path, str(stats_path) + ".bak")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Saved")
    else:
        print(f"  [dry-run] Would save {stats_path}")


# ── 3. Patch data parquet ───────────────────────────────────────────────────
def patch_data_parquets(root: Path, dry_run: bool):
    pattern = str(root / "data" / "chunk-*" / "file-*.parquet")
    files = sorted(glob.glob(pattern))
    print(f"[3/4] Patching {len(files)} data parquet file(s)")

    for fpath in files:
        print(f"  Processing {os.path.basename(fpath)} ...")
        df = pd.read_parquet(fpath)

        # Truncate action
        if "action" in df.columns:
            df["action"] = df["action"].apply(
                lambda x: np.array(x[:NEW_DIM], dtype=np.float32)
            )
        # Truncate observation.state
        if "observation.state" in df.columns:
            df["observation.state"] = df["observation.state"].apply(
                lambda x: np.array(x[:NEW_DIM], dtype=np.float32)
            )

        # Add missing complementary_info columns with defaults
        # NOTE: shape=(1,) fields must be stored as scalar in parquet, not array
        hil_defaults = {
            "complementary_info.policy_action": lambda: np.zeros(NEW_DIM, dtype=np.float32),
            "complementary_info.is_intervention": lambda: np.float32(0.0),   # scalar, not array
            "complementary_info.state": lambda: np.float32(0.0),             # scalar, not array
            "complementary_info.collector_policy_id": lambda: "human",
        }
        for col, default_fn in hil_defaults.items():
            if col not in df.columns:
                df[col] = [default_fn() for _ in range(len(df))]
                print(f"    + added {col} (default)")

        if not dry_run:
            bak = fpath + ".bak"
            # Read original, write backup, then overwrite
            original = pd.read_parquet(fpath)
            original.to_parquet(bak)
            df.to_parquet(fpath)
            print(f"    ✓ Saved (backup: {bak})")
        else:
            print(f"    [dry-run] Would save {fpath}")


# ── 4. Patch episodes parquet ───────────────────────────────────────────────
def patch_episodes_parquets(root: Path, dry_run: bool, episode_success: str = "success"):
    pattern = str(root / "meta" / "episodes" / "chunk-*" / "file-*.parquet")
    files = sorted(glob.glob(pattern))
    print(f"[4/4] Patching {len(files)} episodes parquet file(s)")

    stats_cols_to_truncate = [
        c for c_prefix in ("stats/action/", "stats/observation.state/")
        for c in (c_prefix + s for s in ("min", "max", "mean", "std"))
    ]

    for fpath in files:
        print(f"  Processing {os.path.basename(fpath)} ...")
        df = pd.read_parquet(fpath)

        # Truncate per-episode stats from 10→7
        for col in stats_cols_to_truncate:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: np.array(x[:NEW_DIM], dtype=np.float32)
                    if hasattr(x, "__len__") and len(x) == OLD_DIM
                    else x
                )
                print(f"    truncated {col}")

        # Add episode_success column

        df["episode_success"] = episode_success
        print(f"    + added episode_success='{episode_success}'")

        if not dry_run:
            bak = fpath + ".bak"
            original = pd.read_parquet(fpath)
            original.to_parquet(bak)
            df.to_parquet(fpath)
            print(f"    ✓ Saved (backup: {bak})")
        else:
            print(f"    [dry-run] Would save {fpath}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Patch dataset for piper_follower compatibility")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--episode-success", choices=["success", "failure", "ask"],
                        default="ask",
                        help="episode_success 标注值。success/failure 直接写入；"
                             "ask（默认）在终端交互确认。")
    parser.add_argument("--new-dim", type=int, default=7,
                        help="新的 action/observation.state 维度（默认：7）")
    args = parser.parse_args()

    # 更新全局变量
    global NEW_DIM, NEW_JOINT_NAMES
    NEW_DIM = args.new_dim

    # 根据 NEW_DIM 动态调整 joint names
    if NEW_DIM == 7:
        NEW_JOINT_NAMES = [
            "joint_1.pos", "joint_2.pos", "joint_3.pos",
            "joint_4.pos", "joint_5.pos", "joint_6.pos",
            "gripper.pos",
        ]
    else:
        # 如果不是7维，生成通用名称
        NEW_JOINT_NAMES = [f"joint_{i}.pos" for i in range(1, NEW_DIM + 1)]

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory")
        return

    print(f"Dataset root: {root}")
    print(f"Dry run: {args.dry_run}")
    print(f"New dimension: {NEW_DIM}")
    print(f"New joint names: {NEW_JOINT_NAMES}")
    print()

    patch_info_json(root, args.dry_run)
    print()
    patch_stats_json(root, args.dry_run)
    print()
    patch_data_parquets(root, args.dry_run)
    print()

    # 确定 episode_success 值
    episode_success = args.episode_success
    if episode_success == "ask":
        print("=" * 50)
        print(f"数据集路径: {root}")
        while True:
            val = input("请输入本批数据的 episode_success 标注 [success/failure]: ").strip().lower()
            if val in ("success", "failure"):
                episode_success = val
                break
            print("  无效输入，请输入 success 或 failure")
        print("=" * 50)

    patch_episodes_parquets(root, args.dry_run, episode_success)
    print()
    print("Done!")


if __name__ == "__main__":
    main()
