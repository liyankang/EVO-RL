#!/usr/bin/env python3
"""
修复：将 ep 22~49 重编号为 ep 21~48，同步更新 parquet 内字段和 meta 文件。
当前状态：data/meta 均缺 ep 21，ep 22~49 需要前移一位。
"""

import glob
import json
import os
import shutil
from datetime import datetime

import pandas as pd

ROOT = "/media/Data/huggingface/lerobot/lerobot_v3/data_toiletButton_0701_f"
DATA_DIR = f"{ROOT}/data/chunk-000"
VIDEO_DIR = f"{ROOT}/videos/chunk-000"
META_DIR = f"{ROOT}/meta"

OLD_START = 22
OLD_END = 49
OFFSET = 1  # new = old - 1


def backup():
    bak = f"{ROOT}/meta_bak_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copytree(META_DIR, bak)
    print(f"[备份] meta/ → {bak}")


def renumber_files():
    print("\n===== 重命名文件 =====")
    # 从小到大重命名（ep21 空位，不会冲突）
    for old_ep in range(OLD_START, OLD_END + 1):
        new_ep = old_ep - OFFSET

        # parquet
        old_pq = f"{DATA_DIR}/episode_{old_ep:06d}.parquet"
        new_pq = f"{DATA_DIR}/episode_{new_ep:06d}.parquet"
        if os.path.exists(old_pq):
            os.rename(old_pq, new_pq)
            print(f"  parquet: {old_ep:06d} → {new_ep:06d}")
        else:
            print(f"  [警告] parquet 不存在: episode_{old_ep:06d}.parquet")

        # 视频（两个相机）
        for cam in ["observation.images.primary", "observation.images.wrist"]:
            old_vid = f"{VIDEO_DIR}/{cam}/episode_{old_ep:06d}.mp4"
            new_vid = f"{VIDEO_DIR}/{cam}/episode_{new_ep:06d}.mp4"
            if os.path.exists(old_vid):
                os.rename(old_vid, new_vid)
                print(f"  video/{cam}: {old_ep:06d} → {new_ep:06d}")
            else:
                print(f"  [警告] 视频不存在: {cam}/episode_{old_ep:06d}.mp4")


def patch_parquets():
    print("\n===== 修正 parquet 内字段 =====")
    all_pq = sorted(glob.glob(f"{DATA_DIR}/episode_*.parquet"))
    global_index = 0
    for pq_path in all_pq:
        df = pd.read_parquet(pq_path)
        ep = int(os.path.basename(pq_path).replace("episode_", "").replace(".parquet", ""))
        n = len(df)
        df["episode_index"] = ep
        df["frame_index"] = range(n)
        df["index"] = range(global_index, global_index + n)
        df.to_parquet(pq_path, index=False)
        global_index += n
    print(f"  共处理 {len(all_pq)} 个 parquet，总帧数 {global_index}")


def update_meta():
    print("\n===== 更新 meta =====")

    # episodes.jsonl
    ep_path = f"{META_DIR}/episodes.jsonl"
    with open(ep_path) as f:
        eps = [json.loads(l) for l in f if l.strip()]

    new_eps = []
    for e in eps:
        idx = e["episode_index"]
        if idx >= OLD_START:
            e["episode_index"] = idx - OFFSET
        new_eps.append(e)
    new_eps.sort(key=lambda x: x["episode_index"])

    # 刷新 length
    for e in new_eps:
        pq_path = f"{DATA_DIR}/episode_{e['episode_index']:06d}.parquet"
        df = pd.read_parquet(pq_path)
        e["length"] = len(df)

    with open(ep_path, "w") as f:
        for e in new_eps:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  episodes.jsonl: {len(new_eps)} 条")

    # episodes_stats.jsonl
    stats_path = f"{META_DIR}/episodes_stats.jsonl"
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = [json.loads(l) for l in f if l.strip()]
        for s in stats:
            if s["episode_index"] >= OLD_START:
                s["episode_index"] -= OFFSET
        stats.sort(key=lambda x: x["episode_index"])
        with open(stats_path, "w") as f:
            for s in stats:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  episodes_stats.jsonl: {len(stats)} 条")

    # info.json
    info_path = f"{META_DIR}/info.json"
    with open(info_path) as f:
        info = json.load(f)
    all_pq = glob.glob(f"{DATA_DIR}/episode_*.parquet")
    n_eps = len(all_pq)
    total_frames = sum(len(pd.read_parquet(p)) for p in all_pq)
    total_vids = len(glob.glob(f"{VIDEO_DIR}/**/episode_*.mp4", recursive=True))
    info["total_episodes"] = n_eps
    info["total_frames"] = total_frames
    info["total_videos"] = total_vids
    info["splits"]["train"] = f"0:{n_eps}"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  info.json: total_episodes={n_eps}, total_frames={total_frames}, total_videos={total_vids}")


def validate():
    print("\n===== 校验 =====")
    all_pq = sorted(glob.glob(f"{DATA_DIR}/episode_*.parquet"))
    eps_data = {int(os.path.basename(p).replace("episode_", "").replace(".parquet", "")) for p in all_pq}
    expected = set(range(len(all_pq)))
    ok = True
    if eps_data != expected:
        print(f"  [错误] parquet 编号不连续: 实际={sorted(eps_data)}, 期望={sorted(expected)}")
        ok = False
    else:
        print(f"  parquet 编号 0~{max(expected)}: ✓")

    for cam in ["observation.images.primary", "observation.images.wrist"]:
        vids = sorted(glob.glob(f"{VIDEO_DIR}/{cam}/episode_*.mp4"))
        vid_eps = {int(os.path.basename(v).replace("episode_", "").replace(".mp4", "")) for v in vids}
        if vid_eps != expected:
            print(f"  [错误] {cam} 视频编号不一致")
            ok = False
        else:
            print(f"  {cam} 视频 0~{max(expected)}: ✓")

    # 全局 index 连续性
    dfs = [pd.read_parquet(p) for p in all_pq]
    combined = pd.concat(dfs, ignore_index=True)
    if list(combined["index"]) != list(range(len(combined))):
        print("  [错误] 全局 index 不连续")
        ok = False
    else:
        print(f"  全局 index 0~{len(combined)-1}: ✓")

    print("\n✅ 全部通过!" if ok else "\n❌ 存在错误，请检查")


if __name__ == "__main__":
    backup()
    renumber_files()
    patch_parquets()
    update_meta()
    validate()
