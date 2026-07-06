#!/usr/bin/env python3
"""
可视化 indicator/value/advantage 叠加到视频帧上。
不依赖 LeRobotDataset，直接读 parquet + 视频文件。

用法:
    python scripts/visualize_indicator.py \
        --dataset /media/Data/huggingface/lerobot/lerobot_v3/data_toiletButton_0703 \
        --episodes 0,20,40,60,80,100 \
        --smooth-window 5 \
        --output-dir ./output/indicator_viz
"""

import argparse
import glob
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import savgol_filter
from tqdm import tqdm


# ── 字段名 ────────────────────────────────────────────────────────────────────
VALUE_FIELD     = "complementary_info.value_push_button"
ADVANTAGE_FIELD = "complementary_info.advantage_push_button"
INDICATOR_FIELD = "complementary_info.acp_indicator_push_button"
PRIMARY_CAM     = "observation.images.primary"


def smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(arr) < 5:
        return arr.copy()
    w = min(window, len(arr))
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return arr.copy()
    return savgol_filter(arr, window_length=w, polyorder=3).astype(arr.dtype)


def load_font(size: int = 18):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_overlay(img: Image.Image, step: int, total: int,
                 values: np.ndarray, advantages: np.ndarray, indicators: np.ndarray,
                 font) -> Image.Image:
    W, H = img.size
    overlay_w = max(180, W // 3)
    overlay_h = H

    ov = Image.new("RGBA", (overlay_w, overlay_h), (0, 0, 0, 160))
    draw = ImageDraw.Draw(ov)

    split = int(overlay_h * 0.55)

    def draw_curve(arr, color, y0, y1, label):
        mn, mx = arr.min(), arr.max()
        rng = mx - mn if mx != mn else 1.0
        pts = []
        for i, v in enumerate(arr):
            x = int(i / max(len(arr) - 1, 1) * (overlay_w - 1))
            y = y1 - int((v - mn) / rng * (y1 - y0 - 4)) - 2
            pts.append((x, y))
        if len(pts) >= 2:
            draw.line(pts, fill=color + (200,), width=2)
        # current position dot
        cx, cy = pts[step]
        draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill=(255, 255, 0, 255))
        # current value label
        cur_val = arr[step]
        draw.text((cx + 6, cy - 10), f"{cur_val:.3f}", font=font, fill=(255, 255, 0, 255))
        # axis label
        draw.text((4, y0 + 2), label, font=font, fill=color + (230,))

    draw_curve(values,     (100, 200, 255), 4,       split - 4, "Value")
    draw_curve(advantages, (100, 255, 150), split+4, overlay_h - 20, "Adv")

    # ACP indicator
    is_pos = bool(indicators[step] >= 1)
    ind_color = (0, 255, 80, 255) if is_pos else (255, 60, 60, 255)
    ind_label = "ACP+" if is_pos else "ACP-"
    draw.ellipse([overlay_w//2 - 10, overlay_h - 18, overlay_w//2 + 10, overlay_h - 2],
                 fill=ind_color)
    draw.text((overlay_w//2 + 14, overlay_h - 18), ind_label, font=font, fill=ind_color)

    # frame counter
    draw.text((4, overlay_h - 18), f"{step+1}/{total}", font=font, fill=(200, 200, 200, 200))

    out = img.convert("RGBA")
    out.paste(ov, (0, 0), ov)
    return out.convert("RGB")


def get_video_info(dataset_root: Path, episode_idx: int, cam_key: str, eps_df: pd.DataFrame):
    """返回 (video_path, from_ts, to_ts)，时间戳是视频文件内的绝对时间。"""
    row = eps_df[eps_df["episode_index"] == episode_idx]
    if row.empty:
        raise FileNotFoundError(f"episodes parquet 中找不到 episode {episode_idx}")
    r = row.iloc[0]
    chunk = int(r[f"videos/{cam_key}/chunk_index"])
    fidx  = int(r[f"videos/{cam_key}/file_index"])
    from_ts = float(r[f"videos/{cam_key}/from_timestamp"])
    to_ts   = float(r[f"videos/{cam_key}/to_timestamp"])
    path = dataset_root / "videos" / cam_key / f"chunk-{chunk:03d}" / f"file-{fidx:03d}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"视频文件不存在: {path}")
    return path, from_ts, to_ts


def decode_frames_at_timestamps(video_path: Path, abs_timestamps: np.ndarray) -> list[Image.Image]:
    """从视频中按绝对时间戳提取帧（取最近帧）。"""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    tb = float(stream.time_base)

    all_frames: list[tuple[float, Image.Image]] = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            t = float(frame.pts) * tb
            all_frames.append((t, frame.to_image()))
    container.close()

    if not all_frames:
        raise ValueError(f"视频无帧: {video_path}")

    frame_ts = np.array([t for t, _ in all_frames])
    result = []
    for ts in abs_timestamps:
        idx = int(np.argmin(np.abs(frame_ts - ts)))
        result.append(all_frames[idx][1])
    return result


def parse_episodes(arg: str, total: int) -> list[int]:
    if arg.strip().lower() == "all":
        return list(range(total))
    result = []
    for tok in arg.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-", 1)
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(tok))
    return sorted(set(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="数据集根目录")
    parser.add_argument("--episodes", default="0,20,40,60,80,100")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--output-dir", default="./output/indicator_viz")
    parser.add_argument("--cam-key", default=PRIMARY_CAM)
    parser.add_argument("--vcodec", default="libx264")
    args = parser.parse_args()

    root = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载所有帧数据
    parquets = sorted((root / "data").rglob("*.parquet"))
    frames_df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    print(f"总帧数: {len(frames_df)}  列: {list(frames_df.columns)}")

    # 加载 episodes parquet（含视频文件位置）
    ep_parquets = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    eps_df = pd.concat([pd.read_parquet(p) for p in ep_parquets], ignore_index=True)

    total_episodes = int(frames_df["episode_index"].max()) + 1
    episodes = parse_episodes(args.episodes, total_episodes)
    episodes = [e for e in episodes if e < total_episodes]
    print(f"将处理 episodes: {episodes}")

    font = load_font(16)
    written = []

    for ep in tqdm(episodes, desc="episodes"):
        ep_df = frames_df[frames_df["episode_index"] == ep].sort_values("frame_index").reset_index(drop=True)
        if len(ep_df) == 0:
            print(f"  ep {ep}: 无数据，跳过")
            continue

        values     = smooth(np.array(ep_df[VALUE_FIELD].tolist(),     dtype=np.float32).ravel(), args.smooth_window)
        advantages = smooth(np.array(ep_df[ADVANTAGE_FIELD].tolist(), dtype=np.float32).ravel(), args.smooth_window)
        indicators = np.array(ep_df[INDICATOR_FIELD].tolist(), dtype=np.int64).ravel()
        rel_timestamps = ep_df["timestamp"].values.astype(np.float64)

        try:
            video_path, from_ts, to_ts = get_video_info(root, ep, args.cam_key, eps_df)
        except FileNotFoundError as e:
            print(f"  ep {ep}: {e}")
            continue

        # 帧的绝对视频时间 = episode 起始时间 + 帧相对时间
        abs_timestamps = from_ts + rel_timestamps
        abs_timestamps = np.minimum(abs_timestamps, to_ts)

        try:
            pil_frames = decode_frames_at_timestamps(video_path, abs_timestamps)
        except Exception as e:
            print(f"  ep {ep}: 视频解码失败: {e}")
            continue

        if len(pil_frames) != len(ep_df):
            # 对齐长度
            n = min(len(pil_frames), len(values))
            pil_frames = pil_frames[:n]
            values = values[:n]; advantages = advantages[:n]; indicators = indicators[:n]

        # 合成叠加帧
        overlaid = []
        for i, frame in enumerate(pil_frames):
            overlaid.append(draw_overlay(frame, i, len(pil_frames), values, advantages, indicators, font))

        # 编码视频
        out_path = output_dir / f"episode_{ep:04d}.mp4"
        W, H = overlaid[0].size
        container = av.open(str(out_path), mode="w")
        stream = container.add_stream(args.vcodec, rate=30)
        stream.width = W; stream.height = H
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "fast"}
        for img in overlaid:
            av_frame = av.VideoFrame.from_image(img)
            for pkt in stream.encode(av_frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()

        written.append(out_path)
        pos_ratio = indicators.mean() * 100
        print(f"  ep {ep}: {len(overlaid)} 帧  ACP+={pos_ratio:.1f}%  → {out_path.name}")

    print(f"\n完成，共生成 {len(written)} 个视频 → {output_dir}")


if __name__ == "__main__":
    main()
