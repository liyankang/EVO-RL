"""视频处理工具模块，用于视频帧窗口处理。"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def write_video_frame_window(video_path: Path, temp_output: Path, start_frame: int, end_frame: int, fps: float, width: int, height: int) -> int:
    """使用更兼容的H.264编码器替代mp4v，解决浏览器播放问题。
    
    Args:
        video_path: 原始视频文件路径
        temp_output: 临时输出文件路径
        start_frame: 开始帧索引
        end_frame: 结束帧索引（不包含）
        fps: 帧率
        width: 视频宽度
        height: 视频高度
        
    Returns:
        实际写入的帧数
    """
    # 尝试使用av (PyAV) 库进行更高效的视频处理
    try:
        import av
        from fractions import Fraction
        
        # 使用PyAV进行流式裁剪，更加高效可靠
        input_container = av.open(str(video_path))
        output_container = av.open(str(temp_output), mode='w')
        
        # 获取视频流
        input_stream = input_container.streams.video[0]
        
        # 将帧率转换为分数格式，避免PyAV的float类型错误
        if isinstance(fps, float):
            # 尝试将浮点数转换为最接近的简单分数
            fps_fraction = Fraction(fps).limit_denominator(1000)
        else:
            fps_fraction = Fraction(int(fps), 1)
        
        # 配置输出视频流 - 使用H.264编码器，兼容性更好
        output_stream = output_container.add_stream('libx264', rate=fps_fraction)
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = 'yuv420p'  # 更好的浏览器兼容性
        
        # 配置H.264编码器参数，优化质量和兼容性
        output_stream.options = {
            'crf': '23',  # 质量控制 (18-28之间，越小质量越好)
            'preset': 'fast',  # 编码速度
            'movflags': '+faststart',  # 启用快速启动，支持流式播放
        }
        
        written = 0
        frame_idx = 0
        for packet in input_container.demux(input_stream):
            for frame in packet.decode():
                if frame_idx < start_frame:
                    frame_idx += 1
                    continue
                if frame_idx >= end_frame:
                    break
                
                # 将帧编码并写入输出
                for packet_out in output_stream.encode(frame):
                    output_container.mux(packet_out)
                written += 1
                frame_idx += 1
            else:
                continue
            break
        
        # 写入剩余数据并关闭文件
        for packet_out in output_stream.encode():
            output_container.mux(packet_out)
        
        input_container.close()
        output_container.close()
        
        return written
        
    except ImportError:
        # 如果av不可用，回退到使用OpenCV但使用兼容性更好的编码器
        logger.warning("PyAV not available, falling back to OpenCV with H.264 encoding")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {video_path}")

        # 尝试使用H.264编码器，兼容性更好
        # 优先尝试常见的H.264 FOURCC代码
        fourcc_codes = [
            ('H', '2', '6', '4'),      # H.264
            ('a', 'v', 'c', '1'),      # H.264 AVC
            ('X', '2', '6', '4'),      # H.264 alternative
            ('m', 'p', '4', 'v'),      # MPEG-4 Visual (fallback)
        ]
        
        writer = None
        for fourcc_code in fourcc_codes:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                test_writer = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
                if test_writer.isOpened():
                    test_writer.release()
                    writer = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
                    logger.info(f"Successfully initialized video writer with FOURCC: {''.join(fourcc_code)}")
                    break
            except Exception:
                continue
        
        if writer is None or not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to create temporary video with any compatible codec: {temp_output}")

        written = 0
        frame_idx = 0
        try:
            while frame_idx < end_frame:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx >= start_frame:
                    writer.write(frame)
                    written += 1
                frame_idx += 1
        finally:
            writer.release()
            cap.release()

        return written


def get_truncation_window(original_length: int, timestamps: Optional[np.ndarray], 
                          start_time: Optional[float], end_time: Optional[float],
                          keep_start_frame: Optional[int], keep_end_frame: Optional[int],
                          truncate_frame: Optional[int], discard_mode: str) -> tuple:
    """计算截断窗口的起始和结束帧索引。
    
    Args:
        original_length: 原始长度
        timestamps: 时间戳数组（用于基于时间的截断）
        start_time: 开始时间
        end_time: 结束时间
        keep_start_frame: 保留的起始帧
        keep_end_frame: 保留的结束帧
        truncate_frame: 截断的帧数
        discard_mode: 丢弃模式 ('after' 或 'before')
        
    Returns:
        (start_frame, end_frame) 元组
    """
    # 基于时间戳的截断
    if start_time is not None or end_time is not None:
        if timestamps is None:
            raise ValueError("Timestamps must be provided for time-based truncation")
        
        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time must be provided together")
        
        # 确保时间戳范围有效
        start_time = max(0.0, min(start_time, timestamps[-1]))
        end_time = max(0.0, min(end_time, timestamps[-1]))
        
        if end_time < start_time:
            start_time, end_time = end_time, start_time
        
        # 找到时间戳范围内的帧索引
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        frame_indices = np.where(mask)[0]
        
        if len(frame_indices) == 0:
            raise ValueError(f"No frames found in time range [{start_time}, {end_time}]")
        
        start_frame = int(frame_indices[0])
        end_frame = int(frame_indices[-1]) + 1  # 独区间
        
        return start_frame, end_frame
    
    # 基于帧数的截断（原有逻辑）
    if keep_start_frame is not None or keep_end_frame is not None:
        if keep_start_frame is None or keep_end_frame is None:
            raise ValueError("keep_start_frame and keep_end_frame must be provided together")

        start_frame = max(0, min(keep_start_frame, original_length - 1))
        end_frame_inclusive = max(0, min(keep_end_frame, original_length - 1))
        if end_frame_inclusive < start_frame:
            start_frame, end_frame_inclusive = end_frame_inclusive, start_frame
        return start_frame, end_frame_inclusive + 1

    if truncate_frame is None or truncate_frame <= 0:
        raise ValueError("truncate_frame must be greater than 0")

    keep_frames = min(truncate_frame, original_length)
    if discard_mode == "after":
        start_frame = 0
    elif discard_mode == "before":
        start_frame = original_length - keep_frames
    else:
        raise ValueError(f"Unsupported discard_mode: {discard_mode}")

    return start_frame, start_frame + keep_frames