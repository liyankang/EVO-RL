"""视频裁剪组件，使用关键帧缩略图和双滑块完成区间选择。"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


class VideoPlayerWithSlider:
    """视频裁剪组件。"""
    
    def __init__(self, video_files: List[Tuple[str, Path]], total_frames: int, fps: float = 30.0):
        """初始化视频播放器
        
        Args:
            video_files: 视频文件列表，格式为 [(camera_name, video_path), ...]
            total_frames: 总帧数
            fps: 帧率
        """
        self.video_files = video_files
        self.total_frames = total_frames
        self.fps = fps
        
    def render(self, height: int = 300, action_strength: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """渲染关键帧缩略图和双滑块。
        
        Args:
            height: 视频显示高度
            
        Returns:
            (start_frame, end_frame) 选择的裁剪范围
        """
        if not self.video_files:
            st.warning("没有可用的视频文件")
            return 0, 0

        start_frame, end_frame = self._current_range()
        self._render_keyframes(start_frame, end_frame)

        if action_strength is not None:
            self._render_action_curve(action_strength, start_frame, end_frame)

        start_frame, end_frame = self._render_controls()
        
        return start_frame, end_frame

    def _default_range(self) -> Tuple[int, int]:
        """默认保留中间 50% 区间。"""
        if self.total_frames <= 1:
            return 0, 1
        default_start = max(0, self.total_frames // 4)
        default_end = max(default_start + 1, (self.total_frames * 3) // 4)
        return default_start, min(default_end, self.total_frames)

    def _current_range(self) -> Tuple[int, int]:
        """从 session state 读取当前滑块区间。"""
        start_frame, end_frame = st.session_state.get("crop_range_slider", self._default_range())
        start_frame = max(0, min(int(start_frame), max(0, self.total_frames - 1)))
        end_frame = max(start_frame + 1, min(int(end_frame), self.total_frames))
        return start_frame, end_frame
    
    def _read_frame(self, video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
        """读取指定视频帧并转换为 RGB。"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        safe_idx = max(0, min(frame_idx, self.total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _render_keyframes(self, start_frame: int, end_frame: int):
        """显示两个相机在当前裁剪区间的起止关键帧。"""
        preview_files = self.video_files[:2]
        cols = st.columns(2)
        end_preview_frame = max(start_frame, end_frame - 1)

        if not preview_files:
            with cols[0]:
                st.info("没有可用于缩略图预览的视频")
            return

        for idx, (camera_name, video_path) in enumerate(preview_files):
            start_image = self._read_frame(video_path, start_frame)
            end_image = self._read_frame(video_path, end_preview_frame)

            with cols[idx]:
                st.markdown(f"**{camera_name}**")
                frame_cols = st.columns(2)
                with frame_cols[0]:
                    st.caption(f"起始帧 {start_frame}")
                    if start_image is not None:
                        st.image(start_image, width="stretch")
                    else:
                        st.warning("无法读取")
                with frame_cols[1]:
                    st.caption(f"结束帧 {end_preview_frame}")
                    if end_image is not None:
                        st.image(end_image, width="stretch")
                    else:
                        st.warning("无法读取")

        if len(preview_files) < 2:
            with cols[1]:
                st.info("第二路相机视频不存在")

    def _render_action_curve(self, action_strength: np.ndarray, start_frame: int, end_frame: int):
        """在滑块上方显示动作强度曲线，并高亮保留区间。"""
        import plotly.graph_objects as go

        if len(action_strength) == 0:
            return

        safe_end = max(start_frame + 1, min(end_frame, len(action_strength)))
        x_values = np.arange(len(action_strength))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values,
            y=action_strength,
            mode="lines",
            line=dict(color="#d97706", width=1.4),
            fill="tozeroy",
            fillcolor="rgba(217, 119, 6, 0.10)",
            hovertemplate="帧 %{x}<br>动作强度 %{y:.4f}<extra></extra>"
        ))
        fig.add_vrect(
            x0=start_frame,
            x1=safe_end,
            fillcolor="rgba(14, 165, 233, 0.18)",
            line_width=0
        )
        fig.add_vline(x=start_frame, line_color="#0ea5e9", line_width=2)
        fig.add_vline(x=safe_end, line_color="#0ea5e9", line_width=2)
        fig.update_layout(
            height=170,
            margin=dict(l=0, r=0, t=8, b=0),
            showlegend=False,
            xaxis_title="帧",
            yaxis_title="动作强度",
            hovermode="x unified"
        )
        st.plotly_chart(fig, width="stretch")
    
    def _render_controls(self) -> Tuple[int, int]:
        """渲染一条双滑块轨道，保留中间部分。"""
        state_key = "crop_range_slider"
        max_frame = max(1, self.total_frames)
        default_start, default_end = self._default_range()
        if state_key not in st.session_state:
            st.session_state[state_key] = (default_start, default_end)
        else:
            start_frame, end_frame = st.session_state[state_key]
            start_frame = max(0, min(int(start_frame), max_frame - 1))
            end_frame = max(start_frame + 1, min(int(end_frame), max_frame))
            st.session_state[state_key] = (start_frame, end_frame)

        crop_range = st.slider(
            "拖动两个滑块，保留中间这段视频",
            min_value=0,
            max_value=max_frame,
            step=1,
            key=state_key,
            help="左侧滑块表示起始位置，右侧滑块表示结束位置，中间部分会被保留"
        )

        start_frame, end_frame = crop_range
        return start_frame, end_frame


def create_video_player_with_slider(video_files: List[Tuple[str, Path]], 
                                    total_frames: int, 
                                    fps: float = 30.0) -> VideoPlayerWithSlider:
    """创建视频播放器实例
    
    Args:
        video_files: 视频文件列表
        total_frames: 总帧数
        fps: 帧率
        
    Returns:
        VideoPlayerWithSlider 实例
    """
    return VideoPlayerWithSlider(video_files, total_frames, fps)
