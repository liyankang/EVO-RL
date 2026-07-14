"""轨迹时间轴滑块组件"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TrajectorySlider:
    """轨迹时间轴滑块组件"""
    
    def __init__(self, dataset_path: str, episode_index: int):
        """初始化轨迹滑块
        
        Args:
            dataset_path: 数据集路径
            episode_index: episode 索引
        """
        self.dataset_path = Path(dataset_path)
        self.episode_index = episode_index
        self.trajectory_data = None
        self.timestamps = None
        self.length = 0
        self._load_trajectory_data()
    
    def _load_trajectory_data(self):
        """加载轨迹数据"""
        try:
            data_dir = self.dataset_path / "data" / "chunk-000"
            trajectory_file = data_dir / f"episode_{self.episode_index:06d}.parquet"
            
            if not trajectory_file.exists():
                logger.error(f"轨迹文件不存在: {trajectory_file}")
                return
            
            df = pd.read_parquet(trajectory_file)
            self.trajectory_data = df
            self.length = len(df)
            
            # 提取时间戳
            if 'timestamp' in df.columns:
                self.timestamps = df['timestamp'].values
            else:
                # 如果没有时间戳，创建默认时间戳（假设30fps）
                self.timestamps = np.arange(self.length) / 30.0
                
        except Exception as e:
            logger.error(f"加载轨迹数据失败: {e}")
    
    def get_trajectory_info(self) -> Dict[str, Any]:
        """获取轨迹信息
        
        Returns:
            轨迹信息字典
        """
        if self.trajectory_data is None:
            return {}
        
        return {
            'length': self.length,
            'has_timestamp': self.timestamps is not None,
            'timestamps': self.timestamps,  # 添加时间戳数组
            'start_time': float(self.timestamps[0]) if self.timestamps is not None else None,
            'end_time': float(self.timestamps[-1]) if self.timestamps is not None else None,
            'duration': float(self.timestamps[-1] - self.timestamps[0]) if self.timestamps is not None else None
        }
    
    def render_time_slider(self, mode: str = "time") -> Tuple[Optional[float], Optional[float]]:
        """渲染时间轴滑块
        
        Args:
            mode: 滑块模式 ('time' 或 'frame')
            
        Returns:
            (start_value, end_value) 元组
        """
        if self.trajectory_data is None:
            st.error("无法加载轨迹数据")
            return None, None
        
        st.subheader(f"🎬 Episode {self.episode_index} 时间轴")
        
        # 显示轨迹信息
        trajectory_info = self.get_trajectory_info()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总帧数", trajectory_info['length'])
        with col2:
            if trajectory_info['duration']:
                st.metric("时长", f"{trajectory_info['duration']:.2f}s")
            else:
                st.metric("时长", "N/A")
        with col3:
            if trajectory_info['has_timestamp']:
                st.metric("时间戳", "✅ 有")
            else:
                st.metric("时间戳", "⚠️ 无")
        
        # 根据模式渲染滑块
        if mode == "time" and self.timestamps is not None:
            start_time, end_time = self._render_time_mode()
            return start_time, end_time
        else:
            start_frame, end_frame = self._render_frame_mode()
            # 将帧数转换为时间（如果有时间戳）
            if self.timestamps is not None:
                start_time = self.timestamps[start_frame] if start_frame < len(self.timestamps) else None
                end_time = self.timestamps[end_frame] if end_frame <= len(self.timestamps) else None
                return start_time, end_time
            else:
                # 如果没有时间戳，返回帧数除以帧率（假设30fps）
                start_time = start_frame / 30.0 if start_frame is not None else None
                end_time = end_frame / 30.0 if end_frame is not None else None
                return start_time, end_time
    
    def _render_time_mode(self) -> Tuple[float, float]:
        """渲染时间模式滑块
        
        Returns:
            (start_time, end_time) 元组
        """
        if self.timestamps is None:
            st.error("轨迹数据中没有时间戳信息")
            return 0.0, 0.0
        
        min_time = float(self.timestamps[0])
        max_time = float(self.timestamps[-1])
        default_duration = (max_time - min_time) * 0.5  # 默认选择中间50%
        
        # 时间范围滑块
        time_range = st.slider(
            "选择时间范围",
            min_value=min_time,
            max_value=max_time,
            value=(min_time, min_time + default_duration),
            step=0.01,
            format="%.2f 秒"
        )
        
        start_time, end_time = time_range
        
        # 显示选择的信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("开始时间", f"{start_time:.2f}s")
        with col2:
            st.metric("结束时间", f"{end_time:.2f}s")
        with col3:
            st.metric("选择时长", f"{end_time - start_time:.2f}s")
        
        # 计算对应的帧数范围
        start_frame = np.searchsorted(self.timestamps, start_time)
        end_frame = np.searchsorted(self.timestamps, end_time)
        
        st.info(f"对应帧数范围: {start_frame} - {end_frame} (共 {end_frame - start_frame} 帧)")
        
        return start_time, end_time
    
    def _render_frame_mode(self) -> Tuple[int, int]:
        """渲染帧数模式滑块
        
        Returns:
            (start_frame, end_frame) 元组
        """
        if self.length == 0:
            st.error("轨迹数据为空")
            return 0, 0
        
        default_start = 0
        default_end = max(1, self.length // 2)  # 默认选择前半部分
        
        # 帧数范围滑块
        frame_range = st.slider(
            "选择帧数范围",
            min_value=0,
            max_value=self.length,
            value=(default_start, default_end),
            step=1
        )
        
        start_frame, end_frame = frame_range
        
        # 显示选择的信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("开始帧", start_frame)
        with col2:
            st.metric("结束帧", end_frame)
        with col3:
            st.metric("选择帧数", end_frame - start_frame)
        
        # 如果有时间戳，显示对应的时间范围
        if self.timestamps is not None:
            start_time = self.timestamps[start_frame] if start_frame < len(self.timestamps) else 0
            end_time = self.timestamps[end_frame] if end_frame <= len(self.timestamps) else self.timestamps[-1]
            st.info(f"对应时间范围: {start_time:.2f}s - {end_time:.2f}s")
        else:
            # 假设30fps
            fps = 30.0
            start_time = start_frame / fps
            end_time = end_frame / fps
            st.info(f"对应时间范围 (假设30fps): {start_time:.2f}s - {end_time:.2f}s")
        
        return start_frame, end_frame
    
    def render_preview(self, start_time: float, end_time: float):
        """渲染裁剪预览
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
        """
        if self.timestamps is None:
            st.warning("没有时间戳数据，无法显示预览")
            return
        
        st.subheader("👁️ 裁剪预览")
        
        # 找到对应的帧数
        start_frame = np.searchsorted(self.timestamps, start_time)
        end_frame = np.searchsorted(self.timestamps, end_time)
        
        # 显示预览统计
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("原始帧数", self.length)
        with col2:
            st.metric("裁剪后帧数", end_frame - start_frame)
        with col3:
            ratio = (end_frame - start_frame) / self.length * 100 if self.length > 0 else 0
            st.metric("保留比例", f"{ratio:.1f}%")
        with col4:
            st.metric("删除帧数", self.length - (end_frame - start_frame))
        
        # 显示时间轴可视化
        st.write("**时间轴可视化:**")
        
        # 创建时间轴图表
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # 添加原始时间轴
        fig.add_trace(go.Scatter(
            x=self.timestamps,
            y=np.ones(self.length),
            mode='lines',
            name='原始轨迹',
            line=dict(color='lightgray', width=20),
            hoverinfo='none'
        ))
        
        # 添加保留的时间轴
        mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
        if mask.any():
            fig.add_trace(go.Scatter(
                x=self.timestamps[mask],
                y=np.ones(mask.sum()),
                mode='lines',
                name='保留部分',
                line=dict(color='green', width=20),
                hoverinfo='none'
            ))
        
        # 标记开始和结束点
        fig.add_trace(go.Scatter(
            x=[start_time, end_time],
            y=[1, 1],
            mode='markers',
            name='裁剪点',
            marker=dict(color='red', size=15),
            hovertemplate='时间: %{x:.2f}s<extra></extra>'
        ))
        
        fig.update_layout(
            title='轨迹时间轴裁剪预览',
            xaxis_title='时间 (秒)',
            yaxis_title='',
            yaxis=dict(showticklabels=False),
            hovermode='x unified',
            height=200,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, width='stretch')
    
    def render_action_preview(self, start_time: float, end_time: float):
        """渲染动作数据预览
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
        """
        if self.trajectory_data is None or 'action' not in self.trajectory_data.columns:
            st.warning("轨迹数据中没有动作信息")
            return
        
        st.subheader("🎯 动作数据预览")
        
        # 提取动作数据
        if self.timestamps is not None:
            mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
            preview_data = self.trajectory_data[mask].copy()
            preview_timestamps = self.timestamps[mask]
        else:
            # 如果没有时间戳，显示全部数据
            preview_data = self.trajectory_data.copy()
            preview_timestamps = np.arange(len(preview_data)) / 30.0
        
        if len(preview_data) == 0:
            st.warning("选择的时间范围内没有数据")
            return
        
        # 提取动作数组
        actions = np.stack(preview_data['action'].values)
        
        # 绘制动作曲线
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # 为每个动作维度绘制线条
        for i in range(actions.shape[1]):
            fig.add_trace(go.Scatter(
                x=preview_timestamps,
                y=actions[:, i],
                mode='lines',
                name=f'动作维度 {i}',
                line=dict(width=1)
            ))
        
        fig.update_layout(
            title=f'动作数据预览 ({len(actions)} 帧)',
            xaxis_title='时间 (秒)',
            yaxis_title='动作值',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # 显示统计信息
        st.write("**动作统计:**")
        action_stats = pd.DataFrame({
            '维度': [f'维度 {i}' for i in range(actions.shape[1])],
            '最小值': actions.min(axis=0),
            '最大值': actions.max(axis=0),
            '平均值': actions.mean(axis=0),
            '标准差': actions.std(axis=0)
        })
        
        st.dataframe(action_stats, width='stretch')


def create_trajectory_slider(dataset_path: str, episode_index: int) -> TrajectorySlider:
    """创建轨迹滑块实例
    
    Args:
        dataset_path: 数据集路径
        episode_index: episode 索引
        
    Returns:
        TrajectorySlider 实例
    """
    return TrajectorySlider(dataset_path, episode_index)