"""数据集信息显示组件"""

import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)


class DatasetInfoDisplay:
    """数据集信息显示组件"""
    
    @staticmethod
    def display_overview(dataset_info: Dict[str, Any]) -> None:
        """显示数据集概览信息
        
        Args:
            dataset_info: 数据集信息字典
        """
        if not dataset_info:
            st.warning("没有数据集信息可显示")
            return
        
        st.subheader("📊 数据集概览")
        
        # 基本统计信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Episode 数量", dataset_info.get('total_episodes', 0))
        with col2:
            st.metric("总帧数", dataset_info.get('total_frames', 0))
        with col3:
            st.metric("帧率", f"{dataset_info.get('fps', 30)} fps")
        with col4:
            robot_type = dataset_info.get('robot_type', 'unknown')
            st.metric("机器人类型", robot_type)
        
        # 详细信息
        with st.expander("📋 详细信息"):
            if dataset_info.get('cameras'):
                st.write(f"📷 **相机列表:** {', '.join(dataset_info['cameras'])}")
            
            if dataset_info.get('splits'):
                st.write(f"🔄 **数据划分:** {dataset_info['splits']}")
            
            if dataset_info.get('features'):
                st.write(f"🎯 **特征:**")
                for feature_name, feature_info in dataset_info['features'].items():
                    dtype = feature_info.get('dtype', 'unknown')
                    shape = feature_info.get('shape', 'unknown')
                    st.write(f"  - {feature_name}: {dtype} {shape}")
    
    @staticmethod
    def display_episodes_table(episodes_list: list, show_videos: bool = True) -> None:
        """显示 episode 列表表格
        
        Args:
            episodes_list: episode 列表
            show_videos: 是否显示视频信息
        """
        if not episodes_list:
            st.warning("没有 episode 数据可显示")
            return
        
        st.subheader("🎬 Episode 列表")
        
        # 准备表格数据
        table_data = []
        for ep in episodes_list:
            row = {
                "Episode": ep['episode_index'],
                "帧数": ep['length'],
                "起始时间戳": f"{ep['start_timestamp']:.2f}s" if ep.get('start_timestamp') else "N/A",
                "结束时间戳": f"{ep['end_timestamp']:.2f}s" if ep.get('end_timestamp') else "N/A",
            }
            
            if show_videos:
                row["主相机视频"] = "✅" if ep.get('primary_video_exists') else "❌"
                row["腕部相机视频"] = "✅" if ep.get('wrist_video_exists') else "❌"
            
            table_data.append(row)
        
        # 显示表格
        st.dataframe(table_data, width='stretch')
        
        # 统计信息
        total_frames = sum(ep['length'] for ep in episodes_list)
        avg_length = total_frames / len(episodes_list) if episodes_list else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("总帧数", total_frames)
        with col2:
            st.metric("平均帧数", f"{avg_length:.1f}")
    
    @staticmethod
    def display_episode_detail(episode_data: Dict[str, Any]) -> None:
        """显示单个 episode 的详细信息
        
        Args:
            episode_data: episode 数据字典
        """
        if not episode_data:
            st.warning("没有 episode 信息可显示")
            return
        
        st.subheader(f"📹 Episode {episode_data['episode_index']} 详情")
        
        # 基本信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("帧数", episode_data['length'])
        with col2:
            duration = episode_data.get('end_timestamp', 0) - episode_data.get('start_timestamp', 0)
            st.metric("时长", f"{duration:.2f}s")
        with col3:
            if duration > 0:
                fps = episode_data['length'] / duration
                st.metric("实际帧率", f"{fps:.1f} fps")
        
        # 视频信息
        st.write("**📷 视频文件:**")
        video_info = {
            "主相机": {
                "exists": episode_data.get('primary_video_exists', False),
                "path": episode_data.get('primary_video_path', '')
            },
            "腕部相机": {
                "exists": episode_data.get('wrist_video_exists', False),
                "path": episode_data.get('wrist_video_path', '')
            }
        }
        
        for camera_name, info in video_info.items():
            if info['exists']:
                st.success(f"✅ {camera_name}: `{info['path']}`")
            else:
                st.error(f"❌ {camera_name}: 文件不存在")
    
    @staticmethod
    def display_file_structure(dataset_path: str) -> None:
        """显示数据集文件结构
        
        Args:
            dataset_path: 数据集路径
        """
        if not dataset_path:
            st.warning("没有数据集路径")
            return
        
        st.subheader("📁 数据集文件结构")
        
        path = Path(dataset_path)
        if not path.exists():
            st.error(f"路径不存在: {dataset_path}")
            return
        
        # 显示目录树
        def display_tree(directory: Path, level: int = 0, max_level: int = 3):
            if level > max_level:
                return
            
            try:
                items = sorted(directory.iterdir())
                for item in items:
                    if item.name.startswith('.'):
                        continue
                    
                    indent = "    " * level
                    if item.is_dir():
                        st.text(f"{indent}📁 {item.name}/")
                        # 只显示关键子目录
                        if item.name in ['data', 'meta', 'videos']:
                            display_tree(item, level + 1, max_level)
                    else:
                        st.text(f"{indent}📄 {item.name}")
            except PermissionError:
                st.text(f"{indent}🔒 (无权限访问)")
        
        display_tree(path)
    
    @staticmethod
    def display_validation_results(is_valid: bool, errors: list) -> None:
        """显示验证结果
        
        Args:
            is_valid: 是否验证通过
            errors: 错误列表
        """
        if is_valid:
            st.success("✅ 数据集结构验证通过")
        else:
            st.error("❌ 数据集结构验证失败")
            st.write("**错误详情:**")
            for error in errors:
                st.error(f"• {error}")


def display_dataset_overview(dataset_info: Dict[str, Any]) -> None:
    """便捷函数：显示数据集概览"""
    DatasetInfoDisplay.display_overview(dataset_info)


def display_episodes_list(episodes_list: list, show_videos: bool = True) -> None:
    """便捷函数：显示 episode 列表"""
    DatasetInfoDisplay.display_episodes_table(episodes_list, show_videos)