#!/usr/bin/env python3
"""
数据对齐检查页面 - 检查视频与轨迹数据的对应关系
"""

import streamlit as st
import sys
from pathlib import Path

# 设置页面配置
st.set_page_config(
    page_title="数据对齐检查 - 机器人数据集处理平台",
    page_icon="✅",
    layout="wide"
)

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# 导入模块
from utils.video_truncator import VideoTruncator
from components.dataset_info import DatasetInfoDisplay
from utils.state_manager import StateManager
from utils.dataset_loader import DatasetLoader

# 初始化状态
StateManager.init()

# 页面标题
st.markdown('<h1 style="color: #1f77b4;">✅ 数据对齐检查</h1>', unsafe_allow_html=True)
st.markdown("---")

# 检查是否已加载数据集
if not StateManager.is_dataset_loaded():
    st.warning("⚠️ 请先在 '📂 数据加载' 页面加载数据集")
    st.info("💡 数据对齐检查功能需要加载数据集后才能使用")
else:
    dataset_path = StateManager.get_dataset_path()
    episodes_list = StateManager.get_episodes_list()
    
    # 数据集信息概览
    col1, col2, col3 = st.columns(3)
    with col1:
        dataset_info = StateManager.get_dataset_info()
        if dataset_info:
            st.metric("总 Episode 数", dataset_info.get('total_episodes', 0))
    with col2:
        st.metric("当前数据集", Path(dataset_path).name)
    with col3:
        if episodes_list:
            st.metric("总帧数", sum(ep['length'] for ep in episodes_list))
    
    st.markdown("---")
    
    # 检查模式选择
    st.subheader("🎯 检查模式")
    
    col1, col2 = st.columns(2)
    
    with col1:
        check_mode = st.radio(
            "选择检查范围",
            ["all", "single"],
            format_func=lambda x: "📊 检查所有 Episodes" if x == "all" else "🎬 检查单个 Episode",
            horizontal=True
        )
    
    with col2:
        show_details = st.checkbox("显示详细信息", value=False, 
                                  help="是否显示每个相机的详细信息")
    
    # Episode 选择
    selected_episode_idx = None
    if check_mode == "single" and episodes_list:
        selected_episode_idx = st.selectbox(
            "选择要检查的 Episode",
            options=[ep['episode_index'] for ep in episodes_list],
            format_func=lambda x: next((f"Episode {ep['episode_index']} ({ep['length']} 帧)" 
                                        for ep in episodes_list if ep['episode_index'] == x), str(x))
        )
    
    # 执行检查
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 执行检查", type="primary", key="execute_alignment_check"):
            try:
                with st.spinner("🔍 正在检查数据对齐..."):
                    if check_mode == "single" and selected_episode_idx is not None:
                        # 检查单个 episode
                        truncator = VideoTruncator(
                            dataset_root=Path(dataset_path),
                            episode_index=selected_episode_idx
                        )
                        
                        result = truncator.validate_video_trajectory_alignment()
                        
                        # 显示结果
                        st.markdown(f"### 📋 Episode {selected_episode_idx} 对齐检查结果")
                        
                        if result['valid']:
                            st.success("✅ 数据对齐检查通过！")
                        else:
                            st.error("❌ 发现数据对齐问题！")
                        
                        # 显示轨迹信息
                        st.markdown("#### 📊 轨迹数据信息")
                        traj_info = result['trajectory_info']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("帧数", traj_info['frames'])
                        with col2:
                            has_timestamp = "✅ 有" if traj_info['has_timestamp'] else "❌ 无"
                            st.metric("时间戳", has_timestamp)
                        with col3:
                            if traj_info['time_range']:
                                st.metric("时间范围", f"{traj_info['time_range'][0]:.2f}s - {traj_info['time_range'][1]:.2f}s")
                        
                        # 显示视频信息
                        st.markdown("#### 📷 视频数据信息")
                        alignment_info = result['alignment_check']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("相机数量", alignment_info['total_cameras'])
                        with col2:
                            st.metric("匹配相机", alignment_info['matching_cameras'])
                        with col3:
                            all_match = "✅ 是" if alignment_info['all_match'] else "❌ 否"
                            st.metric("全部匹配", all_match)
                        
                        # 显示每个相机的详细信息
                        if show_details and result['video_info']:
                            st.markdown("#### 🎥 相机详细信息")
                            
                            for camera_name, camera_info in result['video_info'].items():
                                with st.expander(f"📷 {camera_name}"):
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("帧数", camera_info['frames'])
                                    with col2:
                                        st.metric("帧率", f"{camera_info['fps']:.1f} fps")
                                    with col3:
                                        st.metric("时长", f"{camera_info['duration']:.1f}s")
                                    with col4:
                                        if camera_info['frames_match']:
                                            st.success("✅ 帧数匹配")
                                        else:
                                            st.error(f"❌ 差异 {camera_info['frame_difference']} 帧")
                        
                        # 显示时间戳分析
                        if result['alignment_check'].get('timestamp_analysis'):
                            st.markdown("#### ⏰ 时间戳分析")
                            ts_analysis = result['alignment_check']['timestamp_analysis']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("平均间隔", f"{ts_analysis['avg_interval']:.4f}s")
                            with col2:
                                st.metric("最小间隔", f"{ts_analysis['min_interval']:.4f}s")
                            with col3:
                                st.metric("最大间隔", f"{ts_analysis['max_interval']:.4f}s")
                            with col4:
                                is_consistent = "✅ 是" if ts_analysis['is_consistent'] else "❌ 否"
                                st.metric("一致性", is_consistent)
                            
                            if not ts_analysis['is_consistent']:
                                st.warning(f"⚠️ 时间戳间隔不一致！期望 {ts_analysis['expected_interval']:.4f}s，实际平均 {ts_analysis['avg_interval']:.4f}s")
                        
                        # 显示错误信息
                        if result['errors']:
                            st.markdown("#### ❌ 错误信息")
                            for error in result['errors']:
                                st.error(f"• {error}")
                    
                    else:
                        # 检查所有 episodes
                        results = {}
                        total_episodes = len(episodes_list)
                        valid_episodes = 0
                        issues_found = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, episode in enumerate(episodes_list):
                            episode_idx = episode['episode_index']
                            status_text.text(f"正在检查 Episode {episode_idx} ({i+1}/{total_episodes})...")
                            
                            try:
                                truncator = VideoTruncator(
                                    dataset_root=Path(dataset_path),
                                    episode_index=episode_idx
                                )
                                
                                result = truncator.validate_video_trajectory_alignment()
                                results[episode_idx] = result
                                
                                if result['valid']:
                                    valid_episodes += 1
                                else:
                                    issues_found.append({
                                        'episode': episode_idx,
                                        'errors': result['errors']
                                    })
                            
                            except Exception as e:
                                issues_found.append({
                                    'episode': episode_idx,
                                    'errors': [f"检查时出错: {str(e)}"]
                                })
                            
                            progress_bar.progress((i + 1) / total_episodes)
                        
                        status_text.text("✅ 检查完成！")
                        
                        # 显示汇总结果
                        st.markdown("### 📊 对齐检查汇总")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("总 Episodes", total_episodes)
                        with col2:
                            st.metric("通过检查", valid_episodes)
                        with col3:
                            st.metric("发现问题", total_episodes - valid_episodes)
                        with col4:
                            success_rate = (valid_episodes / total_episodes * 100) if total_episodes > 0 else 0
                            st.metric("通过率", f"{success_rate:.1f}%")
                        
                        # 显示问题详情
                        if issues_found:
                            st.markdown("### ❌ 发现的问题")
                            
                            for issue in issues_found:
                                with st.expander(f"❌ Episode {issue['episode']}"):
                                    for error in issue['errors']:
                                        st.error(f"• {error}")
                        else:
                            st.success("🎉 所有 Episodes 都通过了数据对齐检查！")
                        
                        # 详细结果表格
                        if show_details:
                            st.markdown("### 📋 详细检查结果")
                            
                            import pandas as pd
                            detailed_results = []
                            
                            for episode_idx, result in results.items():
                                episode_data = next(ep for ep in episodes_list if ep['episode_index'] == episode_idx)
                                
                                detailed_results.append({
                                    'Episode': episode_idx,
                                    '状态': '✅ 通过' if result['valid'] else '❌ 失败',
                                    '轨迹帧数': result['trajectory_info']['frames'],
                                    '相机数量': result['alignment_check']['total_cameras'],
                                    '匹配相机': result['alignment_check']['matching_cameras'],
                                    '时间戳': '✅ 有' if result['trajectory_info']['has_timestamp'] else '❌ 无',
                                    '问题数量': len(result['errors'])
                                })
                            
                            st.dataframe(pd.DataFrame(detailed_results), width='stretch')
                
                st.success("✅ 数据对齐检查完成！")
                
            except Exception as e:
                st.error(f"❌ 检查过程中出错: {e}")
                import traceback
                st.error(traceback.format_exc())
    
    with col2:
        if st.button("📊 生成检查报告", key="generate_report"):
            try:
                with st.spinner("📄 正在生成报告..."):
                    # 这里可以添加生成详细报告的逻辑
                    st.info("📄 报告生成功能正在开发中...")
                    
            except Exception as e:
                st.error(f"❌ 生成报告时出错: {e}")
    
    with col3:
        if st.button("🔄 重置检查", key="reset_alignment"):
            st.rerun()

# 使用说明
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 数据对齐检查功能说明
    
    **检查内容**:
    - **轨迹数据**: 检查轨迹数据的完整性和时间戳
    - **视频数据**: 检查每个相机的视频文件是否存在
    - **帧数匹配**: 检查视频帧数与轨迹帧数是否一致
    - **时间戳一致性**: 检查时间戳间隔是否合理
    
    **检查模式**:
    - **📊 检查所有 Episodes**: 对整个数据集进行完整检查
    - **🎬 检查单个 Episode**: 对选定的 Episode 进行详细检查
    
    **结果说明**:
    - ✅ **通过**: 轨迹和视频数据完全对齐
    - ❌ **失败**: 发现帧数不匹配或其他问题
    
    **常见问题**:
    - **帧数不匹配**: 视频帧数与轨迹帧数不一致，可能需要重新录制或处理
    - **缺少视频文件**: 某个相机的视频文件丢失
    - **时间戳不一致**: 时间戳间隔异常，可能影响数据同步
    
    ⚠️ **注意事项**:
    - 数据对齐是训练机器人的重要前提
    - 发现问题应及时处理，避免影响训练效果
    - 建议定期检查数据集的对齐状态
    - 对于大型数据集，完整检查可能需要较长时间
    """)

st.markdown("---")
st.info("💡 **提示**: 数据对齐检查可以确保视频和轨迹数据的一致性，这是机器人学习的基础。建议定期检查数据集质量。")

# 检查统计
if StateManager.is_dataset_loaded() and episodes_list:
    with st.expander("📈 数据集质量统计"):
        st.markdown("### 📊 当前数据集统计")
        
        # 计算一些基本统计
        total_episodes = len(episodes_list)
        total_frames = sum(ep['length'] for ep in episodes_list)
        avg_frames = total_frames / total_episodes if total_episodes > 0 else 0
        
        # 视频完整性统计
        primary_video_count = sum(1 for ep in episodes_list if ep.get('primary_video_exists'))
        wrist_video_count = sum(1 for ep in episodes_list if ep.get('wrist_video_exists'))
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("总 Episodes", total_episodes)
        with col2:
            st.metric("总帧数", total_frames)
        with col3:
            st.metric("平均帧数", f"{avg_frames:.0f}")
        with col4:
            st.metric("主相机完整性", f"{primary_video_count/total_episodes*100:.1f}%")
        with col5:
            st.metric("腕部相机完整性", f"{wrist_video_count/total_episodes*100:.1f}%")
        
        # 时长分布
        durations = [ep.get('end_timestamp', 0) - ep.get('start_timestamp', 0) 
                    for ep in episodes_list if ep.get('end_timestamp') and ep.get('start_timestamp')]
        
        if durations:
            st.markdown("### ⏱️ 时长分布")
            
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=durations,
                nbinsx=20,
                name='时长分布',
                marker_color='#1f77b4'
            ))
            
            fig.update_layout(
                title='Episode 时长分布',
                xaxis_title='时长 (秒)',
                yaxis_title='Episode 数量',
                height=300
            )
            
            st.plotly_chart(fig, width='stretch')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("最短时长", f"{min(durations):.1f}s")
            with col2:
                st.metric("平均时长", f"{sum(durations)/len(durations):.1f}s")
            with col3:
                st.metric("最长时长", f"{max(durations):.1f}s")