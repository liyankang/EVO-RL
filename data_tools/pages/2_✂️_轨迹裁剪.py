#!/usr/bin/env python3
"""
轨迹裁剪页面 - 按时间或帧数裁剪轨迹数据
"""

import streamlit as st
import sys
from pathlib import Path
import cv2
import numpy as np

# 设置页面配置
st.set_page_config(
    page_title="轨迹裁剪 - 机器人数据集处理平台",
    page_icon="✂️",
    layout="wide"
)

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# 导入模块
from utils.video_truncator import VideoTruncator
from components.trajectory_slider import create_trajectory_slider
from components.video_player_with_slider import create_video_player_with_slider
from utils.state_manager import StateManager
from utils.dataset_loader import DatasetLoader


def collect_episode_videos(dataset_path: str, episode_index: int):
    """收集当前 episode 的多相机视频文件。"""
    video_files_info = []
    video_dir = Path(dataset_path) / "videos" / "chunk-000"

    if not video_dir.exists():
        return video_files_info

    for camera_subdir in video_dir.iterdir():
        if not camera_subdir.is_dir():
            continue
        video_path = camera_subdir / f"episode_{episode_index:06d}.mp4"
        if video_path.exists():
            video_files_info.append((camera_subdir.name, video_path))

    return video_files_info


def get_action_strength(slider):
    """计算每帧动作强度，用于裁剪滑块上方的动作曲线。"""
    if slider.trajectory_data is None or "action" not in slider.trajectory_data.columns:
        return None

    actions = np.stack(slider.trajectory_data["action"].values)
    return np.linalg.norm(actions, axis=1)


# 初始化状态
StateManager.init()

# 页面标题
st.markdown('<h1 style="color: #1f77b4;">✂️ 轨迹裁剪</h1>', unsafe_allow_html=True)
st.markdown("---")

# 检查是否已加载数据集
if not StateManager.is_dataset_loaded():
    st.warning("⚠️ 请先在 '📂 数据加载' 页面加载数据集")
    st.info("💡 轨迹裁剪功能需要加载数据集后才能使用")
else:
    dataset_path = StateManager.get_dataset_path()
    episodes_list = StateManager.get_episodes_list()
    
    if not episodes_list:
        st.error("❌ 没有可用的 Episode 数据")
    else:
        episode_options = [ep['episode_index'] for ep in episodes_list]
        
        # 尝试加载选中的 episode 并获取基本信息
        try:
            # 创建轨迹滑块实例，以便后续获取 trajectory_info
            # 注意：先选择一个默认 episode，但避免重复创建
            selected_episode_idx = st.session_state.get("selected_episode_crop", episode_options[0])
            
            # 选择 Episode（放在顶部）
            st.markdown("### 🎬 选择 Episode")
            col_sel, col_frames = st.columns([2, 1])
            with col_sel:
                selected_episode_idx = st.selectbox(
                    "Episode",
                    options=episode_options,
                    format_func=lambda x: f"Episode {x}",
                    key="episode_selector_crop",
                    label_visibility="collapsed"
                )
                # 保存到 session_state 以便后续使用
                st.session_state["selected_episode_crop"] = selected_episode_idx
            
            # 创建轨迹滑块对象（根据选中的 episode）
            slider = create_trajectory_slider(dataset_path, selected_episode_idx)
            trajectory_info = slider.get_trajectory_info()
            
            # 收集视频文件
            video_files_info = collect_episode_videos(dataset_path, selected_episode_idx)
            st.session_state.video_files_info = video_files_info

            # 获取总帧数（优先从视频获取）
            total_frames = int(trajectory_info.get("length", 0) or 0)
            fps = 30.0
            if video_files_info:
                first_video_path = video_files_info[0][1]
                cap = cv2.VideoCapture(str(first_video_path))
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or total_frames
                    fps = cap.get(cv2.CAP_PROP_FPS) or fps
                    cap.release()
            
            # 显示总帧数（在右侧）
            with col_frames:
                st.metric("📊 总帧数", total_frames)
            
            st.markdown("---")
            
            # ---------- 裁剪工作区 ----------
            st.markdown("### ✂️ 裁剪工作区")
            
            start_value, end_value = None, None
            if video_files_info:
                # 有视频：使用视频播放器 + 滑块
                action_strength = get_action_strength(slider)
                video_player = create_video_player_with_slider(
                    video_files_info,
                    total_frames,
                    fps
                )
                # 渲染视频播放器，获取裁剪范围（帧索引）
                start_frame, end_frame = video_player.render(height=250, action_strength=action_strength)
                
                # 根据帧索引计算对应的裁剪范围（时间值或帧值）
                if trajectory_info['has_timestamp'] and trajectory_info['timestamps'] is not None:
                    timestamps = trajectory_info['timestamps']
                    start_value = timestamps[start_frame] if start_frame < len(timestamps) else 0
                    end_value = timestamps[end_frame] if end_frame < len(timestamps) else timestamps[-1]
                else:
                    start_value = start_frame / fps
                    end_value = end_frame / fps
                
                # 在滑块下方显示当前裁剪范围帧数
                st.caption(f"📌 当前裁剪范围：帧 {start_frame} → {end_frame}（共 {end_frame - start_frame} 帧）")
            else:
                # 无视频：使用传统轨迹滑块（时间/帧模式）
                st.info("📭 当前 Episode 没有视频，切换到轨迹模式（基于动作数据）")
                mode = st.radio(
                    "选择裁剪模式",
                    ["time", "frame"],
                    format_func=lambda x: "⏰ 时间模式" if x == "time" else "🔢 帧数模式",
                    horizontal=True
                )
                start_value, end_value = slider.render_time_slider(mode=mode)
            
            if start_value is not None and end_value is not None:
                st.markdown("---")
                
                # ---------- 高级设置 & 执行按钮 ----------
                with st.expander("⚙️ 高级设置", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        validate_alignment = st.checkbox(
                            "验证视频轨迹对齐",
                            value=True,
                            help="裁剪前验证视频和轨迹数据的对应关系"
                        )
                    with col2:
                        auto_backup = st.checkbox(
                            "自动创建备份",
                            value=True,
                            help="自动创建数据集备份"
                        )
                
                # 执行按钮（占满宽度）
                if st.button("🚨 执行裁剪", type="primary", use_container_width=True, key="execute_truncation"):
                    try:
                        with st.spinner("✂️ 正在裁剪轨迹数据..."):
                            truncator = VideoTruncator(
                                dataset_root=Path(dataset_path),
                                episode_index=selected_episode_idx,
                                start_time=start_value,
                                end_time=end_value,
                                auto_backup=auto_backup
                            )
                            
                            result = truncator.truncate(validate_alignment=validate_alignment)
                            
                            if result.get('success'):
                                st.success("🎉 轨迹裁剪成功！")
                                
                                result_col1, result_col2, result_col3 = st.columns(3)
                                with result_col1:
                                    st.metric("处理前帧数", result['trajectory_result']['original_length'])
                                with result_col2:
                                    st.metric("处理后帧数", result['trajectory_result']['new_length'])
                                with result_col3:
                                    reduction = result['trajectory_result']['original_length'] - result['trajectory_result']['new_length']
                                    st.metric("删除帧数", reduction)
                                
                                if result.get('successful_videos_count', 0) > 0:
                                    st.success(f"✅ 成功处理 {result['successful_videos_count']} 个视频文件")
                                
                                if result.get('backup_dir'):
                                    st.info(f"💾 备份已创建: `{result['backup_dir']}`")
                                
                                # 更新数据集信息
                                try:
                                    loader = DatasetLoader(Path(dataset_path))
                                    episodes_result = loader.get_episodes_list()
                                    if episodes_result.get('success'):
                                        from utils.state_manager import load_dataset
                                        load_dataset(dataset_path, loader)
                                        st.success("✅ 数据集信息已更新")
                                except Exception as update_error:
                                    st.warning(f"⚠️ 更新数据集信息时出错: {update_error}")
                            else:
                                st.error("❌ 轨迹裁剪失败")
                                if 'error' in result:
                                    st.error(f"错误信息: {result['error']}")
                    except Exception as e:
                        st.error(f"❌ 裁剪过程中出错: {e}")
                        import traceback
                        st.error(traceback.format_exc())
        
        except Exception as e:
            st.error(f"❌ 加载轨迹数据时出错: {e}")
            import traceback
            st.error(traceback.format_exc())

# ---------- 底部使用说明 ----------
with st.expander("📖 使用说明"):
    st.markdown("""
    1. **选择 Episode**：从下拉菜单选择要裁剪的轨迹片段。
    2. **浏览与裁剪**：
       - 如果有视频，使用视频播放器下方的双滑块选择裁剪范围（拖动左右端点）。
       - 如果没有视频，则切换至时间/帧数模式，基于动作数据选择。
    3. **高级设置**（可选）：开启/关闭验证对齐和自动备份。
    4. **执行裁剪**：点击按钮，系统将裁剪轨迹数据和对应的视频文件。
    ⚠️ **注意**：裁剪会修改原始数据，建议先预览范围，确认无误后执行。
    """)

st.markdown("---")
st.caption("💡 提示：裁剪后数据集统计信息会自动更新，可返回「数据加载」页面查看。")