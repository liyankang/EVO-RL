#!/usr/bin/env python3
"""
数据加载页面 - 交互式文件夹网格导航 + 美观统计展示
"""

import streamlit as st
import sys
import pandas as pd
from pathlib import Path

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from utils.state_manager import StateManager, load_dataset, clear_dataset
from utils.validators import validate_dataset_structure
from utils.dataset_loader import DatasetLoader

st.set_page_config(
    page_title="数据加载 - 机器人数据集处理平台",
    page_icon="📂",
    layout="wide"
)

# 初始化状态
StateManager.init()

st.markdown('<h1 style="color: #1f77b4;">📂 加载数据集</h1>', unsafe_allow_html=True)
st.markdown("---")

# ---------- 如果已加载，显示美观统计 ----------
if StateManager.is_dataset_loaded():
    st.success(f"✅ 当前数据集: `{StateManager.get_dataset_path()}`")
    
    # 获取数据集信息
    dataset_info = StateManager.get_dataset_info()
    episodes_list = StateManager.get_episodes_list()
    
    if dataset_info and episodes_list:
        # 1. 三个核心指标卡片
        total_episodes = len(episodes_list)
        total_frames = sum(ep.get('length', 0) for ep in episodes_list)
        # 如果有时长信息（假设每个 episode 有 'duration' 字段），否则省略
        has_duration = any('duration' in ep for ep in episodes_list)
        total_duration = sum(ep.get('duration', 0) for ep in episodes_list) if has_duration else None
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📦 总 Episode 数", total_episodes)
        with col2:
            st.metric("🔄 总帧数", f"{total_frames:,}")
        with col3:
            if total_duration is not None:
                st.metric("⏱️ 总时长", f"{total_duration:.1f}s")
            else:
                st.metric("⏱️ 总时长", "N/A")
        
        st.markdown("---")
        
        # 2. Episode 帧数分布条形图
        st.subheader("📊 Episode 帧数分布")
        # 准备数据
        df = pd.DataFrame({
            "Episode": [ep.get('episode_index', i) for i, ep in enumerate(episodes_list)],
            "帧数": [ep.get('length', 0) for ep in episodes_list]
        })
        # 使用 Streamlit 原生条形图
        st.bar_chart(df.set_index("Episode"), width='stretch')
        
        # 3. 详细统计（折叠）
        with st.expander("📋 详细统计信息", expanded=False):
            # 显示动作/状态统计（如果有）
            if dataset_info:
                st.json(dataset_info)
            else:
                # 显示 Episode 列表
                st.dataframe(df, width='stretch')
    
    # 切换数据集按钮
    if st.button("🔄 切换数据集", type="secondary"):
        clear_dataset()
        st.rerun()
    
    st.stop()

# ---------- 初始化浏览路径 ----------
if "browse_path" not in st.session_state:
    default_root = Path("/media/Data/huggingface/lerobot/")
    st.session_state.browse_path = default_root if default_root.exists() else Path.cwd()

# ---------- 面包屑导航 + 快捷操作 ----------
st.markdown("### 🗂️ 浏览目录")
col_path, col_back, col_home = st.columns([4, 1, 1])
with col_path:
    st.info(f"📍 当前路径: `{st.session_state.browse_path}`")
with col_back:
    if st.button("⬆ 上一级", width='stretch'):
        parent = st.session_state.browse_path.parent
        if parent != st.session_state.browse_path:
            st.session_state.browse_path = parent
            st.rerun()
with col_home:
    default_root = Path("/media/Data/huggingface/lerobot/")
    home_path = default_root if default_root.exists() else Path.cwd()
    if st.button("🏠 根目录", width='stretch'):
        st.session_state.browse_path = home_path
        st.rerun()

st.markdown("---")

# ---------- 子文件夹网格 ----------
st.markdown("#### 📁 子文件夹（点击进入）")

try:
    subdirs = sorted([d for d in st.session_state.browse_path.iterdir() if d.is_dir()])
except PermissionError:
    st.warning("⚠️ 无权限读取该目录")
    subdirs = []

if subdirs:
    cols_per_row = 4
    for i in range(0, len(subdirs), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(subdirs):
                dir_path = subdirs[idx]
                if col.button(
                    f"📁 {dir_path.name}",
                    key=f"dir_{idx}_{dir_path.name}",
                    width='stretch'
                ):
                    st.session_state.browse_path = dir_path
                    st.rerun()
else:
    st.info("📭 当前目录下没有子文件夹")

st.markdown("---")

# ---------- 路径验证 + 加载 ----------
current_path = st.session_state.browse_path
is_valid, errors = validate_dataset_structure(current_path)

col_status, col_action = st.columns([3, 1])
with col_status:
    if is_valid:
        st.success(f"✅ 有效数据集: `{current_path.name}` (包含 data/, meta/, videos/)")
    else:
        st.error("❌ 当前目录不是有效的 LeRobot 数据集")
        if errors:
            with st.expander("查看缺失项"):
                for e in errors:
                    st.warning(f"- {e}")

with col_action:
    if is_valid:
        if st.button("🚀 加载数据集", type="primary", width='stretch'):
            with st.spinner("正在加载数据集..."):
                try:
                    loader = DatasetLoader(current_path)
                    success = load_dataset(str(current_path), loader)
                    if success:
                        st.success("✅ 加载成功！")
                        st.rerun()
                    else:
                        st.error("❌ 加载失败，请查看日志")
                except Exception as e:
                    st.error(f"❌ 加载异常: {e}")

st.markdown("---")
st.caption("💡 点击文件夹按钮即可进入子目录；当路径被识别为有效数据集时，【加载数据集】按钮会亮起。")