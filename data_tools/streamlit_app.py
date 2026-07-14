#!/usr/bin/env python3
"""
机器人数据集处理平台 - Streamlit 主应用
支持多页面架构，提供数据加载、轨迹裁剪、动作平滑、数据对齐检查功能
"""

import streamlit as st
import sys
from pathlib import Path

# 添加 data_tools 目录到 Python 路径
data_tools_dir = Path(__file__).parent
if str(data_tools_dir) not in sys.path:
    sys.path.insert(0, str(data_tools_dir))

# 设置页面配置
st.set_page_config(
    page_title="机器人数据集处理平台",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .dataset-info-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeeba;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 session state
def init_session_state():
    """初始化应用的 session state"""
    if 'dataset_path' not in st.session_state:
        st.session_state.dataset_path = None
    if 'dataset_info' not in st.session_state:
        st.session_state.dataset_info = None
    if 'episodes_list' not in st.session_state:
        st.session_state.episodes_list = None
    if 'current_episode' not in st.session_state:
        st.session_state.current_episode = None
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True

# 初始化 session state
init_session_state()

# 主页面标题
st.markdown('<div class="main-header">🤖 机器人数据集处理平台</div>', unsafe_allow_html=True)

# 侧边栏信息
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 当前数据集状态")
    
    if st.session_state.dataset_path:
        st.success(f"✅ 已加载数据集")
        st.info(f"📁 路径: `{st.session_state.dataset_path}`")
        
        if st.session_state.dataset_info:
            info = st.session_state.dataset_info
            st.metric("Episode 数量", info.get('total_episodes', 0))
            st.metric("总帧数", info.get('total_frames', 0))
            st.metric("帧率", f"{info.get('fps', 30)} fps")
            
            if info.get('robot_type'):
                st.text(f"🤖 机器人类型: {info['robot_type']}")
            
            if info.get('cameras'):
                st.text(f"📷 相机数量: {len(info['cameras'])}")
    else:
        st.warning("⚠️ 未加载数据集")
        st.text("请在 '数据加载' 页面选择数据集")
    
    st.markdown("---")
    st.markdown("### 🔧 功能说明")
    st.markdown("""
    - **📂 数据加载**: 交互式选择数据集
    - **✂️ 轨迹裁剪**: 按时间/帧数裁剪轨迹
    - **🛠️ 动作平滑**: 平滑处理动作数据
    - **✅ 数据对齐**: 检查视频与轨迹对齐
    """)

# 显示当前页面信息
st.markdown("---")
st.info("💡 请从左侧导航栏选择功能页面，首次使用请先在 '📂 数据加载' 页面加载数据集。")

# 页脚
st.markdown("---")
st.markdown('<div style="text-align: center; color: #666;">基于 LeRobot 格式 | 支持 Python Streamlit</div>', unsafe_allow_html=True)