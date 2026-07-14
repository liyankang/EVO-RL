#!/usr/bin/env python3
"""
动作平滑页面 - 对动作数据进行平滑处理
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 设置页面配置
st.set_page_config(
    page_title="动作平滑 - 机器人数据集处理平台",
    page_icon="🛠️",
    layout="wide"
)

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# 导入模块
from utils.action_smoother import ActionSmoother
from utils.state_manager import StateManager
from utils.dataset_loader import DatasetLoader


def format_improvement(original_value: float, smoothed_value: float) -> str:
    """格式化改善百分比，避免原始值为 0 时除零。"""
    if original_value == 0:
        return "0.0%" if smoothed_value == 0 else "N/A"
    return f"{((original_value - smoothed_value) / original_value * 100):.1f}%"


# 初始化状态
StateManager.init()

st.markdown('<h1 style="color: #1f77b4;">🛠️ 动作平滑</h1>', unsafe_allow_html=True)
st.markdown("---")

if not StateManager.is_dataset_loaded():
    st.warning("⚠️ 请先在 '📂 数据加载' 页面加载数据集")
    st.info("💡 动作平滑功能需要加载数据集后才能使用")
    st.stop()

dataset_path = StateManager.get_dataset_path()
episodes_list = StateManager.get_episodes_list()
dataset_info = StateManager.get_dataset_info()

# 数据集信息概览
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("总 Episode 数", dataset_info.get('total_episodes', 0) if dataset_info else 0)
with col2:
    st.metric("当前数据集", Path(dataset_path).name)
with col3:
    if episodes_list:
        st.metric("总帧数", sum(ep['length'] for ep in episodes_list))

st.markdown("---")

# ---------- 参数配置 ----------
st.subheader("⚙️ 平滑参数配置")

col1, col2 = st.columns(2)

with col1:
    smoothing_method = st.selectbox(
        "平滑方法",
        ["savgol", "gaussian"],
        format_func=lambda x: "Savitzky-Golay 滤波器" if x == "savgol" else "高斯滤波",
        help="选择平滑算法"
    )

    if smoothing_method == "savgol":
        window_size = st.slider(
            "窗口大小",
            min_value=3,
            max_value=21,
            value=7,
            step=2,
            help="滤波窗口大小，必须是奇数"
        )
        poly_order = st.slider(
            "多项式阶数",
            min_value=1,
            max_value=5,
            value=2,
            help="多项式阶数，必须小于窗口大小"
        )
        st.info(f"📊 Savitzky-Golay (窗口={window_size}, 阶数={poly_order})")
    else:
        sigma = st.slider(
            "高斯标准差",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="高斯滤波的标准差，值越大平滑程度越高"
        )
        st.info(f"📊 高斯滤波 (σ={sigma})")

with col2:
    st.write("**处理选项**")
    process_all = st.checkbox("处理所有 Episode", value=True,
                             help="是否平滑所有 Episode，还是只处理选定的 Episode")

    if not process_all and episodes_list:
        episode_options = [ep['episode_index'] for ep in episodes_list]
        selected_episodes = st.multiselect(
            "选择要处理的 Episode",
            options=episode_options,
            default=[],
            help="留空表示处理所有 Episode"
        )
    else:
        selected_episodes = []

    auto_backup = st.checkbox("自动创建备份", value=True,
                             help="平滑前自动创建原始数据备份")

    cols_to_smooth = st.multiselect(
        "选择要平滑的列",
        ["action"],
        default=["action"],
        help="目前仅支持 action 列"
    )

st.markdown("---")

# ---------- 操作按钮 ----------
st.subheader("🎯 操作模式")

col1, col2, col3 = st.columns(3)
with col1:
    preview_clicked = st.button("👁️ 预览平滑效果", type="secondary", key="preview_smoothing", width='stretch')
with col2:
    execute_clicked = st.button("🚀 执行平滑处理", type="primary", key="execute_smoothing", width='stretch')
with col3:
    reset_clicked = st.button("🔄 重置参数", key="reset_smoothing", width='stretch')

if reset_clicked:
    st.session_state.smoothing_confirmed = False
    st.rerun()

# ---------- 预览逻辑 ----------
if preview_clicked:
    try:
        with st.spinner("🔍 正在分析平滑效果..."):
            # 创建平滑器实例（预览模式）
            smoother = ActionSmoother(
                dataset_root=Path(dataset_path),
                method=smoothing_method,
                window=window_size if smoothing_method == "savgol" else None,
                polyorder=poly_order if smoothing_method == "savgol" else None,
                sigma=sigma if smoothing_method == "gaussian" else None,
                cols=cols_to_smooth
            )

            # 执行预览（dry_run=True 会返回样本数据）
            result = smoother.smooth_dataset(dry_run=True)

            if result.get('success'):
                st.success("✅ 预览完成！")

                # 1. 处理预览统计
                st.markdown("### 📊 处理预览统计")
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("总文件数", result.get('total_files', 0))
                with c2: st.metric("总帧数", result.get('total_frames', 0))
                with c3: st.metric("总 Episode 数", result.get('total_episodes', 0))
                with c4: st.metric("处理列", ", ".join(result.get('parameters', {}).get('cols', [])))

                # 2. 样本数据可视化（曲线对比）
                sample = result.get('sample_data')
                if sample:
                    original = sample.get('original')
                    smoothed = sample.get('smoothed')
                    timestamps = sample.get('timestamps', np.arange(len(original)) if original is not None else [])

                    if original is not None and smoothed is not None and len(original) > 0:
                        st.markdown("### 📈 动作曲线对比（样本 Episode）")
                        num_total_joints = original.shape[1]
                        # 让用户选择显示哪些关节（默认前6个）
                        default_joints = list(range(min(num_total_joints, 6)))
                        joint_selector = st.multiselect(
                            "选择要显示的关节（可多选）",
                            options=list(range(num_total_joints)),
                            default=default_joints,
                            format_func=lambda x: f"Joint {x+1}"
                        )
                        if not joint_selector:
                            joint_selector = default_joints

                        for joint_idx in joint_selector:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=timestamps,
                                y=original[:, joint_idx],
                                mode='lines',
                                name=f'原始 Joint {joint_idx+1}',
                                line=dict(color='blue', width=1)
                            ))
                            fig.add_trace(go.Scatter(
                                x=timestamps,
                                y=smoothed[:, joint_idx],
                                mode='lines',
                                name=f'平滑 Joint {joint_idx+1}',
                                line=dict(color='red', width=2)
                            ))
                            fig.update_layout(
                                title=f'Joint {joint_idx+1} 平滑效果',
                                xaxis_title='帧数',
                                yaxis_title='动作值',
                                height=250,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                margin=dict(l=40, r=20, t=40, b=30)
                            )
                            st.plotly_chart(fig, width='stretch', config={'responsive': True})
                    else:
                        st.warning("样本数据为空，无法绘制曲线。")

                # 3. 统计表格（改善百分比）
                stats = result.get('statistics')
                if stats and 'improvement' in stats:
                    st.markdown("### 📊 改善幅度统计")
                    imp = stats['improvement']
                    data = []
                    for joint, vals in imp.items():
                        data.append({
                            '关节': joint,
                            'P95改善%': f"{vals.get('p95', 0):.1f}%",
                            'P99改善%': f"{vals.get('p99', 0):.1f}%",
                            '最大值改善%': f"{vals.get('max', 0):.1f}%"
                        })
                    if data:
                        df = pd.DataFrame(data)
                        # 高亮正改善为绿色，负改善为红色（使用 Styler.map 替代 applymap）
                        def highlight_positive(val):
                            if isinstance(val, str) and '%' in val:
                                num = float(val.replace('%', ''))
                                if num > 0:
                                    return 'background-color: #90EE90'
                                elif num < 0:
                                    return 'background-color: #FFCCCB'
                            return ''
                        st.dataframe(df.style.map(highlight_positive, subset=['P95改善%', 'P99改善%', '最大值改善%']),
                                     width='stretch')

                    # 柱状图：各关节P95改善
                    joints = list(imp.keys())
                    p95_vals = [imp[j].get('p95', 0) for j in joints]
                    if p95_vals:
                        fig_bar = go.Figure(data=[go.Bar(x=joints, y=p95_vals, marker_color='green')])
                        fig_bar.update_layout(
                            title='各关节 P95 改善百分比',
                            xaxis_title='关节',
                            yaxis_title='改善 (%)',
                            height=300,
                            margin=dict(l=40, r=20, t=40, b=30)
                        )
                        st.plotly_chart(fig_bar, width='stretch', config={'responsive': True})

                st.success("✅ 预览完成！如需应用平滑效果，请点击 '执行平滑处理' 按钮。")
            else:
                st.error("❌ 预览失败")
                if result.get('error'):
                    st.error(f"错误: {result['error']}")
    except Exception as e:
        st.error(f"❌ 预览过程中出错: {e}")
        import traceback
        st.error(traceback.format_exc())

# ---------- 执行逻辑 ----------
if execute_clicked:
    if not st.session_state.get('smoothing_confirmed', False):
        st.warning("⚠️ 此操作将修改原始数据！")
        if st.checkbox("我理解此操作将修改原始数据，并已确保有备份", key='confirm_smoothing'):
            st.session_state.smoothing_confirmed = True
            st.rerun()
    else:
        try:
            with st.spinner("🛠️ 正在进行平滑处理..."):
                smoother = ActionSmoother(
                    dataset_root=Path(dataset_path),
                    method=smoothing_method,
                    window=window_size if smoothing_method == "savgol" else None,
                    polyorder=poly_order if smoothing_method == "savgol" else None,
                    sigma=sigma if smoothing_method == "gaussian" else None,
                    cols=cols_to_smooth
                )
                result = smoother.smooth_dataset(dry_run=False)

                if result.get('success'):
                    st.success("🎉 平滑处理完成！")

                    c1, c2, c3, c4 = st.columns(4)
                    with c1: st.metric("成功文件", result['successful_files'])
                    with c2: st.metric("失败文件", result['failed_files'])
                    with c3: st.metric("处理帧数", result['total_frames'])
                    with c4: st.metric("处理列", ", ".join(result['parameters']['cols']))

                    if result.get('backup_dir'):
                        st.success(f"💾 备份已创建: `{result['backup_dir']}`")

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

                    st.session_state.smoothing_confirmed = False
                else:
                    st.error("❌ 平滑处理失败")
                    if result.get('error'):
                        st.error(f"错误: {result['error']}")
        except Exception as e:
            st.error(f"❌ 平滑处理过程中出错: {e}")
            import traceback
            st.error(traceback.format_exc())

# ---------- 使用说明 ----------
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 动作平滑功能说明

    **平滑方法**:
    - **Savitzky-Golay 滤波器**: 保留信号特征的同时进行平滑，适用于需要保持数据趋势的场景
    - **高斯滤波**: 基于高斯函数的平滑，适用于去除高频噪声

    **参数说明**:
    - **窗口大小**: 滤波窗口的帧数，较大的窗口产生更平滑的结果但可能过度平滑
    - **多项式阶数**: 仅用于 Savitzky-Golay 滤波器，控制拟合多项式的阶数
    - **高斯标准差**: 仅用于高斯滤波，控制平滑程度，值越大平滑效果越强

    **操作步骤**:
    1. 选择平滑方法和参数
    2. 选择要处理的 Episode 和数据列
    3. 点击 "预览平滑效果" 查看预期结果（含图表对比）
    4. 确认效果后点击 "执行平滑处理" 应用更改

    ⚠️ **注意事项**:
    - 平滑操作会修改原始动作数据，请确保有备份
    - 建议先用预览功能确认平滑效果
    - 过度平滑可能会丢失重要的动作细节
    """)

st.markdown("---")
st.info("💡 **提示**: 动作平滑可以减少噪声和突变，但可能也会丢失一些重要的动作细节，请根据实际需求调整参数。")