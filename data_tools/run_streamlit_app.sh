#!/bin/bash

# 机器人数据集处理平台启动脚本

echo "🤖 机器人数据集处理平台启动中..."
echo "================================"

# 检查是否在 data_tools 目录下
if [[ ! -f "streamlit_app.py" ]]; then
    echo "❌ 错误: 请在 data_tools 目录下运行此脚本"
    exit 1
fi

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python3，请先安装 Python3"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
python3 -c "import streamlit" 2>/dev/null || {
    echo "⚠️  Streamlit 未安装，正在安装依赖..."
    pip3 install -r requirements.txt
}

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 启动 Streamlit 应用
echo "🚀 启动 Streamlit 应用..."
echo "📱 应用将在浏览器中自动打开: http://localhost:8501"
echo "================================"

streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0