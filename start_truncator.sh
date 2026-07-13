#!/bin/bash
# 视频轨迹截断工具启动脚本

echo "========================================="
echo "视频轨迹截断工具启动脚本"
echo "========================================="
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import flask, flask_cors, cv2, pandas, pyarrow" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "正在安装依赖..."
    pip install -r video_truncator_requirements.txt
fi

echo "依赖检查完成！"
echo ""

# 启动服务器
echo "启动视频轨迹截断工具服务器..."
echo "访问地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务器"
echo ""
echo "========================================="
echo ""

python3 video_truncator_server.py "$@"