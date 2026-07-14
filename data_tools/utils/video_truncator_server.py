#!/usr/bin/env python3
"""视频与轨迹截断Web服务器，提供HTTP API接口。"""

import logging
import argparse
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response, send_file
from urllib.parse import unquote

try:
    from flask_cors import CORS
except Exception:  # pragma: no cover - local-only use should not require flask-cors.
    CORS = None

# 导入核心处理模块
from video_processor import (
    DatasetLoader, 
    VideoTruncator, 
    restore_latest_backup,
    _serialize_stats_tree,
    ActionSmoother
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
if CORS is not None:
    CORS(app)

# 全局处理状态
processing_status = {
    'is_processing': False,
    'progress': 0,
    'message': '',
    'error': None
}


@app.route('/')
def index():
    """主页，返回HTML界面"""
    return send_from_directory('.', 'video_truncator.html')


@app.route('/load_dataset', methods=['POST'])
def load_dataset():
    """加载数据集信息"""
    try:
        data = request.json or {}
        dataset_root = data.get('dataset_root')
        if not dataset_root:
            return jsonify({'success': False, 'error': '缺少dataset_root参数'}), 400

        loader = DatasetLoader(dataset_root)
        is_valid, errors = loader.validate_dataset()
        if not is_valid:
            return jsonify({'success': False, 'error': '数据集验证失败', 'details': errors}), 400

        return jsonify(loader.get_episodes_list())
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get_video_info', methods=['POST'])
def get_video_info():
    """获取视频文件信息"""
    try:
        data = request.json or {}
        video_path = data.get('video_path')
        dataset_root = data.get('dataset_root')
        if not video_path or not dataset_root:
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400

        loader = DatasetLoader(dataset_root)
        return jsonify(loader.get_video_info(video_path))
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/truncate', methods=['POST'])
def truncate():
    """执行视频和轨迹截断"""
    try:
        data = request.json or {}
        
        # 验证必要参数
        for field in ['dataset_root', 'episode_index']:
            if field not in data:
                return jsonify({'success': False, 'error': f'缺少必要参数: {field}'}), 400

        # 验证截断参数 - 支持时间戳截断
        has_time_range = 'start_time' in data or 'end_time' in data
        has_keep_window = 'keep_start_frame' in data or 'keep_end_frame' in data
        
        if has_time_range:
            # 基于时间戳的截断
            if 'start_time' not in data or 'end_time' not in data:
                return jsonify({'success': False, 'error': '时间戳截断需要同时提供 start_time 和 end_time'}), 400
        elif has_keep_window:
            # 基于帧窗口的截断
            for field in ['keep_start_frame', 'keep_end_frame']:
                if field not in data:
                    return jsonify({'success': False, 'error': f'缺少必要参数: {field}'}), 400
        elif 'truncate_frame' not in data:
            return jsonify({'success': False, 'error': '缺少必要参数: truncate_frame'}), 400

        # 更新处理状态
        processing_status.update({
            'is_processing': True,
            'progress': 0,
            'message': '准备处理...',
            'error': None,
        })

        # 创建截断器并执行截断
        truncator = VideoTruncator(
            dataset_root=data['dataset_root'],
            episode_index=data['episode_index'],
            truncate_frame=int(data['truncate_frame']) if 'truncate_frame' in data else None,
            discard_mode=data.get('discard_mode', 'after'),
            keep_start_frame=data.get('keep_start_frame'),
            keep_end_frame=data.get('keep_end_frame'),
            # 新增：支持时间戳参数
            start_time=data.get('start_time'),
            end_time=data.get('end_time'),
        )
        
        # 添加进度回调支持
        def update_progress(progress, message):
            processing_status['progress'] = progress
            processing_status['message'] = message
        
        # 执行截断
        result = _serialize_stats_tree(truncator.truncate())
        
        # 更新处理状态
        processing_status['is_processing'] = False
        processing_status['progress'] = 100
        processing_status['message'] = '截断完成'
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to process truncate request: {e}")
        processing_status['is_processing'] = False
        processing_status['error'] = str(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/undo', methods=['POST'])
def undo():
    """回退上一步操作"""
    try:
        data = request.json or {}
        dataset_root = data.get('dataset_root')
        if not dataset_root:
            return jsonify({'success': False, 'error': '缺少dataset_root参数'}), 400

        # 更新处理状态
        processing_status.update({
            'is_processing': True,
            'progress': 0,
            'message': '准备回退...',
            'error': None,
        })

        result = restore_latest_backup(dataset_root)
        
        # 更新处理状态
        processing_status['is_processing'] = False
        processing_status['progress'] = 100
        processing_status['message'] = '回退完成'
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to undo truncate request: {e}")
        processing_status['is_processing'] = False
        processing_status['error'] = str(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/status')
def status():
    """获取当前处理状态"""
    return jsonify(processing_status)


@app.route('/video/<path:filepath>')
def serve_video(filepath):
    """提供视频文件服务，支持HTTP Range请求"""
    try:
        # 安全处理文件路径
        filepath = unquote(filepath).replace('..', '').replace('//', '/')
        if len(filepath) >= 3 and filepath[0] == '/' and filepath[2] == ':':
            filepath = filepath[1:]

        full_path = Path(filepath)
        if not full_path.is_absolute():
            full_path = Path('/') / filepath
        
        # 验证文件存在
        if not full_path.exists():
            return jsonify({'error': '视频文件不存在'}), 404
        if not full_path.is_file():
            return jsonify({'error': '无效的视频文件'}), 400

        # 验证文件格式
        file_extension = full_path.suffix.lower()
        if file_extension not in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return jsonify({'error': f'不支持的视频格式: {file_extension}'}), 400

        # 设置MIME类型
        mimetype = 'video/mp4' if file_extension == '.mp4' else 'video/webm'
        file_size = full_path.stat().st_size
        
        # 处理Range请求（支持视频流式播放）
        range_header = request.headers.get('Range')
        if range_header:
            try:
                units, byte_range = range_header.split('=', 1)
                if units.strip().lower() != 'bytes':
                    raise ValueError(f"Unsupported range unit: {units}")

                # 解析Range范围
                start_text, end_text = byte_range.split('-', 1)
                if start_text:
                    start = int(start_text)
                    end = int(end_text) if end_text else file_size - 1
                else:
                    suffix_length = int(end_text)
                    start = max(file_size - suffix_length, 0)
                    end = file_size - 1

                # 验证范围有效性
                if start >= file_size or end < start:
                    logger.warning(
                        "Ignoring invalid range for %s: %s (file_size=%s)",
                        full_path,
                        range_header,
                        file_size,
                    )
                    start = 0
                    end = file_size - 1
                else:
                    end = min(end, file_size - 1)

                # 读取指定范围的数据
                length = end - start + 1
                with open(full_path, 'rb') as f:
                    f.seek(start)
                    data = f.read(length)

                # 返回206 Partial Content响应
                response = Response(data, 206, mimetype=mimetype, direct_passthrough=True)
                response.headers['Accept-Ranges'] = 'bytes'
                response.headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
                response.headers['Content-Length'] = str(length)
                response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
                response.headers['Pragma'] = 'no-cache'
                return response
            except Exception as range_error:
                logger.warning(f"Failed to process range request {range_header}: {range_error}")

        # 普通文件下载
        response = send_file(
            str(full_path),
            mimetype=mimetype,
            as_attachment=False,
            download_name=full_path.name,
        )
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Type'] = mimetype
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        return response
    except Exception as e:
        logger.error(f"Failed to serve video: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/validate_alignment', methods=['POST'])
def validate_alignment():
    """验证视频和轨迹数据的对应关系"""
    try:
        data = request.json or {}
        
        # 验证必要参数
        for field in ['dataset_root', 'episode_index']:
            if field not in data:
                return jsonify({'success': False, 'error': f'缺少必要参数: {field}'}), 400
        
        # 创建截断器实例用于验证
        truncator = VideoTruncator(
            dataset_root=data['dataset_root'],
            episode_index=data['episode_index']
        )
        
        # 执行验证
        result = truncator.validate_video_trajectory_alignment()
        
        return jsonify({
            'success': result['valid'],
            'validation_result': result
        })
    except Exception as e:
        logger.error(f"Failed to validate alignment: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/smooth_actions', methods=['POST'])
def smooth_actions():
    """对数据集的action列进行平滑处理"""
    try:
        data = request.json or {}
        
        # 验证必要参数
        if 'dataset_root' not in data:
            return jsonify({'success': False, 'error': '缺少dataset_root参数'}), 400

        # 更新处理状态
        processing_status.update({
            'is_processing': True,
            'progress': 0,
            'message': '准备平滑处理...',
            'error': None,
        })

        # 创建平滑器实例
        smoother = ActionSmoother(
            dataset_root=data['dataset_root'],
            method=data.get('method', 'savgol'),
            window=int(data.get('window', 7)),
            polyorder=int(data.get('polyorder', 2)),
            sigma=float(data.get('sigma', 2.0)),
            cols=data.get('cols', ['action'])
        )
        
        # 更新进度
        processing_status['progress'] = 10
        processing_status['message'] = '执行平滑处理...'
        
        # 执行平滑处理
        dry_run = data.get('dry_run', False)
        result = smoother.smooth_dataset(dry_run=dry_run)
        
        # 更新处理状态
        processing_status['is_processing'] = False
        processing_status['progress'] = 100
        processing_status['message'] = '平滑处理完成'
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to smooth actions: {e}")
        processing_status['is_processing'] = False
        processing_status['error'] = str(e)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health')
def health():
    """健康检查接口"""
    return jsonify({'status': 'healthy', 'service': 'video_truncator'})


def main():
    """启动服务器"""
    parser = argparse.ArgumentParser(description='Video truncator server')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()

    logger.info(f"Starting video truncator server at http://{args.host}:{args.port}")
    logger.info("Core processing module: video_processor.py")
    logger.info("Web interface: video_truncator.html")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()