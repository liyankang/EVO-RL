#!/usr/bin/env python3
"""
视频轨迹截断工具服务器
提供Web界面和API来截断视频和对应的轨迹数据
支持 LeRobot v2.1 格式
"""

import os
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
import subprocess
import tempfile

import pandas as pd
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量存储处理状态
processing_status = {
    'is_processing': False,
    'progress': 0,
    'message': '',
    'error': None
}


class DatasetLoader:
    """数据集加载器，用于获取数据集信息 (LeRobot v2.1 格式)"""
    
    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.data_dir = self.dataset_root / "data" / "chunk-000"
        # LeRobot v2.1 格式: videos/chunk-000/{camera_name}/episode_XXXXXX.mp4
        self.video_dir = self.dataset_root / "videos" / "chunk-000"
        self.meta_dir = self.dataset_root / "meta"
        
    def validate_dataset(self):
        """验证数据集目录结构"""
        errors = []
        
        if not self.dataset_root.exists():
            errors.append(f"数据集根目录不存在: {self.dataset_root}")
            return False, errors
        
        if not self.data_dir.exists():
            errors.append(f"数据目录不存在: {self.data_dir}")
        
        if not self.video_dir.exists():
            errors.append(f"视频目录不存在: {self.video_dir}")
        
        if not self.meta_dir.exists():
            errors.append(f"元数据目录不存在: {self.meta_dir}")
        
        return len(errors) == 0, errors
    
    def get_episodes_list(self):
        """获取所有episode的信息 (LeRobot v2.1 格式)"""
        try:
            # 读取info.json获取基本信息
            info_path = self.meta_dir / "info.json"
            if not info_path.exists():
                raise FileNotFoundError(f"info.json不存在: {info_path}")
            
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            # 获取episode信息
            episodes = []
            
            # 从data目录读取parquet文件 (LeRobot v2.1 格式: episode_XXXXXX.parquet)
            parquet_files = sorted(self.data_dir.glob("episode_*.parquet"))
            
            if parquet_files:
                # 读取所有parquet文件获取episode信息
                all_data = []
                for pq_file in parquet_files:
                    df = pd.read_parquet(pq_file)
                    all_data.append(df)
                
                # 合并所有数据
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # 按episode分组
                grouped = combined_df.groupby('episode_index')
                
                for episode_index, group in grouped:
                    # 获取该episode的视频文件 (LeRobot v2.1 格式: videos/chunk-000/{camera_name}/episode_XXXXXX.mp4)
                    primary_video_path = self.video_dir / "observation.images.primary" / f"episode_{episode_index:06d}.mp4"
                    wrist_video_path = self.video_dir / "observation.images.wrist" / f"episode_{episode_index:06d}.mp4"
                    
                    # 检查视频文件是否存在
                    primary_exists = primary_video_path.exists()
                    wrist_exists = wrist_video_path.exists()
                    
                    episode_info = {
                        'episode_index': int(episode_index),
                        'length': len(group),
                        'primary_video_exists': primary_exists,
                        'wrist_video_exists': wrist_exists,
                        'primary_video_path': str(primary_video_path) if primary_exists else None,
                        'wrist_video_path': str(wrist_video_path) if wrist_exists else None,
                        'start_timestamp': float(group['timestamp'].iloc[0]) if 'timestamp' in group.columns else 0,
                        'end_timestamp': float(group['timestamp'].iloc[-1]) if 'timestamp' in group.columns else 0
                    }
                    episodes.append(episode_info)
            
            # 按episode_index排序
            episodes.sort(key=lambda x: x['episode_index'])
            
            return {
                'success': True,
                'dataset_info': {
                    'total_episodes': info.get('total_episodes', len(episodes)),
                    'total_frames': info.get('total_frames', 0),
                    'fps': info.get('fps', 30),
                    'robot_type': info.get('robot_type', 'unknown'),
                    'cameras': list(info.get('features', {}).keys())
                },
                'episodes': episodes
            }
            
        except Exception as e:
            logger.error(f"获取episodes列表失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_video_info(self, video_path):
        """获取视频文件信息"""
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                return {'success': False, 'error': f'视频文件不存在: {video_path}'}
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {'success': False, 'error': f'无法打开视频文件: {video_path}'}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'success': True,
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"获取视频信息失败: {e}")
            return {'success': False, 'error': str(e)}


class VideoTruncator:
    """视频和轨迹数据截断器 (LeRobot v2.1 格式)"""
    
    def __init__(self, dataset_root, episode_index, truncate_frame, discard_mode='after'):
        self.dataset_root = Path(dataset_root)
        self.episode_index = int(episode_index)
        self.truncate_frame = truncate_frame
        self.discard_mode = discard_mode  # 'before' = 丢弃前面保留后面, 'after' = 丢弃后面保留前面
        
        # 设置路径 (LeRobot v2.1 格式)
        self.data_dir = self.dataset_root / "data" / "chunk-000"
        self.video_dir = self.dataset_root / "videos" / "chunk-000"  # v2.1 格式: videos/chunk-000/{camera_name}/
        self.meta_dir = self.dataset_root / "meta"
        
    def validate_paths(self):
        """验证路径是否存在"""
        errors = []
        
        if not self.dataset_root.exists():
            errors.append(f"数据集根目录不存在: {self.dataset_root}")
        
        # 检查parquet文件 (LeRobot v2.1 格式: episode_XXXXXX.parquet)
        parquet_files = list(self.data_dir.glob("episode_*.parquet"))
        if not parquet_files:
            errors.append(f"未找到轨迹数据文件: {self.data_dir}/episode_*.parquet")
        
        # 检查至少有一个视频文件 (LeRobot v2.1 格式: videos/chunk-000/{camera_name}/episode_XXXXXX.mp4)
        has_video = False
        if self.video_dir.exists():
            for camera_dir in self.video_dir.iterdir():
                if camera_dir.is_dir():
                    video_file = camera_dir / f"episode_{self.episode_index:06d}.mp4"
                    if video_file.exists():
                        has_video = True
                        break
        
        if not has_video:
            errors.append(f"未找到episode {self.episode_index}的视频文件")
        
        if not self.meta_dir.exists():
            errors.append(f"元数据目录不存在: {self.meta_dir}")
        
        return errors
    
    def truncate_all_videos(self):
        """截断该episode的所有相机视频文件"""
        if not self.video_dir.exists():
            logger.warning(f"视频目录不存在: {self.video_dir}")
            return {}
        
        processed_videos = {}
        
        # 遍历所有相机目录 (LeRobot v2.1 格式: videos/chunk-000/{camera_name}/episode_XXXXXX.mp4)
        for camera_subdir in self.video_dir.iterdir():
            if not camera_subdir.is_dir():
                continue
                
            camera_name = camera_subdir.name
            video_path = camera_subdir / f"episode_{self.episode_index:06d}.mp4"
            
            if not video_path.exists():
                logger.info(f"视频文件不存在，跳过: {camera_name}/{video_path.name}")
                continue
            
            try:
                # 使用FFmpeg截断视频（更好的浏览器兼容性）
                logger.info(f"处理视频 {camera_name}: {video_path.name}")
                
                # 先获取视频信息
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                logger.info(f"视频信息: FPS={fps}, 分辨率={width}x{height}, 总帧数={total_frames}")
                
                # 使用FFmpeg截断视频到指定帧数
                temp_output = video_path.with_suffix('.temp.mp4')
                
                # 根据discard_mode决定截断策略
                if self.discard_mode == 'after':
                    # 丢弃后面，保留前面：从开头截断到指定帧数
                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-i', str(video_path),           # 输入文件
                        '-vframes', str(self.truncate_frame),  # 精确截断帧数
                        '-c:v', 'libx264',               # 使用H.264编码器
                        '-preset', 'fast',               # 编码速度
                        '-crf', '23',                    # 质量控制（18-28，值越低质量越高）
                        '-c:a', 'copy',                  # 音频直接复制
                        '-r', str(fps),                  # 保持原始帧率
                        '-movflags', '+faststart',       # 优化网络播放
                        '-pix_fmt', 'yuv420p',           # 确保色彩格式兼容性
                        '-y',                            # 覆盖输出文件
                        str(temp_output)
                    ]
                elif self.discard_mode == 'before':
                    # 丢弃前面，保留后面：从指定帧数开始截断到结尾
                    start_frame = total_frames - self.truncate_frame
                    if start_frame < 0:
                        start_frame = 0
                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-i', str(video_path),           # 输入文件
                        '-ss', str(start_frame),         # 从指定帧开始
                        '-c:v', 'libx264',               # 使用H.264编码器
                        '-preset', 'fast',               # 编码速度
                        '-crf', '23',                    # 质量控制（18-28，值越低质量越高）
                        '-c:a', 'copy',                  # 音频直接复制
                        '-r', str(fps),                  # 保持原始帧率
                        '-movflags', '+faststart',       # 优化网络播放
                        '-pix_fmt', 'yuv420p',           # 确保色彩格式兼容性
                        '-y',                            # 覆盖输出文件
                        str(temp_output)
                    ]
                else:
                    raise ValueError(f"不支持的discard_mode: {self.discard_mode}")
                
                # 执行FFmpeg命令
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg执行失败: {result.stderr}")
                    raise RuntimeError(f"FFmpeg处理失败: {result.stderr}")
                
                # 检查输出文件是否成功创建
                if not temp_output.exists():
                    raise RuntimeError("输出文件未创建")
                
                # 验证截断后的视频帧数
                cap_verify = cv2.VideoCapture(str(temp_output))
                truncated_frames = int(cap_verify.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_verify.release()
                
                # 替换原文件
                shutil.move(str(temp_output), str(video_path))
                
                processed_videos[camera_name] = {
                    'original_frames': total_frames,
                    'truncated_frames': truncated_frames,
                    'path': str(video_path)
                }
                
                # 更新进度（只对第一个相机更新进度显示）
                if len(processed_videos) == 1:  # 第一个成功处理
                    progress = 50  # 视频处理占50%进度
                    processing_status['progress'] = progress
                    processing_status['message'] = f"正在处理视频 [{camera_name}]: 完成 {truncated_frames}/{self.truncate_frame} 帧"
                
                logger.info(f"视频 {camera_name} 截断完成: {total_frames} -> {truncated_frames} 帧")
                
            except Exception as e:
                logger.error(f"截断视频 {camera_name} 时出错: {e}")
                processed_videos[camera_name] = {'error': str(e)}
        
        return processed_videos
    
    def truncate_trajectory(self):
        """截断轨迹数据文件 (LeRobot v2.1 格式: episode_XXXXXX.parquet)"""
        parquet_files = sorted(self.data_dir.glob("episode_*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"未找到轨迹数据文件: {self.data_dir}/episode_*.parquet")
        
        # 查找目标episode的文件
        target_file = self.data_dir / f"episode_{self.episode_index:06d}.parquet"
        
        if not target_file.exists():
            raise FileNotFoundError(f"未找到episode {self.episode_index}的轨迹数据文件: {target_file}")
        
        # 读取目标文件
        episode_data = pd.read_parquet(target_file)
        original_length = len(episode_data)
        logger.info(f"原始轨迹长度: {original_length} 帧")
        
        # 截断到指定帧数
        if self.truncate_frame >= original_length:
            logger.warning(f"截断帧数 {self.truncate_frame} 大于等于原始长度 {original_length}，不进行截断")
            return original_length
        
        # 根据discard_mode决定截断策略
        if self.discard_mode == 'after':
            # 丢弃后面，保留前面：从开头截断到指定帧数
            truncated_episode_data = episode_data.iloc[:self.truncate_frame]
            logger.info(f"丢弃后面，保留前面: 保留前 {self.truncate_frame} 帧")
        elif self.discard_mode == 'before':
            # 丢弃前面，保留后面：从指定帧数开始到结尾
            start_index = original_length - self.truncate_frame
            if start_index < 0:
                start_index = 0
            truncated_episode_data = episode_data.iloc[start_index:]
            logger.info(f"丢弃前面，保留后面: 保留后 {self.truncate_frame} 帧 (从索引 {start_index} 开始)")
        else:
            raise ValueError(f"不支持的discard_mode: {self.discard_mode}")
        
        # 重新计算索引
        truncated_episode_data = truncated_episode_data.reset_index(drop=True)
        truncated_episode_data['index'] = range(len(truncated_episode_data))
        truncated_episode_data['frame_index'] = range(len(truncated_episode_data))
        
        # 保存截断后的数据
        truncated_episode_data.to_parquet(target_file, index=False)
        
        logger.info(f"轨迹数据截断完成: {original_length} -> {len(truncated_episode_data)} 帧")
        return len(truncated_episode_data)
    
    def update_meta_files(self, new_length):
        """更新元数据文件 (LeRobot v2.1 格式)"""
        
        # 更新info.json
        info_path = self.meta_dir / "info.json"
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            # 重新计算总帧数
            parquet_files = sorted(self.data_dir.glob("episode_*.parquet"))
            total_frames = sum(len(pd.read_parquet(pq)) for pq in parquet_files)
            
            info['total_frames'] = total_frames
            # total_episodes 从文件数量计算
            info['total_episodes'] = len(parquet_files)
            
            # 保存更新
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"更新info.json完成: 总帧数={total_frames}")
    
    def truncate(self):
        """执行完整的截断流程"""
        logger.info(f"开始截断流程: episode={self.episode_index}, frame={self.truncate_frame}")
        
        # 验证路径
        errors = self.validate_paths()
        if errors:
            raise ValueError("\n".join(errors))
        
        try:
            # 截断所有相机的视频
            processing_status['message'] = "正在截断视频..."
            processed_videos = self.truncate_all_videos()
            
            # 截断轨迹数据
            processing_status['message'] = "正在截断轨迹数据..."
            processing_status['progress'] = 60  # 轨迹处理从60%开始
            trajectory_frames = self.truncate_trajectory()
            
            # 更新元数据
            processing_status['message'] = "正在更新元数据..."
            processing_status['progress'] = 90  # 元数据更新从90%开始
            self.update_meta_files(trajectory_frames)
            
            processing_status['progress'] = 100
            processing_status['message'] = "截断完成！"
            
            # 统计处理的视频数量
            successful_videos = {k: v for k, v in processed_videos.items() if 'error' not in v}
            failed_videos = {k: v for k, v in processed_videos.items() if 'error' in v}
            
            logger.info(f"截断流程完成: 处理了{len(successful_videos)}个视频, 轨迹={trajectory_frames}帧")
            if failed_videos:
                logger.warning(f"部分视频处理失败: {list(failed_videos.keys())}")
            
            return {
                'success': True,
                'processed_videos': processed_videos,
                'successful_videos_count': len(successful_videos),
                'failed_videos_count': len(failed_videos),
                'trajectory_frames': trajectory_frames,
                'episode_number': self.episode_index
            }
            
        except Exception as e:
            logger.error(f"截断过程出错: {e}")
            processing_status['error'] = str(e)
            raise


# Flask路由
@app.route('/')
def index():
    """提供HTML界面"""
    return send_from_directory('.', 'video_truncator.html')


@app.route('/load_dataset', methods=['POST'])
def load_dataset():
    """加载数据集信息"""
    try:
        data = request.json
        dataset_root = data.get('dataset_root')
        
        if not dataset_root:
            return jsonify({
                'success': False,
                'error': '缺少dataset_root参数'
            }), 400
        
        loader = DatasetLoader(dataset_root)
        is_valid, errors = loader.validate_dataset()
        
        if not is_valid:
            return jsonify({
                'success': False,
                'error': '数据集验证失败',
                'details': errors
            }), 400
        
        result = loader.get_episodes_list()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"加载数据集时出错: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/get_video_info', methods=['POST'])
def get_video_info():
    """获取视频文件信息"""
    try:
        data = request.json
        video_path = data.get('video_path')
        dataset_root = data.get('dataset_root')
        
        if not video_path or not dataset_root:
            return jsonify({
                'success': False,
                'error': '缺少必需参数'
            }), 400
        
        loader = DatasetLoader(dataset_root)
        result = loader.get_video_info(video_path)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"获取视频信息时出错: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/truncate', methods=['POST'])
def truncate():
    """处理截断请求"""
    try:
        data = request.json
        
        # 验证必需参数
        required_fields = ['dataset_root', 'episode_index', 'truncate_frame']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'缺少必需参数: {field}'
                }), 400
        
        # 重置处理状态
        processing_status.update({
            'is_processing': True,
            'progress': 0,
            'message': '准备处理...',
            'error': None
        })
        
        # 创建截断器，支持discard_mode参数
        discard_mode = data.get('discard_mode', 'after')  # 默认为after（丢弃后面，保留前面）
        truncator = VideoTruncator(
            dataset_root=data['dataset_root'],
            episode_index=data['episode_index'],
            truncate_frame=int(data['truncate_frame']),
            discard_mode=discard_mode
        )
        
        # 执行截断
        result = truncator.truncate()
        
        processing_status['is_processing'] = False
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"处理截断请求时出错: {e}")
        processing_status['is_processing'] = False
        processing_status['error'] = str(e)
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/status')
def status():
    """获取处理状态"""
    return jsonify(processing_status)


@app.route('/video/<path:filepath>')
def serve_video(filepath):
    """提供视频文件访问，支持范围请求"""
    try:
        # 解码URL编码的路径
        from urllib.parse import unquote
        from flask import send_file, request as flask_request
        
        filepath = unquote(filepath)
        
        # 确保文件路径是安全的，并添加开头的/如果缺失
        filepath = filepath.replace('..', '').replace('//', '/')
        if not filepath.startswith('/'):
            filepath = '/' + filepath
        
        # 构建完整路径
        full_path = Path(filepath)
        
        if not full_path.exists():
            logger.error(f"视频文件不存在: {full_path}")
            return jsonify({'error': '视频文件不存在'}), 404
        
        if not full_path.is_file():
            logger.error(f"不是有效的视频文件: {full_path}")
            return jsonify({'error': '不是有效的视频文件'}), 400
        
        # 检查文件扩展名
        file_extension = full_path.suffix.lower()
        if file_extension not in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            logger.error(f"不支持的视频格式: {file_extension}")
            return jsonify({'error': f'不支持的视频格式: {file_extension}'}), 400
        
        # 设置正确的MIME类型
        mimetype = 'video/mp4' if file_extension == '.mp4' else 'video/webm'
        
        # 使用 send_file 并添加适当的响应头
        response = send_file(
            str(full_path),
            mimetype=mimetype,
            as_attachment=False,
            download_name=full_path.name
        )
        
        # 添加视频流相关的响应头
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Type'] = mimetype
        response.headers['Cache-Control'] = 'no-cache'
        
        logger.info(f"提供视频文件: {full_path} ({full_path.stat().st_size} bytes)")
        
        return response
        
    except Exception as e:
        logger.error(f"提供视频文件时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """健康检查"""
    return jsonify({'status': 'healthy'})


def main():
    """启动服务器"""
    import argparse
    
    parser = argparse.ArgumentParser(description='视频轨迹截断工具服务器 (支持 LeRobot v2.1)')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    logger.info(f"启动视频轨迹截断工具服务器 (LeRobot v2.1): http://{args.host}:{args.port}")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()