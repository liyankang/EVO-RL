#!/usr/bin/env python3
"""
相机诊断脚本 - 用于快速检测和解决OpenCV相机连接问题
基于ffmpeg测试结果，相机设备被识别但无法访问，可能是权限或占用问题
"""

import sys
import time
import subprocess
import os
from pathlib import Path


def check_permissions():
    """检查设备权限"""
    print("🔍 检查设备权限...")
    try:
        result = subprocess.run(['ls', '-l', '/dev/video6'], capture_output=True, text=True)
        print(f"设备权限: {result.stdout}")
        
        # 检查用户组
        user_groups = subprocess.run(['groups', os.environ.get('USER', '')], capture_output=True, text=True)
        print(f"当前用户组: {user_groups.stdout}")
        
        if 'video' not in user_groups.stdout:
            print("⚠️  警告: 用户不在video组中，需要添加权限")
            print("   运行: sudo usermod -a -G video $USER")
            print("   然后重新登录或重启系统")
            return False
        return True
    except Exception as e:
        print(f"❌ 权限检查失败: {e}")
        return False


def check_device_occupied():
    """检查设备是否被占用"""
    print("\n🔍 检查设备占用情况...")
    try:
        result = subprocess.run(['sudo', 'lsof', '/dev/video6'], capture_output=True, text=True)
        if result.stdout and 'COMMAND' not in result.stdout:
            print(f"⚠️  设备被占用:\n{result.stdout}")
            print("   建议结束占用进程或重启设备")
            return False
        elif result.stdout and 'COMMAND' in result.stdout:
            print("✅ 设备未被占用")
            return True
        else:
            print("✅ 设备未被占用")
            return True
    except Exception as e:
        print(f"⚠️  占用检查失败 (可能需要sudo权限): {e}")
        return True  # 假设未被占用


def test_with_v4l2():
    """使用v4l2工具测试"""
    print("\n🔍 使用v4l2工具测试...")
    try:
        # 尝试捕获一帧
        result = subprocess.run([
            'v4l2-ctl', '-d', '/dev/video6', 
            '--stream-mmap', '--stream-count=1', 
            '--stream-to=/tmp/test_camera.yuv'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("✅ v4l2测试成功")
            # 检查文件大小
            if os.path.exists('/tmp/test_camera.yuv'):
                size = os.path.getsize('/tmp/test_camera.yuv')
                print(f"   捕获文件大小: {size} bytes")
                if size > 0:
                    return True
            return False
        else:
            print(f"❌ v4l2测试失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ v4l2测试超时")
        return False
    except FileNotFoundError:
        print("⚠️  v4l2-ctl未安装，跳过此测试")
        return None
    except Exception as e:
        print(f"❌ v4l2测试失败: {e}")
        return False


def test_with_opencv_basic():
    """基础OpenCV测试"""
    print("\n🔍 基础OpenCV测试...")
    try:
        import cv2
        
        print(f"OpenCV版本: {cv2.__version__}")
        
        # 尝试不同的后端
        backends = [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "ANY"),
            (cv2.CAP_FFMPEG, "FFMPEG")
        ]
        
        for backend, backend_name in backends:
            print(f"   测试后端: {backend_name}")
            cap = cv2.VideoCapture(6, backend)
            
            if not cap.isOpened():
                print(f"   ❌ 无法打开设备 (后端: {backend_name})")
                continue
            
            print(f"   ✅ 设备已打开 (后端: {backend_name})")
            print(f"   后端名称: {cap.getBackendName()}")
            
            # 获取属性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"   分辨率: {width}x{height}, FPS: {fps}")
            
            # 尝试读取帧
            for attempt in range(3):
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"   ✅ 成功读取帧 (尝试 {attempt + 1}/3)")
                    print(f"   帧形状: {frame.shape}, 帧大小: {frame.size}")
                    
                    # 保存测试图像
                    cv2.imwrite('/tmp/test_camera_frame.jpg', frame)
                    print(f"   📸 测试帧已保存到 /tmp/test_camera_frame.jpg")
                    
                    cap.release()
                    return True
                else:
                    print(f"   ⚠️  读取帧失败 (尝试 {attempt + 1}/3)")
                    time.sleep(0.5)
            
            cap.release()
        
        return False
        
    except ImportError:
        print("❌ OpenCV未安装")
        return False
    except Exception as e:
        print(f"❌ OpenCV测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_opencv_advanced():
    """高级OpenCV测试 - 使用MJPEG格式"""
    print("\n🔍 高级OpenCV测试 (MJPEG格式)...")
    try:
        import cv2
        
        cap = cv2.VideoCapture(6, cv2.CAP_V4L2)
        
        if not cap.isOpened():
            print("❌ 无法打开设备")
            return False
        
        # 设置MJPEG格式
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        
        # 设置较低的分辨率和FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        # 验证设置
        actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"   设置参数: MJPG, 640x480, 15fps")
        print(f"   实际参数: {actual_fourcc_str}, {actual_width}x{actual_height}, {actual_fps}fps")
        
        # 预热
        print("   预热相机 (3秒)...")
        time.sleep(3)
        
        # 尝试读取多帧
        success_count = 0
        for i in range(10):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                success_count += 1
                if success_count == 1:
                    print(f"   ✅ 第1帧成功读取")
                    cv2.imwrite('/tmp/test_camera_mjpeg.jpg', frame)
            time.sleep(0.1)
        
        cap.release()
        
        if success_count >= 5:
            print(f"   ✅ 成功读取 {success_count}/10 帧 - 相机工作正常!")
            return True
        else:
            print(f"   ⚠️  仅成功读取 {success_count}/10 帧 - 相机不稳定")
            return False
            
    except Exception as e:
        print(f"❌ 高级测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def provide_solutions():
    """提供解决方案建议"""
    print("\n" + "="*60)
    print("🔧 解决方案建议")
    print("="*60)
    
    print("\n1. 权限问题:")
    print("   sudo usermod -a -G video $USER")
    print("   # 然后重新登录或重启系统")
    
    print("\n2. 修改相机配置 (推荐):")
    print("""
camera_config = {
    "index_or_path": 6,
    "fps": 15,              # 降低FPS
    "warmup_s": 5.0,        # 增加预热时间
    "width": 640,           # 降低分辨率
    "height": 480,
    "color_mode": "RGB",
    "fourcc": "MJPG",       # 使用MJPEG格式
    "backend": cv2.CAP_V4L2 # 明确指定后端
}
""")
    
    print("\n3. 检查USB连接:")
    print("   lsusb | grep -i camera")
    print("   # 确保使用USB 3.0端口 (蓝色接口)")
    
    print("\n4. 系统重启:")
    print("   sudo reboot")
    print("   # 有时需要重启来重新初始化USB设备")


def main():
    print("🎯 EVO-RL 相机诊断工具")
    print("="*60)
    
    results = {}
    
    # 运行各项测试
    results['权限'] = check_permissions()
    results['占用'] = check_device_occupied()
    results['v4l2'] = test_with_v4l2()
    results['opencv基础'] = test_with_opencv_basic()
    
    if results['opencv基础']:
        results['opencv高级'] = test_with_opencv_advanced()
    else:
        results['opencv高级'] = False
    
    # 总结结果
    print("\n" + "="*60)
    print("📊 诊断结果总结")
    print("="*60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ 通过"
        elif result is False:
            status = "❌ 失败"
        else:
            status = "⚠️  跳过"
        print(f"{test_name:12}: {status}")
    
    # 提供解决方案
    provide_solutions()
    
    # 最终建议
    print("\n" + "="*60)
    print("🎯 最终建议")
    print("="*60)
    
    if results['opencv高级']:
        print("✅ 相机工作正常，可以继续使用")
        print("   建议修改配置文件中的相机参数以提高稳定性")
    elif results['opencv基础']:
        print("⚠️  相机基本可用，但不稳定")
        print("   强烈建议使用推荐的配置参数")
    else:
        print("❌ 相机无法正常工作")
        print("   请按照上面的解决方案逐步排查问题")
    
    print("\n诊断完成!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n诊断被用户中断")
        sys.exit(1)