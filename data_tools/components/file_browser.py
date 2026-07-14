"""交互式文件浏览器组件"""

import streamlit as st
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)


class FileBrowser:
    """交互式文件浏览器组件"""
    
    def __init__(self, root_path: Optional[Path] = None, max_depth: int = 5):
        """初始化文件浏览器
        
        Args:
            root_path: 根目录路径，默认为当前工作目录
            max_depth: 最大浏览深度
        """
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self.max_depth = max_depth
        self.current_path = self.root_path
        
    def get_directory_tree(self, path: Path, depth: int = 0) -> List[dict]:
        """获取目录树结构
        
        Args:
            path: 当前路径
            depth: 当前深度
            
        Returns:
            目录树列表
        """
        if depth > self.max_depth:
            return []
        
        items = []
        try:
            # 获取所有子目录
            for item in sorted(path.iterdir()):
                if item.is_dir() and not item.name.startswith('.'):
                    is_valid = self._is_dataset_directory(item)
                    
                    items.append({
                        'name': item.name,
                        'path': str(item),
                        'is_dir': True,
                        'is_dataset': is_valid,
                        'depth': depth
                    })
                    
                    # 递归获取子目录
                    items.extend(self.get_directory_tree(item, depth + 1))
                    
        except PermissionError:
            logger.warning(f"没有权限访问目录: {path}")
        except Exception as e:
            logger.error(f"读取目录时出错 {path}: {e}")
        
        return items
    
    def _is_dataset_directory(self, path: Path) -> bool:
        """检查目录是否是有效的数据集目录
        
        Args:
            path: 目录路径
            
        Returns:
            是否是数据集目录
        """
        try:
            # 导入验证函数
            import sys
            from pathlib import Path
            
            # 添加父目录到路径以导入 validators
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            
            from utils.validators import is_valid_dataset_path
            
            return is_valid_dataset_path(path)
            
        except Exception as e:
            logger.error(f"验证数据集目录时出错: {e}")
            return False
    
    def get_breadcrumbs(self, path: Path) -> str:
        """获取面包屑导航
        
        Args:
            path: 当前路径
            
        Returns:
            面包屑字符串
        """
        parts = path.parts
        breadcrumbs = []
        
        # 相对于根路径的面包屑
        try:
            rel_parts = path.relative_to(self.root_path).parts
            for i, part in enumerate(rel_parts):
                level_path = Path(*rel_parts[:i+1])
                full_path = self.root_path / level_path
                breadcrumbs.append((part, str(full_path)))
        except ValueError:
            # 路径不在根路径下
            breadcrumbs.append((path.name, str(path)))
        
        return breadcrumbs
    
    def render(self, show_breadcrumbs: bool = True) -> Optional[Path]:
        """渲染文件浏览器界面
        
        Args:
            show_breadcrumbs: 是否显示面包屑导航
            
        Returns:
            用户选择的数据集路径，如果没有选择则返回 None
        """
        st.subheader("📁 数据集目录浏览器")
        
        # 显示面包屑导航
        if show_breadcrumbs:
            breadcrumbs = self.get_breadcrumbs(self.current_path)
            breadcrumb_html = " > ".join([f"<span style='color: #1f77b4;'>{name}</span>" for name, _ in breadcrumbs])
            st.markdown(f"当前路径: {breadcrumb_html}", unsafe_allow_html=True)
        
        # 获取当前目录的内容
        directories = []
        datasets = []
        
        try:
            for item in sorted(self.current_path.iterdir()):
                if item.is_dir() and not item.name.startswith('.'):
                    is_dataset = self._is_dataset_directory(item)
                    if is_dataset:
                        datasets.append(item)
                    else:
                        directories.append(item)
        except PermissionError:
            st.error("没有权限访问当前目录")
            return None
        
        # 导航控制
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if self.current_path != self.root_path:
                if st.button("⬆️ 返回上级"):
                    self.current_path = self.current_path.parent
                    st.rerun()
            
            if st.button("🏠 返回根目录"):
                self.current_path = self.root_path
                st.rerun()
        
        with col2:
            # 显示目录导航
            if directories:
                st.write("**📂 子目录:**")
                for i, directory in enumerate(directories):
                    if st.button(f"📁 {directory.name}/", key=f"dir_{i}"):
                        self.current_path = directory
                        st.rerun()
            
            # 显示数据集目录
            if datasets:
                st.write("**🎯 有效数据集:**")
                for dataset in datasets:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        if st.button(f"✅ {dataset.name}", key=f"dataset_{dataset.name}"):
                            st.session_state.selected_dataset = dataset
                            st.success(f"已选择数据集: {dataset.name}")
                            return dataset
                    with col_b:
                        # 显示一些基本信息
                        try:
                            meta_file = dataset / "meta" / "info.json"
                            if meta_file.exists():
                                import json
                                with open(meta_file, 'r') as f:
                                    info = json.load(f)
                                episodes = info.get('total_episodes', 'N/A')
                                st.caption(f"{episodes} episodes")
                        except:
                            pass
            else:
                st.info("📭 当前目录下没有有效的数据集")
        
        return None
    
    def render_simple(self) -> Optional[Path]:
        """渲染简化的文件选择器
        
        Returns:
            用户选择的数据集路径
        """
        st.subheader("📁 选择数据集目录")
        
        # 让用户输入或选择路径
        path_input = st.text_input("输入数据集路径", value=str(self.current_path))
        
        if st.button("浏览目录"):
            self.current_path = Path(path_input)
        
        # 验证路径
        if path_input:
            path = Path(path_input)
            if path.exists() and path.is_dir():
                # 检查是否是数据集目录
                if self._is_dataset_directory(path):
                    st.success(f"✅ 检测到有效数据集: {path.name}")
                    
                    # 显示基本信息
                    try:
                        meta_file = path / "meta" / "info.json"
                        if meta_file.exists():
                            import json
                            with open(meta_file, 'r') as f:
                                info = json.load(f)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Episodes", info.get('total_episodes', 'N/A'))
                            with col2:
                                st.metric("总帧数", info.get('total_frames', 'N/A'))
                            with col3:
                                st.metric("帧率", f"{info.get('fps', 30)} fps")
                    except Exception as e:
                        st.warning(f"无法读取数据集信息: {e}")
                    
                    if st.button("加载此数据集", key="load_dataset"):
                        return path
                else:
                    st.warning("⚠️ 该目录不是有效的数据集目录")
                    st.info("数据集目录应包含: data/, meta/, videos/ 子目录")
                    
                    # 显示目录内容
                    st.write("**目录内容:**")
                    try:
                        for item in sorted(path.iterdir()):
                            icon = "📁" if item.is_dir() else "📄"
                            st.write(f"{icon} {item.name}")
                    except PermissionError:
                        st.error("没有权限访问该目录")
            else:
                st.error("❌ 路径不存在或不是目录")
        
        return None
    
    def render_selectbox_navigation(self) -> Optional[Path]:
        """使用 st.selectbox 和 os.scandir 实现的动态文件夹选择器
        
        Returns:
            用户选择的数据集路径，如果没有选择则返回 None
        """
        # 导入状态管理
        import sys
        from pathlib import Path
        
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        from utils.state_manager import StateManager, load_dataset
        from utils.dataset_loader import DatasetLoader
        st.subheader("📂 动态文件夹导航")
        
        # 初始化当前路径状态
        if 'current_path' not in st.session_state:
            st.session_state.current_path = self.root_path
        
        # 显示当前路径
        st.markdown(f"**当前路径:** `{st.session_state.current_path}`")
        st.markdown("---")
        
        # 获取当前路径下的子文件夹
        subfolders = []
        try:
            with os.scandir(st.session_state.current_path) as entries:
                for entry in entries:
                    if entry.is_dir() and not entry.name.startswith('.'):
                        subfolders.append(entry.name)
        except PermissionError:
            st.error("❌ 没有权限访问此目录")
            return None
        except Exception as e:
            st.error(f"❌ 读取目录时出错: {e}")
            return None
        
        # 检查当前路径是否是有效的数据集目录
        current_path_obj = Path(st.session_state.current_path)
        is_dataset = self._is_dataset_directory(current_path_obj)
        
        # 按钮控制区
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 返回上级目录按钮
            if st.session_state.current_path != self.root_path:
                if st.button("⬆️ 返回上级目录", key="go_up"):
                    st.session_state.current_path = str(Path(st.session_state.current_path).parent)
                    st.rerun()
            else:
                st.write("📍 已在根目录")
        
        with col2:
            # 返回根目录按钮
            if st.session_state.current_path != self.root_path:
                if st.button("🏠 返回根目录", key="go_root"):
                    st.session_state.current_path = self.root_path
                    st.rerun()
        
        with col3:
            # 手动输入路径
            manual_path = st.text_input("或手动输入路径", placeholder="输入路径按回车", key="manual_path")
            if manual_path and Path(manual_path).exists() and Path(manual_path).is_dir():
                st.session_state.current_path = manual_path
                st.rerun()
        
        st.markdown("---")
        
        # 显示子文件夹选择器
        if subfolders:
            st.markdown("### 📁 子文件夹列表")
            
            # 创建带有图标的选项列表
            folder_options = []
            for folder in sorted(subfolders):
                folder_path = Path(st.session_state.current_path) / folder
                # 检查是否是数据集目录
                folder_is_dataset = self._is_dataset_directory(folder_path)
                icon = "✅" if folder_is_dataset else "📁"
                folder_options.append(f"{icon} {folder}")
            
            # 添加特殊选项
            folder_options.insert(0, "📍 停留在当前目录")
            
            # 使用 selectbox 显示文件夹列表
            selected = st.selectbox(
                "选择要进入的文件夹",
                folder_options,
                index=0,
                key=f"folder_select_{st.session_state.current_path}"
            )
            
            # 处理选择
            if selected and selected != "📍 停留在当前目录":
                # 提取文件夹名称（去掉图标）
                folder_name = selected.split(" ", 1)[1]
                new_path = Path(st.session_state.current_path) / folder_name
                
                # 更新当前路径
                st.session_state.current_path = str(new_path)
                st.rerun()
        
        else:
            st.info("📭 当前目录下没有子文件夹")
        
        st.markdown("---")
        
        # 显示当前目录信息
        st.markdown("### 📋 当前目录信息")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.metric("子文件夹数量", len(subfolders))
        
        with col_info2:
            try:
                all_items = len(list(Path(st.session_state.current_path).iterdir()))
                st.metric("总项目数", all_items)
            except:
                st.metric("总项目数", "N/A")
        
        with col_info3:
            status = "✅ 数据集目录" if is_dataset else "📁 普通目录"
            st.metric("目录类型", status)
        
        # 如果当前目录是数据集目录，显示加载选项
        if is_dataset:
            st.markdown("---")
            st.success("🎯 当前目录是有效的数据集目录")
            
            # 显示数据集详细信息
            try:
                meta_file = current_path_obj / "meta" / "info.json"
                if meta_file.exists():
                    import json
                    with open(meta_file, 'r') as f:
                        info = json.load(f)
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Episodes", info.get('total_episodes', 'N/A'))
                    with col_b:
                        st.metric("总帧数", info.get('total_frames', 'N/A'))
                    with col_c:
                        st.metric("帧率", f"{info.get('fps', 30)} fps")
            except Exception as e:
                st.warning(f"无法读取数据集信息: {e}")
            
            # 加载按钮 - 直接执行加载操作
            if st.button("🚀 加载此数据集", type="primary", key="load_current_dataset"):
                with st.spinner("📊 正在加载数据集..."):
                    try:
                        # 创建数据集加载器
                        loader = DatasetLoader(current_path_obj)
                        
                        # 加载数据集
                        success = load_dataset(str(current_path_obj), loader)
                        
                        if success:
                            st.success("🎉 数据集加载成功！")
                            st.rerun()
                        else:
                            st.error("❌ 数据集加载失败")
                            
                    except Exception as e:
                        st.error(f"❌ 加载数据集时出错: {e}")
                        import traceback
                        st.error(traceback.format_exc())
        
        return None


def create_file_browser(root_path: Optional[Path] = None, mode: str = "advanced") -> FileBrowser:
    """创建文件浏览器实例
    
    Args:
        root_path: 根目录路径
        mode: 浏览器模式 ('advanced' 或 'simple')
        
    Returns:
        FileBrowser 实例
    """
    return FileBrowser(root_path=root_path)


def select_dataset_directory(root_path: Optional[Path] = None) -> Optional[Path]:
    """选择数据集目录的便捷函数
    
    Args:
        root_path: 根目录路径
        
    Returns:
        选择的数据集路径
    """
    browser = create_file_browser(root_path)
    return browser.render()