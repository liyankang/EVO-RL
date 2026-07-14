"""状态管理模块，处理应用的全局状态"""

import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class StateManager:
    """管理应用的全局状态"""
    
    @staticmethod
    def init():
        """初始化 session state"""
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
    
    @staticmethod
    def set_dataset_path(path: Optional[str]):
        """设置数据集路径"""
        st.session_state.dataset_path = path
    
    @staticmethod
    def get_dataset_path() -> Optional[str]:
        """获取数据集路径"""
        return st.session_state.dataset_path
    
    @staticmethod
    def set_dataset_info(info: Optional[Dict[str, Any]]):
        """设置数据集信息"""
        st.session_state.dataset_info = info
    
    @staticmethod
    def get_dataset_info() -> Optional[Dict[str, Any]]:
        """获取数据集信息"""
        return st.session_state.dataset_info
    
    @staticmethod
    def set_episodes_list(episodes: Optional[list]):
        """设置 episode 列表"""
        st.session_state.episodes_list = episodes
    
    @staticmethod
    def get_episodes_list() -> Optional[list]:
        """获取 episode 列表"""
        return st.session_state.episodes_list
    
    @staticmethod
    def set_current_episode(episode_index: Optional[int]):
        """设置当前选中的 episode"""
        st.session_state.current_episode = episode_index
    
    @staticmethod
    def get_current_episode() -> Optional[int]:
        """获取当前选中的 episode"""
        return st.session_state.current_episode
    
    @staticmethod
    def is_dataset_loaded() -> bool:
        """检查是否已加载数据集"""
        return st.session_state.dataset_path is not None
    
    @staticmethod
    def clear():
        """清除所有状态"""
        st.session_state.dataset_path = None
        st.session_state.dataset_info = None
        st.session_state.episodes_list = None
        st.session_state.current_episode = None


def get_dataset_info() -> Optional[Dict[str, Any]]:
    """便捷函数：获取数据集信息"""
    return StateManager.get_dataset_info()


def load_dataset(dataset_path: str, dataset_loader) -> bool:
    """加载数据集并更新状态
    
    Args:
        dataset_path: 数据集路径
        dataset_loader: DatasetLoader 实例
        
    Returns:
        是否加载成功
    """
    try:
        # 验证数据集结构
        is_valid, errors = dataset_loader.validate_dataset()
        if not is_valid:
            logger.error(f"数据集验证失败: {errors}")
            return False
        
        # 获取 episode 列表
        episodes_result = dataset_loader.get_episodes_list()
        if not episodes_result.get('success'):
            logger.error(f"获取 episode 列表失败: {episodes_result.get('error')}")
            return False
        
        # 更新状态
        StateManager.set_dataset_path(dataset_path)
        StateManager.set_dataset_info(episodes_result['dataset_info'])
        StateManager.set_episodes_list(episodes_result['episodes'])
        
        logger.info(f"成功加载数据集: {dataset_path}")
        return True
        
    except Exception as e:
        logger.error(f"加载数据集时出错: {e}")
        return False


def clear_dataset():
    """清除数据集状态"""
    StateManager.clear()
    logger.info("已清除数据集状态")