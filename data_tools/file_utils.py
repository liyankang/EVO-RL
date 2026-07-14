"""文件操作工具函数模块。"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any


def _json_default(value):
    """JSON序列化默认处理器，支持numpy类型。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return str(value)


def _read_json(path: Path, default: Optional[Dict] = None) -> Optional[Dict]:
    """读取JSON文件，如果文件不存在则返回默认值。"""
    if not path.exists():
        return default
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _write_json(path: Path, data: Dict):
    """写入JSON文件。"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def _read_jsonl(path: Path) -> List[Dict]:
    """读取JSONL文件。"""
    rows = []
    if not path.exists():
        return rows
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict]):
    """写入JSONL文件。"""
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + '\n')


def _flatten_dict(data: Dict, prefix: str = "") -> Dict:
    """扁平化字典结构。"""
    flat = {}
    for key, value in data.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _serialize_stats_tree(stats: Any) -> Any:
    """将numpy/标量统计值转换成JSON可写入的普通Python结构。"""
    if isinstance(stats, dict):
        return {key: _serialize_stats_tree(value) for key, value in stats.items()}
    if isinstance(stats, (list, tuple)):
        return [_serialize_stats_tree(value) for value in stats]
    if isinstance(stats, np.ndarray):
        return stats.tolist()
    if isinstance(stats, np.generic):
        return stats.item()
    if isinstance(stats, (str, int, float)):
        return stats
    if isinstance(stats, (np.bool_, bool)):
        return bool(stats)
    # 处理其他类型
    return str(stats)