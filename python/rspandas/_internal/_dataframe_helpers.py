"""DataFrame 内部辅助函数

由 rspandas/dataframe.py 拆分而来。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import rsnumpy as rnp

from ..rspandas import _Series as _PySeries  # type: ignore


def _is_ndarray(data: Any) -> bool:
    """检查对象是否为 rsnumpy ndarray。"""
    return isinstance(data, rnp.ndarray)


def _convert_to_basic(v: Any) -> Any:
    """将值转换为 Rust 端可接受的基础类型。"""
    if v is None:
        return None
    if isinstance(v, (int, float, str, bool)):
        return v
    # Timestamp -> ISO 字符串（去除时间部分如果为 00:00:00）
    from .. import Timestamp

    if isinstance(v, Timestamp):
        dt = v._dt
        if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
            return dt.strftime("%Y-%m-%d")
        # 用空格替代 ISO 默认的 'T'，与 pandas 显示一致
        return dt.isoformat().replace("T", " ", 1)
    # datetime -> ISO 字符串（去除时间部分如果为 00:00:00，用空格替代 'T'）
    if isinstance(v, datetime):
        if v.hour == 0 and v.minute == 0 and v.second == 0 and v.microsecond == 0:
            return v.strftime("%Y-%m-%d")
        return v.isoformat().replace("T", " ", 1)
    # date -> ISO 字符串
    if hasattr(v, "isoformat"):
        return v.isoformat().replace("T", " ", 1)
    # numpy 标量 -> Python 标量
    if hasattr(v, "item"):
        # 多元素数组无法调用无参 item()，转换为 list
        if hasattr(v, "size") and v.size > 1:
            return list(v)
        return v.item()
    # timedelta -> 'N days HH:MM:SS.ffffff' 格式（对齐 pandas 显示）
    if isinstance(v, timedelta):
        days = v.days
        total_sec = v.seconds
        hours = total_sec // 3600
        minutes = (total_sec % 3600) // 60
        secs = total_sec % 60
        us = v.microseconds
        if us > 0:
            return f"{days} days {hours:02d}:{minutes:02d}:{secs:02d}.{us:06d}"
        if hours == 0 and minutes == 0 and secs == 0:
            return f"{days} days"
        return f"{days} days {hours:02d}:{minutes:02d}:{secs:02d}"
    # rspandas.Timedelta 对象
    if hasattr(v, "_td") and isinstance(getattr(v, "_td", None), timedelta):
        return _convert_to_basic(v._td)
    # 其他可转换对象 -> str
    return str(v) if v is not None else None


def _convert_list_to_basic(values: list) -> list:
    """将列表中的每个值转换为基础类型。"""
    return [_convert_to_basic(v) for v in values]


# Python type / 字符串 -> Rust 端 dtype 字符串的映射
_DTYPE_MAP = {
    bool: "bool",
    int: "int64",
    float: "float64",
    str: "object",
    "bool": "bool",
    "int": "int64",
    "int64": "int64",
    "int32": "int32",
    "float": "float64",
    "float64": "float64",
    "float32": "float32",
    "object": "object",
    "str": "object",
    "datetime64[ns]": "datetime64[ns]",
    "category": "category",
}


def _normalize_dtype(dtype) -> Optional[str]:
    """将 dtype 参数规范化为 Rust 端可接受的字符串。

    接受 Python type（如 ``bool``）、dtype 字符串（如 ``"int64"``）或 None。
    """
    if dtype is None:
        return None
    if isinstance(dtype, str):
        return _DTYPE_MAP.get(dtype, dtype)
    if isinstance(dtype, type):
        return _DTYPE_MAP.get(dtype, dtype.__name__)
    return str(dtype)


def _to_pylist_columns(data: Any, columns: Optional[List[str]]) -> Dict[str, list]:
    """将 dict/list/ndarray 输入解析为 dict[str, list]。"""
    from ..series import Series

    # data=None 且指定 columns: 创建空列的 DataFrame (0 行)
    if data is None:
        if columns is None:
            return {}
        return {c: [] for c in columns}
    if isinstance(data, dict):
        result = {}
        # 检查是否有 Series 带自定义 index
        has_series_with_index = False
        series_indices = set()
        has_dict_values = False
        for k, v in data.items():
            if isinstance(v, Series):
                if v._index is not None:
                    has_series_with_index = True
                    series_indices.add(tuple(v._index))
            elif isinstance(v, dict):
                has_dict_values = True
        # 如果存在 Series 带自定义 index，在 __init__ 中处理对齐
        _series_alignment = None
        if has_series_with_index:
            all_indices = []
            for k, v in data.items():
                if isinstance(v, Series) and v._index is not None:
                    all_indices.append(v._index)
            if all_indices and not all(
                idx == all_indices[0] for idx in all_indices[1:]
            ):
                _series_alignment = all_indices
            elif all_indices:
                # 所有 Series index 相同，也需要传递以便用作 DataFrame 行索引
                _series_alignment = all_indices

        # 如果存在 dict 值，需要收集所有唯一索引
        dict_all_keys = []
        if has_dict_values:
            seen = set()
            for k, v in data.items():
                if isinstance(v, dict):
                    for key in v:
                        if key not in seen:
                            seen.add(key)
                            dict_all_keys.append(key)

        for k, v in data.items():
            if isinstance(v, Series):
                result[k] = list(v.values)
            elif isinstance(v, _PySeries):
                result[k] = list(v.values)
            elif isinstance(v, dict):
                # dict 值：转换为 list，按 dict_all_keys 对齐
                result[k] = [v.get(key) for key in dict_all_keys]
            elif v is None:
                result[k] = []
            elif isinstance(v, (list, tuple)):
                result[k] = list(v)
            elif hasattr(v, "tolist"):
                # 对 ndarray 类型，保留 dtype 信息（如 int32 应保持为 int）
                if hasattr(v, "dtype") and v.dtype is not None:
                    dt_str = str(v.dtype)
                    if dt_str in (
                        "int8",
                        "int16",
                        "int32",
                        "int64",
                        "int",
                        "uint8",
                        "uint16",
                        "uint32",
                        "uint64",
                    ):
                        result[k] = [int(x) for x in v]
                    elif dt_str in ("float32", "float64", "float"):
                        result[k] = [float(x) for x in v]
                    else:
                        result[k] = (
                            list(v)
                            if not isinstance(v, (int, float, str, bool))
                            else [v]
                        )
                else:
                    result[k] = (
                        list(v) if not isinstance(v, (int, float, str, bool)) else [v]
                    )
            elif hasattr(v, "__iter__") and not isinstance(v, (str, dict, bytes)):
                # 可迭代对象（如 Categorical），展开为列表
                result[k] = list(v)
            else:
                # 标量值：先暂存，后续广播到其他列长度
                result[k] = [v]  # 先包装成单元素列表，DataFrame.__init__ 会处理广播
        if _series_alignment is not None:
            result["__series_alignment__"] = _series_alignment
        if has_dict_values and dict_all_keys:
            result["__dict_index__"] = dict_all_keys
        return result

    if isinstance(data, list):
        if not data:
            return {}
        if isinstance(data[0], dict):
            # list[dict]
            if columns is None:
                columns = []
                for row in data:
                    for k in row.keys():
                        if k not in columns:
                            columns.append(k)
            result: Dict[str, list] = {c: [] for c in columns}
            for row in data:
                for c in columns:
                    result[c].append(row.get(c))
            return result
        if isinstance(data[0], (list, tuple)):
            # list[list] 或 list[tuple]
            if hasattr(data[0], "_fields"):
                # list[namedtuple] - namedtuple 继承自 tuple，但有 _fields 属性
                fields = list(data[0]._fields)
                if columns is None:
                    columns = fields
                result = {c: [] for c in columns}
                for row in data:
                    for c in columns:
                        result[c].append(getattr(row, c, None))
                return result
            if columns is None:
                columns = [str(i) for i in range(len(data[0]))]
            result = {c: [] for c in columns}
            for row in data:
                for i, c in enumerate(columns):
                    result[c].append(row[i] if i < len(row) else None)
            return result
        if hasattr(data[0], "__dataclass_fields__"):
            # list[dataclass]
            fields = list(data[0].__dataclass_fields__.keys())
            if columns is None:
                columns = fields
            result = {c: [] for c in columns}
            for row in data:
                for c in columns:
                    result[c].append(getattr(row, c, None))
            return result
        if hasattr(data[0], "_fields"):
            # list[namedtuple]（兜底，当 data[0] 不是 list/tuple 子类时）
            fields = list(data[0]._fields)
            if columns is None:
                columns = fields
            result = {c: [] for c in columns}
            for row in data:
                for c in columns:
                    result[c].append(getattr(row, c, None))
            return result
        # 一维标量列表: 作为单列处理 ()
        # 元素为 int/float/str/bool/None/nan 等标量值
        if columns is None:
            columns = ["0"]
        elif len(columns) != 1:
            raise ValueError(f"一维标量列表需要 1 个列名，得到 {len(columns)} 个")
        return {columns[0]: list(data)}

    if _is_ndarray(data):
        # rsnumpy ndarray: 转换为 list[list] 后按列组织
        raw_list = data.tolist()
        if not isinstance(raw_list, list):
            # 0 维数组
            return {"0": [raw_list]}
        if not raw_list:
            return {}
        if isinstance(raw_list[0], (list, tuple)):
            # 2D 数组或结构化数组（元素为 tuple）
            ncols = len(raw_list[0])
            # 尝试从 ndarray 的 dtype 获取字段名（结构化数组）
            field_names = None
            try:
                if hasattr(data, "dtype") and hasattr(data.dtype, "names"):
                    names = data.dtype.names
                    if names and len(names) == ncols:
                        field_names = names
            except Exception:
                pass
            if columns is None and field_names is not None:
                columns = list(field_names)
            if columns is None:
                columns = [str(i) for i in range(ncols)]
            result = {c: [] for c in columns}
            for row in raw_list:
                for i, c in enumerate(columns):
                    result[c].append(row[i] if i < len(row) else None)
            return result
        # 1D 数组
        if columns is None:
            columns = ["0"]
        result = {c: [] for c in columns}
        for v in raw_list:
            result[columns[0]].append(v)
        return result

    raise TypeError(f"Cannot build DataFrame from {type(data).__name__}")
