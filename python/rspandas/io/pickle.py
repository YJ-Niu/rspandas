"""Pickle 读写：read_pickle / to_pickle

由 rspandas/io.py 拆分而来，向后兼容通过 :mod:`rspandas.io` 包入口保证。
"""

from __future__ import annotations

from ..dataframe import DataFrame
from ..series import Series  # noqa: F401  # 部分函数需要

import pickle as _pickle


def read_pickle(path: str, **kwargs) -> DataFrame:
    """从 Pickle 文件读取 DataFrame。

    Parameters
    ----------
    path : str
        Pickle 文件路径。
    **kwargs
        传递给 pickle.load 的其他参数。

    Returns
    -------
    DataFrame
    """
    with open(path, "rb") as f:
        obj = _pickle.load(f)
    if isinstance(obj, dict) and "columns" in obj and "data" in obj:
        return DataFrame(obj["data"], columns=obj["columns"])
    if isinstance(obj, DataFrame):
        return obj
    raise TypeError(f"Pickle file contains {type(obj).__name__}, not DataFrame")


def to_pickle(df: DataFrame, path: str, **kwargs) -> None:
    """将 DataFrame 写入 Pickle 文件。

    Parameters
    ----------
    df : DataFrame
        要写入的 DataFrame。
    path : str
        输出文件路径。
    **kwargs
        传递给 pickle.dump 的其他参数。
    """
    # 序列化为纯 Python dict 以避免 pickle Rust 对象
    state = {
        "columns": list(df.columns),
        "data": df.values,
    }
    with open(path, "wb") as f:
        _pickle.dump(state, f, **kwargs)


# ============================================================================
# SQL
# ============================================================================
