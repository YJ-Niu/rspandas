"""Parquet 与 Feather (Arrow IPC) 读写

由 rspandas/io.py 拆分而来，向后兼容通过 :mod:`rspandas.io` 包入口保证。
"""

from __future__ import annotations

from ..dataframe import DataFrame
from ..series import Series  # noqa: F401  # 部分函数需要
from typing import Any, Dict, List, Optional, Tuple, Union

import json as _json
import pickle as _pickle


def read_parquet(path: str, **kwargs) -> DataFrame:
    """从 Parquet 文件读取 DataFrame（基于 Rust arrow/parquet crate，无需 pyarrow）。

    Parameters
    ----------
    path : str
        Parquet 文件路径。
    **kwargs
        忽略（兼容 pandas 签名）。

    Returns
    -------
    DataFrame
    """
    from ..rspandas import _DataFrame
    from ..rspandas import read_parquet as _read_parquet_rust

    cols, series_list = _read_parquet_rust(path)
    return DataFrame._from_inner(_DataFrame(cols, series_list))


def _arrow_table_to_dataframe(table) -> DataFrame:
    """将 PyArrow Table 转换为 DataFrame（用于 ORC 读取路径）。

    复用 DataFrame.from_arrow 的 Rust IPC 桥接路径：Table → IPC bytes → Rust 层
    反序列化，避免逐元素 to_pylist() 中转。
    """
    return DataFrame.from_arrow(table)


def to_parquet(
    df: DataFrame,
    path: str,
    compression: Optional[str] = "snappy",
    **kwargs,
) -> None:
    """将 DataFrame 写入 Parquet 文件（基于 Rust arrow/parquet crate，无需 pyarrow）。

    Parameters
    ----------
    df : DataFrame
        要写入的 DataFrame。
    path : str
        输出文件路径。
    compression : str, optional, default 'snappy'
        压缩算法 (snappy, gzip, brotli, zstd, lz4, none)。
    **kwargs
        忽略（兼容 pandas 签名）。
    """
    from ..rspandas import write_parquet as _write_parquet_rust

    cols = list(df.columns)
    series_list = [df._inner.get_column(c) for c in cols]
    _write_parquet_rust(path, cols, series_list, compression or "none")


def _dataframe_to_arrow_table(df: DataFrame):
    """将 DataFrame 转换为 PyArrow Table（用于 ORC 写入路径）。

    复用 DataFrame.to_arrow 的 Rust IPC 桥接路径：Rust 层将 ColumnData 精确映射为
    Arrow 类型并序列化为 IPC bytes，再由 pyarrow.ipc 反序列化为 Table。

    注意：Categorical 列会被 Rust 层展开为 utf8（而非 dictionary 编码），
    与旧实现的差异仅在 ORC 内部编码压缩率上，数据语义不变。
    """
    return df.to_arrow()


# ============================================================================
# Feather (Arrow IPC)
# ============================================================================


def read_feather(path: str, **kwargs) -> DataFrame:
    """从 Feather (Arrow IPC) 文件读取 DataFrame（基于 Rust arrow crate，无需 pyarrow）。

    Parameters
    ----------
    path : str
        Feather 文件路径。
    **kwargs
        忽略（兼容 pandas 签名）。

    Returns
    -------
    DataFrame
    """
    from ..rspandas import _DataFrame
    from ..rspandas import read_feather as _read_feather_rust

    cols, series_list = _read_feather_rust(path)
    return DataFrame._from_inner(_DataFrame(cols, series_list))


def to_feather(
    df: DataFrame,
    path: str,
    compression: Optional[str] = "uncompressed",
    **kwargs,
) -> None:
    """将 DataFrame 写入 Feather (Arrow IPC) 文件（基于 Rust arrow crate，无需 pyarrow）。

    Parameters
    ----------
    df : DataFrame
        要写入的 DataFrame。
    path : str
        输出文件路径。
    compression : str, optional, default 'uncompressed'
        压缩算法 (当前 Arrow IPC v1 仅支持 uncompressed，其他值会静默降级)。
    **kwargs
        忽略（兼容 pandas 签名）。
    """
    from ..rspandas import write_feather as _write_feather_rust

    cols = list(df.columns)
    series_list = [df._inner.get_column(c) for c in cols]
    _write_feather_rust(path, cols, series_list, compression or "uncompressed")


# ============================================================================
# Pickle
# ============================================================================
