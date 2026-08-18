"""ORC / Stata / HDF / SPSS / GBQ 等其他 IO

由 rspandas/io.py 拆分而来，向后兼容通过 :mod:`rspandas.io` 包入口保证。
"""

from __future__ import annotations

from ..dataframe import DataFrame
from ..series import Series  # noqa: F401  # 部分函数需要
from typing import Any, Dict, List, Optional, Tuple, Union

import json as _json
import pickle as _pickle


def read_orc(path: str, **kwargs) -> DataFrame:
    """从 ORC 文件读取 DataFrame。

    需安装 pyarrow：pip install pyarrow
    """
    try:
        import pyarrow.orc as orc

        orc_file = orc.ORCFile(path)
        table = orc_file.read()
        return _arrow_table_to_dataframe(table)
    except ImportError:
        raise ImportError(
            "read_orc requires pyarrow to be installed. "
            "Install with: pip install pyarrow"
        )


def to_orc(df: DataFrame, path: str, **kwargs) -> None:
    """将 DataFrame 写入 ORC 文件。

    需安装 pyarrow：pip install pyarrow
    """
    try:
        import pyarrow.orc as orc

        table = _dataframe_to_arrow_table(df)
        orc.write_table(table, path)
    except ImportError:
        raise ImportError(
            "to_orc requires pyarrow to be installed. "
            "Install with: pip install pyarrow"
        )


def read_stata(path: str, **kwargs) -> DataFrame:
    """从 Stata .dta 文件读取 DataFrame。

    需安装 pyreadstat：pip install pyreadstat
    """
    try:
        import pyreadstat

        df, _ = pyreadstat.read_dta(path)
        # 转换为 rspandas DataFrame
        return DataFrame(df.to_dict(orient="list"))
    except ImportError:
        raise ImportError(
            "read_stata requires pyreadstat to be installed. "
            "Install with: pip install pyreadstat"
        )


def to_stata(df: DataFrame, path: str, **kwargs) -> None:
    """将 DataFrame 写入 Stata .dta 文件。

    由于 rspandas 不依赖 pandas，此函数回退为写入 CSV 格式。
    如需真正的 .dta 格式，请手动转换为 pandas DataFrame 后使用 pyreadstat。
    """
    import warnings

    warnings.warn(
        "to_stata: rspandas 不依赖 pandas，无法直接写入 .dta 格式。"
        "回退为写入 CSV 格式。如需 .dta 格式，请手动处理。"
    )
    to_csv(df, path.replace(".dta", ".csv"))


def read_hdf(path_or_buf, key: str = "data", **kwargs) -> DataFrame:
    """从 HDF5 文件读取 DataFrame。

    需安装 tables：pip install tables
    """
    try:
        import tables

        with tables.open_file(path_or_buf, mode="r") as h5file:
            if key in h5file:
                table = h5file.get_node(key)
                data = {col: table.col(col) for col in table.colnames}
                return DataFrame(data)
            raise KeyError(f"Key '{key}' not found in HDF5 file")
    except ImportError:
        raise ImportError(
            "read_hdf requires tables to be installed. "
            "Install with: pip install tables"
        )


def to_hdf(
    df: DataFrame,
    path_or_buf,
    key: str = "data",
    mode: str = "a",
    **kwargs,
) -> None:
    """将 DataFrame 写入 HDF5 文件。

    需安装 tables：pip install tables
    """
    try:
        import tables

        with tables.open_file(path_or_buf, mode=mode, title="rspandas_hdf") as h5file:
            # 创建或覆盖表格
            if key in h5file:
                h5file.remove_node(key)
            description = {col: tables.Float64Col() for col in df.columns}
            table = h5file.create_table("/", key, description)
            for i in range(len(df)):
                row = table.row
                for col in df.columns:
                    row[col] = (
                        df[col].values[i] if df[col].values[i] is not None else 0.0
                    )
                row.append()
            table.flush()
    except ImportError:
        raise ImportError(
            "to_hdf requires tables to be installed. "
            "Install with: pip install tables"
        )


def read_spss(path: str, **kwargs) -> DataFrame:
    """从 SPSS .sav 文件读取 DataFrame。

    需安装 pyreadstat：pip install pyreadstat
    """
    try:
        import pyreadstat

        df, _ = pyreadstat.read_sav(path)
        return DataFrame(df.to_dict(orient="list"))
    except ImportError:
        raise ImportError(
            "read_spss requires pyreadstat to be installed. "
            "Install with: pip install pyreadstat"
        )


def read_gbq(query: str, project_id: str = None, **kwargs) -> DataFrame:
    """从 Google BigQuery 读取 DataFrame。

    需安装 google-cloud-bigquery：pip install google-cloud-bigquery
    """
    try:
        from google.cloud import bigquery

        client = bigquery.Client(project=project_id)
        query_job = client.query(query)
        results = query_job.result()

        # 转为字典列表
        rows_list = [dict(row) for row in results]
        if not rows_list:
            return DataFrame()

        # 构建 DataFrame
        data = {col: [row.get(col) for row in rows_list] for col in rows_list[0].keys()}
        return DataFrame(data)
    except ImportError:
        raise ImportError(
            "read_gbq requires google-cloud-bigquery to be installed. "
            "Install with: pip install google-cloud-bigquery"
        )


def to_gbq(
    df: DataFrame,
    destination_table: str,
    project_id: str = None,
    if_exists: str = "fail",
    **kwargs,
) -> None:
    """将 DataFrame 写入 Google BigQuery。

    需安装 google-cloud-bigquery：pip install google-cloud-bigquery
    """
    try:
        from google.cloud import bigquery

        client = bigquery.Client(project=project_id)
        table_ref = client.dataset(project_id or "default").table(destination_table)

        # 检查表是否存在
        try:
            client.get_table(table_ref)
            if if_exists == "fail":
                raise ValueError(f"Table {destination_table} already exists")
            elif if_exists == "replace":
                client.delete_table(table_ref)
        except Exception:
            pass

        # 插入数据
        rows_to_insert = [
            {col: df[col].values[i] for col in df.columns} for i in range(len(df))
        ]
        errors = client.insert_rows_json(table_ref, rows_to_insert)
        if errors:
            raise RuntimeError(f"Errors inserting rows: {errors}")
    except ImportError:
        raise ImportError(
            "to_gbq requires google-cloud-bigquery to be installed. "
            "Install with: pip install google-cloud-bigquery"
        )


# ============================================================================
# 辅助函数：to_csv（用于剪贴板等）
# ============================================================================
