"""IO 扩展: JSON / Excel / Parquet / Pickle / SQL 读写。

所有函数都接受/返回 DataFrame，与 pandas IO API 对齐。
"""

from __future__ import annotations

import json as _json
import pickle as _pickle
from typing import Any, Dict, Optional, Union
from .dataframe import DataFrame
# from .series import Series


# ============================================================================
# JSON
# ============================================================================

def read_json(
    path: str,
    orient: str = "records",
    lines: bool = False,
    encoding: str = "utf-8",
) -> DataFrame:
    """从 JSON 文件读取 DataFrame。

    Parameters
    ----------
    path : str
        JSON 文件路径。
    orient : str, default 'records'
        JSON 格式方向：
        - 'records': list[dict] (每行一个 dict)
        - 'columns': dict[str, list] (每列一个 list)
        - 'index': dict[str, dict] (行索引 → 列值)
        - 'split': {'columns': [...], 'data': [[...], ...]}
        - 'values': list[list] (纯二维数组)
    lines : bool, default False
        是否按行读取 JSON (每行一个 JSON 对象)。
    encoding : str, default 'utf-8'
        文件编码。

    Returns
    -------
    DataFrame
    """
    with open(path, "r", encoding=encoding) as f:
        if lines:
            records = [_json.loads(line) for line in f if line.strip()]
            return DataFrame(records)
        raw = _json.load(f)

    if orient == "records":
        return DataFrame(raw)
    elif orient == "columns":
        return DataFrame(raw)
    elif orient == "index":
        records = []
        for idx, row_dict in raw.items():
            record = {"index": idx, **row_dict}
            records.append(record)
        return DataFrame(records)
    elif orient == "split":
        cols = raw.get("columns", [])
        data = raw.get("data", [])
        return DataFrame(data, columns=cols)
    elif orient == "values":
        return DataFrame(raw)
    else:
        raise ValueError(f"Unknown orient: {orient}")


def to_json(
    df: DataFrame,
    path: Optional[str] = None,
    orient: str = "records",
    lines: bool = False,
    force_ascii: bool = False,
    indent: Optional[int] = None,
) -> Optional[str]:
    """将 DataFrame 写入 JSON 文件或返回 JSON 字符串。

    Parameters
    ----------
    df : DataFrame
        要写入的 DataFrame。
    path : str, optional
        输出文件路径。None 则返回字符串。
    orient : str, default 'records'
        JSON 格式方向。
    lines : bool, default False
        是否按行输出 JSON。
    force_ascii : bool, default False
        是否强制 ASCII 编码。
    indent : int, optional
        缩进空格数。

    Returns
    -------
    str or None
    """
    # df.values 返回 list[dict]
    records = df.values

    if orient == "records":
        data = records
    elif orient == "columns":
        data = {col: [row.get(col) for row in records] for col in df.columns}
    elif orient == "index":
        data = {}
        for i, row in enumerate(records):
            data[str(i)] = row
    elif orient == "split":
        data = {
            "columns": list(df.columns),
            "data": [[row.get(c) for c in df.columns] for row in records],
        }
    elif orient == "values":
        data = [[row.get(c) for c in df.columns] for row in records]
    else:
        raise ValueError(f"Unknown orient: {orient}")

    json_kwargs: Dict[str, Any] = {"ensure_ascii": force_ascii}
    if indent is not None:
        json_kwargs["indent"] = indent

    if lines:
        if orient != "records":
            raise ValueError("lines=True requires orient='records'")
        output = "\n".join(_json.dumps(r, **json_kwargs) for r in data)
    else:
        output = _json.dumps(data, **json_kwargs)

    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(output)
        if not output.endswith("\n"):
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n")
        return None
    return output


# ============================================================================
# Excel (使用 Rust 后端 calamine + rust_xlsxwriter，无需 openpyxl)
# ============================================================================

def read_excel(
    path: str,
    sheet_name: Union[str, int] = 0,
    header: int = 0,
    **kwargs,
) -> DataFrame:
    """从 Excel 文件读取 DataFrame (使用 Rust calamine 后端)。

    Parameters
    ----------
    path : str
        Excel 文件路径 (.xlsx / .xls / .ods)。
    sheet_name : str or int, default 0
        工作表名称或索引。
    header : int, default 0
        用作列名的行号。
    **kwargs
        忽略 (兼容 pandas 签名)。

    Returns
    -------
    DataFrame
    """
    from .rspandas import read_xlsx as _read_xlsx, _DataFrame

    if isinstance(sheet_name, int):
        cols, series_list = _read_xlsx(path, None, sheet_name, header)
    else:
        cols, series_list = _read_xlsx(path, sheet_name, None, header)

    return DataFrame._from_inner(_DataFrame(cols, series_list))


def to_excel(
    df: DataFrame,
    path: str,
    sheet_name: str = "Sheet1",
    index: bool = False,
    header: bool = True,
    **kwargs,
) -> None:
    """将 DataFrame 写入 Excel 文件 (使用 Rust rust_xlsxwriter 后端)。

    Parameters
    ----------
    df : DataFrame
        要写入的 DataFrame。
    path : str
        输出文件路径。
    sheet_name : str, default 'Sheet1'
        工作表名称。
    index : bool, default False
        是否写入行索引。
    header : bool, default True
        是否写入列名。
    **kwargs
        忽略 (兼容 pandas 签名)。
    """
    from .rspandas import write_xlsx as _write_xlsx

    cols = list(df.columns)
    series_list = [df._inner.get_column(c) for c in cols]
    _write_xlsx(path, cols, series_list, sheet_name, header, index)


# ============================================================================
# Parquet
# ============================================================================

def read_parquet(path: str, **kwargs) -> DataFrame:
    """从 Parquet 文件读取 DataFrame。

    Parameters
    ----------
    path : str
        Parquet 文件路径。
    **kwargs
        传递给 pyarrow/pandas 的其他参数。

    Returns
    -------
    DataFrame
    """
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path, **kwargs)
        return _arrow_table_to_dataframe(table)
    except ImportError:
        raise ImportError(
            "read_parquet requires pyarrow to be installed. "
            "Install with: pip install pyarrow"
        )


def _arrow_table_to_dataframe(table) -> DataFrame:
    """将 PyArrow Table 转换为 DataFrame。"""
    data: Dict[str, list] = {}
    for col_name in table.column_names:
        col = table.column(col_name)
        data[col_name] = col.to_pylist()
    return DataFrame(data)


def to_parquet(
    df: DataFrame,
    path: str,
    compression: Optional[str] = "snappy",
    **kwargs,
) -> None:
    """将 DataFrame 写入 Parquet 文件。

    Parameters
    ----------
    df : DataFrame
        要写入的 DataFrame。
    path : str
        输出文件路径。
    compression : str, optional, default 'snappy'
        压缩算法 (snappy, gzip, brotli, zstd, none)。
    **kwargs
        传递给 pyarrow/pandas 的其他参数。
    """
    try:
        import pyarrow.parquet as pq

        table = _dataframe_to_arrow_table(df)
        pq.write_table(table, path, compression=compression, **kwargs)
        return
    except ImportError:
        pass

    try:
        pdf = df.to_pandas()
        pdf.to_parquet(path, compression=compression, **kwargs)
        return
    except ImportError:
        raise ImportError(
            "to_parquet requires pyarrow or pandas to be installed. "
            "Install with: pip install pyarrow"
        )


def _dataframe_to_arrow_table(df: DataFrame):
    """将 DataFrame 转换为 PyArrow Table。"""
    import pyarrow as pa

    arrays = []
    for col_name in df.columns:
        col_data = list(df[col_name].values)
        # 推断类型
        non_null = [v for v in col_data if v is not None]
        if not non_null:
            arrays.append(pa.array(col_data, type=pa.string()))
        elif all(isinstance(v, bool) for v in non_null):
            arrays.append(pa.array(col_data, type=pa.bool_()))
        elif all(isinstance(v, int) for v in non_null):
            arrays.append(pa.array(col_data, type=pa.int64()))
        elif all(isinstance(v, float) for v in non_null):
            arrays.append(pa.array(col_data, type=pa.float64()))
        else:
            arrays.append(pa.array([str(v) if v is not None else None for v in col_data]))

    return pa.table(dict(zip(df.columns, arrays)))


# ============================================================================
# Pickle
# ============================================================================

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

def read_sql(
    query: str,
    conn,
    **kwargs,
) -> DataFrame:
    """从 SQL 数据库读取 DataFrame。

    Parameters
    ----------
    query : str
        SQL 查询语句。
    conn : sqlalchemy Engine 或 Connection
        数据库连接。
    **kwargs
        忽略 (兼容 pandas 签名)。

    Returns
    -------
    DataFrame
    """
    try:
        import sqlalchemy as sa
    except ImportError:
        raise ImportError(
            "read_sql requires sqlalchemy to be installed. "
            "Install with: pip install sqlalchemy"
        )

    with conn.connect() as connection:
        result = connection.execute(sa.text(query))
        rows = result.fetchall()
        columns = list(result.keys())

    data = {c: [row[i] for row in rows] for i, c in enumerate(columns)}
    return DataFrame(data)


def to_sql(
    df: DataFrame,
    name: str,
    conn,
    if_exists: str = "fail",
    index: bool = False,
    **kwargs,
) -> None:
    """将 DataFrame 写入 SQL 数据库。

    Parameters
    ----------
    df : DataFrame
        要写入的 DataFrame。
    name : str
        目标表名。
    conn : sqlalchemy Engine 或 Connection
        数据库连接。
    if_exists : str, default 'fail'
        表已存在时的行为：'fail', 'replace', 'append'。
    index : bool, default False
        是否写入行索引。
    **kwargs
        忽略 (兼容 pandas 签名)。
    """
    try:
        import sqlalchemy as sa
    except ImportError:
        raise ImportError(
            "to_sql requires sqlalchemy to be installed. "
            "Install with: pip install sqlalchemy"
        )

    with conn.connect() as connection:
        meta = sa.MetaData()
        meta.reflect(bind=connection)
        if name in meta.tables:
            if if_exists == "replace":
                meta.tables[name].drop(connection)
            elif if_exists == "fail":
                raise ValueError(f"Table '{name}' already exists")
            elif if_exists == "append":
                pass
            else:
                raise ValueError(f"Unknown if_exists: {if_exists}")

        if if_exists == "replace" or name not in meta.tables:
            # Create table from DataFrame schema
            cols = []
            for c in df.columns:
                sample = next((v for v in df[c].values if v is not None), None)
                if isinstance(sample, bool):
                    col_type = sa.Boolean
                elif isinstance(sample, int):
                    col_type = sa.Integer
                elif isinstance(sample, float):
                    col_type = sa.Float
                else:
                    col_type = sa.String
                cols.append(sa.Column(c, col_type))
            sa.Table(name, meta, *cols)
            meta.create_all(connection)

        # Insert data
        rows = [{c: v for c, v in zip(df.columns, row)} for row in df.values]
        if rows:
            connection.execute(sa.text(f"INSERT INTO {name} ({', '.join(df.columns)}) VALUES ({', '.join([':' + c for c in df.columns])})"), rows)
