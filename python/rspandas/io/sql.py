"""SQL 读写：read_sql / read_sql_query / read_sql_table / to_sql / to_sql_batch

由 rspandas/io.py 拆分而来，向后兼容通过 :mod:`rspandas.io` 包入口保证。
"""

from __future__ import annotations

from ..dataframe import DataFrame
from ..series import Series  # noqa: F401  # 部分函数需要
from typing import Any, Dict, List, Optional, Tuple, Union

import json as _json
import pickle as _pickle


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


def read_sql_query(
    sql: str,
    conn,
    index_col=None,
    coerce_float: bool = True,
    params=None,
    **kwargs,
) -> DataFrame:
    """从 SQL 查询语句读取 DataFrame。

    Parameters
    ----------
    sql : str
        SQL 查询语句。
    conn : sqlalchemy Engine 或 Connection
        数据库连接。
    index_col : str 或 list, 可选
        用作索引的列名。
    coerce_float : bool
        尝试将非字符串/数字对象转为浮点数（兼容签名，暂不强制转换）。
    params : 参数绑定
        传递给 SQLAlchemy 的参数。
    """
    try:
        import sqlalchemy as sa
    except ImportError:
        raise ImportError(
            "read_sql_query requires sqlalchemy to be installed. "
            "Install with: pip install sqlalchemy"
        )

    with conn.connect() as connection:
        result = connection.execute(sa.text(sql), params or {})
        rows = result.fetchall()
        columns = list(result.keys())

    data = {c: [row[i] for row in rows] for i, c in enumerate(columns)}
    df = DataFrame(data)
    if index_col is not None:
        if isinstance(index_col, str):
            df._index = list(df[index_col].values) if index_col in df._columns else None
        elif isinstance(index_col, list):
            df._index = [list(df[c].values) for c in index_col]
    return df


def read_sql_table(
    table_name: str,
    conn,
    schema=None,
    index_col=None,
    coerce_float: bool = True,
    columns=None,
    **kwargs,
) -> DataFrame:
    """从 SQL 表名读取 DataFrame。

    Parameters
    ----------
    table_name : str
        数据库表名。
    conn : sqlalchemy Engine 或 Connection
        数据库连接。
    schema : str, 可选
        数据库 schema。
    index_col : str 或 list, 可选
        用作索引的列名。
    coerce_float : bool
        尝试将值转为浮点数（兼容签名，暂不强制转换）。
    columns : list, 可选
        只读取指定列。
    """
    try:
        import sqlalchemy as sa
    except ImportError:
        raise ImportError(
            "read_sql_table requires sqlalchemy to be installed. "
            "Install with: pip install sqlalchemy"
        )

    # 构建查询
    table_ref = f'"{schema}"."{table_name}"' if schema else f'"{table_name}"'
    col_clause = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    query = f"SELECT {col_clause} FROM {table_ref}"

    with conn.connect() as connection:
        result = connection.execute(sa.text(query))
        rows = result.fetchall()
        result_columns = list(result.keys())

    data = {c: [row[i] for row in rows] for i, c in enumerate(result_columns)}
    df = DataFrame(data)
    if index_col is not None:
        if isinstance(index_col, str):
            df._index = list(df[index_col].values) if index_col in df._columns else None
        elif isinstance(index_col, list):
            df._index = [list(df[c].values) for c in index_col]
    return df


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
            # 使用辅助函数 + 列表推导式替代显式 for 循环
            def _infer_sa_type(c):
                sample = next((v for v in df[c].values if v is not None), None)
                if isinstance(sample, bool):
                    return sa.Boolean
                if isinstance(sample, int):
                    return sa.Integer
                if isinstance(sample, float):
                    return sa.Float
                return sa.String

            cols = [sa.Column(c, _infer_sa_type(c)) for c in df.columns]
            sa.Table(name, meta, *cols)
            meta.create_all(connection)

        # Insert data
        records = df.values  # list[dict]
        if records:
            col_names = list(df.columns)
            placeholders = ", ".join([":" + c for c in col_names])
            stmt = sa.text(
                f"INSERT INTO {name} ({', '.join(col_names)}) VALUES ({placeholders})"
            )
            connection.execute(stmt, records)


def to_sql_batch(
    df: DataFrame,
    name: str,
    conn,
    batch_size: int = 500,
    if_exists: str = "fail",
    index: bool = False,
    **kwargs,
) -> None:
    """批量将 DataFrame 写入 SQL 数据库。

    将数据按 batch_size 分批插入，避免单次 INSERT 过大。

    Parameters
    ----------
    df : DataFrame
        要写入的 DataFrame。
    name : str
        目标表名。
    conn : sqlalchemy Engine 或 Connection
        数据库连接。
    batch_size : int
        每批行数（默认 500）。
    if_exists : str
        表已存在时的行为：'fail'/'replace'/'append'。
    index : bool
        是否写入行索引。
    """
    try:
        import sqlalchemy as sa
    except ImportError:
        raise ImportError(
            "to_sql_batch requires sqlalchemy to be installed. "
            "Install with: pip install sqlalchemy"
        )

    col_names = list(df.columns)
    if index:
        col_names = ["index"] + col_names

    with conn.connect() as connection:
        meta = sa.MetaData()
        meta.reflect(bind=connection)
        if name in meta.tables:
            if if_exists == "replace":
                meta.tables[name].drop(connection)
            elif if_exists == "fail":
                raise ValueError(f"Table '{name}' already exists")

        if if_exists in ("replace",) or name not in meta.tables:
            # 推断列类型
            def _infer_sa_type(c):
                sample = next((v for v in df[c].values if v is not None), None)
                if isinstance(sample, bool):
                    return sa.Boolean
                if isinstance(sample, int):
                    return sa.Integer
                if isinstance(sample, float):
                    return sa.Float
                return sa.String

            cols = [sa.Column(c, _infer_sa_type(c)) for c in df.columns]
            sa.Table(name, meta, *cols)
            meta.create_all(connection)

        # 分批插入
        placeholders = ", ".join([":" + c for c in col_names])
        stmt = sa.text(
            f"INSERT INTO {name} ({', '.join(col_names)}) VALUES ({placeholders})"
        )
        records = df.values
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]  # noqa
            connection.execute(stmt, batch)
