"""IO 扩展: JSON / Excel / Parquet / Pickle / SQL 读写。

所有函数都接受/返回 DataFrame，与 pandas IO API 对齐。"""

from __future__ import annotations

from .dataframe import DataFrame
from typing import Any, Dict, List, Optional, Tuple, Union

import json as _json
import pickle as _pickle


class ExcelWriter:
    """Excel 写入器，支持将多个 DataFrame 写入同一个文件的不同 sheet。

    用法:
        with ExcelWriter('output.xlsx') as writer:
            df1.to_excel(writer, sheet_name='Sheet1')
            df2.to_excel(writer, sheet_name='Sheet2')
    """

    def __init__(self, path: str):
        self._path = path
        self._sheets: List[Tuple[str, DataFrame, bool, bool]] = []

    def write(
        self,
        df: DataFrame,
        sheet_name: str = "Sheet1",
        index: bool = False,
        header: bool = True,
    ):
        """将 DataFrame 写入指定 sheet。"""
        self._sheets.append((sheet_name, df, header, index))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()

    def close(self):
        """关闭写入器并保存文件。"""
        from .rspandas import write_xlsx_multi as _write_xlsx_multi

        # 使用列表推导式替代显式 for 循环构建 sheets_data
        sheets_data = [
            (
                sheet_name,
                list(df.columns),
                [df._inner.get_column(c) for c in df.columns],
                include_header,
                include_index,
            )
            for sheet_name, df, include_header, include_index in self._sheets
        ]

        _write_xlsx_multi(self._path, sheets_data)


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
        # 使用列表推导式替代显式 for 循环
        records = [{"index": idx, **row_dict} for idx, row_dict in raw.items()]
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
        # 使用字典推导式替代显式 for 循环
        data = {str(i): row for i, row in enumerate(records)}
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
    from .rspandas import _DataFrame
    from .rspandas import read_xlsx as _read_xlsx

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
    # 使用字典推导式替代显式 for 循环
    data: Dict[str, list] = {
        col_name: table.column(col_name).to_pylist() for col_name in table.column_names
    }
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

    def _infer_array(col_name):
        """为单列推断 PyArrow 类型并构造数组。"""
        col_data = list(df[col_name].values)
        # 推断类型
        non_null = [v for v in col_data if v is not None]
        if not non_null:
            return pa.array(col_data, type=pa.string())
        if all(isinstance(v, bool) for v in non_null):
            return pa.array(col_data, type=pa.bool_())
        if all(isinstance(v, int) for v in non_null):
            return pa.array(col_data, type=pa.int64())
        if all(isinstance(v, float) for v in non_null):
            return pa.array(col_data, type=pa.float64())
        return pa.array([str(v) if v is not None else None for v in col_data])

    # 使用列表推导式替代显式 for 循环
    arrays = [_infer_array(col_name) for col_name in df.columns]
    return pa.table(dict(zip(df.columns, arrays)))


# ============================================================================
# Feather (Arrow IPC)
# ============================================================================


def read_feather(path: str, **kwargs) -> DataFrame:
    """从 Feather 文件读取 DataFrame。

    Parameters
    ----------
    path : str
        Feather 文件路径。
    **kwargs
        传递给 pyarrow.feather.read_table 的其他参数。

    Returns
    -------
    DataFrame
    """
    try:
        import pyarrow.feather as pf

        table = pf.read_table(path, **kwargs)
        return _arrow_table_to_dataframe(table)
    except ImportError:
        raise ImportError(
            "read_feather requires pyarrow to be installed. "
            "Install with: pip install pyarrow"
        )


def to_feather(
    df: DataFrame,
    path: str,
    compression: Optional[str] = "lz4",
    **kwargs,
) -> None:
    """将 DataFrame 写入 Feather 文件。

    Parameters
    ----------
    df : DataFrame
        要写入的 DataFrame。
    path : str
        输出文件路径。
    compression : str, optional, default 'lz4'
        压缩算法 (lz4, zstd, uncompressed)。
    **kwargs
        传递给 pyarrow.feather.write_feather 的其他参数。
    """
    try:
        import pyarrow.feather as pf

        table = _dataframe_to_arrow_table(df)
        pf.write_feather(table, path, compression=compression, **kwargs)
    except ImportError:
        raise ImportError(
            "to_feather requires pyarrow to be installed. "
            "Install with: pip install pyarrow"
        )


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


def read_csv_chunked(
    path: str,
    chunk_size: int = 1000,
    encoding: str = "utf-8",
    delimiter: str = ",",
    header: bool = True,
    **kwargs,
):
    """分块读取大 CSV 文件，返回迭代器。

    每次产出 chunk_size 行的 DataFrame，避免一次性加载全部数据。

    Parameters
    ----------
    path : str
        CSV 文件路径。
    chunk_size : int
        每块行数（默认 1000）。
    encoding : str
        文件编码（默认 utf-8）。
    delimiter : str
        分隔符（默认逗号）。
    header : bool
        是否有表头（默认 True）。

    Yields
    ------
    DataFrame
        每次产出一个 chunk_size 行的 DataFrame。
    """
    import csv as _csv

    with open(path, "r", encoding=encoding, newline="") as f:
        reader = _csv.reader(f, delimiter=delimiter)
        col_names = None
        if header:
            col_names = next(reader, None)

        chunk: List[list] = []
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                data = _rows_to_dict(chunk, col_names)
                yield DataFrame(data)
                chunk = []

        # 输出剩余行
        if chunk:
            data = _rows_to_dict(chunk, col_names)
            yield DataFrame(data)


def _rows_to_dict(rows: List[list], col_names: Optional[List[str]]) -> Dict[str, list]:
    """将行列表转为列字典（列表推导式优化）。"""
    if col_names is None:
        n_cols = max(len(r) for r in rows) if rows else 0
        col_names = [f"col{i}" for i in range(n_cols)]
    return {
        col_names[i]: [r[i] if i < len(r) else None for r in rows]
        for i in range(len(col_names))
    }


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
            batch = records[i : i + batch_size]
            connection.execute(stmt, batch)


class StreamDataFrame:
    """流式 DataFrame，支持链式管道操作。

    用法::

        >>> from rspandas.io import StreamDataFrame
        >>> sdf = StreamDataFrame(read_csv_chunked('big.csv', chunk_size=100))
        >>> result = sdf.filter(lambda df: df['val'] > 0).map(lambda df: df.head(10)).collect()
    """

    def __init__(self, source):
        """初始化。

        Parameters
        ----------
        source : 可迭代的 DataFrame，或生成器函数
        """
        self._source = source
        self._pipes: List = []

    def filter(self, func):
        """添加过滤管道：只保留 func(chunk) 为 True 的 chunk。"""
        self._pipes.append(("filter", func))
        return self

    def map(self, func):
        """添加映射管道：对每个 chunk 应用 func。"""
        self._pipes.append(("map", func))
        return self

    def reduce(self, func, initial=None):
        """添加归约管道：将所有 chunk 归约为单个值。"""
        self._pipes.append(("reduce", func, initial))
        return self

    def collect(self) -> DataFrame:
        """收集所有 chunk 并合并为单个 DataFrame。"""
        chunks = []
        for chunk in self._source:
            result = chunk
            for pipe in self._pipes:
                if pipe[0] == "filter":
                    if not pipe[1](result):
                        result = None
                        break
                elif pipe[0] == "map":
                    result = pipe[1](result)
            if result is not None:
                chunks.append(result)

        if not chunks:
            return DataFrame()
        # 合并所有 chunk
        combined_data: Dict[str, list] = {}
        for chunk in chunks:
            for c in chunk.columns:
                combined_data.setdefault(c, []).extend(list(chunk[c].values))
        return DataFrame(combined_data)

    def __iter__(self):
        """迭代模式：逐个产出处理后的 chunk。"""
        for chunk in self._source:
            result = chunk
            for pipe in self._pipes:
                if pipe[0] == "filter":
                    if not pipe[1](result):
                        result = None
                        break
                elif pipe[0] == "map":
                    result = pipe[1](result)
            if result is not None:
                yield result


# ============================================================================
# 扩展 IO 方法（需第三方库支持，按需启用）
# ============================================================================


def read_html(
    io,
    match=0,
    flavor=None,
    header=0,
    index_col=None,
    skiprows=None,
    attrs=None,
    encoding=None,
    **kwargs,
) -> DataFrame:
    """从 HTML 表格读取 DataFrame。

    需安装 BeautifulSoup4 和 lxml：pip install beautifulsoup4 lxml
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "read_html requires beautifulsoup4 to be installed. "
            "Install with: pip install beautifulsoup4 lxml"
        )

    if isinstance(io, str):
        with open(io, "r", encoding=encoding or "utf-8") as f:
            content = f.read()
    else:
        content = io.read() if hasattr(io, "read") else str(io)

    soup = BeautifulSoup(content, "lxml")
    tables = soup.find_all("table", attrs=attrs or {})

    if not tables:
        return DataFrame()

    if isinstance(match, int):
        table = tables[match] if match < len(tables) else tables[0]
    elif hasattr(match, "__call__"):
        table = next((t for t in tables if match(t)), tables[0])
    else:
        table = tables[0]

    # 解析表格
    rows_data = []
    for tr in table.find_all("tr")[header or 0 :]:
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows_data.append(cells)

    if header is not None and rows_data:
        col_names = rows_data[0]
        data_rows = rows_data[1:]
    else:
        col_names = [f"col{i}" for i in range(len(rows_data[0]))] if rows_data else []
        data_rows = rows_data

    data = {
        col_names[i] if i < len(col_names) else f"col{i}": [
            r[i] if i < len(r) else None for r in data_rows
        ]
        for i in range(max(len(r) for r in data_rows) if data_rows else 0)
    }
    return DataFrame(data)


def to_html(df: DataFrame, path=None, index: bool = True, **kwargs) -> Optional[str]:
    """将 DataFrame 写入 HTML 文件或返回 HTML 字符串。"""
    # 简单实现：手动生成 HTML 表格
    lines = ['<table border="1">']
    # 表头
    lines.append("<tr>")
    if index:
        lines.append("<th></th>")
    for col in df.columns:
        lines.append(f"<th>{col}</th>")
    lines.append("</tr>")
    # 数据行
    for i in range(len(df)):
        lines.append("<tr>")
        if index:
            idx_val = df._index[i] if df._index and i < len(df._index) else i
            lines.append(f"<td>{idx_val}</td>")
        for col in df.columns:
            val = df[col].values[i]
            lines.append(f"<td>{val if val is not None else ''}</td>")
        lines.append("</tr>")
    lines.append("</table>")
    html_content = "\n".join(lines)

    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return None
    return html_content


def read_clipboard(**kwargs) -> DataFrame:
    """从系统剪贴板读取 DataFrame。

    需安装 pyperclip：pip install pyperclip
    """
    try:
        import pyperclip

        text = pyperclip.paste()
        # 尝试用 read_csv 解析（以制表符分隔为默认）
        import io as _io

        from . import read_csv as _read_csv

        return _read_csv(_io.StringIO(text), sep="\t")
    except ImportError:
        raise ImportError(
            "read_clipboard requires pyperclip to be installed. "
            "Install with: pip install pyperclip"
        )


def to_clipboard(df: DataFrame, excel: bool = True, **kwargs) -> None:
    """将 DataFrame 写入系统剪贴板。

    需安装 pyperclip：pip install pyperclip
    """
    try:
        import pyperclip

        content = to_csv(df)
        pyperclip.copy(content)
    except ImportError:
        raise ImportError(
            "to_clipboard requires pyperclip to be installed. "
            "Install with: pip install pyperclip"
        )


def read_xml(
    path_or_buffer,
    xpath_regex: str = ".//row",
    row_name: str = "row",
    **kwargs,
) -> DataFrame:
    """从 XML 文件读取 DataFrame。

    需安装 lxml：pip install lxml
    """
    try:
        from lxml import etree
    except ImportError:
        raise ImportError(
            "read_xml requires lxml to be installed. " "Install with: pip install lxml"
        )

    if isinstance(path_or_buffer, str) and not path_or_buffer.strip().startswith("<"):
        tree = etree.parse(path_or_buffer)
        root = tree.getroot()
    else:
        if hasattr(path_or_buffer, "read"):
            content = path_or_buffer.read()
        else:
            content = path_or_buffer
        root = etree.fromstring(
            content.encode() if isinstance(content, str) else content
        )

    rows = root.findall(xpath_regex)
    if not rows:
        return DataFrame()

    # 收集所有列名
    all_cols = set()
    row_data_list = []
    for row in rows:
        row_dict = {}
        for child in row:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            row_dict[tag] = child.text
            all_cols.add(tag)
        # 也检查属性
        for attr_name, attr_value in row.attrib.items():
            row_dict[attr_name] = attr_value
            all_cols.add(attr_name)
        row_data_list.append(row_dict)

    # 构建 DataFrame
    data = {col: [row.get(col) for row in row_data_list] for col in all_cols}
    return DataFrame(data)


def to_xml(
    df: DataFrame,
    path_or_buffer=None,
    index: bool = True,
    root_name: str = "data",
    row_name: str = "row",
    **kwargs,
) -> Optional[str]:
    """将 DataFrame 写入 XML 文件或返回 XML 字符串。"""
    from xml.etree import ElementTree as ET

    root = ET.Element(root_name)
    for i in range(len(df)):
        row_elem = ET.SubElement(root, row_name)
        if index:
            idx_val = df._index[i] if df._index and i < len(df._index) else i
            row_elem.set("index", str(idx_val))
        for col in df.columns:
            val = df[col].values[i]
            col_elem = ET.SubElement(row_elem, str(col))
            col_elem.text = str(val) if val is not None else ""

    xml_bytes = ET.tostring(root, encoding="unicode", xml_declaration=True)

    if path_or_buffer:
        if hasattr(path_or_buffer, "write"):
            path_or_buffer.write(xml_bytes)
        else:
            with open(path_or_buffer, "w", encoding="utf-8") as f:
                f.write(xml_bytes)
        return None
    return xml_bytes


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

    需安装 pyreadstat：pip install pyreadstat
    """
    try:
        import pyreadstat

        # 转换为 pandas DataFrame（如果可用），否则手动构建
        try:
            import pandas as pd

            pdf = pd.DataFrame({c: list(df[c].values) for c in df.columns})
            pyreadstat.write_dta(path, pdf)
        except ImportError:
            # 简化实现：写入 CSV 格式作为回退
            import warnings

            warnings.warn("pyreadstat requires pandas. Writing as CSV instead.")
            to_csv(df, path.replace(".dta", ".csv"))
    except ImportError:
        raise ImportError(
            "to_stata requires pyreadstat to be installed. "
            "Install with: pip install pyreadstat"
        )


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


def to_csv(
    df: DataFrame, path=None, sep: str = ",", index: bool = True, **kwargs
) -> Optional[str]:
    """将 DataFrame 写入 CSV 文件或返回 CSV 字符串。"""
    from .rspandas import write_csv_string as _write_csv_string

    # 使用 Rust 层的 write_csv_string
    cols = list(df.columns)
    series_list = [df._inner.get_column(c) for c in cols]
    csv_str = _write_csv_string(cols, series_list, sep, index)

    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(csv_str)
        return None
    return csv_str
