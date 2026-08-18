"""IO 扩展: JSON / Excel / Parquet / Pickle / SQL 读写。

所有函数都接受/返回 DataFrame，与 pandas IO API 对齐。"""

from __future__ import annotations

from .dataframe import DataFrame
from typing import Any, Dict, List, Optional, Tuple, Union

import json as _json
import pickle as _pickle


# 哨兵对象：用于表示参数未传入（区别于 None）
class _NoDefault:
    """哨兵类型，用于表示参数未传入。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<no_default>"


_NO_DEFAULT = _NoDefault()


# 辅助函数
def _parse_date_series(values, date_format, dayfirst):
    """将字符串列表解析为 ISO 格式 datetime 字符串列表。"""
    from ._datetime import _parse_iso

    out = []
    for v in values:
        if v is None or v == "":
            out.append(None)
            continue
        if not isinstance(v, str):
            out.append(v)
            continue
        try:
            dt = _parse_iso(v)
            out.append(dt.isoformat())
        except (ValueError, TypeError):
            out.append(v)
    return out


def _apply_dtype(values, dtype_str: str):
    """将值列表应用 dtype 转换。"""
    dtype_str = dtype_str.lower()
    if dtype_str in (
        "int",
        "int64",
        "int32",
        "int16",
        "int8",
        "uint64",
        "uint32",
        "uint16",
        "uint8",
    ):
        return [None if v is None else int(v) for v in values]
    if dtype_str in ("float", "float64", "float32"):
        return [None if v is None else float(v) for v in values]
    if dtype_str == "bool":
        return [None if v is None else bool(v) for v in values]
    if dtype_str in ("str", "object", "string"):
        return [None if v is None else str(v) for v in values]
    # 其他类型不转换
    return values


def _parse_cols_items(parse_dates):
    """规范化 parse_dates 列表项。"""
    return list(parse_dates)


def _infer_column_type(values):
    """自动推断列类型（与 pandas 行为一致）。

    尝试顺序: int -> float -> bool -> str/object
    NaN/None 跳过不参与推断，但保留为 None。
    """
    if not values:
        return values

    # 全为 None 的情况
    non_null = [v for v in values if v is not None]
    if not non_null:
        return values

    # 尝试 int
    all_int = True
    for v in non_null:
        if isinstance(v, bool):
            # bool 是 int 的子类，但我们想单独处理 bool
            all_int = False
            break
        if isinstance(v, int):
            continue
        if isinstance(v, str):
            try:
                int(v)
            except (ValueError, TypeError):
                all_int = False
                break
        else:
            all_int = False
            break

    if all_int:
        return [None if v is None else int(v) for v in values]

    # 尝试 float
    all_float = True
    for v in non_null:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            continue
        if isinstance(v, str):
            try:
                float(v)
            except (ValueError, TypeError):
                all_float = False
                break
        else:
            all_float = False
            break

    if all_float:
        return [None if v is None else float(v) for v in values]

    # 尝试 bool
    bool_set = {"True", "TRUE", "true", "False", "FALSE", "false"}
    all_bool = True
    for v in non_null:
        if isinstance(v, bool):
            continue
        if isinstance(v, str) and v in bool_set:
            continue
        all_bool = False
        break

    if all_bool:
        return [
            None if v is None else (v in {"True", "TRUE", "true"} or v is True)
            for v in values
        ]

    # 默认：保持 str/object
    return [
        None if v is None else str(v) if not isinstance(v, str) else v for v in values
    ]


# ============================================================================
# PyArrow 可选依赖：仅用于 ORC 格式（read_orc/to_orc）及 to_arrow()/from_arrow()。
# Parquet 和 Feather (Arrow IPC) 已由 Rust 层 arrow/parquet crate 实现，无需 pyarrow。
# ORC 的 DataFrame↔Table 转换也复用 Rust 层 IPC bytes 桥接（见 dataframe.to_arrow），
# 此处不再在 Python 层做类型推断循环。
# ============================================================================


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
        index: bool = True,
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
    index: bool = True,
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
    index : bool, default True
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
    from .rspandas import _DataFrame
    from .rspandas import read_parquet as _read_parquet_rust

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
    from .rspandas import write_parquet as _write_parquet_rust

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
    from .rspandas import _DataFrame
    from .rspandas import read_feather as _read_feather_rust

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
    from .rspandas import write_feather as _write_feather_rust

    cols = list(df.columns)
    series_list = [df._inner.get_column(c) for c in cols]
    _write_feather_rust(path, cols, series_list, compression or "uncompressed")


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


def _read_content(filepath_or_buffer, compression, encoding, encoding_errors):
    """读取文件内容，支持压缩格式和 file-like 对象。"""
    import gzip
    import bz2
    import zipfile
    import lzma

    # 处理 file-like 对象
    if hasattr(filepath_or_buffer, "read"):
        content = filepath_or_buffer.read()
        if isinstance(content, bytes):
            content = content.decode(encoding or "utf-8", errors=encoding_errors)
        return content

    # 处理字符串路径
    path = str(filepath_or_buffer)

    # 推断压缩格式
    if compression == "infer":
        if path.endswith(".gz"):
            compression = "gzip"
        elif path.endswith(".bz2"):
            compression = "bz2"
        elif path.endswith(".zip"):
            compression = "zip"
        elif path.endswith(".xz"):
            compression = "xz"
        elif path.endswith(".zst"):
            compression = "zst"
        else:
            compression = None

    enc = encoding or "utf-8"

    if compression == "gzip":
        with gzip.open(path, "rt", encoding=enc, errors=encoding_errors) as f:
            return f.read()
    if compression == "bz2":
        with bz2.open(path, "rt", encoding=enc, errors=encoding_errors) as f:
            return f.read()
    if compression == "xz":
        with lzma.open(path, "rt", encoding=enc, errors=encoding_errors) as f:
            return f.read()
    if compression == "zip":
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if not names:
                return ""
            with zf.open(names[0]) as f:
                content = f.read()
                return content.decode(enc, errors=encoding_errors)
    # 无压缩
    with open(path, "r", encoding=enc, errors=encoding_errors) as f:
        return f.read()


class _TextFileReader:
    """分块读取 CSV 文件的迭代器（与 pandas TextFileReader 行为对齐）。"""

    def __init__(
        self,
        filepath_or_buffer,
        sep=",",
        header="infer",
        names=None,
        index_col=None,
        usecols=None,
        dtype=None,
        nrows=None,
        encoding=None,
        chunksize=None,
        compression="infer",
        skiprows=None,
        skipfooter: int = 0,
        skip_blank_lines: bool = True,
        na_values=None,
        keep_default_na: bool = True,
        na_filter: bool = True,
        parse_dates=None,
        date_format=None,
        dayfirst: bool = False,
        converters=None,
        true_values=None,
        false_values=None,
        quotechar: str = '"',
        quoting=0,
        comment=None,
        decimal: str = ".",
        thousands=None,
        encoding_errors: str = "strict",
        doublequote: bool = True,
        escapechar=None,
        skipinitialspace: bool = False,
        lineterminator=None,
    ):
        self._filepath_or_buffer = filepath_or_buffer
        self._sep = sep
        self._header = header
        self._names = names
        self._index_col = index_col
        self._usecols = usecols
        self._dtype = dtype
        self._nrows = nrows
        self._encoding = encoding
        self._chunksize = chunksize or 1000
        self._compression = compression
        self._skiprows = skiprows
        self._skipfooter = skipfooter
        self._skip_blank_lines = skip_blank_lines
        self._na_values = na_values
        self._keep_default_na = keep_default_na
        self._na_filter = na_filter
        self._parse_dates = parse_dates
        self._date_format = date_format
        self._dayfirst = dayfirst
        self._converters = converters
        self._true_values = true_values
        self._false_values = false_values
        self._quotechar = quotechar
        self._quoting = quoting
        self._comment = comment
        self._decimal = decimal
        self._thousands = thousands
        self._encoding_errors = encoding_errors
        self._doublequote = doublequote
        self._escapechar = escapechar
        self._skipinitialspace = skipinitialspace
        self._lineterminator = lineterminator

        # 读取全部内容并按行分割
        self._content = _read_content(
            filepath_or_buffer, compression, encoding, encoding_errors
        )
        self._lines = self._content.splitlines(keepends=False)
        if skip_blank_lines:
            self._lines = [ln for ln in self._lines if ln.strip() != ""]
        self._pos = 0
        # 应用 skiprows
        if skiprows is not None:
            if callable(skiprows):
                self._lines = [
                    ln for i, ln in enumerate(self._lines) if not skiprows(i)
                ]
            elif isinstance(skiprows, int):
                self._lines = self._lines[skiprows:]
            else:
                skip_set = set(skiprows)
                self._lines = [
                    ln for i, ln in enumerate(self._lines) if i not in skip_set
                ]
        # 应用 skipfooter
        if skipfooter > 0:
            self._lines = (
                self._lines[:-skipfooter] if skipfooter < len(self._lines) else []
            )

    def __iter__(self):
        return self

    def __next__(self):
        chunk = self.get_chunk(self._chunksize)
        if chunk is None or len(chunk) == 0:
            raise StopIteration
        return chunk

    def get_chunk(self, size=None):
        """读取下一个 chunk。"""
        from .dataframe import DataFrame as _DataFrame
        import io as _io

        if size is None:
            size = self._chunksize
        if self._pos >= len(self._lines):
            return _DataFrame()

        # 取出本 chunk 的行
        end = min(self._pos + size + 1, len(self._lines))
        chunk_lines = self._lines[self._pos:end]
        self._pos = end

        # 第一行可能是表头
        has_header = True
        if self._header is None or self._header is False:
            has_header = False
        elif self._header == "infer":
            has_header = True

        if has_header and self._pos == end:  # 第一次读取
            header_line = chunk_lines[0]
            data_lines = chunk_lines[1:]
            # 保存表头供后续 chunk 使用
            self._cached_header = header_line
        elif has_header and hasattr(self, "_cached_header"):
            # 后续 chunk 使用第一次的表头
            header_line = self._cached_header
            data_lines = chunk_lines
            # 已经在本次循环里包含了下一行数据，但需要回退 pos 以便下次不重复
            # 简化：本次不重新加表头
        else:
            header_line = None
            data_lines = chunk_lines

        # 简化：直接调用 read_csv 处理本 chunk
        # 重建 CSV 文本
        if has_header and hasattr(self, "_cached_header"):
            text = self._cached_header + "\n" + "\n".join(data_lines)
        else:
            text = "\n".join(chunk_lines)

        return read_csv(
            _io.StringIO(text),
            sep=self._sep,
            header=self._header if self._pos <= size + 1 else None,
            names=self._names if self._pos > size + 1 else _NO_DEFAULT,
            index_col=self._index_col,
            usecols=self._usecols,
            dtype=self._dtype,
            nrows=size,
            na_values=self._na_values,
            keep_default_na=self._keep_default_na,
            na_filter=self._na_filter,
            parse_dates=self._parse_dates,
            date_format=self._date_format,
            dayfirst=self._dayfirst,
            converters=self._converters,
            true_values=self._true_values,
            false_values=self._false_values,
            quotechar=self._quotechar,
            quoting=self._quoting,
            comment=self._comment,
            decimal=self._decimal,
            thousands=self._thousands,
            encoding_errors=self._encoding_errors,
            doublequote=self._doublequote,
            escapechar=self._escapechar,
            skipinitialspace=self._skipinitialspace,
            lineterminator=self._lineterminator,
        )


def read_csv(
    filepath_or_buffer,
    sep=_NO_DEFAULT,
    delimiter=None,
    header="infer",
    names=_NO_DEFAULT,
    index_col=None,
    usecols=None,
    dtype=None,
    engine=None,
    converters=None,
    true_values=None,
    false_values=None,
    skipinitialspace: bool = False,
    skiprows=None,
    skipfooter: int = 0,
    nrows=None,
    na_values=None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    skip_blank_lines: bool = True,
    parse_dates=None,
    date_format=None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    iterator: bool = False,
    chunksize=None,
    compression: str = "infer",
    encoding: str = None,
    encoding_errors: str = "strict",
    lineterminator=None,
    quotechar: str = '"',
    quoting=0,
    doublequote: bool = True,
    escapechar=None,
    comment=None,
    decimal: str = ".",
    thousands=None,
    storage_options=None,
    dtype_backend=None,
    **kwargs,
) -> DataFrame:
    """读取 CSV 文件为 DataFrame。

    与 ``pandas.read_csv`` 签名对齐。支持的参数子集见下方说明，
    其他参数为兼容性参数（若传入则忽略或部分支持）。

    Parameters
    ----------
    filepath_or_buffer : str / path / file-like
        CSV 文件路径或带 read() 方法的对象。
    sep : str, default ','
        字段分隔符。``delimiter`` 为其别名。
    header : int / Sequence[int] / 'infer' / None, default 'infer'
        用作列名的行号。``None`` 表示无表头。
    names : Sequence, optional
        显式指定列名（若指定则覆盖 ``header`` 行）。
    index_col : int / str / Sequence, optional
        用作行索引的列。
    usecols : list / callable, optional
        要读取的列子集。支持列名、列索引或 callable（对列名求值）。
    dtype : type / dict, optional
        列数据类型。
    engine : {'c', 'python', 'pyarrow'}, optional
        解析引擎（保留参数，当前固定走 Rust 引擎）。
    converters : dict, optional
        列级转换函数。
    true_values / false_values : list, optional
        视作 True/False 的额外值。
    skiprows : int / list / callable, optional
        要跳过的行号或行数。
    skipfooter : int, default 0
        文件尾部要跳过的行数。
    nrows : int, optional
        读取的数据行数（不含表头）。
    na_values : Hashable / Iterable / dict, optional
        额外视作 NaN 的值。
    keep_default_na : bool, default True
        是否保留默认 NaN 值集合。
    na_filter : bool, default True
        是否进行缺失值检测。
    skip_blank_lines : bool, default True
        跳过空行而不是解释为 NaN。
    parse_dates : bool / list / dict, optional
        尝试解析为日期的列。
    dayfirst : bool, default False
        DD/MM 日期格式。
    compression : str, default 'infer'
        压缩格式：'infer'/'gzip'/'bz2'/'zip'/'xz'/'zst'/'tar'/None。
    encoding : str, optional
        文件编码。
    quotechar : str, default '"'
        引用字符。
    comment : str, optional
        行注释字符（该字符之后的内容被忽略）。
    decimal : str, default '.'
        小数点字符。
    thousands : str, optional
        千位分隔符。
    iterator / chunksize : bool / int, optional
        返回 TextFileReader 用于分块读取。
    **kwargs
        其他兼容性参数（忽略）。

    Returns
    -------
    DataFrame
    """
    from .dataframe import DataFrame as _DataFrame
    import csv as _csv
    import io as _io

    # ------------------------------------------------------------------
    # 1. 处理 sep / delimiter 别名
    # ------------------------------------------------------------------
    if delimiter is not None:
        sep = delimiter
    if sep is _NO_DEFAULT or sep is None:
        sep = ","

    # ------------------------------------------------------------------
    # 2. iterator / chunksize 模式：返回 TextFileReader
    # ------------------------------------------------------------------
    if iterator or chunksize is not None:
        return _TextFileReader(
            filepath_or_buffer,
            sep=sep,
            header=header,
            names=names,
            index_col=index_col,
            usecols=usecols,
            dtype=dtype,
            nrows=nrows,
            encoding=encoding,
            chunksize=chunksize,
            compression=compression,
            skiprows=skiprows,
            skipfooter=skipfooter,
            skip_blank_lines=skip_blank_lines,
            na_values=na_values,
            keep_default_na=keep_default_na,
            na_filter=na_filter,
            parse_dates=parse_dates,
            date_format=date_format,
            dayfirst=dayfirst,
            converters=converters,
            true_values=true_values,
            false_values=false_values,
            quotechar=quotechar,
            quoting=quoting,
            comment=comment,
            decimal=decimal,
            thousands=thousands,
            encoding_errors=encoding_errors,
            doublequote=doublequote,
            escapechar=escapechar,
            skipinitialspace=skipinitialspace,
            lineterminator=lineterminator,
        )

    # ------------------------------------------------------------------
    # 3. 读取原始内容（处理压缩）
    # ------------------------------------------------------------------
    content = _read_content(filepath_or_buffer, compression, encoding, encoding_errors)

    if not content:
        return _DataFrame()

    # ------------------------------------------------------------------
    # 4. 处理 skiprows
    # ------------------------------------------------------------------
    lines = content.splitlines(keepends=False)

    # 解析 comment: 行内注释字符
    if comment is not None:
        new_lines = []
        for ln in lines:
            idx = ln.find(comment)
            if idx >= 0:
                ln = ln[:idx]
            new_lines.append(ln)
        lines = new_lines

    # skip_blank_lines: 去掉空行
    if skip_blank_lines:
        lines = [ln for ln in lines if ln.strip() != ""]
    else:
        # 保留空行（后续解释为 NaN）
        pass

    # callable / list / int 形式的 skiprows
    if skiprows is not None:
        if callable(skiprows):
            lines = [ln for i, ln in enumerate(lines) if not skiprows(i)]
        elif isinstance(skiprows, int):
            lines = lines[skiprows:]
        else:
            # list of int
            skip_set = set(skiprows)
            lines = [ln for i, ln in enumerate(lines) if i not in skip_set]

    # ------------------------------------------------------------------
    # 5. 确定 header 行与数据行
    # ------------------------------------------------------------------
    # pandas 行为:
    # - names 显式传入时: header 默认变为 None（不使用文件表头）
    # - header=0 + names: 用 names 覆盖文件第一行表头
    has_header = True
    header_rows_count = 1
    effective_header = header

    # 处理 names 与 header 的交互（pandas 行为）
    if names is not None and names is not _NO_DEFAULT:
        if header == "infer":
            # names 显式传入时，默认不使用文件表头
            effective_header = None
            has_header = False
        elif isinstance(header, int) and header == 0:
            # header=0 + names: 跳过文件第一行，用 names 覆盖
            effective_header = 0
            has_header = True
        elif header is None or header is False:
            has_header = False
        else:
            has_header = True
    else:
        if header is None or header is False:
            has_header = False
        elif header == "infer":
            has_header = True
        elif isinstance(header, int):
            has_header = True
        elif isinstance(header, (list, tuple)):
            header_rows_count = len(header)
            has_header = True

    # 处理 header 为行号（0-indexed）
    if isinstance(effective_header, int) and effective_header > 0:
        lines = lines[effective_header:]
    elif isinstance(effective_header, (list, tuple)):
        header_rows_count = len(effective_header)

    # ------------------------------------------------------------------
    # 6. 跳过 footer
    # ------------------------------------------------------------------
    if skipfooter > 0:
        lines = lines[:-skipfooter] if skipfooter < len(lines) else []

    # ------------------------------------------------------------------
    # 7. 使用 Python csv 解析（统一路径，便于处理各种参数）
    # ------------------------------------------------------------------
    # 处理 skipinitialspace
    csv_sep = sep
    csv_kwargs = {
        "delimiter": csv_sep,
        "quotechar": quotechar,
        "quoting": quoting,
        "doublequote": doublequote,
    }
    if escapechar is not None:
        csv_kwargs["escapechar"] = escapechar
    if skipinitialspace:
        csv_kwargs["skipinitialspace"] = True
    if lineterminator is not None:
        csv_kwargs["lineterminator"] = lineterminator

    reader = _csv.reader(_io.StringIO("\n".join(lines)), **csv_kwargs)
    rows = list(reader)
    if not rows:
        return _DataFrame()

    # ------------------------------------------------------------------
    # 8. 解析表头与数据行
    # ------------------------------------------------------------------
    if has_header:
        if header_rows_count > 1:
            # MultiIndex 表头：合并所有表头行
            header_rows = rows[:header_rows_count]
            data_rows = rows[header_rows_count:]
            # 简化：使用最后一行作为列名（与 pandas 行为有差异，但满足基本需求）
            cols = list(header_rows[-1])
        else:
            cols = list(rows[0])
            data_rows = rows[1:]
    else:
        ncols = max(len(r) for r in rows)
        cols = [str(i) for i in range(ncols)]
        data_rows = rows

    # 显式 names 覆盖
    if names is not None and names is not _NO_DEFAULT:
        names_list = list(names)
        if len(names_list) >= len(cols):
            cols = names_list[: len(cols)]
        else:
            # names 长度小于列数，填充默认列名
            cols = names_list + [str(i) for i in range(len(names_list), len(cols))]

    # 空列名处理（pandas 行为）
    cols = [f"Unnamed: {i}" if c == "" else c for i, c in enumerate(cols)]

    # 列名去重（pandas 行为：foo, foo -> foo, foo.1）
    seen = {}
    new_cols = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    cols = new_cols

    # ------------------------------------------------------------------
    # 9. 对齐数据行长度（补齐或截断）
    # ------------------------------------------------------------------
    aligned_rows = []
    for r in data_rows:
        if len(r) < len(cols):
            r = list(r) + [None] * (len(cols) - len(r))
        else:
            r = list(r)[: len(cols)]
        aligned_rows.append(r)
    data_rows = aligned_rows

    # ------------------------------------------------------------------
    # 10. 处理 nrows
    # ------------------------------------------------------------------
    if nrows is not None:
        data_rows = data_rows[:nrows]

    # ------------------------------------------------------------------
    # 11. 构建 dict[str, list]
    # ------------------------------------------------------------------
    data = {c: [r[i] for r in data_rows] for i, c in enumerate(cols)}

    # ------------------------------------------------------------------
    # 12. 处理 usecols
    # ------------------------------------------------------------------
    if usecols is not None:
        if callable(usecols):
            kept = [c for c in cols if usecols(c)]
        else:
            usecols_list = list(usecols)
            # 区分整数索引与列名
            int_indices = [x for x in usecols_list if isinstance(x, int)]
            str_names = [x for x in usecols_list if isinstance(x, str)]
            kept = []
            for i, c in enumerate(cols):
                if i in int_indices or c in str_names:
                    kept.append(c)
        data = {c: v for c, v in data.items() if c in kept}
        cols = [c for c in cols if c in data]

    # ------------------------------------------------------------------
    # 13. 处理 na_values / keep_default_na / na_filter
    # ------------------------------------------------------------------
    if na_filter:
        # 默认 NaN 值集合
        default_na = {
            "",
            "#N/A",
            "#N/A N/A",
            "#NA",
            "-1.#IND",
            "-1.#QNAN",
            "-NaN",
            "-nan",
            "1.#IND",
            "1.#QNAN",
            "<NA>",
            "N/A",
            "NA",
            "NULL",
            "NaN",
            "None",
            "n/a",
            "nan",
            "null",
        }
        if keep_default_na:
            na_set = set(default_na)
            if na_values is not None:
                if isinstance(na_values, dict):
                    pass  # 逐列处理，下方单独处理
                else:
                    na_set.update(na_values)
        else:
            if na_values is not None:
                if isinstance(na_values, dict):
                    pass
                else:
                    na_set = set(na_values)
            else:
                na_set = set()

        # 应用 NaN 替换
        if isinstance(na_values, dict):
            for col_name, vals in na_values.items():
                col_na_set = set(vals)
                if keep_default_na:
                    col_na_set = col_na_set | default_na
                if col_name in data:
                    data[col_name] = [
                        None if v in col_na_set else v for v in data[col_name]
                    ]
        else:
            if na_set:
                for c in cols:
                    data[c] = [None if v in na_set else v for v in data[c]]

    # ------------------------------------------------------------------
    # 14. 处理 true_values / false_values
    # ------------------------------------------------------------------
    if true_values or false_values:
        tv_set = set(true_values) if true_values else set()
        fv_set = set(false_values) if false_values else set()
        # 加入大小写不敏感的 True/False
        tv_set.update({"True", "TRUE", "true"})
        fv_set.update({"False", "FALSE", "false"})
        for c in cols:
            col_vals = data[c]
            # 只在所有值都可识别为 bool 时才转换
            all_bool = True
            for v in col_vals:
                if v is None:
                    continue
                if v not in tv_set and v not in fv_set:
                    all_bool = False
                    break
            if all_bool:
                data[c] = [None if v is None else (v in tv_set) for v in col_vals]

    # ------------------------------------------------------------------
    # 15. 处理 thousands / decimal
    # ------------------------------------------------------------------
    if thousands is not None or decimal != ".":
        for c in cols:
            col_vals = data[c]
            new_vals = []
            for v in col_vals:
                if isinstance(v, str):
                    s = v
                    if thousands is not None:
                        s = s.replace(thousands, "")
                    if decimal != ".":
                        s = s.replace(decimal, ".")
                    # 尝试转为数值
                    try:
                        if "." in s:
                            new_vals.append(float(s))
                        else:
                            new_vals.append(int(s))
                    except ValueError:
                        new_vals.append(v)
                else:
                    new_vals.append(v)
            data[c] = new_vals

    # ------------------------------------------------------------------
    # 16. 处理 converters
    # ------------------------------------------------------------------
    if converters is not None:
        for key, fn in converters.items():
            if isinstance(key, int):
                if key < len(cols):
                    target_col = cols[key]
                else:
                    continue
            else:
                target_col = key
            if target_col in data:
                data[target_col] = [
                    None if v is None else fn(v) for v in data[target_col]
                ]

    # ------------------------------------------------------------------
    # 17. 处理 parse_dates / date_format / dayfirst
    # ------------------------------------------------------------------
    if parse_dates is not None:

        # 确定要解析的列
        date_cols = []
        if isinstance(parse_dates, bool):
            if parse_dates:
                # 尝试解析所有 object 列（仅限可解析的）
                for c in cols:
                    date_cols.append(c)
        elif isinstance(parse_dates, list):
            for x in _parse_cols_items(parse_dates):
                if isinstance(x, str):
                    date_cols.append(x)
                elif isinstance(x, int):
                    if x < len(cols):
                        date_cols.append(cols[x])
        elif isinstance(parse_dates, dict):
            for new_name, src_cols in parse_dates.items():
                # 多列合并为单个日期列
                if isinstance(src_cols, list):
                    # 合并多列为字符串后再解析
                    combined = []
                    for i in range(len(data[src_cols[0]])):
                        parts = [str(data[c][i]) for c in src_cols]
                        combined.append(" ".join(parts))
                    parsed = _parse_date_series(combined, date_format, dayfirst)
                    data[new_name] = parsed
                else:
                    date_cols.append(src_cols)

        for c in date_cols:
            if c in data:
                data[c] = _parse_date_series(data[c], date_format, dayfirst)

    # ------------------------------------------------------------------
    # 18. 处理 dtype
    # ------------------------------------------------------------------
    if dtype is not None:
        from .series import _dtype_to_str as _norm_dtype

        if isinstance(dtype, dict):
            for col_name, dt in dtype.items():
                if col_name in data:
                    data[col_name] = _apply_dtype(data[col_name], _norm_dtype(dt))
        else:
            dt_str = _norm_dtype(dtype)
            for c in cols:
                data[c] = _apply_dtype(data[c], dt_str)
    else:
        # 自动类型推断（与 pandas 行为一致）
        # 对每列尝试 int -> float -> bool -> str 的顺序推断
        for c in cols:
            if c in data:
                data[c] = _infer_column_type(data[c])

    # ------------------------------------------------------------------
    # 19. 处理 index_col
    # ------------------------------------------------------------------
    if index_col is not None and index_col is not False:
        if isinstance(index_col, (int, str)):
            if isinstance(index_col, int):
                if index_col < 0:
                    index_col = len(cols) + index_col
                col_name = cols[index_col] if index_col < len(cols) else None
            else:
                col_name = index_col
            if col_name in data:
                new_index = list(data[col_name])
                data.pop(col_name)
                df = _DataFrame(data, index=new_index)
                return df
        elif isinstance(index_col, (list, tuple)):
            # MultiIndex
            idx_names = []
            idx_data = []
            for x in index_col:
                col_name = cols[x] if isinstance(x, int) else x
                if col_name in data:
                    idx_names.append(col_name)
                    idx_data.append(list(data[col_name]))
                    data.pop(col_name)
            if idx_data:
                # 简化：使用第一列作为索引
                df = _DataFrame(data, index=list(idx_data[0]))
                return df

    return _DataFrame(data)


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
        分隔符（默认逗号，Rust 层仅支持逗号，其他分隔符回退到 Python 实现）。
    header : bool
        是否有表头（默认 True）。

    Yields
    ------
    DataFrame
        每次产出一个 chunk_size 行的 DataFrame。
    """
    # 优先调用 Rust 层 read_csv_chunks（仅支持逗号分隔符）
    if delimiter == ",":
        try:
            from .rspandas import read_csv_chunks as _read_csv_chunks_rust

            with open(path, "r", encoding=encoding, newline="") as f:
                content = f.read()
            chunks = _read_csv_chunks_rust(content, header, chunk_size)
            from .series import Series as _Series
            from .dataframe import DataFrame as _DataFrame

            for cols, series_list in chunks:
                # 构造 Series 列表
                py_series_list = []
                for s in series_list:
                    py_s = _Series.__new__(_Series)
                    py_s._inner = s
                    py_s._dtype_str = s.dtype
                    py_s._index = list(range(s.size))
                    py_s._name = s.name
                    py_series_list.append(py_s)
                # 直接构造 DataFrame
                df_data = {c: py_s.values for c, py_s in zip(cols, py_series_list)}
                yield _DataFrame(df_data)
            return
        except Exception:
            pass

    # 回退到 Python 实现
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
        col_names = [str(i) for i in range(n_cols)]
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
            batch = records[i : i + batch_size]  # noqa
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
    for tr in table.find_all("tr")[header or 0 :]:  # noqa
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows_data.append(cells)

    if header is not None and rows_data:
        col_names = rows_data[0]
        data_rows = rows_data[1:]
    else:
        col_names = [str(i) for i in range(len(rows_data[0]))] if rows_data else []
        data_rows = rows_data

    data = {
        col_names[i] if i < len(col_names) else str(i): [
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
