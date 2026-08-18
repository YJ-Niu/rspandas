"""CSV 读写：read_csv / read_csv_chunked / to_csv

由 rspandas/io.py 拆分而来，向后兼容通过 :mod:`rspandas.io` 包入口保证。
"""

from __future__ import annotations

from ..dataframe import DataFrame
from ..series import Series  # noqa: F401  # 部分函数需要
from typing import Optional

from ._common import (
    _NO_DEFAULT,
    _TextFileReader,
    _apply_dtype,
    _infer_column_type,
    _parse_cols_items,
    _parse_date_series,
    _read_content,
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
    from ..dataframe import DataFrame as _DataFrame

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
    # 7. 使用 Rust 解析 CSV（释放 GIL，不依赖 Python csv 模块）
    # ------------------------------------------------------------------
    csv_content = "\n".join(lines)
    # 处理 lineterminator：Rust csv crate 默认支持 \n 和 \r\n
    if lineterminator is not None and lineterminator not in ("\n", "\r\n"):
        csv_content = csv_content.replace(lineterminator, "\n")

    # 将 quoting 整数转换为布尔值（QUOTE_NONE=3 时禁用引号处理）
    quoting_enabled = quoting != 3

    # 分隔符必须为单字节字符
    if len(sep) != 1:
        return _DataFrame()

    from ..rspandas import parse_csv_raw as _parse_csv_raw

    try:
        rust_headers, cols_data = _parse_csv_raw(
            csv_content,
            has_header if header_rows_count <= 1 else False,
            delimiter=ord(sep),
            quote=ord(quotechar),
            quoting=quoting_enabled,
            double_quote=doublequote,
            escape=ord(escapechar) if escapechar is not None else None,
            skip_initial_space=skipinitialspace,
        )
    except Exception:
        return _DataFrame()

    if not cols_data and not rust_headers:
        return _DataFrame()

    # ------------------------------------------------------------------
    # 8. 解析表头与数据行
    # ------------------------------------------------------------------
    # cols_data 是列导向的: List[List[Optional[str]]]，每列是一个列表
    # rust_headers 是 Rust 解析出的列名
    if has_header:
        if header_rows_count > 1:
            # MultiIndex 表头：Rust 以 has_header=False 解析，全部数据在 cols_data 中
            # 前 header_rows_count 行的每一列值构成 MultiIndex 表头
            ncols = len(cols_data)
            nrows_total = len(cols_data[0]) if cols_data else 0
            # 提取表头行：每列的前 header_rows_count 个值
            header_rows_list = []
            for r in range(min(header_rows_count, nrows_total)):
                header_rows_list.append(
                    [
                        cols_data[c][r] if r < len(cols_data[c]) else None
                        for c in range(ncols)
                    ]
                )
            # 使用最后一行表头作为列名
            cols = [str(h) if h is not None else "" for h in header_rows_list[-1]]
            # 数据行：跳过表头行
            data_start = header_rows_count
            data_cols = []
            for c in range(ncols):
                col = cols_data[c]
                data_cols.append(col[data_start:])
            cols_data = data_cols
        else:
            # 单行表头：Rust 已分离表头和数据
            cols = list(rust_headers)
            # cols_data 已经是数据行（不含表头）
    else:
        # 无表头：Rust 自动生成 "col0", "col1", ... 作为列名
        # 但我们使用用户期望的默认列名（0, 1, 2, ...）
        ncols = len(cols_data)
        cols = [str(i) for i in range(ncols)]
        # 重命名 rust_headers 为我们的默认列名（实际上不需要，因为 cols_data 直接是数据）

    # 显式 names 覆盖
    if names is not None and names is not _NO_DEFAULT:
        names_list = list(names)
        if len(names_list) >= len(cols):
            cols = names_list[: len(cols)]
        else:
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
    # 9. 对齐数据列数（补齐或截断）
    # ------------------------------------------------------------------
    if len(cols_data) < len(cols):
        # 补齐缺失的列
        nrows = len(cols_data[0]) if cols_data else 0
        for _ in range(len(cols) - len(cols_data)):
            cols_data.append([None] * nrows)
    elif len(cols_data) > len(cols):
        cols_data = cols_data[: len(cols)]

    # ------------------------------------------------------------------
    # 10. 处理 nrows
    # ------------------------------------------------------------------
    if nrows is not None and cols_data:
        cols_data = [col[:nrows] for col in cols_data]

    # ------------------------------------------------------------------
    # 11. 构建 dict[str, list]
    # ------------------------------------------------------------------
    data = {c: list(col_data) for c, col_data in zip(cols, cols_data)}

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
                    # 若 date_format 为 dict，按 new_name 获取对应格式
                    col_fmt = date_format
                    if isinstance(date_format, dict):
                        col_fmt = date_format.get(new_name, None)
                    parsed = _parse_date_series(combined, col_fmt, dayfirst)
                    data[new_name] = parsed
                else:
                    date_cols.append(src_cols)

        for c in date_cols:
            if c in data:
                # 若 date_format 为 dict，按列名获取对应格式
                col_fmt = date_format
                if isinstance(date_format, dict):
                    col_fmt = date_format.get(c, None)
                data[c] = _parse_date_series(data[c], col_fmt, dayfirst)

    # ------------------------------------------------------------------
    # 18. 处理 dtype
    # ------------------------------------------------------------------
    _dtype_overrides = {}  # 收集 dtype 覆盖信息（用户显式指定的 dtype）
    if dtype is not None:
        from ..series import _dtype_to_str as _norm_dtype

        if isinstance(dtype, dict):
            for col_name, dt in dtype.items():
                if col_name in data:
                    dt_str = _norm_dtype(dt)
                    data[col_name] = _apply_dtype(data[col_name], dt_str)
                    _dtype_overrides[col_name] = dt_str
            # 对 dict 中未指定的列，仍进行自动类型推断
            for c in cols:
                if c in data and c not in dtype:
                    data[c] = _infer_column_type(data[c])
        else:
            dt_str = _norm_dtype(dtype)
            for c in cols:
                data[c] = _apply_dtype(data[c], dt_str)
            for c in cols:
                _dtype_overrides[c] = dt_str
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
                if _dtype_overrides:
                    df._col_dtypes.update(_dtype_overrides)
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
                if _dtype_overrides:
                    df._col_dtypes.update(_dtype_overrides)
                return df

    df = _DataFrame(data)
    if _dtype_overrides:
        df._col_dtypes.update(_dtype_overrides)
    return df


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
        分隔符（默认逗号，仅支持单字节字符）。
    header : bool
        是否有表头（默认 True）。

    Yields
    ------
    DataFrame
        每次产出一个 chunk_size 行的 DataFrame。
    """
    # 使用 Rust 层 read_csv_chunks（支持自定义单字节分隔符）
    if len(delimiter) != 1:
        # 多字节分隔符不支持，回退为空
        return

    try:
        from ..rspandas import read_csv_chunks as _read_csv_chunks_rust

        with open(path, "r", encoding=encoding, newline="") as f:
            content = f.read()
        chunks = _read_csv_chunks_rust(
            content,
            header,
            chunk_size,
            delimiter=ord(delimiter),
            quote=None,
            quoting=None,
            double_quote=None,
            escape=None,
            skip_initial_space=None,
        )
        from ..series import Series as _Series
        from ..dataframe import DataFrame as _DataFrame

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
    except Exception:
        return


def to_csv(
    df: DataFrame, path=None, sep: str = ",", index: bool = True, **kwargs
) -> Optional[str]:
    """将 DataFrame 写入 CSV 文件或返回 CSV 字符串。"""
    from ..rspandas import write_csv_string as _write_csv_string

    # 使用 Rust 层的 write_csv_string
    cols = list(df.columns)
    series_list = [df._inner.get_column(c) for c in cols]
    csv_str = _write_csv_string(cols, series_list, True, sep)

    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(csv_str)
        return None
    return csv_str
