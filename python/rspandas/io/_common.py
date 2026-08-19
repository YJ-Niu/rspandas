"""IO 公共辅助：日期解析/dtype 推断/文本读取器等内部工具

由 rspandas/io.py 拆分而来，向后兼容通过 :mod:`rspandas.io` 包入口保证。
"""

from __future__ import annotations

from ..dataframe import DataFrame
from ..series import Series  # noqa: F401  # 部分函数需要
from typing import Dict, List, Optional

from datetime import datetime


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
    """将字符串列表解析为 datetime 对象列表。

    支持 date_format 为 dict（按列名指定格式）或 str（统一格式）。
    与 pandas 行为对齐：date_format 作为 dict 时，key 为列名，value 为格式字符串。
    """
    import datetime as _datetime_module

    out = []
    for v in values:
        if v is None or v == "":
            out.append(None)
            continue
        if not isinstance(v, str):
            out.append(v)
            continue
        # 如果 date_format 是字符串，使用指定格式解析
        if isinstance(date_format, str):
            try:
                dt = _datetime_module.datetime.strptime(v, date_format)
                out.append(dt)
            except (ValueError, TypeError):
                out.append(v)
        else:
            # 无格式或 dict 格式（由调用方按列分发）
            from .._datetime import _parse_iso

            try:
                dt = _parse_iso(v)
                out.append(dt)
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

    n = len(values)
    # 快速判断是否全部为 None
    has_value = False
    for v in values:
        if v is not None:
            has_value = True
            break
    if not has_value:
        return values

    # 尝试 int：单趟边校验边转换，避免 all_int 通过后二次 int() 解析
    int_result = [None] * n
    all_int = True
    for i, v in enumerate(values):
        if v is None:
            continue
        if isinstance(v, bool):
            # bool 是 int 的子类，但我们想单独处理 bool
            all_int = False
            break
        if isinstance(v, int):
            int_result[i] = v
        elif isinstance(v, str):
            try:
                int_result[i] = int(v)
            except (ValueError, TypeError):
                all_int = False
                break
        else:
            all_int = False
            break

    if all_int:
        return int_result

    # 尝试 float：单趟边校验边转换
    float_result = [None] * n
    all_float = True
    for i, v in enumerate(values):
        if v is None:
            continue
        if isinstance(v, bool):
            all_float = False
            break
        if isinstance(v, (int, float)):
            float_result[i] = float(v)
        elif isinstance(v, str):
            try:
                float_result[i] = float(v)
            except (ValueError, TypeError):
                all_float = False
                break
        else:
            all_float = False
            break

    if all_float:
        return float_result

    # 尝试 bool
    bool_set = {"True", "TRUE", "true", "False", "FALSE", "false"}
    all_bool = True
    for v in values:
        if v is None:
            continue
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

    # 默认：逐值类型推断（与 pandas 行为一致：混合类型列保持 object dtype）
    # 如果列中全为 datetime 对象，不转换（保持 datetime 类型）
    if all(v is None or isinstance(v, datetime) for v in values):
        return values

    result = []
    for v in values:
        if v is None:
            result.append(None)
        elif isinstance(v, bool):
            result.append(v)
        elif isinstance(v, (int, float)):
            result.append(v)
        elif isinstance(v, str):
            # 尝试 int → float → bool → str
            try:
                result.append(int(v))
            except (ValueError, TypeError):
                try:
                    result.append(float(v))
                except (ValueError, TypeError):
                    if v in ("True", "TRUE", "true"):
                        result.append(True)
                    elif v in ("False", "FALSE", "false"):
                        result.append(False)
                    else:
                        result.append(v)
        else:
            result.append(v)
    return result


# ============================================================================
# PyArrow 可选依赖：仅用于 ORC 格式（read_orc/to_orc）及 to_arrow()/from_arrow()。
# Parquet 和 Feather (Arrow IPC) 已由 Rust 层 arrow/parquet crate 实现，无需 pyarrow。
# ORC 的 DataFrame↔Table 转换也复用 Rust 层 IPC bytes 桥接（见 dataframe.to_arrow），
# 此处不再在 Python 层做类型推断循环。
# ============================================================================


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
    # 无压缩：使用 Rust 层读取文件内容
    from ..rspandas import read_file_to_string as _read_file_to_string

    return _read_file_to_string(path)


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
        from ..dataframe import DataFrame as _DataFrame
        import io as _io
        from .. import read_csv

        if size is None:
            size = self._chunksize
        if self._pos >= len(self._lines):
            return _DataFrame()

        # 取出本 chunk 的行
        end = min(self._pos + size + 1, len(self._lines))
        chunk_lines = self._lines[self._pos : end]
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


def _rows_to_dict(rows: List[list], col_names: Optional[List[str]]) -> Dict[str, list]:
    """将行列表转为列字典（列表推导式优化）。"""
    if col_names is None:
        n_cols = max(len(r) for r in rows) if rows else 0
        col_names = [str(i) for i in range(n_cols)]
    return {
        col_names[i]: [r[i] if i < len(r) else None for r in rows]
        for i in range(len(col_names))
    }


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
