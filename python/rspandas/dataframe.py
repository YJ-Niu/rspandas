"""DataFrame: pandas-like 2D data structure with Rust backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from rspandas.rspandas import _DataFrame as _PyDataFrame, _Series as _PySeries  # type: ignore
from .series import Series


def _to_pylist_columns(data: Any, columns: Optional[List[str]]) -> Dict[str, list]:
    """将 dict/list 输入解析为 dict[str, list]。"""
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if isinstance(v, Series):
                result[k] = list(v.values)
            elif isinstance(v, _PySeries):
                result[k] = list(v.values)
            else:
                result[k] = list(v) if v is not None else []
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
            # list[list]
            if columns is None:
                columns = [f"col{i}" for i in range(len(data[0]))]
            result = {c: [] for c in columns}
            for row in data:
                for i, c in enumerate(columns):
                    result[c].append(row[i] if i < len(row) else None)
            return result

    raise TypeError(f"Cannot build DataFrame from {type(data).__name__}")


class DataFrame:
    """二维表格，对齐 pandas API。

    Examples:
        >>> df = DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        >>> df.shape
        (3, 2)
        >>> df['a'].sum()
        6
    """

    def __init__(
        self,
        data=None,
        columns: Optional[List[str]] = None,
        index=None,
        dtype=None,
    ):
        """构造 DataFrame。

        :param data: dict[str, list] | list[dict] | list[list]
        :param columns: list[str] | None
        :param index: MVP 忽略
        :param dtype: MVP 忽略
        """
        col_dict = _to_pylist_columns(data, columns)
        col_names = list(col_dict.keys())
        col_values = [col_dict[c] for c in col_names]

        # 校验每列长度一致
        n = len(col_values[0]) if col_values else 0
        for c, vs in zip(col_names, col_values):
            if len(vs) != n:
                raise ValueError(
                    f"column '{c}' has length {len(vs)} != {n}"
                )

        # 构造 Rust 端 Series
        rust_series_list = []
        for c, vs in zip(col_names, col_values):
            rust_series_list.append(_PySeries(vs, c))

        # 构造 Rust 端 DataFrame
        self._inner = _PyDataFrame(col_names, rust_series_list)

        self._columns: List[str] = col_names
        self._nrows: int = n
        self._index = index if index is not None else list(range(n))

    # ---------- 属性 ----------

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._nrows, len(self._columns))

    @property
    def columns(self) -> List[str]:
        return list(self._columns)

    @columns.setter
    def columns(self, value: List[str]) -> None:
        if len(value) != len(self._columns):
            raise ValueError(
                f"new columns length {len(value)} != old {len(self._columns)}"
            )
        # 更新 Rust 端 column 名称 - 通过重命名每个 series
        # MVP 简化: 用 values 重建
        old_data = {c: list(self._inner.get_column(c).values) for c in self._columns}
        new_series = [_PySeries(old_data[c], value[i]) for i, c in enumerate(self._columns)]
        self._inner = _PyDataFrame(list(value), new_series)
        self._columns = list(value)

    @property
    def dtypes(self) -> Dict[str, str]:
        result = {}
        for c in self._columns:
            ser = self._inner.get_column(c)
            result[c] = ser.dtype
        return result

    @property
    def index(self):
        return self._index

    @property
    def empty(self) -> bool:
        return self._nrows == 0 or len(self._columns) == 0

    @property
    def size(self) -> int:
        return self._nrows * len(self._columns)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def values(self) -> list:
        """返回 list[dict]，每行一个 dict。"""
        result = []
        for i in range(self._nrows):
            row = {}
            for c in self._columns:
                ser = self._inner.get_column(c)
                row[c] = ser.values[i]
            result.append(row)
        return result

    # ---------- dunder ----------

    def __len__(self) -> int:
        return self._nrows

    def __repr__(self) -> str:
        return self._format_repr()

    def __str__(self) -> str:
        return self._format_repr()

    def __getitem__(self, key) -> Union[Series, "DataFrame"]:
        # str -> 单列
        if isinstance(key, str):
            return self._get_column_as_series(key)
        # list[str] -> 多列
        if isinstance(key, list) and all(isinstance(x, str) for x in key):
            new_data = {c: list(self._inner.get_column(c).values) for c in key}
            return DataFrame(new_data)
        # list[bool] / Series -> 行 mask 过滤
        if isinstance(key, Series):
            return self._filter_with_mask(list(key.values))
        if isinstance(key, list) and all(isinstance(x, bool) for x in key):
            return self._filter_with_mask(key)
        # int -> 单行 dict
        if isinstance(key, int):
            if key < 0:
                key += self._nrows
            if key < 0 or key >= self._nrows:
                raise IndexError("index out of range")
            return {c: self._inner.get_column(c).values[key] for c in self._columns}
        # slice -> 行切片
        if isinstance(key, slice):
            start, stop, step = key.indices(self._nrows)
            idx = list(range(start, stop, step))
            new_data = {
                c: [self._inner.get_column(c).values[i] for i in idx]
                for c in self._columns
            }
            return DataFrame(new_data)
        raise TypeError(f"Cannot index DataFrame with {type(key).__name__}")

    def _filter_with_mask(self, mask: list) -> "DataFrame":
        if len(mask) != self._nrows:
            raise ValueError(
                f"mask length {len(mask)} != nrows {self._nrows}"
            )
        cols = self._columns
        new_data = {}
        for c in cols:
            ser = self._inner.get_column(c)
            new_data[c] = list(ser.filter([bool(x) for x in mask]).values)
        return DataFrame(new_data)

    def __setitem__(self, key: str, value) -> None:
        """df['new_col'] = values 添加/更新列。"""
        if isinstance(value, Series):
            values = list(value.values)
        elif isinstance(value, _PySeries):
            values = list(value.values)
        else:
            values = list(value)

        if len(values) != self._nrows:
            raise ValueError(
                f"length of values {len(values)} != length of DataFrame {self._nrows}"
            )

        if key in self._columns:
            # 更新现有列：重建 DataFrame
            new_data = {c: list(self._inner.get_column(c).values) for c in self._columns}
            new_data[key] = values
            self._reload(new_data)
        else:
            # 新增列
            new_data = {c: list(self._inner.get_column(c).values) for c in self._columns}
            new_data[key] = values
            self._reload(new_data)
            self._columns.append(key)

    def __contains__(self, col) -> bool:
        return col in self._columns

    def _reload(self, col_dict: Dict[str, list]) -> None:
        cols = list(col_dict.keys())
        n = self._nrows
        rust_series_list = [_PySeries(col_dict[c], c) for c in cols]
        self._inner = _PyDataFrame(cols, rust_series_list)

    def _get_column_as_series(self, name: str) -> Series:
        ser = self._inner.get_column(name)
        return Series(list(ser.values), name=name, dtype=ser.dtype)

    # ---------- 子集 ----------

    def head(self, n: int = 5) -> "DataFrame":
        cols = self._columns
        new_data = {c: list(self._inner.get_column(c).head(n).values) for c in cols}
        return DataFrame(new_data)

    def tail(self, n: int = 5) -> "DataFrame":
        cols = self._columns
        new_data = {c: list(self._inner.get_column(c).tail(n).values) for c in cols}
        return DataFrame(new_data)

    def filter_rows(self, mask: list) -> "DataFrame":
        if len(mask) != self._nrows:
            raise ValueError(
                f"mask length {len(mask)} != nrows {self._nrows}"
            )
        cols = self._columns
        new_data = {}
        for c in cols:
            ser = self._inner.get_column(c)
            new_data[c] = list(ser.filter([bool(x) for x in mask]).values)
        return DataFrame(new_data)

    # ---------- 缺失值 ----------

    def dropna(self) -> "DataFrame":
        """删除任意一列含 None 的行。"""
        return DataFrame._from_inner(self._inner.dropna())

    def fillna(self, value) -> "DataFrame":
        """填充整个 DataFrame 中所有列的缺失值。

        :param value: 标量 -> 应用到所有列; dict -> 按列名填充不同值
        """
        if isinstance(value, dict):
            return DataFrame._from_inner(self._inner.fillna(value))
        # 标量: 对每列单独调用 fillna
        new_data: Dict[str, list] = {}
        for c in self._columns:
            ser = self._inner.get_column(c)
            filled = ser.fillna(value)
            new_data[c] = list(filled.values)
        return DataFrame(new_data)

    @classmethod
    def _from_inner(cls, inner) -> "DataFrame":
        """从 Rust DataFrame 直接构造 Python DataFrame。"""
        cols = list(inner.columns)
        df = cls.__new__(cls)
        df._inner = inner
        df._columns = cols
        df._nrows = inner.nrows
        df._index = list(range(df._nrows))
        return df

    # ---------- 概览 ----------

    def info(self) -> None:
        """打印 DataFrame 概览。"""
        print("<DataFrame>")
        print(f"Shape: {self._nrows} rows x {len(self._columns)} columns")
        for c in self._columns:
            ser = self._inner.get_column(c)
            print(f"  {c}: dtype={ser.dtype}, non_null={ser.count()}/{self._nrows}")

    def describe(self) -> "DataFrame":
        """对数值列做统计。

        返回 DataFrame:
          - 行: 列名 (与 pandas 一致)
          - 列: count, mean, std, min, 50%, max
        """
        stat_names = ["count", "mean", "std", "min", "50%", "max"]
        # 只对数值列做完整统计
        numeric_cols = [
            c for c in self._columns
            if self._inner.get_column(c).dtype in ("int64", "float64")
        ]
        out: Dict[str, list] = {s: [] for s in stat_names}
        for c in self._columns:
            ser = self._get_column_as_series(c)
            if c in numeric_cols:
                out["count"].append(ser.count())
                out["mean"].append(ser.mean())
                out["std"].append(ser.std())
                out["min"].append(ser.min())
                out["50%"].append(ser.median())
                out["max"].append(ser.max())
            else:
                out["count"].append(ser.count())
                out["mean"].append(None)
                out["std"].append(None)
                out["min"].append(None)
                out["50%"].append(None)
                out["max"].append(None)
        df = DataFrame(out)
        # 用列名作为行索引 -> 在打印时把列名放在最左侧
        return df

    # ---------- 显示 ----------

    def _format_repr(self) -> str:
        # 准备每列的字符串化数据
        col_strs: Dict[str, list] = {}
        col_widths: Dict[str, int] = {}
        for c in self._columns:
            ser = self._inner.get_column(c)
            svec = ser.to_string_vec()
            col_strs[c] = svec
            col_widths[c] = max(len(c), max((len(s) for s in svec), default=0))

        # 截断列: > 20 列时显示前 10 + ... + 后 10
        if len(self._columns) > 20:
            shown_cols = self._columns[:10] + self._columns[-10:]
        else:
            shown_cols = list(self._columns)

        # 截断行: > 60 行时显示前 30 + ... + 后 30
        n = self._nrows
        if n > 60:
            shown_rows = list(range(30)) + list(range(n - 30, n))
        else:
            shown_rows = list(range(n))

        # 索引列宽度
        idx_width = max(len(str(max(n - 1, 0))), 1)

        # 表头
        header_cells = [c.ljust(col_widths[c]) for c in shown_cols]
        header = " " * (idx_width + 1) + "  ".join(header_cells)
        lines = [header]

        prev_i = -1
        for i in shown_rows:
            if prev_i >= 0 and i != prev_i + 1:
                # 截断行之间的省略号
                ellipsis_cells = ["." * col_widths[c] for c in shown_cols]
                lines.append("." * (idx_width + 1) + "  " + "  ".join(ellipsis_cells))
            row_cells = [
                col_strs[c][i].ljust(col_widths[c]) if i < len(col_strs[c]) else ""
                for c in shown_cols
            ]
            lines.append(f"{i:>{idx_width}} " + "  ".join(row_cells))
            prev_i = i

        return "\n".join(lines) + f"\n\n[{n} rows x {len(self._columns)} columns]"
