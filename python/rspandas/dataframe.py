"""DataFrame: pandas-like 2D data structure with Rust backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from rspandas.rspandas import (  # type: ignore
    _DataFrame as _PyDataFrame,
    _Series as _PySeries,
    read_csv_path,
    write_csv_path,
    read_csv_string,
    write_csv_string,
)
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
    def loc(self):
        """基于标签的索引器。"""
        return _LocIndexer(self)

    @property
    def iloc(self):
        """基于位置的索引器。"""
        return _ILocIndexer(self)

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

    def sort_values(self, by, ascending: bool = True) -> "DataFrame":
        """按 by 列排序。by 可以是列名或列名列表。"""
        if isinstance(by, str):
            by = [by]
        for c in by:
            if c not in self._columns:
                raise KeyError(f"column not found: {c}")
        n = self._nrows
        # 取出 by 列用于排序
        sort_keys = [
            [self._inner.get_column(c).values[i] for c in by]
            for i in range(n)
        ]

        def key_func(row):
            return tuple(
                (1 if v is None else 0, v) for v in row
            )

        try:
            order = sorted(range(n), key=lambda i: key_func(sort_keys[i]),
                           reverse=not ascending)
        except TypeError:
            raise TypeError("cannot sort mixed types")
        new_data = {
            c: [self._inner.get_column(c).values[i] for i in order]
            for c in self._columns
        }
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

    def merge(
        self,
        other: "DataFrame",
        on=None,
        how: str = "inner",
    ) -> "DataFrame":
        """连接两个 DataFrame (v0.4.0)。"""
        if on is None:
            raise ValueError("on must be specified")
        if isinstance(on, str):
            keys = [on]
        else:
            keys = list(on)
        for k in keys:
            if k not in self._columns:
                raise KeyError(f"column {k!r} not in left")
            if k not in other._columns:
                raise KeyError(f"column {k!r} not in right")

        left = [
            (tuple(self._inner.get_column(k).values[i] for k in keys),
             {c: self._inner.get_column(c).values[i] for c in self._columns})
            for i in range(self._nrows)
        ]
        right = [
            (tuple(other._inner.get_column(k).values[i] for k in keys),
             {c: other._inner.get_column(c).values[i] for c in other._columns})
            for i in range(other._nrows)
        ]
        left_keys = {lk: i for i, (lk, _) in enumerate(left)}
        right_keys = {rk: i for i, (rk, _) in enumerate(right)}

        merged_rows: List[dict] = []
        if how == "inner":
            common = set(left_keys) & set(right_keys)
            for k in common:
                lv = left[left_keys[k]][1]
                rv = right[right_keys[k]][1]
                row = {}
                for c, v in lv.items():
                    row[c] = v
                for c, v in rv.items():
                    row[c] = v
                merged_rows.append(row)
        elif how == "left":
            right_only = [c for c in other._columns if c not in self._columns]
            for lk, lv in left:
                rv = right[right_keys[lk]][1] if lk in right_keys else None
                row = {}
                for c, v in lv.items():
                    row[c] = v
                if rv is None:
                    for c in right_only:
                        row[c] = None
                else:
                    for c, v in rv.items():
                        row[c] = v
                merged_rows.append(row)
        elif how == "outer":
            right_only = [c for c in other._columns if c not in self._columns]
            left_only = [c for c in self._columns if c not in other._columns]
            seen_l = set()
            for lk, lv in left:
                seen_l.add(lk)
                rv = right[right_keys[lk]][1] if lk in right_keys else None
                row = {}
                for c, v in lv.items():
                    row[c] = v
                if rv is None:
                    for c in right_only:
                        row[c] = None
                else:
                    for c, v in rv.items():
                        row[c] = v
                merged_rows.append(row)
            for rk, rv in right:
                if rk in seen_l:
                    continue
                lv = left[left_keys[rk]][1] if rk in left_keys else None
                row = {}
                if lv is None:
                    for c in left_only:
                        row[c] = None
                else:
                    for c, v in lv.items():
                        row[c] = v
                for c, v in rv.items():
                    row[c] = v
                merged_rows.append(row)
        else:
            raise ValueError(f"unsupported how: {how}")

        all_cols: List[str] = list(self._columns)
        for c in other._columns:
            if c not in all_cols:
                all_cols.append(c)
        col_data: Dict[str, list] = {c: [] for c in all_cols}
        for row in merged_rows:
            for c in all_cols:
                col_data[c].append(row.get(c))
        return DataFrame(col_data)

    @staticmethod
    def concat(frames: List["DataFrame"], axis: int = 0) -> "DataFrame":
        """拼接 DataFrame (v0.4.0)。"""
        if not frames:
            return DataFrame({})
        if axis == 0:
            all_cols: List[str] = []
            for f in frames:
                for c in f._columns:
                    if c not in all_cols:
                        all_cols.append(c)
            col_data: Dict[str, list] = {c: [] for c in all_cols}
            for f in frames:
                for c in all_cols:
                    if c in f._columns:
                        col_data[c].extend(f._inner.get_column(c).values)
                    else:
                        col_data[c].extend([None] * f._nrows)
            return DataFrame(col_data)
        elif axis == 1:
            nrows = frames[0]._nrows
            for f in frames[1:]:
                if f._nrows != nrows:
                    raise ValueError("all frames must have the same number of rows")
            all_cols: List[str] = []
            for f in frames:
                for c in f._columns:
                    if c not in all_cols:
                        all_cols.append(c)
            col_data: Dict[str, list] = {c: [] for c in all_cols}
            for f in frames:
                for c in f._columns:
                    col_data[c].extend(f._inner.get_column(c).values)
            return DataFrame(col_data)
        else:
            raise ValueError(f"axis must be 0 or 1, got {axis}")

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

    def apply(self, func, axis: int = 0) -> "Series":
        """应用函数。

        :param axis: 0=按列 (每列传入 Series); 1=按行 (每行传入 dict)
        """
        if axis == 0:
            results = []
            for c in self._columns:
                ser = self[c]
                results.append(func(ser))
            return Series(results, index=list(self._columns))
        else:  # axis == 1
            results = []
            for i in range(self._nrows):
                row = {c: self._inner.get_column(c).values[i] for c in self._columns}
                results.append(func(row))
            return Series(results, index=list(range(self._nrows)))

    def applymap(self, func) -> "DataFrame":
        """对每个元素应用 func。"""
        new_data: Dict[str, list] = {}
        for c in self._columns:
            ser = self[c]
            new_data[c] = [None if v is None else func(v) for v in ser.values]
        return DataFrame(new_data)

    def replace(self, to_replace, value=None) -> "DataFrame":
        """替换 DataFrame 中的值。"""
        new_data: Dict[str, list] = {}
        for c in self._columns:
            ser = self[c]
            new_data[c] = list(ser.replace(to_replace, value).values)
        return DataFrame(new_data)

    def duplicated(self, subset=None, keep: str = "first") -> "Series":
        """标记重复行。"""
        if subset is None:
            subset = self._columns
        elif isinstance(subset, str):
            subset = [subset]
        # 取每行的 key tuple
        n = self._nrows
        row_keys = []
        for i in range(n):
            key = tuple(self._inner.get_column(c).values[i] for c in subset)
            row_keys.append(key)
        seen = set()
        mark = []
        if keep == "first":
            for k in row_keys:
                mark.append(k in seen)
                seen.add(k)
        elif keep == "last":
            for k in reversed(row_keys):
                mark.append(k in seen)
                seen.add(k)
            mark.reverse()
        elif keep is False:
            from collections import Counter
            c = Counter(row_keys)
            dup = {k for k, n in c.items() if n > 1}
            mark = [k in dup for k in row_keys]
        return Series(mark, name=None, index=list(range(n)))

    def drop_duplicates(self, subset=None, keep: str = "first", inplace: bool = False) -> "DataFrame":
        """删除重复行。"""
        if subset is None:
            subset = self._columns
        elif isinstance(subset, str):
            subset = [subset]
        n = self._nrows
        row_keys = [tuple(self._inner.get_column(c).values[i] for c in subset) for i in range(n)]
        seen = set()
        keep_idx = []
        if keep == "first":
            for i, k in enumerate(row_keys):
                if k not in seen:
                    keep_idx.append(i)
                    seen.add(k)
        elif keep == "last":
            for i, k in reversed(list(enumerate(row_keys))):
                if k not in seen:
                    keep_idx.append(i)
                    seen.add(k)
            keep_idx.reverse()
        new_data: Dict[str, list] = {}
        for c in self._columns:
            vals = self._inner.get_column(c).values
            new_data[c] = [vals[i] for i in keep_idx]
        return DataFrame(new_data)

    def nunique(self) -> "Series":
        """每列不同值数量。"""
        out = {}
        for c in self._columns:
            ser = self[c]
            out[c] = ser.nunique()
        return Series(out, name=None, index=list(self._columns))

    def to_pandas(self):
        """转换为 pandas DataFrame。"""
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            raise ImportError("pandas is required for to_pandas()")
        data = {c: list(self._inner.get_column(c).values) for c in self._columns}
        return pd.DataFrame(data)

    @classmethod
    def from_pandas(cls, pdf) -> "DataFrame":
        """从 pandas DataFrame 构造。"""
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            raise ImportError("pandas is required for from_pandas()")
        if not isinstance(pdf, pd.DataFrame):
            raise TypeError("expected pandas DataFrame")
        data = {c: [None if pd.isna(v) else v.item() if hasattr(v, 'item') else v
                    for v in pdf[c].values]
                for c in pdf.columns}
        return cls(data)

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

    # ---------- CSV I/O ----------

    @classmethod
    def read_csv(
        cls,
        path: str,
        has_header: bool = True,
    ) -> "DataFrame":
        """从 CSV 文件读取 DataFrame。"""
        cols, series_list = read_csv_path(path, has_header)
        return cls._from_inner(_PyDataFrame(cols, series_list))

    @classmethod
    def read_csv_from_string(
        cls,
        content: str,
        has_header: bool = True,
    ) -> "DataFrame":
        """从 CSV 字符串构造 DataFrame。"""
        cols, series_list = read_csv_string(content, has_header)
        return cls._from_inner(_PyDataFrame(cols, series_list))

    def to_csv(
        self,
        path: Optional[str] = None,
        include_header: bool = True,
    ) -> Optional[str]:
        """写入 CSV。

        :param path: 文件路径；为 None 时返回字符串
        :param include_header: 是否写入表头
        :return: 如果 path 为 None，返回 CSV 字符串
        """
        if path is None:
            return write_csv_string(
                list(self._columns),
                [self._inner.get_column(c) for c in self._columns],
                include_header,
            )
        write_csv_path(
            path,
            list(self._columns),
            [self._inner.get_column(c) for c in self._columns],
            include_header,
        )
        return None

    # ---------- 索引器辅助 ----------

    def _select_row(self, idx: int) -> "DataFrame":
        if idx < 0:
            idx += self._nrows
        if idx < 0 or idx >= self._nrows:
            raise IndexError("single positional indexer is out-of-bounds")
        new_data = {
            c: [self._inner.get_column(c).values[idx]] for c in self._columns
        }
        return DataFrame(new_data)

    def _select_slice(self, start, stop, step) -> "DataFrame":
        if start is None:
            start = 0
        if stop is None:
            stop = self._nrows
        if step is None:
            step = 1
        if start < 0:
            start += self._nrows
        if stop < 0:
            stop += self._nrows
        if start is None or stop is None or start >= self._nrows:
            return DataFrame({})
        stop = min(stop, self._nrows)
        idx = list(range(start, stop, step))
        new_data = {
            c: [self._inner.get_column(c).values[i] for i in idx]
            for c in self._columns
        }
        return DataFrame(new_data)

    def _select_indices(self, indices: list) -> "DataFrame":
        n = self._nrows
        norm = []
        for i in indices:
            if i < 0:
                i += n
            if i < 0 or i >= n:
                raise IndexError(f"index {i} out of range")
            norm.append(i)
        new_data = {
            c: [self._inner.get_column(c).values[i] for i in norm]
            for c in self._columns
        }
        return DataFrame(new_data)

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
          - 第一列(无标题): 列名 (与 pandas 一致)
          - 其他列: count, mean, std, min, 50%, max
        """
        stat_names = ["count", "mean", "std", "min", "50%", "max"]
        # 只对数值列做完整统计
        numeric_cols = [
            c for c in self._columns
            if self._inner.get_column(c).dtype in ("int64", "float64")
        ]
        out: Dict[str, list] = {s: [] for s in stat_names}
        # 第一列(无名)存放列名 -> 用空字符串作为"列名"
        out[""] = []
        for c in self._columns:
            out[""].append(c)
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
        return DataFrame(out)

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

    def groupby(self, by, as_index: bool = True):
        """按 by 列分组 (v0.4.0)。"""
        return DataFrameGroupBy(self, by, as_index=as_index)


class DataFrameGroupBy:
    """DataFrame 分组结果 (极简版)。"""

    def __init__(self, df: "DataFrame", by, as_index: bool = True):
        if isinstance(by, str):
            self._by = [by]
        else:
            self._by = list(by)
        self._df = df
        self._as_index = as_index
        # 分组: { key_tuple: [row_indices] }
        self._groups: Dict[tuple, list] = {}
        n = df._nrows
        for i in range(n):
            key = tuple(
                df._inner.get_column(c).values[i] for c in self._by
            )
            self._groups.setdefault(key, []).append(i)

    def _agg(self, agg_funcs: Dict[str, str]) -> "DataFrame":
        """对每列应用聚合函数。

        :param agg_funcs: {列名: 'sum' | 'mean' | 'min' | 'max' | 'count' | 'std' | 'var' | 'median' | 'first' | 'last'}
        """
        result: Dict[str, list] = {c: [] for c in self._by}
        agg_cols = list(agg_funcs.keys())
        for c in agg_cols:
            result[c] = []

        for key, idxs in self._groups.items():
            for k, c in zip(key, self._by):
                result[c].append(k)
            for c in agg_cols:
                # 用 iloc 取子集 (Series.iloc 接受 list[int])
                ser = self._df[c]
                sub = ser.iloc(idxs)
                func = agg_funcs[c]
                if func == "sum":
                    result[c].append(sub.sum())
                elif func == "mean":
                    result[c].append(sub.mean())
                elif func == "min":
                    result[c].append(sub.min())
                elif func == "max":
                    result[c].append(sub.max())
                elif func == "count":
                    result[c].append(sub.count())
                elif func == "std":
                    result[c].append(sub.std())
                elif func == "var":
                    result[c].append(sub.var())
                elif func == "median":
                    result[c].append(sub.median())
                elif func == "first":
                    result[c].append(sub.iloc(0) if len(sub) > 0 else None)
                elif func == "last":
                    result[c].append(sub.iloc(-1) if len(sub) > 0 else None)
                else:
                    raise ValueError(f"unsupported agg: {func}")
        return DataFrame(result)

    def sum(self) -> "DataFrame":
        return self._agg({c: "sum" for c in self._df._columns if c not in self._by})

    def mean(self) -> "DataFrame":
        numeric_cols = [
            c for c in self._df._columns
            if c not in self._by
            and self._df._inner.get_column(c).dtype in ("int64", "float64")
        ]
        return self._agg({c: "mean" for c in numeric_cols})

    def min(self) -> "DataFrame":
        return self._agg({c: "min" for c in self._df._columns if c not in self._by})

    def max(self) -> "DataFrame":
        return self._agg({c: "max" for c in self._df._columns if c not in self._by})

    def count(self) -> "DataFrame":
        return self._agg({c: "count" for c in self._df._columns if c not in self._by})

    def agg(self, func) -> "DataFrame":
        """通用聚合: 可以传 str 或 dict[列名->str]。"""
        if isinstance(func, str):
            return self._agg({c: func for c in self._df._columns if c not in self._by})
        if isinstance(func, dict):
            return self._agg(func)
        raise TypeError("agg must be str or dict")


# ---------------------------------------------------------------------------
# 索引器
# ---------------------------------------------------------------------------

class _IndexerBase:
    """loc/iloc 索引器基类。"""

    def __init__(self, df: "DataFrame"):
        self._df = df


class _LocIndexer(_IndexerBase):
    """基于标签的索引器 (MVP 索引为 0..n-1)。"""

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key = key
            col_key = None

        # 1. 行选择
        rows_df = self._select_rows(row_key)

        # 2. 列选择
        if col_key is not None:
            if isinstance(col_key, str):
                return rows_df[col_key]
            if isinstance(col_key, list):
                return rows_df[col_key]
            raise TypeError(f"loc: unsupported column key {type(col_key).__name__}")
        return rows_df

    def _select_rows(self, key):
        if isinstance(key, int):
            return self._df._select_row(int(key))
        if isinstance(key, slice):
            # loc 切片: 双闭区间 (与 pandas 一致)
            start, stop, step = key.start, key.stop, key.step
            if step is None:
                step = 1
            if step <= 0:
                raise ValueError("loc slice step must be positive")
            n = self._df._nrows
            if start is None:
                start = 0
            if stop is None:
                stop = n - 1
            if start < 0:
                start += n
            if stop < 0:
                stop += n
            if start >= n:
                return DataFrame({})
            stop = min(stop, n - 1)
            idx = list(range(start, stop + 1, step))
            new_data = {
                c: [self._df._inner.get_column(c).values[i] for i in idx]
                for c in self._df._columns
            }
            return DataFrame(new_data)
        if isinstance(key, list):
            if not key:
                return DataFrame({})
            if all(isinstance(x, bool) for x in key):
                return self._df[key]
            # list of labels
            idx = list(key)
            return self._df._select_indices(idx)
        if isinstance(key, Series):
            return self._df[key]
        raise TypeError(f"loc: unsupported key {type(key).__name__}")


class _ILocIndexer(_IndexerBase):
    """基于位置的索引器 (整数位置)。"""

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key = key
            col_key = None

        # 1. 行选择
        if isinstance(row_key, int):
            if row_key < 0:
                row_key += self._df._nrows
            rows_df = self._df._select_row(int(row_key))
        elif isinstance(row_key, slice):
            start, stop, step = row_key.indices(self._df._nrows)
            rows_df = self._df._select_slice(start, stop, step)
        elif isinstance(row_key, list):
            if all(isinstance(x, bool) for x in row_key):
                rows_df = self._df[row_key]
            else:
                idx = [int(i) if i >= 0 else int(i) + self._df._nrows for i in row_key]
                rows_df = self._df._select_indices(idx)
        else:
            raise TypeError(f"iloc: unsupported row key {type(row_key).__name__}")

        # 2. 列选择
        if col_key is not None:
            cols = rows_df.columns
            if isinstance(col_key, int):
                col_key = int(col_key) + len(cols) if col_key < 0 else int(col_key)
                return rows_df[cols[col_key]]
            if isinstance(col_key, list):
                if all(isinstance(x, bool) for x in col_key):
                    picked = [c for c, b in zip(cols, col_key) if b]
                else:
                    picked = [cols[int(i)] for i in col_key]
                return rows_df[picked]
            if isinstance(col_key, slice):
                picked = cols[col_key]
                return rows_df[list(picked)]
            raise TypeError(f"iloc: unsupported column key {type(col_key).__name__}")
        return rows_df
