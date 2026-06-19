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

    # ---------- 高级操作 (v1.0.0) ----------

    def assign(self, **kwargs) -> "DataFrame":
        """添加新列 (链式调用友好)。"""
        new_data = {c: list(self._inner.get_column(c).values) for c in self._columns}
        for name, value in kwargs.items():
            if isinstance(value, Series):
                new_data[name] = list(value.values)
            else:
                try:
                    iter(value)
                    new_data[name] = list(value)
                except TypeError:
                    new_data[name] = [value] * self._nrows
        return DataFrame(new_data)

    def eval(self, expr: str):
        """用字符串表达式计算。"""
        local_vars = {c: self._inner.get_column(c).values for c in self._columns}
        return eval(expr, {}, local_vars)

    def query(self, expr: str) -> "DataFrame":
        """用字符串表达式过滤行。"""
        mask = []
        for i in range(self._nrows):
            local_vars = {c: self._inner.get_column(c).values[i] for c in self._columns}
            mask.append(bool(eval(expr, {}, local_vars)))
        return self._filter_with_mask(mask)

    def pipe(self, func, *args, **kwargs):
        """管道方法: df.pipe(func, ...) == func(df, ...)。"""
        return func(self, *args, **kwargs)

    def transform(self, func) -> "DataFrame":
        """对每列应用 func 并返回相同形状的 DataFrame。"""
        new_data: Dict[str, list] = {}
        for c in self._columns:
            ser = self[c]
            result = func(ser)
            if isinstance(result, Series):
                new_data[c] = list(result.values)
            else:
                new_data[c] = [result] * self._nrows
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

    def to_numpy(self):
        """转换为 numpy 二维数组。"""
        try:
            import numpy as np  # type: ignore
        except ImportError:
            raise ImportError("numpy is required for to_numpy()")
        cols = list(self._columns)
        data = [[self._inner.get_column(c).values[i] for c in cols] for i in range(self._nrows)]
        return np.array(data)

    @classmethod
    def from_numpy(cls, arr, columns=None, index=None, dtype=None) -> "DataFrame":
        """从 numpy 二维数组构造 DataFrame。

        Parameters
        ----------
        arr : numpy.ndarray
            二维输入数组。
        columns : list[str], optional
            列名。
        index : list, optional
            行索引。
        dtype : str, optional
            目标类型。

        Returns
        -------
        DataFrame
        """
        try:
            import numpy as np  # type: ignore
        except ImportError:
            raise ImportError("numpy is required for from_numpy()")
        if not isinstance(arr, np.ndarray):
            raise TypeError("expected numpy.ndarray")
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        data = arr.tolist()
        if columns is None:
            columns = [f"col{i}" for i in range(arr.shape[1])]
        return cls(data, columns=columns, index=index, dtype=dtype)

    def to_arrow(self):
        """转换为 PyArrow Table。

        Returns
        -------
        pyarrow.Table
        """
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError("pyarrow is required for to_arrow()")
        arrays = []
        for c in self._columns:
            col_data = list(self._inner.get_column(c).values)
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
        return pa.table(dict(zip(self._columns, arrays)))

    @classmethod
    def from_arrow(cls, table) -> "DataFrame":
        """从 PyArrow Table 构造 DataFrame。

        Parameters
        ----------
        table : pyarrow.Table
            PyArrow 表。

        Returns
        -------
        DataFrame
        """
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError("pyarrow is required for from_arrow()")
        if not isinstance(table, pa.Table):
            raise TypeError("expected pyarrow.Table")
        data = {}
        for col_name in table.column_names:
            col = table.column(col_name)
            data[col_name] = col.to_pylist()
        return cls(data)

    @staticmethod
    def read_json(
        path: str,
        orient: str = "records",
        lines: bool = False,
        encoding: str = "utf-8",
    ) -> "DataFrame":
        """从 JSON 文件读取 DataFrame (v1.2.0)。

        :param path: JSON 文件路径
        :param orient: JSON 格式方向
        :param lines: 是否按行读取 JSON
        :param encoding: 文件编码
        """
        from .io import read_json as _read_json
        return _read_json(path, orient=orient, lines=lines, encoding=encoding)

    @staticmethod
    def read_excel(
        path: str,
        sheet_name=0,
        header: int = 0,
        **kwargs,
    ) -> "DataFrame":
        """从 Excel 文件读取 DataFrame (v1.2.0)。

        :param path: Excel 文件路径
        :param sheet_name: 工作表名称或索引
        :param header: 用作列名的行号
        """
        from .io import read_excel as _read_excel
        return _read_excel(path, sheet_name=sheet_name, header=header, **kwargs)

    @staticmethod
    def read_parquet(path: str, **kwargs) -> "DataFrame":
        """从 Parquet 文件读取 DataFrame (v1.2.0)。

        :param path: Parquet 文件路径
        """
        from .io import read_parquet as _read_parquet
        return _read_parquet(path, **kwargs)

    @staticmethod
    def read_feather(path: str, **kwargs) -> "DataFrame":
        """从 Feather 文件读取 DataFrame (v1.5.0)。

        :param path: Feather 文件路径
        """
        from .io import read_feather as _read_feather
        return _read_feather(path, **kwargs)

    @staticmethod
    def read_pickle(path: str, **kwargs) -> "DataFrame":
        """从 Pickle 文件读取 DataFrame (v1.2.0)。

        :param path: Pickle 文件路径
        """
        from .io import read_pickle as _read_pickle
        return _read_pickle(path, **kwargs)

    @staticmethod
    def read_sql(query: str, conn, **kwargs) -> "DataFrame":
        """从 SQL 数据库读取 DataFrame (v1.2.0)。

        :param query: SQL 查询语句
        :param conn: 数据库连接
        """
        from .io import read_sql as _read_sql
        return _read_sql(query, conn, **kwargs)

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

    # ---------- 索引操作 (v1.0.0) ----------

    def drop(self, labels, axis: int = 0) -> "DataFrame":
        """删除行或列。
        :param labels: 要删除的标签 (str/int 或 list)
        :param axis: 0=行, 1=列
        """
        if axis == 0:
            if not isinstance(labels, (list, tuple)):
                labels = [labels]
            keep_idx = []
            for i in range(self._nrows):
                label = self._index[i] if self._index else i
                if label not in labels:
                    keep_idx.append(i)
            new_data = {c: [self._inner.get_column(c).values[i] for i in keep_idx]
                        for c in self._columns}
            return DataFrame(new_data)
        else:
            if not isinstance(labels, (list, tuple)):
                labels = [labels]
            new_cols = [c for c in self._columns if c not in labels]
            new_data = {c: list(self._inner.get_column(c).values) for c in new_cols}
            return DataFrame(new_data)

    def rename(self, mapper, axis: int = 0) -> "DataFrame":
        """重命名行或列。
        :param mapper: dict {old_name: new_name} 或 callable
        :param axis: 0=行索引, 1=列
        """
        if axis == 1:
            if isinstance(mapper, dict):
                new_cols = [mapper.get(c, c) for c in self._columns]
            else:
                new_cols = [mapper(c) for c in self._columns]
            new_data = {new_cols[i]: list(self._inner.get_column(c).values)
                        for i, c in enumerate(self._columns)}
            return DataFrame(new_data)
        else:
            new_index = []
            for i in range(self._nrows):
                label = self._index[i] if self._index else i
                if isinstance(mapper, dict):
                    new_index.append(mapper.get(label, label))
                else:
                    new_index.append(mapper(label))
            new_data = {c: list(self._inner.get_column(c).values) for c in self._columns}
            df = DataFrame(new_data)
            df._index = new_index
            return df

    def reindex(self, index=None, columns=None) -> "DataFrame":
        """重新索引。"""
        if index is None:
            index = self._index
        if columns is None:
            columns = self._columns
        if not isinstance(index, list):
            index = list(index)
        if not isinstance(columns, list):
            columns = list(columns)
        old_index_map = {self._index[i] if self._index else i: i
                         for i in range(self._nrows)}
        new_data = {c: [] for c in columns}
        for new_idx_label in index:
            old_pos = old_index_map.get(new_idx_label)
            for c in columns:
                if c in self._columns and old_pos is not None:
                    new_data[c].append(self._inner.get_column(c).values[old_pos])
                else:
                    new_data[c].append(None)
        df = DataFrame(new_data)
        df._index = index
        return df

    def set_index(self, keys) -> "DataFrame":
        """设置索引列。"""
        if isinstance(keys, str):
            keys = [keys]
        else:
            keys = list(keys)
        for k in keys:
            if k not in self._columns:
                raise KeyError(f"column not found: {k}")
        new_index = []
        for i in range(self._nrows):
            if len(keys) == 1:
                new_index.append(self._inner.get_column(keys[0]).values[i])
            else:
                new_index.append(tuple(self._inner.get_column(k).values[i] for k in keys))
        new_data = {c: list(self._inner.get_column(c).values)
                    for c in self._columns if c not in keys}
        df = DataFrame(new_data)
        df._index = new_index
        return df

    def reset_index(self, drop: bool = False) -> "DataFrame":
        """重置索引为默认 RangeIndex。"""
        new_data = {c: list(self._inner.get_column(c).values) for c in self._columns}
        if not drop and self._index is not None:
            new_data["index"] = list(self._index)
            cols = ["index"] + [c for c in self._columns]
            df = DataFrame(new_data)
            df._columns = cols
        else:
            df = DataFrame(new_data)
        df._index = list(range(self._nrows))
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

    # ---------- IO 扩展 (v1.2.0) ----------

    def to_json(
        self,
        path: Optional[str] = None,
        orient: str = "records",
        lines: bool = False,
        force_ascii: bool = False,
        indent: Optional[int] = None,
    ) -> Optional[str]:
        """将 DataFrame 写入 JSON 文件或返回 JSON 字符串。

        :param path: 文件路径；为 None 时返回字符串
        :param orient: JSON 格式方向 ('records' / 'columns' / 'index' / 'split' / 'values')
        :param lines: 是否按行输出 JSON
        :param force_ascii: 是否强制 ASCII 编码
        :param indent: 缩进空格数
        :return: 如果 path 为 None，返回 JSON 字符串
        """
        from .io import to_json as _to_json
        return _to_json(self, path, orient=orient, lines=lines,
                        force_ascii=force_ascii, indent=indent)

    def to_excel(
        self,
        path: str,
        sheet_name: str = "Sheet1",
        index: bool = False,
        header: bool = True,
        **kwargs,
    ) -> None:
        """将 DataFrame 写入 Excel 文件。

        :param path: 输出文件路径
        :param sheet_name: 工作表名称
        :param index: 是否写入行索引
        :param header: 是否写入列名
        """
        from .io import to_excel as _to_excel
        _to_excel(self, path, sheet_name=sheet_name, index=index, header=header, **kwargs)

    def to_parquet(
        self,
        path: str,
        compression: Optional[str] = "snappy",
        **kwargs,
    ) -> None:
        """将 DataFrame 写入 Parquet 文件。

        :param path: 输出文件路径
        :param compression: 压缩算法 (snappy, gzip, brotli, zstd, none)
        """
        from .io import to_parquet as _to_parquet
        _to_parquet(self, path, compression=compression, **kwargs)

    def to_feather(
        self,
        path: str,
        compression: Optional[str] = "lz4",
        **kwargs,
    ) -> None:
        """将 DataFrame 写入 Feather 文件 (v1.5.0)。

        :param path: 输出文件路径
        :param compression: 压缩算法 (lz4, zstd, uncompressed)
        """
        from .io import to_feather as _to_feather
        _to_feather(self, path, compression=compression, **kwargs)

    def to_pickle(self, path: str, **kwargs) -> None:
        """将 DataFrame 写入 Pickle 文件。

        :param path: 输出文件路径
        """
        from .io import to_pickle as _to_pickle
        _to_pickle(self, path, **kwargs)

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

    def melt(
        self,
        id_vars=None,
        value_vars=None,
        var_name: str = "variable",
        value_name: str = "value",
        ignore_index: bool = True,
    ) -> "DataFrame":
        """将宽表转为长表 (v1.0.0)。

        将 id_vars 之外 (或 value_vars 指定) 的列"折叠"成 variable + value 两列。

        :param id_vars: 用作标识符的列 (str | list[str] | None)
        :param value_vars: 要展开为值的列 (str | list[str] | None)，None 表示其余列
        :param var_name: 存放原列名的列名
        :param value_name: 存放原值的列名
        :param ignore_index: 是否重置索引
        :return: DataFrame

        Examples:
            >>> df = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
            >>> df.melt(id_vars=['A'])
               A variable  value
            0  1        B      3
            1  1        C      5
            2  2        B      4
            3  2        C      6
        """
        # 解析 id_vars
        if id_vars is None:
            id_vars = []
        elif isinstance(id_vars, str):
            id_vars = [id_vars]
        else:
            id_vars = list(id_vars)
        for c in id_vars:
            if c not in self._columns:
                raise KeyError(f"id_var column not found: {c}")

        # 解析 value_vars
        if value_vars is None:
            value_vars = [c for c in self._columns if c not in id_vars]
        elif isinstance(value_vars, str):
            value_vars = [value_vars]
        else:
            value_vars = list(value_vars)
        for c in value_vars:
            if c not in self._columns:
                raise KeyError(f"value_var column not found: {c}")

        if not value_vars:
            raise ValueError("value_vars cannot be empty")

        # 构造结果
        new_data: Dict[str, list] = {c: [] for c in id_vars}
        new_data[var_name] = []
        new_data[value_name] = []

        for i in range(self._nrows):
            for vc in value_vars:
                for iv in id_vars:
                    new_data[iv].append(self._inner.get_column(iv).values[i])
                new_data[var_name].append(vc)
                new_data[value_name].append(self._inner.get_column(vc).values[i])

        return DataFrame(new_data)

    def pivot(
        self,
        index=None,
        columns=None,
        values=None,
    ) -> "DataFrame":
        """将长表转为宽表 (v1.0.0)。

        与 pivot_table 不同，pivot 不支持聚合，仅在每个 (index, columns) 组合
        对应一个唯一 value 时使用。

        :param index: 用作行索引的列 (str | None -> 使用现有 index)
        :param columns: 用作列的列 (str)
        :param values: 填充值的列 (str | list[str] | None)
        :return: DataFrame

        Examples:
            >>> df = DataFrame({
            ...     'foo': ['one', 'one', 'two', 'two'],
            ...     'bar': ['A', 'B', 'A', 'B'],
            ...     'baz': [1, 2, 3, 4],
            ... })
            >>> df.pivot(index='foo', columns='bar', values='baz')
            bar    A    B
            foo
            one    1    2
            two    3    4
        """
        if columns is None:
            raise ValueError("columns must be specified")
        if values is None:
            raise ValueError("values must be specified")
        if isinstance(values, str):
            values = [values]
        else:
            values = list(values)
        if isinstance(index, str):
            index = [index]
        elif index is None:
            index = []

        for c in [columns] + values + index:
            if c not in self._columns:
                raise KeyError(f"column not found: {c}")

        n = self._nrows
        # 取出关键列
        col_vals = list(self._inner.get_column(columns).values)
        idx_tuples = [
            tuple(self._inner.get_column(c).values[i] for c in index)
            for i in range(n)
        ]
        # 收集所有 column 值
        new_cols_set: list = []
        seen = set()
        for v in col_vals:
            if v not in seen:
                new_cols_set.append(v)
                seen.add(v)

        # 收集所有 index 值 (保持顺序)
        new_idx_set: list = []
        idx_seen = set()
        for t in idx_tuples:
            if t not in idx_seen:
                new_idx_set.append(t)
                idx_seen.add(t)

        # 构造 (idx_tuple, col_val) -> value dict
        cell: Dict[tuple, list] = {}
        for i in range(n):
            key = (idx_tuples[i], col_vals[i])
            cell.setdefault(key, []).extend(
                self._inner.get_column(v).values[i] for v in values
            )

        # 构造结果 DataFrame
        new_data: Dict[str, list] = {}
        for ic in index:
            new_data[ic] = [t[index.index(ic)] for t in new_idx_set]
        for cv in new_cols_set:
            for j, v in enumerate(values):
                col_name = str(cv) if len(values) == 1 else f"{cv}_{v}"
                col_data = []
                for t in new_idx_set:
                    vals = cell.get((t, cv))
                    if vals is None:
                        col_data.append(None)
                    else:
                        col_data.append(vals[j] if j < len(vals) else None)
                new_data[col_name] = col_data

        return DataFrame(new_data)

    def pivot_table(
        self,
        values=None,
        index=None,
        columns=None,
        aggfunc: str = "mean",
        fill_value=None,
    ) -> "DataFrame":
        """创建透视表 (v1.0.0)。

        :param values: 聚合的列 (str | list[str] | None -> 所有数值列)
        :param index: 行分组列 (str | list[str])
        :param columns: 列分组列 (str | list[str])
        :param aggfunc: 聚合函数 ('sum' / 'mean' / 'count' / 'min' / 'max' / 'median' / 'std')
        :param fill_value: 用于替换缺失值的标量
        :return: DataFrame

        Examples:
            >>> df = DataFrame({
            ...     'A': ['foo', 'foo', 'bar', 'bar'],
            ...     'B': ['one', 'two', 'one', 'two'],
            ...     'C': [1, 2, 3, 4],
            ...     'D': [10, 20, 30, 40],
            ... })
            >>> df.pivot_table(values='C', index='A', columns='B')
            B      one  two
            A
            bar      3    4
            foo      1    2
        """
        # 解析 values
        if values is None:
            # 默认选所有数值列
            values = [
                c for c in self._columns
                if self._inner.get_column(c).dtype in ("int64", "float64")
            ]
        if isinstance(values, str):
            values = [values]
        else:
            values = list(values)
        for v in values:
            if v not in self._columns:
                raise KeyError(f"value column not found: {v}")

        # 解析 index
        if index is None:
            index_cols: list = []
        elif isinstance(index, str):
            index_cols = [index]
        else:
            index_cols = list(index)
        for c in index_cols:
            if c not in self._columns:
                raise KeyError(f"index column not found: {c}")

        # 解析 columns
        if columns is None:
            raise ValueError("columns must be specified")
        if isinstance(columns, str):
            col_cols = [columns]
        else:
            col_cols = list(columns)
        for c in col_cols:
            if c not in self._columns:
                raise KeyError(f"column key not found: {c}")

        n = self._nrows
        idx_tuples = [
            tuple(self._inner.get_column(c).values[i] for c in index_cols)
            for i in range(n)
        ]
        col_tuples = [
            tuple(self._inner.get_column(c).values[i] for c in col_cols)
            for i in range(n)
        ]

        # 收集所有 index 值
        idx_set: list = []
        idx_seen = set()
        for t in idx_tuples:
            if t not in idx_seen:
                idx_set.append(t)
                idx_seen.add(t)

        # 收集所有 column 值
        col_set: list = []
        col_seen = set()
        for t in col_tuples:
            if t not in col_seen:
                col_set.append(t)
                col_seen.add(t)

        # 构造 (idx, col) -> list of values
        groups: Dict[tuple, Dict[str, list]] = {}
        for i in range(n):
            key = (idx_tuples[i], col_tuples[i])
            if key not in groups:
                groups[key] = {v: [] for v in values}
            for v in values:
                groups[key][v].append(self._inner.get_column(v).values[i])

        # 构造结果
        new_data: Dict[str, list] = {}
        for ic in index_cols:
            new_data[ic] = [t[index_cols.index(ic)] for t in idx_set]

        for ct in col_set:
            for v in values:
                col_name_parts = [str(x) for x in ct] + [v]
                col_name = "_".join(col_name_parts) if len(col_name_parts) > 1 else col_name_parts[0]
                col_data = []
                for it in idx_set:
                    g = groups.get((it, ct))
                    if g is None or not g[v]:
                        col_data.append(fill_value)
                    else:
                        vals = g[v]
                        if aggfunc == "sum":
                            col_data.append(sum(x for x in vals if x is not None))
                        elif aggfunc == "mean":
                            nums = [x for x in vals if x is not None]
                            col_data.append(sum(nums) / len(nums) if nums else fill_value)
                        elif aggfunc == "count":
                            col_data.append(sum(1 for x in vals if x is not None))
                        elif aggfunc == "min":
                            nums = [x for x in vals if x is not None]
                            col_data.append(min(nums) if nums else fill_value)
                        elif aggfunc == "max":
                            nums = [x for x in vals if x is not None]
                            col_data.append(max(nums) if nums else fill_value)
                        elif aggfunc == "median":
                            nums = sorted([x for x in vals if x is not None])
                            if not nums:
                                col_data.append(fill_value)
                            elif len(nums) % 2:
                                col_data.append(nums[len(nums) // 2])
                            else:
                                col_data.append((nums[len(nums) // 2 - 1] + nums[len(nums) // 2]) / 2)
                        elif aggfunc == "std":
                            nums = [x for x in vals if x is not None]
                            if len(nums) < 2:
                                col_data.append(fill_value)
                            else:
                                m = sum(nums) / len(nums)
                                var = sum((x - m) ** 2 for x in nums) / len(nums)
                                col_data.append(var ** 0.5)
                        else:
                            raise ValueError(f"unsupported aggfunc: {aggfunc}")
                new_data[col_name] = col_data

        return DataFrame(new_data)

    def stack(self, level: int = -1) -> "DataFrame":
        """将列堆叠为行 (v1.0.0)。"""
        # 简化版: 仅支持单层
        n = self._nrows
        new_data: Dict[str, list] = {self._index_name or "index": []}
        # 取所有列名作为 stack 后的变量
        new_data["variable"] = []
        new_data["value"] = []
        for i in range(n):
            for c in self._columns:
                new_data[self._index_name or "index"].append(i)
                new_data["variable"].append(c)
                new_data["value"].append(self._inner.get_column(c).values[i])
        return DataFrame(new_data)

    @property
    def _index_name(self) -> Optional[str]:
        """返回 index 列名 (None 表示 RangeIndex)。"""
        return None

    def unstack(self) -> "DataFrame":
        """stack 的反操作 (v1.0.0) - 简化版。"""
        # 如果 DataFrame 包含 'variable' 和 'value' 列, 尝试 pivot
        if "variable" in self._columns and "value" in self._columns:
            other_cols = [c for c in self._columns if c not in ("variable", "value")]
            if other_cols:
                return self.pivot(
                    index=other_cols[0],
                    columns="variable",
                    values="value",
                )
        raise NotImplementedError("unstack requires 'variable' and 'value' columns")

    # ---------- v2.0.0: compare / equals / copy ----------

    def compare(
        self,
        other: "DataFrame",
        align_axis: int = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
        result_names: tuple = ("self", "other"),
    ) -> "DataFrame":
        """与另一个 DataFrame 逐元素比较，返回差异。

        :param other: 要比较的 DataFrame
        :param align_axis: 1=列对齐, 0=行对齐
        :param keep_shape: 是否保持原始形状（用 None 填充相同位置）
        :param keep_equal: 是否保留相同值
        :param result_names: 差异列的多级列名
        :return: 差异 DataFrame
        """
        if self.shape != other.shape:
            raise ValueError(
                f"Can only compare identically-labeled DataFrame objects, "
                f"shapes: {self.shape} vs {other.shape}"
            )

        left_name, right_name = result_names
        n = self._nrows
        cols = self._columns
        other_cols = other._columns

        if cols != other_cols:
            raise ValueError(
                "Can only compare identically-labeled DataFrame objects"
            )

        diff_data: Dict[str, list] = {}
        for c in cols:
            self_vals = list(self._inner.get_column(c).values)
            other_vals = list(other._inner.get_column(c).values)
            for i in range(n):
                sv = self_vals[i]
                ov = other_vals[i]
                if keep_equal or sv != ov:
                    diff_data.setdefault((c, left_name), []).append(sv)
                    diff_data.setdefault((c, right_name), []).append(ov)
                elif keep_shape:
                    diff_data.setdefault((c, left_name), []).append(sv)
                    diff_data.setdefault((c, right_name), []).append(ov)

        if not diff_data:
            return DataFrame({})

        df = DataFrame({str(k): v for k, v in diff_data.items()})
        return df

    def equals(self, other: "DataFrame") -> bool:
        """检查两个 DataFrame 是否完全相等。

        :param other: 另一个 DataFrame
        :return: bool
        """
        if not isinstance(other, DataFrame):
            return False
        if self.shape != other.shape:
            return False
        if self._columns != other._columns:
            return False
        for c in self._columns:
            self_vals = list(self._inner.get_column(c).values)
            other_vals = list(other._inner.get_column(c).values)
            if self_vals != other_vals:
                return False
        return True

    def copy(self, deep: bool = True) -> "DataFrame":
        """创建 DataFrame 的副本。

        :param deep: 是否深拷贝 (默认 True)
        :return: DataFrame
        """
        new_data = {c: list(self._inner.get_column(c).values) for c in self._columns}
        return DataFrame(new_data)

    # ---------- v2.0.0: pop / insert ----------

    def pop(self, item: str) -> "Series":
        """删除一列并返回它。

        :param item: 列名
        :return: Series
        """
        if item not in self._columns:
            raise KeyError(f"column not found: {item}")
        ser = self._get_column_as_series(item)
        new_cols = [c for c in self._columns if c != item]
        new_data = {c: list(self._inner.get_column(c).values) for c in new_cols}
        self._reload(new_data)
        self._columns = new_cols
        return ser

    def insert(self, loc: int, column: str, value) -> None:
        """在指定位置插入一列。

        :param loc: 插入位置 (0-based)
        :param column: 列名
        :param value: 列数据 (list / Series / 标量)
        """
        if column in self._columns:
            raise ValueError(f"cannot insert {column}, already exists")

        if isinstance(value, Series):
            vals = list(value.values)
        elif isinstance(value, _PySeries):
            vals = list(value.values)
        else:
            try:
                vals = list(value)
            except TypeError:
                vals = [value] * self._nrows

        if len(vals) != self._nrows:
            raise ValueError(
                f"length of values {len(vals)} != length of DataFrame {self._nrows}"
            )

        new_cols = list(self._columns)
        new_cols.insert(loc, column)
        new_data = {c: list(self._inner.get_column(c).values) for c in self._columns}
        new_data[column] = vals
        self._reload(new_data)
        self._columns = new_cols

    # ---------- v2.0.0: filter / select_dtypes ----------

    def filter(
        self,
        items=None,
        like: Optional[str] = None,
        regex: Optional[str] = None,
        axis: int = 1,
    ) -> "DataFrame":
        """根据列名过滤 DataFrame。

        :param items: 要保留的列名列表
        :param like: 保留包含此字符串的列
        :param regex: 保留匹配正则表达式的列
        :param axis: 1=列, 0=行
        :return: DataFrame
        """
        import re

        if axis == 0:
            # 按行索引过滤
            if items is not None:
                indices = [i for i, idx in enumerate(self._index) if idx in items]
            elif like is not None:
                indices = [i for i, idx in enumerate(self._index) if like in str(idx)]
            elif regex is not None:
                pat = re.compile(regex)
                indices = [i for i, idx in enumerate(self._index) if pat.search(str(idx))]
            else:
                return self.copy()
            new_data = {c: [self._inner.get_column(c).values[i] for i in indices] for c in self._columns}
            return DataFrame(new_data)
        else:
            if items is not None:
                cols = [c for c in self._columns if c in items]
            elif like is not None:
                cols = [c for c in self._columns if like in c]
            elif regex is not None:
                pat = re.compile(regex)
                cols = [c for c in self._columns if pat.search(c)]
            else:
                return self.copy()
            new_data = {c: list(self._inner.get_column(c).values) for c in cols}
            return DataFrame(new_data)

    def select_dtypes(self, include=None, exclude=None) -> "DataFrame":
        """根据 dtype 选择列。

        :param include: 要包含的类型 (str / list[str] / type)
        :param exclude: 要排除的类型 (str / list[str] / type)
        :return: DataFrame
        """
        # 类型映射
        type_map = {
            "int": "int64",
            "int64": "int64",
            "float": "float64",
            "float64": "float64",
            "bool": "bool",
            "object": "object",
            "string": "object",
            "str": "object",
            "number": ("int64", "float64"),
        }

        def _to_dtype_set(types):
            if types is None:
                return set()
            if isinstance(types, str):
                types = [types]
            result = set()
            for t in types:
                if isinstance(t, type):
                    if t in (int,):
                        result.add("int64")
                    elif t in (float,):
                        result.add("float64")
                    elif t in (bool,):
                        result.add("bool")
                    elif t in (str,):
                        result.add("object")
                elif t in type_map:
                    mapped = type_map[t]
                    if isinstance(mapped, tuple):
                        result.update(mapped)
                    else:
                        result.add(mapped)
                else:
                    result.add(t)
            return result

        include_set = _to_dtype_set(include)
        exclude_set = _to_dtype_set(exclude)

        cols = []
        for c in self._columns:
            dt = self._inner.get_column(c).dtype
            if include_set and dt not in include_set:
                continue
            if exclude_set and dt in exclude_set:
                continue
            cols.append(c)

        new_data = {c: list(self._inner.get_column(c).values) for c in cols}
        return DataFrame(new_data)

    # ---------- v2.0.0: swapaxes / take / xs / get / lookup ----------

    def swapaxes(self, axis1, axis2, copy: bool = True) -> "DataFrame":
        """交换两个轴。

        :param axis1: 第一个轴 (0 或 1)
        :param axis2: 第二个轴 (0 或 1)
        :param copy: 是否返回副本
        :return: DataFrame
        """
        if {axis1, axis2} != {0, 1}:
            raise ValueError("axis must be 0 and 1")
        return self.transpose() if copy else self

    def transpose(self) -> "DataFrame":
        """转置 DataFrame (v0.2.0)。"""
        n = self._nrows
        new_data: Dict[str, list] = {}
        # 每行变成一列
        for i in range(n):
            new_data[str(i)] = [
                self._inner.get_column(c).values[i]
                for c in self._columns
            ]
        return DataFrame(new_data)

    @property
    def T(self) -> "DataFrame":
        return self.transpose()

    def take(self, indices, axis: int = 0) -> "DataFrame":
        """返回指定索引位置的元素。

        :param indices: 索引列表
        :param axis: 0=行, 1=列
        :return: DataFrame
        """
        if axis == 0:
            if isinstance(indices, int):
                indices = [indices]
            self._validate_indices(indices)
            new_data = {
                c: [self._inner.get_column(c).values[i] for i in indices]
                for c in self._columns
            }
            return DataFrame(new_data)
        else:
            if isinstance(indices, int):
                indices = [indices]
            cols = [self._columns[i] for i in indices]
            new_data = {c: list(self._inner.get_column(c).values) for c in cols}
            return DataFrame(new_data)

    def _validate_indices(self, indices):
        n = self._nrows
        for i in indices:
            if i < 0:
                i += n
            if i < 0 or i >= n:
                raise IndexError(f"index {i} out of range for axis 0 with size {n}")

    def xs(self, key, axis: int = 0, level=None, drop_level: bool = True) -> "Series":
        """返回跨截面 (cross-section)。

        :param key: 标签
        :param axis: 0=行, 1=列
        :param level: 多级索引层级
        :param drop_level: 是否删除层级
        :return: Series 或 DataFrame
        """
        if axis == 0:
            if isinstance(key, int):
                if key < 0:
                    key += self._nrows
                row = {c: self._inner.get_column(c).values[key] for c in self._columns}
                return Series(row, name=str(key))
            else:
                # 按标签查找
                try:
                    idx = self._index.index(key)
                except ValueError:
                    raise KeyError(f"label {key!r} not found in index")
                row = {c: self._inner.get_column(c).values[idx] for c in self._columns}
                return Series(row, name=str(key))
        else:
            # axis == 1: 按列名取
            if key in self._columns:
                return self._get_column_as_series(key)
            raise KeyError(f"column {key!r} not found")

    def get(self, key, default=None):
        """获取列，如果不存在则返回默认值。

        :param key: 列名
        :param default: 默认值
        :return: Series 或 default
        """
        if key in self._columns:
            return self._get_column_as_series(key)
        return default

    def lookup(self, row_labels, col_labels) -> list:
        """基于标签的查找 (已弃用于 pandas 2.1+)。

        :param row_labels: 行标签列表
        :param col_labels: 列标签列表
        :return: 值列表
        """
        result = []
        for rl, cl in zip(row_labels, col_labels):
            try:
                idx = self._index.index(rl)
            except ValueError:
                raise KeyError(f"row label {rl!r} not found")
            if cl not in self._columns:
                raise KeyError(f"column {cl!r} not found")
            result.append(self._inner.get_column(cl).values[idx])
        return result

    # ---------- v2.0.0: first / last / truncate ----------

    def first(self, offset) -> "DataFrame":
        """根据日期偏移选择前几段时间的数据。

        :param offset: 日期偏移字符串 (如 '5D')
        :return: DataFrame
        """
        return self._time_slice(offset, mode="first")

    def last(self, offset) -> "DataFrame":
        """根据日期偏移选择最后几段时间的数据。

        :param offset: 日期偏移字符串 (如 '5D')
        :return: DataFrame
        """
        return self._time_slice(offset, mode="last")

    def _time_slice(self, offset: str, mode: str) -> "DataFrame":
        """时间切片辅助方法。"""
        from datetime import datetime, timedelta

        # 解析 offset
        offset = offset.strip().upper()
        num = int(offset[:-1])
        unit = offset[-1]
        unit_map = {
            "D": "days", "H": "hours", "M": "minutes", "S": "seconds",
            "W": "weeks",
        }

        if unit not in unit_map:
            raise ValueError(f"unsupported offset: {offset}")

        # 尝试找到第一个日期时间索引
        idx_vals = self._index
        times = []
        for v in idx_vals:
            if isinstance(v, (datetime, str)):
                try:
                    if isinstance(v, str):
                        v = datetime.fromisoformat(v)
                    times.append(v)
                except (ValueError, TypeError):
                    continue
            elif isinstance(v, (int, float)):
                times.append(v)
            else:
                continue

        if not times:
            raise TypeError("first/last requires a datetime-like index")

        if isinstance(times[0], datetime):
            if mode == "first":
                start = min(times)
                end = start + timedelta(**{unit_map[unit]: num - 1})
                end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
                indices = [
                    i for i in range(self._nrows)
                    if isinstance(self._index[i], datetime) and self._index[i] <= end
                ]
            else:
                end = max(times)
                start = end - timedelta(**{unit_map[unit]: num - 1})
                indices = [
                    i for i in range(self._nrows)
                    if isinstance(self._index[i], datetime) and self._index[i] >= start
                ]
        else:
            # 数值索引
            if mode == "first":
                start = min(times)
                end = start + num
                indices = [
                    i for i in range(self._nrows)
                    if isinstance(self._index[i], (int, float)) and self._index[i] <= end
                ]
            else:
                end = max(times)
                start = end - num
                indices = [
                    i for i in range(self._nrows)
                    if isinstance(self._index[i], (int, float)) and self._index[i] >= start
                ]

        new_data = {
            c: [self._inner.get_column(c).values[i] for i in indices]
            for c in self._columns
        }
        return DataFrame(new_data)

    def truncate(self, before=None, after=None, axis: int = 0) -> "DataFrame":
        """截断 DataFrame 在某个索引值之前或之后。

        :param before: 截断此日期/值之前的数据
        :param after: 截断此日期/值之后的数据
        :param axis: 0=行, 1=列
        :return: DataFrame
        """
        if axis == 0:
            indices = list(range(self._nrows))
            if before is not None:
                from datetime import datetime
                if isinstance(before, str):
                    try:
                        before = datetime.fromisoformat(before)
                    except ValueError:
                        pass
                # 过滤掉 before 之前的行
                indices = [
                    i for i in indices
                    if not (
                        isinstance(self._index[i], type(before))
                        and self._index[i] < before
                    )
                ]
            if after is not None:
                from datetime import datetime
                if isinstance(after, str):
                    try:
                        after = datetime.fromisoformat(after)
                    except ValueError:
                        pass
                indices = [
                    i for i in indices
                    if not (
                        isinstance(self._index[i], type(after))
                        and self._index[i] > after
                    )
                ]
            new_data = {
                c: [self._inner.get_column(c).values[i] for i in indices]
                for c in self._columns
            }
            return DataFrame(new_data)
        else:
            # axis == 1: 截断列
            cols = list(self._columns)
            if before is not None:
                try:
                    idx = cols.index(before)
                    cols = cols[idx:]
                except ValueError:
                    pass
            if after is not None:
                try:
                    idx = cols.index(after)
                    cols = cols[:idx + 1]
                except ValueError:
                    pass
            new_data = {c: list(self._inner.get_column(c).values) for c in cols}
            return DataFrame(new_data)

    # ---------- v2.0.0: asfreq / tz_localize / tz_convert / between_time / at_time ----------

    def asfreq(self, freq, method=None, normalize: bool = False) -> "DataFrame":
        """将时间序列转换为指定频率。

        :param freq: 频率字符串 ('D'/'H'/'M'/'W'/'Y' 等)
        :param method: 填充方法 ('ffill'/'bfill'/None)
        :param normalize: 是否将时间归一化到午夜
        :return: DataFrame
        """
        from datetime import datetime, timedelta

        # 解析 freq
        freq = freq.strip().upper()
        unit_map = {
            "D": ("days", 1), "H": ("hours", 1), "h": ("hours", 1),
            "M": ("minutes", 0), "T": ("minutes", 1), "min": ("minutes", 1),
            "S": ("seconds", 1), "W": ("weeks", 1),
        }

        if freq not in unit_map:
            raise ValueError(f"unsupported freq: {freq!r}")

        unit_name, _ = unit_map[freq]

        # 尝试解析 index 为 datetime
        idx_vals = self._index
        times = []
        for v in idx_vals:
            if isinstance(v, datetime):
                times.append(v)
            elif isinstance(v, str):
                try:
                    times.append(datetime.fromisoformat(v))
                except (ValueError, TypeError):
                    raise TypeError(f"asfreq requires a DatetimeIndex, got {type(v).__name__}")
            else:
                raise TypeError(f"asfreq requires a DatetimeIndex, got {type(v).__name__}")

        if not times:
            return DataFrame({})

        # 生成目标频率的时间范围
        start = min(times)
        end = max(times)

        if normalize:
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = end.replace(hour=0, minute=0, second=0, microsecond=0)

        # 生成目标索引
        target_idx = []
        current = start
        while current <= end:
            target_idx.append(current)
            if unit_name == "days":
                current = current + timedelta(days=1)
            elif unit_name == "weeks":
                current = current + timedelta(weeks=1)
            elif unit_name == "hours":
                current = current + timedelta(hours=1)
            elif unit_name == "minutes":
                current = current + timedelta(minutes=1)
            elif unit_name == "seconds":
                current = current + timedelta(seconds=1)
            else:
                current = current + timedelta(days=1)

        # 对每一列，按目标索引重采样
        new_data: Dict[str, list] = {}
        for c in self._columns:
            col_vals = list(self._inner.get_column(c).values)
            col_data = []
            for t_target in target_idx:
                # 找到最接近的时间点
                matched_val = None
                for i, t_orig in enumerate(times):
                    if t_orig == t_target:
                        matched_val = col_vals[i]
                        break
                    elif t_orig <= t_target:
                        if method == "ffill":
                            matched_val = col_vals[i]
                        elif method is None:
                            matched_val = col_vals[i]
                    elif t_orig > t_target and method == "bfill":
                        if matched_val is None:
                            matched_val = col_vals[i]
                        break

                if matched_val is None and method is None:
                    matched_val = None

                col_data.append(matched_val)

            new_data[c] = col_data

        df = DataFrame(new_data)
        df._index = target_idx
        return df

    def tz_localize(self, tz, axis: int = 0, level=None, copy: bool = True) -> "DataFrame":
        """将 tz-naive 的 datetime 索引本地化为时区感知。

        :param tz: 时区字符串 (如 'Asia/Shanghai', 'UTC', 'US/Eastern')
        :param axis: 0=行索引, 1=列索引
        :param level: 多级索引层级
        :param copy: 是否返回副本
        :return: DataFrame
        """
        from datetime import datetime

        if axis == 0:
            # 尝试解析时区
            tzinfo = self._parse_timezone(tz)

            # 检查索引是否已经是时区感知的
            if self._index and any(isinstance(v, datetime) and v.tzinfo is not None for v in self._index):
                raise TypeError("Index is already tz-aware. Use tz_convert instead.")

            # 本地化 index
            new_index = []
            for v in self._index:
                if isinstance(v, datetime):
                    new_index.append(v.replace(tzinfo=tzinfo))
                elif isinstance(v, str):
                    try:
                        dt = datetime.fromisoformat(v)
                        new_index.append(dt.replace(tzinfo=tzinfo))
                    except (ValueError, TypeError):
                        new_index.append(v)
                else:
                    new_index.append(v)

            new_data = {c: list(self._inner.get_column(c).values) for c in self._columns}
            df = DataFrame(new_data)
            df._index = new_index
            return df
        else:
            # axis == 1: 列索引 (不常用)
            return self.copy() if copy else self

    def tz_convert(self, tz, axis: int = 0, level=None, copy: bool = True) -> "DataFrame":
        """将 tz-aware 的 datetime 索引转换为另一个时区。

        :param tz: 目标时区字符串 (如 'Asia/Shanghai', 'UTC', 'US/Eastern')
        :param axis: 0=行索引, 1=列索引
        :param level: 多级索引层级
        :param copy: 是否返回副本
        :return: DataFrame
        """
        from datetime import datetime

        if axis == 0:
            # 检查索引是否是时区感知的
            has_tz = any(isinstance(v, datetime) and v.tzinfo is not None for v in self._index)
            if not has_tz:
                raise TypeError("Index is not tz-aware. Use tz_localize first.")

            tzinfo = self._parse_timezone(tz)

            # 转换时区
            new_index = []
            for v in self._index:
                if isinstance(v, datetime) and v.tzinfo is not None:
                    new_index.append(v.astimezone(tzinfo))
                else:
                    new_index.append(v)

            new_data = {c: list(self._inner.get_column(c).values) for c in self._columns}
            df = DataFrame(new_data)
            df._index = new_index
            return df
        else:
            return self.copy() if copy else self

    @staticmethod
    def _parse_timezone(tz: str):
        """解析时区字符串为 tzinfo 对象。"""
        from datetime import timezone, timedelta as td

        if tz is None:
            return None

        # 尝试固定偏移量格式: +08:00, -05:00, UTC+8 等
        tz = str(tz).strip()

        if tz.upper() == "UTC":
            return timezone.utc

        # 尝试 UTC+X 或 UTC-X 格式
        if tz.upper().startswith("UTC"):
            offset_str = tz[3:].strip()
            try:
                hours = float(offset_str)
                return timezone(td(hours=hours))
            except ValueError:
                pass

        # 尝试 +HH:MM 或 -HH:MM 格式
        if tz.startswith("+") or tz.startswith("-"):
            try:
                # Parse +08:00 format
                sign = 1 if tz.startswith("+") else -1
                parts = tz[1:].split(":")
                hours = int(parts[0])
                minutes = int(parts[1]) if len(parts) > 1 else 0
                return timezone(td(hours=sign * hours, minutes=sign * minutes))
            except (ValueError, IndexError):
                pass

        # 尝试使用 pytz (如果可用)
        try:
            import pytz
            return pytz.timezone(tz)
        except ImportError:
            pass

        # 尝试使用 zoneinfo (Python 3.9+)
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(tz)
        except (ImportError, Exception):
            pass

        raise ValueError(f"Unable to parse timezone: {tz!r}")

    def between_time(
        self,
        start_time,
        end_time,
        include_start: bool = True,
        include_end: bool = True,
        axis: int = 0,
    ) -> "DataFrame":
        """选择一天中特定时间段内的值。

        :param start_time: 起始时间 (datetime.time 或 str 如 '09:00:00')
        :param end_time: 结束时间
        :param include_start: 是否包含起始时间
        :param include_end: 是否包含结束时间
        :param axis: 0=行, 1=列
        :return: DataFrame
        """
        from datetime import datetime, time

        # 解析 start_time / end_time
        if isinstance(start_time, str):
            start_time = time.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = time.fromisoformat(end_time)

        if axis == 0:
            indices = []
            for i in range(self._nrows):
                idx_val = self._index[i]

                # 提取时间部分
                if isinstance(idx_val, datetime):
                    t = idx_val.time()
                elif isinstance(idx_val, str):
                    try:
                        t = datetime.fromisoformat(idx_val).time()
                    except (ValueError, TypeError):
                        continue
                else:
                    continue

                # 判断是否在时间范围内
                if include_start and include_end:
                    in_range = start_time <= t <= end_time
                elif include_start:
                    in_range = start_time <= t < end_time
                elif include_end:
                    in_range = start_time < t <= end_time
                else:
                    in_range = start_time < t < end_time

                if in_range:
                    indices.append(i)

            new_data = {
                c: [self._inner.get_column(c).values[i] for i in indices]
                for c in self._columns
            }
            return DataFrame(new_data)
        else:
            # axis == 1: 不常用
            return self.copy()

    def at_time(self, time, axis: int = 0) -> "DataFrame":
        """选择一天中特定时间点的值。

        :param time: 目标时间 (datetime.time 或 str 如 '09:00:00')
        :param axis: 0=行, 1=列
        :return: DataFrame
        """
        from datetime import datetime, time as dt_time

        if isinstance(time, str):
            time = dt_time.fromisoformat(time)

        if axis == 0:
            indices = []
            for i in range(self._nrows):
                idx_val = self._index[i]

                if isinstance(idx_val, datetime):
                    t = idx_val.time()
                elif isinstance(idx_val, str):
                    try:
                        t = datetime.fromisoformat(idx_val).time()
                    except (ValueError, TypeError):
                        continue
                else:
                    continue

                if t == time:
                    indices.append(i)

            new_data = {
                c: [self._inner.get_column(c).values[i] for i in indices]
                for c in self._columns
            }
            return DataFrame(new_data)
        else:
            return self.copy()

    # ---------- v2.0.0: 累计操作 ----------

    def cumsum(self, axis: int = 0, skipna: bool = True) -> "DataFrame":
        """返回每列的累计和。

        :param axis: 0=列方向, 1=行方向
        :param skipna: 是否跳过 NaN
        """
        if axis == 0:
            new_data = {}
            for c in self._columns:
                vals = list(self._inner.get_column(c).values)
                result = []
                acc = 0
                for v in vals:
                    if v is None:
                        result.append(None if not skipna else acc)
                    else:
                        acc = v + (acc if isinstance(v, (int, float)) else 0)
                        result.append(acc)
                new_data[c] = result
            return DataFrame(new_data)
        else:
            return self.T.cumsum(axis=0).T

    def cumprod(self, axis: int = 0, skipna: bool = True) -> "DataFrame":
        """返回每列的累计积。

        :param axis: 0=列方向, 1=行方向
        :param skipna: 是否跳过 NaN
        """
        if axis == 0:
            new_data = {}
            for c in self._columns:
                vals = list(self._inner.get_column(c).values)
                result = []
                acc = 1
                for v in vals:
                    if v is None:
                        result.append(None if not skipna else acc)
                    else:
                        acc = v * (acc if isinstance(v, (int, float)) else 1)
                        result.append(acc)
                new_data[c] = result
            return DataFrame(new_data)
        else:
            return self.T.cumprod(axis=0).T

    def cummax(self, axis: int = 0, skipna: bool = True) -> "DataFrame":
        """返回每列的累计最大值。

        :param axis: 0=列方向, 1=行方向
        :param skipna: 是否跳过 NaN
        """
        if axis == 0:
            new_data = {}
            for c in self._columns:
                vals = list(self._inner.get_column(c).values)
                result = []
                acc = None
                for v in vals:
                    if v is not None:
                        acc = v if acc is None else max(acc, v)
                    result.append(acc)
                new_data[c] = result
            return DataFrame(new_data)
        else:
            return self.T.cummax(axis=0).T

    def cummin(self, axis: int = 0, skipna: bool = True) -> "DataFrame":
        """返回每列的累计最小值。

        :param axis: 0=列方向, 1=行方向
        :param skipna: 是否跳过 NaN
        """
        if axis == 0:
            new_data = {}
            for c in self._columns:
                vals = list(self._inner.get_column(c).values)
                result = []
                acc = None
                for v in vals:
                    if v is not None:
                        acc = v if acc is None else min(acc, v)
                    result.append(acc)
                new_data[c] = result
            return DataFrame(new_data)
        else:
            return self.T.cummin(axis=0).T

    def cumcount(self, axis: int = 0) -> "Series":
        """返回每列的累计计数 (跳过 None 值)。

        :param axis: 0=列方向, 1=行方向
        """
        if axis == 0:
            new_data = {}
            for c in self._columns:
                vals = list(self._inner.get_column(c).values)
                result = []
                cnt = 0
                for v in vals:
                    if v is not None:
                        cnt += 1
                    result.append(cnt)
                new_data[c] = result
            return DataFrame(new_data)
        else:
            return self.T.cumcount(axis=0)

    # ---------- v2.0.0: 时序操作 ----------

    def shift(self, periods: int = 1, axis: int = 0) -> "DataFrame":
        """将数据按行/列平移。

        :param periods: 平移的步数 (正数向下/右, 负数向上/左)
        :param axis: 0=行方向, 1=列方向
        """
        if axis == 0:
            new_data = {}
            for c in self._columns:
                vals = list(self._inner.get_column(c).values)
                if periods >= 0:
                    shifted = ([None] * periods) + vals[:len(vals) - periods]
                else:
                    p = -periods
                    shifted = vals[p:] + ([None] * p)
                new_data[c] = shifted
            return DataFrame(new_data)
        else:
            return self.T.shift(periods, axis=0).T

    def diff(self, periods: int = 1, axis: int = 0) -> "DataFrame":
        """计算每列的差分。

        :param periods: 差分步数
        :param axis: 0=列方向, 1=行方向
        """
        if axis == 0:
            new_data = {}
            for c in self._columns:
                vals = list(self._inner.get_column(c).values)
                result = []
                for i in range(len(vals)):
                    if i < periods:
                        result.append(None)
                    elif vals[i] is None or vals[i - periods] is None:
                        result.append(None)
                    else:
                        result.append(vals[i] - vals[i - periods])
                new_data[c] = result
            return DataFrame(new_data)
        else:
            return self.T.diff(periods, axis=0).T

    def pct_change(self, periods: int = 1) -> "DataFrame":
        """计算每列的百分比变化。

        :param periods: 差分步数
        """
        new_data = {}
        for c in self._columns:
            vals = list(self._inner.get_column(c).values)
            result = []
            for i in range(len(vals)):
                if i < periods or vals[i - periods] is None or vals[i - periods] == 0:
                    result.append(None)
                elif vals[i] is None:
                    result.append(None)
                else:
                    result.append((vals[i] - vals[i - periods]) / vals[i - periods])
            new_data[c] = result
        return DataFrame(new_data)

    # ---------- v2.0.0: 统计方法 ----------

    def rank(self, axis: int = 0, method: str = "average", ascending: bool = True) -> "DataFrame":
        """计算每列的排名。

        :param axis: 0=列方向, 1=行方向
        :param method: 'average'/'min'/'max'/'first'/'dense'
        :param ascending: 是否升序
        """
        if axis == 0:
            new_data = {}
            for c in self._columns:
                vals = list(self._inner.get_column(c).values)
                # 建立 (value, original_index) 对，排除 None
                indexed = [(v, i) for i, v in enumerate(vals) if v is not None]
                if not indexed:
                    new_data[c] = [None] * len(vals)
                    continue
                # 排序
                indexed.sort(key=lambda x: x[0], reverse=not ascending)
                ranks = [None] * len(vals)
                if method == "dense":
                    rank = 0
                    prev = None
                    for v, i in indexed:
                        if prev is None or v != prev:
                            rank += 1
                        ranks[i] = rank
                        prev = v
                elif method == "min":
                    rank = 0
                    for j, (v, i) in enumerate(indexed):
                        if j == 0 or v != indexed[j - 1][0]:
                            ranks[i] = j + 1
                        else:
                            ranks[i] = ranks[indexed[j - 1][1]]
                elif method == "max":
                    # 先计算 min，再按组替换
                    min_ranks = [None] * len(vals)
                    for j, (v, i) in enumerate(indexed):
                        if j == 0 or v != indexed[j - 1][0]:
                            min_ranks[i] = j + 1
                        else:
                            min_ranks[i] = min_ranks[indexed[j - 1][1]]
                    # 反向遍历替换为 max
                    for j in range(len(indexed) - 1, -1, -1):
                        v, i = indexed[j]
                        if j == len(indexed) - 1 or v != indexed[j + 1][0]:
                            ranks[i] = j + 1
                        else:
                            ranks[i] = ranks[indexed[j + 1][1]]
                elif method == "first":
                    for j, (v, i) in enumerate(indexed):
                        ranks[i] = j + 1
                else:  # average
                    group_start = 0
                    for j in range(1, len(indexed) + 1):
                        if j == len(indexed) or indexed[j][0] != indexed[group_start][0]:
                            n = j - group_start
                            avg_rank = group_start + 1 + (n - 1) / 2.0
                            for k in range(group_start, j):
                                ranks[indexed[k][1]] = avg_rank
                            group_start = j
                new_data[c] = ranks
            return DataFrame(new_data)
        else:
            return self.T.rank(axis=0, method=method, ascending=ascending).T

    def quantile(self, q=0.5, axis: int = 0) -> "Series":
        """计算每列的分位数。

        :param q: 分位数 (0-1) 或 list
        :param axis: 0=列方向, 1=行方向
        """
        from .series import Series
        if axis == 0:
            q_list = [q] if not isinstance(q, (list, tuple)) else list(q)
            new_data = {}
            for c in self._columns:
                vals = [v for v in self._inner.get_column(c).values if v is not None]
                if not vals:
                    new_data[c] = [None] * len(q_list)
                    continue
                vals.sort()
                col_quants = []
                for qv in q_list:
                    pos = qv * (len(vals) - 1)
                    lo = int(pos)
                    hi = min(lo + 1, len(vals) - 1)
                    frac = pos - lo
                    col_quants.append(vals[lo] + (vals[hi] - vals[lo]) * frac)
                new_data[c] = col_quants
            if len(q_list) == 1:
                return Series({c: new_data[c][0] for c in self._columns})
            return DataFrame(dict((c, new_data[c]) for c in self._columns))
        else:
            return self.T.quantile(q, axis=0)

    def mode(self, axis: int = 0, dropna: bool = True) -> "DataFrame":
        """计算每列的众数。

        :param axis: 0=列方向, 1=行方向
        :param dropna: 是否忽略 NaN
        """
        if axis == 0:
            new_data = {}
            for c in self._columns:
                vals = list(self._inner.get_column(c).values)
                if dropna:
                    vals = [v for v in vals if v is not None]
                from collections import Counter
                cnt = Counter(vals)
                max_count = max(cnt.values()) if cnt else 0
                modes = [k for k, v in cnt.items() if v == max_count]
                new_data[c] = modes if modes else [None]
            # 对齐长度
            max_len = max(len(v) for v in new_data.values()) if new_data else 0
            for c in new_data:
                if len(new_data[c]) < max_len:
                    new_data[c].extend([None] * (max_len - len(new_data[c])))
            return DataFrame(new_data)
        else:
            return self.T.mode(axis=0, dropna=dropna)

    def skew(self, axis: int = 0, skipna: bool = True) -> "Series":
        """计算每列的偏度。

        :param axis: 0=列方向, 1=行方向
        :param skipna: 是否跳过 NaN
        """
        from .series import Series
        if axis == 0:
            result = {}
            for c in self._columns:
                vals = list(self._inner.get_column(c).values)
                if skipna:
                    vals = [v for v in vals if v is not None]
                n = len(vals)
                if n < 3:
                    result[c] = None
                    continue
                mean = sum(vals) / n
                m2 = sum((v - mean) ** 2 for v in vals)
                m3 = sum((v - mean) ** 3 for v in vals)
                if m2 == 0:
                    result[c] = None
                else:
                    result[c] = (n ** 0.5 * m3) / (m2 ** 1.5)
            return Series(result)
        else:
            return self.T.skew(axis=0, skipna=skipna)

    def kurt(self, axis: int = 0, skipna: bool = True) -> "Series":
        """计算每列的峰度 (Fisher 定义，正态分布峰度=0)。

        :param axis: 0=列方向, 1=行方向
        :param skipna: 是否跳过 NaN
        """
        from .series import Series
        if axis == 0:
            result = {}
            for c in self._columns:
                vals = list(self._inner.get_column(c).values)
                if skipna:
                    vals = [v for v in vals if v is not None]
                n = len(vals)
                if n < 4:
                    result[c] = None
                    continue
                mean = sum(vals) / n
                m2 = sum((v - mean) ** 2 for v in vals)
                m4 = sum((v - mean) ** 4 for v in vals)
                if m2 == 0:
                    result[c] = None
                else:
                    result[c] = (n * (n + 1) * m4) / ((n - 1) * (n - 2) * (n - 3) * m2 ** 2) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
            return Series(result)
        else:
            return self.T.kurt(axis=0, skipna=skipna)

    def mad(self, axis: int = 0, skipna: bool = True) -> "Series":
        """计算每列的平均绝对偏差 (Mean Absolute Deviation)。

        :param axis: 0=列方向, 1=行方向
        :param skipna: 是否跳过 NaN
        """
        from .series import Series
        if axis == 0:
            result = {}
            for c in self._columns:
                vals = list(self._inner.get_column(c).values)
                if skipna:
                    vals = [v for v in vals if v is not None]
                if not vals:
                    result[c] = None
                    continue
                mean = sum(vals) / len(vals)
                result[c] = sum(abs(v - mean) for v in vals) / len(vals)
            return Series(result)
        else:
            return self.T.mad(axis=0, skipna=skipna)

    def idxmax(self, axis: int = 0, skipna: bool = True) -> "Series":
        """返回每列最大值所在的索引。

        :param axis: 0=列方向, 1=行方向
        :param skipna: 是否跳过 NaN
        """
        from .series import Series
        if axis == 0:
            result = {}
            for c in self._columns:
                vals = list(self._inner.get_column(c).values)
                best_idx = None
                best_val = None
                for i, v in enumerate(vals):
                    if skipna and v is None:
                        continue
                    if best_val is None or v > best_val:
                        best_val = v
                        best_idx = self._index[i] if self._index and i < len(self._index) else i
                result[c] = best_idx
            return Series(result)
        else:
            return self.T.idxmax(axis=0, skipna=skipna)

    def idxmin(self, axis: int = 0, skipna: bool = True) -> "Series":
        """返回每列最小值所在的索引。

        :param axis: 0=列方向, 1=行方向
        :param skipna: 是否跳过 NaN
        """
        from .series import Series
        if axis == 0:
            result = {}
            for c in self._columns:
                vals = list(self._inner.get_column(c).values)
                best_idx = None
                best_val = None
                for i, v in enumerate(vals):
                    if skipna and v is None:
                        continue
                    if best_val is None or v < best_val:
                        best_val = v
                        best_idx = self._index[i] if self._index and i < len(self._index) else i
                result[c] = best_idx
            return Series(result)
        else:
            return self.T.idxmin(axis=0, skipna=skipna)

    # ---------- v2.0.0: 排序 ----------

    def sort_index(self, axis: int = 0, ascending: bool = True) -> "DataFrame":
        """按索引排序。

        :param axis: 0=行方向, 1=列方向
        :param ascending: 是否升序
        """
        if axis == 0:
            if not self._index:
                return self.copy()
            indexed = list(enumerate(self._index))
            # 排序时 None 放最后
            indexed.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else ""), reverse=not ascending)
            new_order = [i for i, _ in indexed]
            new_data = {
                c: [self._inner.get_column(c).values[i] for i in new_order]
                for c in self._columns
            }
            df = DataFrame(new_data)
            df._index = [self._index[i] for i in new_order]
            return df
        else:
            cols = sorted(self._columns, reverse=not ascending)
            return self[cols]

    def sort_columns(self) -> "DataFrame":
        """按列名排序。"""
        return self.sort_index(axis=1)

    # ---------- v2.0.0: 转换 ----------

    def clip(self, lower=None, upper=None, axis: int = 0) -> "DataFrame":
        """裁剪每列的值到指定范围。

        :param lower: 下界
        :param upper: 上界
        :param axis: 0=列方向, 1=行方向
        """
        if axis == 0:
            new_data = {}
            for c in self._columns:
                vals = list(self._inner.get_column(c).values)
                clipped = []
                for v in vals:
                    if v is None:
                        clipped.append(None)
                    else:
                        result = v
                        if lower is not None and result < lower:
                            result = lower
                        if upper is not None and result > upper:
                            result = upper
                        clipped.append(result)
                new_data[c] = clipped
            return DataFrame(new_data)
        else:
            return self.T.clip(lower, upper, axis=0).T

    def astype(self, dtype: str) -> "DataFrame":
        """转换每列的数据类型。

        :param dtype: 目标类型 (如 'int64'/'float64'/'object'/'bool')
        """
        if isinstance(dtype, dict):
            new_data = {}
            for c in self._columns:
                target = dtype.get(c, None)
                if target is None:
                    new_data[c] = list(self._inner.get_column(c).values)
                else:
                    ser = self._get_column_as_series(c)
                    new_data[c] = list(ser.astype(target).values)
            return DataFrame(new_data)
        else:
            new_data = {}
            for c in self._columns:
                ser = self._get_column_as_series(c)
                new_data[c] = list(ser.astype(dtype).values)
            return DataFrame(new_data)

    # ---------- v2.0.0: 概览 ----------

    def memory_usage(self, index: bool = True, deep: bool = False) -> "Series":
        """返回每列的内存使用量 (字节)。

        :param index: 是否包含索引
        :param deep: 是否深度计算 (字符串等)
        """
        import sys
        from .series import Series
        result = {}
        for c in self._columns:
            total = 0
            for v in self._inner.get_column(c).values:
                if deep and isinstance(v, str):
                    total += sys.getsizeof(v)
                else:
                    total += 8  # 指针大小估计
            result[c] = total
        if index:
            result["Index"] = len(self._index) * 8 if self._index else 0
        return Series(result)

    # ---------- v2.0.0: 数据访问 ----------

    def first_valid_index(self) -> Any:
        """返回第一个非 NaN 行所在的索引。"""
        for i in range(self._nrows):
            row = [self._inner.get_column(c).values[i] for c in self._columns]
            if any(v is not None for v in row):
                return self._index[i] if self._index and i < len(self._index) else i
        return None

    def last_valid_index(self) -> Any:
        """返回最后一个非 NaN 行所在的索引。"""
        for i in range(self._nrows - 1, -1, -1):
            row = [self._inner.get_column(c).values[i] for c in self._columns]
            if any(v is not None for v in row):
                return self._index[i] if self._index and i < len(self._index) else i
        return None

    # ---------- v2.0.0: 其他 ----------

    def rename_axis(self, mapper, axis: int = 0) -> "DataFrame":
        """重命名轴标签。

        :param mapper: 标量或函数
        :param axis: 0=行, 1=列
        """
        if axis == 0:
            new_name = mapper(self._index_name()) if callable(mapper) else mapper
            df = self.copy()
            df._index_name_val = new_name
            return df
        else:
            new_name = mapper(self._columns_name) if callable(mapper) else mapper
            df = self.copy()
            df._columns_name = new_name
            return df

    def explode(self, column, ignore_index: bool = False) -> "DataFrame":
        """将列表类列展开为多行。

        :param column: 要展开的列名
        :param ignore_index: 是否重置索引
        """
        if isinstance(column, str):
            column = [column]
        col_vals = {}
        for c in self._columns:
            col_vals[c] = list(self._inner.get_column(c).values)

        # 对每个展开列，计算展开后的行数
        new_data = {c: [] for c in self._columns}
        for i in range(self._nrows):
            # 计算展开倍数
            explode_lens = []
            for ec in column:
                v = col_vals[ec][i]
                if isinstance(v, (list, tuple)):
                    explode_lens.append(len(v))
                else:
                    explode_lens.append(1)
            max_len = max(explode_lens) if explode_lens else 1

            for j in range(max_len):
                for c in self._columns:
                    v = col_vals[c][i]
                    if c in column:
                        if isinstance(v, (list, tuple)):
                            new_data[c].append(v[j] if j < len(v) else None)
                        else:
                            new_data[c].append(v if j == 0 else None)
                    else:
                        new_data[c].append(v)

        df = DataFrame(new_data)
        if ignore_index:
            df._index = list(range(len(df)))
        return df


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
                    result[c].append(sub.values[0] if len(sub) > 0 else None)
                elif func == "last":
                    result[c].append(sub.values[-1] if len(sub) > 0 else None)
                else:
                    raise ValueError(f"unsupported agg: {func}")
        return DataFrame(result)

    def sum(self) -> "DataFrame":
        return self._agg({c: "sum" for c in self._df._columns if c not in self._by})

    def mean(self) -> "DataFrame":
        numeric_cols = [
            c for c in self._df._columns
            if c not in self._by and self._df._inner.get_column(c).dtype in ("int64", "float64")
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

    # ---------- 分组取值扩展 (v1.4.0) ----------

    def first(self) -> "DataFrame":
        """返回每个分组的第一行。"""
        return self._agg({c: "first" for c in self._df._columns if c not in self._by})

    def last(self) -> "DataFrame":
        """返回每个分组的最后一行。"""
        return self._agg({c: "last" for c in self._df._columns if c not in self._by})

    def nth(self, n: int) -> "DataFrame":
        """返回每个分组的第 n 行。

        :param n: 行索引 (0-based, 支持负数)
        """
        result: Dict[str, list] = {c: [] for c in self._by}
        other_cols = [c for c in self._df._columns if c not in self._by]
        for c in other_cols:
            result[c] = []

        for key, idxs in self._groups.items():
            for k, c in zip(key, self._by):
                result[c].append(k)
            for c in other_cols:
                ser = self._df[c]
                sub_vals = ser.iloc(idxs).values
                if n < 0:
                    actual_n = len(sub_vals) + n
                else:
                    actual_n = n
                if 0 <= actual_n < len(sub_vals):
                    result[c].append(sub_vals[actual_n])
                else:
                    result[c].append(None)

        return DataFrame(result)

    # ---------- v2.0.0: GroupBy 扩展 ----------

    def ngroup(self) -> "Series":
        """返回每个分组的编号 (0-based)。"""
        from .series import Series
        group_ids = {}
        for i, key in enumerate(self._groups):
            group_ids[key] = i
        # 为每行分配组号
        n = self._df._nrows
        group_nums = [None] * n
        for key, idxs in self._groups.items():
            gid = group_ids[key]
            for idx in idxs:
                group_nums[idx] = gid
        return Series(group_nums)

    def cumcount(self, ascending: bool = True) -> "Series":
        """返回每个分组内的累计计数 (0-based)。"""
        from .series import Series
        n = self._df._nrows
        result = [None] * n
        for idxs in self._groups.values():
            if ascending:
                for i, idx in enumerate(idxs):
                    result[idx] = i
            else:
                for i, idx in enumerate(reversed(idxs)):
                    result[idx] = i
        return Series(result)

    def rank(self, method: str = "average", ascending: bool = True) -> "DataFrame":
        """返回每个分组内的排名。

        :param method: 'average'/'min'/'max'/'first'/'dense'
        :param ascending: 是否升序
        """
        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {c: [None] * self._df._nrows for c in other_cols}

        for idxs in self._groups.values():
            for c in other_cols:
                ser = self._df[c]
                sub_vals = [ser.values[i] for i in idxs]
                # 在每个组内排名
                indexed = [(v, i) for i, v in enumerate(sub_vals) if v is not None]
                if not indexed:
                    for i in idxs:
                        result[c][i] = None
                    continue
                indexed.sort(key=lambda x: x[0], reverse=not ascending)
                ranks = [None] * len(sub_vals)
                if method == "dense":
                    rank = 0
                    prev = None
                    for v, i in indexed:
                        if prev is None or v != prev:
                            rank += 1
                        ranks[i] = rank
                        prev = v
                elif method == "min":
                    for j, (v, i) in enumerate(indexed):
                        if j == 0 or v != indexed[j - 1][0]:
                            ranks[i] = j + 1
                        else:
                            ranks[i] = ranks[indexed[j - 1][1]]
                elif method == "max":
                    min_ranks = [None] * len(sub_vals)
                    for j, (v, i) in enumerate(indexed):
                        if j == 0 or v != indexed[j - 1][0]:
                            min_ranks[i] = j + 1
                        else:
                            min_ranks[i] = min_ranks[indexed[j - 1][1]]
                    for j in range(len(indexed) - 1, -1, -1):
                        v, i = indexed[j]
                        if j == len(indexed) - 1 or v != indexed[j + 1][0]:
                            ranks[i] = j + 1
                        else:
                            ranks[i] = ranks[indexed[j + 1][1]]
                elif method == "first":
                    for j, (v, i) in enumerate(indexed):
                        ranks[i] = j + 1
                else:  # average
                    group_start = 0
                    for j in range(1, len(indexed) + 1):
                        if j == len(indexed) or indexed[j][0] != indexed[group_start][0]:
                            n_g = j - group_start
                            avg_rank = group_start + 1 + (n_g - 1) / 2.0
                            for k in range(group_start, j):
                                ranks[indexed[k][1]] = avg_rank
                            group_start = j
                for j, idx in enumerate(idxs):
                    result[c][idx] = ranks[j]

        return DataFrame(result)

    def quantile(self, q=0.5) -> "DataFrame":
        """返回每个分组内的分位数。

        :param q: 分位数 (0-1)
        """
        result: Dict[str, list] = {c: [] for c in self._by}
        other_cols = [c for c in self._df._columns if c not in self._by]
        for c in other_cols:
            result[c] = []

        for key, idxs in self._groups.items():
            for k, c in zip(key, self._by):
                result[c].append(k)
            for c in other_cols:
                vals = [v for v in self._df[c].iloc(idxs).values if v is not None]
                if not vals:
                    result[c].append(None)
                    continue
                vals.sort()
                pos = q * (len(vals) - 1)
                lo = int(pos)
                hi = min(lo + 1, len(vals) - 1)
                frac = pos - lo
                result[c].append(vals[lo] + (vals[hi] - vals[lo]) * frac)

        return DataFrame(result)

    def corr(self, other_col: str) -> "DataFrame":
        """计算每个分组内两列的相关系数。

        :param other_col: 目标列名
        """
        numeric_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {c: [] for c in self._by}
        for c in numeric_cols:
            if c != other_col:
                result[c] = []

        for key, idxs in self._groups.items():
            for k, c in zip(key, self._by):
                result[c].append(k)
            # 获取 other_col 的值
            other_vals = [self._df._inner.get_column(other_col).values[i] for i in idxs]
            for c in numeric_cols:
                if c == other_col:
                    continue
                col_vals = [self._df._inner.get_column(c).values[i] for i in idxs]
                pairs = [(a, b) for a, b in zip(col_vals, other_vals) if a is not None and b is not None]
                if len(pairs) < 2:
                    result[c].append(None)
                    continue
                ma = sum(a for a, b in pairs) / len(pairs)
                mb = sum(b for a, b in pairs) / len(pairs)
                num = sum((a - ma) * (b - mb) for a, b in pairs)
                da = (sum((a - ma) ** 2 for a, b in pairs)) ** 0.5
                db = (sum((b - mb) ** 2 for a, b in pairs)) ** 0.5
                if da == 0 or db == 0:
                    result[c].append(None)
                else:
                    result[c].append(num / (da * db))

        return DataFrame(result)

    def cov(self, other_col: str) -> "DataFrame":
        """计算每个分组内两列的协方差。

        :param other_col: 目标列名
        """
        numeric_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {c: [] for c in self._by}
        for c in numeric_cols:
            if c != other_col:
                result[c] = []

        for key, idxs in self._groups.items():
            for k, c in zip(key, self._by):
                result[c].append(k)
            other_vals = [self._df._inner.get_column(other_col).values[i] for i in idxs]
            for c in numeric_cols:
                if c == other_col:
                    continue
                col_vals = [self._df._inner.get_column(c).values[i] for i in idxs]
                pairs = [(a, b) for a, b in zip(col_vals, other_vals) if a is not None and b is not None]
                if len(pairs) < 2:
                    result[c].append(None)
                    continue
                ma = sum(a for a, b in pairs) / len(pairs)
                mb = sum(b for a, b in pairs) / len(pairs)
                result[c].append(sum((a - ma) * (b - mb) for a, b in pairs) / len(pairs))

        return DataFrame(result)

    def corrwith(self, other: "DataFrame") -> "Series":
        """计算每个分组内与另一个 DataFrame 的列相关系数。

        :param other: 另一个 DataFrame
        """
        from .series import Series
        result: Dict[str, float] = {}
        for c in self._df._columns:
            if c in self._by or c not in other._columns:
                continue
            all_pairs = []
            for idxs in self._groups.values():
                col_a = [self._df._inner.get_column(c).values[i] for i in idxs]
                col_b = [other._inner.get_column(c).values[i] for i in idxs]
                all_pairs.extend([(a, b) for a, b in zip(col_a, col_b) if a is not None and b is not None])
            if len(all_pairs) < 2:
                result[c] = None
                continue
            ma = sum(a for a, b in all_pairs) / len(all_pairs)
            mb = sum(b for a, b in all_pairs) / len(all_pairs)
            num = sum((a - ma) * (b - mb) for a, b in all_pairs)
            da = (sum((a - ma) ** 2 for a, b in all_pairs)) ** 0.5
            db = (sum((b - mb) ** 2 for a, b in all_pairs)) ** 0.5
            result[c] = num / (da * db) if da > 0 and db > 0 else None
        return Series(result)

    def pct_change(self, periods: int = 1) -> "DataFrame":
        """返回每个分组内的百分比变化。"""
        result: Dict[str, list] = {}
        for c in self._df._columns:
            result[c] = [None] * self._df._nrows

        for idxs in self._groups.values():
            for c in self._df._columns:
                if c in self._by:
                    continue
                vals = [self._df._inner.get_column(c).values[i] for i in idxs]
                for j, idx in enumerate(idxs):
                    if j < periods:
                        result[c][idx] = None
                    elif vals[j - periods] is None or vals[j - periods] == 0 or vals[j] is None:
                        result[c][idx] = None
                    else:
                        result[c][idx] = (vals[j] - vals[j - periods]) / vals[j - periods]

        return DataFrame(result)

    def rolling(self, window: int, min_periods=None) -> "DataFrame":
        """返回每个分组内的滚动窗口聚合结果 (按组应用 rolling)。"""
        from .series import Rolling
        if min_periods is None:
            min_periods = window
        result: Dict[str, list] = {}
        for c in self._df._columns:
            result[c] = [None] * self._df._nrows

        for idxs in self._groups.values():
            for c in self._df._columns:
                if c in self._by:
                    continue
                vals = [self._df._inner.get_column(c).values[i] for i in idxs]
                r = Rolling(Series(vals), window, min_periods)
                means = r.mean().values
                for j, idx in enumerate(idxs):
                    result[c][idx] = means[j]

        return DataFrame(result)

    def expanding(self, min_periods: int = 1) -> "DataFrame":
        """返回每个分组内的扩展窗口聚合结果 (按组应用 expanding)。"""
        from .series import Expanding
        result: Dict[str, list] = {}
        for c in self._df._columns:
            result[c] = [None] * self._df._nrows

        for idxs in self._groups.values():
            for c in self._df._columns:
                if c in self._by:
                    continue
                vals = [self._df._inner.get_column(c).values[i] for i in idxs]
                e = Expanding(Series(vals), min_periods)
                means = e.mean().values
                for j, idx in enumerate(idxs):
                    result[c][idx] = means[j]

        return DataFrame(result)

    def ewm(self, **kwargs) -> "DataFrame":
        """返回每个分组内的指数加权移动窗口 (按组应用 ewm)。"""
        from .series import EWM
        result: Dict[str, list] = {}
        for c in self._df._columns:
            result[c] = [None] * self._df._nrows

        for idxs in self._groups.values():
            for c in self._df._columns:
                if c in self._by:
                    continue
                vals = [self._df._inner.get_column(c).values[i] for i in idxs]
                ew = EWM(Series(vals), **kwargs)
                means = ew.mean().values
                for j, idx in enumerate(idxs):
                    result[c][idx] = means[j]

        return DataFrame(result)

    def resample(self, freq: str) -> "DataFrame":
        """返回每个分组内的重采样聚合结果 (按组应用 resample)。"""
        from .series import Resampler
        result: Dict[str, list] = {}
        for c in self._df._columns:
            result[c] = [None] * self._df._nrows

        for idxs in self._groups.values():
            for c in self._df._columns:
                if c in self._by:
                    continue
                vals = [self._df._inner.get_column(c).values[i] for i in idxs]
                r = Resampler(Series(vals), freq)
                sums = r.sum().values
                for j, idx in enumerate(idxs):
                    result[c][idx] = sums[j]

        return DataFrame(result)


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
