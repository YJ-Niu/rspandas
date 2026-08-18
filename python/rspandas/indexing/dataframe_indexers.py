"""DataFrame 索引器 (Loc/ILoc/Iat/At)

由 rspandas/dataframe.py 拆分而来。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from ..dataframe import DataFrame
    from ..series import Series
from typing import Any, Optional, Union


class _IndexerBase:
    """loc/iloc 索引器基类。"""

    def __init__(self, df: "DataFrame"):
        self._df = df


class _AtIndexer(_IndexerBase):
    """基于标签的标量索引器。"""

    def _find_row_idx(self, row_label) -> int:
        """在索引中查找行标签的位置。"""
        index = self._df._index
        for i, idx in enumerate(index):
            if idx == row_label:
                return i
            if isinstance(row_label, str) and str(idx) == row_label:
                return i

        # 如果是字符串，尝试解析为 datetime
        if isinstance(row_label, str):
            try:
                from .._datetime import to_datetime

                target = to_datetime(row_label)
                for i, idx in enumerate(index):
                    if idx == target:
                        return i
            except Exception:
                pass

        raise KeyError(f"row label {row_label!r} not found")

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("at[] requires a tuple (row_label, col_label)")
        row_label, col_label = key
        if col_label not in self._df._columns:
            raise KeyError(f"column label {col_label!r} not found")
        row_idx = self._find_row_idx(row_label)
        return self._df[col_label].values[row_idx]

    def __setitem__(self, key, value):
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("at[] requires a tuple (row_label, col_label)")
        row_label, col_label = key
        if col_label not in self._df._columns:
            raise KeyError(f"column label {col_label!r} not found")
        # 简化实现：通过重建 DataFrame 修改值
        row_idx = self._find_row_idx(row_label)
        new_data = {c: list(self._df[c].values) for c in self._df._columns}
        new_data[col_label][row_idx] = value
        self._df._reload_inplace(new_data)


class _IatIndexer(_IndexerBase):
    """基于位置的标量索引器。"""

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("iat[] requires a tuple (row_pos, col_pos)")
        row_pos, col_pos = key
        if not (0 <= row_pos < self._df._nrows):
            raise IndexError(f"row position {row_pos} out of range")
        if not (0 <= col_pos < len(self._df._columns)):
            raise IndexError(f"column position {col_pos} out of range")
        col_name = self._df._columns[col_pos]
        return self._df[col_name].values[row_pos]

    def __setitem__(self, key, value):
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("iat[] requires a tuple (row_pos, col_pos)")
        row_pos, col_pos = key
        if not (0 <= row_pos < self._df._nrows):
            raise IndexError(f"row position {row_pos} out of range")
        if not (0 <= col_pos < len(self._df._columns)):
            raise IndexError(f"column position {col_pos} out of range")
        col_name = self._df._columns[col_pos]
        new_data = {c: list(self._df[c].values) for c in self._df._columns}
        new_data[col_name][row_pos] = value
        self._df._reload_inplace(new_data)


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
                result = rows_df[col_key]
                # 单行 + 单列 -> 返回标量
                if isinstance(result, Series) and result.size == 1:
                    return result.values[0]
                return result
            if isinstance(col_key, list):
                return rows_df[col_key]
            raise TypeError(f"loc: unsupported column key {type(col_key).__name__}")

        # 单行无列选择 -> 返回 Series
        if isinstance(rows_df, DataFrame) and rows_df._nrows == 1:
            return rows_df._to_series_row(0)
        return rows_df

    def __setitem__(self, key, value):
        """df.loc[row_key, col_key] = value 支持赋值。"""
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("loc[] requires a tuple (row_key, col_key) for assignment")
        row_key, col_key = key

        # 找到行索引
        row_indices = []
        from datetime import datetime

        if isinstance(row_key, datetime):
            for i, idx in enumerate(self._df._index):
                if idx == row_key:
                    row_indices.append(i)
                    break
        elif isinstance(row_key, str):
            for i, idx in enumerate(self._df._index):
                if str(idx) == row_key:
                    row_indices.append(i)
                    break
            if not row_indices:
                try:
                    from .._datetime import to_datetime

                    target = to_datetime(row_key)
                    for i, idx in enumerate(self._df._index):
                        if idx == target:
                            row_indices.append(i)
                            break
                except Exception:
                    pass
        elif isinstance(row_key, slice):
            # 切片赋值: 处理 datetime/str 起始值
            start, stop, step = row_key.start, row_key.stop, row_key.step
            from datetime import datetime

            # 解析起始位置
            if isinstance(start, datetime):
                for i, idx in enumerate(self._df._index):
                    if idx == start:
                        start = i
                        break
                else:
                    start = 0
            elif isinstance(start, str):
                for i, idx in enumerate(self._df._index):
                    if str(idx) == start:
                        start = i
                        break
                else:
                    try:
                        from .._datetime import to_datetime

                        target = to_datetime(start)
                        for i, idx in enumerate(self._df._index):
                            if idx == target:
                                start = i
                                break
                        else:
                            start = 0
                    except Exception:
                        start = 0
            elif start is None:
                start = 0

            # 解析结束位置
            if isinstance(stop, datetime):
                for i, idx in enumerate(self._df._index):
                    if idx == stop:
                        stop = i
                        break
                else:
                    stop = self._df._nrows - 1
            elif isinstance(stop, str):
                for i, idx in enumerate(self._df._index):
                    if str(idx) == stop:
                        stop = i
                        break
                else:
                    try:
                        from .._datetime import to_datetime

                        target = to_datetime(stop)
                        for i, idx in enumerate(self._df._index):
                            if idx == target:
                                stop = i
                                break
                        else:
                            stop = self._df._nrows - 1
                    except Exception:
                        stop = self._df._nrows - 1
            elif stop is None:
                stop = self._df._nrows - 1

            if isinstance(start, int) and start < 0:
                start += self._df._nrows
            if isinstance(stop, int) and stop < 0:
                stop += self._df._nrows
            if step is None:
                step = 1
            if isinstance(start, int) and isinstance(stop, int):
                row_indices = list(range(start, stop + 1, step))
            else:
                row_indices = []
        elif isinstance(row_key, list):
            row_indices = list(row_key)
        elif isinstance(row_key, int):
            row_indices = [row_key]
        else:
            raise TypeError(f"loc: unsupported row key {type(row_key).__name__}")

        if not row_indices:
            return

        # 处理列赋值
        if isinstance(col_key, str):
            # df.loc[:, "D"] = value
            values = (
                list(value)
                if hasattr(value, "__iter__") and not isinstance(value, str)
                else [value] * len(row_indices)
            )
            new_data = {c: list(self._df[c].values) for c in self._df._columns}
            # 保持原列 dtype（整数赋值给 float 列时保持 float 类型）
            orig_dtype = self._df._inner.get_column(col_key).dtype
            if orig_dtype in ("float64", "float32", "float16", "float"):
                converted = []
                for v in values:
                    if v is None:
                        converted.append(None)
                    elif isinstance(v, bool):
                        converted.append(float(v))
                    elif isinstance(v, int):
                        converted.append(float(v))
                    else:
                        converted.append(v)
                values = converted
            for i, row_idx in enumerate(row_indices):
                if row_idx < len(new_data[col_key]):
                    new_data[col_key][row_idx] = (
                        values[i] if i < len(values) else values[0]
                    )
            self._df._reload_inplace(new_data)
        elif isinstance(col_key, list) and all(isinstance(x, str) for x in col_key):
            # df.loc[:, ["a", "b"]] = value  (对指定列列表赋值，保留原 dtype)
            # pandas 行为：loc 赋值会尽量适配现有 dtype，不改变列的 dtype
            # value 可以是 DataFrame（按位置取列）或标量/列表
            if isinstance(value, DataFrame):
                value_cols = value._columns
                value_cache = value._cache_columns(value_cols)
                new_data = {c: list(self._df[c].values) for c in self._df._columns}
                for i, col_name in enumerate(col_key):
                    if col_name not in new_data:
                        new_data[col_name] = [None] * self._df._nrows
                        self._df._columns.append(col_name)
                        self._df._raw_columns.append(col_name)
                    if i < len(value_cols):
                        src_vals = list(value_cache[value_cols[i]])
                        # 按行索引赋值（row_indices 为空表示全行）
                        if row_indices:
                            for j, row_idx in enumerate(row_indices):
                                if row_idx < len(new_data[col_name]):
                                    new_data[col_name][row_idx] = (
                                        src_vals[j]
                                        if j < len(src_vals)
                                        else src_vals[0]
                                    )
                        else:
                            new_data[col_name] = src_vals
            else:
                # 标量或列表：广播到指定列
                values = (
                    list(value)
                    if hasattr(value, "__iter__") and not isinstance(value, str)
                    else [value] * len(row_indices)
                )
                new_data = {c: list(self._df[c].values) for c in self._df._columns}
                for col_name in col_key:
                    if col_name not in new_data:
                        new_data[col_name] = [None] * self._df._nrows
                        self._df._columns.append(col_name)
                        self._df._raw_columns.append(col_name)
                    for i, row_idx in enumerate(row_indices):
                        if row_idx < len(new_data[col_name]):
                            new_data[col_name][row_idx] = (
                                values[i] if i < len(values) else values[0]
                            )
            self._df._reload_inplace(new_data)
        elif (
            isinstance(col_key, slice)
            and col_key.start is None
            and col_key.stop is None
        ):
            # df.loc[:, col_key] = value  (对所有列赋值)
            values = (
                list(value)
                if hasattr(value, "__iter__") and not isinstance(value, str)
                else [value] * len(row_indices)
            )
            new_data = {c: list(self._df[c].values) for c in self._df._columns}
            for col_name in self._df._columns:
                for i, row_idx in enumerate(row_indices):
                    if row_idx < len(new_data[col_name]):
                        new_data[col_name][row_idx] = (
                            values[i] if i < len(values) else values[0]
                        )
            self._df._reload_inplace(new_data)
        else:
            raise TypeError(
                f"loc: unsupported column key {type(col_key).__name__} for assignment"
            )

    def _select_rows(self, key):
        from datetime import datetime

        # datetime 或 str 键: 在索引中查找位置
        if isinstance(key, datetime):
            for i, idx in enumerate(self._df._index):
                if idx == key:
                    return self._df._select_row(i)
            raise KeyError(f"loc: key {key} not found in index")

        if isinstance(key, str):
            # 尝试直接匹配
            for i, idx in enumerate(self._df._index):
                if str(idx) == key:
                    return self._df._select_row(i)
            # 尝试解析为 datetime
            try:
                from .._datetime import to_datetime

                target = to_datetime(key)
                for i, idx in enumerate(self._df._index):
                    if idx == target:
                        return self._df._select_row(i)
            except Exception:
                pass
            raise KeyError(f"loc: key '{key}' not found in index")

        if isinstance(key, int):
            return self._df._select_row(int(key))

        if isinstance(key, slice):
            # loc 切片: 双闭区间
            start, stop, step = key.start, key.stop, key.step
            if step is None:
                step = 1
            if step <= 0:
                raise ValueError("loc slice step must be positive")
            n = self._df._nrows

            # 处理字符串/日期索引切片
            if (
                isinstance(start, str)
                or isinstance(stop, str)
                or isinstance(start, datetime)
                or isinstance(stop, datetime)
            ):
                from .._datetime import to_datetime

                start_idx = None
                stop_idx = None

                if isinstance(start, (str, datetime)):
                    target = to_datetime(start) if isinstance(start, str) else start
                    for i, idx in enumerate(self._df._index):
                        if idx == target:
                            start_idx = i
                            break
                    if start_idx is None:
                        raise KeyError(f"loc: start key '{start}' not found")

                if isinstance(stop, (str, datetime)):
                    target = to_datetime(stop) if isinstance(stop, str) else stop
                    for i, idx in enumerate(self._df._index):
                        if idx == target:
                            stop_idx = i
                            break
                    if stop_idx is None:
                        raise KeyError(f"loc: stop key '{stop}' not found")

                if start_idx is None:
                    start_idx = 0
                if stop_idx is None:
                    stop_idx = n - 1

                idx = list(range(start_idx, stop_idx + 1, step))
            else:
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
            new_index = [self._df._index[i] for i in idx]
            return DataFrame(new_data, index=new_index)

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
                result = rows_df[cols[col_key]]
                # 单行 + 单列 -> 返回标量
                if isinstance(result, Series) and result.size == 1:
                    return result.values[0]
                return result
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

        # 单行无列选择 -> 返回 Series
        if isinstance(rows_df, DataFrame) and rows_df._nrows == 1:
            return rows_df._to_series_row(0)
        return rows_df

    def __setitem__(self, key, value) -> None:
        """df.iloc[row_key] = value 或 df.iloc[row_key, col_key] = value。

        :param key: 行键，或 (行键, 列键) 元组
        :param value: 标量、列表或 DataFrame
        """
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key = key
            col_key = None

        df = self._df

        # 解析行索引为位置列表
        if isinstance(row_key, int):
            idx = int(row_key)
            if idx < 0:
                idx += df._nrows
            row_indices = [idx]
        elif isinstance(row_key, slice):
            start, stop, step = row_key.indices(df._nrows)
            row_indices = list(range(start, stop, step))
        elif isinstance(row_key, list):
            row_indices = [int(i) + df._nrows if i < 0 else int(i) for i in row_key]
        else:
            raise TypeError(f"iloc: unsupported row key {type(row_key).__name__}")

        # 解析列索引为列名列表
        if col_key is not None:
            cols = df._columns
            if isinstance(col_key, int):
                ci = int(col_key) + len(cols) if col_key < 0 else int(col_key)
                col_names = [cols[ci]]
            elif isinstance(col_key, list):
                if all(isinstance(x, bool) for x in col_key):
                    col_names = [c for c, b in zip(cols, col_key) if b]
                else:
                    col_names = [
                        cols[int(i) + len(cols) if i < 0 else int(i)] for i in col_key
                    ]
            elif isinstance(col_key, slice):
                col_names = list(cols[col_key])
            else:
                raise TypeError(
                    f"iloc: unsupported column key {type(col_key).__name__}"
                )
        else:
            col_names = list(df._columns)

        # 判断 value 类型
        is_scalar = isinstance(value, (int, float, bool)) or value is None

        # 构建新数据并赋值
        new_data = {}
        for c in df._columns:
            vals = list(df._inner.get_column(c).values)
            if c in col_names:
                if is_scalar:
                    # 标量赋值: 所有选中行设为同一值
                    for idx in row_indices:
                        if 0 <= idx < len(vals):
                            vals[idx] = value
                elif hasattr(value, "__iter__") and not isinstance(value, str):
                    # 列表/Series 赋值: 按顺序赋值
                    val_list = list(value)
                    for i, idx in enumerate(row_indices):
                        if i < len(val_list) and 0 <= idx < len(vals):
                            vals[idx] = val_list[i]
            new_data[c] = vals

        df._reload(new_data)


def __getattr__(name: str):
    """模块级延迟导入，避免与 dataframe.py / series.py 的循环导入。"""
    if name == "DataFrame":
        from ..dataframe import DataFrame as _DF

        return _DF
    if name == "Series":
        from ..series import Series as _S

        return _S
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# 懒加载代理：避免循环导入
# ---------------------------------------------------------------------------
import importlib as _importlib


class _LazyProxy:
    """延迟导入代理，支持 __call__（构造）和 __getattr__（属性访问）。"""

    def __init__(self, mod_path: str, attr_name: str):
        self._mod_path = mod_path
        self._attr_name = attr_name
        self._obj = None

    def _load(self):
        if self._obj is None:
            mod = _importlib.import_module(self._mod_path)
            self._obj = getattr(mod, self._attr_name)
        return self._obj

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __instancecheck__(self, instance):
        return isinstance(instance, self._load())


# 安装懒加载代理
if "Series" not in globals() or not callable(globals().get("Series", None)):
    Series = _LazyProxy("rspandas.series", "Series")
if "DataFrame" not in globals() or not callable(globals().get("DataFrame", None)):
    DataFrame = _LazyProxy("rspandas.dataframe", "DataFrame")
