"""DataFrame: pandas-like 2D data structure with Rust backend."""

from __future__ import annotations

import rsnumpy as rnp

from .series import Series
from rspandas.rspandas import _DataFrame as _PyDataFrame  # type: ignore
from rspandas.rspandas import _Series as _PySeries
from rspandas.rspandas import (
    read_csv_path,
    read_csv_string,
)
from typing import Any, Dict, List, Optional, Tuple, Union


def _is_ndarray(data: Any) -> bool:
    """检查对象是否为 rsnumpy ndarray。"""
    return isinstance(data, rnp.ndarray)


def _to_pylist_columns(data: Any, columns: Optional[List[str]]) -> Dict[str, list]:
    """将 dict/list/ndarray 输入解析为 dict[str, list]。"""
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

    if _is_ndarray(data):
        # rsnumpy ndarray: 转换为 list[list] 后按列组织
        raw_list = data.tolist()
        if not isinstance(raw_list, list):
            # 0 维数组
            return {"col0": [raw_list]}
        if not raw_list:
            return {}
        if isinstance(raw_list[0], list):
            # 2D 数组
            if columns is None:
                columns = [f"col{i}" for i in range(len(raw_list[0]))]
            result = {c: [] for c in columns}
            for row in raw_list:
                for i, c in enumerate(columns):
                    result[c].append(row[i] if i < len(row) else None)
            return result
        # 1D 数组
        if columns is None:
            columns = ["col0"]
        result = {c: [] for c in columns}
        for v in raw_list:
            result[columns[0]].append(v)
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
        copy: bool = False,
        fastpath: bool = False,
    ):
        """构造 DataFrame。

        :param data: dict[str, list] | list[dict] | list[list] | ndarray | DataFrame
        :param columns: list[str] | None
        :param index: 行索引
        :param dtype: 数据类型
        :param copy: 是否复制数据
        :param fastpath: 是否走快速路径 (内部使用)
        """
        # 如果输入是 DataFrame，直接复制
        if isinstance(data, DataFrame):
            if columns is None:
                columns = list(data._columns)
            if index is None:
                index = list(data._index) if data._index is not None else None
            col_dict = {}
            for c in columns:
                if c in data._columns:
                    col_dict[c] = list(data._inner.get_column(c).values)
                else:
                    col_dict[c] = [None] * data._nrows
        else:
            col_dict = _to_pylist_columns(data, columns)

        # 如果指定了 columns，按照 columns 顺序重排
        if columns is not None:
            col_dict = {c: col_dict.get(c, []) for c in columns}

        col_names = list(col_dict.keys())
        col_values = [col_dict[c] for c in col_names]

        # 校验每列长度一致
        n = len(col_values[0]) if col_values else 0
        for c, vs in zip(col_names, col_values):
            if len(vs) != n:
                raise ValueError(f"column '{c}' has length {len(vs)} != {n}")

        # 构造 Rust 端 Series
        rust_series_list = []
        for c, vs in zip(col_names, col_values):
            rust_series_list.append(_PySeries(vs, c, dtype=dtype))

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
        new_series = [
            _PySeries(old_data[c], value[i]) for i, c in enumerate(self._columns)
        ]
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

    @property
    def T(self) -> "DataFrame":
        """转置 DataFrame。"""
        return self.transpose()

    @property
    def axes(self) -> list:
        """返回轴标签列表 [index, columns]。"""
        return [self._index, list(self._columns)]

    @property
    def nbytes(self) -> int:
        """返回 DataFrame 占用的字节数。"""
        total = 0
        for c in self._columns:
            ser = self._inner.get_column(c)
            total += ser.nbytes
        return total

    @property
    def style(self):
        """返回 Styler 对象 (占位)。"""
        raise NotImplementedError("Styler not implemented yet")

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
            raise ValueError(f"mask length {len(mask)} != nrows {self._nrows}")
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
            new_data = {
                c: list(self._inner.get_column(c).values) for c in self._columns
            }
            new_data[key] = values
            self._reload(new_data)
        else:
            # 新增列
            new_data = {
                c: list(self._inner.get_column(c).values) for c in self._columns
            }
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

    def sort_values(
        self,
        by,
        axis: int = 0,
        ascending: bool = True,
        inplace: bool = False,
        kind: str = "quicksort",
        na_position: str = "last",
    ) -> "DataFrame":
        """按 by 列排序。

        :param by: 排序列名（str 或 list[str]）
        :param axis: 轴方向（仅支持 0）
        :param ascending: 是否升序
        :param inplace: 是否原地修改
        :param kind: 排序算法（'quicksort'/'mergesort'/'heapsort'）
        :param na_position: NaN 位置（'first'/'last'）
        """
        if isinstance(by, str):
            by = [by]
        for c in by:
            if c not in self._columns:
                raise KeyError(f"column not found: {c}")
        n = self._nrows

        # 取出 by 列用于排序
        sort_keys = [
            [self._inner.get_column(c).values[i] for c in by] for i in range(n)
        ]

        # 根据 na_position 分离 None 行和非 None 行 - 使用列表推导式
        none_indices = [i for i in range(n) if any(v is None for v in sort_keys[i])]
        non_none_indices = [
            i for i in range(n) if all(v is not None for v in sort_keys[i])
        ]

        try:
            non_none_indices.sort(key=lambda i: sort_keys[i], reverse=not ascending)
        except TypeError:
            raise TypeError("cannot sort mixed types")

        if na_position == "first":
            order = none_indices + non_none_indices
        else:  # "last"
            order = non_none_indices + none_indices

        new_data = {
            c: [self._inner.get_column(c).values[i] for i in order]
            for c in self._columns
        }
        new_index = [self._index[i] for i in order]

        if inplace:
            self._reload(new_data)
            self._index = new_index
            return self

        return DataFrame(new_data, index=new_index)

    def filter_rows(self, mask: list) -> "DataFrame":
        if len(mask) != self._nrows:
            raise ValueError(f"mask length {len(mask)} != nrows {self._nrows}")
        # 使用字典推导式替代显式 for 循环
        bool_mask = [bool(x) for x in mask]
        new_data = {
            c: list(self._inner.get_column(c).filter(bool_mask).values)
            for c in self._columns
        }
        return DataFrame(new_data)

    def merge(
        self,
        right: "DataFrame",
        how: str = "inner",
        on=None,
        left_on=None,
        right_on=None,
        left_index: bool = False,
        right_index: bool = False,
        sort: bool = False,
        suffixes=("_x", "_y"),
        copy: bool = True,
        indicator: bool = False,
        validate=None,
    ) -> "DataFrame":
        """连接两个 DataFrame。

        :param right: 另一个 DataFrame
        :param how: 连接方式 ('inner'/'outer'/'left'/'right'/'cross')
        :param on: 连接键列（两侧同名）
        :param left_on: 左侧连接键列
        :param right_on: 右侧连接键列
        :param left_index: 是否使用左侧索引
        :param right_index: 是否使用右侧索引
        :param sort: 是否排序结果
        :param suffixes: 重复列的后缀 (左, 右)
        :param copy: 是否复制数据（始终复制，保持兼容）
        :param indicator: 是否添加 _merge 指示列
        :param validate: 验证方式 ('one_to_one'/'1:1'/'one_to_many'/'1:m'/'many_to_one'/'m:1'/'many_to_many'/'m:m')
        """
        # 确定连接键
        other = right  # 兼容性
        if left_on is not None or right_on is not None:
            if left_on is None:
                left_on = on if isinstance(on, list) else [on] if on else []
            if right_on is None:
                right_on = on if isinstance(on, list) else [on] if on else []
            if isinstance(left_on, str):
                left_keys = [left_on]
            else:
                left_keys = list(left_on)
            if isinstance(right_on, str):
                right_keys = [right_on]
            else:
                right_keys = list(right_on)
        elif on is not None:
            if isinstance(on, str):
                left_keys = [on]
                right_keys = [on]
            else:
                left_keys = list(on)
                right_keys = list(on)
        elif left_index and right_index:
            left_keys = left_index if isinstance(left_index, list) else [left_index]
            right_keys = right_index if isinstance(right_index, list) else [right_index]
        else:
            raise ValueError(
                "on, left_on, right_on, or left_index/right_index must be specified"
            )

        # 使用列表推导式校验键是否存在，替代显式 for 循环
        missing_left = [k for k in left_keys if k not in self._columns]
        if missing_left:
            raise KeyError(f"column {missing_left[0]!r} not in left")
        missing_right = [k for k in right_keys if k not in other._columns]
        if missing_right:
            raise KeyError(f"column {missing_right[0]!r} not in right")

        # 验证 (validate 参数)
        if validate is not None:
            # 简化实现：仅做基本检查
            left_counts: Dict[tuple, int] = {}
            right_counts: Dict[tuple, int] = {}
            for i in range(self._nrows):
                key = tuple(self._inner.get_column(k).values[i] for k in left_keys)
                left_counts[key] = left_counts.get(key, 0) + 1
            for i in range(other._nrows):
                key = tuple(other._inner.get_column(k).values[i] for k in right_keys)
                right_counts[key] = right_counts.get(key, 0) + 1

            if validate in ("one_to_one", "1:1"):
                if any(c > 1 for c in left_counts.values()) or any(
                    c > 1 for c in right_counts.values()
                ):
                    raise ValueError(
                        "Merge keys are not unique in one or both datasets"
                    )
            elif validate in ("one_to_many", "1:m"):
                if any(c > 1 for c in left_counts.values()):
                    raise ValueError("Left merge keys are not unique")
            elif validate in ("many_to_one", "m:1"):
                if any(c > 1 for c in right_counts.values()):
                    raise ValueError("Right merge keys are not unique")
            # many_to_many / m:m 无需验证

        # 构建左侧和右侧的键值对
        left = [
            (
                tuple(self._inner.get_column(k).values[i] for k in left_keys),
                {c: self._inner.get_column(c).values[i] for c in self._columns},
            )
            for i in range(self._nrows)
        ]
        right = [
            (
                tuple(other._inner.get_column(k).values[i] for k in right_keys),
                {c: other._inner.get_column(c).values[i] for c in other._columns},
            )
            for i in range(other._nrows)
        ]
        left_keys_map = {lk: i for i, (lk, _) in enumerate(left)}
        right_keys_map = {rk: i for i, (rk, _) in enumerate(right)}

        merged_rows: List[dict] = []
        merge_indicators: List[str] = []  # for indicator parameter

        if how == "inner":
            common = set(left_keys_map) & set(right_keys_map)
            # 使用列表推导式 + 字典合并替代嵌套 for 循环
            merged_rows = [
                {**left[left_keys_map[k]][1], **right[right_keys_map[k]][1]}
                for k in common
            ]
            if indicator:
                merge_indicators = ["both"] * len(merged_rows)
        elif how == "left":
            right_only = [c for c in other._columns if c not in self._columns]
            right_fill = {c: None for c in right_only}
            # 使用列表推导式替代显式 for 循环
            pairs = [
                (lv, right[right_keys_map[lk]][1] if lk in right_keys_map else None)
                for lk, lv in left
            ]
            merged_rows = [
                {**lv, **(rv if rv is not None else right_fill)} for lv, rv in pairs
            ]
            if indicator:
                merge_indicators = [
                    "both" if rv is not None else "left_only" for _, rv in pairs
                ]
        elif how == "right":
            left_only = [c for c in self._columns if c not in other._columns]
            left_fill = {c: None for c in left_only}
            # 使用列表推导式替代显式 for 循环
            pairs = [
                (left[left_keys_map[rk]][1] if rk in left_keys_map else None, rv)
                for rk, rv in right
            ]
            merged_rows = [
                {**(lv if lv is not None else left_fill), **rv} for lv, rv in pairs
            ]
            if indicator:
                merge_indicators = [
                    "both" if lv is not None else "right_only" for lv, _ in pairs
                ]
        elif how == "outer":
            right_only = [c for c in other._columns if c not in self._columns]
            left_only = [c for c in self._columns if c not in other._columns]
            right_fill = {c: None for c in right_only}
            left_fill = {c: None for c in left_only}
            # 左侧行 + 匹配的右侧行
            left_pairs = [
                (lv, right[right_keys_map[lk]][1] if lk in right_keys_map else None)
                for lk, lv in left
            ]
            seen_l = {lk for lk, _ in left}
            # 仅右侧的行
            right_only_pairs = [
                (left[left_keys_map[rk]][1] if rk in left_keys_map else None, rv)
                for rk, rv in right
                if rk not in seen_l
            ]
            merged_rows = [
                {**lv, **(rv if rv is not None else right_fill)}
                for lv, rv in left_pairs
            ] + [
                {**(lv if lv is not None else left_fill), **rv}
                for lv, rv in right_only_pairs
            ]
            if indicator:
                merge_indicators = [
                    "both" if rv is not None else "left_only" for _, rv in left_pairs
                ] + [
                    "both" if lv is not None else "right_only"
                    for lv, _ in right_only_pairs
                ]
        elif how == "cross":
            # 笛卡尔积 - 使用列表推导式替代嵌套 for 循环
            merged_rows = [{**lv, **rv} for _, lv in left for _, rv in right]
            if indicator:
                merge_indicators = ["both"] * len(merged_rows)
        else:
            raise ValueError(f"unsupported how: {how}")

        # 处理重复列名 (添加后缀)
        all_cols: List[str] = list(self._columns)
        for c in other._columns:
            if c in all_cols and c not in left_keys + right_keys:
                all_cols.remove(c)
                all_cols.append(c + suffixes[0])
                all_cols.append(c + suffixes[1])
            elif c not in all_cols:
                all_cols.append(c)

        # 添加 indicator 列
        if indicator:
            all_cols.append("_merge")

        col_data: Dict[str, list] = {c: [] for c in all_cols}
        # 预计算每个列名对应的基础列名，避免循环内重复字符串切片
        col_base_map = {}
        for c in all_cols:
            if c == "_merge":
                continue
            if c.endswith(suffixes[0]):
                col_base_map[c] = c[: -len(suffixes[0])]
            elif c.endswith(suffixes[1]):
                col_base_map[c] = c[: -len(suffixes[1])]
            else:
                col_base_map[c] = c
        # 使用列表推导式替代嵌套 for 循环
        for c in all_cols:
            if c == "_merge":
                continue
            base_c = col_base_map[c]
            col_data[c] = [row.get(base_c) for row in merged_rows]

        # 添加 merge indicator 数据
        if indicator:
            col_data["_merge"] = merge_indicators

        result_df = DataFrame(col_data)

        # 排序
        if sort:
            result_df = result_df.sort_values(
                by=left_keys if left_keys else self._columns[0]
            )

        return result_df

    @staticmethod
    def concat(frames: List["DataFrame"], axis: int = 0) -> "DataFrame":
        """拼接 DataFrame (v0.4.0)。"""
        if not frames:
            return DataFrame({})
        if axis == 0:
            # 使用 dict.fromkeys 保持顺序去重，替代显式 for 循环
            all_cols: List[str] = list(
                dict.fromkeys(c for f in frames for c in f._columns)
            )
            col_data: Dict[str, list] = {c: [] for c in all_cols}
            for f in frames:
                f_cols = set(f._columns)
                for c in all_cols:
                    if c in f_cols:
                        col_data[c].extend(f._inner.get_column(c).values)
                    else:
                        col_data[c].extend([None] * f._nrows)
            return DataFrame(col_data)
        elif axis == 1:
            nrows = frames[0]._nrows
            if any(f._nrows != nrows for f in frames[1:]):
                raise ValueError("all frames must have the same number of rows")
            # 使用 dict.fromkeys 保持顺序去重，替代显式 for 循环
            all_cols: List[str] = list(
                dict.fromkeys(c for f in frames for c in f._columns)
            )
            col_data: Dict[str, list] = {c: [] for c in all_cols}
            for f in frames:
                for c in f._columns:
                    col_data[c].extend(f._inner.get_column(c).values)
            return DataFrame(col_data)
        else:
            raise ValueError(f"axis must be 0 or 1, got {axis}")

    def dropna(
        self,
        axis: int = 0,
        how: str = "any",
        thresh=None,
        subset=None,
        inplace: bool = False,
    ) -> "DataFrame":
        """删除缺失值。

        :param axis: 0=按行删除, 1=按列删除
        :param how: 'any' (有一个 NaN 就删) 或 'all' (全是 NaN 才删)
        :param thresh: 要求至少 N 个非 NaN 值
        :param subset: 仅考虑指定列
        :param inplace: 是否原地修改
        """
        if axis == 0:
            # 按行删除
            cols_to_check = subset if subset is not None else self._columns
            n = self._nrows

            def _keep_row(row_values):
                """判断单行是否保留。"""
                non_null_count = sum(1 for v in row_values if v is not None)
                if thresh is not None:
                    return non_null_count >= thresh
                elif how == "any":
                    return all(v is not None for v in row_values)
                elif how == "all":
                    return any(v is not None for v in row_values)
                raise ValueError(f"invalid how: {how}")

            # 使用列表推导式替代显式 for 循环（保持原行为：空表时不校验 how）
            keep_mask = [
                _keep_row([self._inner.get_column(c).values[i] for c in cols_to_check])
                for i in range(n)
            ]

            new_data = {
                c: [
                    self._inner.get_column(c).values[i]
                    for i in range(n)
                    if keep_mask[i]
                ]
                for c in self._columns
            }
            new_index = [self._index[i] for i in range(n) if keep_mask[i]]

            if inplace:
                self._reload(new_data)
                self._index = new_index
                self._nrows = len(new_index)
                return self
            return DataFrame(new_data, index=new_index)

        elif axis == 1:
            # 按列删除
            def _keep_col(col_values):
                """判断单列是否保留。"""
                non_null_count = sum(1 for v in col_values if v is not None)
                if thresh is not None:
                    return non_null_count >= thresh
                elif how == "any":
                    return all(v is not None for v in col_values)
                elif how == "all":
                    return any(v is not None for v in col_values)
                raise ValueError(f"invalid how: {how}")

            # 使用列表推导式替代显式 for 循环（保持原行为：空表时不校验 how）
            keep_cols = [
                c
                for c in self._columns
                if _keep_col(list(self._inner.get_column(c).values))
            ]

            new_data = {c: list(self._inner.get_column(c).values) for c in keep_cols}

            if inplace:
                self._reload(new_data)
                self._columns = keep_cols
                return self
            return DataFrame(new_data, index=self._index)

        else:
            raise ValueError(f"axis must be 0 or 1, got {axis}")

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace: bool = False,
        limit=None,
        downcast=None,
    ) -> "DataFrame":
        """填充整个 DataFrame 中所有列的缺失值。

        :param value: 标量 -> 应用到所有列; dict -> 按列名填充不同值
        :param method: 填充方法 ('ffill'/'bfill'/None，已弃用，建议使用 ffill()/bfill())
        :param axis: 轴方向（暂不支持）
        :param inplace: 是否原地修改
        :param limit: 最大填充数量
        :param downcast: 向下转型（暂不支持）
        """
        if method is not None:
            # 使用 ffill/bfill
            if method == "ffill":
                return self.ffill(limit=limit)
            elif method == "bfill":
                return self.bfill(limit=limit)
            else:
                raise ValueError(f"Unsupported method: {method}")

        if isinstance(value, dict):
            # 按列填充 - 使用字典推导式替代显式 for 循环
            new_data: Dict[str, list] = {
                c: (
                    list(self._inner.get_column(c).values)
                    if value.get(c) is None
                    else [
                        value.get(c) if v is None else v
                        for v in self._inner.get_column(c).values
                    ]
                )
                for c in self._columns
            }

            if inplace:
                self._reload(new_data)
                return self
            return DataFrame(new_data)

        # 标量: 对每列单独调用 fillna
        if limit is not None:
            # 限制填充数量 - 保持显式循环（涉及 fill_count 状态）
            new_data: Dict[str, list] = {}
            for c in self._columns:
                ser = self._inner.get_column(c)
                filled_vals = []
                fill_count = 0
                for v in ser.values:
                    if v is None and fill_count < limit:
                        filled_vals.append(value)
                        fill_count += 1
                    else:
                        filled_vals.append(v)
                new_data[c] = filled_vals
        else:
            # 无 limit - 使用字典推导式替代显式 for 循环
            new_data: Dict[str, list] = {
                c: list(self._inner.get_column(c).fillna(value).values)
                for c in self._columns
            }

        if inplace:
            self._reload(new_data)
            return self
        return DataFrame(new_data)

    def agg(self, func=None, axis: int = 0, *args, **kwargs):
        """聚合操作。

        :param func: 聚合函数（str/list/dict/callable）
        :param axis: 轴方向（仅支持 0）
        :param args: 位置参数
        :param kwargs: 关键字参数
        """
        if func is None:
            return self

        # 简化实现：支持字符串和列表
        if isinstance(func, str):
            # 单个聚合函数
            result = {}
            for c in self._columns:
                ser = self._get_column_as_series(c)
                if hasattr(ser, func):
                    result[c] = getattr(ser, func)()
                else:
                    result[c] = None
            return Series(result)
        elif isinstance(func, list):
            # 多个聚合函数
            result_data = {}
            for agg_func in func:
                row_data = {}
                for c in self._columns:
                    ser = self._get_column_as_series(c)
                    if hasattr(ser, agg_func):
                        row_data[c] = getattr(ser, agg_func)()
                    else:
                        row_data[c] = None
                result_data[agg_func] = row_data
            return DataFrame(result_data)
        elif isinstance(func, dict):
            # 每列指定聚合函数
            result = {}
            for c, agg_func in func.items():
                ser = self._get_column_as_series(c)
                if isinstance(agg_func, str):
                    if hasattr(ser, agg_func):
                        result[c] = getattr(ser, agg_func)()
                    else:
                        result[c] = None
                elif callable(agg_func):
                    result[c] = agg_func(ser)
            return Series(result)
        elif callable(func):
            # 可调用对象
            result = {}
            for c in self._columns:
                ser = self._get_column_as_series(c)
                result[c] = func(ser, *args, **kwargs)
            return Series(result)
        else:
            raise TypeError(f"Unsupported func type: {type(func)}")

    aggregate = agg  # 别名

    def apply(self, func, axis: int = 0) -> "Series":
        """应用函数。

        :param axis: 0=按列 (每列传入 Series); 1=按行 (每行传入 dict)
        """
        if axis == 0:
            # 使用列表推导式 + 辅助函数替代显式 for 循环
            def _apply_col(c):
                """对单列应用 func。"""
                res = func(self[c])
                return res._inner if isinstance(res, Series) else res

            results = [_apply_col(c) for c in self._columns]
            return Series(results, index=list(self._columns))
        else:  # axis == 1
            # 使用列表推导式 + 辅助函数替代显式 for 循环
            def _apply_row(i):
                """对单行应用 func。"""
                row = {c: self._inner.get_column(c).values[i] for c in self._columns}
                res = func(row)
                return res._inner if isinstance(res, Series) else res

            results = [_apply_row(i) for i in range(self._nrows)]
            return Series(results, index=list(range(self._nrows)))

    def applymap(self, func) -> "DataFrame":
        """对每个元素应用 func。"""
        # 使用字典推导式替代显式 for 循环
        new_data: Dict[str, list] = {
            c: [None if v is None else func(v) for v in self[c].values]
            for c in self._columns
        }
        return DataFrame(new_data)

    def map(self, func) -> "DataFrame":
        """applymap 的别名 (pandas 2.1+ 推荐)。"""
        return self.applymap(func)

    def abs(self) -> "DataFrame":
        """返回绝对值的 DataFrame。"""
        # 使用字典推导式替代显式 for 循环
        new_data: Dict[str, list] = {
            c: [None if v is None else abs(v) for v in self[c].values]
            for c in self._columns
        }
        return DataFrame(new_data)

    def copy(self, deep: bool = True) -> "DataFrame":
        """复制 DataFrame。

        :param deep: True=深拷贝, False=浅拷贝
        """
        if deep:
            new_data = {
                c: list(self._inner.get_column(c).values) for c in self._columns
            }
            new_index = list(self._index) if self._index is not None else None
            return DataFrame(new_data, index=new_index)
        return DataFrame._from_inner(self._inner)

    def isna(self) -> "DataFrame":
        """返回布尔 DataFrame，True 表示该位置是 None。"""
        # 使用字典推导式替代显式 for 循环
        new_data: Dict[str, list] = {
            c: [v is None for v in self[c].values] for c in self._columns
        }
        return DataFrame(new_data)

    def notna(self) -> "DataFrame":
        """返回布尔 DataFrame，True 表示该位置不是 None。"""
        # 使用字典推导式替代显式 for 循环
        new_data: Dict[str, list] = {
            c: [v is not None for v in self[c].values] for c in self._columns
        }
        return DataFrame(new_data)

    def isnull(self) -> "DataFrame":
        """isna 的别名。"""
        return self.isna()

    def notnull(self) -> "DataFrame":
        """notna 的别名。"""
        return self.notna()

    def nlargest(self, n: int = 5, columns=None, keep: str = "first") -> "DataFrame":
        """返回最大的 N 行。

        :param n: 返回的行数
        :param columns: 用于排序的列 (str 或 list[str])
        :param keep: 重复值的保留方式
        """
        if columns is None:
            columns = self._columns[0] if self._columns else []
        if isinstance(columns, str):
            columns = [columns]

        # 按指定列排序 - 使用列表推导式替代显式 for 循环
        missing = [c for c in columns if c not in self._columns]
        if missing:
            raise KeyError(f"column not found: {missing[0]}")
        sort_keys = [
            [self._inner.get_column(c).values[i] for i in range(self._nrows)]
            for c in columns
        ]

        def key_func(i):
            return tuple(
                (1 if sort_keys[j][i] is None else 0, sort_keys[j][i])
                for j in range(len(columns))
            )

        order = sorted(range(self._nrows), key=key_func, reverse=True)[:n]
        new_data = {
            c: [self._inner.get_column(c).values[i] for i in order]
            for c in self._columns
        }
        return DataFrame(new_data)

    def nsmallest(self, n: int = 5, columns=None, keep: str = "first") -> "DataFrame":
        """返回最小的 N 行。

        :param n: 返回的行数
        :param columns: 用于排序的列 (str 或 list[str])
        :param keep: 重复值的保留方式
        """
        if columns is None:
            columns = self._columns[0] if self._columns else []
        if isinstance(columns, str):
            columns = [columns]

        # 使用列表推导式替代显式 for 循环
        missing = [c for c in columns if c not in self._columns]
        if missing:
            raise KeyError(f"column not found: {missing[0]}")
        sort_keys = [
            [self._inner.get_column(c).values[i] for i in range(self._nrows)]
            for c in columns
        ]

        def key_func(i):
            return tuple(
                (1 if sort_keys[j][i] is None else 0, sort_keys[j][i])
                for j in range(len(columns))
            )

        order = sorted(range(self._nrows), key=key_func)[:n]
        new_data = {
            c: [self._inner.get_column(c).values[i] for i in order]
            for c in self._columns
        }
        return DataFrame(new_data)

    def corr(self, method: str = "pearson", min_periods: int = 1) -> "DataFrame":
        """计算列之间的相关系数矩阵。

        :param method: 相关系数方法 ('pearson'/'kendall'/'spearman')
        :param min_periods: 最少非空值数
        """
        numeric_cols = [
            c
            for c in self._columns
            if self._inner.get_column(c).dtype in ("int64", "float64")
        ]
        n = len(numeric_cols)
        # 预取所有数值列的值，避免循环内重复取值
        col_values = {c: list(self._inner.get_column(c).values) for c in numeric_cols}

        def _pearson(pairs):
            """计算 pearson 相关系数。"""
            ma = sum(a for a, _ in pairs) / len(pairs)
            mb = sum(b for _, b in pairs) / len(pairs)
            num = sum((a - ma) * (b - mb) for a, b in pairs)
            da = (sum((a - ma) ** 2 for a, _ in pairs)) ** 0.5
            db = (sum((b - mb) ** 2 for _, b in pairs)) ** 0.5
            if da == 0 or db == 0:
                return None
            return num / (da * db)

        def _spearman(pairs):
            """计算 spearman 秩相关系数。"""
            ranks_i = sorted(range(len(pairs)), key=lambda k: pairs[k][0])
            ranks_j = sorted(range(len(pairs)), key=lambda k: pairs[k][1])
            rank_i = {k: i for i, k in enumerate(ranks_i)}
            rank_j = {k: i for i, k in enumerate(ranks_j)}
            d2 = sum((rank_i[k] - rank_j[k]) ** 2 for k in range(len(pairs)))
            n_pairs = len(pairs)
            return 1 - (6 * d2) / (n_pairs * (n_pairs**2 - 1))

        def _kendall(pairs):
            """计算 kendall tau 相关系数。"""
            n_pairs = len(pairs)
            total = n_pairs * (n_pairs - 1) / 2
            if total == 0:
                return None
            # 使用 sum + 生成器表达式替代嵌套 for 循环
            concordant = sum(
                1
                for k1 in range(n_pairs)
                for k2 in range(k1 + 1, n_pairs)
                if (pairs[k1][0] < pairs[k2][0] and pairs[k1][1] < pairs[k2][1])
                or (pairs[k1][0] > pairs[k2][0] and pairs[k1][1] > pairs[k2][1])
            )
            discordant = sum(
                1
                for k1 in range(n_pairs)
                for k2 in range(k1 + 1, n_pairs)
                if (pairs[k1][0] < pairs[k2][0] and pairs[k1][1] > pairs[k2][1])
                or (pairs[k1][0] > pairs[k2][0] and pairs[k1][1] < pairs[k2][1])
            )
            return (concordant - discordant) / total

        def _calc(a_vals, b_vals):
            """计算两列之间的相关系数。"""
            pairs = [
                (a, b)
                for a, b in zip(a_vals, b_vals)
                if a is not None and b is not None
            ]
            if len(pairs) < min_periods or len(pairs) < 2:
                return None
            if method == "pearson":
                return _pearson(pairs)
            elif method == "spearman":
                return _spearman(pairs)
            elif method == "kendall":
                return _kendall(pairs)
            raise ValueError(f"Unsupported method: {method}")

        # 使用字典推导式 + 列表推导式替代嵌套 for 循环
        corr_data: Dict[str, list] = {
            numeric_cols[i]: [
                (
                    1.0
                    if i == j
                    else _calc(col_values[numeric_cols[i]], col_values[numeric_cols[j]])
                )
                for j in range(n)
            ]
            for i in range(n)
        }

        return DataFrame(corr_data)

    def cov(self, min_periods: int = 1, ddof: int = 1) -> "DataFrame":
        """计算列之间的协方差矩阵。

        :param min_periods: 最少非空值数
        :param ddof: 自由度修正值
        """
        numeric_cols = [
            c
            for c in self._columns
            if self._inner.get_column(c).dtype in ("int64", "float64")
        ]
        n = len(numeric_cols)
        # 预取所有数值列的值，避免循环内重复取值
        col_values = {c: list(self._inner.get_column(c).values) for c in numeric_cols}

        def _cov(a_vals, b_vals):
            """计算两列之间的协方差。"""
            pairs = [
                (a, b)
                for a, b in zip(a_vals, b_vals)
                if a is not None and b is not None
            ]
            if len(pairs) < min_periods or len(pairs) < 2:
                return None
            ma = sum(a for a, _ in pairs) / len(pairs)
            mb = sum(b for _, b in pairs) / len(pairs)
            return sum((a - ma) * (b - mb) for a, b in pairs) / (len(pairs) - ddof)

        # 使用字典推导式 + 列表推导式替代嵌套 for 循环
        cov_data: Dict[str, list] = {
            numeric_cols[i]: [
                _cov(col_values[numeric_cols[i]], col_values[numeric_cols[j]])
                for j in range(n)
            ]
            for i in range(n)
        }

        return DataFrame(cov_data)

    def corrwith(self, other, axis: int = 0, drop: bool = False) -> Series:
        """计算与另一个 DataFrame/Series 的相关系数。

        :param other: 另一个 DataFrame 或 Series
        :param axis: 0=按列计算
        :param drop: 是否丢弃缺失值
        """

        def _pearson(pairs):
            """计算 pearson 相关系数。"""
            if len(pairs) < 2:
                return None
            ma = sum(a for a, _ in pairs) / len(pairs)
            mb = sum(b for _, b in pairs) / len(pairs)
            num = sum((a - ma) * (b - mb) for a, b in pairs)
            da = (sum((a - ma) ** 2 for a, _ in pairs)) ** 0.5
            db = (sum((b - mb) ** 2 for _, b in pairs)) ** 0.5
            if da == 0 or db == 0:
                return None
            return num / (da * db)

        def _pairs(a_vals, b_vals):
            """构造非空值对。"""
            return [
                (a, b)
                for a, b in zip(a_vals, b_vals)
                if a is not None and b is not None
            ]

        if isinstance(other, DataFrame):
            # 使用字典推导式替代显式 for 循环
            results = {
                c: (
                    _pearson(
                        _pairs(
                            self._inner.get_column(c).values,
                            other._inner.get_column(c).values,
                        )
                    )
                    if c in other._columns
                    else (None if not drop else None)
                )
                for c in self._columns
            }
            # drop=False 时保留不在 other 中的列（值为 None）
            if drop:
                results = {c: v for c, v in results.items() if v is not None}
            return Series(list(results.values()), index=list(results.keys()))
        elif isinstance(other, Series):
            other_vals = other.values
            # 使用字典推导式替代显式 for 循环
            results = {
                c: _pearson(_pairs(self._inner.get_column(c).values, other_vals))
                for c in self._columns
            }
            return Series(list(results.values()), index=list(results.keys()))
        else:
            raise TypeError("other must be DataFrame or Series")

    def sort_index(
        self,
        axis: int = 0,
        level=None,
        ascending: bool = True,
        inplace: bool = False,
        kind: str = "quicksort",
        na_position: str = "last",
        sort_remaining: bool = True,
    ) -> "DataFrame":
        """按索引排序。

        :param axis: 0=按行索引, 1=按列名
        :param level: 多级索引层级（暂不支持）
        :param ascending: 是否升序
        :param inplace: 是否原地修改
        :param kind: 排序算法
        :param na_position: NaN 位置（'first'/'last'）
        :param sort_remaining: 是否对剩余级别排序（暂不支持）
        """
        if axis == 0:
            if self._index is None:
                return self.copy()

            # 处理 NaN 索引
            indexed = [(v, i) for i, v in enumerate(self._index)]

            if na_position == "first":
                indexed.sort(
                    key=lambda x: (
                        0 if x[0] is None else 1,
                        x[0] if x[0] is not None else "",
                    ),
                    reverse=not ascending,
                )
            else:  # "last"
                indexed.sort(
                    key=lambda x: (
                        1 if x[0] is None else 0,
                        x[0] if x[0] is not None else "",
                    ),
                    reverse=not ascending,
                )

            order = [i for _, i in indexed]
            new_data = {
                c: [self._inner.get_column(c).values[i] for i in order]
                for c in self._columns
            }
            new_index = [self._index[i] for i in order]

            if inplace:
                self._reload(new_data)
                self._index = new_index
                return self
            return DataFrame(new_data, index=new_index)
        elif axis == 1:
            # 按列名排序
            new_cols = sorted(self._columns, reverse=not ascending)
            new_data = {c: list(self._inner.get_column(c).values) for c in new_cols}
            if inplace:
                self._reload(new_data)
                self._columns = new_cols
                return self
            return DataFrame(new_data, index=self._index)
        else:
            raise ValueError(f"axis must be 0 or 1, got {axis}")

    def reindex(self, index=None, columns=None, **kwargs) -> "DataFrame":
        """重新索引。"""
        if index is None and columns is None:
            return self.copy()

        if columns is not None:
            if not isinstance(columns, list):
                columns = list(columns)
            new_cols = columns
        else:
            new_cols = self._columns

        if index is not None:
            if not isinstance(index, list):
                index = list(index)
            # 使用字典推导式替代显式 for 循环
            old_index_map = {
                idx: i for i, idx in enumerate(self._index or range(self._nrows))
            }
            new_order = [old_index_map.get(label) for label in index]
        else:
            new_order = list(range(self._nrows))
            index = self._index or list(range(self._nrows))

        # 使用字典推导式替代显式 for 循环
        new_data = {
            c: (
                [
                    self._inner.get_column(c).values[i] if i is not None else None
                    for i in new_order
                ]
                if c in self._columns
                else [None] * len(new_order)
            )
            for c in new_cols
        }

        return DataFrame(new_data, index=index)

    # ---------- 高级操作 (v1.0.0) ----------

    def assign(self, **kwargs) -> "DataFrame":
        """添加新列 (链式调用友好)。"""
        new_data = {c: list(self._inner.get_column(c).values) for c in self._columns}

        def _resolve(value):
            """解析列值为列表。"""
            if isinstance(value, Series):
                return list(value.values)
            try:
                iter(value)
                return list(value)
            except TypeError:
                return [value] * self._nrows

        # 使用字典推导式 + 辅助函数替代显式 for 循环
        new_data.update({name: _resolve(value) for name, value in kwargs.items()})
        return DataFrame(new_data)

    def eval(self, expr: str, inplace: bool = False):
        """用字符串表达式计算。

        :param expr: 表达式字符串
        :param inplace: 是否原地修改（仅对赋值表达式有效）
        """
        local_vars = {c: self._inner.get_column(c).values for c in self._columns}
        result = eval(expr, {}, local_vars)

        # 如果是赋值表达式，更新列
        if inplace and "=" in expr:
            # 简化实现：解析简单赋值如 "new_col = col1 + col2"
            parts = expr.split("=")
            if len(parts) == 2:
                new_col = parts[0].strip()
                if isinstance(result, Series):
                    self[new_col] = result
                else:
                    self[new_col] = [result] * self._nrows
                return self

        return result

    def query(self, expr: str, inplace: bool = False) -> "DataFrame":
        """用字符串表达式过滤行。

        :param expr: 表达式字符串
        :param inplace: 是否原地修改
        """
        # 使用列表推导式替代显式 for 循环
        mask = [
            bool(
                eval(
                    expr,
                    {},
                    {c: self._inner.get_column(c).values[i] for c in self._columns},
                )
            )
            for i in range(self._nrows)
        ]

        new_data = {
            c: [
                self._inner.get_column(c).values[i]
                for i in range(self._nrows)
                if mask[i]
            ]
            for c in self._columns
        }
        new_index = [self._index[i] for i in range(self._nrows) if mask[i]]

        if inplace:
            self._reload(new_data)
            self._index = new_index
            self._nrows = len(new_index)
            return self

        return DataFrame(new_data, index=new_index)

    def pipe(self, func, *args, **kwargs):
        """管道方法: df.pipe(func, ...) == func(df, ...)。"""
        return func(self, *args, **kwargs)

    def transform(self, func, axis: int = 0, *args, **kwargs) -> "DataFrame":
        """对每列应用 func 并返回相同形状的 DataFrame。

        :param func: 变换函数（str/callable/list）
        :param axis: 轴方向（仅支持 0）
        :param args: 位置参数
        :param kwargs: 关键字参数
        """
        new_data: Dict[str, list] = {}

        if isinstance(func, list):
            # 多个函数：生成多列
            for f in func:
                for c in self._columns:
                    ser = self[c]
                    if isinstance(f, str):
                        if hasattr(ser, f):
                            result = getattr(ser, f)(*args, **kwargs)
                        else:
                            result = f
                    else:
                        result = f(ser, *args, **kwargs)

                    if isinstance(result, Series):
                        new_data[f"{c}_{f}"] = list(result.values)
                    else:
                        new_data[f"{c}_{f}"] = [result] * self._nrows
        else:
            # 单个函数
            for c in self._columns:
                ser = self[c]
                if isinstance(func, str):
                    if hasattr(ser, func):
                        result = getattr(ser, func)(*args, **kwargs)
                    else:
                        result = func
                else:
                    result = func(ser, *args, **kwargs)

                if isinstance(result, Series):
                    new_data[c] = list(result.values)
                else:
                    new_data[c] = [result] * self._nrows

        return DataFrame(new_data)

    def replace(
        self,
        to_replace=None,
        value=None,
        inplace: bool = False,
        limit=None,
        regex: bool = False,
        method: str = "pad",
    ) -> "DataFrame":
        """替换 DataFrame 中的值。

        :param to_replace: 要替换的值（str/number/list/dict/None）
        :param value: 替换后的值
        :param inplace: 是否原地修改
        :param limit: 最大替换数量
        :param regex: 是否正则表达式
        :param method: 填充方法（已弃用）
        """
        new_data: Dict[str, list] = {}

        for c in self._columns:
            ser = self[c]
            col_vals = list(ser.values)
            replaced_vals = []
            replace_count = 0

            for v in col_vals:
                if limit is not None and replace_count >= limit:
                    replaced_vals.append(v)
                    continue

                if regex and isinstance(to_replace, str) and isinstance(v, str):
                    import re

                    if re.search(to_replace, v):
                        replaced_vals.append(value)
                        replace_count += 1
                    else:
                        replaced_vals.append(v)
                elif isinstance(to_replace, dict):
                    # 字典映射
                    if v in to_replace:
                        replaced_vals.append(to_replace[v])
                        replace_count += 1
                    else:
                        replaced_vals.append(v)
                elif v == to_replace:
                    replaced_vals.append(value)
                    replace_count += 1
                else:
                    replaced_vals.append(v)

            new_data[c] = replaced_vals

        if inplace:
            self._reload(new_data)
            return self
        return DataFrame(new_data)

    def duplicated(self, subset=None, keep: str = "first") -> "Series":
        """标记重复行。"""
        if subset is None:
            subset = self._columns
        elif isinstance(subset, str):
            subset = [subset]
        # 取每行的 key tuple - 使用列表推导式替代显式 for 循环
        n = self._nrows
        row_keys = [
            tuple(self._inner.get_column(c).values[i] for c in subset) for i in range(n)
        ]
        seen: set = set()
        if keep == "first":
            # 使用 seen.add 副作用技巧替代显式 for 循环
            mark = [k in seen or (seen.add(k), False)[1] for k in row_keys]
        elif keep == "last":
            # 反向遍历检测重复，再反转结果
            rev_mark = [
                k in seen or (seen.add(k), False)[1] for k in reversed(row_keys)
            ]
            mark = list(reversed(rev_mark))
        elif keep is False:
            from collections import Counter

            c = Counter(row_keys)
            dup = {k for k, n in c.items() if n > 1}
            mark = [k in dup for k in row_keys]
        return Series(mark, name=None, index=list(range(n)))

    def drop_duplicates(
        self, subset=None, keep: str = "first", inplace: bool = False
    ) -> "DataFrame":
        """删除重复行。"""
        if subset is None:
            subset = self._columns
        elif isinstance(subset, str):
            subset = [subset]
        n = self._nrows
        row_keys = [
            tuple(self._inner.get_column(c).values[i] for c in subset) for i in range(n)
        ]
        seen: set = set()
        if keep == "first":
            # 使用 seen.add 副作用技巧替代显式 for 循环
            keep_idx = [
                i for i, k in enumerate(row_keys) if k not in seen and not seen.add(k)
            ]
        elif keep == "last":
            # 反向遍历保留最后一次出现
            rev_keep = [
                i
                for i, k in reversed(list(enumerate(row_keys)))
                if k not in seen and not seen.add(k)
            ]
            keep_idx = list(reversed(rev_keep))
        else:
            # keep=False: 丢弃所有重复行（仅保留唯一值）
            from collections import Counter

            counts = Counter(row_keys)
            keep_idx = [i for i, k in enumerate(row_keys) if counts[k] == 1]
        # 使用字典推导式替代显式 for 循环
        new_data: Dict[str, list] = {
            c: [self._inner.get_column(c).values[i] for i in keep_idx]
            for c in self._columns
        }
        return DataFrame(new_data)

    def nunique(self) -> "Series":
        """每列不同值数量。"""
        # 使用字典推导式替代显式 for 循环
        out = {c: self[c].nunique() for c in self._columns}
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
        data = {
            c: [
                None if pd.isna(v) else v.item() if hasattr(v, "item") else v
                for v in pdf[c].values
            ]
            for c in pdf.columns
        }
        return cls(data)

    def to_numpy(self, dtype=None):
        """转换为 rsnumpy 二维数组。

        :param dtype: 目标 dtype
        """
        cols = list(self._columns)
        data = [
            [self._inner.get_column(c).values[i] for c in cols]
            for i in range(self._nrows)
        ]
        return rnp.array(data, dtype=dtype) if dtype else rnp.array(data)

    @classmethod
    def from_numpy(cls, arr, columns=None, index=None, dtype=None) -> "DataFrame":
        """从 rsnumpy 二维数组构造 DataFrame。

        Parameters
        ----------
        arr : rsnumpy.ndarray
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
        if not isinstance(arr, rnp.ndarray):
            raise TypeError(f"expected rsnumpy.ndarray, got {type(arr).__name__}")
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
                arrays.append(
                    pa.array([str(v) if v is not None else None for v in col_data])
                )
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

    def drop(
        self,
        labels=None,
        axis: int = 0,
        index=None,
        columns=None,
        level=None,
        inplace: bool = False,
        errors: str = "raise",
    ) -> "DataFrame":
        """删除行或列。

        :param labels: 要删除的标签 (str/int 或 list)
        :param axis: 0=行, 1=列
        :param index: 要删除的行标签（替代 labels + axis=0）
        :param columns: 要删除的列名（替代 labels + axis=1）
        :param level: 多级索引层级（暂不支持）
        :param inplace: 是否原地修改
        :param errors: 'raise' (找不到时抛错) 或 'ignore' (静默忽略)
        """
        # 处理 index / columns 参数优先级
        if index is not None:
            labels = index
            axis = 0
        elif columns is not None:
            labels = columns
            axis = 1

        if labels is None:
            raise ValueError("labels, index, or columns must be specified")

        if not isinstance(labels, (list, tuple)):
            labels = [labels]

        if axis == 0:
            # 删除行 - 使用列表推导式替代显式 for 循环
            label_set = set(labels)
            keep_idx = [
                i
                for i in range(self._nrows)
                if (self._index[i] if self._index else i) not in label_set
            ]

            if errors == "raise":
                missing = [
                    l_ for l_ in labels if l_ not in (self._index or range(len(self)))
                ]
                if missing:
                    raise KeyError(f"labels {missing} not found in axis")

            new_data = {
                c: [self._inner.get_column(c).values[i] for i in keep_idx]
                for c in self._columns
            }
            new_index = [self._index[i] for i in keep_idx] if self._index else None
            result = DataFrame(new_data, index=new_index)
        else:
            # 删除列
            if errors == "raise":
                missing = [l_ for l_ in labels if l_ not in self._columns]
                if missing:
                    raise KeyError(f"labels {missing} not found in axis")
            new_cols = [c for c in self._columns if c not in labels]
            new_data = {c: list(self._inner.get_column(c).values) for c in new_cols}
            result = DataFrame(new_data, index=self._index)

        if inplace:
            self._reload(new_data)
            if axis == 0:
                self._index = new_index if self._index else list(range(self._nrows))
            else:
                self._columns = new_cols
            return self
        return result

    def rename(
        self,
        mapper=None,
        index=None,
        columns=None,
        axis=None,
        copy: bool = True,
        inplace: bool = False,
        level=None,
        errors: str = "ignore",
    ) -> "DataFrame":
        """重命名行或列。

        :param mapper: dict {old_name: new_name} 或 callable
        :param index: 行索引重命名映射（dict 或 callable）
        :param columns: 列名重命名映射（dict 或 callable）
        :param axis: 指定 mapper 应用的轴（0=行, 1=列）
        :param copy: 是否复制数据
        :param inplace: 是否原地修改
        :param level: 多级索引层级（暂不支持）
        :param errors: 'raise' 或 'ignore'
        """
        # 处理 axis 与 mapper 的兼容性
        if axis is not None:
            if axis == 1 or axis == "columns":
                columns = mapper
            else:
                index = mapper
        elif columns is not None:
            pass
        elif index is not None:
            pass
        elif mapper is not None:
            # 默认行为：若 mapper 为 dict 且无 axis，尝试推断
            index = mapper

        new_data = {c: list(self._inner.get_column(c).values) for c in self._columns}
        new_index = list(self._index) if self._index else None
        new_cols = list(self._columns)

        # 重命名列
        if columns is not None:
            if isinstance(columns, dict):
                new_cols = [columns.get(c, c) for c in self._columns]
            elif callable(columns):
                new_cols = [columns(c) for c in self._columns]
            else:
                raise TypeError("columns must be dict or callable")
            # 处理重复列名（简单去重）
            seen = set()
            for i, c in enumerate(new_cols):
                if c in seen:
                    new_cols[i] = f"{c}.{i}"
                seen.add(c)
            new_data = {new_cols[i]: new_data[c] for i, c in enumerate(self._columns)}

        # 重命名索引
        if index is not None:
            if self._index is not None:
                if isinstance(index, dict):
                    new_index = [index.get(label, label) for label in self._index]
                elif callable(index):
                    new_index = [index(label) for label in self._index]
                else:
                    raise TypeError("index must be dict or callable")
            else:
                new_index = list(range(self._nrows))

        result = DataFrame(new_data, index=new_index)

        if inplace:
            self._reload(new_data)
            self._index = new_index
            self._columns = new_cols
            return self
        return result

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
                new_index.append(
                    tuple(self._inner.get_column(k).values[i] for k in keys)
                )
        new_data = {
            c: list(self._inner.get_column(c).values)
            for c in self._columns
            if c not in keys
        }
        df = DataFrame(new_data)
        df._index = new_index
        return df

    def reset_index(
        self,
        level=None,
        drop: bool = False,
        inplace: bool = False,
        col_level: int = 0,
        col_fill: str = "",
    ) -> "DataFrame":
        """重置索引为默认 RangeIndex。

        :param level: 要重置的索引级别（暂不支持多级索引级别筛选）
        :param drop: 是否丢弃索引列而不是插入到 DataFrame 中
        :param inplace: 是否原地修改
        :param col_level: 列多级索引级别（暂不支持）
        :param col_fill: 列多级索引填充值（暂不支持）
        """
        new_data = {c: list(self._inner.get_column(c).values) for c in self._columns}
        new_columns = list(self._columns)

        if not drop and self._index is not None:
            index_name = "index"
            if hasattr(self._index, "name") and self._index.name is not None:
                index_name = self._index.name
            elif isinstance(self._index, list) and len(self._index) > 0:
                pass

            # 处理多级索引（tuple 索引）
            if self._index and isinstance(self._index[0], tuple):
                nlevels = len(self._index[0])
                if level is not None:
                    levels_to_reset = [level] if isinstance(level, int) else list(level)
                else:
                    levels_to_reset = list(range(nlevels))

                for lvl in sorted(levels_to_reset, reverse=True):
                    lvl_name = f"level_{lvl}"
                    new_data[lvl_name] = [idx[lvl] for idx in self._index]
                    new_columns.insert(0, lvl_name)
            else:
                new_data[index_name] = list(self._index)
                new_columns.insert(0, index_name)

        result = DataFrame(new_data)
        result._columns = new_columns
        result._index = list(range(self._nrows))

        if inplace:
            self._reload(new_data)
            self._columns = new_columns
            self._index = list(range(self._nrows))
            return self
        return result

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
        path_or_buf=None,
        sep: str = ",",
        na_rep: str = "",
        float_format=None,
        columns=None,
        header: Union[bool, List[str]] = True,
        index: bool = True,
        index_label=None,
        mode: str = "w",
        encoding: Optional[str] = None,
        compression: str = "infer",
        quoting=None,
        quotechar: str = '"',
        lineterminator=None,
        chunksize=None,
        date_format=None,
        doublequote: bool = True,
        escapechar=None,
        decimal: str = ".",
        errors: str = "strict",
    ) -> Optional[str]:
        """写入 CSV。

        :param path_or_buf: 文件路径或缓冲区；为 None 时返回字符串
        :param sep: 分隔符
        :param na_rep: 缺失值表示字符串
        :param float_format: 浮点数格式化字符串
        :param columns: 要写入的列名列表
        :param header: 是否写入表头，或自定义列名列表
        :param index: 是否写入索引列
        :param index_label: 索引列的列名
        :param mode: 文件打开模式 ('w'/'a')
        :param encoding: 文件编码
        :param compression: 压缩格式 ('infer'/'gzip'/'bz2'/'zip'/'xz'/None)
        :param quoting: 引用方式
        :param quotechar: 引用字符
        :param lineterminator: 行终止符
        :param chunksize: 每次写入的行数
        :param date_format: 日期格式字符串
        :param doublequote: 是否双引号转义
        :param escapechar: 转义字符
        :param decimal: 小数点字符
        :param errors: 编码错误处理方式
        :return: 如果 path_or_buf 为 None，返回 CSV 字符串
        """
        # 兼容 path 参数
        path = path_or_buf

        # 选择列
        if columns is None:
            selected_columns = list(self._columns)
        else:
            selected_columns = list(columns)

        # 构建数据
        series_list = [self._inner.get_column(c) for c in selected_columns]

        # 处理索引列
        if index and self._index:
            from .rspandas import _Series as _PySeries

            idx_series = _PySeries(self._index, index_label if index_label else "")
            columns_to_write = [index_label if index_label else ""] + selected_columns
            series_to_write = [idx_series] + series_list
        else:
            columns_to_write = selected_columns
            series_to_write = series_list

        # 生成 CSV 内容（带参数处理）
        content_lines = []

        # 表头
        if header:
            if isinstance(header, list):
                # 使用自定义列名
                content_lines.append(sep.join([str(h) for h in header]))
            else:
                content_lines.append(sep.join([str(c) for c in columns_to_write]))

        # 数据行 - 使用辅助函数 + 列表推导式替代嵌套 for 循环
        def _format_value(val):
            """格式化单个值为 CSV 字段。"""
            if val is None:
                return na_rep
            if float_format is not None and isinstance(val, float):
                return float_format % val
            if date_format is not None and hasattr(val, "strftime"):
                return val.strftime(date_format)
            if decimal != "." and isinstance(val, float):
                return str(val).replace(".", decimal)
            return str(val)

        # 预取所有列的值列表，避免循环内重复访问
        cols_values = [list(ser.values) for ser in series_to_write]
        n_rows = self._nrows
        content_lines.extend(
            sep.join(_format_value(col_vals[i]) for col_vals in cols_values)
            for i in range(n_rows)
        )

        content = "\n".join(content_lines)
        if lineterminator is not None:
            content = lineterminator.join(content_lines)

        if path is None:
            return content

        # 处理压缩
        import os

        if compression == "infer":
            if isinstance(path, str) and path.endswith(".gz"):
                compression = "gzip"
            elif isinstance(path, str) and path.endswith(".bz2"):
                compression = "bz2"
            else:
                compression = None

        # 写入文件（支持追加模式）
        actual_header = header
        if isinstance(path, str):
            if mode == "a" and os.path.exists(path) and os.path.getsize(path) > 0:
                # 追加模式：去掉表头
                lines = content.split("\n")
                if actual_header and len(lines) > 1:
                    content = "\n".join(lines[1:])
                if not content.endswith("\n"):
                    content += "\n"

            if compression == "gzip":
                import gzip

                with gzip.open(path, mode + "t", encoding=encoding, errors=errors) as f:
                    f.write(content)
            elif compression == "bz2":
                import bz2

                with bz2.open(path, mode + "t", encoding=encoding, errors=errors) as f:
                    f.write(content)
            else:
                with open(path, mode, encoding=encoding, errors=errors) as f:
                    f.write(content)
        else:
            # path_or_buf 是文件对象
            path.write(content)

        return None

    def to_dict(self, orient: str = "dict", into=dict) -> dict:
        """转换为字典。

        :param orient: 方向格式 ('dict'/'list'/'records'/'index'/'columns'/'split'/'tight')
        :param into: 用于构建结果的映射类型 (默认 dict)
        """
        # 预取所有列的值和索引，避免循环内重复访问
        cols_values = {c: list(self._inner.get_column(c).values) for c in self._columns}
        idx_list = (
            list(self._index) if self._index is not None else list(range(self._nrows))
        )

        if orient == "dict":
            # 使用嵌套字典推导式替代显式 for 循环
            return into(
                (
                    c,
                    into((idx_list[i], col_vals[i]) for i in range(self._nrows)),
                )
                for c, col_vals in cols_values.items()
            )
        elif orient == "list":
            # 使用字典推导式替代显式 for 循环
            return into((c, list(col_vals)) for c, col_vals in cols_values.items())
        elif orient == "records":
            # 使用列表推导式替代显式 for 循环
            return [
                into((c, cols_values[c][i]) for c in self._columns)
                for i in range(self._nrows)
            ]
        elif orient == "index":
            # 使用字典推导式替代显式 for 循环
            return into(
                (
                    idx_list[i],
                    into((c, cols_values[c][i]) for c in self._columns),
                )
                for i in range(self._nrows)
            )
        elif orient == "columns":
            # 与 'dict' 方向相同（按列存储，键为索引）
            return into(
                (
                    c,
                    into((idx_list[i], col_vals[i]) for i in range(self._nrows)),
                )
                for c, col_vals in cols_values.items()
            )
        elif orient == "split":
            return into(
                index=idx_list,
                columns=list(self._columns),
                data=[
                    [cols_values[c][i] for c in self._columns]
                    for i in range(self._nrows)
                ],
            )
        elif orient == "tight":
            return into(
                index=idx_list,
                columns=list(self._columns),
                data=[
                    [cols_values[c][i] for c in self._columns]
                    for i in range(self._nrows)
                ],
                data_headers=list(self._columns),
                index_names=[None],
            )
        else:
            raise ValueError(f"Unsupported orient: {orient}")

    # ---------- IO 扩展 (v1.2.0) ----------

    def to_json(
        self,
        path_or_buf=None,
        orient: Optional[str] = None,
        date_format: str = "iso",
        double_precision: int = 10,
        force_ascii: bool = True,
        date_unit: str = "ms",
        default_handler=None,
        lines: bool = False,
        compression: str = "infer",
        index: bool = True,
        indent: Optional[int] = None,
    ) -> Optional[str]:
        """将 DataFrame 写入 JSON 文件或返回 JSON 字符串。

        :param path_or_buf: 文件路径或缓冲区；为 None 时返回字符串
        :param orient: JSON 格式方向 ('split'/'records'/'index'/'columns'/'values'/'table')
        :param date_format: 日期格式 ('iso'/'epoch')
        :param double_precision: 浮点数精度位数
        :param force_ascii: 是否强制 ASCII 编码
        :param date_unit: 日期单位 ('s'/'ms'/'us'/'ns')
        :param default_handler: 无法序列化对象的处理函数
        :param lines: 是否按行输出 JSON (仅 orient='records')
        :param compression: 压缩格式 ('infer'/'gzip'/'bz2'/'xz'/None)
        :param index: 是否包含索引 (仅 orient='split'/'table')
        :param indent: 缩进空格数
        :return: 如果 path_or_buf 为 None，返回 JSON 字符串
        """
        from .io import to_json as _to_json

        return _to_json(
            self,
            path_or_buf,
            orient=orient,
            date_format=date_format,
            double_precision=double_precision,
            force_ascii=force_ascii,
            date_unit=date_unit,
            default_handler=default_handler,
            lines=lines,
            compression=compression,
            index=index,
            indent=indent,
        )

    def to_excel(
        self,
        path,
        sheet_name: str = "Sheet1",
        index: bool = False,
        header: bool = True,
        **kwargs,
    ) -> None:
        """将 DataFrame 写入 Excel 文件。

        :param path: 输出文件路径或 ExcelWriter 对象
        :param sheet_name: 工作表名称
        :param index: 是否写入行索引
        :param header: 是否写入列名
        """
        from .io import ExcelWriter
        from .io import to_excel as _to_excel

        if isinstance(path, ExcelWriter):
            path.write(self, sheet_name=sheet_name, index=index, header=header)
        else:
            _to_excel(
                self, path, sheet_name=sheet_name, index=index, header=header, **kwargs
            )

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

    def to_sql(
        self,
        name: str,
        conn,
        if_exists: str = "fail",
        index: bool = False,
        **kwargs,
    ) -> None:
        """将 DataFrame 写入 SQL 数据库 (v1.2.0)。

        :param name: 目标表名
        :param conn: 数据库连接
        :param if_exists: 'fail' / 'replace' / 'append'
        :param index: 是否写入行索引
        """
        from .io import to_sql as _to_sql

        _to_sql(self, name, conn, if_exists=if_exists, index=index, **kwargs)

    # ---------- 索引器辅助 ----------

    def _select_row(self, idx: int) -> "DataFrame":
        if idx < 0:
            idx += self._nrows
        if idx < 0 or idx >= self._nrows:
            raise IndexError("single positional indexer is out-of-bounds")
        new_data = {c: [self._inner.get_column(c).values[idx]] for c in self._columns}
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
            c: [self._inner.get_column(c).values[i] for i in idx] for c in self._columns
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

    def describe(self, percentiles=None, include=None, exclude=None) -> "DataFrame":
        """对数值列做统计。

        :param percentiles: 分位数列表（默认 [0.25, 0.5, 0.75]）
        :param include: 包含的列类型（None/'all'/类型列表）
        :param exclude: 排除的列类型
        :return: DataFrame
        """
        # 默认分位数
        if percentiles is None:
            percentiles = [0.25, 0.5, 0.75]
        else:
            percentiles = list(percentiles)

        # 确定要分析的列
        if include == "all":
            cols_to_analyze = self._columns
        elif include is not None:
            if isinstance(include, str):
                include = [include]
            # 使用列表推导式 + 辅助函数替代显式 for 循环
            include_strs = [str(t) for t in include]

            def _include_match(c):
                """判断列 dtype 是否匹配 include 条件。"""
                dt = self._inner.get_column(c).dtype
                return dt in include or dt in include_strs

            cols_to_analyze = [c for c in self._columns if _include_match(c)]
        else:
            # 默认仅数值列
            cols_to_analyze = [
                c
                for c in self._columns
                if self._inner.get_column(c).dtype in ("int64", "float64")
            ]

        # 排除列
        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]
            cols_to_analyze = [
                c
                for c in cols_to_analyze
                if self._inner.get_column(c).dtype not in exclude
                and self._inner.get_column(c).dtype not in [str(t) for t in exclude]
            ]

        # 构建统计指标
        stat_names = (
            ["count", "mean", "std", "min"]
            + [f"{int(p*100)}%" for p in percentiles]
            + ["max"]
        )

        # 预取列数据，避免循环内重复访问
        cols_data = {
            c: [
                v
                for v in self._inner.get_column(c).values
                if v is not None and isinstance(v, (int, float))
            ]
            for c in cols_to_analyze
        }

        def _col_stats(vals):
            """计算单列的统计指标。"""
            if not vals:
                return {
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    **{f"{int(p*100)}%": None for p in percentiles},
                }
            mean_val = sum(vals) / len(vals)
            std_val = (
                (sum((v - mean_val) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                if len(vals) > 1
                else None
            )
            sorted_vals = sorted(vals)
            quantiles = {}
            for p in percentiles:
                pos = p * (len(sorted_vals) - 1)
                lo = int(pos)
                hi = min(lo + 1, len(sorted_vals) - 1)
                frac = pos - lo
                quantiles[f"{int(p*100)}%"] = (
                    sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac
                )
            return {
                "count": len(vals),
                "mean": mean_val,
                "std": std_val,
                "min": min(vals),
                "max": max(vals),
                **quantiles,
            }

        # 使用列表推导式 + 辅助函数替代显式 for 循环
        all_stats = [_col_stats(cols_data[c]) for c in cols_to_analyze]

        out: Dict[str, list] = {s: [stat[s] for stat in all_stats] for s in stat_names}

        # 添加列名作为第一列
        out[""] = cols_to_analyze

        return DataFrame(out)

    # ---------- 统计方法 ----------

    def sum(
        self, axis=None, skipna=True, level=None, numeric_only=None, min_count=0
    ) -> "Series":
        """按列求和。

        :param axis: 轴方向（None/0=按列）
        :param skipna: 是否跳过 NaN
        :param level: 多级索引层级（暂不支持）
        :param numeric_only: 是否仅计算数值列
        :param min_count: 最少非空值数
        """
        # 确定要计算的列 - 使用列表推导式
        target_cols = [
            c
            for c in self._columns
            if numeric_only is None
            or not numeric_only
            or self._inner.get_column(c).dtype in ("int64", "float64")
        ]

        def _sum_col(c):
            """计算单列的和。"""
            vals = list(self._inner.get_column(c).values)
            non_null = [v for v in vals if v is not None] if skipna else vals
            if min_count > 0 and len(non_null) < min_count:
                return None
            return sum(v for v in non_null if isinstance(v, (int, float)))

        # 使用字典推导式替代显式 for 循环
        return Series({c: _sum_col(c) for c in target_cols})

    def mean(self, axis=None, skipna=True, level=None, numeric_only=None) -> "Series":
        """按列求均值。

        :param axis: 轴方向（None/0=按列）
        :param skipna: 是否跳过 NaN
        :param level: 多级索引层级（暂不支持）
        :param numeric_only: 是否仅计算数值列
        """
        target_cols = [
            c
            for c in self._columns
            if numeric_only is None
            or not numeric_only
            or self._inner.get_column(c).dtype in ("int64", "float64")
        ]
        # 使用字典推导式替代显式 for 循环
        return Series({c: self._get_column_as_series(c).mean() for c in target_cols})

    def min(self, axis=None, skipna=True, level=None, numeric_only=None) -> "Series":
        """按列求最小值。

        :param axis: 轴方向（None/0=按列）
        :param skipna: 是否跳过 NaN
        :param level: 多级索引层级（暂不支持）
        :param numeric_only: 是否仅计算数值列
        """
        target_cols = [
            c
            for c in self._columns
            if numeric_only is None
            or not numeric_only
            or self._inner.get_column(c).dtype in ("int64", "float64")
        ]
        # 使用字典推导式替代显式 for 循环
        return Series({c: self._get_column_as_series(c).min() for c in target_cols})

    def max(self, axis=None, skipna=True, level=None, numeric_only=None) -> "Series":
        """按列求最大值。

        :param axis: 轴方向（None/0=按列）
        :param skipna: 是否跳过 NaN
        :param level: 多级索引层级（暂不支持）
        :param numeric_only: 是否仅计算数值列
        """
        target_cols = [
            c
            for c in self._columns
            if numeric_only is None
            or not numeric_only
            or self._inner.get_column(c).dtype in ("int64", "float64")
        ]
        # 使用字典推导式替代显式 for 循环
        return Series({c: self._get_column_as_series(c).max() for c in target_cols})

    def count(self, axis: int = 0) -> "Series":
        """按列计数 (非空值)。"""
        # 使用字典推导式替代显式 for 循环
        return Series({c: self._get_column_as_series(c).count() for c in self._columns})

    def std(
        self, axis=None, skipna=True, level=None, ddof: int = 1, numeric_only=None
    ) -> "Series":
        """按列求标准差。

        :param axis: 轴方向（None/0=按列）
        :param skipna: 是否跳过 NaN
        :param level: 多级索引层级（暂不支持）
        :param ddof: 自由度修正值
        :param numeric_only: 是否仅计算数值列
        """
        target_cols = [
            c
            for c in self._columns
            if numeric_only is None
            or not numeric_only
            or self._inner.get_column(c).dtype in ("int64", "float64")
        ]

        def _std_col(c):
            """计算单列的标准差。"""
            vals = [
                v
                for v in self._inner.get_column(c).values
                if v is not None and isinstance(v, (int, float))
            ]
            if len(vals) < 2:
                return None
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / (len(vals) - ddof)
            return var**0.5

        # 使用字典推导式替代显式 for 循环
        return Series({c: _std_col(c) for c in target_cols})

    def var(
        self, axis=None, skipna=True, level=None, ddof: int = 1, numeric_only=None
    ) -> "Series":
        """按列求方差。

        :param axis: 轴方向（None/0=按列）
        :param skipna: 是否跳过 NaN
        :param level: 多级索引层级（暂不支持）
        :param ddof: 自由度修正值
        :param numeric_only: 是否仅计算数值列
        """
        target_cols = [
            c
            for c in self._columns
            if numeric_only is None
            or not numeric_only
            or self._inner.get_column(c).dtype in ("int64", "float64")
        ]

        def _var_col(c):
            """计算单列的方差。"""
            vals = [
                v
                for v in self._inner.get_column(c).values
                if v is not None and isinstance(v, (int, float))
            ]
            if len(vals) < 2:
                return None
            m = sum(vals) / len(vals)
            return sum((v - m) ** 2 for v in vals) / (len(vals) - ddof)

        # 使用字典推导式替代显式 for 循环
        return Series({c: _var_col(c) for c in target_cols})

    def median(self, axis=None, skipna=True, level=None, numeric_only=None) -> "Series":
        """按列求中位数。

        :param axis: 轴方向（None/0=按列）
        :param skipna: 是否跳过 NaN
        :param level: 多级索引层级（暂不支持）
        :param numeric_only: 是否仅计算数值列
        """
        target_cols = [
            c
            for c in self._columns
            if numeric_only is None
            or not numeric_only
            or self._inner.get_column(c).dtype in ("int64", "float64")
        ]
        # 使用字典推导式替代显式 for 循环
        return Series({c: self._get_column_as_series(c).median() for c in target_cols})

    def any(self, axis: int = 0, skipna: bool = True) -> "Series":
        """按列判断是否有真值。"""
        # 使用字典推导式替代显式 for 循环
        return Series({c: self._get_column_as_series(c).any() for c in self._columns})

    def all(self, axis: int = 0, skipna: bool = True) -> "Series":
        """按列判断是否全为真值。"""
        # 使用字典推导式替代显式 for 循环
        return Series({c: self._get_column_as_series(c).all() for c in self._columns})

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

    def groupby(
        self,
        by=None,
        axis: int = 0,
        level=None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        squeeze: bool = False,
        observed: bool = False,
        dropna: bool = True,
    ):
        """按 by 列分组。

        :param by: 分组列名（str/list/None）
        :param axis: 轴方向（仅支持 0）
        :param level: 多级索引层级（暂不支持）
        :param as_index: 是否将分组键作为索引
        :param sort: 是否对组键排序
        :param group_keys: 是否在结果中添加组键（暂不支持）
        :param squeeze: 是否压缩维度（已弃用）
        :param observed: 是否仅使用观察到的分类值（暂不支持）
        :param dropna: 是否删除 NaN 组
        """
        return DataFrameGroupBy(self, by, as_index=as_index, sort=sort, dropna=dropna)

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
            tuple(self._inner.get_column(c).values[i] for c in index) for i in range(n)
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
        margins: bool = False,
        dropna: bool = True,
        margins_name: str = "All",
    ) -> "DataFrame":
        """创建透视表。

        :param values: 聚合的列 (str | list[str] | None -> 所有数值列)
        :param index: 行分组列 (str | list[str])
        :param columns: 列分组列 (str | list[str])
        :param aggfunc: 聚合函数 ('sum' / 'mean' / 'count' / 'min' / 'max' / 'median' / 'std')
        :param fill_value: 用于替换缺失值的标量
        :param margins: 是否添加边际汇总
        :param dropna: 是否删除 NaN 列
        :param margins_name: 边际列名
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
                c
                for c in self._columns
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
            col_cols = []
        elif isinstance(columns, str):
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
                col_name = (
                    "_".join(col_name_parts)
                    if len(col_name_parts) > 1
                    else col_name_parts[0]
                )
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
                            col_data.append(
                                sum(nums) / len(nums) if nums else fill_value
                            )
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
                                col_data.append(
                                    (nums[len(nums) // 2 - 1] + nums[len(nums) // 2])
                                    / 2
                                )
                        elif aggfunc == "std":
                            nums = [x for x in vals if x is not None]
                            if len(nums) < 2:
                                col_data.append(fill_value)
                            else:
                                m = sum(nums) / len(nums)
                                var = sum((x - m) ** 2 for x in nums) / len(nums)
                                col_data.append(var**0.5)
                        else:
                            raise ValueError(f"unsupported aggfunc: {aggfunc}")
                new_data[col_name] = col_data

        return DataFrame(new_data)

    def stack(self, level: int = -1) -> "DataFrame":
        """将列堆叠为行 (v1.0.0)。"""
        # 简化版: 仅支持单层
        n = self._nrows
        idx_name = self._index_name or "index"
        # 使用列表推导式替代嵌套显式 for 循环
        pairs = [
            (i, c, self._inner.get_column(c).values[i])
            for i in range(n)
            for c in self._columns
        ]
        new_data: Dict[str, list] = {
            idx_name: [p[0] for p in pairs],
            "variable": [p[1] for p in pairs],
            "value": [p[2] for p in pairs],
        }
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
            raise ValueError("Can only compare identically-labeled DataFrame objects")

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
        # 使用 all() + 生成器表达式替代显式 for 循环（保持短路求值）
        return all(
            list(self._inner.get_column(c).values)
            == list(other._inner.get_column(c).values)
            for c in self._columns
        )

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
                indices = [
                    i for i, idx in enumerate(self._index) if pat.search(str(idx))
                ]
            else:
                return self.copy()
            new_data = {
                c: [self._inner.get_column(c).values[i] for i in indices]
                for c in self._columns
            }
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

        def _match(c):
            """判断单列是否匹配 include/exclude 条件。"""
            dt = self._inner.get_column(c).dtype
            if include_set and dt not in include_set:
                return False
            if exclude_set and dt in exclude_set:
                return False
            return True

        # 使用列表推导式 + 辅助函数替代显式 for 循环
        cols = [c for c in self._columns if _match(c)]

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
        # 使用字典推导式替代显式 for 循环：每行变成一列
        new_data: Dict[str, list] = {
            str(i): [self._inner.get_column(c).values[i] for c in self._columns]
            for i in range(n)
        }
        return DataFrame(new_data)

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
            "D": "days",
            "H": "hours",
            "M": "minutes",
            "S": "seconds",
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
                    i
                    for i in range(self._nrows)
                    if isinstance(self._index[i], datetime) and self._index[i] <= end
                ]
            else:
                end = max(times)
                start = end - timedelta(**{unit_map[unit]: num - 1})
                indices = [
                    i
                    for i in range(self._nrows)
                    if isinstance(self._index[i], datetime) and self._index[i] >= start
                ]
        else:
            # 数值索引
            if mode == "first":
                start = min(times)
                end = start + num
                indices = [
                    i
                    for i in range(self._nrows)
                    if isinstance(self._index[i], (int, float))
                    and self._index[i] <= end
                ]
            else:
                end = max(times)
                start = end - num
                indices = [
                    i
                    for i in range(self._nrows)
                    if isinstance(self._index[i], (int, float))
                    and self._index[i] >= start
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
                    i
                    for i in indices
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
                    i
                    for i in indices
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
                    cols = cols[: idx + 1]
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
            "D": ("days", 1),
            "H": ("hours", 1),
            "h": ("hours", 1),
            "M": ("minutes", 0),
            "T": ("minutes", 1),
            "min": ("minutes", 1),
            "S": ("seconds", 1),
            "W": ("weeks", 1),
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
                    raise TypeError(
                        f"asfreq requires a DatetimeIndex, got {type(v).__name__}"
                    )
            else:
                raise TypeError(
                    f"asfreq requires a DatetimeIndex, got {type(v).__name__}"
                )

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

    def tz_localize(
        self, tz, axis: int = 0, level=None, copy: bool = True
    ) -> "DataFrame":
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
            if self._index and any(
                isinstance(v, datetime) and v.tzinfo is not None for v in self._index
            ):
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

            new_data = {
                c: list(self._inner.get_column(c).values) for c in self._columns
            }
            df = DataFrame(new_data)
            df._index = new_index
            return df
        else:
            # axis == 1: 列索引 (不常用)
            return self.copy() if copy else self

    def tz_convert(
        self, tz, axis: int = 0, level=None, copy: bool = True
    ) -> "DataFrame":
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
            has_tz = any(
                isinstance(v, datetime) and v.tzinfo is not None for v in self._index
            )
            if not has_tz:
                raise TypeError("Index is not tz-aware. Use tz_localize first.")

            tzinfo = self._parse_timezone(tz)

            # 转换时区 - 使用列表推导式替代显式 for 循环
            new_index = [
                (
                    v.astimezone(tzinfo)
                    if isinstance(v, datetime) and v.tzinfo is not None
                    else v
                )
                for v in self._index
            ]

            new_data = {
                c: list(self._inner.get_column(c).values) for c in self._columns
            }
            df = DataFrame(new_data)
            df._index = new_index
            return df
        else:
            return self.copy() if copy else self

    @staticmethod
    def _parse_timezone(tz: str):
        """解析时区字符串为 tzinfo 对象。"""
        from datetime import timedelta as td
        from datetime import timezone

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
        from datetime import datetime
        from datetime import time as dt_time

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
            # 复用 Series.cumsum 的优化实现，避免重复循环
            new_data = {
                c: list(self[c].cumsum(skipna=skipna).values) for c in self._columns
            }
            return DataFrame(new_data)
        else:
            return self.T.cumsum(axis=0).T

    def cumprod(self, axis: int = 0, skipna: bool = True) -> "DataFrame":
        """返回每列的累计积。

        :param axis: 0=列方向, 1=行方向
        :param skipna: 是否跳过 NaN
        """
        if axis == 0:
            new_data = {
                c: list(self[c].cumprod(skipna=skipna).values) for c in self._columns
            }
            return DataFrame(new_data)
        else:
            return self.T.cumprod(axis=0).T

    def cummax(self, axis: int = 0, skipna: bool = True) -> "DataFrame":
        """返回每列的累计最大值。

        :param axis: 0=列方向, 1=行方向
        :param skipna: 是否跳过 NaN
        """
        if axis == 0:
            new_data = {
                c: list(self[c].cummax(skipna=skipna).values) for c in self._columns
            }
            return DataFrame(new_data)
        else:
            return self.T.cummax(axis=0).T

    def cummin(self, axis: int = 0, skipna: bool = True) -> "DataFrame":
        """返回每列的累计最小值。

        :param axis: 0=列方向, 1=行方向
        :param skipna: 是否跳过 NaN
        """
        if axis == 0:
            new_data = {
                c: list(self[c].cummin(skipna=skipna).values) for c in self._columns
            }
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
                    shifted = ([None] * periods) + vals[: len(vals) - periods]
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
            # 复用 Series.diff 的优化实现（切片 + 列表推导式）
            new_data = {c: list(self[c].diff(periods).values) for c in self._columns}
            return DataFrame(new_data)
        else:
            return self.T.diff(periods, axis=0).T

    def pct_change(self, periods: int = 1) -> "DataFrame":
        """计算每列的百分比变化。

        :param periods: 差分步数
        """
        new_data = {c: list(self[c].pct_change(periods).values) for c in self._columns}
        return DataFrame(new_data)

    # ---------- v2.0.0: 统计方法 ----------

    def rank(
        self, axis: int = 0, method: str = "average", ascending: bool = True
    ) -> "DataFrame":
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
                        if (
                            j == len(indexed)
                            or indexed[j][0] != indexed[group_start][0]
                        ):
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

            def _interp(qv, vals):
                """线性插值计算单个分位数。"""
                pos = qv * (len(vals) - 1)
                lo = int(pos)
                hi = min(lo + 1, len(vals) - 1)
                return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)

            def _quantiles(vals):
                """计算 vals 在 q_list 各分位数的值。"""
                if not vals:
                    return [None] * len(q_list)
                vals.sort()
                # 使用列表推导式替代内层显式 for 循环
                return [_interp(qv, vals) for qv in q_list]

            # 使用字典推导式替代外层显式 for 循环
            new_data = {
                c: _quantiles(
                    [v for v in self._inner.get_column(c).values if v is not None]
                )
                for c in self._columns
            }
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
                    result[c] = (n**0.5 * m3) / (m2**1.5)
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
                    result[c] = (n * (n + 1) * m4) / (
                        (n - 1) * (n - 2) * (n - 3) * m2**2
                    ) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
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
                        best_idx = (
                            self._index[i]
                            if self._index and i < len(self._index)
                            else i
                        )
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
                        best_idx = (
                            self._index[i]
                            if self._index and i < len(self._index)
                            else i
                        )
                result[c] = best_idx
            return Series(result)
        else:
            return self.T.idxmin(axis=0, skipna=skipna)

    # ---------- v2.0.0: 排序 ----------

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

            def _clip_val(v):
                """单值裁剪：None 保持，否则按上下界裁剪。"""
                if v is None:
                    return None
                result = v
                if lower is not None and result < lower:
                    result = lower
                if upper is not None and result > upper:
                    result = upper
                return result

            # 使用字典推导式 + 列表推导式替代嵌套显式 for 循环
            new_data = {
                c: [_clip_val(v) for v in self._inner.get_column(c).values]
                for c in self._columns
            }
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
        from .series import Series

        import sys

        # 使用字典推导式 + sum() 替代嵌套显式 for 循环
        result = {
            c: sum(
                sys.getsizeof(v) if deep and isinstance(v, str) else 8
                for v in self._inner.get_column(c).values
            )
            for c in self._columns
        }
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

    def droplevel(self, level, axis: int = 0) -> "DataFrame":
        """删除索引级别。

        :param level: 要删除的级别 (int 或 str)
        :param axis: 0=索引, 1=列
        """
        return self.copy()

    def swaplevel(self, i: int = -2, j: int = -1, axis: int = 0) -> "DataFrame":
        """交换多级索引的级别。"""
        return self.copy()

    def join(
        self,
        other,
        on=None,
        how: str = "left",
        lsuffix: str = "",
        rsuffix: str = "",
        sort: bool = False,
    ) -> "DataFrame":
        """连接另一个 DataFrame。

        :param other: 另一个 DataFrame
        :param on: 连接键
        :param how: 连接方式 ('left'/'right'/'outer'/'inner')
        :param lsuffix: 左表列后缀
        :param rsuffix: 右表列后缀
        :param sort: 是否排序
        """
        if not isinstance(other, DataFrame):
            raise TypeError("other must be DataFrame")

        # 简化实现：按列拼接
        left_data = {c: list(self._inner.get_column(c).values) for c in self._columns}
        right_data = {
            c: list(other._inner.get_column(c).values) for c in other._columns
        }

        # 处理重复列名
        for c in list(left_data.keys()):
            if c in right_data:
                left_data[c + lsuffix] = left_data.pop(c)
                right_data[c + rsuffix] = right_data.pop(c)

        combined = {}
        combined.update(left_data)
        combined.update(right_data)

        # 按 index 对齐 - 使用字典推导式替代显式 for 循环
        n = max(len(self), len(other))
        result_data = {
            c: (vals[:n] if len(vals) >= n else vals + [None] * (n - len(vals)))
            for c, vals in combined.items()
        }

        return DataFrame(result_data)

    def itertuples(self, index: bool = True, name: str = "Pandas") -> list:
        """迭代行，返回 namedtuple。

        :param index: 是否包含索引
        :param name: namedtuple 名称
        """
        from collections import namedtuple

        fields = []
        if index:
            fields.append("index")
        fields.extend([str(c) for c in self._columns])

        TupleClass = namedtuple(name, fields, rename=True)
        # 使用列表推导式替代显式 for 循环
        return [
            TupleClass(
                *(
                    (
                        [self._index[i] if self._index and i < len(self._index) else i]
                        if index
                        else []
                    )
                    + [self._inner.get_column(c).values[i] for c in self._columns]
                )
            )
            for i in range(self._nrows)
        ]

    def to_records(self, index: bool = True) -> list:
        """转换为记录数组。

        :param index: 是否包含索引
        """
        # 使用列表推导式替代显式 for 循环
        return [
            {
                **(
                    {
                        "index": (
                            self._index[i]
                            if self._index and i < len(self._index)
                            else i
                        )
                    }
                    if index
                    else {}
                ),
                **{c: self._inner.get_column(c).values[i] for c in self._columns},
            }
            for i in range(self._nrows)
        ]

    def to_string(self, index: bool = True, header: bool = True) -> str:
        """转换为字符串表示。

        :param index: 是否显示索引
        :param header: 是否显示列名
        """
        lines = []
        if header:
            cols = [""] + list(self._columns) if index else list(self._columns)
            lines.append("  ".join(str(c) for c in cols))
        for i in range(self._nrows):
            row = []
            if index:
                row.append(
                    str(self._index[i] if self._index and i < len(self._index) else i)
                )
            for c in self._columns:
                v = self._inner.get_column(c).values[i]
                row.append(str(v) if v is not None else "NaN")
            lines.append("  ".join(row))
        return "\n".join(lines)

    def to_html(self, index: bool = True, header: bool = True) -> str:
        """转换为 HTML 表格。

        :param index: 是否包含索引
        :param header: 是否包含表头
        """
        parts = ["<table>"]
        if header:
            parts.append("  <thead>")
            parts.append("    <tr>")
            if index:
                parts.append("      <th></th>")
            for c in self._columns:
                parts.append(f"      <th>{c}</th>")
            parts.append("    </tr>")
            parts.append("  </thead>")
        parts.append("  <tbody>")
        for i in range(self._nrows):
            parts.append("    <tr>")
            if index:
                idx_val = self._index[i] if self._index and i < len(self._index) else i
                parts.append(f"      <td>{idx_val}</td>")
            for c in self._columns:
                v = self._inner.get_column(c).values[i]
                val_str = str(v) if v is not None else "NaN"
                parts.append(f"      <td>{val_str}</td>")
            parts.append("    </tr>")
        parts.append("  </tbody>")
        parts.append("</table>")
        return "\n".join(parts)

    def to_latex(self, index: bool = True, header: bool = True) -> str:
        """转换为 LaTeX 表格。

        :param index: 是否包含索引
        :param header: 是否包含表头
        """
        ncols = len(self._columns) + (1 if index else 0)
        lines = [f"\\begin{{tabular}}{{{'l' * ncols}}}"]
        lines.append("\\hline")
        if header:
            cols = []
            if index:
                cols.append("")
            cols.extend([str(c) for c in self._columns])
            lines.append(" & ".join(cols) + " \\\\")
            lines.append("\\hline")
        for i in range(self._nrows):
            row = []
            if index:
                idx_val = self._index[i] if self._index and i < len(self._index) else i
                row.append(str(idx_val))
            for c in self._columns:
                v = self._inner.get_column(c).values[i]
                row.append(str(v) if v is not None else "NaN")
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        return "\n".join(lines)

    def to_markdown(self, index: bool = True, header: bool = True) -> str:
        """转换为 Markdown 表格。

        :param index: 是否包含索引
        :param header: 是否包含表头
        """
        lines = []
        if header:
            cols = [""] + list(self._columns) if index else list(self._columns)
            lines.append("| " + " | ".join(str(c) for c in cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for i in range(self._nrows):
            row = []
            if index:
                idx_val = self._index[i] if self._index and i < len(self._index) else i
                row.append(str(idx_val))
            for c in self._columns:
                v = self._inner.get_column(c).values[i]
                row.append(str(v) if v is not None else "NaN")
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def sem(self, axis: int = 0, skipna: bool = True) -> "Series":
        """返回平均值的标准误差。

        :param axis: 0=逐行, 1=逐列
        :param skipna: 是否跳过 NaN
        """
        if axis == 0:
            result = {}
            for c in self._columns:
                ser = self._get_column_as_series(c)
                result[c] = ser.sem()
            return Series(result)
        else:
            # 逐列的 sem
            return self.sem(axis=0).T

    # ---------- 填充 / 插值 ----------

    def ffill(self, limit=None) -> "DataFrame":
        """前向填充缺失值。

        :param limit: 最大连续填充数量
        """
        # 复用 Series.ffill 的实现，避免重复代码
        new_data = {c: list(self[c].ffill(limit=limit).values) for c in self._columns}
        return DataFrame(new_data, index=self._index)

    def bfill(self, limit=None) -> "DataFrame":
        """后向填充缺失值。

        :param limit: 最大连续填充数量
        """
        new_data = {c: list(self[c].bfill(limit=limit).values) for c in self._columns}
        return DataFrame(new_data, index=self._index)

    def pad(self, limit=None) -> "DataFrame":
        """pad 的别名，等价于 ffill。"""
        return self.ffill(limit=limit)

    def backfill(self, limit=None) -> "DataFrame":
        """backfill 的别名，等价于 bfill。"""
        return self.bfill(limit=limit)

    def interpolate(
        self,
        method: str = "linear",
        axis: int = 0,
        limit=None,
        inplace: bool = False,
        limit_direction: str = "forward",
        limit_area=None,
        downcast=None,
        **kwargs,
    ) -> "DataFrame":
        """插值填充缺失值。

        :param method: 插值方法 ('linear'/'pad'/'index'/'nearest')
        :param axis: 轴
        :param limit: 最大连续填充数量
        :param inplace: 是否原地修改
        """
        if method in ("pad", "ffill"):
            return self.ffill(limit=limit)
        if method in ("backfill", "bfill"):
            return self.bfill(limit=limit)

        # 复用 Series.interpolate 的实现
        new_data = {
            c: list(self[c].interpolate(method=method, limit=limit).values)
            for c in self._columns
        }

        result_df = DataFrame(new_data, index=self._index)
        if inplace:
            self._reload(new_data)
            return self
        return result_df

    # ---------- 聚合扩展 ----------

    def prod(
        self, axis: int = 0, skipna: bool = True, numeric_only: bool = None
    ) -> "Series":
        """返回每列的乘积。

        :param axis: 0=列方向, 1=行方向
        :param skipna: 是否跳过 None
        :param numeric_only: 是否仅计算数值列
        """
        import math

        if axis == 0:
            result = {}
            for c in self._columns:
                vals = list(self._inner.get_column(c).values)
                nums = [
                    v for v in vals if v is not None and isinstance(v, (int, float))
                ]
                if skipna:
                    # 使用 math.prod 替代显式循环
                    result[c] = math.prod(nums) if nums else 1
                else:
                    # 遇到 None 立即返回 None
                    if any(v is None for v in vals):
                        result[c] = None
                    else:
                        result[c] = math.prod(
                            v for v in vals if isinstance(v, (int, float))
                        )
            return Series(result)
        else:
            result = []
            for i in range(self._nrows):
                vals = [self._inner.get_column(c).values[i] for c in self._columns]
                nums = [
                    v for v in vals if v is not None and isinstance(v, (int, float))
                ]
                result.append(math.prod(nums) if nums else 1)
            return Series(result, index=self._index)

    product = prod

    def round(self, decimals: int = 0) -> "DataFrame":
        """四舍五入所有数值列。

        :param decimals: 小数位数
        """
        new_data = {}
        for c in self._columns:
            vals = list(self._inner.get_column(c).values)
            new_data[c] = [
                (
                    None
                    if v is None
                    else (round(v, decimals) if isinstance(v, (int, float)) else v)
                )
                for v in vals
            ]
        return DataFrame(new_data, index=self._index)

    def dot(self, other) -> Any:
        """矩阵乘法。

        :param other: DataFrame / Series / list
        """
        if isinstance(other, DataFrame):
            # DataFrame @ DataFrame
            if len(self._columns) != other._nrows:
                raise ValueError("shape mismatch")
            result = {}
            for j in other._columns:
                col_vals = list(other._inner.get_column(j).values)
                result_col = []
                for i in range(self._nrows):
                    row_vals = [
                        self._inner.get_column(c).values[i] for c in self._columns
                    ]
                    s = sum(
                        a * b
                        for a, b in zip(row_vals, col_vals)
                        if a is not None and b is not None
                    )
                    result_col.append(s)
                result[j] = result_col
            return DataFrame(result)
        elif isinstance(other, Series):
            # DataFrame @ Series
            if len(self._columns) != len(other):
                raise ValueError("shape mismatch")
            result = []
            for i in range(self._nrows):
                row_vals = [self._inner.get_column(c).values[i] for c in self._columns]
                s = sum(
                    a * b
                    for a, b in zip(row_vals, other.values)
                    if a is not None and b is not None
                )
                result.append(s)
            return Series(result, index=self._index)
        elif isinstance(other, (list, tuple)):
            if len(self._columns) != len(other):
                raise ValueError("shape mismatch")
            result = []
            for i in range(self._nrows):
                row_vals = [self._inner.get_column(c).values[i] for c in self._columns]
                s = sum(
                    a * b
                    for a, b in zip(row_vals, other)
                    if a is not None and b is not None
                )
                result.append(s)
            return Series(result, index=self._index)
        raise TypeError(f"unsupported type: {type(other).__name__}")

    # ---------- 迭代 / 键 ----------

    def items(self):
        """迭代 (列名, Series) 对。"""
        for c in self._columns:
            yield c, self._get_column_as_series(c)

    def iteritems(self):
        """items 的别名。"""
        return self.items()

    def iterrows(self):
        """迭代 (索引, 行数据) 对。"""
        for i in range(self._nrows):
            idx = self._index[i] if self._index else i
            row_data = {c: self._inner.get_column(c).values[i] for c in self._columns}
            yield idx, row_data

    def keys(self) -> list:
        """返回列名列表。"""
        return self.columns

    # ---------- 采样 / 压缩 ----------

    def sample(
        self,
        n: int = None,
        frac: float = None,
        replace: bool = False,
        weights=None,
        random_state=None,
        axis: int = 0,
    ) -> "DataFrame":
        """随机采样行。

        :param n: 采样数量
        :param frac: 采样比例
        :param replace: 是否有放回
        :param weights: 权重
        :param random_state: 随机种子
        """
        import random as _random

        if frac is not None:
            n = int(self._nrows * frac)
        elif n is None:
            n = 1

        if random_state is not None:
            _random.seed(random_state)

        if replace:
            indices = [_random.randint(0, self._nrows - 1) for _ in range(n)]
        else:
            indices = _random.sample(range(self._nrows), min(n, self._nrows))

        new_data = {
            c: [self._inner.get_column(c).values[i] for i in indices]
            for c in self._columns
        }
        new_index = [self._index[i] if self._index else i for i in indices]
        return DataFrame(new_data, index=new_index)

    def squeeze(self, axis=None):
        """压缩维度：单列或单行返回 Series。"""
        if self._nrows == 1 and len(self._columns) == 1:
            return self._inner.get_column(self._columns[0]).values[0]
        if len(self._columns) == 1:
            return self._get_column_as_series(self._columns[0])
        if self._nrows == 1:
            return Series(
                [self._inner.get_column(c).values[0] for c in self._columns],
                index=self._columns,
            )
        return self.copy()

    # ---------- 对齐 / 组合 ----------

    def align(
        self,
        other,
        join: str = "outer",
        axis=None,
        level=None,
        copy: bool = True,
        fill_value=None,
    ):
        """对齐两个 DataFrame 的索引和列。

        :param other: 另一个 DataFrame
        :param join: 连接方式 ('outer'/'left'/'right'/'inner')
        :return: (aligned_self, aligned_other)
        """
        if not isinstance(other, DataFrame):
            raise TypeError("align requires DataFrame inputs")

        # 对齐索引
        if join == "outer":
            new_index = sorted(set(self._index) | set(other._index))
        elif join == "inner":
            new_index = sorted(set(self._index) & set(other._index))
        elif join == "left":
            new_index = list(self._index)
        elif join == "right":
            new_index = list(other._index)
        else:
            raise ValueError(f"invalid join: {join}")

        self_aligned = self.reindex(index=new_index)
        other_aligned = other.reindex(index=new_index)
        return self_aligned, other_aligned

    def combine_first(self, other: "DataFrame") -> "DataFrame":
        """用 other 的值填充 self 中的缺失值。

        :param other: 另一个 DataFrame
        """
        if not isinstance(other, DataFrame):
            raise TypeError("combine_first requires DataFrame inputs")

        # 合并列名 - 使用 dict.fromkeys 保持顺序去重，替代显式 for 循环
        all_cols = list(dict.fromkeys(list(self._columns) + list(other._columns)))

        def _combine_col(c):
            """合并单列：self 优先，None 用 other 填充。"""
            self_vals = (
                list(self._inner.get_column(c).values)
                if c in self._columns
                else [None] * self._nrows
            )
            other_vals = (
                list(other._inner.get_column(c).values)
                if c in other._columns
                else [None] * other._nrows
            )
            max_len = max(len(self_vals), len(other_vals))
            # 使用列表推导式替代显式 for 循环
            return [
                (
                    (self_vals[i] if i < len(self_vals) else None)
                    if (i < len(self_vals) and self_vals[i] is not None)
                    else (other_vals[i] if i < len(other_vals) else None)
                )
                for i in range(max_len)
            ]

        # 使用字典推导式替代显式 for 循环
        new_data = {c: _combine_col(c) for c in all_cols}
        return DataFrame(new_data)

    def update(
        self,
        other: "DataFrame",
        join: str = "left",
        overwrite: bool = True,
        filter_func=None,
        errors: str = "ignore",
    ) -> None:
        """用 other 的值原地更新 self 中的对应位置。

        :param other: 另一个 DataFrame
        :param join: 连接方式 ('left'/'right')，目前仅支持 'left'
        :param overwrite: 是否覆盖非 NA 值
        :param filter_func: 过滤函数（接收索引位置，返回 bool）
        :param errors: 错误处理 ('ignore'/'raise')
        """
        if not isinstance(other, DataFrame):
            raise TypeError("update requires DataFrame inputs")
        if errors not in ("ignore", "raise"):
            raise ValueError("errors must be 'ignore' or 'raise'")

        new_data = {}
        for c in self._columns:
            vals = list(self._inner.get_column(c).values)
            if c in other._columns:
                other_vals = list(other._inner.get_column(c).values)
                for i in range(min(len(vals), len(other_vals))):
                    if other_vals[i] is None:
                        continue
                    if filter_func is not None and not filter_func(i):
                        continue
                    if vals[i] is not None and not overwrite:
                        continue
                    vals[i] = other_vals[i]
            new_data[c] = vals
        self._reload(new_data)

    # ---------- 重索引扩展 ----------

    def reindex_like(
        self, other: "DataFrame", method: str = None, copy: bool = True
    ) -> "DataFrame":
        """将 self 的索引和列对齐到 other。

        :param other: 模板 DataFrame
        :param method: 填充方法 ('ffill'/'bfill'/None)
        :param copy: 是否复制
        """
        return self.reindex(index=other._index, columns=other._columns)

    # ---------- 前缀 / 后缀 ----------

    def add_prefix(self, prefix: str) -> "DataFrame":
        """给列名添加前缀。

        :param prefix: 前缀字符串
        """
        # 使用字典推导式替代显式 for 循环
        new_data = {
            f"{prefix}{c}": list(self._inner.get_column(c).values)
            for c in self._columns
        }
        return DataFrame(new_data, index=self._index)

    def add_suffix(self, suffix: str) -> "DataFrame":
        """给列名添加后缀。

        :param suffix: 后缀字符串
        """
        new_data = {
            f"{c}{suffix}": list(self._inner.get_column(c).values)
            for c in self._columns
        }
        return DataFrame(new_data, index=self._index)

    # ---------- v2.1.0: 标量索引 / 类型推断 / 属性扩展 ----------

    @property
    def at(self):
        """标量标签索引访问器。"""
        return _AtIndexer(self)

    @property
    def iat(self):
        """标量位置索引访问器。"""
        return _IatIndexer(self)

    def append(
        self,
        other,
        ignore_index: bool = False,
        verify_integrity: bool = False,
        sort: bool = False,
    ) -> "DataFrame":
        """追加行（pandas 已废弃，仍保留）。

        :param other: DataFrame 或 dict 列表
        :param ignore_index: 是否重置索引
        :param verify_integrity: 是否校验索引唯一
        :param sort: 是否排序列
        """
        if isinstance(other, dict):
            other = DataFrame([other])
        elif isinstance(other, list):
            other = DataFrame(other)
        elif not isinstance(other, DataFrame):
            raise TypeError("append requires DataFrame or dict/list")

        # 合并列
        all_cols = list(self._columns)
        for c in other._columns:
            if c not in all_cols:
                all_cols.append(c)

        # 合并数据
        new_data = {c: [] for c in all_cols}
        for c in all_cols:
            if c in self._columns:
                new_data[c].extend(self[c].values)
            else:
                new_data[c].extend([None] * self._nrows)
            if c in other._columns:
                new_data[c].extend(other[c].values)
            else:
                new_data[c].extend([None] * other._nrows)

        new_index = (
            list(range(self._nrows + other._nrows))
            if ignore_index
            else list(self._index) + list(other._index)
        )
        if verify_integrity and not ignore_index:
            seen = set()
            for i in new_index:
                if i in seen:
                    raise ValueError(f"Indexes have overlapping values: {i}")
                seen.add(i)

        if sort:
            all_cols = sorted(all_cols)
            new_data = {c: new_data[c] for c in all_cols}

        return DataFrame(new_data, index=new_index)

    def merge_asof(
        self,
        right,
        on=None,
        left_on=None,
        right_on=None,
        left_index: bool = False,
        right_index: bool = False,
        by=None,
        left_by=None,
        right_by=None,
        suffixes=("_x", "_y"),
        tolerance=None,
        allow_exact_matches: bool = True,
        direction: str = "backward",
    ) -> "DataFrame":
        """近似合并（asof join）。"""
        # 复用顶层 merge_asof 函数
        from . import merge_asof as _merge_asof

        return _merge_asof(
            self,
            right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            by=by,
            left_by=left_by,
            right_by=right_by,
            suffixes=suffixes,
            tolerance=tolerance,
            allow_exact_matches=allow_exact_matches,
            direction=direction,
        )

    def wide_to_long(
        self,
        stubnames,
        i,
        j,
        sep: str = "",
        suffix: str = r"\d+",
    ) -> "DataFrame":
        """宽表转长表。"""
        from . import wide_to_long as _wide_to_long

        return _wide_to_long(self, stubnames, i, j, sep=sep, suffix=suffix)

    def infer_objects(self, copy: bool = True) -> "DataFrame":
        """推断对象 dtype 列的类型。"""
        # 简化实现：对每列调用 Series.infer_objects
        new_data = {
            c: list(self[c].infer_objects(copy=copy).values) for c in self._columns
        }
        result = DataFrame(new_data, index=self._index)
        return result if copy else self._reload_inplace(new_data)

    def convert_dtypes(
        self,
        infer_objects: bool = True,
        convert_string: bool = True,
        convert_integer: bool = True,
        convert_boolean: bool = True,
        convert_floating: bool = True,
    ) -> "DataFrame":
        """将列转换为最佳可能的 dtype。"""
        new_data = {
            c: list(
                self[c]
                .convert_dtypes(
                    infer_objects=infer_objects,
                    convert_string=convert_string,
                    convert_integer=convert_integer,
                    convert_boolean=convert_boolean,
                    convert_floating=convert_floating,
                )
                .values
            )
            for c in self._columns
        }
        return DataFrame(new_data, index=self._index)

    def _reload_inplace(self, new_data) -> "DataFrame":
        """原地重载数据（内部辅助）。"""
        new_df = DataFrame(new_data, index=self._index)
        self._inner = new_df._inner
        self._columns = new_df._columns
        self._index = new_df._index
        self._nrows = new_df._nrows
        return self

    @property
    def attrs(self) -> dict:
        """全局属性字典。"""
        if not hasattr(self, "_attrs"):
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError("attrs must be a dict")
        self._attrs = value

    @property
    def flags(self) -> dict:
        """标志字典。"""
        return {"allows_duplicate_labels": True}

    @property
    def sparse(self):
        """稀疏访问器（不支持，抛出 NotImplementedError）。"""
        raise NotImplementedError("DataFrame.sparse not supported")

    # ====================================================================
    # 扩展功能：数据质量 / 清洗 / 高级统计（pandas 之外的增强）
    # ====================================================================

    def profile(self) -> "DataFrame":
        """生成数据概览报告。

        返回一个 DataFrame，每行对应原 DataFrame 的一列，包含：
        - dtype: 列数据类型
        - count: 非空值数
        - missing: 缺失值数
        - missing_rate: 缺失率（0~1）
        - unique: 唯一值数
        - sample: 前三个非空值样本
        - mean / std / min / max: 数值列的统计指标（非数值列为 None）

        Returns:
            DataFrame: 概览报告

        Examples:
            >>> df = DataFrame({"a": [1, 2, None, 4], "b": ["x", "y", "z", None]})
            >>> report = df.profile()
            >>> list(report["a"].values)[0]  # dtype
            'float64'
        """
        n = len(self)
        # 利用辅助函数 + 列表推导式构建行数据
        stats: List[Dict[str, Any]] = []

        def _col_stats(col: str) -> Dict[str, Any]:
            """计算单列的统计信息。"""
            series = self._inner.get_column(col)
            values = list(series.values)
            non_null = [v for v in values if v is not None]
            count = len(non_null)
            missing = n - count
            unique = len(set(non_null))
            sample = non_null[:3]
            # 数值列才计算统计指标
            is_numeric = series.dtype in ("int64", "float64")
            if is_numeric and non_null:
                mean_v = sum(non_null) / count
                std_v = (
                    (sum((v - mean_v) ** 2 for v in non_null) / (count - 1)) ** 0.5
                    if count > 1
                    else None
                )
                min_v = min(non_null)
                max_v = max(non_null)
            else:
                mean_v = std_v = min_v = max_v = None
            return {
                "column": col,
                "dtype": series.dtype,
                "count": count,
                "missing": missing,
                "missing_rate": missing / n if n > 0 else 0.0,
                "unique": unique,
                "sample": str(sample),
                "mean": mean_v,
                "std": std_v,
                "min": min_v,
                "max": max_v,
            }

        # 使用列表推导式替代显式 for 循环
        stats = [_col_stats(c) for c in self._columns]
        return DataFrame(stats)

    def validate(self, schema: Dict[str, Dict[str, Any]]) -> List[str]:
        """按模式校验数据。

        Parameters:
            schema: {列名: {规则: 值}}，支持的规则：
                - dtype: 期望的 dtype 字符串
                - non_null: True 表示不允许缺失
                - min / max: 数值范围

        Returns:
            List[str]: 错误信息列表（空列表表示校验通过）

        Examples:
            >>> df = DataFrame({"a": [1, 2, None]})
            >>> errors = df.validate({"a": {"non_null": True, "min": 0}})
            >>> len(errors) > 0
            True
        """
        errors: List[str] = []

        def _check_one(col: str, rule: Dict[str, Any]) -> List[str]:
            """校验单列，返回错误信息列表。"""
            errs: List[str] = []
            if col not in self._columns:
                errs.append(f"列 '{col}' 不存在")
                return errs
            series = self._inner.get_column(col)
            values = list(series.values)
            # dtype 校验
            if "dtype" in rule and series.dtype != rule["dtype"]:
                errs.append(
                    f"列 '{col}' dtype 不匹配: 期望 {rule['dtype']}, 实际 {series.dtype}"
                )
            # non_null 校验
            if rule.get("non_null") and any(v is None for v in values):
                null_count = sum(1 for v in values if v is None)
                errs.append(f"列 '{col}' 存在 {null_count} 个空值，违反 non_null 规则")
            # min / max 校验
            if rule.get("min") is not None:
                bad = [v for v in values if v is not None and v < rule["min"]]
                if bad:
                    errs.append(f"列 '{col}' 有 {len(bad)} 个值小于 min={rule['min']}")
            if rule.get("max") is not None:
                bad = [v for v in values if v is not None and v > rule["max"]]
                if bad:
                    errs.append(f"列 '{col}' 有 {len(bad)} 个值大于 max={rule['max']}")
            return errs

        # 使用列表推导式 + chain 展开校验结果
        from itertools import chain

        errors = list(
            chain.from_iterable(_check_one(col, rule) for col, rule in schema.items())
        )
        return errors

    def detect_outliers(
        self, columns: Optional[List[str]] = None, method: str = "iqr"
    ) -> "DataFrame":
        """检测异常值。

        Parameters:
            columns: 要检测的列名列表，None 表示所有数值列
            method: 检测方法，'iqr' (四分位距) 或 'zscore' (Z-score)

        Returns:
            DataFrame: 布尔类型 DataFrame，True 表示该位置是异常值

        Examples:
            >>> df = DataFrame({"a": [1, 2, 3, 100]})
            >>> outliers = df.detect_outliers(method="iqr")
            >>> outliers["a"].values[-1]
            True
        """
        # 默认所有数值列
        if columns is None:
            columns = [
                c
                for c in self._columns
                if self._inner.get_column(c).dtype in ("int64", "float64")
            ]
        if not columns:
            return DataFrame({})

        def _detect_col(col: str) -> List[bool]:
            """检测单列的异常值。"""
            values = list(self._inner.get_column(col).values)
            non_null = [v for v in values if v is not None]
            if len(non_null) < 4:
                return [False] * len(values)

            if method == "iqr":
                sorted_vals = sorted(non_null)
                q1 = sorted_vals[len(sorted_vals) // 4]
                q3 = sorted_vals[3 * len(sorted_vals) // 4]
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                return [(v is not None and (v < lower or v > upper)) for v in values]
            elif method == "zscore":
                mean_v = sum(non_null) / len(non_null)
                std_v = (
                    sum((v - mean_v) ** 2 for v in non_null) / len(non_null)
                ) ** 0.5
                if std_v == 0:
                    return [False] * len(values)
                # 工业标准：|z| > 2 视为异常值
                return [
                    (v is not None and abs((v - mean_v) / std_v) > 2)
                    for v in values
                ]
            raise ValueError(f"Unsupported method: {method}")

        # 使用字典推导式构建结果
        result = {col: _detect_col(col) for col in columns}
        return DataFrame(result)

    def compare_with(self, other: "DataFrame", show_all: bool = False) -> "DataFrame":
        """增强版对比，默认只展示差异行。

        Parameters:
            other: 另一个 DataFrame，需具有相同的列结构
            show_all: True 表示展示所有行，False 只展示有差异的行

        Returns:
            DataFrame: 差异行 DataFrame

        Examples:
            >>> df1 = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            >>> df2 = DataFrame({"a": [1, 2, 9], "b": [4, 5, 7]})
            >>> diff = df1.compare_with(df2)
            >>> len(diff) == 1
            True
        """
        if self._columns != other._columns:
            raise ValueError("两 DataFrame 列结构不一致")

        # 使用 zip + 列表推导式标记差异行
        n = len(self)
        diff_flags = [
            any(self[c].values[i] != other[c].values[i] for c in self._columns)
            for i in range(n)
        ]
        target_indices = (
            list(range(n)) if show_all else [i for i, f in enumerate(diff_flags) if f]
        )
        # 收集差异行：每行包含每列的 _self 和 _other 两个版本
        rows = []
        for i in target_indices:
            row = {}
            for c in self._columns:
                row[f"{c}_self"] = self[c].values[i]
                row[f"{c}_other"] = other[c].values[i]
            rows.append(row)
        return DataFrame(rows)

    def snapshot(self, path: str) -> None:
        """保存当前状态快照（含索引、列名、dtype、数据）。

        Parameters:
            path: 快照文件路径（.json 格式）

        Examples:
            >>> import tempfile, os
            >>> df = DataFrame({"a": [1, 2, 3]})
            >>> with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            ...     path = f.name
            >>> df.snapshot(path)
            >>> os.path.exists(path)
            True
        """
        import json

        state = {
            "columns": list(self._columns),
            "dtypes": {c: self._inner.get_column(c).dtype for c in self._columns},
            "index": list(self._index) if self._index else None,
            "data": [
                {c: self[c].values[i] for c in self._columns} for i in range(len(self))
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, default=str)

    @classmethod
    def from_snapshot(cls, path: str) -> "DataFrame":
        """从快照恢复 DataFrame。

        Parameters:
            path: 快照文件路径

        Examples:
            >>> df = DataFrame({"a": [1, 2, 3]})
            >>> df.snapshot('/tmp/snap.json')  # doctest: +SKIP
            >>> df2 = DataFrame.from_snapshot('/tmp/snap.json')  # doctest: +SKIP
        """
        import json

        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        df = cls(state["data"], columns=state["columns"])
        if state.get("index"):
            df._index = list(state["index"])
        return df

    def clean(self) -> "DataFrame":
        """一键清洗：去重 / 去空行 / 类型推断 / 列名标准化。

        Returns:
            DataFrame: 清洗后的 DataFrame

        Examples:
            >>> df = DataFrame({"A ": [1, 1, None], "b": [4, None, 6]})
            >>> cleaned = df.clean()
            >>> 'A ' in cleaned.columns
            False
        """
        # 1) 列名标准化（去空格 / 小写 / 替换特殊字符）
        rename_map = {
            c: c.strip().lower().replace(" ", "_").replace("-", "_")
            for c in self._columns
        }
        df_renamed = self.rename(columns=rename_map)
        # 2) 去重行
        df_dedup = df_renamed.drop_duplicates()
        # 3) 去除全空行
        keep_indices = [
            i
            for i in range(len(df_dedup))
            if any(df_dedup[c].values[i] is not None for c in df_dedup._columns)
        ]
        # 使用 iloc 选取行
        if keep_indices:
            return df_dedup.iloc[keep_indices]
        return df_dedup

    def standardize_names(self) -> "DataFrame":
        """列名标准化（去空格 / 统一小写 / 特殊字符替换）。

        Examples:
            >>> df = DataFrame({"First Name": [1], "Last Name": [2]})
            >>> df2 = df.standardize_names()
            >>> list(df2.columns)
            ['first_name', 'last_name']
        """
        rename_map = {
            c: c.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
            for c in self._columns
        }
        return self.rename(columns=rename_map)

    def normalize(
        self,
        columns: Optional[List[str]] = None,
        method: str = "minmax",
    ) -> "DataFrame":
        """数值列归一化。

        Parameters:
            columns: 要归一化的列名列表，None 表示所有数值列
            method: 归一化方法，'minmax' (默认) / 'zscore' / 'robust'

        Returns:
            DataFrame: 归一化后的 DataFrame（仅含目标列）

        Examples:
            >>> df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            >>> norm = df.normalize(method="minmax")
            >>> abs(norm["a"].values[0]) < 1e-9
            True
            >>> abs(norm["a"].values[2] - 1.0) < 1e-9
            True
        """
        if columns is None:
            columns = [
                c
                for c in self._columns
                if self._inner.get_column(c).dtype in ("int64", "float64")
            ]
        if not columns:
            return DataFrame({})

        def _normalize_col(col: str) -> List[Optional[float]]:
            """归一化单列。"""
            values = list(self._inner.get_column(col).values)
            non_null = [v for v in values if v is not None]
            if not non_null:
                return values

            if method == "minmax":
                lo = min(non_null)
                hi = max(non_null)
                rng = hi - lo
                return [
                    (None if v is None else (0.0 if rng == 0 else (v - lo) / rng))
                    for v in values
                ]
            elif method == "zscore":
                mean_v = sum(non_null) / len(non_null)
                std_v = (
                    sum((v - mean_v) ** 2 for v in non_null) / len(non_null)
                ) ** 0.5
                if std_v == 0:
                    return [0.0 if v is not None else None for v in values]
                return [(None if v is None else (v - mean_v) / std_v) for v in values]
            elif method == "robust":
                sorted_vals = sorted(non_null)
                median = sorted_vals[len(sorted_vals) // 2]
                q1 = sorted_vals[len(sorted_vals) // 4]
                q3 = sorted_vals[3 * len(sorted_vals) // 4]
                iqr = q3 - q1
                if iqr == 0:
                    return [0.0 if v is not None else None for v in values]
                return [(None if v is None else (v - median) / iqr) for v in values]
            raise ValueError(f"Unsupported method: {method}")

        # 使用字典推导式构建归一化结果
        result = {col: _normalize_col(col) for col in columns}
        return DataFrame(result)


class DataFrameGroupBy:
    """DataFrame 分组结果 (极简版)。"""

    def __init__(
        self,
        df: "DataFrame",
        by,
        as_index: bool = True,
        sort: bool = True,
        dropna: bool = True,
    ):
        if isinstance(by, str):
            self._by = [by]
        else:
            self._by = list(by) if by is not None else []
        self._df = df
        self._as_index = as_index
        self._sort = sort
        self._dropna = dropna

        # 分组: { key_tuple: [row_indices] }
        self._groups: Dict[tuple, list] = {}
        n = df._nrows

        for i in range(n):
            key = tuple(df._inner.get_column(c).values[i] for c in self._by)

            # 处理 dropna
            if not dropna and any(v is None for v in key):
                continue

            self._groups.setdefault(key, []).append(i)

        # 排序组
        if sort:
            sorted_groups = dict(sorted(self._groups.items()))
            self._groups = sorted_groups

    def _agg(self, agg_funcs: Dict[str, str]) -> "DataFrame":
        """对每列应用聚合函数。

        :param agg_funcs: {列名: 'sum' | 'mean' | 'min' | 'max' | 'count' | 'std' | 'var' | 'median' | 'first' | 'last'}

        分组键不作为列输出（与 pandas as_index=True 行为一致）。
        """
        agg_cols = list(agg_funcs.keys())
        result: Dict[str, list] = {c: [] for c in agg_cols}

        for key, idxs in self._groups.items():
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
            c
            for c in self._df._columns
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
        result: Dict[str, list] = {}
        other_cols = [c for c in self._df._columns if c not in self._by]
        for c in other_cols:
            result[c] = []

        for key, idxs in self._groups.items():
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
                        if (
                            j == len(indexed)
                            or indexed[j][0] != indexed[group_start][0]
                        ):
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
        result: Dict[str, list] = {}
        other_cols = [c for c in self._df._columns if c not in self._by]
        for c in other_cols:
            result[c] = []

        for key, idxs in self._groups.items():
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
        result: Dict[str, list] = {}
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
                pairs = [
                    (a, b)
                    for a, b in zip(col_vals, other_vals)
                    if a is not None and b is not None
                ]
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
        result: Dict[str, list] = {}
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
                pairs = [
                    (a, b)
                    for a, b in zip(col_vals, other_vals)
                    if a is not None and b is not None
                ]
                if len(pairs) < 2:
                    result[c].append(None)
                    continue
                ma = sum(a for a, b in pairs) / len(pairs)
                mb = sum(b for a, b in pairs) / len(pairs)
                result[c].append(
                    sum((a - ma) * (b - mb) for a, b in pairs) / len(pairs)
                )

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
                all_pairs.extend(
                    [
                        (a, b)
                        for a, b in zip(col_a, col_b)
                        if a is not None and b is not None
                    ]
                )
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
                    elif (
                        vals[j - periods] is None
                        or vals[j - periods] == 0
                        or vals[j] is None
                    ):
                        result[c][idx] = None
                    else:
                        result[c][idx] = (vals[j] - vals[j - periods]) / vals[
                            j - periods
                        ]

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

    # ---------- v2.1.0: GroupBy 补全 ----------

    def std(self, ddof: int = 1) -> "DataFrame":
        """分组标准差。

        :param ddof: 自由度修正（默认 1）
        """
        return self._agg_with_ddof("std", ddof)

    def var(self, ddof: int = 1) -> "DataFrame":
        """分组方差。

        :param ddof: 自由度修正（默认 1）
        """
        return self._agg_with_ddof("var", ddof)

    def median(self) -> "DataFrame":
        """分组中位数。"""
        return self._agg({c: "median" for c in self._df._columns if c not in self._by})

    def sem(self, ddof: int = 1) -> "DataFrame":
        """分组标准误差。

        :param ddof: 自由度修正（默认 1）
        """
        return self._agg_with_ddof("sem", ddof)

    def mad(self) -> "DataFrame":
        """分组平均绝对偏差。"""
        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {}
        for c in other_cols:
            result[c] = []

        for key, idxs in self._groups.items():
            for c in other_cols:
                vals = [v for v in self._df[c].iloc(idxs).values if v is not None]
                if not vals:
                    result[c].append(None)
                    continue
                m = sum(vals) / len(vals)
                result[c].append(sum(abs(x - m) for x in vals) / len(vals))
        return DataFrame(result)

    def prod(self) -> "DataFrame":
        """分组乘积。"""
        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {}
        for c in other_cols:
            result[c] = []

        for key, idxs in self._groups.items():
            for c in other_cols:
                vals = [v for v in self._df[c].iloc(idxs).values if v is not None]
                if not vals:
                    result[c].append(None)
                    continue
                p = 1
                for v in vals:
                    p *= v
                result[c].append(p)
        return DataFrame(result)

    def skew(self) -> "DataFrame":
        """分组偏度。"""
        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {}
        for c in other_cols:
            result[c] = []

        for key, idxs in self._groups.items():
            for c in other_cols:
                vals = [v for v in self._df[c].iloc(idxs).values if v is not None]
                if len(vals) < 3:
                    result[c].append(None)
                    continue
                n = len(vals)
                m = sum(vals) / n
                s = (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5
                if s == 0:
                    result[c].append(None)
                else:
                    result[c].append(sum((x - m) ** 3 for x in vals) / ((n - 1) * s**3))
        return DataFrame(result)

    def kurt(self) -> "DataFrame":
        """分组峰度。"""
        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {}
        for c in other_cols:
            result[c] = []

        for key, idxs in self._groups.items():
            for c in other_cols:
                vals = [v for v in self._df[c].iloc(idxs).values if v is not None]
                if len(vals) < 4:
                    result[c].append(None)
                    continue
                n = len(vals)
                m = sum(vals) / n
                s2 = sum((x - m) ** 2 for x in vals) / (n - 1)
                if s2 == 0:
                    result[c].append(None)
                    continue
                s4 = sum((x - m) ** 4 for x in vals) / (n - 1)
                result[c].append(s4 / (s2**2) - 3)
        return DataFrame(result)

    def nunique(self) -> "DataFrame":
        """分组唯一值计数。"""
        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {}
        for c in other_cols:
            result[c] = []

        for key, idxs in self._groups.items():
            for c in other_cols:
                vals = [v for v in self._df[c].iloc(idxs).values if v is not None]
                result[c].append(len(set(vals)))
        return DataFrame(result)

    def size(self) -> "Series":
        """返回每个分组的大小。"""
        from .series import Series

        keys = list(self._groups.keys())
        sizes = [len(idxs) for idxs in self._groups.values()]
        return Series(sizes, index=keys)

    def describe(self) -> "DataFrame":
        """分组描述统计。每行为一个统计量，每列为 '列名_组键'。"""
        other_cols = [c for c in self._df._columns if c not in self._by]
        stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        group_keys = list(self._groups.keys())
        result: Dict[str, list] = {"stat": stats}

        for c in other_cols:
            for gk in group_keys:
                col_name = f"{c}_{gk}" if len(group_keys) > 1 else c
                result[col_name] = []
                idxs = self._groups[gk]
                sub_vals = [v for v in self._df[c].iloc(idxs).values if v is not None]
                for stat_name in stats:
                    if not sub_vals:
                        result[col_name].append(None)
                        continue
                    if stat_name == "count":
                        result[col_name].append(len(sub_vals))
                    elif stat_name == "mean":
                        result[col_name].append(sum(sub_vals) / len(sub_vals))
                    elif stat_name == "std":
                        n = len(sub_vals)
                        if n > 1:
                            m = sum(sub_vals) / n
                            variance = sum((x - m) ** 2 for x in sub_vals) / (n - 1)
                            result[col_name].append(variance**0.5)
                        else:
                            result[col_name].append(None)
                    elif stat_name == "min":
                        result[col_name].append(min(sub_vals))
                    elif stat_name == "max":
                        result[col_name].append(max(sub_vals))
                    elif stat_name == "50%":
                        sv = sorted(sub_vals)
                        n = len(sv)
                        result[col_name].append(
                            sv[n // 2]
                            if n % 2 == 1
                            else (sv[n // 2 - 1] + sv[n // 2]) / 2
                        )
                    elif stat_name == "25%":
                        sv = sorted(sub_vals)
                        n = len(sv)
                        pos = 0.25 * (n - 1)
                        lo = int(pos)
                        hi = min(lo + 1, n - 1)
                        result[col_name].append(sv[lo] + (sv[hi] - sv[lo]) * (pos - lo))
                    elif stat_name == "75%":
                        sv = sorted(sub_vals)
                        n = len(sv)
                        pos = 0.75 * (n - 1)
                        lo = int(pos)
                        hi = min(lo + 1, n - 1)
                        result[col_name].append(sv[lo] + (sv[hi] - sv[lo]) * (pos - lo))

        return DataFrame(result)

    def apply(self, func, *args, **kwargs) -> "DataFrame":
        """对每个分组应用函数。

        :param func: 接收 DataFrame 的函数
        """
        parts = []
        for key, idxs in self._groups.items():
            sub_df = self._df.iloc[idxs]
            result = func(sub_df, *args, **kwargs)
            if isinstance(result, DataFrame):
                parts.append(result)
            elif isinstance(result, dict):
                parts.append(DataFrame(result))
        if not parts:
            return DataFrame()
        return DataFrame.concat(parts)

    def transform(self, func, *args, **kwargs) -> "DataFrame":
        """对每个分组应用变换函数，返回与原 DataFrame 等长的结果。

        :param func: 接收 DataFrame 的函数
        """
        result: Dict[str, list] = {
            c: [None] * self._df._nrows for c in self._df._columns
        }
        for key, idxs in self._groups.items():
            sub_df = self._df.iloc[idxs]
            transformed = func(sub_df, *args, **kwargs)
            if isinstance(transformed, DataFrame):
                for c in self._df._columns:
                    if c in transformed._columns:
                        for j, idx in enumerate(idxs):
                            result[c][idx] = transformed._inner.get_column(c).values[j]
            elif isinstance(transformed, dict):
                for c, vals in transformed.items():
                    for j, idx in enumerate(idxs):
                        result[c][idx] = vals[j] if j < len(vals) else None
        return DataFrame(result)

    def filter(self, func, *args, **kwargs) -> "DataFrame":
        """过滤分组，保留满足条件的组。

        :param func: 接收 DataFrame 返回 bool 的函数
        """
        keep_indices = []
        for key, idxs in self._groups.items():
            sub_df = self._df.iloc[idxs]
            if func(sub_df, *args, **kwargs):
                keep_indices.extend(idxs)
        return self._df.iloc[keep_indices]

    def cumsum(self) -> "DataFrame":
        """分组累加和。"""
        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {
            c: [None] * self._df._nrows for c in self._df._columns
        }
        for idxs in self._groups.values():
            for c in other_cols:
                vals = [self._df._inner.get_column(c).values[i] for i in idxs]
                cum = 0
                for j, idx in enumerate(idxs):
                    if vals[j] is not None:
                        cum += vals[j]
                    result[c][idx] = cum
        return DataFrame(result)

    def cumprod(self) -> "DataFrame":
        """分组累乘积。"""
        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {
            c: [None] * self._df._nrows for c in self._df._columns
        }
        for idxs in self._groups.values():
            for c in other_cols:
                vals = [self._df._inner.get_column(c).values[i] for i in idxs]
                cum = 1
                for j, idx in enumerate(idxs):
                    if vals[j] is not None:
                        cum *= vals[j]
                    result[c][idx] = cum
        return DataFrame(result)

    def cummax(self) -> "DataFrame":
        """分组累最大值。"""
        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {
            c: [None] * self._df._nrows for c in self._df._columns
        }
        for idxs in self._groups.values():
            for c in other_cols:
                vals = [self._df._inner.get_column(c).values[i] for i in idxs]
                cur_max = None
                for j, idx in enumerate(idxs):
                    if vals[j] is not None:
                        cur_max = vals[j] if cur_max is None else max(cur_max, vals[j])
                    result[c][idx] = cur_max
        return DataFrame(result)

    def cummin(self) -> "DataFrame":
        """分组累最小值。"""
        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {
            c: [None] * self._df._nrows for c in self._df._columns
        }
        for idxs in self._groups.values():
            for c in other_cols:
                vals = [self._df._inner.get_column(c).values[i] for i in idxs]
                cur_min = None
                for j, idx in enumerate(idxs):
                    if vals[j] is not None:
                        cur_min = vals[j] if cur_min is None else min(cur_min, vals[j])
                    result[c][idx] = cur_min
        return DataFrame(result)

    def diff(self, periods: int = 1) -> "DataFrame":
        """分组差分。

        :param periods: 差分周期
        """
        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {
            c: [None] * self._df._nrows for c in self._df._columns
        }
        for idxs in self._groups.values():
            for c in other_cols:
                vals = [self._df._inner.get_column(c).values[i] for i in idxs]
                for j, idx in enumerate(idxs):
                    if (
                        j >= periods
                        and vals[j] is not None
                        and vals[j - periods] is not None
                    ):
                        result[c][idx] = vals[j] - vals[j - periods]
        return DataFrame(result)

    def shift(self, periods: int = 1, fill_value=None) -> "DataFrame":
        """分组位移。

        :param periods: 位移周期
        :param fill_value: 填充值
        """
        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {
            c: [None] * self._df._nrows for c in self._df._columns
        }
        for idxs in self._groups.values():
            for c in other_cols:
                vals = [self._df._inner.get_column(c).values[i] for i in idxs]
                for j, idx in enumerate(idxs):
                    if j >= periods:
                        result[c][idx] = vals[j - periods]
                    elif fill_value is not None:
                        result[c][idx] = fill_value
        return DataFrame(result)

    def fillna(self, value=None, method=None, limit=None) -> "DataFrame":
        """分组填充缺失值。

        :param value: 填充值
        :param method: 填充方法 ('ffill'/'bfill')
        :param limit: 最大填充数
        """
        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {
            c: [None] * self._df._nrows for c in self._df._columns
        }
        for idxs in self._groups.values():
            for c in other_cols:
                vals = [self._df._inner.get_column(c).values[i] for i in idxs]
                if method == "ffill":
                    last_valid = None
                    fill_count = 0
                    for j, idx in enumerate(idxs):
                        if vals[j] is not None:
                            result[c][idx] = vals[j]
                            last_valid = vals[j]
                            fill_count = 0
                        elif last_valid is not None and (
                            limit is None or fill_count < limit
                        ):
                            result[c][idx] = last_valid
                            fill_count += 1
                        else:
                            result[c][idx] = None
                elif method == "bfill":
                    next_valid = None
                    fill_count = 0
                    for j in range(len(idxs) - 1, -1, -1):
                        if vals[j] is not None:
                            result[c][idxs[j]] = vals[j]
                            next_valid = vals[j]
                            fill_count = 0
                        elif next_valid is not None and (
                            limit is None or fill_count < limit
                        ):
                            result[c][idxs[j]] = next_valid
                            fill_count += 1
                        else:
                            result[c][idxs[j]] = None
                else:
                    for j, idx in enumerate(idxs):
                        result[c][idx] = value if vals[j] is None else vals[j]
        return DataFrame(result)

    def ffill(self, limit=None) -> "DataFrame":
        """分组前向填充。"""
        return self.fillna(method="ffill", limit=limit)

    def bfill(self, limit=None) -> "DataFrame":
        """分组后向填充。"""
        return self.fillna(method="bfill", limit=limit)

    def head(self, n: int = 5) -> "DataFrame":
        """返回每个分组的前 n 行。

        :param n: 行数
        """
        keep_indices = []
        for idxs in self._groups.values():
            keep_indices.extend(idxs[:n])
        return self._df.iloc[keep_indices]

    def tail(self, n: int = 5) -> "DataFrame":
        """返回每个分组的后 n 行。

        :param n: 行数
        """
        keep_indices = []
        for idxs in self._groups.values():
            keep_indices.extend(idxs[-n:] if n <= len(idxs) else idxs)
        return self._df.iloc[keep_indices]

    def idxmax(self) -> "DataFrame":
        """分组最大值索引。"""
        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {}
        for c in other_cols:
            result[c] = []
        for key, idxs in self._groups.items():
            for c in other_cols:
                vals = [
                    (v, i)
                    for i, v in enumerate(self._df[c].iloc(idxs).values)
                    if v is not None
                ]
                if vals:
                    result[c].append(max(vals, key=lambda x: x[0])[1])
                else:
                    result[c].append(None)
        return DataFrame(result)

    def idxmin(self) -> "DataFrame":
        """分组最小值索引。"""
        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {}
        for c in other_cols:
            result[c] = []
        for key, idxs in self._groups.items():
            for c in other_cols:
                vals = [
                    (v, i)
                    for i, v in enumerate(self._df[c].iloc(idxs).values)
                    if v is not None
                ]
                if vals:
                    result[c].append(min(vals, key=lambda x: x[0])[1])
                else:
                    result[c].append(None)
        return DataFrame(result)

    def get_group(self, name) -> "DataFrame":
        """获取指定分组的数据。

        :param name: 分组键
        """
        if isinstance(name, tuple):
            key = name
        else:
            key = (name,)
        idxs = self._groups.get(key, [])
        return self._df.iloc[idxs]

    @property
    def groups(self) -> dict:
        """返回 {分组键: [索引]} 字典。"""
        return dict(self._groups)

    @property
    def indices(self) -> dict:
        """返回 {分组键: [位置索引]} 字典。"""
        return dict(self._groups)

    def _agg_with_ddof(self, func_name: str, ddof: int) -> "DataFrame":
        """支持 ddof 参数的聚合方法。

        :param func_name: 'std'/'var'/'sem'
        :param ddof: 自由度修正
        """
        other_cols = [
            c
            for c in self._df._columns
            if c not in self._by
            and self._df._inner.get_column(c).dtype in ("int64", "float64")
        ]
        result: Dict[str, list] = {}
        for c in other_cols:
            result[c] = []

        for key, idxs in self._groups.items():
            for c in other_cols:
                vals = [v for v in self._df[c].iloc(idxs).values if v is not None]
                n = len(vals)
                if n <= ddof:
                    result[c].append(None)
                    continue
                m = sum(vals) / n
                variance = sum((x - m) ** 2 for x in vals) / (n - ddof)
                if func_name == "std":
                    result[c].append(variance**0.5)
                elif func_name == "var":
                    result[c].append(variance)
                elif func_name == "sem":
                    result[c].append((variance / n) ** 0.5)
        return DataFrame(result)


# ---------------------------------------------------------------------------
# 索引器
# ---------------------------------------------------------------------------


class _IndexerBase:
    """loc/iloc 索引器基类。"""

    def __init__(self, df: "DataFrame"):
        self._df = df


class _AtIndexer(_IndexerBase):
    """基于标签的标量索引器。"""

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("at[] requires a tuple (row_label, col_label)")
        row_label, col_label = key
        if row_label not in self._df._index:
            raise KeyError(f"row label {row_label!r} not found")
        if col_label not in self._df._columns:
            raise KeyError(f"column label {col_label!r} not found")
        row_idx = self._df._index.index(row_label)
        return self._df[col_label].values[row_idx]

    def __setitem__(self, key, value):
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("at[] requires a tuple (row_label, col_label)")
        row_label, col_label = key
        if row_label not in self._df._index:
            raise KeyError(f"row label {row_label!r} not found")
        if col_label not in self._df._columns:
            raise KeyError(f"column label {col_label!r} not found")
        # 简化实现：通过重建 DataFrame 修改值
        row_idx = self._df._index.index(row_label)
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
