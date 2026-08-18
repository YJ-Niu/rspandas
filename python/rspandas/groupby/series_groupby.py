"""SeriesGroupBy Series 分组

由 rspandas/series.py 拆分而来。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from ..dataframe import DataFrame
    from ..series import Series
from typing import Any, Optional, Union


class SeriesGroupBy:
    """Series 的 groupby 操作。"""

    def __init__(
        self,
        series: Series,
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
        self._s = series
        self._by = by
        self._axis = axis
        self._level = level
        self._as_index = as_index
        self._sort = sort
        self._group_keys = group_keys
        self._squeeze = squeeze
        self._observed = observed
        self._dropna = dropna
        self._groups = self._compute_groups()

    def _compute_groups(self) -> dict:
        """计算分组（列表推导式优化版）。"""
        values = self._s.values
        keys = [self._by_key(i, v) for i, v in enumerate(values)]
        # 使用 dict.setdefault + enumerate 一步构建分组，避免 if/else 嵌套
        groups: dict = {}
        for i, (v, key) in enumerate(zip(values, keys)):
            if key is None and self._dropna:
                continue
            groups.setdefault(key, []).append((i, v))
        return groups

    def _by_key(self, i: int, v) -> Any:
        """获取分组键。"""
        if self._level is not None:
            # 按 level 分组：从 Series 索引中取第 level 层
            idx = self._s._index
            if idx is None or i >= len(idx):
                return None
            return idx[i]
        if callable(self._by):
            return self._by(v)
        elif isinstance(self._by, Series):
            return self._by.values[i] if i < len(self._by.values) else None
        elif isinstance(self._by, list):
            return self._by[i] if i < len(self._by) else None
        else:
            return self._by

    def std(self, ddof: int = 1) -> Series:
        """分组标准差。

        :param ddof: 自由度修正（默认 1）
        """
        return self.agg("std", ddof=ddof)

    def var(self, ddof: int = 1) -> Series:
        """分组方差。

        :param ddof: 自由度修正（默认 1）
        """
        return self.agg("var", ddof=ddof)

    def median(self) -> Series:
        """分组中位数。"""
        return self.agg("median")

    def sem(self, ddof: int = 1) -> Series:
        """分组标准误差。

        :param ddof: 自由度修正（默认 1）
        """
        return self.agg("sem", ddof=ddof)

    def prod(self) -> Series:
        """分组乘积。"""
        return self.agg("prod")

    def first(self) -> Series:
        """分组第一个值。"""
        return self.agg("first")

    def last(self) -> Series:
        """分组最后一个值。"""
        return self.agg("last")

    def _sorted_keys(self, keys: list) -> list:
        """对 keys 排序（按 self._sort 决定）。"""
        if not self._sort:
            return list(keys)
        try:
            return sorted(keys)
        except TypeError:
            return list(keys)

    def _build_grouped_series(self, compute) -> Series:
        """通用辅助：对每个分组应用 compute(key, items)，返回 Series。"""
        # 使用字典推导式构建结果，再统一排序后构造 Series
        result = {key: compute(key, items) for key, items in self._groups.items()}
        keys = self._sorted_keys(result.keys())
        return Series([result[k] for k in keys], index=keys, name=self._s.name)

    def nth(self, n: int) -> Series:
        """返回每个分组的第 n 个值。

        :param n: 位置索引（支持负数）
        """

        def _nth(key, items):
            actual_n = len(items) + n if n < 0 else n
            return items[actual_n][1] if 0 <= actual_n < len(items) else None

        return self._build_grouped_series(_nth)

    def idxmax(self) -> Series:
        """分组最大值的位置索引。"""

        def _idxmax(key, items):
            vals = [(v, i) for i, v in items if v is not None]
            return max(vals, key=lambda x: x[0])[1] if vals else None

        return self._build_grouped_series(_idxmax)

    def idxmin(self) -> Series:
        """分组最小值的位置索引。"""

        def _idxmin(key, items):
            vals = [(v, i) for i, v in items if v is not None]
            return min(vals, key=lambda x: x[0])[1] if vals else None

        return self._build_grouped_series(_idxmin)

    def nunique(self) -> Series:
        """分组唯一值计数。"""

        def _nunique(key, items):
            return len({v for _, v in items if v is not None})

        return self._build_grouped_series(_nunique)

    def size(self) -> Series:
        """返回每个分组的大小。"""
        return self._build_grouped_series(lambda _, items: len(items))

    def describe(self) -> "DataFrame":
        """分组描述统计。"""
        from ..dataframe import DataFrame

        # 预计算每个分组的非空值（按 self._groups.values() 顺序）
        group_vals = [
            [v for _, v in items if v is not None] for items in self._groups.values()
        ]

        def _compute_stat(vals, stat_name):
            """对单组数值计算指定统计量。"""
            if not vals:
                return None
            n = len(vals)
            m = sum(vals) / n
            if stat_name == "count":
                return n
            if stat_name == "mean":
                return m
            if stat_name == "std":
                if n > 1:
                    variance = sum((x - m) ** 2 for x in vals) / (n - 1)
                    return variance**0.5
                return None
            if stat_name == "min":
                return min(vals)
            if stat_name == "max":
                return max(vals)
            if stat_name == "50%":
                sv = sorted(vals)
                return sv[n // 2] if n % 2 == 1 else (sv[n // 2 - 1] + sv[n // 2]) / 2
            if stat_name in ("25%", "75%"):
                sv = sorted(vals)
                pos = (0.25 if stat_name == "25%" else 0.75) * (n - 1)
                lo = int(pos)
                hi = min(lo + 1, n - 1)
                return sv[lo] + (sv[hi] - sv[lo]) * (pos - lo)
            return None

        stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        # 使用字典推导式 + 嵌套列表推导式替代嵌套 for 循环
        data: dict = {
            stat: [_compute_stat(vals, stat) for vals in group_vals] for stat in stats
        }
        data["stat"] = stats
        return DataFrame(data)

    def cumsum(self) -> Series:
        """分组累加和。"""
        result = [None] * len(self._s)
        for items in self._groups.values():
            cum = 0
            for i, v in items:
                if v is not None:
                    cum += v
                result[i] = cum
        return Series(result, name=self._s.name, index=self._s._index)

    def cumprod(self) -> Series:
        """分组累乘积。"""
        result = [None] * len(self._s)
        for items in self._groups.values():
            cum = 1
            for i, v in items:
                if v is not None:
                    cum *= v
                result[i] = cum
        return Series(result, name=self._s.name, index=self._s._index)

    def cummax(self) -> Series:
        """分组累最大值。"""
        result = [None] * len(self._s)
        for items in self._groups.values():
            cur_max = None
            for i, v in items:
                if v is not None:
                    cur_max = v if cur_max is None else max(cur_max, v)
                result[i] = cur_max
        return Series(result, name=self._s.name, index=self._s._index)

    def cummin(self) -> Series:
        """分组累最小值。"""
        result = [None] * len(self._s)
        for items in self._groups.values():
            cur_min = None
            for i, v in items:
                if v is not None:
                    cur_min = v if cur_min is None else min(cur_min, v)
                result[i] = cur_min
        return Series(result, name=self._s.name, index=self._s._index)

    def diff(self, periods: int = 1) -> Series:
        """分组差分。

        :param periods: 差分周期
        """
        result = [None] * len(self._s)
        for items in self._groups.values():
            for j, (i, v) in enumerate(items):
                if j >= periods and v is not None and items[j - periods][1] is not None:
                    result[i] = v - items[j - periods][1]
        return Series(result, name=self._s.name, index=self._s._index)

    def shift(self, periods: int = 1, fill_value=None) -> Series:
        """分组位移。

        :param periods: 位移周期
        :param fill_value: 填充值
        """
        result = [None] * len(self._s)
        for items in self._groups.values():
            for j, (i, v) in enumerate(items):
                if j >= periods:
                    result[i] = items[j - periods][1]
                elif fill_value is not None:
                    result[i] = fill_value
        return Series(result, name=self._s.name, index=self._s._index)

    def fillna(self, value=None, method=None, limit=None) -> Series:
        """分组填充缺失值。

        :param value: 填充值
        :param method: 填充方法 ('ffill'/'bfill')
        :param limit: 最大填充数
        """
        result = [None] * len(self._s)
        for items in self._groups.values():
            if method == "ffill":
                last_valid = None
                fill_count = 0
                for i, v in items:
                    if v is not None:
                        result[i] = v
                        last_valid = v
                        fill_count = 0
                    elif last_valid is not None and (
                        limit is None or fill_count < limit
                    ):
                        result[i] = last_valid
                        fill_count += 1
            elif method == "bfill":
                next_valid = None
                fill_count = 0
                for i, v in reversed(items):
                    if v is not None:
                        result[i] = v
                        next_valid = v
                        fill_count = 0
                    elif next_valid is not None and (
                        limit is None or fill_count < limit
                    ):
                        result[i] = next_valid
                        fill_count += 1
            else:
                for i, v in items:
                    result[i] = value if v is None else v
        return Series(result, name=self._s.name, index=self._s._index)

    def ffill(self, limit=None) -> Series:
        """分组前向填充。"""
        return self.fillna(method="ffill", limit=limit)

    def bfill(self, limit=None) -> Series:
        """分组后向填充。"""
        return self.fillna(method="bfill", limit=limit)

    def head(self, n: int = 5) -> Series:
        """返回每个分组的前 n 个值。

        :param n: 值数量
        """
        keep_indices = []
        for items in self._groups.values():
            keep_indices.extend(i for i, _ in items[:n])
        return self._s.iloc[keep_indices]

    def tail(self, n: int = 5) -> Series:
        """返回每个分组的后 n 个值。

        :param n: 值数量
        """
        keep_indices = []
        for items in self._groups.values():
            if n <= len(items):
                keep_indices.extend(i for i, _ in items[-n:])
            else:
                keep_indices.extend(i for i, _ in items)
        return self._s.iloc[keep_indices]

    def get_group(self, name) -> Series:
        """获取指定分组的数据。

        :param name: 分组键
        """
        items = self._groups.get(name, [])
        return Series([v for _, v in items], name=self._s.name)

    @property
    def groups(self) -> dict:
        """返回 {分组键: [(索引, 值)]} 字典。"""
        return dict(self._groups)

    @property
    def indices(self) -> dict:
        """返回 {分组键: [位置索引]} 字典。"""
        return {k: [i for i, _ in v] for k, v in self._groups.items()}

    def rank(self, method: str = "average", ascending: bool = True) -> Series:
        """分组内排名。

        :param method: 'average'/'min'/'max'/'first'/'dense'
        :param ascending: 是否升序
        """
        result = [None] * len(self._s)
        for items in self._groups.values():
            vals = [(v, i) for i, v in items if v is not None]
            if not vals:
                continue
            vals.sort(key=lambda x: x[0], reverse=not ascending)
            if method == "first":
                for j, (v, i) in enumerate(vals):
                    result[i] = j + 1
            elif method == "dense":
                rank = 0
                prev = None
                for v, i in vals:
                    if prev is None or v != prev:
                        rank += 1
                    result[i] = rank
                    prev = v
            else:
                grouped = {}
                for v, i in vals:
                    grouped.setdefault(v, []).append(i)
                if method == "min":
                    for rank, (v, idxs) in enumerate(grouped.items(), 1):
                        for i in idxs:
                            result[i] = rank
                elif method == "max":
                    rank = 0
                    for v, idxs in grouped.items():
                        rank += len(idxs)
                        for i in idxs:
                            result[i] = rank
                else:
                    pos = 1
                    for v, idxs in grouped.items():
                        avg = pos + (len(idxs) - 1) / 2.0
                        for i in idxs:
                            result[i] = avg
                        pos += len(idxs)
        return Series(result, name=self._s.name, index=self._s._index)

    def corr(self, other: Series) -> Series:
        """计算每个分组内与另一个 Series 的相关系数。

        :param other: 另一个 Series
        """
        result = {}
        for key, items in self._groups.items():
            pairs = [
                (v, other.values[i])
                for i, v in items
                if v is not None
                and i < len(other.values)
                and other.values[i] is not None
            ]
            if len(pairs) < 2:
                result[key] = None
                continue
            ma = sum(a for a, b in pairs) / len(pairs)
            mb = sum(b for a, b in pairs) / len(pairs)
            num = sum((a - ma) * (b - mb) for a, b in pairs)
            da = (sum((a - ma) ** 2 for a, b in pairs)) ** 0.5
            db = (sum((b - mb) ** 2 for a, b in pairs)) ** 0.5
            result[key] = num / (da * db) if da > 0 and db > 0 else None
        keys = list(result.keys())
        if self._sort:
            try:
                keys.sort()
            except TypeError:
                pass
        return Series([result[k] for k in keys], index=keys, name=self._s.name)

    def cov(self, other: Series) -> Series:
        """计算每个分组内与另一个 Series 的协方差。

        :param other: 另一个 Series
        """
        result = {}
        for key, items in self._groups.items():
            pairs = [
                (v, other.values[i])
                for i, v in items
                if v is not None
                and i < len(other.values)
                and other.values[i] is not None
            ]
            if len(pairs) < 2:
                result[key] = None
                continue
            ma = sum(a for a, b in pairs) / len(pairs)
            mb = sum(b for a, b in pairs) / len(pairs)
            result[key] = sum((a - ma) * (b - mb) for a, b in pairs) / (len(pairs) - 1)
        keys = list(result.keys())
        if self._sort:
            try:
                keys.sort()
            except TypeError:
                pass
        return Series([result[k] for k in keys], index=keys, name=self._s.name)

    def filter(self, func, *args, **kwargs) -> Series:
        """过滤分组，保留满足条件的组。

        :param func: 接收 Series 返回 bool 的函数
        """
        keep_indices = []
        for key, items in self._groups.items():
            group_series = Series([v for _, v in items], name=self._s.name)
            if func(group_series, *args, **kwargs):
                keep_indices.extend(i for i, _ in items)
        return self._s.iloc[keep_indices]

    def _agg_single(self, func, axis: int = 0, *args, **kwargs) -> Series:
        """单函数聚合（原始 agg 逻辑）。"""
        import math

        # 优先调用 Rust 层 groupby_agg_series（仅支持内置聚合名，不支持 callable）
        if isinstance(func, str) and not args and not kwargs:
            try:
                # 构造分组键字符串列表（按原 Series 顺序）
                values = self._s.values
                keys_list = [self._by_key(i, v) for i, v in enumerate(values)]
                by_str = [k if k is not None else "" for k in keys_list]
                rust_keys, rust_vals = self._s._inner.groupby_agg_series(by_str, func)
                rust_keys_list = list(rust_keys)
                rust_vals_list = [(v if v is not None else None) for v in rust_vals]
                return Series(rust_vals_list, index=rust_keys_list, name=self._s.name)
            except Exception:
                pass

        result = {}
        for key, items in self._groups.items():
            vals = [v for _, v in items if v is not None]
            if vals:
                if callable(func):
                    result[key] = func(vals, *args, **kwargs)
                elif func == "sum":
                    result[key] = sum(vals)
                elif func == "mean":
                    result[key] = sum(vals) / len(vals)
                elif func == "min":
                    result[key] = min(vals)
                elif func == "max":
                    result[key] = max(vals)
                elif func == "count":
                    result[key] = len(vals)
                elif func == "std":
                    ddof = kwargs.pop("ddof", 1)
                    n = len(vals)
                    if n > ddof:
                        m = sum(vals) / n
                        variance = sum((x - m) ** 2 for x in vals) / (n - ddof)
                        result[key] = variance**0.5
                    else:
                        result[key] = None
                elif func == "var":
                    ddof = kwargs.pop("ddof", 1)
                    n = len(vals)
                    if n > ddof:
                        m = sum(vals) / n
                        variance = sum((x - m) ** 2 for x in vals) / (n - ddof)
                        result[key] = variance
                    else:
                        result[key] = None
                elif func == "sem":
                    ddof = kwargs.pop("ddof", 1)
                    n = len(vals)
                    if n > ddof:
                        m = sum(vals) / n
                        variance = sum((x - m) ** 2 for x in vals) / (n - ddof)
                        result[key] = (variance / n) ** 0.5
                    else:
                        result[key] = None
                elif func == "median":
                    sv = sorted(vals)
                    n = len(sv)
                    result[key] = (
                        sv[n // 2] if n % 2 == 1 else (sv[n // 2 - 1] + sv[n // 2]) / 2
                    )
                elif func == "prod":
                    result[key] = math.prod(vals)
                elif func == "first":
                    result[key] = vals[0]
                elif func == "last":
                    result[key] = vals[-1]
                else:
                    result[key] = None
            else:
                result[key] = None

        keys = list(result.keys())
        if self._sort:
            try:
                keys.sort()
            except TypeError:
                pass
        values_list = [result[k] for k in keys]
        return Series(values_list, index=keys, name=self._s.name)

    def apply(self, func, *args, **kwargs) -> Series:
        """应用函数到每个分组。"""
        result = {}
        for key, items in self._groups.items():
            group_series = Series([v for _, v in items], name=self._s.name)
            result[key] = func(group_series, *args, **kwargs)
        keys = list(result.keys())
        if self._sort:
            try:
                keys.sort()
            except TypeError:
                pass
        values_list = [result[k] for k in keys]
        return Series(values_list, index=keys, name=self._s.name)

    def sum(self) -> Series:
        """分组求和。"""
        return self._agg_single("sum")

    def mean(self) -> Series:
        """分组求均值。"""
        return self._agg_single("mean")

    def min(self) -> Series:
        """分组求最小值。"""
        return self._agg_single("min")

    def max(self) -> Series:
        """分组求最大值。"""
        return self._agg_single("max")

    def count(self) -> Series:
        """分组计数。"""
        return self._agg_single("count")

    def transform(self, func, axis: int = 0, *args, **kwargs) -> Series:
        """对每个分组应用函数并返回原始长度的 Series。

        :param func: 可调用函数，或聚合函数名字符串 ('sum'/'mean' 等)
        :param axis: 轴 (未使用，保持兼容性)
        :param args: 传递给 func 的额外位置参数
        :param kwargs: 传递给 func 的关键字参数
        """
        # 字符串形式：映射到 _agg_single 的内置聚合名，取聚合结果做广播
        if isinstance(func, str):
            agg_result = self._agg_single(func, *args, **kwargs)
            agg_map = dict(zip(agg_result._index or [], agg_result.values))
            result = [None] * len(self._s)
            for key, items in self._groups.items():
                val = agg_map.get(key, None)
                for i, _ in items:
                    result[i] = val
            return Series(result, name=self._s.name, index=self._s._index)

        result = [None] * len(self._s)
        for key, items in self._groups.items():
            group_series = Series([v for _, v in items], name=self._s.name)
            transformed = func(group_series, *args, **kwargs)
            if isinstance(transformed, Series):
                for (i, _), tv in zip(items, transformed.values):
                    result[i] = tv
            else:
                for i, _ in items:
                    result[i] = transformed
        return Series(result, name=self._s.name, index=self._s._index)

    def quantile(self, q=0.5) -> Series:
        """分组分位数。

        :param q: 分位数值 (0.0-1.0) 或列表
        """
        from ..dataframe import DataFrame

        if isinstance(q, (list, tuple)):
            # 多分位数 -> 返回 DataFrame
            result_rows = {}
            for q_val in q:
                result_rows[q_val] = [
                    self._quantile_for_group(items, q_val)
                    for items in self._groups.values()
                ]
            keys = self._sorted_keys(self._groups.keys())
            df = DataFrame(result_rows)
            df = df.T
            df._columns = list(keys)
            df._index = list(q)
            return df
        return self._build_grouped_series(
            lambda _, items: self._quantile_for_group(items, q)
        )

    @staticmethod
    def _quantile_for_group(items, q):
        """单组分位数计算。"""
        vals = sorted(v for _, v in items if v is not None)
        if not vals:
            return None
        n = len(vals)
        if n == 1:
            return vals[0]
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return vals[lo] + (vals[hi] - vals[lo]) * frac

    def skew(self) -> Series:
        """分组偏度。"""

        def _skew(key, items):
            vals = [v for _, v in items if v is not None]
            n = len(vals)
            if n < 3:
                return None
            m = sum(vals) / n
            s = (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5
            if s == 0:
                return None
            return sum((x - m) ** 3 for x in vals) / ((n - 1) * s**3)

        return self._build_grouped_series(_skew)

    def kurt(self) -> Series:
        """分组峰度（excess kurtosis）。"""

        def _kurt(key, items):
            vals = [v for _, v in items if v is not None]
            n = len(vals)
            if n < 4:
                return None
            m = sum(vals) / n
            s2 = sum((x - m) ** 2 for x in vals) / (n - 1)
            if s2 == 0:
                return None
            s4 = sum((x - m) ** 4 for x in vals) / (n - 1)
            return s4 / (s2**2) - 3

        return self._build_grouped_series(_kurt)

    def mad(self) -> Series:
        """分组平均绝对偏差。"""

        def _mad(key, items):
            vals = [v for _, v in items if v is not None]
            if not vals:
                return None
            m = sum(vals) / len(vals)
            return sum(abs(x - m) for x in vals) / len(vals)

        return self._build_grouped_series(_mad)

    def ngroup(self, ascending: bool = True) -> Series:
        """返回每个元素所属的分组编号 (0-based)。"""
        n = len(self._s)
        result = [None] * n
        if ascending:
            id_map = {
                k: i for i, k in enumerate(self._sorted_keys(self._groups.keys()))
            }
        else:
            id_map = {
                k: i
                for i, k in enumerate(reversed(self._sorted_keys(self._groups.keys())))
            }
        for key, items in self._groups.items():
            gid = id_map[key]
            for i, _ in items:
                result[i] = gid
        return Series(result, name=self._s.name, index=self._s._index, dtype="int64")

    def cumcount(self, ascending: bool = True) -> Series:
        """返回每个分组内的累计计数 (0-based)。"""
        n = len(self._s)
        result = [None] * n
        for items in self._groups.values():
            if ascending:
                for j, (i, _) in enumerate(items):
                    result[i] = j
            else:
                m = len(items)
                for j, (i, _) in enumerate(items):
                    result[i] = m - j - 1
        return Series(result, name=self._s.name, index=self._s._index, dtype="int64")

    def agg(self, func, axis: int = 0, *args, **kwargs) -> Series:
        """聚合操作（扩展支持 list/dict 多函数）。

        :param func: 聚合函数/名，或函数名列表，或 {新列名: 函数} 字典
        :param axis: 轴 (未使用，保持兼容性)
        """
        from ..dataframe import DataFrame

        # 单函数/名单函数调用 -> 保持原始行为，返回 Series
        if (
            isinstance(func, (str,))
            or callable(func)
            and not isinstance(func, (list, tuple, dict))
        ):
            return self._agg_single(func, *args, **kwargs)
        if isinstance(func, dict):
            # dict: {新列名: 聚合函数名} -> 返回 DataFrame
            result: Dict[str, list] = {}
            keys = self._sorted_keys(self._groups.keys())
            for col_name, f in func.items():
                series_result = self._agg_single(f, *args, **kwargs)
                result[col_name] = [
                    series_result[k] if k in (series_result._index or []) else None
                    for k in keys
                ]
            df = DataFrame(result)
            df._index = list(keys)
            return df
        if isinstance(func, (list, tuple)):
            # list: [函数名1, 函数名2] -> 返回 MultiIndex 风格 DataFrame
            result: Dict[str, list] = {}
            keys = self._sorted_keys(self._groups.keys())
            for fname in func:
                series_result = self._agg_single(fname, *args, **kwargs)
                result[str(fname)] = [
                    series_result[k] if k in (series_result._index or []) else None
                    for k in keys
                ]
            df = DataFrame(result)
            df._index = list(keys)
            return df
        raise TypeError(f"agg: unsupported func type {type(func).__name__}")

    aggregate = agg


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
