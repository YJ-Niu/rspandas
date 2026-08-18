"""DataFrameGroupBy DataFrame 分组

由 rspandas/dataframe.py 拆分而来。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from ..dataframe import DataFrame
    from ..series import Series


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


class DataFrameGroupBy:
    """DataFrame 分组结果 (极简版)。"""

    def __init__(
        self,
        df: "DataFrame",
        by,
        as_index: bool = True,
        sort: bool = True,
        dropna: bool = True,
        observed: bool = False,
    ):
        if isinstance(by, str):
            self._by = [by]
        else:
            self._by = list(by) if by is not None else []
        self._df = df
        self._as_index = as_index
        self._sort = sort
        self._dropna = dropna
        self._observed = observed

        # 分组: { key_tuple: [row_indices] } （优化：预取列 + 列表推导式提取 key）
        self._groups: Dict[tuple, list] = {}
        n = df._nrows
        by_cols = [df._inner.get_column(c).values for c in self._by]

        for i in range(n):
            key = tuple(col[i] for col in by_cols)
            # dropna=True 时丢弃含 None 的键；dropna=False 保留所有键
            if dropna and any(v is None for v in key):
                continue
            self._groups.setdefault(key, []).append(i)

        # 排序组
        if sort:
            try:
                self._groups = dict(sorted(self._groups.items()))
            except TypeError:
                # 不可排序键类型（含 None/混合类型）保持原顺序
                pass

    def __getitem__(self, key):
        """按列名或列名列表选择分组后的列。"""
        if isinstance(key, str):
            # 返回 SeriesGroupBy
            from ..series import SeriesGroupBy

            return SeriesGroupBy(self._df, self._by, key)
        elif isinstance(key, list):
            # 返回仅包含指定列的新 DataFrameGroupBy
            # 需要包含分组键列，因为它们需要用于分组
            all_cols = self._by + [c for c in key if c not in self._by]
            new_df = self._df[all_cols]
            return DataFrameGroupBy(
                new_df,
                self._by,
                self._as_index,
                self._sort,
                self._dropna,
                observed=self._observed,
            )
        else:
            raise TypeError(f"GroupBy 不支持的 key 类型: {type(key).__name__}")

    def _agg(self, agg_funcs: Dict[str, str]) -> "DataFrame":
        """对每列应用聚合函数。

        :param agg_funcs: {列名: 'sum' | 'mean' | 'min' | 'max' | 'count' | 'std' | 'var' | 'median' | 'first' | 'last'}

        分组键作为 index 输出
        """
        agg_cols = list(agg_funcs.keys())
        group_keys = list(self._groups.keys())
        # 优先调用 Rust 层（仅支持单列 by、所有列同一 agg 且在 Rust 支持列表中）
        if len(self._by) == 1 and len(set(agg_funcs.values())) == 1:
            single_agg = agg_funcs[agg_cols[0]]
            if single_agg in ("sum", "mean", "count", "min", "max"):
                by_col = self._by[0]
                by_idx = self._df._columns.index(by_col)
                # 检查 by 列是否有 None 值（Rust 层 None 处理为空字符串，行为不一致）
                by_values = self._df._inner.get_column(by_col).values
                has_none = any(v is None for v in by_values)
                if not has_none:
                    try:
                        rust_keys, rust_df = self._df._inner.groupby_agg(
                            by_idx, single_agg
                        )
                        # 构建 {rust_key: row_idx} 映射
                        rust_key_to_idx = {k: i for i, k in enumerate(rust_keys)}
                        # 按 self._groups 的顺序提取行（保持 sort/dropna 行为一致）
                        ordered_indices = []
                        for key_tuple in group_keys:
                            key_str = str(key_tuple[0])
                            if key_str in rust_key_to_idx:
                                ordered_indices.append(rust_key_to_idx[key_str])
                        # 所有组都找到时按顺序提取结果
                        if len(ordered_indices) == len(group_keys):
                            new_data = {
                                **{c: [] for c in agg_cols},
                            }
                            for c in agg_cols:
                                col_vals = rust_df.get_column(c).values
                                new_data[c] = [col_vals[i] for i in ordered_indices]
                            out = DataFrame(new_data)
                            if self._as_index:
                                out._index = [k[0] for k in group_keys]
                            return out
                    except Exception:
                        pass

        # 回退到 Python 实现
        result: Dict[str, list] = {c: [] for c in agg_cols}
        if not self._as_index:
            for i, by_col in enumerate(self._by):
                result[by_col] = []

        for key in group_keys:
            idxs = self._groups[key]
            if not self._as_index:
                if isinstance(key, tuple):
                    for i, by_col in enumerate(self._by):
                        result[by_col].append(key[i])
                else:
                    result[self._by[0]].append(key)
            for c in agg_cols:
                ser = self._df[c]
                sub = ser.iloc[idxs]
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
        out = DataFrame(result)
        if self._as_index and len(self._by) > 0:
            if len(self._by) == 1:
                out._index = [k[0] if isinstance(k, tuple) else k for k in group_keys]
            else:
                out._index = list(group_keys)
        return out

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
        """通用聚合: 支持 str / dict[列名->str] / list[func]。"""
        if isinstance(func, str):
            return self._agg({c: func for c in self._df._columns if c not in self._by})
        if isinstance(func, dict):
            # 允许 dict 值为 list: {col: [func1, func2]} -> 展开为 MultiIndex 风格列
            flat_funcs: Dict[str, str] = {}
            multi_mode = False
            for col, f in func.items():
                if isinstance(f, (list, tuple)):
                    multi_mode = True
                    for single_f in f:
                        flat_funcs[f"{col}_{single_f}"] = single_f
                else:
                    flat_funcs[col] = f
            if multi_mode:
                return self._agg(flat_funcs)
            return self._agg(func)
        if isinstance(func, (list, tuple)):
            # 所有列都应用这些函数
            result_parts: Dict[str, "DataFrame"] = {}
            other_cols = [c for c in self._df._columns if c not in self._by]
            for fname in func:
                sub = self._agg({c: fname for c in other_cols})
                # 重命名列以避免冲突
                rename_map = {c: f"{c}_{fname}" for c in other_cols}
                sub = sub.rename(columns=rename_map)
                result_parts[str(fname)] = sub
            # 横向拼接：取第一个 DataFrame 为基准，追加其他列
            if not result_parts:
                return DataFrame()
            it = iter(result_parts.values())
            merged = next(it)
            for rest_df in it:
                # 简单按列顺序横向合并
                for col in rest_df._columns:
                    merged[col] = rest_df[col].to_list()
            return merged
        raise TypeError("agg must be str, dict or list")

    aggregate = agg

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
                sub_vals = ser.iloc[idxs].values
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
        from ..series import Series

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
        from ..series import Series

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
                vals = [v for v in self._df[c].iloc[idxs].values if v is not None]
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
        from ..series import Series

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
        from ..series import Rolling

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
        from ..series import Expanding

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
        from ..series import EWM

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
        from ..series import Resampler

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
                vals = [v for v in self._df[c].iloc[idxs].values if v is not None]
                if not vals:
                    result[c].append(None)
                    continue
                m = sum(vals) / len(vals)
                result[c].append(sum(abs(x - m) for x in vals) / len(vals))
        return DataFrame(result)

    def prod(self) -> "DataFrame":
        """分组乘积（math.prod 优化版）。"""
        import math

        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {c: [] for c in other_cols}

        for key, idxs in self._groups.items():
            for c in other_cols:
                vals = [v for v in self._df[c].iloc[idxs].values if v is not None]
                result[c].append(math.prod(vals) if vals else None)
        return DataFrame(result)

    def skew(self) -> "DataFrame":
        """分组偏度。"""
        other_cols = [c for c in self._df._columns if c not in self._by]
        result: Dict[str, list] = {}
        for c in other_cols:
            result[c] = []

        for key, idxs in self._groups.items():
            for c in other_cols:
                vals = [v for v in self._df[c].iloc[idxs].values if v is not None]
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
                vals = [v for v in self._df[c].iloc[idxs].values if v is not None]
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
                vals = [v for v in self._df[c].iloc[idxs].values if v is not None]
                result[c].append(len(set(vals)))
        return DataFrame(result)

    def size(self) -> "Series":
        """返回每个分组的大小。"""
        from ..series import Series

        keys = list(self._groups.keys())
        sizes = [len(idxs) for idxs in self._groups.values()]

        # 当 observed=False 时，包含所有类别（即使计数为 0）
        if not self._observed:
            for by_col in self._by:
                # 检查 by 列是否为 category 类型
                if (
                    by_col in self._df._col_dtypes
                    and self._df._col_dtypes[by_col] == "category"
                ):
                    # 获取所有 categories
                    if (
                        by_col in self._df._col_categories
                        and self._df._col_categories[by_col]
                    ):
                        all_cats = self._df._col_categories[by_col]
                        # 为缺失的类别添加 0 计数
                        existing_keys = set(
                            k[0] if isinstance(k, tuple) else k for k in keys
                        )
                        for cat in all_cats:
                            if cat not in existing_keys:
                                keys.append((cat,) if len(self._by) > 1 else cat)
                                sizes.append(0)
                        # 排序
                        if self._sort:
                            sorted_pairs = sorted(
                                zip(keys, sizes),
                                key=lambda x: (
                                    (
                                        all_cats.index(x[0])
                                        if x[0] in all_cats
                                        else len(all_cats)
                                    ),
                                    str(x[0]),
                                ),
                            )
                            keys = [p[0] for p in sorted_pairs]
                            sizes = [p[1] for p in sorted_pairs]
                        break

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
                sub_vals = [v for v in self._df[c].iloc[idxs].values if v is not None]
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
                    for i, v in enumerate(self._df[c].iloc[idxs].values)
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
                    for i, v in enumerate(self._df[c].iloc[idxs].values)
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
                vals = [v for v in self._df[c].iloc[idxs].values if v is not None]
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
