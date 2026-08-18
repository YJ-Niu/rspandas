"""分箱与列联表工具函数 (v1.3.0)。

包含 cut / qcut / crosstab / get_dummies 等与 pandas 兼容的工具函数。
"""

from __future__ import annotations

from ..rspandas import _DataFrame as rspandas_DataFrame  # type: ignore
from ..series import Series
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

# ============================================================================
# 工具函数: get_dummies / cut / qcut / crosstab
# ============================================================================


def get_dummies(
    data,
    prefix: Optional[Union[str, List[str]]] = None,
    prefix_sep: str = "_",
    columns: Optional[List[str]] = None,
) -> rspandas_DataFrame:
    """将分类变量转换为哑变量（one-hot 编码）。

    Parameters
    ----------
    data : Series or DataFrame
        输入数据。
    prefix : str or list[str], optional
        列名前缀。
    prefix_sep : str, default '_'
        前缀与类别之间的分隔符。
    columns : list[str], optional
        如果是 DataFrame，指定要转换的列。

    Returns
    -------
    DataFrame
    """
    from ..dataframe import DataFrame as _DataFrame

    if isinstance(data, Series):
        values = data.values
        name = data.name or "col"
        return _get_dummies_series(values, prefix or name, prefix_sep)

    if isinstance(data, _DataFrame):
        if columns is None:
            # 使用列表推导式替代显式 for 循环收集待转换列
            columns = [
                c
                for c in data.columns
                if data[c].dtype in ("object", "category", "bool")
            ]

        # 为每列构建哑变量 DataFrame，再用 _concat_frames 一次性拼接
        def _build_dummies_for_col(col_idx, col):
            """为单列构建哑变量 DataFrame。"""
            actual_prefix = (
                prefix
                if isinstance(prefix, str)
                else (prefix[col_idx] if prefix else col)
            )
            return _get_dummies_series(
                list(data[col].values), prefix=actual_prefix, sep=prefix_sep
            )

        all_dummies = [_build_dummies_for_col(i, c) for i, c in enumerate(columns)]
        return _concat_frames([data] + all_dummies) if all_dummies else data

    raise TypeError(
        f"get_dummies expected Series or DataFrame, got {type(data).__name__}"
    )


def _get_dummies_series(values: list, prefix: str, sep: str) -> rspandas_DataFrame:
    """对单个 Series 做 one-hot 编码。"""
    from ..dataframe import DataFrame as _DataFrame

    # 获取唯一值（使用 dict.fromkeys 保序去重，过滤 None）
    unique_vals = list(dict.fromkeys(v for v in values if v is not None))

    # 使用字典推导式一次性生成所有列（每列为该值的 one-hot 编码）
    result_data = {
        f"{prefix}{sep}{uv}": [1 if v == uv else 0 for v in values]
        for uv in unique_vals
    }

    return _DataFrame(result_data)


def _concat_frames(frames: list) -> rspandas_DataFrame:
    """横向拼接 DataFrame，去掉重复列。"""
    from ..dataframe import DataFrame as _DataFrame

    if not frames:
        return _DataFrame({})
    if len(frames) == 1:
        return frames[0]

    # 保留每个列名首次出现的列数据 - 使用普通 dict 按 key 去重（Python 3.7+ dict 保持插入顺序）
    result_data: Dict[str, list] = {}
    for df in frames:
        # 使用 update 仅插入尚未存在的 key（用 dict 推导式过滤）
        new_cols = {
            col: list(df[col].values) for col in df.columns if col not in result_data
        }
        result_data.update(new_cols)

    return _DataFrame(result_data)


def cut(
    x,
    bins: Union[int, list],
    right: bool = True,
    labels=None,
    include_lowest: bool = False,
) -> Series:
    """将连续值分割为离散区间。

    Parameters
    ----------
    x : list or Series
        输入数据。
    bins : int or list
        区间数或区间边界。
    right : bool, default True
        区间是否右闭。
    labels : list, optional
        区间标签。
    include_lowest : bool, default False
        第一个区间是否包含最小值。

    Returns
    -------
    Series
    """
    import math

    values = list(x.values) if isinstance(x, Series) else list(x)

    # 过滤缺失值
    def _is_missing(v) -> bool:
        if v is None:
            return True
        try:
            return v != v  # type: ignore[operator]
        except TypeError:
            return False

    non_null = [v for v in values if not _is_missing(v)]
    if not non_null:
        return Series([None] * len(values), dtype="category")

    # 计算 bins
    if isinstance(bins, int):
        min_val = min(non_null)
        max_val = max(non_null)
        if min_val == max_val:
            # 只有一个唯一值，构造一个包含它的区间
            bins = [min_val - 0.5, min_val + 0.5]
        else:
            bin_width = (max_val - min_val) / bins
            bins = [min_val + i * bin_width for i in range(bins + 1)]
            # 修正浮点精度: 确保最后一个边界等于 max_val
            bins[-1] = max_val
        # 当 bins 为 int 时，自动包含最小值 (与 pandas 一致)
        include_lowest = True
    else:
        bins = list(bins)

    n_bins = len(bins) - 1

    # 格式化边界值用于标签
    def _fmt(v) -> str:
        """格式化数值为字符串。"""
        if v == float("inf"):
            return "inf"
        if v == float("-inf"):
            return "-inf"
        # 整数边界用整数格式
        if isinstance(v, (int,)) or (isinstance(v, float) and v.is_integer()):
            return str(int(v))
        return repr(v)

    # 生成标签
    if labels is None:
        if right:
            labels = [f"({_fmt(bins[i])}, {_fmt(bins[i + 1])}]" for i in range(n_bins)]
            if include_lowest:
                labels[0] = f"[{_fmt(bins[0])}, {_fmt(bins[1])}]"
        else:
            labels = [f"[{_fmt(bins[i])}, {_fmt(bins[i + 1])})" for i in range(n_bins)]
            if include_lowest:
                labels[-1] = f"[{_fmt(bins[-2])}, {_fmt(bins[-1])}]"
    else:
        labels = list(labels)

    # 分配区间
    def _find_bin(v):
        """返回 v 所在区间的标签，未匹配则返回 None。"""
        if _is_missing(v):
            return None
        # 处理 inf 值
        is_inf = math.isinf(v) if isinstance(v, float) else False

        for i in range(n_bins):
            lo = bins[i]
            hi = bins[i + 1]
            if right:
                # 区间 (lo, hi]，第一个区间可能改为 [lo, hi]
                if i == 0 and include_lowest:
                    if lo <= v <= hi:
                        return labels[i]
                else:
                    if lo < v <= hi:
                        return labels[i]
            else:
                # 区间 [lo, hi)，最后一个区间可能改为 [lo, hi]
                if i == n_bins - 1 and include_lowest:
                    if lo <= v <= hi:
                        return labels[i]
                else:
                    if lo <= v < hi:
                        return labels[i]

        # 浮点精度兜底: 最大值/最小值可能因精度问题未匹配
        if not is_inf:
            if v <= bins[0]:
                return labels[0]
            if v >= bins[-1]:
                return labels[-1]
        return None

    result = [_find_bin(v) for v in values]

    return Series(result, dtype="category")


def qcut(
    x,
    q: Union[int, list],
    labels=None,
) -> Series:
    """基于分位数将连续值分割为离散区间。

    Parameters
    ----------
    x : list or Series
        输入数据。
    q : int or list
        分位数数量或分位点列表 (0-1)。
    labels : list, optional
        区间标签。

    Returns
    -------
    Series
    """
    values = list(x.values) if isinstance(x, Series) else list(x)

    # 过滤缺失值
    def _is_missing(v) -> bool:
        if v is None:
            return True
        try:
            return v != v  # type: ignore[operator]
        except TypeError:
            return False

    non_null = sorted([v for v in values if not _is_missing(v)])
    if not non_null:
        return Series([None] * len(values), dtype="category")

    n = len(non_null)

    if isinstance(q, int):
        n_bins = q

        # 计算分位点
        def _quantile_at(i):
            pos = i * (n - 1) / n_bins
            lo = int(pos)
            hi = min(lo + 1, n - 1)
            frac = pos - lo
            return non_null[lo] + frac * (non_null[hi] - non_null[lo])

        quantiles = [_quantile_at(i) for i in range(n_bins + 1)]
    else:
        # q 是分位数列表 (如 [0, 0.25, 0.5, 0.75, 1])
        # 需要计算每个分位数对应的实际值
        q_list = sorted(q)

        def _quantile_at(p):
            """计算分位数 p 对应的实际值 (线性插值法)。"""
            if p <= 0:
                return non_null[0]
            if p >= 1:
                return non_null[-1]
            pos = p * (n - 1)
            lo = int(pos)
            hi = min(lo + 1, n - 1)
            frac = pos - lo
            return non_null[lo] + frac * (non_null[hi] - non_null[lo])

        quantiles = [_quantile_at(p) for p in q_list]
        n_bins = len(quantiles) - 1

    # 使用 cut 进行分箱，包含最低值
    return cut(values, bins=quantiles, right=True, labels=labels, include_lowest=True)


def crosstab(
    index,
    columns,
    values=None,
    aggfunc: str = "count",
    rownames=None,
    colnames=None,
    margins: bool = False,
    normalize: Union[bool, str] = False,
) -> rspandas_DataFrame:
    """计算交叉表。

    Parameters
    ----------
    index : list or Series
        行分组变量。
    columns : list or Series
        列分组变量。
    values : list or Series, optional
        聚合的值。
    aggfunc : str, default 'count'
        聚合函数：'count', 'sum', 'mean', 'min', 'max'。
    rownames : list, optional
        行名称。
    colnames : list, optional
        列名称。
    margins : bool, default False
        是否添加边际汇总。
    normalize : bool or str, default False
        True/'all' 归一化所有值，'index' 按行归一化，'columns' 按列归一化。

    Returns
    -------
    DataFrame
    """
    from ..dataframe import DataFrame as _DataFrame

    idx_vals = list(index.values) if isinstance(index, Series) else list(index)
    col_vals = list(columns.values) if isinstance(columns, Series) else list(columns)

    if values is not None:
        val_vals = list(values.values) if isinstance(values, Series) else list(values)
    else:
        val_vals = [1] * len(idx_vals)

    # 收集所有 index 和 column 值（使用 dict.fromkeys 保序去重）
    unique_idx = list(dict.fromkeys(idx_vals))
    unique_col = list(dict.fromkeys(col_vals))

    # 构建交叉表 - 使用 dict.setdefault 简化分组
    groups: Dict[tuple, list] = {}
    for i, (iv, cv) in enumerate(zip(idx_vals, col_vals)):
        groups.setdefault((iv, cv), []).append(val_vals[i])

    # 聚合
    def _agg(vals):
        if not vals:
            return 0
        if aggfunc == "count":
            return len(vals)
        nums = [v for v in vals if v is not None]
        if not nums:
            return 0
        if aggfunc == "sum":
            return sum(nums)
        if aggfunc == "mean":
            return sum(nums) / len(nums)
        if aggfunc == "min":
            return min(nums)
        if aggfunc == "max":
            return max(nums)
        return 0

    # 构建结果（使用字典推导式 + 嵌套列表推导式替代显式 for 循环）
    result_data: Dict[str, list] = {
        "": unique_idx,  # 行索引列
    }
    result_data.update(
        {
            str(cv): [_agg(groups.get((iv, cv), [])) for iv in unique_idx]
            for cv in unique_col
        }
    )

    df = _DataFrame(result_data)

    # 边际汇总
    if margins:
        # 行汇总：每行所有列的聚合值之和
        row_sums = [
            sum(_agg(groups.get((iv, cv), [])) for cv in unique_col)
            for iv in unique_idx
        ]
        df["All"] = row_sums

        # 列汇总：每列所有行的聚合值之和
        col_sums = (
            [""]
            + [
                sum(_agg(groups.get((iv, cv), [])) for iv in unique_idx)
                for cv in unique_col
            ]
            + [sum(row_sums)]
        )

        # 添加汇总行
        df_data = {
            c: list(df[c].values) + [col_sums[i]] for i, c in enumerate(df.columns)
        }
        df = _DataFrame(df_data)

    # 归一化
    if normalize is True or normalize == "all":
        total = sum(
            sum(v for v in df[c].values if v is not None) for c in df.columns if c != ""
        )
        if total > 0:
            # 使用字典推导式批量归一化所有列
            df_data = {
                c: (
                    [v / total for v in df[c].values] if c != "" else list(df[c].values)
                )
                for c in df.columns
            }
            df = _DataFrame(df_data)
    elif normalize == "index":
        # 按行归一化：每行总和为 1
        row_totals = [
            sum(df[c].values[i] for c in df.columns if c != "") for i in range(len(df))
        ]
        df_data = {
            c: (
                [
                    df[c].values[i] / row_totals[i] if row_totals[i] > 0 else 0
                    for i in range(len(df))
                ]
                if c != ""
                else list(df[c].values)
            )
            for c in df.columns
        }
        df = _DataFrame(df_data)
    elif normalize == "columns":
        # 按列归一化：每列总和为 1
        col_totals = {
            c: sum(v for v in df[c].values if v is not None)
            for c in df.columns
            if c != ""
        }
        df_data = {
            c: (
                [v / col_totals[c] if col_totals[c] > 0 else 0 for v in df[c].values]
                if c != ""
                else list(df[c].values)
            )
            for c in df.columns
        }
        df = _DataFrame(df_data)

    return df
