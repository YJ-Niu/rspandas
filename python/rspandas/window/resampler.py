"""Resampler 重采样

由 rspandas/series.py 拆分而来。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from ..dataframe import DataFrame
    from ..series import Series
from typing import Any, Optional, Union


class Resampler:
    """时间序列重采样 (v1.0.0)。

    Examples:
        >>> import rspandas as rpd
        >>> from datetime import datetime
        >>> idx = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
        >>> s = rpd.Series([1, 2, 3], index=idx)
        >>> s.resample('D').sum().values
        [1, 2, 3]
    """

    _FREQ_MAP = {
        "D": "day",
        "W": "week",
        "M": "month",
        "Y": "year",
        "H": "hour",
        "h": "hour",
        "S": "second",
        "s": "second",
        "T": "minute",
        "min": "minute",
        "Min": "minute",
        "m": "minute",
    }

    def __init__(self, series: _PySeries, freq: str, index: list):
        # 解析频率字符串，支持数字前缀如 "5Min"
        self._freq_num = 1
        self._freq_unit = freq

        # 尝试解析数字前缀
        import re

        match = re.match(r"^(\d+)(.*)", freq)
        if match:
            self._freq_num = int(match.group(1))
            self._freq_unit = match.group(2)

        if self._freq_unit not in self._FREQ_MAP:
            raise ValueError(f"unsupported freq: {freq!r}")
        self._s = series
        self._freq = self._freq_unit
        self._index = index
        self._values = series.values

    def _bucket_key(self, dt):
        """生成桶 key。"""
        # 根据频率单位和倍数计算桶
        if self._freq_unit == "D":
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if self._freq_unit == "W":
            # 周一开始
            start = dt - timedelta(days=dt.weekday())
            return start.replace(hour=0, minute=0, second=0, microsecond=0)
        if self._freq_unit == "M":
            return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if self._freq_unit == "Y":
            return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        if self._freq_unit in ("H", "h"):
            # 小时桶，支持倍数
            hour_bucket = (dt.hour // self._freq_num) * self._freq_num
            return dt.replace(hour=hour_bucket, minute=0, second=0, microsecond=0)
        if self._freq_unit in ("T", "min", "Min", "m"):
            # 分钟桶，支持倍数如 "5Min"
            minute_bucket = (dt.minute // self._freq_num) * self._freq_num
            return dt.replace(minute=minute_bucket, second=0, microsecond=0)
        if self._freq_unit in ("S", "s"):
            # 秒桶，支持倍数
            second_bucket = (dt.second // self._freq_num) * self._freq_num
            return dt.replace(second=second_bucket, microsecond=0)
        return dt

    def _aggregate(self, aggfunc: str) -> _PySeries:
        # 优先调用 Rust 层 resample
        try:
            from datetime import datetime

            # 将索引转为 epoch 秒
            timestamps = []
            for dt in self._index:
                if isinstance(dt, datetime):
                    timestamps.append(dt.timestamp())
                elif isinstance(dt, (int, float)):
                    timestamps.append(float(dt))
                else:
                    timestamps.append(0.0)
            # 计算 freq_seconds，支持倍数
            freq_map = {
                "D": 86400.0,
                "W": 86400.0 * 7,
                "M": 86400.0 * 30,
                "Y": 86400.0 * 365,
                "H": 3600.0,
                "h": 3600.0,
                "T": 60.0,
                "min": 60.0,
                "Min": 60.0,
                "m": 60.0,
                "S": 1.0,
                "s": 1.0,
            }
            if self._freq_unit not in freq_map:
                raise ValueError(f"unsupported freq for Rust: {self._freq_unit}")
            freq_seconds = freq_map[self._freq_unit] * self._freq_num
            ts_list, val_list = self._s._inner.resample(
                timestamps, freq_seconds, aggfunc
            )
            # 将 epoch 秒转回 datetime
            from datetime import datetime as _dt

            out_index = [_dt.fromtimestamp(ts) for ts in ts_list]
            out_values = [v for v in val_list]
            # 推断 dtype（count 和 sum 对整数输入应返回 int）
            result = Series(out_values, name=self._s.name, index=out_index)
            # 如果原始值全是整数，且聚合函数为 sum/count，转换为 int64
            if aggfunc in ("sum", "count") and self._s._inner is not None:
                raw_vals = list(self._s._inner.values)
                if all(
                    isinstance(v, int) and not isinstance(v, bool)
                    for v in raw_vals
                    if v is not None
                ):
                    int_vals = [int(v) if v is not None else None for v in out_values]
                    result = Series(
                        int_vals, name=self._s.name, index=out_index, dtype="int64"
                    )
            # 设置频率信息
            result._freq = (
                f"{self._freq_num}{self._freq_unit}"
                if self._freq_num > 1
                else self._freq_unit
            )
            return result
        except Exception:
            pass

        # 回退到 Python 实现
        # 按桶分组 - 使用 dict.setdefault 简化分组逻辑
        buckets: dict = {}
        bucket_order: list = []
        for i, dt in enumerate(self._index):
            key = self._bucket_key(dt)
            if key not in buckets:
                buckets[key] = []
                bucket_order.append(key)
            buckets[key].append(self._values[i])

        def _agg_bucket(nums):
            """对单个桶内的非空值执行聚合。"""
            if not nums:
                return None
            if aggfunc == "sum":
                return sum(nums)
            if aggfunc == "mean":
                return sum(nums) / len(nums)
            if aggfunc == "count":
                return len(nums)
            if aggfunc == "min":
                return min(nums)
            if aggfunc == "max":
                return max(nums)
            if aggfunc == "median":
                s = sorted(nums)
                return (
                    s[len(s) // 2]
                    if len(s) % 2
                    else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2
                )
            if aggfunc == "std":
                if len(nums) < 2:
                    return None
                m = sum(nums) / len(nums)
                return (sum((x - m) ** 2 for x in nums) / len(nums)) ** 0.5
            if aggfunc == "first":
                return nums[0]
            if aggfunc == "last":
                return nums[-1]
            raise ValueError(f"unsupported aggfunc: {aggfunc}")

        # 对每个桶计算聚合值，过滤掉结果为 None 的桶
        agg_pairs = [
            (k, _agg_bucket([v for v in buckets[k] if v is not None]))
            for k in bucket_order
        ]
        agg_pairs = [(k, v) for k, v in agg_pairs if v is not None]
        out_values = [v for _, v in agg_pairs]
        out_index = [k for k, _ in agg_pairs]
        result = Series(out_values, name=self._s.name, index=out_index)
        # 设置频率信息
        result._freq = (
            f"{self._freq_num}{self._freq_unit}"
            if self._freq_num > 1
            else self._freq_unit
        )
        return result

    def sum(self) -> _PySeries:
        return self._aggregate("sum")

    def mean(self) -> _PySeries:
        return self._aggregate("mean")

    def count(self) -> _PySeries:
        return self._aggregate("count")

    def min(self) -> _PySeries:
        return self._aggregate("min")

    def max(self) -> _PySeries:
        return self._aggregate("max")

    def median(self) -> _PySeries:
        return self._aggregate("median")

    def std(self) -> _PySeries:
        return self._aggregate("std")

    def first(self) -> _PySeries:
        return self._aggregate("first")

    def last(self) -> _PySeries:
        return self._aggregate("last")

    def agg(self, func: str) -> _PySeries:
        return self._aggregate(func)


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
