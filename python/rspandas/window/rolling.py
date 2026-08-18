"""Rolling 滑动窗口

由 rspandas/series.py 拆分而来。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from ..dataframe import DataFrame
    from ..series import Series
from typing import Any, Optional, Union


class Rolling:
    """Rolling 滚动窗口。

    Examples:
        >>> s = Series([1, 2, 3, 4, 5])
        >>> s.rolling(3).mean().values
        [None, None, 2.0, 3.0, 4.0]
    """

    def __init__(
        self,
        series: _PySeries,
        window: int,
        min_periods: int,
        center: bool = False,
        win_type: Optional[str] = None,
        closed: Optional[str] = None,
    ):
        self._s = series
        self._window = window
        self._min_periods = min_periods
        self._center = center
        self._win_type = win_type
        self._closed = closed or "right"

    def _apply(self, func) -> _PySeries:
        """应用窗口函数 func(window_values) -> scalar。"""
        from ..series import Series

        values = self._s.values
        n = len(values)
        out = []

        # 计算权重 (如果指定了 win_type)
        weights = None
        if self._win_type is not None:
            if self._win_type == "boxcar":
                weights = [1.0] * self._window
            elif self._win_type == "triang":
                w = self._window
                weights = [(i + 1) / w for i in range(w)]
            elif self._win_type == "blackman":
                import math

                w = self._window
                if w == 1:
                    weights = [1.0]
                else:
                    weights = [
                        0.42
                        - 0.5 * math.cos(2 * math.pi * i / (w - 1))
                        + 0.08 * math.cos(4 * math.pi * i / (w - 1))
                        for i in range(w)
                    ]
            else:
                raise ValueError(f"unsupported window type: {self._win_type}")

        for i in range(n):
            if self._center:
                start = max(0, i - self._window // 2)
                end = min(n, i + self._window // 2 + 1)
            else:
                start = max(0, i - self._window + 1)
                end = i + 1

            win = values[start:end]

            # 根据 closed 参数调整
            if self._closed == "left":
                if len(win) > 0:
                    win = win[:-1]
            elif self._closed == "neither":
                if len(win) >= 2:
                    win = win[1:-1]
                else:
                    win = []

            non_null = [v for v in win if v is not None]
            if len(non_null) < self._min_periods:
                out.append(None)
            else:
                try:
                    if weights is not None:
                        # 加权计算
                        w = weights[: len(win)]
                        out.append(func(win, w))
                    else:
                        out.append(func(win))
                except Exception:
                    out.append(None)
        return Series(out, name=self._s.name, index=self._s._index)

    def sum(self) -> _PySeries:
        # 优先调用 Rust 层（仅支持默认参数: center=False, win_type=None, closed=right）
        if not self._center and self._win_type is None and self._closed == "right":
            try:
                result = self._s._inner.rolling_sum(self._window, self._min_periods)
                return Series(result, name=self._s.name, index=self._s._index)
            except Exception:
                pass

        def f(win, w=None):
            return sum(v for v in win if v is not None)

        return self._apply(f)

    def mean(self) -> _PySeries:
        # 优先调用 Rust 层（仅支持默认参数: center=False, win_type=None, closed=right）
        if not self._center and self._win_type is None and self._closed == "right":
            try:
                result = self._s._inner.rolling_mean(self._window, self._min_periods)
                return Series(result, name=self._s.name, index=self._s._index)
            except Exception:
                pass

        def f(win, w=None):
            if w is not None:
                nums = [(v, wt) for v, wt in zip(win, w) if v is not None]
                if not nums:
                    return None
                total = sum(v * wt for v, wt in nums)
                wt_sum = sum(wt for _, wt in nums)
                return total / wt_sum if wt_sum > 0 else None
            nums = [v for v in win if v is not None]
            return sum(nums) / len(nums) if nums else None

        return self._apply(f)

    def min(self) -> _PySeries:
        def f(win, w=None):
            nums = [v for v in win if v is not None]
            return min(nums) if nums else None

        return self._apply(f)

    def max(self) -> _PySeries:
        def f(win, w=None):
            nums = [v for v in win if v is not None]
            return max(nums) if nums else None

        return self._apply(f)

    def std(self) -> _PySeries:
        # 优先调用 Rust 层（仅支持默认参数: center=False, win_type=None, closed=right）
        if not self._center and self._win_type is None and self._closed == "right":
            try:
                result = self._s._inner.rolling_std(self._window, self._min_periods)
                return Series(result, name=self._s.name, index=self._s._index)
            except Exception:
                pass

        def f(win, w=None):
            nums = [v for v in win if v is not None]
            if len(nums) < 2:
                return None
            m = sum(nums) / len(nums)
            var = sum((x - m) ** 2 for x in nums) / len(nums)
            return var**0.5

        return self._apply(f)

    def var(self) -> _PySeries:
        def f(win, w=None):
            nums = [v for v in win if v is not None]
            if len(nums) < 2:
                return None
            m = sum(nums) / len(nums)
            return sum((x - m) ** 2 for x in nums) / len(nums)

        return self._apply(f)

    def median(self) -> _PySeries:
        def f(win, w=None):
            nums = sorted([v for v in win if v is not None])
            if not nums:
                return None
            if len(nums) % 2:
                return nums[len(nums) // 2]
            return (nums[len(nums) // 2 - 1] + nums[len(nums) // 2]) / 2

        return self._apply(f)

    def count(self) -> _PySeries:
        values = self._s.values
        n = len(values)
        out = []
        for i in range(n):
            start = max(0, i - self._window + 1)
            win = values[start : i + 1]  # noqa
            cnt = sum(1 for v in win if v is not None)
            if cnt < self._min_periods:
                out.append(None)
            else:
                out.append(cnt)
        return Series(out, name=self._s.name, index=self._s._index)

    def corr(self, other: _PySeries) -> _PySeries:
        """滚动相关系数。"""
        if len(other) != len(self._s):
            raise ValueError("lengths must match")
        values_a = self._s.values
        values_b = other.values
        n = len(values_a)
        out = []
        for i in range(n):
            start = max(0, i - self._window + 1)
            wa = values_a[start : i + 1]  # noqa
            wb = values_b[start : i + 1]  # noqa
            pairs = [(a, b) for a, b in zip(wa, wb) if a is not None and b is not None]
            if len(pairs) < self._min_periods or len(pairs) < 2:
                out.append(None)
                continue
            ma = sum(a for a, b in pairs) / len(pairs)
            mb = sum(b for a, b in pairs) / len(pairs)
            num = sum((a - ma) * (b - mb) for a, b in pairs)
            da = (sum((a - ma) ** 2 for a, b in pairs)) ** 0.5
            db = (sum((b - mb) ** 2 for a, b in pairs)) ** 0.5
            if da == 0 or db == 0:
                out.append(None)
            else:
                out.append(num / (da * db))
        return Series(out, name=self._s.name, index=self._s._index)

    def cov(self, other: _PySeries) -> _PySeries:
        """滚动协方差。"""
        if len(other) != len(self._s):
            raise ValueError("lengths must match")
        values_a = self._s.values
        values_b = other.values
        n = len(values_a)
        out = []
        for i in range(n):
            start = max(0, i - self._window + 1)
            wa = values_a[start : i + 1]  # noqa
            wb = values_b[start : i + 1]  # noqa
            pairs = [(a, b) for a, b in zip(wa, wb) if a is not None and b is not None]
            if len(pairs) < self._min_periods or len(pairs) < 2:
                out.append(None)
                continue
            ma = sum(a for a, b in pairs) / len(pairs)
            mb = sum(b for a, b in pairs) / len(pairs)
            cov = sum((a - ma) * (b - mb) for a, b in pairs) / len(pairs)
            out.append(cov)
        return Series(out, name=self._s.name, index=self._s._index)

    def apply(self, func) -> _PySeries:
        """应用自定义窗口函数。"""
        return self._apply(func)

    # ---------- 滚动统计扩展 (v1.4.0) ----------

    def quantile(self, q: float = 0.5) -> _PySeries:
        """滚动分位数。"""

        def f(win, w=None):
            nums = sorted([float(v) for v in win if v is not None])
            if not nums:
                return None
            n = len(nums)
            if n == 1:
                return nums[0]
            pos = q * (n - 1)
            lo = int(pos)
            hi = min(lo + 1, n - 1)
            frac = pos - lo
            return nums[lo] * (1 - frac) + nums[hi] * frac

        return self._apply(f)

    def skew(self) -> _PySeries:
        """滚动偏度。"""

        def f(win, w=None):
            nums = [v for v in win if v is not None]
            n = len(nums)
            if n < 3:
                return None
            m = sum(nums) / n
            var = sum((x - m) ** 2 for x in nums) / n
            if var == 0:
                return 0.0
            m3 = sum((x - m) ** 3 for x in nums) / n
            return m3 / (var**1.5)

        return self._apply(f)

    def kurt(self) -> _PySeries:
        """滚动峰度 (excess kurtosis)。"""

        def f(win, w=None):
            nums = [v for v in win if v is not None]
            n = len(nums)
            if n < 4:
                return None
            m = sum(nums) / n
            var = sum((x - m) ** 2 for x in nums) / n
            if var == 0:
                return None
            m4 = sum((x - m) ** 4 for x in nums) / n
            return m4 / (var**2) - 3.0

        return self._apply(f)

    # ---------- v2.0.0: sem ----------

    def sem(self) -> _PySeries:
        """滚动标准误差 (Standard Error of Mean)。"""

        def f(win, w=None):
            nums = [v for v in win if v is not None]
            if len(nums) < 2:
                return None
            m = sum(nums) / len(nums)
            var = sum((x - m) ** 2 for x in nums) / (len(nums) - 1)
            return (var / len(nums)) ** 0.5

        return self._apply(f)


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
