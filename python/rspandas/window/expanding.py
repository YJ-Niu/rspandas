"""Expanding 扩展窗口

由 rspandas/series.py 拆分而来。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from ..dataframe import DataFrame
    from ..series import Series
from typing import Any, Optional, Union


class Expanding:
    """Expanding 扩展窗口 (从开始到当前位置)。"""

    def __init__(self, series: _PySeries, min_periods: int):
        self._s = series
        self._min_periods = min_periods

    def _apply(self, func) -> _PySeries:
        values = self._s.values
        n = len(values)
        out = []
        for i in range(n):
            win = values[: i + 1]
            non_null = [v for v in win if v is not None]
            if len(non_null) < self._min_periods:
                out.append(None)
            else:
                try:
                    out.append(func(win))
                except Exception:
                    out.append(None)
        return Series(out, name=self._s.name, index=self._s._index)

    def sum(self) -> _PySeries:
        # 优先调用 Rust 层
        try:
            result = self._s._inner.expanding_sum(self._min_periods)
            return Series(result, name=self._s.name, index=self._s._index)
        except Exception:
            pass
        return self._apply(lambda win: sum(v for v in win if v is not None))

    def mean(self) -> _PySeries:
        # 优先调用 Rust 层
        try:
            result = self._s._inner.expanding_mean(self._min_periods)
            return Series(result, name=self._s.name, index=self._s._index)
        except Exception:
            pass

        def f(win):
            nums = [v for v in win if v is not None]
            return sum(nums) / len(nums) if nums else None

        return self._apply(f)

    def min(self) -> _PySeries:
        def f(win):
            nums = [v for v in win if v is not None]
            return min(nums) if nums else None

        return self._apply(f)

    def max(self) -> _PySeries:
        def f(win):
            nums = [v for v in win if v is not None]
            return max(nums) if nums else None

        return self._apply(f)

    def std(self) -> _PySeries:
        def f(win):
            nums = [v for v in win if v is not None]
            if len(nums) < 2:
                return None
            m = sum(nums) / len(nums)
            return (sum((x - m) ** 2 for x in nums) / len(nums)) ** 0.5

        return self._apply(f)

    def var(self) -> _PySeries:
        def f(win):
            nums = [v for v in win if v is not None]
            if len(nums) < 2:
                return None
            m = sum(nums) / len(nums)
            return sum((x - m) ** 2 for x in nums) / len(nums)

        return self._apply(f)

    def count(self) -> _PySeries:
        values = self._s.values
        min_periods = self._min_periods

        def _count_one(i):
            """统计窗口 [0, i+1) 中非空值数量。"""
            cnt = sum(1 for v in values[: i + 1] if v is not None)
            return cnt if cnt >= min_periods else None

        # 使用列表推导式替代显式 for 循环
        out = [_count_one(i) for i in range(len(values))]
        return Series(out, name=self._s.name, index=self._s._index)


# ---------------------------------------------------------------------------
# EWM 指数加权移动窗口 (v1.4.0)
# ---------------------------------------------------------------------------


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
