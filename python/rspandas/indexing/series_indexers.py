"""Series 索引器 (Loc/Iat/ILoc)

由 rspandas/series.py 拆分而来。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from ..dataframe import DataFrame
    from ..series import Series


class _LocIndexer:
    """Series 的标签索引器。"""

    def __init__(self, series: Series):
        self._s = series

    def __getitem__(self, key):
        """按标签取值。"""
        from ..series import Series

        index = (
            self._s._index if self._s._index is not None else list(range(len(self._s)))
        )
        if isinstance(key, slice):
            start = index.index(key.start) if key.start in index else 0
            stop = index.index(key.stop) + 1 if key.stop in index else len(self._s)
            vals = self._s.values[start:stop]
            return Series(vals, name=self._s.name, index=index[start:stop])
        elif isinstance(key, list):
            idxs = [index.index(k) for k in key if k in index]
            vals = [self._s.values[i] for i in idxs]
            return Series(
                vals,
                name=self._s.name,
                index=[key[i] for i, k in enumerate(key) if k in index],
            )
        else:
            if key in index:
                return self._s.values[index.index(key)]
            raise KeyError(f"label {key!r} not in index")

    def __setitem__(self, key, value):
        """按标签赋值。"""
        index = (
            self._s._index if self._s._index is not None else list(range(len(self._s)))
        )
        if key in index:
            idx = index.index(key)
            self._s._inner.values[idx] = value
        else:
            raise KeyError(f"label {key!r} not in index")


class _IatIndexer:
    """Series 的标量位置索引器。"""

    def __init__(self, series: Series):
        self._s = series

    def __getitem__(self, key):
        """按位置取标量值。"""
        if isinstance(key, (int, float)):
            idx = int(key)
            if 0 <= idx < len(self._s):
                return self._s.values[idx]
            raise IndexError("index out of range")
        raise TypeError("iat requires integer index")

    def __setitem__(self, key, value):
        """按位置赋标量值。"""
        if isinstance(key, (int, float)):
            idx = int(key)
            if 0 <= idx < len(self._s):
                self._s._inner.values[idx] = value
            else:
                raise IndexError("index out of range")
        else:
            raise TypeError("iat requires integer index")


class _ILocIndexer:
    """Series 的位置索引器（按整数位置索引）。"""

    def __init__(self, series: Series):
        self._s = series

    def __getitem__(self, key):
        """按位置取值。"""
        from ..series import Series

        n = len(self._s)
        if isinstance(key, int):
            if key < 0:
                key += n
            if key < 0 or key >= n:
                raise IndexError("index out of range")
            return self._s.values[key]
        if isinstance(key, slice):
            values = self._s.values[key]
            new_index = self._s._index[key] if self._s._index is not None else None
            return Series(
                values, name=self._s.name, index=new_index, dtype=self._s._dtype_str
            )
        if isinstance(key, (list, tuple)):
            if all(isinstance(x, bool) for x in key):
                return self._s._filter_mask(key)
            # 整数列表
            indices = [int(x) + n if int(x) < 0 else int(x) for x in key]
            values = [self._s.values[i] for i in indices]
            if self._s._index is not None:
                new_index = [self._s._index[i] for i in indices]
            else:
                new_index = None
            return Series(
                values, name=self._s.name, index=new_index, dtype=self._s._dtype_str
            )
        raise TypeError(f"iloc: unsupported key {type(key).__name__}")


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
