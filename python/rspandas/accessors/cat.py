"""CatAccessor 分类访问器

由 rspandas/series.py 拆分而来。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from ..dataframe import DataFrame
    from ..series import Series
from typing import Any, Optional, Union


class CatAccessor:
    """Series.cat Categorical 访问器 (对齐 pandas 的 .cat 访问器)。"""

    def __init__(self, series: Series):
        self._s = series

    def _wrap_cat(self, inner_result) -> Series:
        """将 Rust 端返回的 PySeries 包装为 Series，保持 category dtype。"""
        s = Series.__new__(Series)
        s._inner = inner_result
        s._dtype_str = "category"
        s._index = list(range(inner_result.size))
        return s

    @property
    def categories(self) -> list:
        """返回 categories 列表。"""
        if self._s.dtype != "category":
            raise AttributeError("Can only use .cat accessor with 'category' dtype")
        # 优先使用 set_categories 保存的完整 categories 列表
        if hasattr(self._s, "_categories") and self._s._categories:
            return list(self._s._categories)
        inner = self._s._inner
        if hasattr(inner, "cat_categories"):
            cats = inner.cat_categories()
            if cats is not None:
                return list(cats)
        return sorted(set(v for v in self._s.values if v is not None))

    @property
    def codes(self) -> list:
        """返回 codes 列表。"""
        if self._s.dtype != "category":
            raise AttributeError("Can only use .cat accessor with 'category' dtype")
        inner = self._s._inner
        if hasattr(inner, "cat_codes"):
            codes = inner.cat_codes()
            if codes is not None:
                return [c if c is not None else -1 for c in codes]
        return []

    @property
    def ordered(self) -> bool:
        """是否有序。"""
        if self._s.dtype != "category":
            raise AttributeError("Can only use .cat accessor with 'category' dtype")
        inner = self._s._inner
        if hasattr(inner, "cat_ordered"):
            return inner.cat_ordered() or False
        return False

    def add_categories(self, new_categories: list) -> Series:
        """添加新的 categories。"""
        if self._s.dtype != "category":
            raise AttributeError("Can only use .cat accessor with 'category' dtype")
        inner = self._s._inner
        if hasattr(inner, "cat_add_categories"):
            result = inner.cat_add_categories(list(new_categories))
            if result is not None:
                return self._wrap_cat(result)
        return self._s

    def remove_unused_categories(self) -> Series:
        """移除未使用的 categories。"""
        if self._s.dtype != "category":
            raise AttributeError("Can only use .cat accessor with 'category' dtype")
        inner = self._s._inner
        if hasattr(inner, "cat_remove_unused_categories"):
            result = inner.cat_remove_unused_categories()
            if result is not None:
                return self._wrap_cat(result)
        return self._s

    def rename_categories(self, new_categories: list) -> Series:
        """重命名 categories。"""
        if self._s.dtype != "category":
            raise AttributeError("Can only use .cat accessor with 'category' dtype")
        inner = self._s._inner
        if hasattr(inner, "cat_rename_categories"):
            result = inner.cat_rename_categories(list(new_categories))
            if result is not None:
                return self._wrap_cat(result)
        # Python 回退：按位置重命名（使用实际 category 顺序，而非排序）
        old_cats = self.categories
        cat_map = {}
        for i, old in enumerate(old_cats):
            if i < len(new_categories):
                cat_map[old] = new_categories[i]
            else:
                cat_map[old] = old
        new_vals = [
            cat_map.get(v, v) if v is not None else None for v in self._s.values
        ]
        s = Series(new_vals, name=self._s.name, dtype="object")
        s._dtype_str = "category"
        return s

    def set_categories(
        self, new_categories, ordered: bool = False, rename: bool = False
    ) -> Series:
        """设置新的 categories（替换现有 categories 列表）。"""
        if self._s.dtype != "category":
            raise AttributeError("Can only use .cat accessor with 'category' dtype")
        # 简化实现：用 Python 构建新 Series
        cat_map = {}
        if rename:
            # 重命名模式：按位置映射
            old_cats = list(self._s.unique())
            for i, old in enumerate(old_cats):
                if i < len(new_categories):
                    cat_map[old] = new_categories[i]
                else:
                    cat_map[old] = old

        # 构建新值列表，None 保持不变
        new_vals = []
        for v in self._s.values:
            if v is None:
                new_vals.append(None)
            elif rename and v in cat_map:
                new_vals.append(cat_map[v])
            elif v in new_categories:
                new_vals.append(v)
            else:
                # 不在新 categories 中的值变为 NaN
                new_vals.append(None)
        s = Series(new_vals, name=self._s.name, dtype="object")
        s._dtype_str = "category"
        # 保存所有 new_categories 方便后续显示
        s._categories = list(new_categories)
        return s

    def as_ordered(self) -> Series:
        """设置为 ordered。"""
        if self._s.dtype != "category":
            raise AttributeError("Can only use .cat accessor with 'category' dtype")
        inner = self._s._inner
        if hasattr(inner, "cat_as_ordered"):
            result = inner.cat_as_ordered()
            if result is not None:
                return self._wrap_cat(result)
        return self._s

    def as_unordered(self) -> Series:
        """设置为 unordered。"""
        if self._s.dtype != "category":
            raise AttributeError("Can only use .cat accessor with 'category' dtype")
        inner = self._s._inner
        if hasattr(inner, "cat_as_unordered"):
            result = inner.cat_as_unordered()
            if result is not None:
                return self._wrap_cat(result)
        return self._s


# ==============================================================================
# DatetimeAccessor  - 日期时间访问器
# ==============================================================================


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
