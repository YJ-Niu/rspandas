"""StringAccessor 字符串访问器

由 rspandas/series.py 拆分而来。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from ..dataframe import DataFrame
    from ..series import Series


def __getattr__(name: str):
    """模块级延迟导入，避免与 series.py 的循环导入。"""
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


class StringAccessor:
    """Series.str 字符串访问器。"""

    def __init__(self, series: "Series"):
        self._s = series

    def _wrap(self, values: list, name: str = None) -> "Series":
        from ..series import Series

        return Series(
            values, name=name or self._s.name, index=self._s._index, dtype="str"
        )

    def _ensure_str(self, v):
        return (
            None
            if v is None or (isinstance(v, float) and v != v) or v == "NaN"
            else str(v)
        )

    def upper(self) -> _PySeries:
        from ..series import Series

        try:
            new_inner = self._s._inner.str_upper()
            return Series(
                new_inner, name=self._s.name, index=self._s._index, dtype="str"
            )
        except Exception:
            pass
        return self._wrap(
            [
                self._ensure_str(v).upper() if v is not None else None
                for v in self._s.values
            ]
        )

    def lower(self) -> _PySeries:
        from ..series import Series

        # 优先调用 Rust 层（使用 rayon 并行 to_lowercase，避免 Python 层多次遍历）
        try:
            new_inner = self._s._inner.str_lower()
            return Series(
                new_inner, name=self._s.name, index=self._s._index, dtype="str"
            )
        except Exception:
            pass
        # 回退：Python 层实现（Rust 失败时使用）
        vals = self._s.values
        str_vals = [self._ensure_str(v) for v in vals]
        return self._wrap([v.lower() if v is not None else None for v in str_vals])

    def title(self) -> _PySeries:
        return self._wrap(
            [
                self._ensure_str(v).title() if v is not None else None
                for v in self._s.values
            ]
        )

    def capitalize(self) -> _PySeries:
        return self._wrap(
            [
                self._ensure_str(v).capitalize() if v is not None else None
                for v in self._s.values
            ]
        )

    def strip(self) -> _PySeries:
        from ..series import Series

        try:
            new_inner = self._s._inner.str_strip()
            return Series(new_inner, name=self._s.name, index=self._s._index)
        except Exception:
            pass
        return self._wrap(
            [
                self._ensure_str(v).strip() if v is not None else None
                for v in self._s.values
            ]
        )

    def lstrip(self) -> _PySeries:
        return self._wrap(
            [
                self._ensure_str(v).lstrip() if v is not None else None
                for v in self._s.values
            ]
        )

    def rstrip(self) -> _PySeries:
        return self._wrap(
            [
                self._ensure_str(v).rstrip() if v is not None else None
                for v in self._s.values
            ]
        )

    def len(self) -> _PySeries:
        from ..series import Series

        try:
            new_inner = self._s._inner.str_len()
            return Series(new_inner, name=self._s.name, index=self._s._index)
        except Exception:
            pass
        return self._wrap([len(v) if v is not None else None for v in self._s.values])

    def contains(self, pat, case: bool = True, na=None) -> _PySeries:
        # case=True 时尝试调用 Rust 层加速
        if case:
            try:
                mask = self._s._inner.str_contains(pat)
                # Rust 层 None 视为 false，Python 层 None 返回 na
                out = [
                    na if v is None else mask[i] for i, v in enumerate(self._s.values)
                ]
                return self._wrap(out)
            except Exception:
                pass
        # 回退到原 Python 实现（case=False 或 Rust 调用失败时）
        needle = pat if case else pat.lower()

        def _contains_one(v):
            if v is None:
                return na
            target = str(v) if case else str(v).lower()
            return needle in target

        out = [_contains_one(v) for v in self._s.values]
        return self._wrap(out)

    def startswith(self, pat) -> _PySeries:
        return self._wrap(
            [str(v).startswith(pat) if v is not None else None for v in self._s.values]
        )

    def endswith(self, pat) -> _PySeries:
        return self._wrap(
            [str(v).endswith(pat) if v is not None else None for v in self._s.values]
        )

    def replace(self, pat, repl) -> _PySeries:
        from ..series import Series

        try:
            new_inner = self._s._inner.str_replace(pat, repl)
            return Series(new_inner, name=self._s.name, index=self._s._index)
        except Exception:
            pass
        return self._wrap(
            [
                str(v).replace(pat, repl) if v is not None else None
                for v in self._s.values
            ]
        )

    def split(self, pat: str = None, n: int = -1, expand: bool = False):
        """字符串分割。

        :param pat: 分隔符
        :param n: 最大分割次数 (-1 表示全部)
        :param expand: 是否展开为多列 DataFrame (暂不支持)
        :return: Series，每个元素为分割后的 list
        """
        from ..series import Series

        if expand:
            raise NotImplementedError("expand=True is not supported yet")
        result = [
            str(v).split(pat, n) if v is not None else None for v in self._s.values
        ]
        return Series(result, name=self._s.name, index=self._s._index, dtype="object")

    def slice(self, start=None, stop=None, step=None) -> _PySeries:
        s = slice(start, stop, step)
        return self._wrap(
            [str(v)[s] if v is not None else None for v in self._s.values]
        )

    def cat(self, sep: str = "") -> str:
        return sep.join(str(v) for v in self._s.values if v is not None)

    def find(self, sub, start=0, end=None) -> _PySeries:
        # 使用列表推导式替代显式 for 循环
        out = [
            (str(v).find(sub, start, end)) if v is not None else None
            for v in self._s.values
        ]
        return self._wrap(out)

    def rfind(self, sub, start=0, end=None) -> _PySeries:
        # 使用列表推导式替代显式 for 循环
        out = [
            (str(v).rfind(sub, start, end)) if v is not None else None
            for v in self._s.values
        ]
        return self._wrap(out)

    def index(self, sub, start=0, end=None) -> _PySeries:
        # 使用辅助函数 + 列表推导式替代显式 for 循环
        def _index_one(v):
            if v is None:
                return None
            try:
                return str(v).index(sub, start, end)
            except ValueError:
                return -1

        out = [_index_one(v) for v in self._s.values]
        return self._wrap(out)

    def rindex(self, sub, start=0, end=None) -> _PySeries:
        # 使用辅助函数 + 列表推导式替代显式 for 循环
        def _rindex_one(v):
            if v is None:
                return None
            try:
                return str(v).rindex(sub, start, end)
            except ValueError:
                return -1

        out = [_rindex_one(v) for v in self._s.values]
        return self._wrap(out)

    def match(self, pat, case=True, flags=0, na=None) -> _PySeries:
        import re

        actual_flags = flags if case else flags | re.IGNORECASE

        def _match_one(v):
            if v is None:
                return na
            return bool(re.match(pat, str(v), flags=actual_flags))

        out = [_match_one(v) for v in self._s.values]
        return self._wrap(out)

    def fullmatch(self, pat, case=True, flags=0, na=None) -> _PySeries:
        import re

        actual_flags = flags if case else flags | re.IGNORECASE

        def _fullmatch_one(v):
            if v is None:
                return na
            return bool(re.fullmatch(pat, str(v), flags=actual_flags))

        out = [_fullmatch_one(v) for v in self._s.values]
        return self._wrap(out)

    def extract(self, pat, flags=0, expand=True):
        import re

        def _extract_one(v):
            if v is None:
                return None
            m = re.search(pat, str(v), flags)
            if not m:
                return None
            if expand and m.groups():
                groups = list(m.groups())
                return groups[0] if len(groups) == 1 else groups
            return m.group(0)

        # 使用列表推导式替代显式 for 循环
        results = [_extract_one(v) for v in self._s.values]
        return results

    def extractall(self, pat, flags=0):
        import re

        # 使用列表推导式替代显式 for 循环
        results = [
            (re.findall(pat, str(v), flags) if v is not None else [])
            for v in self._s.values
        ]
        return results

    def count(self, pat, flags=0) -> _PySeries:
        import re

        return self._wrap(
            [
                len(re.findall(pat, str(v), flags)) if v is not None else None
                for v in self._s.values
            ]
        )

    def swapcase(self) -> _PySeries:
        return self._wrap(
            [
                self._ensure_str(v).swapcase() if v is not None else None
                for v in self._s.values
            ]
        )

    def casefold(self) -> _PySeries:
        return self._wrap(
            [
                self._ensure_str(v).casefold() if v is not None else None
                for v in self._s.values
            ]
        )

    def isalnum(self) -> _PySeries:
        return self._wrap(
            [str(v).isalnum() if v is not None else None for v in self._s.values]
        )

    def isalpha(self) -> _PySeries:
        return self._wrap(
            [str(v).isalpha() if v is not None else None for v in self._s.values]
        )

    def isdigit(self) -> _PySeries:
        return self._wrap(
            [str(v).isdigit() if v is not None else None for v in self._s.values]
        )

    def isspace(self) -> _PySeries:
        return self._wrap(
            [str(v).isspace() if v is not None else None for v in self._s.values]
        )

    def islower(self) -> _PySeries:
        return self._wrap(
            [str(v).islower() if v is not None else None for v in self._s.values]
        )

    def isupper(self) -> _PySeries:
        return self._wrap(
            [str(v).isupper() if v is not None else None for v in self._s.values]
        )

    def istitle(self) -> _PySeries:
        return self._wrap(
            [str(v).istitle() if v is not None else None for v in self._s.values]
        )

    def zfill(self, width) -> _PySeries:
        return self._wrap(
            [str(v).zfill(width) if v is not None else None for v in self._s.values]
        )

    def wrap(self, width, **kwargs) -> _PySeries:
        import textwrap

        return self._wrap(
            [
                textwrap.fill(str(v), width, **kwargs) if v is not None else None
                for v in self._s.values
            ]
        )

    def pad(self, width, side="left", fillchar=" ") -> _PySeries:
        # 使用辅助函数 + 列表推导式替代显式 for 循环
        def _pad_one(v):
            if v is None:
                return None
            s = str(v)
            if side == "left":
                return s.rjust(width, fillchar)
            if side == "right":
                return s.ljust(width, fillchar)
            if side == "both":
                return s.center(width, fillchar)
            raise ValueError(f"side must be 'left', 'right', or 'both', got {side!r}")

        out = [_pad_one(v) for v in self._s.values]
        return self._wrap(out)

    def center(self, width, fillchar=" ") -> _PySeries:
        return self._wrap(
            [
                str(v).center(width, fillchar) if v is not None else None
                for v in self._s.values
            ]
        )

    def ljust(self, width, fillchar=" ") -> _PySeries:
        return self._wrap(
            [
                str(v).ljust(width, fillchar) if v is not None else None
                for v in self._s.values
            ]
        )

    def rjust(self, width, fillchar=" ") -> _PySeries:
        return self._wrap(
            [
                str(v).rjust(width, fillchar) if v is not None else None
                for v in self._s.values
            ]
        )

    def partition(self, sep=" ") -> list:
        return [
            list(str(v).partition(sep)) if v is not None else None
            for v in self._s.values
        ]

    def rpartition(self, sep=" ") -> list:
        return [
            list(str(v).rpartition(sep)) if v is not None else None
            for v in self._s.values
        ]

    def rsplit(self, pat=None, n=-1) -> list:
        return [
            str(v).rsplit(pat, n) if v is not None else None for v in self._s.values
        ]

    def slice_replace(self, start=None, stop=None, repl=None) -> _PySeries:
        # 使用列表推导式替代显式 for 循环
        actual_repl = repl if repl is not None else ""

        def _slice_one(v):
            if v is None:
                return None
            s = str(v)
            return s[:start] + actual_repl + s[stop:]

        out = [_slice_one(v) for v in self._s.values]
        return self._wrap(out)

    def get(self, i) -> _PySeries:
        """从每个元素中取第 i 个值。

        - list/tuple: 取第 i 个元素 (支持负索引)
        - str: 取第 i 个字符 (支持负索引)
        """
        from ..series import Series

        import ast

        def _get_one(v):
            if v is None:
                return None
            if isinstance(v, (list, tuple)):
                if -len(v) <= i < len(v):
                    return v[i]
                return None
            # 字符串: 尝试解析为 list (str.split 的结果)
            s = str(v)
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        if -len(parsed) <= i < len(parsed):
                            return parsed[i]
                        return None
                except (ValueError, SyntaxError):
                    pass
            # 普通字符串取字符
            if -len(s) <= i < len(s):
                return s[i]
            return None

        return Series(
            [_get_one(v) for v in self._s.values],
            name=self._s.name,
            index=self._s._index,
            dtype="object",
        )

    def get_dummies(self, sep="|"):
        """返回 one-hot 编码的 DataFrame。"""
        from ..dataframe import DataFrame

        # 收集所有唯一值（使用集合推导式替代嵌套 for 循环）
        all_values = {
            part for v in self._s.values if v is not None for part in str(v).split(sep)
        }
        cols = sorted(all_values)

        # 对每行收集其 parts 集合，再使用列表推导式批量生成 one-hot 列
        rows_parts = [
            (set(str(v).split(sep)) if v is not None else None) for v in self._s.values
        ]
        # 使用字典推导式 + 嵌套列表推导式替代显式 for 循环
        data = {
            c: [1 if (rp is not None and c in rp) else 0 for rp in rows_parts]
            for c in cols
        }
        return DataFrame(data)

    def encode(self, encoding, errors="strict"):
        return [
            str(v).encode(encoding, errors) if v is not None else None
            for v in self._s.values
        ]

    def decode(self, encoding, errors="strict"):
        return [
            (
                v.decode(encoding, errors)
                if isinstance(v, bytes)
                else (str(v) if v is not None else None)
            )
            for v in self._s.values
        ]


def __getattr__(name: str):
    """模块级延迟导入，避免与 dataframe.py / series.py 的循环导入。"""
    if name == "DataFrame":
        from ..dataframe import DataFrame as _DF

        return _DF
    if name == "Series":
        from ..series import Series as _S

        return _S
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
