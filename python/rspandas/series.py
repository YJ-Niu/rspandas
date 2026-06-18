"""Series: pandas-like 1D data structure with Rust backend."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Iterator, Optional, Tuple
from .rspandas import _Series as _PySeries, _DataFrame as _PyDataFrame  # type: ignore


# ---------------------------------------------------------------------------
# 类型推断
# ---------------------------------------------------------------------------

def _infer_dtype(values: list) -> str:
    """根据数据推断 dtype（对齐 pandas 的行为）。"""
    if not values:
        return "object"

    has_non_null = False
    all_bool = True
    all_int = True
    all_float = True
    all_str = True

    for v in values:
        if v is None:
            continue
        has_non_null = True
        # bool 优先于 int（True/False is int in Python）
        if isinstance(v, bool):
            all_int = False
            all_float = False
            all_str = False
        elif isinstance(v, int):
            all_bool = False
            all_float = False
            all_str = False
        elif isinstance(v, float):
            all_bool = False
            all_int = False
            all_str = False
        elif isinstance(v, str):
            all_bool = False
            all_int = False
            all_float = False
        else:
            # 不支持类型 -> object
            all_bool = False
            all_int = False
            all_float = False
            all_str = False
            return "object"

    # 全 None -> object
    if not has_non_null:
        return "object"
    if all_bool:
        return "bool"
    if all_int:
        return "int64"
    if all_float:
        return "float64"
    if all_str:
        return "object"
    return "object"


def _to_python_list(data: Any) -> list:
    """将输入标准化为 Python list。"""
    if isinstance(data, _PySeries):
        return list(data.values)
    if isinstance(data, (list, tuple)):
        return list(data)
    if isinstance(data, dict):
        # dict: 默认用 values
        return list(data.values())
    if data is None:
        return []
    raise TypeError(f"Cannot convert {type(data).__name__} to Series")


def _is_range_index(index) -> bool:
    """判断 index 是否为默认的 RangeIndex (0, 1, 2, ...)。"""
    if not index:
        return True
    return list(index) == list(range(len(index)))


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------

class Series:
    """一维带标签数组，对齐 pandas API。

    Examples:
        >>> s = Series([1, 2, 3], name='a')
        >>> s.shape
        (3,)
        >>> s.dtype
        'int64'
        >>> s.sum()
        6
    """

    def __init__(
        self,
        data=None,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
        index=None,
        copy: bool = False,
    ):
        """构造 Series。

        :param data: list / tuple / scalar
        :param name: 列名
        :param dtype: 可选类型 ('int64' / 'float64' / 'bool' / 'object')
        :param index: MVP 忽略，使用 RangeIndex
        :param copy: MVP 忽略
        """
        # 标准化数据
        values = _to_python_list(data)

        # 推断 dtype
        if dtype is None:
            dtype = _infer_dtype(values)

        # 构造 Rust 端 Series
        self._inner = _PySeries(values, name)

        # 缓存 dtype
        self._dtype_str: str = self._inner.dtype

        # MVP: RangeIndex
        self._index = index if index is not None else list(range(len(values)))

    # ---------- 属性 ----------

    @property
    def shape(self) -> Tuple[int]:
        return self._inner.shape

    @property
    def dtype(self) -> str:
        return self._dtype_str

    @property
    def name(self) -> Optional[str]:
        return self._inner.name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        self._inner.name = value

    @property
    def values(self) -> list:
        return list(self._inner.values)

    @property
    def size(self) -> int:
        return self._inner.size

    @property
    def empty(self) -> bool:
        return self._inner.empty

    @property
    def nbytes(self) -> int:
        return self._inner.nbytes

    @property
    def index(self):
        return self._index

    @property
    def ndim(self) -> int:
        return 1

    # ---------- dunder ----------

    def __len__(self) -> int:
        return self._inner.size

    def __iter__(self) -> Iterator:
        return iter(self.values)

    def __repr__(self) -> str:
        return self._format_repr()

    def __str__(self) -> str:
        return self._format_repr()

    def __getitem__(self, key):
        # 自定义 index: 优先按 label 查找
        if self._index is not None and not _is_range_index(self._index):
            if isinstance(key, (str, int, float, bool)):
                try:
                    pos = self._index.index(key)
                    return self.values[pos]
                except ValueError:
                    raise KeyError(key)
        # RangeIndex 或其他: 走位置
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("index out of range")
            return self.values[key]
        if isinstance(key, slice):
            values = self.values[key]
            new_index = self._index[key] if self._index is not None else None
            return Series(values, name=self.name, index=new_index)
        if isinstance(key, (list, tuple)) and all(isinstance(x, bool) for x in key):
            # bool mask
            return self._filter_mask(key)
        raise TypeError(f"Cannot index Series with {type(key).__name__}")

    def _filter_mask(self, mask: list) -> _PySeries:
        if len(mask) != len(self):
            raise ValueError(f"mask length {len(mask)} != series length {len(self)}")
        rust_mask = [bool(x) for x in mask]
        return Series(_PySeries_filter(self._inner, rust_mask), name=self.name)

    def __eq__(self, other) -> _PySeries:
        mask = self._inner.eq_scalar(other)
        return Series(mask, name=self.name, dtype="bool")

    def __ne__(self, other) -> _PySeries:
        # ne = not eq
        eq_mask = self._inner.eq_scalar(other)
        return Series([not x for x in eq_mask], name=self.name, dtype="bool")

    def __lt__(self, other) -> _PySeries:
        mask = self._inner.lt_scalar(other)
        return Series(mask, name=self.name, dtype="bool")

    def __gt__(self, other) -> _PySeries:
        mask = self._inner.gt_scalar(other)
        return Series(mask, name=self.name, dtype="bool")

    def __le__(self, other) -> _PySeries:
        mask = self._inner.le_scalar(other)
        return Series(mask, name=self.name, dtype="bool")

    def __ge__(self, other) -> _PySeries:
        mask = self._inner.ge_scalar(other)
        return Series(mask, name=self.name, dtype="bool")

    # ---------- 算术运算符 (v0.3.0) ----------

    def _arith(self, other, op: str) -> _PySeries:
        """逐元素算术运算，缺失值用 None。"""
        if isinstance(other, Series):
            if len(other) != len(self):
                raise ValueError("Series lengths must match")
            other_vals = other.values
        else:
            # 标量广播
            other_vals = [other] * len(self)

        result = []
        for a, b in zip(self.values, other_vals):
            if a is None or b is None:
                result.append(None)
                continue
            try:
                if op == "add":
                    result.append(a + b)
                elif op == "sub":
                    result.append(a - b)
                elif op == "mul":
                    result.append(a * b)
                elif op == "truediv":
                    if b == 0:
                        result.append(None)
                    else:
                        result.append(a / b)
                elif op == "floordiv":
                    if b == 0:
                        result.append(None)
                    else:
                        result.append(a // b)
                elif op == "mod":
                    if b == 0:
                        result.append(None)
                    else:
                        result.append(a % b)
                elif op == "pow":
                    result.append(a ** b)
            except (TypeError, ValueError):
                result.append(None)
        # 推断结果 dtype
        nums = [v for v in result if isinstance(v, (int, float))]
        if not nums:
            return Series(result, name=self.name, dtype="object")
        if any(isinstance(v, float) for v in nums):
            return Series(result, name=self.name, dtype="float64")
        return Series(result, name=self.name, dtype="int64")

    def __add__(self, other) -> _PySeries:
        return self._arith(other, "add")

    def __radd__(self, other) -> _PySeries:
        return self._arith(other, "add")

    def __sub__(self, other) -> _PySeries:
        return self._arith(other, "sub")

    def __rsub__(self, other) -> _PySeries:
        return self._arith(other, "sub")

    def __mul__(self, other) -> _PySeries:
        return self._arith(other, "mul")

    def __rmul__(self, other) -> _PySeries:
        return self._arith(other, "mul")

    def __truediv__(self, other) -> _PySeries:
        return self._arith(other, "truediv")

    def __rtruediv__(self, other) -> _PySeries:
        return self._arith(other, "truediv")

    def __floordiv__(self, other) -> _PySeries:
        return self._arith(other, "floordiv")

    def __rfloordiv__(self, other) -> _PySeries:
        return self._arith(other, "floordiv")

    def __mod__(self, other) -> _PySeries:
        return self._arith(other, "mod")

    def __rmod__(self, other) -> _PySeries:
        return self._arith(other, "mod")

    def __pow__(self, other) -> _PySeries:
        return self._arith(other, "pow")

    def __neg__(self) -> _PySeries:
        return self._arith(-1, "mul")

    def __pos__(self) -> _PySeries:
        return self

    def __abs__(self) -> _PySeries:
        result = [None if v is None else abs(v) for v in self.values]
        return Series(result, name=self.name, dtype=self._dtype_str)

    @property
    def str(self):
        """字符串访问器。"""
        return StringAccessor(self)

    def isin(self, other) -> _PySeries:
        """判断每个元素是否在 other 中。"""
        s = set(other)
        out = [v in s for v in self.values]
        return Series(out, name=self.name, index=self._index, dtype="bool")

    def between(self, left, right, inclusive: str = "both") -> _PySeries:
        """判断每个元素是否在 [left, right] 范围内。"""
        if inclusive == "both":
            out = [v is not None and left <= v <= right for v in self.values]
        elif inclusive == "left":
            out = [v is not None and left <= v < right for v in self.values]
        elif inclusive == "right":
            out = [v is not None and left < v <= right for v in self.values]
        elif inclusive == "neither":
            out = [v is not None and left < v < right for v in self.values]
        else:
            raise ValueError("inclusive must be one of: both/left/right/neither")
        return Series(out, name=self.name, index=self._index, dtype="bool")

    def to_pandas(self):
        """转换为 pandas Series。"""
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            raise ImportError("pandas is required for to_pandas()")
        index = self._index if self._index is not None else None
        return pd.Series(list(self.values), name=self.name, index=index)

    @classmethod
    def from_pandas(cls, ps) -> _PySeries:
        """从 pandas Series 构造。"""
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            raise ImportError("pandas is required for from_pandas()")
        if not isinstance(ps, pd.Series):
            raise TypeError("expected pandas Series")
        vals = [None if pd.isna(v) else v.item() if hasattr(v, 'item') else v
                for v in ps.values]
        index = list(ps.index) if ps.index is not None else None
        return cls(vals, name=ps.name, index=index)

    # ---------- 转换方法 (v1.0.0) ----------

    def to_list(self) -> list:
        """转换为 Python list。"""
        return list(self.values)

    def to_numpy(self):
        """转换为 numpy array。"""
        try:
            import numpy as np  # type: ignore
        except ImportError:
            raise ImportError("numpy is required for to_numpy()")
        return np.array(self.values)

    def to_dict(self) -> dict:
        """转换为 dict (index -> value)。"""
        return {self._index[i] if self._index else i: v
                for i, v in enumerate(self.values)}

    def to_frame(self, name=None) -> _PyDataFrame:
        """转换为 DataFrame。"""
        return _PyDataFrame({name or self.name: self.values})

    # ---------- 展开方法 (v1.0.0) ----------
    def explode(self) -> _PySeries:
        """展开列表元素为单独行。"""
        values = self.values
        out = []
        for v in values:
            if v is None:
                out.append(None)
            elif isinstance(v, (list, tuple)):
                out.extend(v)
            else:
                out.append(v)
        return Series(out, name=self.name)

    def repeat(self, repeats) -> _PySeries:
        """重复元素。
        :param repeats: 重复次数 (int 或 list[int])
        """
        values = self.values
        out = []
        if isinstance(repeats, int):
            for v in values:
                out.extend([v] * repeats)
        else:
            if len(repeats) != len(values):
                raise ValueError("repeats length must match series length")
            for v, rep in zip(values, repeats):
                out.extend([v] * rep)
        return Series(out, name=self.name)

    def __bool__(self) -> bool:
        if len(self) == 1:
            v = self.values[0]
            if v is None:
                raise ValueError("truth value of a None element is ambiguous")
            return bool(v)
        raise ValueError(f"truth value of a Series with {len(self)} elements is ambiguous")

    # ---------- 子集 ----------

    def head(self, n: int = 5) -> _PySeries:
        return Series(self._inner.head(n), name=self.name)

    def tail(self, n: int = 5) -> _PySeries:
        return Series(self._inner.tail(n), name=self.name)

    def iloc(self, key) -> _PySeries:
        """按位置索引。key: int / list[int] / slice / bool mask。"""
        n = len(self)
        if isinstance(key, int):
            if key < 0:
                key += n
            if key < 0 or key >= n:
                raise IndexError("index out of range")
            return Series([self.values[key]], name=self.name, dtype=self._dtype_str)
        if isinstance(key, slice):
            values = self.values[key]
            if self._index is not None:
                new_index = self._index[key]
            else:
                new_index = None
            return Series(values, name=self.name, index=new_index, dtype=self._dtype_str)
        if isinstance(key, (list, tuple)):
            if all(isinstance(x, bool) for x in key):
                return self._filter_mask(key)
            # 整数列表
            indices = [int(x) + n if int(x) < 0 else int(x) for x in key]
            values = [self.values[i] for i in indices]
            if self._index is not None:
                new_index = [self._index[i] for i in indices]
            else:
                new_index = None
            return Series(values, name=self.name, index=new_index, dtype=self._dtype_str)
        raise TypeError(f"iloc: unsupported key {type(key).__name__}")

    def sort_values(self, ascending: bool = True, inplace: bool = False) -> _PySeries:
        """按值排序。None 始终在末尾。"""
        pairs = [(i, v) for i, v in enumerate(self.values)]
        # stable sort: None 在末尾
        non_none = [(i, v) for i, v in pairs if v is not None]
        none_items = [(i, v) for i, v in pairs if v is None]
        try:
            non_none.sort(key=lambda x: x[1], reverse=not ascending)
        except TypeError:
            raise TypeError("cannot sort mixed types")
        sorted_pairs = non_none + none_items
        new_values = [v for _, v in sorted_pairs]
        if self._index is not None:
            new_index = [self._index[i] for i, _ in sorted_pairs]
        else:
            new_index = None
        if inplace:
            self._inner = _PySeries(new_values, self.name)
            self._index = new_index
            return self
        return Series(new_values, name=self.name, index=new_index)

    def apply(self, func) -> _PySeries:
        """对每个非 None 元素应用 func。

        :param func: callable (scalar) -> scalar
        """
        out = [None if v is None else func(v) for v in self.values]
        return Series(out, name=self.name, index=self._index)

    def map(self, arg) -> _PySeries:
        """映射: 可以传 dict 或 callable。

        - dict: 用值匹配, 缺失 -> None
        - callable: 逐元素应用, 同 apply
        """
        if isinstance(arg, dict):
            out = [None if v is None else arg.get(v, None) for v in self.values]
        else:
            out = [None if v is None else arg(v) for v in self.values]
        return Series(out, name=self.name, index=self._index)

    def replace(self, to_replace, value=None) -> _PySeries:
        """替换值。

        三种用法:
        - df.replace(old, new)
        - df.replace({old: new})
        - df.replace([old1, old2], [new1, new2])
        """
        # 形式 1: scalar
        if not isinstance(to_replace, (list, tuple, dict)):
            out = [value if v == to_replace else v for v in self.values]
            return Series(out, name=self.name, index=self._index)
        # 形式 3: list[old] + list[new]
        if isinstance(to_replace, (list, tuple)) and isinstance(value, (list, tuple)):
            if len(to_replace) != len(value):
                raise ValueError("to_replace and value must have the same length")
            mapping = dict(zip(to_replace, value))
        elif isinstance(to_replace, dict):
            mapping = to_replace
        else:
            raise TypeError("invalid replace arguments")
        out = [mapping.get(v, v) if v is not None else None for v in self.values]
        return Series(out, name=self.name, index=self._index)

    def duplicated(self, keep: str = "first") -> _PySeries:
        """返回 bool Series 标记重复行。

        :param keep: 'first' / 'last' / False
        """
        if keep == "first":
            seen = set()
            out = []
            for v in self.values:
                if v in seen:
                    out.append(True)
                else:
                    out.append(False)
                    seen.add(v)
        elif keep == "last":
            seen = set()
            rev_out = []
            for v in reversed(self.values):
                if v in seen:
                    rev_out.append(True)
                else:
                    rev_out.append(False)
                    seen.add(v)
            out = list(reversed(rev_out))
        elif keep is False:
            from collections import Counter
            c = Counter(self.values)
            dup = {k for k, n in c.items() if n > 1}
            out = [v in dup for v in self.values]
        else:
            raise ValueError("keep must be 'first', 'last', or False")
        return Series(out, name=self.name, index=self._index, dtype="bool")

    def drop_duplicates(self, keep: str = "first", inplace: bool = False) -> _PySeries:
        """删除重复值。"""
        seen = set()
        out = []
        out_idx = []
        for v, i in zip(self.values,
                        self._index if self._index else range(len(self))):
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
            out_idx.append(i)
        if keep == "last":
            seen = set()
            out = []
            out_idx = []
            for v, i in zip(reversed(self.values),
                            reversed(self._index if self._index else range(len(self)))):
                if v in seen:
                    continue
                seen.add(v)
                out.append(v)
                out_idx.append(i)
            out.reverse()
            out_idx.reverse()
        new_index = out_idx if self._index is not None else None
        if inplace:
            self._inner = _PySeries(out, self.name)
            self._index = new_index
            return self
        return Series(out, name=self.name, index=new_index)

    def where(self, cond, other=None) -> _PySeries:
        """三元: cond 为 True 保留 self, 否则取 other。"""
        if isinstance(cond, Series):
            cond = cond.values
        out = [
            (v if c else other) if v is not None and c else (other if not c else v)
            for v, c in zip(self.values, cond)
        ]
        return Series(out, name=self.name, index=self._index)

    def mask(self, cond, other=None) -> _PySeries:
        """where 的反义: cond 为 True 替换为 other, 否则保留 self。"""
        if isinstance(cond, Series):
            cond = cond.values
        out = [
            (other if c else v) if v is not None or not c else other
            for v, c in zip(self.values, cond)
        ]
        return Series(out, name=self.name, index=self._index)

    def astype(self, dtype: str) -> _PySeries:
        """类型转换（简化版）。"""
        target = dtype.lower()
        if target == self._dtype_str:
            return Series(self.values, name=self.name, dtype=dtype)
        if target == "int64":
            vals = [None if v is None else int(v) for v in self.values]
        elif target == "float64":
            vals = [None if v is None else float(v) for v in self.values]
        elif target == "bool":
            vals = [None if v is None else bool(v) for v in self.values]
        elif target == "object":
            vals = [None if v is None else str(v) for v in self.values]
        else:
            raise TypeError(f"unsupported dtype: {dtype}")
        return Series(vals, name=self.name, dtype=target)

    # ---------- 聚合 ----------

    def sum(self) -> Any:
        return self._inner.sum()

    def mean(self) -> Any:
        return self._inner.mean()

    def min(self) -> Any:
        return self._inner.min()

    def max(self) -> Any:
        return self._inner.max()

    def count(self) -> int:
        return self._inner.count()

    def std(self) -> Any:
        return self._inner.std()

    def var(self) -> Any:
        return self._inner.var()

    def median(self) -> Any:
        return self._inner.median()

    def any(self) -> Any:
        return self._inner.any()

    def all(self) -> Any:
        return self._inner.all()

    def describe(self) -> _PySeries:
        """返回统计摘要 Series。"""
        stats = {
            "count": self.count(),
            "mean": self.mean(),
            "std": self.std(),
            "min": self.min(),
            "50%": self.median(),
            "max": self.max(),
        }
        # 构建一个 Series
        result = Series(list(stats.values()), index=list(stats.keys()))
        return result

    # ---------- 极值位置 (v1.0.0) ----------

    def argmax(self) -> int:
        """返回最大值的位置索引 (整数位置)。"""
        values = self.values
        max_val = None
        max_idx = None
        for i, v in enumerate(values):
            if v is None:
                continue
            if max_val is None or v > max_val:
                max_val = v
                max_idx = i
        return max_idx

    def argmin(self) -> int:
        """返回最小值的位置索引 (整数位置)。"""
        values = self.values
        min_val = None
        min_idx = None
        for i, v in enumerate(values):
            if v is None:
                continue
            if min_val is None or v < min_val:
                min_val = v
                min_idx = i
        return min_idx

    def idxmax(self):
        """返回最大值的标签索引。"""
        idx = self.argmax()
        if idx is None:
            return None
        return self._index[idx] if self._index is not None else idx

    def idxmin(self):
        """返回最小值的标签索引。"""
        idx = self.argmin()
        if idx is None:
            return None
        return self._index[idx] if self._index is not None else idx

    # ---------- 缺失值 ----------

    def isnull(self) -> _PySeries:
        """返回 bool Series，True 表示该位置是 None。"""
        mask = self._inner.isnull()
        return Series(mask, name=self.name, dtype="bool")

    def notnull(self) -> _PySeries:
        """返回 bool Series，True 表示该位置不是 None。"""
        mask = self._inner.notnull()
        return Series(mask, name=self.name, dtype="bool")

    def dropna(self) -> _PySeries:
        """删除缺失值所在行。"""
        return Series(self._inner.dropna(), name=self.name)

    def fillna(self, value) -> _PySeries:
        """用 value 填充缺失值。"""
        return Series(self._inner.fillna(value), name=self.name)

    # ---------- 唯一值 ----------

    def unique(self) -> _PySeries:
        """返回去重后的 Series (保持首次出现顺序)。"""
        return Series(self._inner.unique(), name=self.name)

    def nunique(self) -> int:
        """返回不同值的数量 (None 不计入)。"""
        return self._inner.nunique()

    def value_counts(self) -> _PySeries:
        """统计每个值出现的次数，返回按出现顺序排序的 Series，索引为值。"""
        values, counts = self._inner.value_counts()
        s = Series(counts, index=values, name=self.name)
        return s

    # ---------- 统计方法 (v1.0.0) ----------

    def rank(self, method: str = "average") -> _PySeries:
        """计算排名。
        :param method: 'average' / 'min' / 'max' / 'first' / 'dense'
        """
        values = self.values
        n = len(values)
        # 创建 (value, original_index) 对
        indexed = [(v, i) for i, v in enumerate(values) if v is not None]
        # 排序
        indexed.sort(key=lambda x: x[0])
        ranks = [None] * n
        if not indexed:
            return Series(ranks, name=self.name, index=self._index)

        if method == "first":
            for rank, (_, idx) in enumerate(indexed, 1):
                ranks[idx] = rank
        elif method == "min":
            i = 0
            while i < len(indexed):
                current_val = indexed[i][0]
                start = i
                while i < len(indexed) and indexed[i][0] == current_val:
                    i += 1
                rank = start + 1
                for j in range(start, i):
                    ranks[indexed[j][1]] = rank
        elif method == "max":
            i = 0
            while i < len(indexed):
                current_val = indexed[i][0]
                start = i
                while i < len(indexed) and indexed[i][0] == current_val:
                    i += 1
                rank = i
                for j in range(start, i):
                    ranks[indexed[j][1]] = rank
        elif method == "dense":
            dense_rank = 1
            i = 0
            while i < len(indexed):
                current_val = indexed[i][0]
                while i < len(indexed) and indexed[i][0] == current_val:
                    ranks[indexed[i][1]] = dense_rank
                    i += 1
                dense_rank += 1
        else:
            i = 0
            while i < len(indexed):
                current_val = indexed[i][0]
                start = i
                while i < len(indexed) and indexed[i][0] == current_val:
                    i += 1
                avg_rank = (start + 1 + i) / 2
                for j in range(start, i):
                    ranks[indexed[j][1]] = avg_rank
        return Series(ranks, name=self.name, index=self._index)

    def quantile(self, q=0.5) -> float:
        """计算分位数。
        :param q: 分位数值 (0.0-1.0)
        """
        values = [v for v in self.values if v is not None]
        if not values:
            return None
        values.sort()
        n = len(values)
        if n == 1:
            return values[0]
        pos = q * (n - 1)
        lower = int(pos)
        upper = min(lower + 1, n - 1)
        frac = pos - lower
        return values[lower] * (1 - frac) + values[upper] * frac

    def mode(self, dropna: bool = True) -> _PySeries:
        """返回众数。"""
        from collections import Counter
        values = self.values
        if dropna:
            values = [v for v in values if v is not None]
        if not values:
            return Series([])
        counter = Counter(values)
        max_count = max(counter.values())
        modes = [v for v, cnt in counter.items() if cnt == max_count]
        return Series(sorted(modes), name=self.name)

    def skew(self) -> float:
        """计算偏度。"""
        values = [v for v in self.values if v is not None]
        n = len(values)
        if n < 3:
            return None
        m = sum(values) / n
        var = sum((x - m) ** 2 for x in values) / n
        if var == 0:
            return 0.0
        std = var ** 0.5
        skew = sum((x - m) ** 3 for x in values) / (n * std ** 3)
        return skew

    def kurt(self) -> float:
        """计算峰度。"""
        values = [v for v in self.values if v is not None]
        n = len(values)
        if n < 4:
            return None
        m = sum(values) / n
        var = sum((x - m) ** 2 for x in values) / n
        if var == 0:
            return 0.0
        std = var ** 0.5
        kurt = sum((x - m) ** 4 for x in values) / (n * std ** 4) - 3
        return kurt

    # ---------- 过滤 ----------

    def filter(self, mask: list) -> _PySeries:
        return self._filter_mask(mask)

    # ---------- 时序操作 (v1.0.0) ----------

    def shift(self, periods: int = 1) -> _PySeries:
        """将数据移动 periods 位。
        :param periods: 移动位数 (正数向后, 负数向前)
        """
        values = self.values
        n = len(values)
        out = [None] * n
        if periods > 0:
            for i in range(periods, n):
                out[i] = values[i - periods]
        elif periods < 0:
            for i in range(n + periods):
                out[i] = values[i - periods]
        return Series(out, name=self.name, index=self._index)

    def diff(self, periods: int = 1) -> _PySeries:
        """计算相邻元素的差。
        :param periods: 间隔位数
        """
        values = self.values
        n = len(values)
        out = []
        for i in range(n):
            prev_idx = i - periods
            if prev_idx < 0 or prev_idx >= n:
                out.append(None)
            else:
                a, b = values[i], values[prev_idx]
                if a is None or b is None:
                    out.append(None)
                else:
                    out.append(a - b)
        return Series(out, name=self.name, index=self._index)

    def pct_change(self, periods: int = 1, fill_method: str = "pad") -> _PySeries:
        """计算百分比变化。
        :param periods: 间隔位数
        :param fill_method: 'pad' (填充前值) / 'backfill' / None
        """
        values = self.values
        n = len(values)
        out = []
        for i in range(n):
            prev_idx = i - periods
            if prev_idx < 0 or prev_idx >= n:
                out.append(None)
            else:
                a, b = values[i], values[prev_idx]
                if b is None:
                    if fill_method == "pad":
                        out.append(None)
                    elif fill_method == "backfill":
                        out.append(None)
                    else:
                        out.append(None)
                elif a is None:
                    out.append(None)
                elif b == 0:
                    out.append(None)
                else:
                    out.append((a - b) / b)
        return Series(out, name=self.name, index=self._index)

    def cumsum(self, skipna: bool = True) -> _PySeries:
        """累加和。"""
        values = self.values
        out = []
        acc = None
        for v in values:
            if v is None:
                if skipna:
                    out.append(acc)
                else:
                    out.append(None)
                    acc = None
            else:
                if acc is None:
                    acc = v
                else:
                    acc = acc + v
                out.append(acc)
        return Series(out, name=self.name, index=self._index)

    def cumprod(self, skipna: bool = True) -> _PySeries:
        """累乘积。"""
        values = self.values
        out = []
        acc = None
        for v in values:
            if v is None:
                if skipna:
                    out.append(acc)
                else:
                    out.append(None)
                    acc = None
            else:
                if acc is None:
                    acc = v
                else:
                    acc = acc * v
                out.append(acc)
        return Series(out, name=self.name, index=self._index)

    def cummax(self, skipna: bool = True) -> _PySeries:
        """累计最大值。"""
        values = self.values
        out = []
        acc = None
        for v in values:
            if v is None:
                if skipna:
                    out.append(acc)
                else:
                    out.append(None)
                    acc = None
            else:
                if acc is None:
                    acc = v
                else:
                    acc = max(acc, v)
                out.append(acc)
        return Series(out, name=self.name, index=self._index)

    def cummin(self, skipna: bool = True) -> _PySeries:
        """累计最小值。"""
        values = self.values
        out = []
        acc = None
        for v in values:
            if v is None:
                if skipna:
                    out.append(acc)
                else:
                    out.append(None)
                    acc = None
            else:
                if acc is None:
                    acc = v
                else:
                    acc = min(acc, v)
                out.append(acc)
        return Series(out, name=self.name, index=self._index)

    # ---------- 窗口函数 (v1.0.0) ----------

    def rolling(self, window: int, min_periods: Optional[int] = None) -> "Rolling":
        """返回 Rolling 窗口对象。

        :param window: 窗口大小
        :param min_periods: 最少非空值数 (默认 = window)
        """
        if window < 1:
            raise ValueError("window must be >= 1")
        if min_periods is None:
            min_periods = window
        return Rolling(self, window, min_periods)

    def expanding(self, min_periods: int = 1) -> "Expanding":
        """返回 Expanding 窗口对象。"""
        if min_periods < 1:
            raise ValueError("min_periods must be >= 1")
        return Expanding(self, min_periods)

    def resample(self, freq: str) -> "Resampler":
        """时间序列重采样 (简化版 v1.0.0)。

        :param freq: 频率字符串 ('D'日, 'W'周, 'M'月, 'Y'年, 'H'时)
        :return: Resampler 对象，可调用 .sum()/.mean() 等聚合方法
        """
        from datetime import datetime
        # 解析 index -> datetime
        index = self._index if self._index is not None else list(range(len(self)))
        if not all(isinstance(i, datetime) for i in index):
            raise TypeError(
                "resample requires a datetime index; "
                "use to_datetime() to convert"
            )
        return Resampler(self, freq, index)

    # ---------- 显示 ----------

    def _format_repr(self) -> str:
        # 字符串化每个值
        strs = [
            str(v) if v is not None else "NaN"
            for v in self.values
        ]

        n = len(strs)

        # 准备索引字符串
        idx_strs = (
            [str(i) for i in self._index] if self._index is not None
            else [str(i) for i in range(n)]
        )
        # 截断：> 60 行
        if n > 60:
            head_strs = strs[:30]
            tail_strs = strs[-30:]
            head_idx = idx_strs[:30]
            tail_idx = idx_strs[-30:]
            strs = head_strs + ["..."] + tail_strs
            idx_strs = head_idx + ["..."] + tail_idx

        # 索引列宽度
        idx_width = max(
            (len(s) for s in idx_strs), default=1
        )

        lines = []
        pos = 0
        for s, idx_s in zip(strs, idx_strs):
            if s == "...":
                lines.append("..")
            else:
                lines.append(f"{idx_s:>{idx_width}}    {s}")
                pos += 1

        col_name = self.name if self.name is not None else ""
        body = "\n".join(lines)
        return f"{body}\nName: {col_name}, dtype: {self._dtype_str}"


def _PySeries_filter(inner: _PySeries, mask: list) -> _PySeries:
    """辅助函数：调用 Rust 端 filter。"""
    return inner.filter(mask)


# ---------------------------------------------------------------------------
# 窗口函数类 (v1.0.0)
# ---------------------------------------------------------------------------

class Rolling:
    """Rolling 滚动窗口。

    Examples:
        >>> s = Series([1, 2, 3, 4, 5])
        >>> s.rolling(3).mean().values
        [None, None, 2.0, 3.0, 4.0]
    """

    def __init__(self, series: _PySeries, window: int, min_periods: int):
        self._s = series
        self._window = window
        self._min_periods = min_periods

    def _apply(self, func) -> _PySeries:
        """应用窗口函数 func(window_values) -> scalar。"""
        values = self._s.values
        n = len(values)
        out = []
        for i in range(n):
            start = max(0, i - self._window + 1)
            win = values[start:i + 1]
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
        def f(win):
            return sum(v for v in win if v is not None)
        return self._apply(f)

    def mean(self) -> _PySeries:
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
            var = sum((x - m) ** 2 for x in nums) / len(nums)
            return var ** 0.5
        return self._apply(f)

    def var(self) -> _PySeries:
        def f(win):
            nums = [v for v in win if v is not None]
            if len(nums) < 2:
                return None
            m = sum(nums) / len(nums)
            return sum((x - m) ** 2 for x in nums) / len(nums)
        return self._apply(f)

    def median(self) -> _PySeries:
        def f(win):
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
            win = values[start:i + 1]
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
            wa = values_a[start:i + 1]
            wb = values_b[start:i + 1]
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
            wa = values_a[start:i + 1]
            wb = values_b[start:i + 1]
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
            win = values[:i + 1]
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
        return self._apply(lambda win: sum(v for v in win if v is not None))

    def mean(self) -> _PySeries:
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
        out = []
        for i in range(len(values)):
            win = values[:i + 1]
            cnt = sum(1 for v in win if v is not None)
            if cnt < self._min_periods:
                out.append(None)
            else:
                out.append(cnt)
        return Series(out, name=self._s.name, index=self._s._index)


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
    }

    def __init__(self, series: _PySeries, freq: str, index: list):
        if freq not in self._FREQ_MAP:
            raise ValueError(f"unsupported freq: {freq!r}")
        self._s = series
        self._freq = freq
        self._index = index
        self._values = series.values

    def _bucket_key(self, dt):
        """生成桶 key。"""
        if self._freq == "D":
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if self._freq == "W":
            # 周一开始
            start = dt - timedelta(days=dt.weekday())
            return start.replace(hour=0, minute=0, second=0, microsecond=0)
        if self._freq == "M":
            return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if self._freq == "Y":
            return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        if self._freq in ("H", "h"):
            return dt.replace(minute=0, second=0, microsecond=0)
        if self._freq == "S":
            return dt.replace(microsecond=0)
        return dt

    def _aggregate(self, aggfunc: str) -> _PySeries:
        # 按桶分组
        buckets: dict = {}
        bucket_order: list = []
        for i, dt in enumerate(self._index):
            key = self._bucket_key(dt)
            if key not in buckets:
                buckets[key] = []
                bucket_order.append(key)
            buckets[key].append(self._values[i])

        out_values = []
        out_index = []
        for k in bucket_order:
            vals = buckets[k]
            nums = [v for v in vals if v is not None]
            if not nums:
                continue
            if aggfunc == "sum":
                out_values.append(sum(nums))
            elif aggfunc == "mean":
                out_values.append(sum(nums) / len(nums))
            elif aggfunc == "count":
                out_values.append(len(nums))
            elif aggfunc == "min":
                out_values.append(min(nums))
            elif aggfunc == "max":
                out_values.append(max(nums))
            elif aggfunc == "median":
                s = sorted(nums)
                if len(s) % 2:
                    out_values.append(s[len(s) // 2])
                else:
                    out_values.append((s[len(s) // 2 - 1] + s[len(s) // 2]) / 2)
            elif aggfunc == "std":
                if len(nums) < 2:
                    out_values.append(None)
                else:
                    m = sum(nums) / len(nums)
                    out_values.append((sum((x - m) ** 2 for x in nums) / len(nums)) ** 0.5)
            elif aggfunc == "first":
                out_values.append(nums[0])
            elif aggfunc == "last":
                out_values.append(nums[-1])
            else:
                raise ValueError(f"unsupported aggfunc: {aggfunc}")
            out_index.append(k)
        return Series(out_values, name=self._s.name, index=out_index)

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


class StringAccessor:
    """Series.str 字符串访问器。"""

    def __init__(self, series: _PySeries):
        self._s = series

    def _wrap(self, values: list, name: str = None) -> _PySeries:
        return Series(values, name=name or self._s.name, index=self._s._index)

    def _ensure_str(self, v):
        return None if v is None else str(v)

    def upper(self) -> _PySeries:
        return self._wrap([self._ensure_str(v).upper() if v is not None else None for v in self._s.values])

    def lower(self) -> _PySeries:
        return self._wrap([self._ensure_str(v).lower() if v is not None else None for v in self._s.values])

    def title(self) -> _PySeries:
        return self._wrap([self._ensure_str(v).title() if v is not None else None for v in self._s.values])

    def capitalize(self) -> _PySeries:
        return self._wrap([self._ensure_str(v).capitalize() if v is not None else None for v in self._s.values])

    def strip(self) -> _PySeries:
        return self._wrap([self._ensure_str(v).strip() if v is not None else None for v in self._s.values])

    def lstrip(self) -> _PySeries:
        return self._wrap([self._ensure_str(v).lstrip() if v is not None else None for v in self._s.values])

    def rstrip(self) -> _PySeries:
        return self._wrap([self._ensure_str(v).rstrip() if v is not None else None for v in self._s.values])

    def len(self) -> _PySeries:
        return self._wrap([len(v) if v is not None else None for v in self._s.values])

    def contains(self, pat, case: bool = True, na=None) -> _PySeries:
        out = []
        for v in self._s.values:
            if v is None:
                out.append(na)
            else:
                target = str(v) if case else str(v).lower()
                needle = pat if case else pat.lower()
                out.append(needle in target)
        return self._wrap(out)

    def startswith(self, pat) -> _PySeries:
        return self._wrap([str(v).startswith(pat) if v is not None else None for v in self._s.values])

    def endswith(self, pat) -> _PySeries:
        return self._wrap([str(v).endswith(pat) if v is not None else None for v in self._s.values])

    def replace(self, pat, repl) -> _PySeries:
        return self._wrap([str(v).replace(pat, repl) if v is not None else None for v in self._s.values])

    def split(self, pat: str = None, n: int = -1) -> list:
        """字符串分割。返回 list[list[str]]。"""
        return [
            str(v).split(pat, n) if v is not None else None
            for v in self._s.values
        ]

    def slice(self, start=None, stop=None, step=None) -> _PySeries:
        s = slice(start, stop, step)
        return self._wrap([
            str(v)[s] if v is not None else None
            for v in self._s.values
        ])

    def cat(self, sep: str = "") -> str:
        return sep.join(str(v) for v in self._s.values if v is not None)
