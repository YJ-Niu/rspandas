"""Series: pandas-like 1D data structure with Rust backend."""

from __future__ import annotations

from typing import Any, Iterator, Optional, Tuple, Union

from rspandas.rspandas import _Series as _PySeries  # type: ignore


# ---------------------------------------------------------------------------
# 类型推断
# ---------------------------------------------------------------------------

def _infer_dtype(values: list) -> str:
    """根据数据推断 dtype（对齐 pandas 的行为）。"""
    if not values:
        return "object"

    has_none = False
    has_non_null = False
    all_bool = True
    all_int = True
    all_float = True
    all_str = True

    for v in values:
        if v is None:
            has_none = True
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

    def _filter_mask(self, mask: list) -> "Series":
        if len(mask) != len(self):
            raise ValueError(f"mask length {len(mask)} != series length {len(self)}")
        rust_mask = [bool(x) for x in mask]
        return Series(_PySeries_filter(self._inner, rust_mask), name=self.name)

    def __eq__(self, other) -> "Series":
        mask = self._inner.eq_scalar(other)
        return Series(mask, name=self.name, dtype="bool")

    def __ne__(self, other) -> "Series":
        # ne = not eq
        eq_mask = self._inner.eq_scalar(other)
        return Series([not x for x in eq_mask], name=self.name, dtype="bool")

    def __lt__(self, other) -> "Series":
        mask = self._inner.lt_scalar(other)
        return Series(mask, name=self.name, dtype="bool")

    def __gt__(self, other) -> "Series":
        mask = self._inner.gt_scalar(other)
        return Series(mask, name=self.name, dtype="bool")

    def __le__(self, other) -> "Series":
        mask = self._inner.le_scalar(other)
        return Series(mask, name=self.name, dtype="bool")

    def __ge__(self, other) -> "Series":
        mask = self._inner.ge_scalar(other)
        return Series(mask, name=self.name, dtype="bool")

    def __bool__(self) -> bool:
        if len(self) == 1:
            v = self.values[0]
            if v is None:
                raise ValueError("truth value of a None element is ambiguous")
            return bool(v)
        raise ValueError(f"truth value of a Series with {len(self)} elements is ambiguous")

    # ---------- 子集 ----------

    def head(self, n: int = 5) -> "Series":
        return Series(self._inner.head(n), name=self.name)

    def tail(self, n: int = 5) -> "Series":
        return Series(self._inner.tail(n), name=self.name)

    def astype(self, dtype: str) -> "Series":
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

    def describe(self) -> "Series":
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

    # ---------- 缺失值 ----------

    def isnull(self) -> "Series":
        """返回 bool Series，True 表示该位置是 None。"""
        mask = self._inner.isnull()
        return Series(mask, name=self.name, dtype="bool")

    def notnull(self) -> "Series":
        """返回 bool Series，True 表示该位置不是 None。"""
        mask = self._inner.notnull()
        return Series(mask, name=self.name, dtype="bool")

    def dropna(self) -> "Series":
        """删除缺失值所在行。"""
        return Series(self._inner.dropna(), name=self.name)

    def fillna(self, value) -> "Series":
        """用 value 填充缺失值。"""
        return Series(self._inner.fillna(value), name=self.name)

    # ---------- 唯一值 ----------

    def unique(self) -> "Series":
        """返回去重后的 Series (保持首次出现顺序)。"""
        return Series(self._inner.unique(), name=self.name)

    def nunique(self) -> int:
        """返回不同值的数量 (None 不计入)。"""
        return self._inner.nunique()

    def value_counts(self) -> "Series":
        """统计每个值出现的次数，返回按出现顺序排序的 Series，索引为值。"""
        values, counts = self._inner.value_counts()
        s = Series(counts, index=values, name=self.name)
        return s

    # ---------- 过滤 ----------

    def filter(self, mask: list) -> "Series":
        return self._filter_mask(mask)

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
