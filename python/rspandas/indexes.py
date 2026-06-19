"""高级索引: Index / RangeIndex / MultiIndex + 工具函数 (v1.3.0)。

提供与 pandas 兼容的索引类型和工具函数。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from .rspandas import _DataFrame as rspandas_DataFrame  # type: ignore
from .series import Series


# ============================================================================
# Index
# ============================================================================

class Index:
    """不可变的一维标签数组，与 pandas.Index 对齐。

    Examples:
        >>> idx = Index([1, 2, 3])
        >>> len(idx)
        3
        >>> 2 in idx
        True
    """

    def __init__(self, data=None, name: Optional[str] = None):
        if data is None:
            self._data: list = []
        elif isinstance(data, Index):
            self._data = list(data._data)
        elif isinstance(data, (list, tuple)):
            self._data = list(data)
        else:
            self._data = list(data)
        self._name = name

    # ---------- 属性 ----------

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, value: Optional[str]):
        self._name = value

    @property
    def values(self) -> list:
        return list(self._data)

    @property
    def dtype(self) -> str:
        if not self._data:
            return "object"
        non_null = [v for v in self._data if v is not None]
        if not non_null:
            return "object"
        if all(isinstance(v, bool) for v in non_null):
            return "bool"
        if all(isinstance(v, int) for v in non_null):
            return "int64"
        if all(isinstance(v, float) for v in non_null):
            return "float64"
        return "object"

    @property
    def shape(self) -> Tuple[int]:
        return (len(self._data),)

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def empty(self) -> bool:
        return len(self._data) == 0

    @property
    def is_unique(self) -> bool:
        return len(set(self._data)) == len(self._data)

    @property
    def is_monotonic_increasing(self) -> bool:
        if len(self._data) < 2:
            return True
        for i in range(1, len(self._data)):
            if self._data[i] is not None and self._data[i - 1] is not None:
                if self._data[i] < self._data[i - 1]:
                    return False
        return True

    @property
    def is_monotonic_decreasing(self) -> bool:
        if len(self._data) < 2:
            return True
        for i in range(1, len(self._data)):
            if self._data[i] is not None and self._data[i - 1] is not None:
                if self._data[i] > self._data[i - 1]:
                    return False
        return True

    # ---------- dunder ----------

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        name = f", name='{self._name}'" if self._name else ""
        return f"Index({self._data}{name})"

    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            result = self._data[key]
            if isinstance(key, int):
                return result
            return Index(result, name=self._name)
        if isinstance(key, (list, tuple)):
            return Index([self._data[i] for i in key], name=self._name)
        raise TypeError(f"Index key must be int/slice/list, not {type(key).__name__}")

    def __contains__(self, item) -> bool:
        return item in self._data

    def __eq__(self, other) -> bool:
        if isinstance(other, Index):
            return self._data == other._data
        return False

    # ---------- 方法 ----------

    def tolist(self) -> list:
        return list(self._data)

    def to_list(self) -> list:
        return self.tolist()

    def get_loc(self, key) -> int:
        """返回 key 在 Index 中的位置。"""
        try:
            return self._data.index(key)
        except ValueError:
            raise KeyError(key)

    def append(self, other: "Index") -> "Index":
        return Index(self._data + list(other._data), name=self._name)

    def difference(self, other: "Index") -> "Index":
        other_set = set(other._data)
        return Index([v for v in self._data if v not in other_set])

    def intersection(self, other: "Index") -> "Index":
        other_set = set(other._data)
        seen = set()
        result = []
        for v in self._data:
            if v in other_set and v not in seen:
                result.append(v)
                seen.add(v)
        return Index(result)

    def union(self, other: "Index") -> "Index":
        seen = set()
        result = []
        for v in self._data:
            if v not in seen:
                result.append(v)
                seen.add(v)
        for v in other._data:
            if v not in seen:
                result.append(v)
                seen.add(v)
        return Index(result)

    def unique(self) -> "Index":
        seen = set()
        result = []
        for v in self._data:
            if v not in seen:
                result.append(v)
                seen.add(v)
        return Index(result)

    def sort_values(self, ascending: bool = True) -> "Index":
        sorted_data = sorted(
            [v for v in self._data if v is not None],
            reverse=not ascending,
        )
        return Index(sorted_data, name=self._name)

    def astype(self, dtype: str) -> "Index":
        """转换 Index 类型。"""
        result = list(self._data)
        if dtype in ("int64", "int"):
            result = [int(v) if v is not None else None for v in result]
        elif dtype in ("float64", "float"):
            result = [float(v) if v is not None else None for v in result]
        elif dtype in ("str", "object", "string"):
            result = [str(v) if v is not None else None for v in result]
        return Index(result, name=self._name)

    def rename(self, name: str) -> "Index":
        return Index(self._data, name=name)

    def fillna(self, value) -> "Index":
        return Index([v if v is not None else value for v in self._data], name=self._name)

    def dropna(self) -> "Index":
        return Index([v for v in self._data if v is not None], name=self._name)

    def isin(self, values) -> list:
        val_set = set(values)
        return [v in val_set for v in self._data]

    def min(self):
        non_null = [v for v in self._data if v is not None]
        return min(non_null) if non_null else None

    def max(self):
        non_null = [v for v in self._data if v is not None]
        return max(non_null) if non_null else None

    def argmin(self) -> int:
        non_null = [(i, v) for i, v in enumerate(self._data) if v is not None]
        if not non_null:
            raise ValueError("empty index")
        return min(non_null, key=lambda x: x[1])[0]

    def argmax(self) -> int:
        non_null = [(i, v) for i, v in enumerate(self._data) if v is not None]
        if not non_null:
            raise ValueError("empty index")
        return max(non_null, key=lambda x: x[1])[0]

    def duplicated(self, keep: str = "first") -> list:
        seen = set()
        result = []
        for v in self._data:
            if v in seen:
                result.append(True)
            else:
                result.append(False)
                seen.add(v)
        if keep == "last":
            # 反转检测
            seen.clear()
            for i in range(len(self._data) - 1, -1, -1):
                v = self._data[i]
                if v in seen:
                    result[i] = True
                else:
                    result[i] = False
                    seen.add(v)
        return result

    def copy(self) -> "Index":
        return Index(self._data, name=self._name)

    # ---------- v2.0.0: Index 扩展 ----------

    def map(self, mapper) -> "Index":
        """对索引中的每个值应用映射函数。

        :param mapper: dict 或 callable
        """
        if callable(mapper):
            new_data = [mapper(v) for v in self._data]
        elif isinstance(mapper, dict):
            new_data = [mapper.get(v, v) for v in self._data]
        else:
            raise TypeError("mapper must be a dict or callable")
        return Index(new_data, name=self._name)

    def where(self, cond: list) -> "Index":
        """条件筛选，不满足条件的替换为 None。

        :param cond: bool 列表
        """
        new_data = [v if i < len(cond) and cond[i] else None for i, v in enumerate(self._data)]
        return Index(new_data, name=self._name)

    def mask(self, cond: list) -> "Index":
        """条件筛选，满足条件的替换为 None。

        :param cond: bool 列表
        """
        new_data = [None if i < len(cond) and cond[i] else v for i, v in enumerate(self._data)]
        return Index(new_data, name=self._name)

    def set_names(self, names, level=None) -> "Index":
        """设置索引名称。

        :param names: 名称或名称列表
        """
        if isinstance(names, (list, tuple)):
            name = names[0] if names else None
        else:
            name = names
        return Index(self._data, name=name)

    def symmetric_difference(self, other: "Index") -> "Index":
        """返回两个索引的对称差集。

        :param other: 另一个 Index
        """
        other_set = set(other._data)
        self_set = set(self._data)
        result = list(self_set.symmetric_difference(other_set))
        return Index(result)

    def any(self) -> bool:
        """返回是否所有值中存在 True。"""
        return any(bool(v) for v in self._data)

    def all(self) -> bool:
        """返回是否所有值都为 True。"""
        return all(bool(v) for v in self._data)

    def to_numpy(self):
        """转换为 numpy 数组。"""
        try:
            import numpy as np
            return np.array(self._data)
        except ImportError:
            raise ImportError("numpy is required for to_numpy()")

    def to_frame(self, index: bool = True, name: Optional[str] = None):
        """转换为 DataFrame。"""
        from .dataframe import DataFrame
        col_name = name or self._name or "index"
        df = DataFrame({col_name: self._data})
        if not index:
            df._index = list(range(len(self._data)))
        return df


# ============================================================================
# RangeIndex
# ============================================================================

class RangeIndex(Index):
    """优化的范围索引，类似 pandas.RangeIndex。

    Examples:
        >>> ri = RangeIndex(5)
        >>> len(ri)
        5
        >>> list(ri)
        [0, 1, 2, 3, 4]
        >>> ri = RangeIndex(1, 10, 2)
        >>> list(ri)
        [1, 3, 5, 7, 9]
    """

    def __init__(self, start=0, stop=None, step=1, name: Optional[str] = None):
        if stop is None:
            stop = start
            start = 0
        self._start = start
        self._stop = stop
        self._step = step
        self._name = name

    @property
    def start(self) -> int:
        return self._start

    @property
    def stop(self) -> int:
        return self._stop

    @property
    def step(self) -> int:
        return self._step

    # ---------- 重写 Index 方法 ----------

    def __len__(self) -> int:
        if self._step > 0:
            return max(0, (self._stop - self._start + self._step - 1) // self._step)
        return max(0, (self._start - self._stop + abs(self._step) - 1) // abs(self._step))

    def __repr__(self) -> str:
        name = f", name='{self._name}'" if self._name else ""
        return f"RangeIndex(start={self._start}, stop={self._stop}, step={self._step}{name})"

    def __iter__(self):
        return iter(range(self._start, self._stop, self._step))

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("RangeIndex index out of range")
            return self._start + key * self._step
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            new_start = self._start + start * self._step
            new_stop = self._start + stop * self._step
            new_step = self._step * step
            return RangeIndex(new_start, new_stop, new_step, name=self._name)
        if isinstance(key, (list, tuple)):
            return Index([self._start + k * self._step for k in key], name=self._name)
        raise TypeError("RangeIndex key must be int/slice/list")

    def __contains__(self, item) -> bool:
        if not isinstance(item, int):
            return False
        if self._step > 0:
            return self._start <= item < self._stop and (item - self._start) % self._step == 0
        return self._stop < item <= self._start and (item - self._start) % self._step == 0

    def __eq__(self, other) -> bool:
        if isinstance(other, RangeIndex):
            return (self._start == other._start and self._stop == other._stop and self._step == other._step)
        return False

    # ---------- 方法 ----------

    def tolist(self) -> list:
        return list(range(self._start, self._stop, self._step))

    def to_list(self) -> list:
        return self.tolist()

    def get_loc(self, key) -> int:
        if not isinstance(key, int):
            raise KeyError(key)
        if key not in self:
            raise KeyError(key)
        return (key - self._start) // self._step

    @property
    def values(self) -> list:
        return self.tolist()

    @property
    def size(self) -> int:
        return len(self)

    @property
    def empty(self) -> bool:
        return len(self) == 0

    @property
    def is_unique(self) -> bool:
        return self._step != 0 and len(self) > 0

    @property
    def is_monotonic_increasing(self) -> bool:
        return self._step > 0

    @property
    def is_monotonic_decreasing(self) -> bool:
        return self._step < 0

    @property
    def dtype(self) -> str:
        return "int64"

    def copy(self) -> "RangeIndex":
        return RangeIndex(self._start, self._stop, self._step, name=self._name)

    @staticmethod
    def from_range(start: int, stop: int, step: int = 1, name=None) -> "RangeIndex":
        return RangeIndex(start, stop, step, name=name)


# ============================================================================
# MultiIndex
# ============================================================================

class MultiIndex(Index):
    """多级索引，类似 pandas.MultiIndex。

    Examples:
        >>> arrays = [['a', 'a', 'b', 'b'], [1, 2, 1, 2]]
        >>> mi = MultiIndex.from_arrays(arrays, names=['first', 'second'])
        >>> len(mi)
        4
        >>> mi.nlevels
        2
    """

    def __init__(self, levels, codes, names=None):
        self._levels = [list(level) for level in levels]
        self._codes = [list(c) for c in codes]
        self._names = names if names is not None else [None] * len(levels)
        self._name = None  # MultiIndex 本身没有 name

        # 验证
        if len(self._levels) != len(self._codes):
            raise ValueError("levels and codes must have same length")
        n = len(self._codes[0]) if self._codes else 0
        for c in self._codes:
            if len(c) != n:
                raise ValueError("all codes must have same length")

    # ---------- 属性 ----------

    @property
    def nlevels(self) -> int:
        return len(self._levels)

    @property
    def levels(self) -> List[list]:
        return self._levels

    @property
    def codes(self) -> List[list]:
        return self._codes

    @property
    def names(self) -> List[Optional[str]]:
        return self._names

    @names.setter
    def names(self, value: List[Optional[str]]):
        if len(value) != self.nlevels:
            raise ValueError(f"names length {len(value)} != nlevels {self.nlevels}")
        self._names = list(value)

    @property
    def values(self) -> list:
        return self.tolist()

    @property
    def dtype(self) -> str:
        return "object"

    @property
    def is_unique(self) -> bool:
        return len(set(self.tolist())) == len(self)

    # ---------- dunder ----------

    def __len__(self) -> int:
        return len(self._codes[0]) if self._codes else 0

    def __repr__(self) -> str:
        names_str = f", names={self._names}" if any(n is not None for n in self._names) else ""
        return f"MultiIndex(levels={self._levels}, codes={self._codes}{names_str})"

    def __iter__(self):
        return iter(self.tolist())

    def __getitem__(self, key):
        if isinstance(key, int):
            return tuple(
                self._levels[level][self._codes[level][key]]
                if self._codes[level][key] >= 0 else None
                for level in range(self.nlevels)
            )
        if isinstance(key, slice):
            indices = range(len(self))[key]
            return self._slice_by_indices(indices)
        if isinstance(key, (list, tuple)):
            if all(isinstance(k, bool) for k in key):
                return self._slice_by_indices([i for i, b in enumerate(key) if b])
            return self._slice_by_indices(key)
        raise TypeError("MultiIndex key must be int/slice/list")

    def __contains__(self, item) -> bool:
        return item in self.tolist()

    def __eq__(self, other) -> bool:
        if isinstance(other, MultiIndex):
            return (self._levels == other._levels and self._codes == other._codes and self._names == other._names)
        return False

    # ---------- 构造方法 ----------

    @staticmethod
    def from_arrays(arrays, names=None) -> MultiIndex:
        """从数组列表构造 MultiIndex。

        Parameters
        ----------
        arrays : list[list]
            每个 level 一个数组。
        names : list[str], optional
            每个 level 的名称。
        """
        arrays = [list(a) for a in arrays]
        n = len(arrays[0])
        for a in arrays:
            if len(a) != n:
                raise ValueError("all arrays must have same length")

        levels = []
        codes = []
        for arr in arrays:
            unique = []
            seen = {}
            for v in arr:
                if v not in seen:
                    seen[v] = len(unique)
                    unique.append(v)
            levels.append(unique)
            codes.append([seen[v] for v in arr])

        return MultiIndex(levels, codes, names=names)

    @staticmethod
    def from_tuples(tuples, names=None) -> "MultiIndex":
        """从 tuple 列表构造 MultiIndex。
        Parameters
        ----------
        tuples : list[tuple]
            每个元素是一个 tuple，表示一行索引。
        names : list[str], optional
            每个 level 的名称。
        """
        if not tuples:
            return MultiIndex([], [], names=names or [])

        nlevels = len(tuples[0])
        arrays = [[] for _ in range(nlevels)]
        for t in tuples:
            for i in range(nlevels):
                arrays[i].append(t[i])

        return MultiIndex.from_arrays(arrays, names=names)

    @staticmethod
    def from_product(iterables, names=None) -> MultiIndex:
        """从笛卡尔积构造 MultiIndex。
        Parameters
        ----------
        iterables : list[list]
            每个 level 的可能值列表。
        names : list[str], optional
            每个 level 的名称。
        """
        iterables = [list(it) for it in iterables]
        nlevels = len(iterables)

        # 笛卡尔积
        def _product(idx, current):
            if idx == nlevels:
                return [tuple(current)]
            result = []
            for v in iterables[idx]:
                result.extend(_product(idx + 1, current + [v]))
            return result

        tuples = _product(0, [])
        return MultiIndex.from_tuples(tuples, names=names)

    # ---------- 方法 ----------

    def tolist(self) -> list:
        """返回 tuple 列表。"""
        result = []
        for i in range(len(self)):
            result.append(self[i])
        return result

    def to_list(self) -> list:
        return self.tolist()

    def get_loc(self, key) -> Union[int, slice]:
        """返回 key 在 MultiIndex 中的位置。"""
        if isinstance(key, tuple):
            try:
                return self.tolist().index(key)
            except ValueError:
                raise KeyError(key)
        # 部分 key: 只匹配第一层
        for i in range(len(self)):
            if self[i][0] == key:
                return i
        raise KeyError(key)

    def get_level_values(self, level: Union[int, str]) -> Index:
        """返回指定 level 的值。

        Parameters
        ----------
        level : int or str
            level 索引或名称。

        Returns
        -------
        Index
        """
        level_idx = self._resolve_level(level)
        values = []
        for i in range(len(self)):
            c = self._codes[level_idx][i]
            values.append(self._levels[level_idx][c] if c >= 0 else None)
        return Index(values, name=self._names[level_idx])

    def swaplevel(self, i: int = -2, j: int = -1) -> MultiIndex:
        """交换两个 level。

        Parameters
        ----------
        i, j : int
            要交换的两个 level 索引。

        Returns
        -------
        MultiIndex
        """
        i = i if i >= 0 else self.nlevels + i
        j = j if j >= 0 else self.nlevels + j

        new_levels = list(self._levels)
        new_codes = list(self._codes)
        new_names = list(self._names)

        new_levels[i], new_levels[j] = new_levels[j], new_levels[i]
        new_codes[i], new_codes[j] = new_codes[j], new_codes[i]
        new_names[i], new_names[j] = new_names[j], new_names[i]

        return MultiIndex(new_levels, new_codes, names=new_names)

    def reorder_levels(self, order: List[int]) -> MultiIndex:
        """重新排列 level 顺序。

        Parameters
        ----------
        order : list[int]
            新的 level 顺序。

        Returns
        -------
        MultiIndex
        """
        new_levels = [self._levels[i] for i in order]
        new_codes = [self._codes[i] for i in order]
        new_names = [self._names[i] for i in order]
        return MultiIndex(new_levels, new_codes, names=new_names)

    def droplevel(self, level: Union[int, str, List[Union[int, str]]]) -> Index:
        """删除指定 level，返回 Index 或 MultiIndex。

        Parameters
        ----------
        level : int, str, or list
            要删除的 level。

        Returns
        -------
        Index or MultiIndex
        """
        if isinstance(level, (int, str)):
            levels_to_drop = [level]
        else:
            levels_to_drop = list(level)

        drop_indices = [self._resolve_level(lev) for lev in levels_to_drop]
        keep_indices = [i for i in range(self.nlevels) if i not in drop_indices]

        if len(keep_indices) == 0:
            return Index([], name=None)
        if len(keep_indices) == 1:
            return self.get_level_values(keep_indices[0])
        return self.reorder_levels(keep_indices)

    def _resolve_level(self, level: Union[int, str]) -> int:
        """将 level 名称解析为索引。"""
        if isinstance(level, int):
            if level < 0 or level >= self.nlevels:
                raise IndexError(f"level {level} out of range [0, {self.nlevels})")
            return level
        try:
            return self._names.index(level)
        except ValueError:
            raise KeyError(f"level name '{level}' not found")

    def _slice_by_indices(self, indices) -> MultiIndex:
        """按索引列表切片。"""
        new_codes = [[c[i] for i in indices] for c in self._codes]
        return MultiIndex(self._levels, new_codes, names=self._names)

    def copy(self) -> MultiIndex:
        return MultiIndex([list(level) for level in self._levels], [list(c) for c in self._codes], names=list(self._names))

    # ---------- v2.0.0: MultiIndex 扩展 ----------

    def set_levels(self, levels, level=None, verify_integrity: bool = True) -> MultiIndex:
        """设置 levels。

        :param levels: 新的 levels 列表
        :param level: 要设置的 level 索引 (None 表示全部)
        :param verify_integrity: 是否验证完整性
        """
        if level is not None:
            if isinstance(level, int):
                new_levels = [list(self._levels[i]) if i != level else list(levels) for i in range(self.nlevels)]
            else:
                new_levels = [list(lev) for lev in self._levels]
            new_names = list(self._names)
        else:
            new_levels = [list(lev) for lev in levels]
            new_names = list(self._names)
        return MultiIndex(new_levels, [list(c) for c in self._codes], names=new_names)

    def set_codes(self, codes, level=None, verify_integrity: bool = True) -> MultiIndex:
        """设置 codes。

        :param codes: 新的 codes 列表
        :param level: 要设置的 level 索引 (None 表示全部)
        :param verify_integrity: 是否验证完整性
        """
        if level is not None:
            if isinstance(level, int):
                new_codes = [list(self._codes[i]) if i != level else list(codes) for i in range(self.nlevels)]
            else:
                new_codes = [list(c) for c in self._codes]
        else:
            new_codes = [list(c) for c in codes]
        return MultiIndex([list(lvl) for lvl in self._levels], new_codes, names=list(self._names))


# ============================================================================
# IntervalIndex - 区间索引 (v2.0.0)
# ============================================================================

class IntervalIndex(Index):
    """区间索引，用于存储区间（左闭右闭或左闭右开）。

    Examples:
        >>> ii = IntervalIndex.from_breaks([0, 1, 2, 3])
        >>> len(ii)
        3
        >>> ii[0]
        (0, 1)
    """

    def __init__(self, data, closed: str = "right", name: Optional[str] = None):
        """构造 IntervalIndex。

        :param data: 区间列表，每个元素为 (left, right) tuple
        :param closed: 'left' / 'right' / 'both' / 'neither'
        :param name: 索引名称
        """
        self._data = [tuple(d) if d is not None else None for d in data]
        self._closed = closed
        self._name = name

    @property
    def closed(self) -> str:
        return self._closed

    @property
    def left(self) -> Index:
        return Index([d[0] if d is not None else None for d in self._data])

    @property
    def right(self) -> Index:
        return Index([d[1] if d is not None else None for d in self._data])

    @property
    def mid(self) -> Index:
        return Index([(d[0] + d[1]) / 2 if d is not None else None for d in self._data])

    @property
    def length(self) -> Index:
        return Index([d[1] - d[0] if d is not None else None for d in self._data])

    @property
    def is_empty(self) -> list:
        return [d[0] >= d[1] if d is not None else None for d in self._data]

    def __repr__(self) -> str:
        name = f", name='{self._name}'" if self._name else ""
        return f"IntervalIndex({self._data}, closed='{self._closed}'{name})"

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._data[key]
        if isinstance(key, slice):
            return IntervalIndex(self._data[key], closed=self._closed, name=self._name)
        if isinstance(key, (list, tuple)):
            return IntervalIndex([self._data[i] for i in key], closed=self._closed, name=self._name)
        raise TypeError(f"IntervalIndex key must be int/slice/list, not {type(key).__name__}")

    def __contains__(self, item) -> bool:
        if isinstance(item, tuple) and len(item) == 2:
            return item in self._data
        # 单值查找: 判断属于哪个区间
        for d in self._data:
            if d is None:
                continue
            if self._closed == "right":
                if d[0] < item <= d[1]:
                    return True
            elif self._closed == "left":
                if d[0] <= item < d[1]:
                    return True
            elif self._closed == "both":
                if d[0] <= item <= d[1]:
                    return True
            else:
                if d[0] < item < d[1]:
                    return True
        return False

    def __eq__(self, other) -> bool:
        if isinstance(other, IntervalIndex):
            return self._data == other._data and self._closed == other._closed
        return False

    def tolist(self) -> list:
        return list(self._data)

    def to_list(self) -> list:
        return self.tolist()

    def get_loc(self, key) -> int:
        """返回 key 在 IntervalIndex 中的位置。"""
        if isinstance(key, tuple) and len(key) == 2:
            try:
                return self._data.index(key)
            except ValueError:
                raise KeyError(key)
        # 单值查找
        for i, d in enumerate(self._data):
            if d is None:
                continue
            if self._closed == "right":
                if d[0] < key <= d[1]:
                    return i
            elif self._closed == "left":
                if d[0] <= key < d[1]:
                    return i
            elif self._closed == "both":
                if d[0] <= key <= d[1]:
                    return i
            else:
                if d[0] < key < d[1]:
                    return i
        raise KeyError(key)

    def contains(self, other) -> list:
        """检查每个值是否在某个区间内。"""
        return [v in self for v in other]

    def overlaps(self, other) -> list:
        """检查每个区间是否与另一个区间重叠。"""
        if isinstance(other, tuple) and len(other) == 2:
            other_itv = other
            return [self._overlaps(d, other_itv) if d is not None else None for d in self._data]
        if isinstance(other, IntervalIndex):
            return [self._overlaps(d1, d2) if d1 is not None and d2 is not None else None
                    for d1, d2 in zip(self._data, other._data)]
        raise TypeError("overlaps requires IntervalIndex or tuple")

    def _overlaps(self, a, b) -> bool:
        if a is None or b is None:
            return False
        return a[0] < b[1] and b[0] < a[1]

    @staticmethod
    def from_breaks(breaks, closed: str = "right", name=None) -> IntervalIndex:
        """从断点列表构造 IntervalIndex。

        :param breaks: 断点列表 (如 [0, 1, 2, 3])
        :param closed: 'left' / 'right' / 'both' / 'neither'
        :param name: 索引名称
        """
        intervals = [(breaks[i], breaks[i + 1]) for i in range(len(breaks) - 1)]
        return IntervalIndex(intervals, closed=closed, name=name)

    @staticmethod
    def from_arrays(left, right, closed: str = "right", name=None) -> IntervalIndex:
        """从左右端点数组构造 IntervalIndex。

        :param left: 左端点列表
        :param right: 右端点列表
        :param closed: 'left' / 'right' / 'both' / 'neither'
        :param name: 索引名称
        """
        intervals = list(zip(left, right))
        return IntervalIndex(intervals, closed=closed, name=name)

    @staticmethod
    def from_tuples(tuples, closed: str = "right", name=None) -> IntervalIndex:
        """从 tuple 列表构造 IntervalIndex。

        :param tuples: (left, right) tuple 列表
        :param closed: 'left' / 'right' / 'both' / 'neither'
        :param name: 索引名称
        """
        return IntervalIndex(tuples, closed=closed, name=name)


# ============================================================================
# CategoricalIndex - 分类索引 (v2.0.0)
# ============================================================================

class CategoricalIndex(Index):
    """分类索引，用于存储有限类别。

    Examples:
        >>> ci = CategoricalIndex(['a', 'b', 'a', 'c'])
        >>> ci.categories
        ['a', 'b', 'c']
        >>> ci.codes
        [0, 1, 0, 2]
    """

    def __init__(self, data, categories=None, ordered: bool = False, name: Optional[str] = None):
        """构造 CategoricalIndex。

        :param data: 数据列表
        :param categories: 预定义类别列表 (None 则自动推断)
        :param ordered: 是否有序
        :param name: 索引名称
        """
        self._data = list(data)
        if categories is None:
            seen = set()
            self._categories = []
            for v in self._data:
                if v is not None and v not in seen:
                    self._categories.append(v)
                    seen.add(v)
        else:
            self._categories = list(categories)
        self._ordered = ordered
        self._name = name

        # 构建 codes
        cat_to_code = {c: i for i, c in enumerate(self._categories)}
        self._codes = [cat_to_code.get(v, -1) if v is not None else -1 for v in self._data]

    @property
    def categories(self) -> list:
        return list(self._categories)

    @property
    def codes(self) -> list:
        return list(self._codes)

    @property
    def ordered(self) -> bool:
        return self._ordered

    def __repr__(self) -> str:
        name = f", name='{self._name}'" if self._name else ""
        return f"CategoricalIndex({self._data}, categories={self._categories}{name})"

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._data[key]
        if isinstance(key, slice):
            return CategoricalIndex(self._data[key], categories=self._categories,
                                    ordered=self._ordered, name=self._name)
        if isinstance(key, (list, tuple)):
            return CategoricalIndex([self._data[i] for i in key], categories=self._categories,
                                    ordered=self._ordered, name=self._name)
        raise TypeError(f"CategoricalIndex key must be int/slice/list, not {type(key).__name__}")

    def __contains__(self, item) -> bool:
        return item in self._data

    def tolist(self) -> list:
        return list(self._data)

    def to_list(self) -> list:
        return self.tolist()

    def get_loc(self, key) -> int:
        try:
            return self._data.index(key)
        except ValueError:
            raise KeyError(key)

    def add_categories(self, new_categories) -> CategoricalIndex:
        """添加新类别。"""
        new_cats = list(self._categories)
        for c in new_categories:
            if c not in new_cats:
                new_cats.append(c)
        return CategoricalIndex(self._data, categories=new_cats, ordered=self._ordered, name=self._name)

    def remove_unused_categories(self) -> CategoricalIndex:
        """移除未使用的类别。"""
        used = set(v for v in self._data if v is not None)
        new_cats = [c for c in self._categories if c in used]
        return CategoricalIndex(self._data, categories=new_cats, ordered=self._ordered, name=self._name)

    def rename_categories(self, new_categories) -> CategoricalIndex:
        """重命名类别。"""
        if len(new_categories) != len(self._categories):
            raise ValueError("new_categories must have same length as categories")
        mapping = dict(zip(self._categories, new_categories))
        new_data = [mapping.get(v, v) if v is not None else None for v in self._data]
        return CategoricalIndex(new_data, categories=list(new_categories), ordered=self._ordered, name=self._name)

    def as_ordered(self) -> CategoricalIndex:
        return CategoricalIndex(self._data, categories=self._categories, ordered=True, name=self._name)

    def as_unordered(self) -> CategoricalIndex:
        return CategoricalIndex(self._data, categories=self._categories, ordered=False, name=self._name)


# ============================================================================
# DatetimeIndex - 日期时间索引 (v2.0.0)
# ============================================================================

class DatetimeIndex(Index):
    """日期时间索引，用于存储 datetime 值。

    Examples:
        >>> di = DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03'])
        >>> di.year
        [2024, 2024, 2024]
        >>> di.month
        [1, 1, 1]
    """

    def __init__(self, data, name: Optional[str] = None, tz=None):
        """构造 DatetimeIndex。

        :param data: 日期字符串列表或 datetime 对象列表
        :param name: 索引名称
        :param tz: 时区信息
        """
        self._data = []
        for v in data:
            if v is None:
                self._data.append(None)
            elif isinstance(v, datetime):
                self._data.append(v)
            elif isinstance(v, str):
                try:
                    self._data.append(datetime.fromisoformat(v))
                except (ValueError, TypeError):
                    from .datetime import _parse_iso
                    self._data.append(_parse_iso(v))
            else:
                self._data.append(v)
        self._name = name
        self._tz = tz

    @property
    def year(self) -> Index:
        return Index([v.year if isinstance(v, datetime) else None for v in self._data])

    @property
    def month(self) -> Index:
        return Index([v.month if isinstance(v, datetime) else None for v in self._data])

    @property
    def day(self) -> Index:
        return Index([v.day if isinstance(v, datetime) else None for v in self._data])

    @property
    def hour(self) -> Index:
        return Index([v.hour if isinstance(v, datetime) else None for v in self._data])

    @property
    def minute(self) -> Index:
        return Index([v.minute if isinstance(v, datetime) else None for v in self._data])

    @property
    def second(self) -> Index:
        return Index([v.second if isinstance(v, datetime) else None for v in self._data])

    @property
    def weekday(self) -> Index:
        return Index([v.weekday() if isinstance(v, datetime) else None for v in self._data])

    @property
    def dayofyear(self) -> Index:
        return Index([v.timetuple().tm_yday if isinstance(v, datetime) else None for v in self._data])

    @property
    def quarter(self) -> Index:
        return Index([(v.month - 1) // 3 + 1 if isinstance(v, datetime) else None for v in self._data])

    @property
    def tz(self):
        return self._tz

    def __repr__(self) -> str:
        name = f", name='{self._name}'" if self._name else ""
        tz_str = f", tz='{self._tz}'" if self._tz else ""
        return f"DatetimeIndex({self._data}{name}{tz_str})"

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._data[key]
        if isinstance(key, slice):
            return DatetimeIndex(self._data[key], name=self._name, tz=self._tz)
        if isinstance(key, (list, tuple)):
            return DatetimeIndex([self._data[i] for i in key], name=self._name, tz=self._tz)
        raise TypeError(f"DatetimeIndex key must be int/slice/list, not {type(key).__name__}")

    def __contains__(self, item) -> bool:
        return item in self._data

    def tolist(self) -> list:
        return list(self._data)

    def to_list(self) -> list:
        return self.tolist()

    def get_loc(self, key) -> int:
        # 支持字符串 key 查找
        if isinstance(key, str):
            try:
                key_dt = datetime.fromisoformat(key)
            except (ValueError, TypeError):
                raise KeyError(key)
            for i, v in enumerate(self._data):
                if isinstance(v, datetime) and v == key_dt:
                    return i
            raise KeyError(key)
        try:
            return self._data.index(key)
        except ValueError:
            raise KeyError(key)

    def strftime(self, fmt: str) -> Index:
        return Index([v.strftime(fmt) if isinstance(v, datetime) else None for v in self._data])

    def normalize(self) -> DatetimeIndex:
        """将时间归一化到午夜。"""
        return DatetimeIndex(
            [v.replace(hour=0, minute=0, second=0, microsecond=0) if isinstance(v, datetime) else None
             for v in self._data],
            name=self._name, tz=self._tz,
        )

    def floor(self, freq: str) -> DatetimeIndex:
        """向下舍入到指定频率。"""
        freq = freq.strip().upper()
        
        def _floor(dt):
            if dt is None:
                return None
            if freq == "D":
                return dt.replace(hour=0, minute=0, second=0, microsecond=0)
            elif freq == "H":
                return dt.replace(minute=0, second=0, microsecond=0)
            elif freq in ("M", "T", "min"):
                return dt.replace(second=0, microsecond=0)
            elif freq == "S":
                return dt.replace(microsecond=0)
            return dt
        return DatetimeIndex([_floor(v) for v in self._data], name=self._name, tz=self._tz)

    def ceil(self, freq: str) -> DatetimeIndex:
        """向上舍入到指定频率。"""
        freq = freq.strip().upper()

        def _ceil(dt):
            if dt is None:
                return None
            floored = None
            if freq == "D":
                floored = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            elif freq == "H":
                floored = dt.replace(minute=0, second=0, microsecond=0)
            elif freq in ("M", "T", "min"):
                floored = dt.replace(second=0, microsecond=0)
            elif freq == "S":
                floored = dt.replace(microsecond=0)
            if floored == dt:
                return dt
            if freq == "D":
                return floored + timedelta(days=1)
            elif freq == "H":
                return floored + timedelta(hours=1)
            elif freq in ("M", "T", "min"):
                return floored + timedelta(minutes=1)
            elif freq == "S":
                return floored + timedelta(seconds=1)
            return dt
        return DatetimeIndex([_ceil(v) for v in self._data], name=self._name, tz=self._tz)

    def round(self, freq: str) -> DatetimeIndex:
        """四舍五入到指定频率。"""
        freq = freq.strip().upper()

        def _round(dt):
            if dt is None:
                return None
            floored = None
            next_tick = None
            if freq == "D":
                floored = dt.replace(hour=0, minute=0, second=0, microsecond=0)
                next_tick = floored + timedelta(days=1)
            elif freq == "H":
                floored = dt.replace(minute=0, second=0, microsecond=0)
                next_tick = floored + timedelta(hours=1)
            elif freq in ("M", "T", "min"):
                floored = dt.replace(second=0, microsecond=0)
                next_tick = floored + timedelta(minutes=1)
            elif freq == "S":
                floored = dt.replace(microsecond=0)
                next_tick = floored + timedelta(seconds=1)
            if dt - floored < next_tick - dt:
                return floored
            return next_tick
        return DatetimeIndex([_round(v) for v in self._data], name=self._name, tz=self._tz)


# ============================================================================
# TimedeltaIndex - 时间差索引 (v2.0.0)
# ============================================================================

class TimedeltaIndex(Index):
    """时间差索引，用于存储 timedelta 值。

    Examples:
        >>> ti = TimedeltaIndex([timedelta(days=1), timedelta(days=2)])
        >>> ti.days
        [1, 2]
    """

    def __init__(self, data, name: Optional[str] = None):
        """构造 TimedeltaIndex。

        :param data: timedelta 对象列表或字符串列表
        :param name: 索引名称
        """
        self._data = []
        for v in data:
            if v is None:
                self._data.append(None)
            elif isinstance(v, timedelta):
                self._data.append(v)
            elif isinstance(v, (int, float)):
                self._data.append(timedelta(seconds=float(v)))
            elif isinstance(v, str):
                self._data.append(self._parse_timedelta(v))
            else:
                self._data.append(v)
        self._name = name

    @staticmethod
    def _parse_timedelta(s: str) -> timedelta:
        """解析 timedelta 字符串 (如 '1 days 02:00:00')。"""
        s = s.strip()
        if "days" in s:
            parts = s.split(" days ")
            days = int(parts[0].strip())
            time_str = parts[1] if len(parts) > 1 else "00:00:00"
            time_parts = time_str.split(":")
            hours = int(time_parts[0]) if len(time_parts) > 0 else 0
            minutes = int(time_parts[1]) if len(time_parts) > 1 else 0
            seconds = float(time_parts[2]) if len(time_parts) > 2 else 0
            return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        # 尝试 HH:MM:SS 格式
        time_parts = s.split(":")
        if len(time_parts) >= 2:
            hours = int(time_parts[0]) if len(time_parts) > 0 else 0
            minutes = int(time_parts[1]) if len(time_parts) > 1 else 0
            seconds = float(time_parts[2]) if len(time_parts) > 2 else 0
            return timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return timedelta(seconds=float(s))

    @property
    def days(self) -> Index:
        return Index([v.days if isinstance(v, timedelta) else None for v in self._data])

    @property
    def seconds(self) -> Index:
        return Index([v.seconds if isinstance(v, timedelta) else None for v in self._data])

    @property
    def microseconds(self) -> Index:
        return Index([v.microseconds if isinstance(v, timedelta) else None for v in self._data])

    @property
    def total_seconds(self) -> Index:
        return Index([v.total_seconds() if isinstance(v, timedelta) else None for v in self._data])

    def __repr__(self) -> str:
        name = f", name='{self._name}'" if self._name else ""
        return f"TimedeltaIndex({self._data}{name})"

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._data[key]
        if isinstance(key, slice):
            return TimedeltaIndex(self._data[key], name=self._name)
        if isinstance(key, (list, tuple)):
            return TimedeltaIndex([self._data[i] for i in key], name=self._name)
        raise TypeError(f"TimedeltaIndex key must be int/slice/list, not {type(key).__name__}")

    def __contains__(self, item) -> bool:
        return item in self._data

    def tolist(self) -> list:
        return list(self._data)

    def to_list(self) -> list:
        return self.tolist()

    def get_loc(self, key) -> int:
        # 支持字符串 key 查找
        if isinstance(key, str):
            key_td = self._parse_timedelta(key)
            for i, v in enumerate(self._data):
                if isinstance(v, timedelta) and v == key_td:
                    return i
            raise KeyError(key)
        try:
            return self._data.index(key)
        except ValueError:
            raise KeyError(key)


# ============================================================================
# PeriodIndex - 时期索引 (v2.0.0)
# ============================================================================

class PeriodIndex(Index):
    """时期索引，用于存储固定频率的时期。

    Examples:
        >>> pi = PeriodIndex(['2024-01', '2024-02', '2024-03'], freq='M')
        >>> pi.year
        [2024, 2024, 2024]
        >>> pi.month
        [1, 2, 3]
    """

    def __init__(self, data, freq: str = "M", name: Optional[str] = None):
        """构造 PeriodIndex。

        :param data: 时期字符串列表或 datetime 对象列表
        :param freq: 频率 ('M'/'Q'/'Y'/'D'/'H' 等)
        :param name: 索引名称
        """
        self._data = []
        self._freq = freq
        for v in data:
            if v is None:
                self._data.append(None)
            elif isinstance(v, datetime):
                self._data.append(v)
            elif isinstance(v, str):
                from .datetime import _parse_iso
                # Try to parse as period string (e.g. '2024-01', '2024-Q1')
                try:
                    self._data.append(_parse_iso(v))
                except ValueError:
                    # Try period-specific parsing
                    try:
                        # '2024-01' -> treat as first day of month
                        if '-' in v and len(v) == 7:  # 'YYYY-MM'
                            self._data.append(datetime.strptime(v + "-01", "%Y-%m-%d"))
                        elif '-' in v and len(v) == 4:  # 'YYYY'
                            self._data.append(datetime.strptime(v + "-01-01", "%Y-%m-%d"))
                        else:
                            self._data.append(v)
                    except ValueError:
                        self._data.append(v)
            else:
                self._data.append(v)
        self._name = name

    @property
    def freq(self) -> str:
        return self._freq

    @property
    def year(self) -> Index:
        return Index([v.year if isinstance(v, datetime) else None for v in self._data])

    @property
    def month(self) -> Index:
        return Index([v.month if isinstance(v, datetime) else None for v in self._data])

    @property
    def day(self) -> Index:
        return Index([v.day if isinstance(v, datetime) else None for v in self._data])

    @property
    def quarter(self) -> Index:
        return Index([(v.month - 1) // 3 + 1 if isinstance(v, datetime) else None for v in self._data])

    @property
    def start_time(self) -> DatetimeIndex:
        return DatetimeIndex([v if isinstance(v, datetime) else None for v in self._data])

    @property
    def end_time(self) -> DatetimeIndex:
        import calendar

        def _end(dt):
            if dt is None:
                return None
            if self._freq == "M":
                last_day = calendar.monthrange(dt.year, dt.month)[1]
                return dt.replace(day=last_day, hour=23, minute=59, second=59)
            elif self._freq == "Q":
                m = ((dt.month - 1) // 3 + 1) * 3
                last_day = calendar.monthrange(dt.year, m)[1]
                return dt.replace(month=m, day=last_day, hour=23, minute=59, second=59)
            elif self._freq == "Y":
                return dt.replace(month=12, day=31, hour=23, minute=59, second=59)
            return dt
        return DatetimeIndex([_end(v) for v in self._data])

    def __repr__(self) -> str:
        name = f", name='{self._name}'" if self._name else ""
        return f"PeriodIndex({self._data}, freq='{self._freq}'{name})"

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._data[key]
        if isinstance(key, slice):
            return PeriodIndex(self._data[key], freq=self._freq, name=self._name)
        if isinstance(key, (list, tuple)):
            return PeriodIndex([self._data[i] for i in key], freq=self._freq, name=self._name)
        raise TypeError(f"PeriodIndex key must be int/slice/list, not {type(key).__name__}")

    def __contains__(self, item) -> bool:
        return item in self._data

    def tolist(self) -> list:
        return list(self._data)

    def to_list(self) -> list:
        return self.tolist()

    def get_loc(self, key) -> int:
        if isinstance(key, str):
            from .datetime import _parse_iso
            key_dt = _parse_iso(key)
            for i, v in enumerate(self._data):
                if isinstance(v, datetime) and v == key_dt:
                    return i
            raise KeyError(key)
        try:
            return self._data.index(key)
        except ValueError:
            raise KeyError(key)

    def strftime(self, fmt: str) -> Index:
        return Index([v.strftime(fmt) if isinstance(v, datetime) else None for v in self._data])

    def asfreq(self, freq: str) -> PeriodIndex:
        """转换为不同频率。"""
        return PeriodIndex(self._data, freq=freq, name=self._name)


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
    from .dataframe import DataFrame as _DataFrame

    if isinstance(data, Series):
        values = data.values
        name = data.name or "col"
        return _get_dummies_series(values, prefix or name, prefix_sep)

    if isinstance(data, _DataFrame):
        if columns is None:
            columns = [c for c in data.columns
                       if data[c].dtype in ("object", "category", "bool")]
        result = data
        for col in columns:
            dummies = _get_dummies_series(
                list(data[col].values),
                prefix=prefix if isinstance(prefix, str) else (prefix[columns.index(col)] if prefix else col),
                sep=prefix_sep,
            )
            result = _concat_frames([result, dummies])
        return result

    raise TypeError(f"get_dummies expected Series or DataFrame, got {type(data).__name__}")


def _get_dummies_series(values: list, prefix: str, sep: str) -> rspandas_DataFrame:
    """对单个 Series 做 one-hot 编码。"""
    from .dataframe import DataFrame as _DataFrame

    # 获取唯一值
    unique_vals = []
    seen = set()
    for v in values:
        if v is not None and v not in seen:
            unique_vals.append(v)
            seen.add(v)

    result_data = {}
    for uv in unique_vals:
        col_name = f"{prefix}{sep}{uv}"
        result_data[col_name] = [1 if v == uv else 0 for v in values]

    return _DataFrame(result_data)


def _concat_frames(frames: list) -> rspandas_DataFrame:
    """横向拼接 DataFrame，去掉重复列。"""
    from .dataframe import DataFrame as _DataFrame

    if not frames:
        return _DataFrame({})
    if len(frames) == 1:
        return frames[0]

    result_data = {}
    for df in frames:
        for col in df.columns:
            if col not in result_data:
                result_data[col] = list(df[col].values)

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
    values = list(x.values) if isinstance(x, Series) else list(x)

    # 计算 bins
    non_null = [v for v in values if v is not None]
    if not non_null:
        return Series([None] * len(values), dtype="category")

    if isinstance(bins, int):
        min_val = min(non_null)
        max_val = max(non_null)
        if min_val == max_val:
            # 只有一个唯一值
            bins = [min_val - 0.5, min_val + 0.5]
        else:
            bin_width = (max_val - min_val) / bins
            bins = [min_val + i * bin_width for i in range(bins + 1)]
    else:
        bins = list(bins)

    n_bins = len(bins) - 1

    # 生成标签
    if labels is None:
        labels = []
        for i in range(n_bins):
            left = bins[i]
            right_val = bins[i + 1]
            if right:
                labels.append(f"({left}, {right_val}]")
            else:
                labels.append(f"[{left}, {right_val})")
        if include_lowest:
            if right:
                labels[0] = f"[{bins[0]}, {bins[1]}]"
            else:
                labels[-1] = f"[{bins[-2]}, {bins[-1]}]"
    else:
        labels = list(labels)

    # 分配区间
    result = []
    for v in values:
        if v is None:
            result.append(None)
            continue

        assigned = False
        for i in range(n_bins):
            if right:
                if i == 0 and include_lowest:
                    if bins[0] <= v <= bins[1]:
                        result.append(labels[i])
                        assigned = True
                        break
                if bins[i] < v <= bins[i + 1]:
                    result.append(labels[i])
                    assigned = True
                    break
            else:
                if i == n_bins - 1 and include_lowest:
                    if bins[-2] <= v <= bins[-1]:
                        result.append(labels[i])
                        assigned = True
                        break
                if bins[i] <= v < bins[i + 1]:
                    result.append(labels[i])
                    assigned = True
                    break

        if not assigned:
            result.append(None)

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
        分位数数量或分位点列表。
    labels : list, optional
        区间标签。

    Returns
    -------
    Series
    """
    values = list(x.values) if isinstance(x, Series) else list(x)

    non_null = sorted([v for v in values if v is not None])
    if not non_null:
        return Series([None] * len(values), dtype="category")

    n = len(non_null)

    if isinstance(q, int):
        n_bins = q
        # 计算分位点
        quantiles = []
        for i in range(n_bins + 1):
            pos = i * (n - 1) / n_bins
            lo = int(pos)
            hi = min(lo + 1, n - 1)
            frac = pos - lo
            quantiles.append(non_null[lo] + frac * (non_null[hi] - non_null[lo]))
    else:
        quantiles = sorted(q)
        n_bins = len(quantiles) - 1

    # 使用 cut 进行分箱
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
    from .dataframe import DataFrame as _DataFrame

    idx_vals = list(index.values) if isinstance(index, Series) else list(index)
    col_vals = list(columns.values) if isinstance(columns, Series) else list(columns)

    if values is not None:
        val_vals = list(values.values) if isinstance(values, Series) else list(values)
    else:
        val_vals = [1] * len(idx_vals)

    # 收集所有 index 和 column 值
    unique_idx = []
    idx_seen = set()
    for v in idx_vals:
        if v not in idx_seen:
            unique_idx.append(v)
            idx_seen.add(v)

    unique_col = []
    col_seen = set()
    for v in col_vals:
        if v not in col_seen:
            unique_col.append(v)
            col_seen.add(v)

    # 构建交叉表
    groups: Dict[tuple, list] = {}
    for i, iv, cv in zip(range(len(idx_vals)), idx_vals, col_vals):
        key = (iv, cv)
        if key not in groups:
            groups[key] = []
        groups[key].append(val_vals[i])

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

    # 构建结果
    result_data: Dict[str, list] = {}
    result_data[""] = unique_idx  # 行索引列

    for cv in unique_col:
        col_data = []
        for iv in unique_idx:
            col_data.append(_agg(groups.get((iv, cv), [])))
        result_data[str(cv)] = col_data

    df = _DataFrame(result_data)

    # 边际汇总
    if margins:
        row_sums = []
        for iv in unique_idx:
            row_sum = 0
            for cv in unique_col:
                row_sum += _agg(groups.get((iv, cv), []))
            row_sums.append(row_sum)
        df["All"] = row_sums

        col_sums = [""]  # 索引列名
        for cv in unique_col:
            col_sum = sum(_agg(groups.get((iv, cv), [])) for iv in unique_idx)
            col_sums.append(col_sum)
        if margins:
            col_sums.append(sum(row_sums))

        # 添加汇总行
        df_data = {c: list(df[c].values) + [col_sums[i]]
                   for i, c in enumerate(df.columns)}
        df = _DataFrame(df_data)

    # 归一化
    if normalize is True or normalize == "all":
        total = sum(
            sum(v for v in df[c].values if v is not None)
            for c in df.columns if c != ""
        )
        if total > 0:
            for c in df.columns:
                if c != "":
                    _ = [v / total for v in df[c].values]  # noqa: F841
    elif normalize == "index":
        pass  # 按行归一化 (预留)
    elif normalize == "columns":
        pass  # 按列归一化 (预留)

    return df
