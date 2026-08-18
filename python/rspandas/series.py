"""Series: pandas-like 1D data structure with Rust backend."""

from __future__ import annotations

import rsnumpy as rnp

from .rspandas import _DataFrame as _PyDataFrame
from .rspandas import _Series as _PySeries  # type: ignore
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple

if TYPE_CHECKING:
    # 仅用于类型注解，运行时通过函数内 import 避免循环引用
    from .dataframe import DataFrame

# ---------------------------------------------------------------------------
# 内部辅助函数与工具类（已迁移到 _internal/_series_helpers）
# ---------------------------------------------------------------------------
from ._internal._series_helpers import (
    _AlignmentResult,
    _DtypeScalar,
    _ExtensionArray,
    _PySeries_filter,
    _dtype_to_str,
    _format_timedelta,
    _infer_dtype,
    _is_missing,
    _is_range_index,
    _to_python_list,
    _to_python_list_and_index,
)

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
        index=None,
        dtype: Optional[str] = None,
        name: Optional[str] = None,
        copy: bool = False,
        fastpath: bool = False,
    ):
        """构造 Series。

        :param data: list / tuple / scalar / Series / dict
        :param index: 索引 (list / RangeIndex)
        :param dtype: 可选类型 ('int64' / 'float64' / 'bool' / 'object')
        :param name: 列名
        :param copy: 是否复制数据
        :param fastpath: 是否走快速路径 (内部使用)
        """
        from ._datetime import DatetimeSeries  # 延迟 import 避免循环引用
        from .indexes import PeriodIndex

        # 额外的 datetime 缓存（当源是 DatetimeSeries / datetime 对象列表时保留）
        self._dt_values: Optional[list] = None
        self._dt_tz: Any = None
        # Period 频率缓存（当源是 PeriodIndex 时保留）
        self._period_freq: Optional[str] = None
        # timedelta 缓存（当源含 timedelta 对象时保留）
        self._td_values: Optional[list] = None

        # 如果输入是 Series，直接复制
        if isinstance(data, Series):
            if copy:
                values = list(data.values)
                index = list(data._index) if data._index is not None else None
                name = data.name if name is None else name
                dtype = data._dtype_str if dtype is None else dtype
            else:
                values = list(data.values)
                index = list(data._index) if data._index is not None else None
                name = data.name if name is None else name
                dtype = data._dtype_str if dtype is None else dtype
            # 如果原始 Series 有 datetime 缓存，同步
            if getattr(data, "_dt_values", None) is not None:
                self._dt_values = list(data._dt_values)
                self._dt_tz = getattr(data, "_dt_tz", None)
            # 如果原始 Series 有 timedelta 缓存，同步
            if getattr(data, "_td_values", None) is not None:
                self._td_values = list(data._td_values)
            # 如果原始 Series 有 period 频率，同步
            if getattr(data, "_period_freq", None) is not None:
                self._period_freq = data._period_freq
        elif isinstance(data, DatetimeSeries):
            # 直接取内部 ISO 字符串 values，同时保留 datetime 列表缓存
            values = list(data._inner.values)
            if name is None:
                name = data.name
            if index is None and data._index is not None:
                index = list(data._index)
            self._dt_values = list(data.values)  # datetime 对象列表
            self._dt_tz = data._tz
            if dtype is None:
                dtype = "datetime64[ns]"  # 语义 dtype（底层仍用 object 存 ISO）
        elif isinstance(data, PeriodIndex):
            # PeriodIndex: 保留 datetime 列表和 freq 信息
            values = [
                (v.strftime("%Y-%m-%d") if isinstance(v, datetime) else v)
                for v in data._data
            ]
            if name is None:
                name = data._name
            self._dt_values = list(data._data)
            self._period_freq = data._freq
            if dtype is None:
                dtype = f"period[{data._freq}]"
        elif isinstance(data, dict):
            values, index = _to_python_list_and_index(data, index)
        else:
            # 检查是否为标量输入（int/float/str/bool），如果是且有 index，则广播
            # 排除 list/tuple/_PySeries/Series/dict/range 以及有 tolist 方法的数组类型
            if (
                index is not None
                and not isinstance(data, (list, tuple, _PySeries, Series, dict, range))
                and not hasattr(data, "tolist")
            ):
                # 标量广播到 index 长度
                index_len = (
                    len(index) if hasattr(index, "__len__") else len(list(index))
                )
                values = [data] * index_len
            else:
                # 检测原始数据是否含 datetime/timedelta（在转换前先保留缓存）
                raw_iter = None
                if isinstance(data, (list, tuple)):
                    raw_iter = data
                elif hasattr(data, "__iter__") and not hasattr(data, "tolist"):
                    raw_iter = list(data)
                values = _to_python_list(data)
                # 如果原始数据含 datetime/date，则保留对象缓存（用于 to_numpy dtype=object）
                if raw_iter is not None and any(
                    isinstance(v, (datetime, date)) and not isinstance(v, bool)
                    for v in raw_iter
                ):
                    self._dt_values = [
                        v
                        for v in (
                            raw_iter if isinstance(raw_iter, list) else list(raw_iter)
                        )
                    ]
                    tz_infos = {
                        v.tzinfo
                        for v in self._dt_values
                        if isinstance(v, datetime) and v.tzinfo is not None
                    }
                    if len(tz_infos) == 1:
                        self._dt_tz = next(iter(tz_infos))
                # 如果原始数据含 timedelta，则保留对象缓存
                if raw_iter is not None and any(
                    isinstance(v, timedelta) for v in raw_iter
                ):
                    self._td_values = [
                        v
                        for v in (
                            raw_iter if isinstance(raw_iter, list) else list(raw_iter)
                        )
                        if isinstance(v, timedelta)
                    ]

        # 推断 dtype
        if dtype is None:
            dtype = _infer_dtype(values)

        # 规范化 dtype：type 对象（如 np.uint8/bool/int）→ 字符串
        # Rust 层 _PySeries 要求 dtype: Option<&str>，不能接受 type 对象
        if dtype is not None and not isinstance(dtype, str):
            dtype_str = _dtype_to_str(dtype)
        else:
            dtype_str = dtype

        # 构造 Rust 端 Series (传递 dtype 以支持 category 等类型)
        self._inner = _PySeries(values, name, dtype=dtype_str)

        # 缓存 dtype
        if dtype is not None:
            nd = dtype_str.lower()
            if nd == "category":
                self._dtype_str = "category"
            elif nd in ("str", "string"):
                self._dtype_str = "str"
            elif nd in (
                "float32",
                "float64",
                "float",
                "int8",
                "int16",
                "int32",
                "int64",
                "int",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
            ):
                # 保留显式指定的数值子类型（对齐 pandas 行为）
                # Rust 层仅支持 int64/float64，但 Python 层 _dtype_str 追踪精确子类型
                self._dtype_str = nd
            elif nd.startswith("period["):
                # 保留 period[freq] 格式（与 pandas 一致）
                self._dtype_str = dtype_str
            else:
                self._dtype_str = self._inner.dtype
        else:
            self._dtype_str: str = self._inner.dtype
            # Rust 层将 str 映射为 object，但若推断为 str 则使用 str
            if self._dtype_str == "object":
                inferred = _infer_dtype(values)
                if inferred == "str":
                    self._dtype_str = "str"

        # 若存在 datetime 缓存，dtype 应为 datetime64[us]（与 pandas 一致）
        # 但 period 类型优先使用 period[freq] dtype
        if self._dt_values and self._period_freq is None:
            tz = getattr(self, "_dt_tz", None)
            if tz is not None:
                tz_name = str(tz)
                self._dtype_str = f"datetime64[us, {tz_name}]"
            else:
                self._dtype_str = "datetime64[us]"

        # 若存在 timedelta 缓存，dtype 应为 timedelta64[us]（与 pandas 一致）
        if self._td_values and self._period_freq is None and not self._dt_values:
            self._dtype_str = "timedelta64[us]"

        # RangeIndex 或自定义索引
        from .indexes import Index, RangeIndex, MultiIndex, DatetimeIndex

        # 缓存传入的 Index 对象引用，用于索引共享 (rs.index is df.index)
        self._cached_index_ref: Optional[object] = None
        if isinstance(index, (Index, RangeIndex, MultiIndex, DatetimeIndex)):
            self._cached_index_ref = index
            self._index = list(index._data) if hasattr(index, "_data") else list(index)
        elif index is not None:
            self._index = list(index)
        else:
            self._index = list(range(len(values)))

        # 频率信息（从 DatetimeIndex 等传入）
        self._freq: Optional[str] = (
            getattr(index, "_freq", None) if index is not None else None
        )

    # ---------- 属性 ----------

    @property
    def shape(self) -> Tuple[int]:
        return self._inner.shape

    @property
    def dtype(self) -> str:
        return self._dtype_str

    @property
    def dtypes(self) -> str:
        """dtype 的别名。"""
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
        from .indexes import Index, RangeIndex

        # 优先返回缓存的 Index 对象引用，实现索引共享
        if self._cached_index_ref is not None:
            return self._cached_index_ref
        if _is_range_index(self._index):
            result = RangeIndex(len(self._index))
        else:
            result = Index(self._index)
        # 缓存创建的 Index 对象，后续访问返回同一引用
        self._cached_index_ref = result
        return result

    @property
    def ndim(self) -> int:
        return 1

    @property
    def T(self) -> _PySeries:
        """返回自身 (Series 的 T 是自身)。"""
        return self

    @property
    def dt(self):
        """日期时间访问器 (简化版)。"""
        return DatetimeAccessor(self)

    @property
    def plot(self):
        """绘图访问器 - 使用 rsplotlib 绘图。"""
        try:
            from rsplotlib import pyplot as plt

            class _PlotAccessor:
                def __init__(self, series):
                    self._series = series

                def __call__(self, **kwargs):
                    fig, ax = plt.subplots()
                    ax.plot(
                        list(self._series._index), list(self._series.values), **kwargs
                    )
                    return ax

                def line(self, **kwargs):
                    return self.__call__(**kwargs)

                def hist(self, **kwargs):
                    fig, ax = plt.subplots()
                    ax.hist(list(self._series.values), **kwargs)
                    return ax

            return _PlotAccessor(self)
        except ImportError:
            raise NotImplementedError("plot accessor requires rsplotlib")

    @property
    def memory_usage(self) -> int:
        """返回 Series 内存使用量 (字节)。"""
        return self.nbytes

    # ---------- dunder ----------

    def __len__(self) -> int:
        return self._inner.size

    def __iter__(self) -> Iterator:
        return iter(self.values)

    def __contains__(self, item) -> bool:
        """检查值是否在 Series 索引中。"""
        if self._index is None:
            return False
        return item in self._index

    def __repr__(self) -> str:
        return self._format_repr()

    def __str__(self) -> str:
        return self._format_repr()

    def _wrap_scalar(self, value):
        """将标量值包装为带 dtype 的 _DtypeScalar。"""
        if isinstance(value, _DtypeScalar) or value is None:
            return value
        if isinstance(value, bool):
            return _DtypeScalar(value, "bool")
        if isinstance(value, int):
            return _DtypeScalar(value, "int64")
        if isinstance(value, float):
            dtype_str = getattr(self, "_dtype_str", None)
            return _DtypeScalar(value, dtype_str if dtype_str else "float64")
        if isinstance(value, str):
            return _DtypeScalar(value, "str")
        return _DtypeScalar(value, "object")

    def __getitem__(self, key):
        # 自定义 index: 优先按 label 查找
        if self._index is not None and not _is_range_index(self._index):
            if isinstance(key, (str, int, float, bool)):
                try:
                    pos = self._index.index(key)
                    return self._wrap_scalar(self.values[pos])
                except ValueError:
                    raise KeyError(key)
        # RangeIndex 或其他: 走位置
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("index out of range")
            return self._wrap_scalar(self.values[key])
        if isinstance(key, slice):
            values = self.values[key]
            new_index = self._index[key] if self._index is not None else None
            return Series(values, name=self.name, index=new_index)
        if isinstance(key, (list, tuple)) and all(isinstance(x, bool) for x in key):
            # 布尔列表 mask
            return self._filter_mask(key)
        if isinstance(key, Series):
            # 布尔 Series mask
            return self._filter_mask(list(key.values))
        raise TypeError(f"Cannot index Series with {type(key).__name__}")

    def __setitem__(self, key, value):
        """按标签或位置赋值。"""
        if isinstance(key, slice):
            # 切片赋值: 按位置 (与 pandas 整数切片一致)
            start, stop, step = key.indices(len(self))
            indices = list(range(start, stop, step))
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                # 可迭代值: 逐元素赋值
                val_list = list(value)
                if len(val_list) != len(indices):
                    raise ValueError(
                        f"无法将长度 {len(val_list)} 的值赋给长度 {len(indices)} 的切片"
                    )
                for i, v in zip(indices, val_list):
                    self._inner.set_value(i, v)
            else:
                # 标量广播
                for i in indices:
                    self._inner.set_value(i, value)
            return
        if self._index is not None and not _is_range_index(self._index):
            # 自定义 index: 按 label 查找
            if isinstance(key, (str, int, float, bool)):
                try:
                    pos = self._index.index(key)
                    self._inner.set_value(pos, value)
                    return
                except ValueError:
                    raise KeyError(key)
        # RangeIndex 或其他: 走位置
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("index out of range")
            self._inner.set_value(key, value)
            return
        raise TypeError(f"Cannot index Series with {type(key).__name__}")

    def get(self, key, default=None):
        """获取指定索引处的值，若不存在则返回默认值。"""
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def _filter_mask(self, mask: list, preserve_dtype: bool = False) -> _PySeries:
        if len(mask) != len(self):
            raise ValueError(f"mask length {len(mask)} != series length {len(self)}")
        rust_mask = [bool(x) for x in mask]
        dtype = self._dtype_str if preserve_dtype else None
        # 保留原始索引：只保留 mask 为 True 的索引
        new_index = (
            [idx for idx, m in zip(self._index, rust_mask) if m]
            if self._index is not None
            else None
        )
        return Series(
            _PySeries_filter(self._inner, rust_mask),
            name=self.name,
            dtype=dtype,
            index=new_index,
        )

    def __eq__(self, other) -> _PySeries:
        # 标量走 Rust 快速路径；list-like 走 _compare 逐元素比较
        if isinstance(other, (int, float, bool, str, type(None))):
            mask = self._inner.eq_scalar(other)
            return Series(mask, name=self.name, dtype="bool")
        return self._compare(other, "eq")

    def __ne__(self, other) -> _PySeries:
        if isinstance(other, (int, float, bool, str, type(None))):
            eq_mask = self._inner.eq_scalar(other)
            return Series([not x for x in eq_mask], name=self.name, dtype="bool")
        return self._compare(other, "ne")

    def _cmp_scalar(self, other, op_name: str) -> _PySeries:
        """对标量进行比较，自动处理类型转换。"""
        # 转换 other 为与 Series 相同的类型
        dt = self._dtype_str
        if dt == "float64" and not isinstance(other, float):
            other = float(other)
        elif dt == "int64" and not isinstance(other, int):
            other = int(other)

        # 调用 Rust 层对应方法
        if op_name == "lt":
            mask = self._inner.lt_scalar(other)
        elif op_name == "gt":
            mask = self._inner.gt_scalar(other)
        elif op_name == "le":
            mask = self._inner.le_scalar(other)
        elif op_name == "ge":
            mask = self._inner.ge_scalar(other)
        else:
            raise ValueError(f"unsupported op: {op_name}")
        return Series(mask, name=self.name, dtype="bool")

    def __lt__(self, other) -> _PySeries:
        if isinstance(other, (int, float, bool, str, type(None))):
            return self._cmp_scalar(other, "lt")
        return self._compare(other, "lt")

    def __gt__(self, other) -> _PySeries:
        if isinstance(other, (int, float, bool, str, type(None))):
            return self._cmp_scalar(other, "gt")
        return self._compare(other, "gt")

    def __le__(self, other) -> _PySeries:
        if isinstance(other, (int, float, bool, str, type(None))):
            return self._cmp_scalar(other, "le")
        return self._compare(other, "le")

    def __ge__(self, other) -> _PySeries:
        if isinstance(other, (int, float, bool, str, type(None))):
            return self._cmp_scalar(other, "ge")
        return self._compare(other, "ge")

    # ---------- 算术运算符 (v0.3.0) ----------

    def _compare(self, other, op_name: str, fill_value=None) -> _PySeries:
        """逐元素比较运算，支持索引对齐和 fill_value。

        :param op_name: "eq", "ne", "lt", "gt", "le", "ge"
        :param fill_value: 当"至多一个"操作数缺失时用此值替换缺失方后比较；
            两者均缺失时遵循 NaN 比较规则 (ne → True, 其他 → False)。
        """
        _ops = {
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
            "lt": lambda a, b: a < b,
            "gt": lambda a, b: a > b,
            "le": lambda a, b: a <= b,
            "ge": lambda a, b: a >= b,
        }
        fn = _ops[op_name]

        def _is_missing(v) -> bool:
            if v is None:
                return True
            try:
                return v != v  # type: ignore[operator]
            except TypeError:
                return False

        def _compare_pair(a, b) -> bool:
            a_missing = _is_missing(a)
            b_missing = _is_missing(b)
            # 两者都缺失 → NaN 比较规则: ne → True, 其他 → False
            if a_missing and b_missing:
                return op_name == "ne"
            # 至多一个缺失：用 fill_value 替换缺失方
            if a_missing:
                a = fill_value
            if b_missing:
                b = fill_value
            # 替换后仍缺失（fill_value 为 None 时）
            if _is_missing(a) or _is_missing(b):
                return op_name == "ne"
            try:
                return bool(fn(a, b))
            except (TypeError, ValueError):
                return op_name == "ne"

        if isinstance(other, Series):
            # Series vs Series: 按索引对齐
            self_index = (
                list(self._index) if self._index is not None else list(range(len(self)))
            )
            other_index = (
                list(other._index)
                if other._index is not None
                else list(range(len(other)))
            )
            union_index = list(self_index)
            seen = set(self_index)
            for idx in other_index:
                if idx not in seen:
                    seen.add(idx)
                    union_index.append(idx)
            try:
                union_index = sorted(union_index)
            except TypeError:
                pass
            self_map = dict(zip(self_index, self.values))
            other_map = dict(zip(other_index, other.values))
            self_vals = [self_map.get(idx) for idx in union_index]
            other_vals = [other_map.get(idx) for idx in union_index]
        elif isinstance(other, (list, tuple)):
            # Series vs list/tuple: 按位置对齐
            self_vals = list(self.values)
            other_vals = list(other)
            if len(other_vals) != len(self_vals):
                raise ValueError(
                    f"Length mismatch: {len(self_vals)} vs {len(other_vals)}"
                )
            union_index = (
                list(self._index) if self._index is not None else list(range(len(self)))
            )
        elif hasattr(other, "_data") and hasattr(other, "values"):
            # Index 等有 _data/values 属性的对象
            self_vals = list(self.values)
            other_vals = list(other.values)
            if len(other_vals) != len(self_vals):
                raise ValueError(
                    f"Length mismatch: {len(self_vals)} vs {len(other_vals)}"
                )
            union_index = (
                list(self._index) if self._index is not None else list(range(len(self)))
            )
        elif hasattr(other, "tolist"):
            # ndarray 等有 tolist 方法的对象
            self_vals = list(self.values)
            other_vals = list(other.tolist())
            if len(other_vals) != len(self_vals):
                raise ValueError(
                    f"Length mismatch: {len(self_vals)} vs {len(other_vals)}"
                )
            union_index = (
                list(self._index) if self._index is not None else list(range(len(self)))
            )
        else:
            # 标量广播
            self_vals = list(self.values)
            other_vals = [other] * len(self)
            union_index = (
                list(self._index) if self._index is not None else list(range(len(self)))
            )

        result = [_compare_pair(a, b) for a, b in zip(self_vals, other_vals)]
        return Series(result, name=self.name, dtype="bool", index=union_index)

    def _arith(
        self, other, op: str, reverse: bool = False, fill_value=None
    ) -> _PySeries:
        """逐元素算术运算，缺失值用 None。

        :param reverse: 为 True 时交换操作数顺序（用于反向运算符 __rsub__ 等）
        :param fill_value: 当"至多一个"操作数缺失 (None/NaN) 时用此值替换缺失方
            后再做运算；两个操作数均缺失时结果仍为 NaN (与 pandas 行为一致)。
        """

        def _is_missing(v) -> bool:
            """判断值是否为缺失 (None 或 NaN)。"""
            if v is None:
                return True
            try:
                return v != v  # type: ignore[operator]
            except TypeError:
                return False

        # 算术运算 lambda (已处理 reverse: x=self侧, y=other侧)
        _ops = {
            "add": lambda x, y: x + y,
            "sub": lambda x, y: x - y,
            "mul": lambda x, y: x * y,
            "truediv": lambda x, y: (x / y) if y != 0 else None,
            "floordiv": lambda x, y: (x // y) if y != 0 else None,
            "mod": lambda x, y: (x % y) if y != 0 else None,
            "pow": lambda x, y: x**y,
        }
        fn = _ops[op]

        if isinstance(other, Series):
            # Series + Series: 按索引对齐
            self_index = (
                list(self._index) if self._index is not None else list(range(len(self)))
            )
            other_index = (
                list(other._index)
                if other._index is not None
                else list(range(len(other)))
            )
            # 计算索引并集
            union_index = list(self_index)
            seen = set(self_index)
            for idx in other_index:
                if idx not in seen:
                    seen.add(idx)
                    union_index.append(idx)
            # 排序并集以匹配 pandas 行为
            try:
                union_index = sorted(union_index)
            except TypeError:
                pass
            # 构建值映射
            self_map = dict(zip(self_index, self.values))
            other_map = dict(zip(other_index, other.values))
            # 按并集索引对齐
            self_vals = [self_map.get(idx) for idx in union_index]
            other_vals = [other_map.get(idx) for idx in union_index]
        else:
            # 标量广播
            self_vals = list(self.values)
            other_vals = [other] * len(self)
            union_index = (
                list(self._index) if self._index is not None else list(range(len(self)))
            )

        result = []
        for a, b in zip(self_vals, other_vals):
            a_missing = _is_missing(a)
            b_missing = _is_missing(b)
            # 两者都缺失 → None (NaN)
            if a_missing and b_missing:
                result.append(None)
                continue
            # 至多一个缺失：用 fill_value 替换缺失方
            if a_missing:
                a = fill_value
            if b_missing:
                b = fill_value
            # 替换后仍可能为缺失（fill_value 为 None 时）
            if _is_missing(a) or _is_missing(b):
                result.append(None)
                continue
            # 反向运算符交换左右操作数
            x, y = (b, a) if reverse else (a, b)
            try:
                result.append(fn(x, y))
            except (TypeError, ValueError, ZeroDivisionError):
                result.append(None)
        # 推断结果 dtype
        nums = [v for v in result if isinstance(v, (int, float))]
        has_none = any(v is None for v in result)
        if not nums:
            return Series(result, name=self.name, dtype="object", index=union_index)
        # 如果有 NaN/None，必须用 float64（因为 NaN 是浮点数）
        if has_none:
            return Series(result, name=self.name, dtype="float64", index=union_index)

        # 检查原始操作数中是否有 float 类型
        self_has_float = any(isinstance(v, float) for v in self_vals if v is not None)
        other_has_float = any(isinstance(v, float) for v in other_vals if v is not None)

        if any(isinstance(v, float) for v in nums):
            if self_has_float or other_has_float:
                # 原始操作数中有 float，保持 float 类型（避免 5.0 + 5 → 10）
                return Series(
                    result, name=self.name, dtype="float64", index=union_index
                )
            # 检查是否所有浮点数都是整数值（纯 int 运算产生的 float 结果）
            if all(isinstance(v, float) and v == int(v) for v in nums):
                # 转换为整数
                int_result = [
                    int(v) if isinstance(v, float) and v == int(v) else v
                    for v in result
                ]
                return Series(
                    int_result, name=self.name, dtype="int64", index=union_index
                )
            return Series(result, name=self.name, dtype="float64", index=union_index)
        return Series(result, name=self.name, dtype="int64", index=union_index)

    def __add__(self, other) -> _PySeries:
        return self._arith(other, "add")

    def __radd__(self, other) -> _PySeries:
        return self._arith(other, "add")

    def __sub__(self, other) -> _PySeries:
        return self._arith(other, "sub")

    def __rsub__(self, other) -> _PySeries:
        return self._arith(other, "sub", reverse=True)

    def __mul__(self, other) -> _PySeries:
        return self._arith(other, "mul")

    def __rmul__(self, other) -> _PySeries:
        return self._arith(other, "mul")

    def __truediv__(self, other) -> _PySeries:
        return self._arith(other, "truediv")

    def __rtruediv__(self, other) -> _PySeries:
        return self._arith(other, "truediv", reverse=True)

    def __floordiv__(self, other) -> _PySeries:
        return self._arith(other, "floordiv")

    def __rfloordiv__(self, other) -> _PySeries:
        return self._arith(other, "floordiv", reverse=True)

    def __mod__(self, other) -> _PySeries:
        return self._arith(other, "mod")

    def __rmod__(self, other) -> _PySeries:
        return self._arith(other, "mod", reverse=True)

    def __pow__(self, other) -> _PySeries:
        return self._arith(other, "pow")

    def __rpow__(self, other) -> _PySeries:
        """反向幂运算: other ** self"""
        # 使用列表推导式替代显式 for 循环
        result = [
            None if (v is None or other is None) else other**v for v in self.values
        ]
        return Series(result, name=self.name, index=self._index, dtype=self._dtype_str)

    def __divmod__(self, other):
        """divmod 运算: self divmod other

        支持 other 为标量、list、Series 或 rsnumpy.ndarray，按位置对齐做元素级 divmod。
        """
        self_vals = list(self.values)
        # 解析 other 为值列表或标量
        if isinstance(other, Series):
            other_vals = list(other.values)
            scalar_other = None
        elif isinstance(other, (list, tuple)):
            other_vals = list(other)
            scalar_other = None
        elif hasattr(other, "tolist"):
            other_vals = list(other.tolist())
            scalar_other = None
        else:
            other_vals = None
            scalar_other = other

        if other_vals is not None:
            if len(other_vals) != len(self_vals):
                raise ValueError(
                    f"operands could not be broadcast together with shapes "
                    f"({len(self_vals)},) ({len(other_vals)},)"
                )
            pairs = [
                (None, None) if (v is None or o is None) else divmod(v, o)
                for v, o in zip(self_vals, other_vals)
            ]
        else:
            pairs = [
                (
                    (None, None)
                    if (v is None or scalar_other is None)
                    else divmod(v, scalar_other)
                )
                for v in self_vals
            ]
        quot = [p[0] for p in pairs]
        rem = [p[1] for p in pairs]
        # 推断 dtype：含 None 则 float64，否则 int64
        has_none = any(q is None for q in quot)
        dtype = "float64" if has_none else "int64"
        return (
            Series(quot, name=self.name, index=self._index, dtype=dtype),
            Series(rem, name=self.name, index=self._index, dtype=dtype),
        )

    def __rdivmod__(self, other):
        """反向 divmod 运算: other divmod self

        支持 other 为标量、list、Series 或 rsnumpy.ndarray，按位置对齐做元素级 divmod。
        """
        self_vals = list(self.values)
        if isinstance(other, Series):
            other_vals = list(other.values)
            scalar_other = None
        elif isinstance(other, (list, tuple)):
            other_vals = list(other)
            scalar_other = None
        elif hasattr(other, "tolist"):
            other_vals = list(other.tolist())
            scalar_other = None
        else:
            other_vals = None
            scalar_other = other

        if other_vals is not None:
            if len(other_vals) != len(self_vals):
                raise ValueError(
                    f"operands could not be broadcast together with shapes "
                    f"({len(other_vals)},) ({len(self_vals)},)"
                )
            pairs = [
                (None, None) if (v is None or o is None) else divmod(o, v)
                for v, o in zip(self_vals, other_vals)
            ]
        else:
            pairs = [
                (
                    (None, None)
                    if (v is None or scalar_other is None)
                    else divmod(scalar_other, v)
                )
                for v in self_vals
            ]
        quot = [p[0] for p in pairs]
        rem = [p[1] for p in pairs]
        has_none = any(q is None for q in quot)
        dtype = "float64" if has_none else "int64"
        return (
            Series(quot, name=self.name, index=self._index, dtype=dtype),
            Series(rem, name=self.name, index=self._index, dtype=dtype),
        )

    def _bitwise_series(self, result, force_int: bool = False) -> "_PySeries":
        """根据位运算结果构造 Series，自动推断 dtype。

        与 pandas 行为对齐：
        - bool & bool -> bool
        - int & int -> int64
        - bool & int -> int64（Python 中 True & 1 = 1）
        - 移位运算（<<, >>）结果恒为 int64

        :param result: 位运算结果值列表
        :param force_int: 移位运算时为 True，强制 dtype='int64'
        """
        if force_int:
            dtype = "int64"
        else:
            non_null = [v for v in result if v is not None]
            if non_null and all(isinstance(v, bool) for v in non_null):
                dtype = "bool"
            elif non_null and all(
                isinstance(v, int) and not isinstance(v, bool) for v in non_null
            ):
                dtype = "int64"
            else:
                dtype = self._dtype_str
        return Series(result, name=self.name, index=self._index, dtype=dtype)

    def __invert__(self):
        """按位取反: ~self

        - bool 类型: 逻辑取反 (True -> False)，dtype 保持 bool
        - int 类型: 按位取反 (~v)，dtype 保持 int64
        """
        if self._dtype_str == "bool":
            result = [None if v is None else not v for v in self.values]
            return Series(result, name=self.name, index=self._index, dtype="bool")
        result = [None if v is None else ~int(v) for v in self.values]
        return self._bitwise_series(result)

    def __and__(self, other):
        """按位与: self & other

        保留操作数原始类型（bool/int），与 pandas 行为一致。
        """
        if isinstance(other, Series):
            other_vals = other.values
            result = [
                (
                    None
                    if (v is None or i >= len(other_vals) or other_vals[i] is None)
                    else v & other_vals[i]
                )
                for i, v in enumerate(self.values)
            ]
        else:
            result = [
                None if (v is None or other is None) else v & other for v in self.values
            ]
        return self._bitwise_series(result)

    def __rand__(self, other):
        """反向按位与: other & self"""
        return self.__and__(other)

    def __or__(self, other):
        """按位或: self | other

        保留操作数原始类型（bool/int），与 pandas 行为一致。
        """
        if isinstance(other, Series):
            other_vals = other.values
            result = [
                (
                    None
                    if (v is None or i >= len(other_vals) or other_vals[i] is None)
                    else v | other_vals[i]
                )
                for i, v in enumerate(self.values)
            ]
        else:
            result = [
                None if (v is None or other is None) else v | other for v in self.values
            ]
        return self._bitwise_series(result)

    def __ror__(self, other):
        """反向按位或: other | self"""
        return self.__or__(other)

    def __xor__(self, other):
        """按位异或: self ^ other

        保留操作数原始类型（bool/int），与 pandas 行为一致。
        """
        if isinstance(other, Series):
            other_vals = other.values
            result = [
                (
                    None
                    if (v is None or i >= len(other_vals) or other_vals[i] is None)
                    else v ^ other_vals[i]
                )
                for i, v in enumerate(self.values)
            ]
        else:
            result = [
                None if (v is None or other is None) else v ^ other for v in self.values
            ]
        return self._bitwise_series(result)

    def __rxor__(self, other):
        """反向按位异或: other ^ self"""
        return self.__xor__(other)

    def __lshift__(self, other):
        """左移: self << other

        移位运算结果恒为 int64（与 pandas 一致：bool << 1 -> int64）。
        """
        if isinstance(other, Series):
            other_vals = other.values
            result = [
                (
                    None
                    if (v is None or i >= len(other_vals) or other_vals[i] is None)
                    else int(v) << int(other_vals[i])
                )
                for i, v in enumerate(self.values)
            ]
        else:
            result = [
                None if (v is None or other is None) else int(v) << int(other)
                for v in self.values
            ]
        return self._bitwise_series(result, force_int=True)

    def __rlshift__(self, other):
        """反向左移: other << self"""
        if isinstance(other, Series):
            other_vals = other.values
            result = [
                (
                    None
                    if (v is None or i >= len(other_vals) or other_vals[i] is None)
                    else int(other_vals[i]) << int(v)
                )
                for i, v in enumerate(self.values)
            ]
        else:
            result = [
                None if (v is None or other is None) else int(other) << int(v)
                for v in self.values
            ]
        return self._bitwise_series(result, force_int=True)

    def __rshift__(self, other):
        """右移: self >> other

        移位运算结果恒为 int64（与 pandas 一致）。
        """
        if isinstance(other, Series):
            other_vals = other.values
            result = [
                (
                    None
                    if (v is None or i >= len(other_vals) or other_vals[i] is None)
                    else int(v) >> int(other_vals[i])
                )
                for i, v in enumerate(self.values)
            ]
        else:
            result = [
                None if (v is None or other is None) else int(v) >> int(other)
                for v in self.values
            ]
        return self._bitwise_series(result, force_int=True)

    def __rrshift__(self, other):
        """反向右移: other >> self"""
        if isinstance(other, Series):
            other_vals = other.values
            result = [
                (
                    None
                    if (v is None or i >= len(other_vals) or other_vals[i] is None)
                    else int(other_vals[i]) >> int(v)
                )
                for i, v in enumerate(self.values)
            ]
        else:
            result = [
                None if (v is None or other is None) else int(other) >> int(v)
                for v in self.values
            ]
        return self._bitwise_series(result, force_int=True)

    def __neg__(self) -> _PySeries:
        if self._dtype_str == "bool":
            # bool 类型的取负相当于逻辑 NOT（True -> False, False -> True）
            return self.__invert__()
        return self._arith(-1, "mul")

    def __pos__(self) -> _PySeries:
        return self

    def __abs__(self) -> _PySeries:
        result = [None if v is None else abs(v) for v in self.values]
        return Series(result, name=self.name, dtype=self._dtype_str)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """支持 rsnumpy 通用函数 (ufunc) 作用于 Series。

        当 rnp.exp(series) / rnp.remainder(s1, s2) 等调用时，
        rsnumpy 会调用此方法，返回 Series 而非 list。
        """
        if method != "__call__":
            return NotImplemented

        # 将 Series 转换为值列表，其他保持不变
        processed_inputs = []
        series_inputs = []  # 记录哪些输入是 Series
        for inp in inputs:
            if isinstance(inp, Series):
                processed_inputs.append(list(inp.values))
                series_inputs.append(inp)
            else:
                processed_inputs.append(inp)
                series_inputs.append(None)

        # 处理 out 参数
        out = kwargs.get("out", None)

        if out is not None:
            # out 参数暂不支持，忽略
            pass

        # 应用 ufunc
        result_values = []
        n = len(processed_inputs[0])
        for i in range(n):
            args = []
            for inp in processed_inputs:
                if isinstance(inp, list):
                    args.append(inp[i])
                else:
                    args.append(inp)  # 标量
            # 跳过 None 值
            if any(a is None for a in args):
                result_values.append(None)
            else:
                try:
                    result_values.append(ufunc(*args))
                except (TypeError, ValueError):
                    result_values.append(None)

        # 推断结果 dtype
        dtype = self._dtype_str
        if result_values and any(v is not None for v in result_values):
            non_null = [v for v in result_values if v is not None]
            if non_null and all(isinstance(v, bool) for v in non_null):
                dtype = "bool"
            elif non_null and all(isinstance(v, int) for v in non_null):
                dtype = "int64"
            elif non_null and all(isinstance(v, (int, float)) for v in non_null):
                dtype = "float64"

        # 确定结果的 index 和 name
        # 如果有多个 Series 输入且 index 不同，对齐索引
        series_with_index = [s for s in series_inputs if s is not None]
        if len(series_with_index) > 1:
            # 多 Series 输入：按第一个 Series 的 index 对齐
            first_series = series_with_index[0]
            result_index = (
                list(first_series._index)
                if first_series._index is not None
                else list(range(n))
            )
        elif series_with_index:
            s = series_with_index[0]
            result_index = list(s._index) if s._index is not None else list(range(n))
        else:
            result_index = list(range(n))

        result_name = self.name

        return Series(result_values, name=result_name, dtype=dtype, index=result_index)

    def __array__(self, dtype=None):
        """支持 rnp.array(series) 转换。"""
        return rnp.array(self.values, dtype=dtype)

    # ---------- 命名算术方法 ----------

    def radd(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """反向加法: other + self"""
        return self._arith(other, "add", reverse=True, fill_value=fill_value)

    def rsub(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """反向减法: other - self"""
        return self._arith(other, "sub", reverse=True, fill_value=fill_value)

    def rmul(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """反向乘法: other * self"""
        return self._arith(other, "mul", reverse=True, fill_value=fill_value)

    def rdiv(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """反向除法: other / self"""
        return self._arith(other, "truediv", reverse=True, fill_value=fill_value)

    def rtruediv(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """反向真除法: other / self"""
        return self._arith(other, "truediv", reverse=True, fill_value=fill_value)

    def rfloordiv(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """反向整除: other // self"""
        return self._arith(other, "floordiv", reverse=True, fill_value=fill_value)

    def rmod(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """反向取模: other % self"""
        return self._arith(other, "mod", reverse=True, fill_value=fill_value)

    def rpow(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """反向幂运算: other ** self"""
        return self._arith(other, "pow", reverse=True, fill_value=fill_value)

    def rdivmod(self, other):
        """反向 divmod: divmod(other, self)"""
        return self.__rdivmod__(other)

    # ---------- 比较命名方法 ----------

    def eq(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """等于: self == other"""
        return self._compare(other, "eq", fill_value=fill_value)

    def ne(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """不等于: self != other"""
        return self._compare(other, "ne", fill_value=fill_value)

    def lt(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """小于: self < other"""
        return self._compare(other, "lt", fill_value=fill_value)

    def gt(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """大于: self > other"""
        return self._compare(other, "gt", fill_value=fill_value)

    def le(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """小于等于: self <= other"""
        return self._compare(other, "le", fill_value=fill_value)

    def ge(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """大于等于: self >= other"""
        return self._compare(other, "ge", fill_value=fill_value)

    # ---------- 合并方法 ----------

    def combine(self, other, func, fill_value=None) -> _PySeries:
        """按元素合并两个 Series。

        :param other: 另一个 Series
        :param func: 接收两个标量的函数
        :param fill_value: 缺失值填充
        """
        # 使用列表推导式替代显式 for 循环
        other_vals = other.values
        self_vals = self.values
        other_len = len(other_vals)

        def _combine_one(i, v):
            ov = other_vals[i] if i < other_len else None
            a = v if v is not None else fill_value
            b = ov if ov is not None else fill_value
            return func(a, b) if (a is not None and b is not None) else None

        result = [_combine_one(i, v) for i, v in enumerate(self_vals)]
        return Series(result, name=self.name, index=self._index, dtype=self._dtype_str)

    def combine_first(self, other) -> _PySeries:
        """用 other 的非空值填充 self 的空值。

        按 index 标签对齐：self 优先，self 缺失时取 other，
        两者均缺失则结果为 NaN。结果索引为两者的并集。

        :param other: 另一个 Series
        """

        def _is_missing(v) -> bool:
            if v is None:
                return True
            try:
                return v != v  # type: ignore[operator]
            except TypeError:
                return False

        self_index = (
            list(self._index) if self._index is not None else list(range(len(self)))
        )
        other_index = (
            list(other._index) if other._index is not None else list(range(len(other)))
        )
        # 索引并集 (保持顺序)
        union_index = list(self_index)
        seen = set(self_index)
        for idx in other_index:
            if idx not in seen:
                seen.add(idx)
                union_index.append(idx)
        # 构建值映射
        self_map = dict(zip(self_index, self.values))
        other_map = dict(zip(other_index, other.values))
        # 按并集索引逐行合并
        result = []
        for idx in union_index:
            sv = self_map.get(idx)
            if not _is_missing(sv):
                result.append(sv)
            else:
                ov = other_map.get(idx)
                result.append(ov)  # other 也可能缺失 → None/NaN
        return Series(result, name=self.name, index=union_index)

    @property
    def str(self):
        """字符串访问器。"""
        return StringAccessor(self)

    @property
    def cat(self):
        """Categorical 访问器。"""
        return CatAccessor(self)

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

    # ---------- 转换方法 (v1.0.0) ----------

    def to_list(self) -> list:
        """转换为 Python list。"""
        return list(self.values)

    def tolist(self) -> list:
        """to_list 的别名。"""
        return self.to_list()

    def to_frame(self, name: Optional[str] = None) -> "_PyDataFrame":
        """转换为 DataFrame。

        :param name: 列名 (默认使用 Series.name)
        """
        from .dataframe import DataFrame

        col_name = name if name is not None else self.name
        if col_name is None:
            col_name = 0
        return DataFrame({col_name: list(self.values)}, index=self._index)

    def to_numpy(self, dtype=None, copy: bool = True, na_value: Any = None):
        """转换为 rsnumpy array（对齐 pandas Series.to_numpy）。

        当 Series 内部有 datetime 缓存（源自 DatetimeSeries 或 datetime 列表），
        支持 `dtype=object`（返回 datetime 对象数组）和
        `dtype='datetime64[ns]'`（返回 int64 纳秒纪元数组）。

        :param dtype: 目标 dtype ('object' / 'datetime64[ns]' / None 等)
        :param copy: 是否拷贝 (保留参数对齐 pandas，本实现始终返回新数组)
        :param na_value: 替换 None 的值 (仅在 dtype=object / datetime64[ns] 场景生效)
        """
        import rsnumpy as rnp

        has_dt_cache = self._dt_values is not None and len(self._dt_values) > 0
        if dtype is None:
            dtype_str = ""
            dtype_is_object = False
            dtype_is_datetime64 = False
        elif dtype is object:
            dtype_str = "object"
            dtype_is_object = True
            dtype_is_datetime64 = False
        elif isinstance(dtype, str):
            dtype_str = dtype.lower().replace(" ", "")
            dtype_is_object = dtype_str in ("object", "o")
            dtype_is_datetime64 = dtype_str.startswith("datetime64")
        else:
            dtype_str = str(dtype).lower()
            dtype_is_object = "'object'" in dtype_str or dtype_str == "<class 'object'>"
            dtype_is_datetime64 = "datetime64" in dtype_str

        if has_dt_cache and (dtype_is_object or dtype_is_datetime64 or dtype_str == ""):
            vals = list(self._dt_values)
            if na_value is not None:
                vals = [na_value if v is None else v for v in vals]
            if dtype_is_object:
                # rsnumpy 不支持 datetime 对象，退回 ISO 字符串 (object dtype)
                from ._datetime import _to_iso

                iso_strs = [
                    (
                        _to_iso(v)
                        if isinstance(v, (datetime, date)) and not isinstance(v, bool)
                        else v
                    )
                    for v in vals
                ]
                return rnp.array(iso_strs)
            # 默认或 datetime64[ns]: 转纳秒纪元
            sample_tz = (
                vals[0].tzinfo
                if vals and isinstance(vals[0], datetime) and vals[0].tzinfo is not None
                else None
            )
            epoch = datetime(1970, 1, 1, tzinfo=sample_tz)
            nano_vals: list = []
            for v in vals:
                if v is None:
                    nano_vals.append(float("nan"))
                elif isinstance(v, datetime):
                    delta = v - epoch
                    ns = (
                        delta.days * 86_400_000_000_000
                        + delta.seconds * 1_000_000_000
                        + delta.microseconds * 1000
                    )
                    nano_vals.append(ns)
                elif isinstance(v, date):
                    dt = datetime(v.year, v.month, v.day, tzinfo=sample_tz)
                    delta = dt - epoch
                    nano_vals.append(
                        delta.days * 86_400_000_000_000 + delta.seconds * 1_000_000_000
                    )
                else:
                    nano_vals.append(float("nan"))
            return rnp.array(nano_vals)

        # 普通路径
        # rsnumpy 不支持 None 与数值混合，将 None 转为 nan (float 数组)
        vals = list(self.values)
        if any(v is None for v in vals):
            non_none = [v for v in vals if v is not None]
            # 仅当其余值为数值时，将 None 转为 nan
            if non_none and all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in non_none
            ):
                vals = [float("nan") if v is None else float(v) for v in vals]
        if dtype:
            return rnp.array(vals, dtype=dtype)
        return rnp.array(vals)

    @classmethod
    def from_numpy(cls, arr, name=None, index=None) -> "Series":
        """从 rsnumpy array 构造 Series。

        :param arr: rsnumpy.ndarray 输入数组
        :param name: Series 名称
        :param index: 索引
        """
        if not isinstance(arr, rnp.ndarray):
            raise TypeError(f"expected rsnumpy.ndarray, got {type(arr).__name__}")
        vals = arr.tolist()
        return cls(vals, name=name, index=index)

    def to_dict(self) -> dict:
        """转换为 dict (index -> value)。"""
        return {
            self._index[i] if self._index else i: v for i, v in enumerate(self.values)
        }

    def unstack(self, level=-1) -> "_PyDataFrame":
        """将 Series 的 MultiIndex 展开为 DataFrame。

        :param level: 要展开的层级 (默认 -1，即最后一级)
        :return: DataFrame

        Examples:
            >>> s = Series([1, 2, 3, 4], index=[('a', 'x'), ('a', 'y'), ('b', 'x'), ('b', 'y')])
            >>> s.unstack()
               x  y
            a  1  2
            b  3  4
        """
        from .dataframe import DataFrame

        n = len(self)
        if n == 0:
            return DataFrame()

        # 解析索引
        if self._index is None:
            raise ValueError("Series has no index")

        # 将每个索引解析为 (row_key, col_key) 元组
        # 如果索引是 tuple，取前 level 层为 row，第 level 层为 col
        pairs = []
        for i in range(n):
            idx = self._index[i]
            if isinstance(idx, tuple):
                if level == -1 or level == len(idx) - 1:
                    row_key = idx[:-1] if len(idx) > 1 else idx[0]
                    col_key = idx[-1]
                else:
                    row_key = (
                        idx[:level] + idx[level + 1 :]  # noqa
                        if len(idx) > 1 and level < len(idx)
                        else idx[0]
                    )
                    col_key = idx[level]
            else:
                row_key = 0
                col_key = idx
            pairs.append((row_key, col_key, self.values[i]))

        # 收集所有行键和列键
        row_keys = list(dict.fromkeys(p[0] for p in pairs))
        col_keys = list(dict.fromkeys(p[1] for p in pairs))

        # 构建结果
        # 构建数据
        data: Dict[str, list] = {}
        for ck in col_keys:
            col_name = str(ck)
            col_vals = []
            for rk in row_keys:
                val = None
                for p in pairs:
                    if p[0] == rk and p[1] == ck:
                        val = p[2]
                        break
                col_vals.append(val)
            data[col_name] = col_vals

        result = DataFrame(data)
        # 设置行索引
        result._index = row_keys
        return result

    # ---------- 展开方法 (v1.0.0) ----------
    def explode(self) -> _PySeries:
        """展开列表元素为单独行。"""
        from itertools import chain

        values = self.values
        # 使用列表推导式生成每项的展开列表，再用 chain 拼接
        out = list(
            chain.from_iterable(
                (
                    [None]
                    if v is None
                    else list(v) if isinstance(v, (list, tuple)) else [v]
                )
                for v in values
            )
        )
        return Series(out, name=self.name)

    def repeat(self, repeats) -> _PySeries:
        """重复元素。
        :param repeats: 重复次数 (int 或 list[int])
        """
        values = self.values
        # 使用列表推导式 + itertools.chain 替代显式 for 循环
        from itertools import chain

        if isinstance(repeats, int):
            out = list(chain.from_iterable([v] * repeats for v in values))
        else:
            if len(repeats) != len(values):
                raise ValueError("repeats length must match series length")
            out = list(
                chain.from_iterable([v] * rep for v, rep in zip(values, repeats))
            )
        return Series(out, name=self.name)

    def __bool__(self) -> bool:
        if len(self) == 1:
            v = self.values[0]
            if v is None:
                raise ValueError("truth value of a None element is ambiguous")
            return bool(v)
        raise ValueError(
            f"truth value of a Series with {len(self)} elements is ambiguous"
        )

    # ---------- 子集 ----------

    def head(self, n: int = 5) -> _PySeries:
        n = min(n, len(self))
        sliced_index = self._index[:n] if self._index is not None else None
        return Series(
            self._inner.head(n),
            name=self.name,
            dtype=self._dtype_str,
            index=sliced_index,
        )

    def tail(self, n: int = 5) -> _PySeries:
        n = min(n, len(self))
        sliced_index = self._index[-n:] if self._index is not None else None
        return Series(
            self._inner.tail(n),
            name=self.name,
            dtype=self._dtype_str,
            index=sliced_index,
        )

    @property
    def iloc(self):
        """按位置索引访问器。"""
        return _ILocIndexer(self)

    def sort_values(
        self,
        axis: int = 0,
        ascending: bool = True,
        inplace: bool = False,
        kind: str = "quicksort",
        na_position: str = "last",
        ignore_index: bool = False,
        key=None,
    ) -> _PySeries:
        """按值排序。

        :param axis: 轴方向（Series 仅支持 0）
        :param ascending: 是否升序 (默认 True)
        :param inplace: 是否原地修改 (默认 False)
        :param kind: 排序算法 ('quicksort'/'mergesort'/'heapsort'/'stable')
        :param na_position: None 值的位置 ('first' 或 'last', 默认 'last')
        :param ignore_index: 是否忽略索引并从 0 开始重新生成
        :param key: 应用于待排序值的排序键函数（如 ``lambda x: x.str.lower()``）
        """
        if axis != 0:
            raise ValueError(f"axis must be 0 for Series, got {axis}")

        n = len(self.values)

        # 使用 Python 实现（保留原始索引标签，对齐 pandas 行为；
        # Rust 层 sort_values 会重排索引，无法保留原始索引标签，故不使用）
        values = list(self.values)

        # 应用 key 函数（若有）
        sort_vals = values
        if key is not None:
            tmp = Series(values, name=self.name, index=self._index)
            transformed = key(tmp)
            if hasattr(transformed, "values"):
                sort_vals = list(transformed.values)
            elif hasattr(transformed, "tolist"):
                sort_vals = list(transformed.tolist())
            else:
                sort_vals = list(transformed)
            if len(sort_vals) != n:
                raise ValueError("key function must return an index of the same length")

        # 分离 None 和非 None
        non_none = [i for i in range(n) if sort_vals[i] is not None]
        none_items = [i for i in range(n) if sort_vals[i] is None]

        try:
            non_none.sort(key=lambda i: sort_vals[i], reverse=not ascending)
        except TypeError:
            raise TypeError("cannot sort mixed types")

        # 根据 na_position 决定 None 的位置
        if na_position == "first":
            order = none_items + non_none
        else:  # 'last'
            order = non_none + none_items

        new_values = [values[i] for i in order]
        if self._index is not None:
            new_index = [self._index[i] for i in order]
        else:
            new_index = None
        if ignore_index:
            new_index = list(range(n))

        if inplace:
            self._inner = _PySeries(new_values, self.name)
            self._index = new_index
            return self
        return Series(new_values, name=self.name, index=new_index)

    def apply(self, func, convert_dtype: bool = True, args=(), **kwargs) -> _PySeries:
        """对每个非 None 元素应用 func。

        :param func: callable (scalar) -> scalar
        :param convert_dtype: 是否尝试转换 dtype (默认 True)
        :param args: 传递给 func 的额外位置参数
        :param kwargs: 传递给 func 的关键字参数
        """

        # 使用辅助函数 + 列表推导式替代显式 for 循环
        def _apply_one(v):
            if v is None:
                return None
            try:
                return func(v, *args, **kwargs)
            except Exception:
                return None

        out = [_apply_one(v) for v in self.values]
        return Series(out, name=self.name, index=self._index)

    def map(self, arg, na_action: Optional[str] = None) -> _PySeries:
        """映射: 可以传 dict、callable 或 Series。

        :param arg: 映射字典、可调用函数或 Series (用索引作映射键)
        :param na_action: None 值处理方式 ('ignore' 跳过 NA, None 对 NA 也调用函数)
        """
        if isinstance(arg, dict):
            if na_action == "ignore":
                out = [arg.get(v, v) if not _is_missing(v) else v for v in self.values]
            else:
                out = [arg.get(v, None) for v in self.values]
        elif isinstance(arg, Series):
            # Series 映射: 用 arg 的 index->value 构建字典
            mapping = dict(zip(arg.index, arg.values))
            if na_action == "ignore":
                out = [
                    mapping.get(v, v) if not _is_missing(v) else v for v in self.values
                ]
            else:
                out = [mapping.get(v, float("nan")) for v in self.values]
        else:
            # callable
            if na_action == "ignore":
                out = [v if _is_missing(v) else arg(v) for v in self.values]
            else:
                # na_action=None: 对所有值调用 func (包括 None/NaN)
                # None 转为 float('nan') 以匹配 pandas 行为 (str(nan)="nan")
                out = [arg(float("nan") if v is None else v) for v in self.values]
        return Series(out, name=self.name, index=self._index)

    def replace(
        self,
        to_replace=None,
        value=None,
        inplace: bool = False,
        limit=None,
        regex: bool = False,
        method: str = "pad",
    ) -> _PySeries:
        """替换值。

        :param to_replace: 要替换的值 (标量、列表、字典或正则)
        :param value: 替换后的值 (标量、列表，字典时忽略)
        :param inplace: 是否原地修改 (默认 False)
        :param limit: 最大替换次数 (未实现)
        :param regex: 是否使用正则表达式 (默认 False)
        :param method: 填充方法 ('pad'/'ffill'/'bfill')
        """
        if to_replace is None:
            raise ValueError("'to_replace' must be specified")

        out = []
        if regex:
            # 正则替换
            import re
            from functools import reduce

            if isinstance(to_replace, str) and isinstance(value, str):
                pattern = re.compile(to_replace)
                out = [
                    None if v is None else pattern.sub(value, str(v))
                    for v in self.values
                ]
            elif isinstance(to_replace, dict):
                # 预编译所有 pattern，避免循环内重复编译
                compiled = [(re.compile(k), repl) for k, repl in to_replace.items()]
                out = [
                    (
                        None
                        if v is None
                        else reduce(lambda s, pr: pr[0].sub(pr[1], s), compiled, str(v))
                    )
                    for v in self.values
                ]
            else:
                raise TypeError("regex replace requires str or dict patterns")
        else:
            # 形式 1: scalar
            if not isinstance(to_replace, (list, tuple, dict)):
                out = [value if v == to_replace else v for v in self.values]
            # 形式 3: list[old] + list[new]
            elif isinstance(to_replace, (list, tuple)) and isinstance(
                value, (list, tuple)
            ):
                if len(to_replace) != len(value):
                    raise ValueError("to_replace and value must have the same length")
                mapping = dict(zip(to_replace, value))
                out = [
                    mapping.get(v, v) if v is not None else None for v in self.values
                ]
            elif isinstance(to_replace, dict):
                mapping = to_replace
                out = [
                    mapping.get(v, v) if v is not None else None for v in self.values
                ]
            else:
                raise TypeError("invalid replace arguments")

        if inplace:
            self._inner = _PySeries(out, self.name, dtype=self._dtype_str)
            return self
        return Series(out, name=self.name, dtype=self._dtype_str, index=self._index)

    def duplicated(self, keep: str = "first") -> _PySeries:
        """返回 bool Series 标记重复行。

        :param keep: 'first' / 'last' / False
        """
        if keep == "first":
            # 首次出现返回 False，后续返回 True
            seen: set = set()
            out = [v in seen or (seen.add(v), False)[1] for v in self.values]
        elif keep == "last":
            # 最后一次出现返回 False，之前的返回 True
            seen = set()
            rev_out = [
                v in seen or (seen.add(v), False)[1] for v in reversed(self.values)
            ]
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
        src_index = self._index if self._index else range(len(self))
        if keep == "last":
            # 反向遍历，保留最后一次出现
            seen: set = set()
            rev_pairs = [
                (v, i)
                for v, i in zip(reversed(self.values), reversed(src_index))
                if v not in seen and not seen.add(v)
            ]
            pairs = list(reversed(rev_pairs))
        else:
            # 默认 keep="first"，保留首次出现
            seen = set()
            pairs = [
                (v, i)
                for v, i in zip(self.values, src_index)
                if v not in seen and not seen.add(v)
            ]
        out = [v for v, _ in pairs]
        out_idx = [i for _, i in pairs]
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

    def astype(self, dtype, copy: bool = True, errors: str = "raise") -> _PySeries:
        """类型转换。

        :param dtype: 目标类型 (str/type 对象/numpy dtype 对象)
        :param copy: 是否复制数据 (默认 True)
        :param errors: 错误处理 ('raise' 或 'ignore')
        """
        # 统一规范化 dtype 为字符串（支持 np.uint8/np.float32 等 type 对象）
        dtype_str = _dtype_to_str(dtype)
        target = dtype_str.lower()

        if target == self._dtype_str and not copy:
            return self

        if target == self._dtype_str:
            return Series(
                self.values, name=self.name, dtype=dtype_str, index=self._index
            )

        vals = []
        try:
            if target in (
                "int8",
                "int16",
                "int32",
                "int64",
                "int",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
            ):
                # 整数族：Rust 层统一存储为 int64，Python 层 _dtype_str 追踪精确子类型
                vals = [None if v is None else int(v) for v in self.values]
            elif target in ("float32", "float64", "float"):
                # 浮点族：Rust 层统一存储为 float64，Python 层 _dtype_str 追踪精确子类型
                vals = [None if v is None else float(v) for v in self.values]
            elif target == "bool":
                vals = [None if v is None else bool(v) for v in self.values]
            elif target in ("object", "str", "string"):
                vals = [None if v is None else str(v) for v in self.values]
            elif target == "category":
                # 转换为 Categorical（Rust 层不支持 category dtype，强制设置 _dtype_str）
                s = Series(self.values, name=self.name, dtype="object")
                s._dtype_str = "category"
                return s
            else:
                raise TypeError(f"unsupported dtype: {dtype}")
        except (ValueError, TypeError) as e:
            if errors == "raise":
                raise e
            # errors == 'ignore': 返回原始 Series
            return self.copy()

        # 传递规范化后的 dtype 字符串以保留子类型信息（如 int8/uint8/float32）
        return Series(vals, name=self.name, dtype=dtype_str, index=self._index)

    def abs(self) -> _PySeries:
        """返回绝对值 Series。"""
        out = [None if v is None else abs(v) for v in self.values]
        return Series(out, name=self.name, dtype=self._dtype_str, index=self._index)

    def sqrt(self) -> _PySeries:
        """逐元素求平方根。"""
        import math

        def _sqrt(v):
            if v is None or _is_missing(v):
                return None
            if v < 0:
                return float("nan")
            return math.sqrt(v)

        out = [_sqrt(v) for v in self.values]
        return Series(out, name=self.name, dtype="float64", index=self._index)

    def copy(self, deep: bool = True) -> _PySeries:
        """复制 Series。

        :param deep: True=深拷贝, False=浅拷贝
        """
        if deep:
            return Series(
                list(self.values),
                name=self.name,
                dtype=self._dtype_str,
                index=list(self._index) if self._index is not None else None,
            )
        return Series(self._inner, name=self.name, dtype=self._dtype_str)

    def drop(
        self,
        labels=None,
        axis: int = 0,
        index=None,
        columns=None,
        level=None,
        inplace: bool = False,
        errors: str = "raise",
    ) -> _PySeries:
        """删除指定索引的元素。

        :param labels: 要删除的索引标签 (标量或列表)
        :param axis: 0=索引 (仅支持 0)
        :param index: 要删除的索引标签（替代 labels）
        :param columns: 不支持（Series 无列）
        :param level: 多级索引层级（暂不支持）
        :param inplace: 是否原地修改
        :param errors: 'raise' (找不到时抛错) 或 'ignore' (静默忽略)
        """
        if columns is not None:
            raise ValueError("Series.drop does not support columns")
        if index is not None:
            labels = index
        if labels is None:
            return self if inplace else self.copy()
        if not isinstance(labels, (list, tuple)):
            labels = [labels]

        # 使用列表推导式替代显式 for 循环（按 index 过滤保留项）
        labels_set = set(labels)
        src_index = self._index if self._index is not None else list(range(len(self)))
        kept = [
            (v, idx) for v, idx in zip(self.values, src_index) if idx not in labels_set
        ]
        new_values = [v for v, _ in kept]
        new_index = [idx for _, idx in kept]

        if errors == "raise":
            missing = [
                l_ for l_ in labels if l_ not in (self._index or range(len(self)))
            ]
            if missing:
                raise KeyError(f"labels not found: {missing}")

        if inplace:
            self._inner = _PySeries(new_values, self.name)
            self._index = new_index
            return self
        return Series(
            new_values, name=self.name, dtype=self._dtype_str, index=new_index
        )

    def dropna(
        self,
        axis: int = 0,
        inplace: bool = False,
        how: str = "any",
        thresh=None,
        subset=None,
    ) -> _PySeries:
        """删除缺失值。

        :param axis: 0=索引 (仅支持 0)
        :param inplace: 是否原地修改
        :param how: 'any' (有一个 NaN 就删) 或 'all' (全是 NaN 才删)
        :param thresh: 要求至少 N 个非 NaN 值
        :param subset: 对 Series 无意义（为兼容 DataFrame API 保留）
        """
        values = self.values
        index = self._index or list(range(len(values)))

        if thresh is not None:
            # thresh 模式: 保留至少 thresh 个非 NaN 值
            non_null_count = sum(1 for v in values if v is not None)
            if non_null_count < thresh:
                result = Series([], name=self.name, dtype=self._dtype_str)
                if inplace:
                    self._inner = result._inner
                    self._index = []
                    return self
                return result
            return self if inplace else self.copy()

        if how == "any":
            mask = [v is not None for v in values]
        elif how == "all":
            mask = [True] * len(values)  # Series 不会全是 NaN 除非全是
            if all(v is None for v in values):
                mask = [False] * len(values)
        else:
            raise ValueError(f"invalid how: {how}")

        new_values = [v for v, m in zip(values, mask) if m]
        new_index = [i for i, m in zip(index, mask) if m]

        if inplace:
            self._inner = _PySeries(new_values, self.name)
            self._index = new_index
            return self
        return Series(
            new_values, name=self.name, dtype=self._dtype_str, index=new_index
        )

    def isna(self) -> _PySeries:
        """isnull 的别名。"""
        return self.isnull()

    def notna(self) -> _PySeries:
        """notnull 的别名。"""
        return self.notnull()

    def nlargest(self, n: int = 5, keep: str = "first") -> _PySeries:
        """返回最大的 N 个元素。

        等价于 ``pandas.Series.nlargest``，使用 Rust 稳定排序实现，
        None/NaN 一律跳过。

        :param n: 返回的元素数量
        :param keep: 重复值的保留方式 ('first' / 'last' / 'all')
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        # 调用 Rust 层 arg_top_n 获取原始索引
        idx_list = list(self._inner.arg_top_n(n, keep, True))
        values = self.values
        result_values = [values[i] for i in idx_list]
        if self._index:
            result_index = [self._index[i] for i in idx_list]
        else:
            result_index = list(idx_list)
        return Series(
            result_values, name=self.name, dtype=self._dtype_str, index=result_index
        )

    def nsmallest(self, n: int = 5, keep: str = "first") -> _PySeries:
        """返回最小的 N 个元素。

        等价于 ``pandas.Series.nsmallest``，使用 Rust 稳定排序实现，
        None/NaN 一律跳过。

        :param n: 返回的元素数量
        :param keep: 重复值的保留方式 ('first' / 'last' / 'all')
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        idx_list = list(self._inner.arg_top_n(n, keep, False))
        values = self.values
        result_values = [values[i] for i in idx_list]
        if self._index:
            result_index = [self._index[i] for i in idx_list]
        else:
            result_index = list(idx_list)
        return Series(
            result_values, name=self.name, dtype=self._dtype_str, index=result_index
        )

    def droplevel(self, level, axis: int = 0) -> _PySeries:
        """删除索引级别 (多级索引时)。"""
        return self.copy()

    def reindex(
        self, index=None, method=None, copy=True, limit=None, tolerance=None, **kwargs
    ) -> _PySeries:
        """重新索引。

        :param index: 新的索引
        :param method: 填充方法 ('ffill'/'bfill'/'nearest'/None)
        :param copy: 是否复制数据
        :param limit: 最大连续填充次数
        :param tolerance: 最大距离限制（如 '1 day'、2.0）
        """
        if index is None:
            return self.copy()
        from .indexes import Index, RangeIndex, MultiIndex, DatetimeIndex

        # 记录原始 Index 对象引用，用于索引共享
        index_obj_ref = None
        if isinstance(index, (Index, RangeIndex, MultiIndex, DatetimeIndex)):
            index_obj_ref = index
            index_list = list(index._data) if hasattr(index, "_data") else list(index)
        elif not isinstance(index, list):
            index_list = list(index)
        else:
            index_list = index

        # 使用字典推导式构建旧索引映射
        src_index = self._index if self._index is not None else list(range(len(self)))
        old_index_map = {idx: i for i, idx in enumerate(src_index)}

        # 使用列表推导式构建新值和新索引
        self_vals = self.values
        new_values = [
            self_vals[old_index_map[label]] if label in old_index_map else None
            for label in index_list
        ]
        new_index_list = list(index_list)

        # 如果指定了 method，对缺失值进行填充
        if method is not None:
            new_values = self._apply_fill_method(
                new_values,
                src_index,
                old_index_map,
                index_list,
                method,
                limit=limit,
                tolerance=tolerance,
            )

        result = Series(
            new_values, name=self.name, dtype=self._dtype_str, index=new_index_list
        )
        # 如果传入的是 Index 对象，共享其引用
        if index_obj_ref is not None:
            result._cached_index_ref = index_obj_ref
        return result

    @staticmethod
    def _parse_tolerance(tolerance):
        """解析 tolerance 参数为统一的比较值。

        支持字符串形式的 Timedelta（如 "1 day", "2 hours", "30 minutes"）
        或数值形式（用于数值索引）。
        """
        if tolerance is None:
            return None

        from datetime import timedelta

        if isinstance(tolerance, timedelta):
            return tolerance

        if isinstance(tolerance, str):
            # 解析 pandas 风格的时间字符串
            parts = tolerance.strip().split()
            if len(parts) == 1:
                # 无空格格式: "1day", "2h", "30min"
                return Series._parse_tolerance_short(parts[0])
            elif len(parts) == 2:
                # 有空格格式: "1 day", "2 hours"
                try:
                    val = float(parts[0])
                    unit = parts[1].lower()
                    return Series._timedelta_from_unit(val, unit)
                except (ValueError, KeyError):
                    pass
            raise ValueError(f"Cannot parse tolerance: {tolerance!r}")

        if isinstance(tolerance, (int, float)):
            return float(tolerance)

        # 尝试 timedelta 转换
        try:
            return timedelta(tolerance)
        except Exception:
            return float(tolerance)

    @staticmethod
    def _parse_tolerance_short(s):
        """解析短格式如 '1day', '2h', '30min'。"""
        s = s.strip().lower()
        # 尝试提取数字部分
        num_str = ""
        unit = ""
        for ch in s:
            if ch.isdigit() or ch == ".":
                num_str += ch
            else:
                unit += ch
        if not num_str:
            raise ValueError(f"Cannot parse tolerance: {s!r}")
        val = float(num_str)
        return Series._timedelta_from_unit(val, unit)

    @staticmethod
    def _timedelta_from_unit(val, unit):
        """将数值和单位转为 timedelta。"""
        from datetime import timedelta

        unit_map = {
            "d": "days",
            "day": "days",
            "days": "days",
            "h": "hours",
            "hour": "hours",
            "hours": "hours",
            "hr": "hours",
            "m": "minutes",
            "minute": "minutes",
            "minutes": "minutes",
            "min": "minutes",
            "s": "seconds",
            "second": "seconds",
            "seconds": "seconds",
            "ms": "milliseconds",
            "millisecond": "milliseconds",
            "milliseconds": "milliseconds",
            "us": "microseconds",
            "microsecond": "microseconds",
            "microseconds": "microseconds",
            "w": "weeks",
            "week": "weeks",
            "weeks": "weeks",
        }
        if unit not in unit_map:
            raise ValueError(f"Unknown tolerance unit: {unit!r}")
        return timedelta(**{unit_map[unit]: val})

    @staticmethod
    def _compute_distance(idx_a, idx_b):
        """计算两个索引值之间的距离，返回用于比较的数值。

        对于 datetime，返回 timedelta.abs()
        对于数值，返回 abs(diff)
        其他类型返回 None（不支持）
        """
        from datetime import datetime, date

        a, b = idx_a, idx_b

        # 提取 Timestamp 的 datetime 值
        if hasattr(a, "to_pydatetime"):
            a = a.to_pydatetime()
        if hasattr(b, "to_pydatetime"):
            b = b.to_pydatetime()

        if isinstance(a, (datetime, date)) and isinstance(b, (datetime, date)):
            return abs(a - b)

        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b))

        # 尝试 timedelta 运算
        try:
            return abs(a - b)
        except (TypeError, Exception):
            return None

    @staticmethod
    def _check_tolerance(dist, tolerance):
        """检查距离是否在 tolerance 范围内。"""
        if tolerance is None or dist is None:
            return True

        from datetime import timedelta

        if isinstance(tolerance, timedelta):
            if isinstance(dist, timedelta):
                return dist <= tolerance
            # dist 是数值时，将 tolerance 转为数值（秒）
            return dist.total_seconds() <= tolerance.total_seconds()

        # tolerance 是数值，dist 可能是 timedelta
        if isinstance(dist, timedelta):
            return dist.total_seconds() <= float(tolerance)

        # 两者都是数值
        return dist <= float(tolerance)

    @staticmethod
    def _apply_fill_method(
        new_values,
        src_index,
        old_index_map,
        new_index_list,
        method,
        limit=None,
        tolerance=None,
    ):
        """对 reindex 结果中的 None 应用填充方法。

        :param new_values: reindex 后的新值列表
        :param src_index: 原始索引
        :param old_index_map: 旧索引到位置的映射
        :param new_index_list: 新索引列表
        :param method: 填充方法 ('ffill'/'bfill'/'nearest')
        :param limit: 最大连续填充次数
        :param tolerance: 最大距离限制
        :return: 填充后的值列表
        """
        if method not in ("ffill", "bfill", "nearest"):
            raise ValueError(f"invalid method: {method!r}")

        if tolerance is not None:
            tolerance = Series._parse_tolerance(tolerance)

        n = len(new_values)
        filled = list(new_values)

        # 收集有值的位置
        valid_positions = []
        for i, val in enumerate(filled):
            if val is not None:
                valid_positions.append(i)

        if not valid_positions:
            return filled

        if method == "ffill":
            # 向前填充：逐段处理，从每个有效值向后填充最多 limit 个
            for vp in valid_positions:
                count = 0
                for j in range(vp + 1, n):
                    if filled[j] is not None:
                        break
                    if limit is not None and count >= limit:
                        break
                    if tolerance is not None:
                        dist = Series._compute_distance(
                            new_index_list[j], new_index_list[vp]
                        )
                        if dist is None or not Series._check_tolerance(dist, tolerance):
                            break
                    filled[j] = filled[vp]
                    count += 1

        elif method == "bfill":
            # 向后填充：逐段处理，从每个有效值向前填充最多 limit 个
            for vp in valid_positions:
                count = 0
                for j in range(vp - 1, -1, -1):
                    if filled[j] is not None:
                        break
                    if limit is not None and count >= limit:
                        break
                    if tolerance is not None:
                        dist = Series._compute_distance(
                            new_index_list[j], new_index_list[vp]
                        )
                        if dist is None or not Series._check_tolerance(dist, tolerance):
                            break
                    filled[j] = filled[vp]
                    count += 1

        elif method == "nearest":
            # 最近填充：对每个 None，找最近的有效值
            for i in range(n):
                if filled[i] is not None:
                    continue

                best_pos = None
                best_dist = float("inf")
                for vp in valid_positions:
                    pos_dist = abs(i - vp)
                    if pos_dist < best_dist or (
                        pos_dist == best_dist and vp < best_pos
                    ):
                        # 检查 tolerance
                        if tolerance is not None:
                            idx_dist = Series._compute_distance(
                                new_index_list[i], new_index_list[vp]
                            )
                            if idx_dist is None or not Series._check_tolerance(
                                idx_dist, tolerance
                            ):
                                continue
                        best_dist = pos_dist
                        best_pos = vp
                if best_pos is not None:
                    # 检查 limit：nearest 时 limit 限制每段连续填充
                    if limit is not None:
                        # 查找周围连续 None 的段长度
                        segment_len = 0
                        for j in range(i - 1, -1, -1):
                            if filled[j] is None:
                                segment_len += 1
                            else:
                                break
                        # 只有当本段未超过 limit 时才填充
                        if segment_len >= limit:
                            continue
                    filled[i] = filled[best_pos]

        return filled

    def sort_index(
        self,
        ascending: bool = True,
        inplace: bool = False,
        kind: str = "quicksort",
        na_position: str = "last",
    ) -> _PySeries:
        """按索引排序。

        :param ascending: 是否升序
        :param inplace: 是否原地修改
        :param kind: 排序算法
        :param na_position: NaN 位置 ('first' 或 'last')
        """
        # 尝试调用 Rust 层加速（仅在默认 RangeIndex 时）
        # Rust 层 sort_index: ascending=True 保持原顺序，ascending=False 反转
        # 对默认 RangeIndex 而言，按索引值排序等价于保持/反转原顺序
        if _is_range_index(self._index):
            try:
                new_inner = self._inner.sort_index(ascending=ascending)
                new_values = list(new_inner.values)
                n = len(new_values)
                new_index = list(range(n)) if ascending else list(range(n - 1, -1, -1))
                if inplace:
                    self._inner = _PySeries(
                        new_values, self.name, dtype=self._dtype_str
                    )
                    self._index = new_index
                    return self
                return Series(
                    new_values,
                    name=self.name,
                    dtype=self._dtype_str,
                    index=new_index,
                )
            except Exception:
                pass
        # 回退到原 Python 实现（自定义索引或 Rust 调用失败时）
        if self._index is None:
            return self.copy()
        pairs = list(zip(self._index, self.values))
        try:
            pairs.sort(key=lambda x: x[0], reverse=not ascending)
        except TypeError:
            raise TypeError("cannot sort mixed types in index")
        new_index = [idx for idx, _ in pairs]
        new_values = [val for _, val in pairs]
        if inplace:
            self._inner = _PySeries(new_values, self.name, dtype=self._dtype_str)
            self._index = new_index
            return self
        return Series(
            new_values, name=self.name, dtype=self._dtype_str, index=new_index
        )

    def clip(self, lower=None, upper=None) -> _PySeries:
        """裁剪值到指定范围。

        :param lower: 下界
        :param upper: 上界
        """
        out = [
            (
                None
                if v is None
                else (
                    lower
                    if lower is not None and v < lower
                    else upper if upper is not None and v > upper else v
                )
            )
            for v in self.values
        ]
        return Series(out, name=self.name, dtype=self._dtype_str, index=self._index)

    def compare(self, other, align_axis: int = 1) -> "DataFrame":
        """与另一个 Series 比较差异。

        :param other: 另一个 Series
        :param align_axis: 对齐轴 (1=按索引对齐)
        """
        from .dataframe import DataFrame

        if not isinstance(other, Series):
            raise TypeError("other must be Series")
        self_idx = self._index if self._index is not None else range(len(self))
        other_idx = other._index if other._index is not None else range(len(other))
        all_keys = sorted(set(self_idx) | set(other_idx))
        # 列表推导式构造三元组并筛选差异
        diff_pairs = [
            (k, v1, v2)
            for k, v1, v2 in (
                (
                    key,
                    self[key] if key in self_idx else None,
                    other[key] if key in other_idx else None,
                )
                for key in all_keys
            )
            if v1 != v2
        ]
        if not diff_pairs:
            return DataFrame({"self": [], "other": []})
        keys = [k for k, _, _ in diff_pairs]
        self_vals = [v for _, v, _ in diff_pairs]
        other_vals = [v for _, _, v in diff_pairs]
        df = DataFrame({"self": self_vals, "other": other_vals})
        df._index = keys
        return df

    def transform(self, func, axis: int = 0, *args, **kwargs) -> Any:
        """对 Series 应用函数并返回相同长度的结果。

        :param func: 可调用函数、函数名或其列表
        :param axis: 轴 (未使用，保持兼容性)
        :param args: 传递给 func 的额外位置参数
        :param kwargs: 传递给 func 的关键字参数
        """
        if isinstance(func, list):
            # 多个函数 → DataFrame, 列名为函数名
            from .dataframe import DataFrame

            new_data = {}
            for f in func:
                # 确定函数名
                if isinstance(f, str):
                    fname = f
                elif callable(f):
                    fname = getattr(f, "__name__", "<lambda>")
                else:
                    fname = str(f)

                # 应用函数
                if isinstance(f, str):
                    if hasattr(self, f):
                        result = getattr(self, f)(*args, **kwargs)
                    else:
                        raise ValueError(f"Unknown function: {f}")
                else:
                    result = f(self, *args, **kwargs)

                if isinstance(result, Series):
                    new_data[fname] = list(result.values)
                else:
                    new_data[fname] = [result] * len(self)

            return DataFrame(new_data, index=self._index)
        elif isinstance(func, str):
            # 字符串方法名
            if hasattr(self, func):
                result = getattr(self, func)(*args, **kwargs)
            else:
                raise ValueError(f"Unknown function: {func}")
            if isinstance(result, Series):
                return result
            return Series([result] * len(self), name=self.name, index=self._index)
        elif callable(func):
            result = func(self, *args, **kwargs)
            if isinstance(result, Series):
                return result
            return Series([result] * len(self), name=self.name, index=self._index)
        raise TypeError("func must be callable, string, or list")

    def agg(self, func, axis: int = 0, *args, **kwargs) -> Any:
        """聚合操作。

        :param func: 聚合函数、函数名或其列表
        :param axis: 轴 (未使用，保持兼容性)
        :param args: 传递给 func 的额外位置参数
        :param kwargs: 传递给 func 的关键字参数
        """
        if isinstance(func, list):
            # 多个聚合函数 → Series, 索引为函数名
            results = []
            names = []
            for agg_func in func:
                if isinstance(agg_func, str):
                    name = agg_func
                elif callable(agg_func):
                    name = getattr(agg_func, "__name__", "<lambda>")
                else:
                    name = str(agg_func)

                if isinstance(agg_func, str):
                    if hasattr(self, agg_func):
                        results.append(getattr(self, agg_func)())
                    else:
                        raise ValueError(f"Unknown aggregation function: {agg_func}")
                elif callable(agg_func):
                    results.append(agg_func(self))
                else:
                    results.append(None)
                names.append(name)
            return Series(results, index=names, name=self.name)
        elif callable(func):
            return func(self, *args, **kwargs)
        elif isinstance(func, str):
            if func == "sum":
                return self.sum()
            elif func == "mean":
                return self.mean()
            elif func == "min":
                return self.min()
            elif func == "max":
                return self.max()
            elif func == "std":
                return self.std()
            elif func == "var":
                return self.var()
            elif func == "count":
                return self.count()
            elif func == "median":
                return self.median()
            else:
                raise ValueError(f"Unknown aggregation function: {func}")
        else:
            raise TypeError("func must be callable, string, or list")

    aggregate = agg  # 别名

    def groupby(
        self,
        by=None,
        axis: int = 0,
        level=None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        squeeze: bool = False,
        observed: bool = False,
        dropna: bool = True,
    ) -> "SeriesGroupBy":
        """分组操作。

        :param by: 分组依据 (列名、函数或 Series)
        :param axis: 分组轴 (未使用，保持兼容性)
        :param level: 多级索引的级别 (未实现)
        :param as_index: 是否将分组键作为索引 (默认 True)
        :param sort: 是否排序分组键 (默认 True)
        :param group_keys: 是否添加分组键到索引 (未实现)
        :param squeeze: 是否压缩维度 (未实现)
        :param observed: 是否仅使用观察到的分类值 (未实现)
        :param dropna: 是否丢弃 None 组 (默认 True)
        """
        return SeriesGroupBy(
            self,
            by,
            axis=axis,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            squeeze=squeeze,
            observed=observed,
            dropna=dropna,
        )

    def reindex_like(self, other, method: str = None, copy: bool = True) -> _PySeries:
        """按另一个 Series 的索引重新索引。"""
        if not isinstance(other, Series):
            raise TypeError("other must be Series")
        return self.reindex(other._index)

    def align(
        self,
        other,
        join: str = "outer",
        axis=None,
        level=None,
        copy: bool = True,
        fill_value=None,
        method=None,
    ):
        """同时按索引对齐两个 Series。

        :param other: 另一个 Series
        :param join: 连接方式 ('outer'/'left'/'right'/'inner')
        :param axis: 保留参数（Series 仅为 0），与 pandas 签名对齐
        :param level: 多级索引级别（未实现，仅占位）
        :param copy: 是否复制数据（未实现，仅占位）
        :param fill_value: 缺失值填充（填充所有 NaN，与 pandas 行为一致）
        :param method: 填充方法（未实现，仅占位）
        :return: (aligned_self, aligned_other) 两个重新索引后的 Series
        """
        if not isinstance(other, Series):
            raise TypeError(f"align requires Series input, got {type(other).__name__}")

        self_index = (
            list(self._index) if self._index is not None else list(range(len(self)))
        )
        other_index = (
            list(other._index) if other._index is not None else list(range(len(other)))
        )

        # 根据 join 计算新索引
        # pandas 行为：outer/inner 返回 sorted 顺序，left/right 保持原顺序
        if join == "outer":
            new_index = sorted(set(self_index) | set(other_index))
        elif join == "inner":
            new_index = sorted(set(self_index) & set(other_index))
        elif join == "left":
            new_index = list(self_index)
        elif join == "right":
            new_index = list(other_index)
        else:
            raise ValueError(f"invalid join: {join!r}")

        self_aligned = self.reindex(index=new_index)
        other_aligned = other.reindex(index=new_index)

        # fill_value 填充所有 NaN（包括原有与新增的），与 pandas 行为一致
        if fill_value is not None:
            self_aligned = self_aligned.fillna(fill_value)
            other_aligned = other_aligned.fillna(fill_value)

        return _AlignmentResult((self_aligned, other_aligned))

    def swaplevel(self, i: int = -2, j: int = -1) -> _PySeries:
        """交换多级索引的级别。"""
        return self.copy()

    def rename(
        self,
        index=None,
        mapper=None,
        axis: int = 0,
        copy: bool = True,
        inplace: bool = False,
        level=None,
        errors: str = "ignore",
    ) -> _PySeries:
        """重命名 Series 或索引。

        :param index: 新索引 (list) 或 None
        :param mapper: 映射函数或 dict
        :param axis: 轴
        :param copy: 是否复制
        :param inplace: 是否原地修改
        :param level: 多级索引级别
        :param errors: 错误处理 ('ignore' 或 'raise')
        """
        if isinstance(index, str):
            # 字符串参数：重命名 Series 本身
            if inplace:
                self.name = index
                return self
            return Series(
                list(self.values), name=index, dtype=self._dtype_str, index=self._index
            )

        if index is not None:
            # 非字符串：重命名索引标签（dict / callable / list-like）
            if inplace:
                if callable(index):
                    self._index = (
                        [index(i) for i in self._index]
                        if self._index is not None
                        else None
                    )
                elif isinstance(index, dict):
                    self._index = (
                        [index.get(i, i) for i in self._index]
                        if self._index is not None
                        else None
                    )
                else:
                    self._index = list(index) if not isinstance(index, list) else index
                return self
            if callable(index):
                new_index = (
                    [index(i) for i in self._index] if self._index is not None else None
                )
            elif isinstance(index, dict):
                new_index = (
                    [index.get(i, i) for i in self._index]
                    if self._index is not None
                    else None
                )
            else:
                new_index = list(index) if not isinstance(index, list) else index
            return Series(
                list(self.values),
                name=self.name,
                dtype=self._dtype_str,
                index=new_index,
            )

        if mapper is not None:
            if inplace:
                if self._index is not None:
                    self._index = [mapper(idx) for idx in self._index]
                return self
            if self._index is not None:
                return Series(
                    list(self.values),
                    name=self.name,
                    dtype=self._dtype_str,
                    index=[mapper(idx) for idx in self._index],
                )
            return self.copy()

        return self.copy()

    def rename_axis(
        self, mapper=None, axis: int = 0, copy: bool = True, inplace: bool = False
    ) -> _PySeries:
        """重命名索引轴。"""
        # Series 只有一个轴，这里简化处理
        return self.rename(mapper=mapper, axis=axis, copy=copy, inplace=inplace)

    def add(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """加法运算。"""
        return self._arith(other, "add", fill_value=fill_value)

    def sub(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """减法运算。"""
        return self._arith(other, "sub", fill_value=fill_value)

    def mul(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """乘法运算。"""
        return self._arith(other, "mul", fill_value=fill_value)

    def div(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """除法运算。"""
        return self._arith(other, "truediv", fill_value=fill_value)

    def divide(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """div 的别名。"""
        return self.div(other, level=level, fill_value=fill_value, axis=axis)

    def floordiv(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """整除运算。"""
        return self._arith(other, "floordiv", fill_value=fill_value)

    def mod(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """取模运算。"""
        return self._arith(other, "mod", fill_value=fill_value)

    def pow(self, other, level=None, fill_value=None, axis: int = 0) -> _PySeries:
        """幂运算。"""
        return self._arith(other, "pow", fill_value=fill_value)

    def divmod(self, other, level=None, fill_value=None, axis: int = 0) -> tuple:
        """同时返回整除和取模的结果。"""
        return (
            self.floordiv(other, level=level, fill_value=fill_value, axis=axis),
            self.mod(other, level=level, fill_value=fill_value, axis=axis),
        )

    def compress(self, condition) -> _PySeries:
        """按条件过滤元素。

        :param condition: 布尔列表或 Series
        """
        if isinstance(condition, Series):
            condition = condition.values
        out_values = [v for v, c in zip(self.values, condition) if c]
        out_index = [i for i, c in zip(self._index or range(len(self)), condition) if c]
        return Series(
            out_values, name=self.name, dtype=self._dtype_str, index=out_index
        )

    # ---------- 聚合 ----------

    def sum(
        self,
        axis=None,
        skipna: bool = True,
        level=None,
        numeric_only=None,
        min_count: int = 0,
    ) -> Any:
        """求和。

        :param axis: 轴 (未使用，保持兼容性)
        :param skipna: 是否忽略 None/NaN 值 (默认 True)
        :param level: 多级索引级别 (未实现)
        :param numeric_only: 是否仅计算数值 (未实现)
        :param min_count: 最少非空值数 (默认 0)
        """
        values = list(self.values)
        non_null = [v for v in values if not _is_missing(v)]
        if not skipna and len(non_null) < len(values):
            return None
        if len(non_null) < min_count:
            return None
        return self._inner.sum()

    def mean(
        self,
        axis=None,
        skipna: bool = True,
        level=None,
        numeric_only=None,
    ) -> Any:
        """均值。

        :param axis: 轴 (未使用，保持兼容性)
        :param skipna: 是否忽略 None/NaN 值 (默认 True)
        :param level: 多级索引级别 (未实现)
        :param numeric_only: 是否仅计算数值 (未实现)
        """
        values = list(self.values)
        non_null = [v for v in values if not _is_missing(v)]
        if not skipna and len(non_null) < len(values):
            return None
        return self._inner.mean()

    def min(
        self,
        axis=None,
        skipna: bool = True,
        level=None,
        numeric_only=None,
    ) -> Any:
        """最小值。

        :param axis: 轴 (未使用，保持兼容性)
        :param skipna: 是否忽略 None/NaN 值 (默认 True)
        :param level: 多级索引级别 (未实现)
        :param numeric_only: 是否仅计算数值 (未实现)
        """
        values = list(self.values)
        non_null = [v for v in values if not _is_missing(v)]
        if not skipna and len(non_null) < len(values):
            return None
        return self._inner.min()

    def max(
        self,
        axis=None,
        skipna: bool = True,
        level=None,
        numeric_only=None,
    ) -> Any:
        """最大值。

        :param axis: 轴 (未使用，保持兼容性)
        :param skipna: 是否忽略 None/NaN 值 (默认 True)
        :param level: 多级索引级别 (未实现)
        :param numeric_only: 是否仅计算数值 (未实现)
        """
        values = list(self.values)
        non_null = [v for v in values if not _is_missing(v)]
        if not skipna and len(non_null) < len(values):
            return None
        return self._inner.max()

    def count(self) -> int:
        # 过滤 None 和 NaN (np.nan 存为 Some(f64::NAN), Rust 层 count 不过滤 NaN)
        return sum(1 for v in self.values if not _is_missing(v))

    def std(
        self,
        axis=None,
        skipna: bool = True,
        level=None,
        ddof: int = 1,
        numeric_only=None,
    ) -> Any:
        """标准差。

        :param axis: 轴 (未使用，保持兼容性)
        :param skipna: 是否忽略 None/NaN 值 (默认 True)
        :param level: 多级索引级别 (未实现)
        :param ddof: 自由度调整 (默认 1)
        :param numeric_only: 是否仅计算数值 (未实现)
        """
        values = list(self.values)
        non_null = [v for v in values if not _is_missing(v)]
        if not skipna and len(non_null) < len(values):
            return None
        return self._inner.std()

    def var(
        self,
        axis=None,
        skipna: bool = True,
        level=None,
        ddof: int = 1,
        numeric_only=None,
    ) -> Any:
        """方差。

        :param axis: 轴 (未使用，保持兼容性)
        :param skipna: 是否忽略 None/NaN 值 (默认 True)
        :param level: 多级索引级别 (未实现)
        :param ddof: 自由度调整 (默认 1)
        :param numeric_only: 是否仅计算数值 (未实现)
        """
        values = list(self.values)
        non_null = [v for v in values if not _is_missing(v)]
        if not skipna and len(non_null) < len(values):
            return None
        return self._inner.var()

    def sem(
        self,
        axis=None,
        skipna: bool = True,
        level=None,
        ddof: int = 1,
        numeric_only=None,
    ) -> Any:
        """返回平均值的标准误差。

        :param axis: 轴 (未使用，保持兼容性)
        :param skipna: 是否忽略 None/NaN 值 (默认 True)
        :param level: 多级索引级别 (未实现)
        :param ddof: 自由度调整 (默认 1)
        :param numeric_only: 是否仅计算数值 (未实现)
        """
        values = list(self.values)
        non_null = [v for v in values if not _is_missing(v)]
        if not skipna and len(non_null) < len(values):
            return None
        n = len(non_null)
        if n - ddof <= 0:
            return None
        m = sum(non_null) / n
        var = sum((v - m) ** 2 for v in non_null) / (n - ddof)
        return (var**0.5) / (n**0.5)

    def median(
        self,
        axis=None,
        skipna: bool = True,
        level=None,
        numeric_only=None,
    ) -> Any:
        """中位数。

        :param axis: 轴 (未使用，保持兼容性)
        :param skipna: 是否忽略 None/NaN 值 (默认 True)
        :param level: 多级索引级别 (未实现)
        :param numeric_only: 是否仅计算数值 (未实现)
        """
        values = list(self.values)
        non_null = [v for v in values if not _is_missing(v)]
        if not skipna and len(non_null) < len(values):
            return None
        return self._inner.median()

    def any(self) -> Any:
        return self._inner.any()

    def all(self) -> Any:
        return self._inner.all()

    def describe(
        self,
        percentiles=None,
        include=None,
        exclude=None,
    ) -> _PySeries:
        """返回统计摘要 Series。

        数值型: count/mean/std/min/分位数/max
        非数值型: count/unique/top/freq

        :param percentiles: 要包含的分位数列表 (默认 [0.25, 0.5, 0.75])
        :param include: 要包含的数据类型 (Series 未使用，保持签名兼容)
        :param exclude: 要排除的数据类型 (Series 未使用，保持签名兼容)
        """
        from collections import Counter

        non_null = [v for v in self.values if not _is_missing(v)]

        # 判断是否为数值型 (dtype 为 int64/float64，或所有非空值均为数值)
        is_numeric = self._dtype_str in ("int64", "float64") or (
            non_null
            and all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in non_null
            )
        )

        if not is_numeric:
            # 非数值型: count/unique/top/freq
            counter = Counter(non_null)
            if counter:
                # most_common(1) 返回频次最高且首次出现的值 (与 pandas 一致)
                top, freq = counter.most_common(1)[0]
            else:
                top, freq = None, 0
            stats = {
                "count": len(non_null),
                "unique": len(counter),
                "top": top,
                "freq": freq,
            }
            return Series(
                list(stats.values()), index=list(stats.keys()), dtype="object"
            )

        # 数值型：调用 Rust 层 batch_agg 一次性得到 count/mean/std/min/max
        if percentiles is None:
            percentiles = [0.25, 0.5, 0.75]
        else:
            percentiles = list(percentiles)
        if 0.5 not in percentiles:
            percentiles = list(percentiles) + [0.5]
        for p in percentiles:
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"percentiles 应在 [0, 1] 范围内，得到 {p}")
        percentiles = sorted(set(percentiles))

        # Rust 批量聚合: count, mean, std, min, max
        aggs = ["count", "mean", "std", "min", "max"]
        try:
            agg_result = list(self._inner.batch_agg(aggs))
            count_val = agg_result[0] if len(agg_result) > 0 else None
            mean_val = agg_result[1] if len(agg_result) > 1 else None
            std_val = agg_result[2] if len(agg_result) > 2 else None
            min_val = agg_result[3] if len(agg_result) > 3 else None
            max_val = agg_result[4] if len(agg_result) > 4 else None
        except Exception:
            agg_result = None
            count_val = mean_val = std_val = min_val = max_val = None

        # 分位数仍然用 Python 层计算（Rust quantile 要多次调用排序，
        # 这里 Python 一次 sort 可复用）
        vals = [
            float(v)
            for v in non_null
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        n = len(vals)

        if n == 0:
            stat_names = (
                ["count", "mean", "std", "min"]
                + [f"{int(p * 100)}%" for p in percentiles]
                + ["max"]
            )
            return Series(
                [0.0] + [float("nan")] * (len(stat_names) - 1),
                index=stat_names,
            )

        # 若 Rust 聚合失败则回退 Python 计算
        if count_val is None:
            count_val = float(n)
        if mean_val is None:
            mean_val = sum(vals) / n
        if std_val is None:
            if n > 1:
                m = mean_val
                std_val = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5
            else:
                std_val = float("nan")

        sorted_vals = sorted(vals)
        if min_val is None:
            min_val = sorted_vals[0]
        if max_val is None:
            max_val = sorted_vals[-1]

        def _quantile(q: float) -> float:
            """线性插值法计算分位数。"""
            if len(sorted_vals) == 1:
                return sorted_vals[0]
            pos = q * (len(sorted_vals) - 1)
            lo = int(pos)
            hi = min(lo + 1, len(sorted_vals) - 1)
            frac = pos - lo
            return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac

        stats = {
            "count": float(count_val),
            "mean": float(mean_val),
            "std": float(std_val),
            "min": float(min_val),
        }
        stats.update({f"{int(p * 100)}%": _quantile(p) for p in percentiles})
        stats["max"] = float(max_val)

        return Series(list(stats.values()), index=list(stats.keys()))

    # ---------- 极值位置 (v1.0.0) ----------

    def argmax(self) -> int:
        """返回最大值的位置索引 (整数位置)。"""
        values = self.values
        # 使用列表推导式收集非空索引，再用 max + key 查找最大值索引
        non_null_indices = [i for i, v in enumerate(values) if v is not None]
        if not non_null_indices:
            return None
        return max(non_null_indices, key=lambda i: values[i])

    def argmin(self) -> int:
        """返回最小值的位置索引 (整数位置)。"""
        values = self.values
        # 使用列表推导式收集非空索引，再用 min + key 查找最小值索引
        non_null_indices = [i for i, v in enumerate(values) if v is not None]
        if not non_null_indices:
            return None
        return min(non_null_indices, key=lambda i: values[i])

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

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace: bool = False,
        limit=None,
        downcast=None,
    ) -> _PySeries:
        """填充缺失值。

        :param value: 用于填充的值 (标量、字典或 Series)
        :param method: 填充方法 ('backfill'/'bfill'/'pad'/'ffill'/None)
        :param axis: 轴 (未使用，保持兼容性)
        :param inplace: 是否原地修改 (默认 False)
        :param limit: 填充的最大连续数量 (未实现)
        :param downcast: 类型降级 (未实现)
        """
        if method is not None and value is not None:
            raise ValueError("Cannot specify both 'value' and 'method'")

        if method is not None:
            # 实现填充方法
            from itertools import accumulate

            values = self.values

            if method in ("pad", "ffill"):
                # 前向填充: 用上一个非 None 值填充当前 None
                values = list(
                    accumulate(values, lambda last, v: v if v is not None else last)
                )
            elif method in ("backfill", "bfill"):
                # 后向填充: 反向 accumulate 后再反转
                values = list(
                    accumulate(
                        reversed(values),
                        lambda last, v: v if v is not None else last,
                    )
                )[::-1]
            else:
                raise ValueError(f"Invalid fill method: {method}")

            if inplace:
                self._inner = _PySeries(values, self.name, dtype=self._dtype_str)
                return self
            return Series(
                values, name=self.name, dtype=self._dtype_str, index=self._index
            )

        # 标准值填充
        if value is None:
            raise ValueError("Must specify a fill 'value' or 'method'")

        # 优先 Python 层实现（避免 Rust 层跨边界类型不匹配）
        # Rust 层 fillna: Int64 列要求 value 为 int，Float64 要求 f64，Object 要求 String
        # Python 层实现能兼容 value=0.0 等标量，并自动处理 upcasting
        vals = list(self.values)
        filled = [value if _is_missing(v) else v for v in vals]
        # 推断新的 dtype（int 列 + float value → float64）
        new_dtype = self._dtype_str
        if new_dtype in ("int8", "int16", "int32", "int64", "int") and isinstance(
            value, float
        ):
            new_dtype = "float64"
        if inplace:
            self._inner = _PySeries(filled, self.name, dtype=new_dtype)
            self._dtype_str = new_dtype
            return self
        return Series(filled, name=self.name, dtype=new_dtype, index=self._index)

    # ---------- 唯一值 ----------

    def unique(self) -> _PySeries:
        """返回去重后的 Series (保持首次出现顺序)。"""
        return Series(self._inner.unique(), name=self.name, dtype=self._dtype_str)

    def nunique(self, dropna: bool = True) -> int:
        """返回不同值的数量。

        :param dropna: 是否排除 None/NaN (默认 True)
        """
        values = list(self.values)
        if dropna:
            values = [v for v in values if not _is_missing(v)]
        try:
            return len(set(values))
        except TypeError:
            # 含不可哈希类型，用列表去重
            seen = []
            for v in values:
                if v not in seen:
                    seen.append(v)
            return len(seen)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ) -> _PySeries:
        """统计每个值出现的次数。

        :param normalize: 是否返回比例而非计数 (默认 False)
        :param sort: 是否按计数排序 (默认 True)
        :param ascending: 是否升序排序 (默认 False，即降序)
        :param bins: 暂不支持，pandas 用于数值分箱
        :param dropna: 是否忽略 None 值 (默认 True)
        """
        if bins is not None:
            raise NotImplementedError("bins parameter is not supported yet")
        if not dropna:
            # dropna=False 保留 None：回退到 Python 实现
            values_list = self.values
            from collections import Counter

            counter = Counter(values_list)
            items = list(counter.items())
            if sort:
                items.sort(key=lambda x: x[1], reverse=not ascending)
            unique_values = [v for v, _ in items]
            counts = [c for _, c in items]
            if normalize:
                total = sum(counts)
                counts = [c / total for c in counts]
            return Series(counts, index=unique_values, name=self.name)

        # 默认路径 (dropna=True)：调用 Rust 层快速实现
        vals, cnts = self._inner.value_counts(sort, ascending)
        counts = list(cnts)
        if normalize:
            total = sum(counts)
            counts = [c / total for c in counts]
        return Series(counts, index=list(vals), name=self.name)

    # ---------- 统计方法 (v1.0.0) ----------

    def rank(
        self,
        method: str = "average",
        numeric_only=None,
        na_option: str = "keep",
        ascending: bool = True,
        pct: bool = False,
    ) -> _PySeries:
        """计算排名。

        :param method: 排名方法 ('average'/'min'/'max'/'first'/'dense')
        :param numeric_only: 是否仅计算数值 (未实现)
        :param na_option: None 值处理 ('keep'/'top'/'bottom')
        :param ascending: 是否升序排名 (默认 True)
        :param pct: 是否返回百分比排名 (默认 False)
        """
        # 优先调用 Rust 层（现已支持 dense + na_option=keep/top/bottom）
        if (
            method in ("average", "min", "max", "first", "dense")
            and na_option in ("keep", "top", "bottom")
            and numeric_only is None
        ):
            ranks = self._inner.rank(method, ascending, na_option)
            if pct:
                # pct=True: 除以最大排名（非 None）
                valid_ranks = [r for r in ranks if r is not None]
                if valid_ranks:
                    max_rank = max(valid_ranks)
                    ranks = [(r / max_rank) if r is not None else None for r in ranks]
            return Series(ranks, name=self.name, index=self._index)

        # 未知 method 才回退到 Python
        raise ValueError(f"Unsupported rank method: {method}")

    def quantile(self, q=0.5, interpolation: str = "linear") -> float:
        """计算分位数。

        :param q: 分位数值 (0.0-1.0) 或列表
        :param interpolation: 插值方法 ('linear'/'lower'/'higher'/'midpoint'/'nearest')
        """
        # 优先调用 Rust 层（仅支持单个 q 且 linear 插值）
        if not isinstance(q, (list, tuple)) and interpolation == "linear":
            try:
                result = self._inner.quantile(q)
                if result is not None:
                    return result
            except Exception:
                pass

        # 回退到 Python 实现
        values = [v for v in self.values if v is not None]
        if not values:
            return None

        # 支持多个分位数
        if isinstance(q, (list, tuple)):
            return [self._quantile_single(q_val, values, interpolation) for q_val in q]

        return self._quantile_single(q, values, interpolation)

    def _quantile_single(self, q: float, values: list, interpolation: str) -> float:
        """计算单个分位数。"""
        values = sorted(values)
        n = len(values)
        if n == 1:
            return values[0]

        pos = q * (n - 1)
        lower = int(pos)
        upper = min(lower + 1, n - 1)
        frac = pos - lower

        if interpolation == "linear":
            return values[lower] * (1 - frac) + values[upper] * frac
        elif interpolation == "lower":
            return values[lower]
        elif interpolation == "higher":
            return values[upper]
        elif interpolation == "midpoint":
            return (values[lower] + values[upper]) / 2
        elif interpolation == "nearest":
            return values[lower] if frac < 0.5 else values[upper]
        else:
            raise ValueError(f"Invalid interpolation method: {interpolation}")

    def searchsorted(self, value, side: str = "left", sorter=None) -> "rnp.ndarray":
        """在已升序的 Series 中查找 value 的插入位置（二分查找）。

        等价于 ``numpy.searchsorted`` / ``pandas.Series.searchsorted``。

        :param value: 标量或 list/tuple/ndarray，要查找的目标值
        :param side: ``"left"``（默认，bisect_left）或 ``"right"``（bisect_right）
        :param sorter: 可选整数索引数组（长度与 self 相同），使 ``self[sorter]``
                       为升序。当 self 本身无序时通过传入 ``np.argsort(self)``
                       实现正确的 searchsorted。可为 list/tuple/ndarray。
        :return: rsnumpy 一维 ndarray（dtype=int64），长度与 ``value`` 扁平化后相同。
                 单个标量输入时仍返回长度为 1 的数组（与 pandas 行为一致）。
        """
        import rsnumpy as rnp

        # 预处理 sorter：若传入 rsnumpy.ndarray 则 list(tuple) 化以适配 Rust 端
        # sorter 也可能是 numpy-like，取 tolist/迭代即可。
        if sorter is not None:
            if hasattr(sorter, "tolist"):
                _sorter = sorter.tolist()
            else:
                _sorter = list(sorter)
        else:
            _sorter = None
        indices = list(self._inner.searchsorted(value, side, _sorter))
        arr = rnp.ndarray(indices, _dtype="int64")
        return arr

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

    def skew(
        self,
        axis=None,
        skipna: bool = True,
        level=None,
        numeric_only=None,
        bias: bool = True,
    ) -> float:
        """计算样本偏度 (3rd moment)。

        :param axis: 轴 (未使用，保持兼容性)
        :param skipna: 是否忽略 None/NaN 值 (默认 True)
        :param level: 多级索引级别 (未实现)
        :param numeric_only: 是否仅计算数值 (未实现)
        :param bias: 是否计算有偏估计 (默认 True; False 时进行无偏修正)
        """
        values = list(self.values)
        non_null = [v for v in values if not _is_missing(v)]
        if not skipna and len(non_null) < len(values):
            return None
        n = len(non_null)
        if n < 3:
            return None
        m = sum(non_null) / n
        var = sum((x - m) ** 2 for x in non_null) / n
        if var == 0:
            return 0.0
        std = var**0.5
        g = sum((x - m) ** 3 for x in non_null) / (n * std**3)
        if not bias:
            # 无偏修正 (与 pandas/scipy 一致)
            g *= (n * (n - 1)) ** 0.5 / (n - 2)
        return g

    def kurt(
        self,
        axis=None,
        skipna: bool = True,
        level=None,
        numeric_only=None,
        bias: bool = True,
        fisher: bool = True,
    ) -> float:
        """计算样本峰度 (4th moment)。

        :param axis: 轴 (未使用，保持兼容性)
        :param skipna: 是否忽略 None/NaN 值 (默认 True)
        :param level: 多级索引级别 (未实现)
        :param numeric_only: 是否仅计算数值 (未实现)
        :param bias: 是否计算有偏估计 (默认 True; False 时进行无偏修正)
        :param fisher: True 时返回 Fisher 峰度 (正态分布=0); False 时返回 Pearson 峰度 (+3)
        """
        values = list(self.values)
        non_null = [v for v in values if not _is_missing(v)]
        if not skipna and len(non_null) < len(values):
            return None
        n = len(non_null)
        if n < 4:
            return None
        m = sum(non_null) / n
        var = sum((x - m) ** 2 for x in non_null) / n
        if var == 0:
            return 0.0
        std = var**0.5
        g2 = sum((x - m) ** 4 for x in non_null) / (n * std**4) - 3
        if not bias and n > 3:
            # 无偏修正 (与 pandas/scipy 一致)
            factor = ((n - 1) * (n + 1)) / ((n - 2) * (n - 3))
            g2 = factor * (g2 + 6 / (n + 1))
        if not fisher:
            g2 += 3
        return g2

    # ---------- 过滤 ----------

    def filter(self, mask: list) -> _PySeries:
        return self._filter_mask(mask, preserve_dtype=True)

    # ---------- 时序操作 (v1.0.0) ----------

    def shift(self, periods: int = 1) -> _PySeries:
        """将数据移动 periods 位。
        :param periods: 移动位数 (正数向后, 负数向前)
        """
        values = self.values
        n = len(values)
        out: list = [None] * n
        if periods > 0 and periods < n:
            # 切片批量赋值，比逐元素循环快
            out[periods:] = values[: n - periods]
        elif periods < 0 and -periods < n:
            out[: n + periods] = values[-periods:]
        elif periods == 0:
            out = list(values)
        return Series(out, name=self.name, index=self._index)

    def diff(self, periods: int = 1) -> _PySeries:
        """计算相邻元素的差。
        :param periods: 间隔位数
        """
        from datetime import datetime as _dt, timedelta as _td

        values = self.values
        n = len(values)
        # 判断是否为 datetime 列（DatetimeSeries 存储为 ISO 字符串）
        is_datetime = False
        if values:
            first_non_none = next((v for v in values if v is not None), None)
            if isinstance(first_non_none, str):
                # 尝试解析为 datetime
                from ._datetime import _parse_iso

                try:
                    _parse_iso(first_non_none)
                    is_datetime = True
                except (ValueError, TypeError):
                    pass
            elif isinstance(first_non_none, (_dt, _td)):
                is_datetime = True

        if is_datetime:
            from ._datetime import _parse_iso

            # 将字符串解析回 datetime/timedelta 对象
            def _parse_v(v):
                if v is None:
                    return None
                if isinstance(v, _dt):
                    return v
                if isinstance(v, _td):
                    return v
                try:
                    return _parse_iso(v)
                except (ValueError, TypeError):
                    return None

            parsed = [_parse_v(v) for v in values]
        else:
            parsed = values

        if periods == 0:
            out = [
                _td(0) if is_datetime else 0.0 if v is not None else None
                for v in parsed
            ]
            return Series(out, name=self.name, index=self._index)
        if periods > 0:
            if periods >= n:
                out = [None] * n
            else:
                head = [None] * periods
                tail = [
                    None if a is None or b is None else a - b
                    for a, b in zip(parsed[periods:], parsed[: n - periods])
                ]
                out = head + tail
        else:
            if -periods >= n:
                out = [None] * n
            else:
                tail_pad = [None] * (-periods)
                head = [
                    None if a is None or b is None else a - b
                    for a, b in zip(parsed[: n + periods], parsed[-periods:])
                ]
                out = head + tail_pad
        # datetime 列的 diff 返回 timedelta，转换为 pandas 兼容字符串
        if is_datetime:
            from .dataframe import _convert_to_basic

            out = [_convert_to_basic(v) if v is not None else None for v in out]
            result = Series(out, name=self.name, index=self._index)
            # 标记为 timedelta64[us] dtype（与 pandas 行为一致）
            result._dtype_str = "timedelta64[us]"
            return result
        return Series(out, name=self.name, index=self._index)

    def pct_change(self, periods: int = 1, fill_method: str = "pad") -> _PySeries:
        """计算百分比变化。
        :param periods: 间隔位数
        :param fill_method: 'pad' (填充前值) / 'backfill' / None
        """
        values = self.values
        n = len(values)
        if periods == 0 or abs(periods) >= n:
            out = [None] * n
            return Series(out, name=self.name, index=self._index)
        if periods > 0:
            cur = values[periods:]
            prev = values[: n - periods]
            pad = [None] * periods
        else:
            cur = values[: n + periods]
            prev = values[-periods:]
            pad = [None] * (-periods)
        # 列表推导式：批量计算 (a - b) / b
        body = [
            None if a is None or b is None or b == 0 else (a - b) / b
            for a, b in zip(cur, prev)
        ]
        # periods > 0 时，前 periods 个没有前值，需要前部填 None
        # periods < 0 时，后 |periods| 个没有前值，需要后部填 None
        out = pad + body if periods > 0 else body + pad
        return Series(out, name=self.name, index=self._index)

    def cumsum(
        self,
        axis=None,
        skipna: bool = True,
        level=None,
        numeric_only=None,
        *args,
        **kwargs,
    ) -> _PySeries:
        """累加和。

        :param axis: 轴 (未使用，保持兼容性)
        :param skipna: 是否跳过 None/NaN 值 (默认 True)
        :param level: 多级索引级别 (未实现)
        :param numeric_only: 是否仅计算数值 (未实现)
        """
        from itertools import accumulate

        values = self.values
        if skipna:
            # 跳过 None，保留前一个有效累加值
            acc: Any = None
            out = []
            for v in values:
                if v is None:
                    out.append(acc)
                else:
                    acc = v if acc is None else acc + v
                    out.append(acc)
        else:
            # 遇到 None 即重置为 None
            out = list(
                accumulate(
                    values, lambda a, b: None if a is None or b is None else a + b
                )
            )
        return Series(out, name=self.name, index=self._index)

    def cumprod(
        self,
        axis=None,
        skipna: bool = True,
        level=None,
        numeric_only=None,
        *args,
        **kwargs,
    ) -> _PySeries:
        """累乘积。

        :param axis: 轴 (未使用，保持兼容性)
        :param skipna: 是否跳过 None/NaN 值 (默认 True)
        :param level: 多级索引级别 (未实现)
        :param numeric_only: 是否仅计算数值 (未实现)
        """
        from itertools import accumulate

        values = self.values
        if skipna:
            acc: Any = None
            out = []
            for v in values:
                if v is None:
                    out.append(acc)
                else:
                    acc = v if acc is None else acc * v
                    out.append(acc)
        else:
            out = list(
                accumulate(
                    values, lambda a, b: None if a is None or b is None else a * b
                )
            )
        return Series(out, name=self.name, index=self._index)

    def cummax(
        self,
        axis=None,
        skipna: bool = True,
        level=None,
        numeric_only=None,
        *args,
        **kwargs,
    ) -> _PySeries:
        """累计最大值。

        :param axis: 轴 (未使用，保持兼容性)
        :param skipna: 是否跳过 None/NaN 值 (默认 True)
        :param level: 多级索引级别 (未实现)
        :param numeric_only: 是否仅计算数值 (未实现)
        """
        from itertools import accumulate

        values = self.values
        if skipna:
            acc: Any = None
            out = []
            for v in values:
                if v is None:
                    out.append(acc)
                else:
                    acc = v if acc is None else max(acc, v)
                    out.append(acc)
        else:
            out = list(
                accumulate(
                    values, lambda a, b: None if a is None or b is None else max(a, b)
                )
            )
        return Series(out, name=self.name, index=self._index)

    def cummin(
        self,
        axis=None,
        skipna: bool = True,
        level=None,
        numeric_only=None,
        *args,
        **kwargs,
    ) -> _PySeries:
        """累计最小值。

        :param axis: 轴 (未使用，保持兼容性)
        :param skipna: 是否跳过 None/NaN 值 (默认 True)
        :param level: 多级索引级别 (未实现)
        :param numeric_only: 是否仅计算数值 (未实现)
        """
        from itertools import accumulate

        values = self.values
        if skipna:
            acc: Any = None
            out = []
            for v in values:
                if v is None:
                    out.append(acc)
                else:
                    acc = v if acc is None else min(acc, v)
                    out.append(acc)
        else:
            out = list(
                accumulate(
                    values, lambda a, b: None if a is None or b is None else min(a, b)
                )
            )
        return Series(out, name=self.name, index=self._index)

    # ---------- 窗口函数 (v1.0.0) ----------

    def rolling(
        self,
        window: int,
        min_periods: Optional[int] = None,
        center: bool = False,
        win_type: Optional[str] = None,
        closed: Optional[str] = None,
    ) -> "Rolling":
        """返回 Rolling 窗口对象。

        :param window: 窗口大小
        :param min_periods: 最少非空值数 (默认 = window)
        :param center: 是否居中窗口
        :param win_type: 窗口类型 (boxcar/triang/blackman 等)
        :param closed: 闭合方式 (right/left/both/neither)
        """
        if window < 1:
            raise ValueError("window must be >= 1")
        if min_periods is None:
            min_periods = window
        return Rolling(
            self, window, min_periods, center=center, win_type=win_type, closed=closed
        )

    def expanding(self, min_periods: int = 1) -> "Expanding":
        """返回 Expanding 窗口对象。"""
        if min_periods < 1:
            raise ValueError("min_periods must be >= 1")
        return Expanding(self, min_periods)

    def ewm(
        self,
        alpha: Optional[float] = None,
        span: Optional[int] = None,
        halflife: Optional[float] = None,
        com: Optional[float] = None,
        adjust: bool = True,
    ) -> "EWM":
        """返回 EWM 指数加权移动窗口对象 (v1.4.0)。

        :param alpha: 平滑因子 (0 < alpha <= 1)
        :param span: N 日跨度，alpha = 2/(span+1)
        :param halflife: 半衰期，alpha = 1 - exp(-log(2)/halflife)
        :param com: 质心，alpha = 1/(com+1)
        :param adjust: 是否使用调整因子
        :return: EWM 对象
        """
        return EWM(
            self, alpha=alpha, span=span, halflife=halflife, com=com, adjust=adjust
        )

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
                "resample requires a datetime index; " "use to_datetime() to convert"
            )
        return Resampler(self, freq, index)

    # ---------- 填充 / 插值 ----------

    def ffill(self, limit=None) -> _PySeries:
        """前向填充缺失值。

        :param limit: 最大连续填充数量
        """
        # 尝试调用 Rust 层加速（limit=None 时，Rust 层暂不支持 limit 参数）
        if limit is None:
            try:
                new_inner = self._inner.ffill()
                return Series(new_inner, name=self.name, index=self._index)
            except Exception:
                pass
        # 回退到原 Python 实现
        values = self.values
        result = []
        last_valid = None
        fill_count = 0
        for v in values:
            if v is not None:
                result.append(v)
                last_valid = v
                fill_count = 0
            else:
                if last_valid is not None and (limit is None or fill_count < limit):
                    result.append(last_valid)
                    fill_count += 1
                else:
                    result.append(None)
        return Series(result, name=self.name, dtype=self._dtype_str, index=self._index)

    def bfill(self, limit=None) -> _PySeries:
        """后向填充缺失值。

        :param limit: 最大连续填充数量
        """
        # 尝试调用 Rust 层加速（limit=None 时，Rust 层暂不支持 limit 参数）
        if limit is None:
            try:
                new_inner = self._inner.bfill()
                return Series(new_inner, name=self.name, index=self._index)
            except Exception:
                pass
        # 回退到原 Python 实现
        values = self.values
        n = len(values)
        result = [None] * n
        next_valid = None
        fill_count = 0
        for i in range(n - 1, -1, -1):
            if values[i] is not None:
                result[i] = values[i]
                next_valid = values[i]
                fill_count = 0
            else:
                if next_valid is not None and (limit is None or fill_count < limit):
                    result[i] = next_valid
                    fill_count += 1
                else:
                    result[i] = None
        return Series(result, name=self.name, dtype=self._dtype_str, index=self._index)

    def pad(self, limit=None) -> _PySeries:
        """pad 的别名，等价于 ffill。"""
        return self.ffill(limit=limit)

    def backfill(self, limit=None) -> _PySeries:
        """backfill 的别名，等价于 bfill。"""
        return self.bfill(limit=limit)

    def interpolate(
        self,
        method: str = "linear",
        axis: int = 0,
        limit=None,
        inplace: bool = False,
        limit_direction: str = "forward",
        limit_area=None,
        downcast=None,
        **kwargs,
    ) -> _PySeries:
        """插值填充缺失值。

        :param method: 插值方法 ('linear'/'pad'/'index'/'nearest'/'zero'/'slinear')
        :param axis: 轴（仅支持 0）
        :param limit: 最大连续填充数量
        :param inplace: 是否原地修改
        :param limit_direction: 填充方向 ('forward'/'backward'/'both')
        :param limit_area: 区域限制 ('inside'/'outside'/None)
        :param downcast: 类型降级（未实现）
        """
        # 优先调用 Rust 层 interpolate
        if method == "linear":
            try:
                limit_n = int(limit) if limit is not None else None
                result_inner = self._inner.interpolate("linear", limit_n)
                new_s = Series.__new__(Series)
                new_s._inner = result_inner
                new_s._dtype_str = "float64"
                new_s._index = list(self._index) if self._index is not None else None
                new_s._name = self.name
                if inplace:
                    self._inner = result_inner
                    self._dtype_str = "float64"
                    return self
                return new_s
            except Exception:
                pass

        values = self.values
        n = len(values)

        if method in ("pad", "ffill"):
            result = self.ffill(limit=limit)
            return result if not inplace else self._apply_inplace(result)

        if method in ("backfill", "bfill"):
            result = self.bfill(limit=limit)
            return result if not inplace else self._apply_inplace(result)

        # 线性插值
        result = list(values)
        i = 0
        while i < n:
            if result[i] is not None:
                i += 1
                continue
            # 找到连续的 None 区间 [i, j)
            j = i
            while j < n and result[j] is None:
                j += 1
            # 左侧有效值
            left_val = result[i - 1] if i > 0 and result[i - 1] is not None else None
            # 右侧有效值
            right_val = result[j] if j < n and result[j] is not None else None

            if left_val is not None and right_val is not None:
                # 线性插值
                gap = j - i
                for k in range(i, j):
                    if limit is not None and (k - i) >= limit:
                        break
                    frac = (k - i + 1) / (gap + 1)
                    result[k] = left_val + (right_val - left_val) * frac
            elif left_val is not None and method == "linear":
                # 只有左侧值，前向填充
                for k in range(i, j):
                    if limit is not None and (k - i) >= limit:
                        break
                    result[k] = left_val
            elif right_val is not None and method == "linear":
                # 只有右侧值，后向填充
                for k in range(i, j):
                    if limit is not None and (j - k - 1) >= limit:
                        continue
                    result[k] = right_val
            i = j

        new_s = Series(result, name=self.name, dtype="float64", index=self._index)
        if inplace:
            self._inner = _PySeries(result, self.name)
            return self
        return new_s

    def _apply_inplace(self, result):
        """辅助方法：原地应用结果。"""
        self._inner = _PySeries(list(result.values), self.name)
        self._index = list(result._index) if result._index is not None else None
        return self

    # ---------- 聚合扩展 ----------

    def prod(
        self,
        axis=None,
        skipna: bool = True,
        level=None,
        numeric_only=None,
        min_count: int = 0,
    ) -> Any:
        """返回所有元素的乘积。

        :param axis: 轴 (未使用，保持兼容性)
        :param skipna: 是否跳过 None/NaN 值 (默认 True)
        :param level: 多级索引级别 (未实现)
        :param numeric_only: 是否仅计算数值 (未实现)
        :param min_count: 最少非空值数 (默认 0)
        """
        import math

        values = list(self.values)
        non_null = [v for v in values if not _is_missing(v)]
        if not skipna and len(non_null) < len(values):
            return None
        if len(non_null) < min_count:
            return None
        if not non_null:
            return 1
        return math.prod(non_null)

    product = prod

    def dot(self, other) -> Any:
        """点积。

        :param other: Series 或 list
        """
        if isinstance(other, Series):
            other_vals = other.values
        elif isinstance(other, (list, tuple)):
            other_vals = list(other)
        else:
            raise TypeError(f"unsupported type: {type(other).__name__}")

        if len(self) != len(other_vals):
            raise ValueError("lengths must match")
        return sum(
            a * b
            for a, b in zip(self.values, other_vals)
            if a is not None and b is not None
        )

    def autocorr(self, lag: int = 1) -> float:
        """自相关系数。

        :param lag: 滞后阶数
        """
        values = [v for v in self.values if v is not None]
        n = len(values)
        if n <= lag:
            return None
        s1 = values[: n - lag]
        s2 = values[lag:]
        mean1 = sum(s1) / len(s1)
        mean2 = sum(s2) / len(s2)
        num = sum((a - mean1) * (b - mean2) for a, b in zip(s1, s2))
        den1 = sum((a - mean1) ** 2 for a in s1) ** 0.5
        den2 = sum((b - mean2) ** 2 for b in s2) ** 0.5
        if den1 == 0 or den2 == 0:
            return None
        return num / (den1 * den2)

    # ---------- 四舍五入 ----------

    def round(self, decimals: int = 0) -> _PySeries:
        """四舍五入。

        :param decimals: 小数位数
        """
        out = [None if v is None else round(v, decimals) for v in self.values]
        return Series(out, name=self.name, dtype=self._dtype_str, index=self._index)

    # ---------- 索引操作扩展 ----------

    def reset_index(
        self,
        level=None,
        drop: bool = False,
        inplace: bool = False,
        col_level: int = 0,
        col_fill: str = "",
    ) -> "_PySeries":
        """重置索引。

        :param level: 要重置的索引级别
        :param drop: 是否丢弃索引列
        :param inplace: 是否原地修改
        """
        if drop:
            new_s = Series(list(self.values), name=self.name, dtype=self._dtype_str)
        else:
            # 将索引作为新列返回（对于 Series，转为 DataFrame 更合适）
            # pandas 行为：返回 DataFrame，包含 index 列和原 Series 列
            from .dataframe import DataFrame

            index_name = "index"
            col_name = self.name if self.name else "0"
            new_data = {
                index_name: (
                    list(self._index)
                    if self._index is not None
                    else list(range(len(self)))
                ),
                col_name: list(self.values),
            }
            df = DataFrame(new_data)
            if inplace:
                raise TypeError(
                    "Cannot use inplace=True when reset_index returns a DataFrame"
                )
            return df

        if inplace:
            self._inner = _PySeries(list(new_s.values), self.name)
            self._index = None
            return self
        return new_s

    def pop(self, item) -> Any:
        """弹出索引项并返回值。

        :param item: 索引标签
        """
        if self._index is not None and item in self._index:
            pos = self._index.index(item)
            val = self.values[pos]
            new_values = self.values[:pos] + self.values[pos + 1 :]  # noqa
            new_index = self._index[:pos] + self._index[pos + 1 :]  # noqa
            self._inner = _PySeries(new_values, self.name)
            self._index = new_index
            return val
        raise KeyError(f"'{item}' not found in index")

    def keys(self) -> list:
        """返回索引。"""
        return self.index

    def items(self) -> list:
        """返回 (index, value) 列表。"""
        return list(zip(self.index, self.values))

    def iteritems(self) -> list:
        """items 的别名。"""
        return self.items()

    def first_valid_index(self):
        """返回第一个非 None 值的索引。"""
        # 使用 next() + 生成器表达式查找首个非空值
        idx = self._index if self._index is not None else list(range(len(self)))
        try:
            return next(idx[i] for i, v in enumerate(self.values) if v is not None)
        except StopIteration:
            return None

    def last_valid_index(self):
        """返回最后一个非 None 值的索引。"""
        # 使用 next() + 生成器表达式查找末个非空值
        idx = self._index if self._index is not None else list(range(len(self)))
        n = len(self)
        try:
            return next(
                idx[i] for i in range(n - 1, -1, -1) if self.values[i] is not None
            )
        except StopIteration:
            return None

    def truncate(
        self,
        before=None,
        after=None,
        axis: int = 0,
        copy: bool = True,
    ) -> _PySeries:
        """截断索引，保留 before 和 after 之间的值。

        :param before: 截断起始索引
        :param after: 截断结束索引
        :param axis: 轴（仅支持 0）
        :param copy: 是否复制
        """
        index = self._index if self._index is not None else list(range(len(self)))
        mask = [
            (before is None or idx >= before) and (after is None or idx <= after)
            for idx in index
        ]
        new_values = [v for v, m in zip(self.values, mask) if m]
        new_index = [i for i, m in zip(index, mask) if m]
        return Series(
            new_values, name=self.name, dtype=self._dtype_str, index=new_index
        )

    def add_prefix(self, prefix: str) -> _PySeries:
        """给索引添加前缀。

        :param prefix: 前缀字符串
        """
        index = self._index if self._index is not None else list(range(len(self)))
        new_index = [f"{prefix}{i}" for i in index]
        return Series(
            list(self.values), name=self.name, dtype=self._dtype_str, index=new_index
        )

    def add_suffix(self, suffix: str) -> _PySeries:
        """给索引添加后缀。

        :param suffix: 后缀字符串
        """
        index = self._index if self._index is not None else list(range(len(self)))
        new_index = [f"{i}{suffix}" for i in index]
        return Series(
            list(self.values), name=self.name, dtype=self._dtype_str, index=new_index
        )

    def squeeze(self):
        """压缩维度：1 元素 Series 返回标量，否则返回自身。"""
        if len(self) == 1:
            return self.values[0]
        return self

    def take(self, indices) -> _PySeries:
        """按位置索引取值。

        :param indices: 位置索引列表
        """
        vals = self.values
        result = [vals[i] if 0 <= i < len(vals) else None for i in indices]
        return Series(result, name=self.name, index=indices, dtype=self._dtype_str)

    def mad(self) -> float:
        """平均绝对偏差。"""
        vals = [v for v in self.values if v is not None]
        if not vals:
            return None
        m = sum(vals) / len(vals)
        return sum(abs(x - m) for x in vals) / len(vals)

    @property
    def loc(self):
        """标签索引访问器。"""
        return _LocIndexer(self)

    @property
    def at(self):
        """标量标签索引访问器。"""
        return _LocIndexer(self)

    @property
    def iat(self):
        """标量位置索引访问器。"""
        return _IatIndexer(self)

    def sample(
        self,
        n: int = None,
        frac: float = None,
        replace: bool = False,
        weights=None,
        random_state=None,
        axis: int = 0,
    ) -> _PySeries:
        """随机采样。

        :param n: 采样数量
        :param frac: 采样比例
        :param replace: 是否有放回
        :param weights: 权重
        :param random_state: 随机种子
        """
        # 优先调用 Rust 层 sample（不支持 weights，回退到 Python 实现）
        if weights is None:
            try:
                # 将 random_state 转为 u64 种子
                seed = None
                if random_state is not None:
                    if isinstance(random_state, int):
                        seed = random_state & 0xFFFFFFFFFFFFFFFF
                    else:
                        import hashlib

                        h = hashlib.md5(str(random_state).encode()).digest()
                        seed = int.from_bytes(h[:8], "little")
                n_int = int(n) if n is not None else None
                frac_f = float(frac) if frac is not None else None
                result_inner = self._inner.sample(n_int, frac_f, replace, seed)
                new_s = Series.__new__(Series)
                new_s._inner = result_inner
                new_s._dtype_str = "float64"
                # 索引按采样结果重新生成（Rust 层未保留原索引）
                new_s._index = list(range(result_inner.size))
                new_s._name = self.name
                return new_s
            except Exception:
                pass

        import random as _random

        values = self.values
        index = self._index if self._index is not None else list(range(len(values)))

        if frac is not None:
            n = int(len(values) * frac)
        elif n is None:
            n = 1

        if random_state is not None:
            _random.seed(random_state)

        if replace:
            indices = [_random.randint(0, len(values) - 1) for _ in range(n)]
        else:
            indices = _random.sample(range(len(values)), min(n, len(values)))

        new_values = [values[i] for i in indices]
        new_index = [index[i] for i in indices]
        return Series(
            new_values, name=self.name, dtype=self._dtype_str, index=new_index
        )

    def argsort(self, ascending: bool = True, kind: str = "quicksort") -> _PySeries:
        """返回排序后的索引位置。

        :param ascending: 是否升序
        :param kind: 排序算法
        """
        indexed = list(enumerate(self.values))
        non_none = [(i, v) for i, v in indexed if v is not None]
        none_items = [(i, v) for i, v in indexed if v is None]
        non_none.sort(key=lambda x: x[1], reverse=not ascending)
        order = [i for i, _ in non_none + none_items]
        return Series(order, name=self.name, dtype="int64")

    # ---------- v2.1.0: IO 输出 ----------

    def to_csv(
        self,
        path=None,
        index: bool = True,
        sep: str = ",",
        na_rep: str = "",
        header: bool = True,
        mode: str = "w",
        encoding: str = "utf-8",
        errors: str = "strict",
        compression: str = "infer",
        quoting=None,
        quotechar: str = '"',
        lineterminator=None,
        chunksize=None,
        date_format=None,
        doublequote: bool = True,
        escapechar=None,
        decimal: str = ".",
    ):
        """将 Series 写入 CSV 文件。

        :param path: 文件路径，若为 None 则返回字符串
        :param index: 是否写索引
        :param sep: 分隔符
        :param na_rep: 缺失值表示
        :param header: 是否写表头
        """
        # 构造 CSV 文本（使用列表推导式替代显式 for 循环）
        idx = self._index if self._index is not None else list(range(len(self)))
        col_name = self.name if self.name is not None else "0"
        header_line = (
            sep.join(["", col_name] if index else [col_name]) if header else None
        )
        # 数据行通过列表推导式生成
        data_lines = [
            (
                sep.join([str(i), na_rep if v is None else str(v)])
                if index
                else (na_rep if v is None else str(v))
            )
            for i, v in zip(idx, self.values)
        ]
        # 拼接表头与数据行
        all_lines = ([header_line] if header_line is not None else []) + data_lines
        text = "\n".join(all_lines) + "\n"

        if path is None:
            return text
        with open(path, mode, encoding=encoding, errors=errors) as f:
            f.write(text)
        return None

    def to_excel(
        self,
        excel_writer,
        sheet_name: str = "Sheet1",
        na_rep: str = "",
        float_format=None,
        columns=None,
        header=True,
        index=True,
        index_label=None,
        startrow=0,
        startcol=0,
        engine=None,
        merge_cells: bool = True,
        inf_rep: str = "inf",
        freeze_panes=None,
        storage_options=None,
    ):
        """将 Series 写入 Excel 文件。"""
        from .dataframe import DataFrame

        # 将 Series 转为单列 DataFrame 后复用 to_excel
        col_name = self.name if self.name is not None else 0
        df = DataFrame({col_name: list(self.values)}, index=self._index)
        return df.to_excel(
            excel_writer,
            sheet_name=sheet_name,
            na_rep=na_rep,
            float_format=float_format,
            columns=columns,
            header=header,
            index=index,
            index_label=index_label,
            startrow=startrow,
            startcol=startcol,
            engine=engine,
            merge_cells=merge_cells,
            inf_rep=inf_rep,
            freeze_panes=freeze_panes,
            storage_options=storage_options,
        )

    def to_json(
        self,
        path_or_buf=None,
        orient: str = "records",
        date_format: str = "epoch",
        double_precision: int = 10,
        force_ascii: bool = True,
        date_unit: str = "ms",
        default_handler=None,
        lines: bool = False,
        compression: str = "infer",
        index: bool = True,
        indent: int = 0,
    ):
        """将 Series 转换为 JSON 字符串。"""
        import json

        vals = [None if v is None else v for v in self.values]
        idx = self._index if self._index is not None else list(range(len(vals)))

        if orient == "records":
            data = [{str(i): v} for i, v in zip(idx, vals)]
        elif orient == "index":
            data = {str(i): v for i, v in zip(idx, vals)}
        elif orient == "values":
            data = vals
        elif orient == "split":
            data = {"name": self.name, "index": list(idx), "data": vals}
        else:
            data = vals

        if lines:
            if orient != "records":
                raise ValueError("'lines' is only valid with orient='records'")
            text = "\n".join(json.dumps(row, ensure_ascii=force_ascii) for row in data)
        else:
            text = json.dumps(
                data,
                ensure_ascii=force_ascii,
                default=default_handler,
                indent=indent if indent > 0 else None,
            )

        if path_or_buf is None:
            return text
        with open(path_or_buf, "w", encoding="utf-8") as f:
            f.write(text)
        return None

    def to_parquet(
        self,
        path=None,
        engine: str = "auto",
        compression: str = "snappy",
        index: bool = None,
        partition_cols=None,
        storage_options=None,
        **kwargs,
    ):
        """将 Series 写入 Parquet 文件（基于 Rust arrow/parquet crate，无需 pyarrow）。"""
        from .dataframe import DataFrame

        col_name = self.name if self.name is not None else "0"
        df = DataFrame({col_name: list(self.values)}, index=self._index)
        return df.to_parquet(
            path,
            engine=engine,
            compression=compression,
            index=index,
            partition_cols=partition_cols,
            storage_options=storage_options,
            **kwargs,
        )

    def to_string(
        self,
        buf=None,
        na_rep: str = "NaN",
        float_format=None,
        header: bool = True,
        index: bool = True,
        length: bool = False,
        dtype: bool = False,
        name: bool = False,
        max_rows=None,
        min_rows=None,
    ) -> str:
        """将 Series 格式化为字符串。"""
        vals = self.values
        idx = self._index if self._index is not None else list(range(len(vals)))
        if max_rows is not None and len(vals) > max_rows:
            half = max_rows // 2
            head_vals = vals[:half]
            tail_vals = vals[-half:]
            head_idx = idx[:half]
            tail_idx = idx[-half:]
            show_vals = head_vals + ["..."] + tail_vals
            show_idx = head_idx + ["..."] + tail_idx
        else:
            show_vals = vals
            show_idx = idx

        def fmt(v):
            if v is None:
                return na_rep
            if float_format is not None and isinstance(v, float):
                return float_format(v)
            return str(v)

        idx_width = max((len(str(i)) for i in show_idx), default=1)
        # 使用列表推导式替代显式 for 循环
        lines = []
        if header:
            name_str = self.name if self.name else "0"
            lines.append(f"{'':>{idx_width}}    {name_str}")
        lines.extend(
            f"{str(i):>{idx_width}}    {fmt(v) if v != '...' else '...'}"
            for i, v in zip(show_idx, show_vals)
        )
        text = "\n".join(lines)

        if length:
            text += f"\nLength: {len(vals)}"
        if dtype:
            text += f"\ndtype: {self._dtype_str}"
        if name and self.name is not None:
            text += f"\nName: {self.name}"

        if buf is None:
            return text
        with open(buf, "w", encoding="utf-8") as f:
            f.write(text)
        return None

    def to_markdown(
        self,
        buf=None,
        mode: str = "wt",
        index: bool = True,
        **kwargs,
    ) -> str:
        """将 Series 转换为 Markdown 表格。"""
        idx = self._index if self._index is not None else list(range(len(self)))
        idx_name = "index" if index else ""
        col_name = self.name if self.name is not None else "0"
        if index:
            lines = [
                f"| {idx_name} | {col_name} |",
                "|---|---|",
                *[
                    f"| {i} | {'NaN' if v is None else str(v)} |"
                    for i, v in zip(idx, self.values)
                ],
            ]
        else:
            lines = [
                f"| {col_name} |",
                "|---|",
                *[f"| {'NaN' if v is None else str(v)} |" for v in self.values],
            ]

        text = "\n".join(lines)
        if buf is None:
            return text
        with open(buf, mode, encoding="utf-8") as f:
            f.write(text)
        return None

    # ---------- v2.1.0: 类型推断与时间序列 ----------

    def infer_objects(self, copy: bool = True) -> _PySeries:
        """推断对象 dtype 列的类型。"""
        # 简化实现：直接返回副本
        return self.copy() if copy else self

    def convert_dtypes(
        self,
        infer_objects: bool = True,
        convert_string: bool = True,
        convert_integer: bool = True,
        convert_boolean: bool = True,
        convert_floating: bool = True,
    ) -> _PySeries:
        """将列转换为最佳可能的 dtype。"""
        # 简化实现：根据值推断
        vals = self.values
        non_none = [v for v in vals if v is not None]
        if not non_none:
            return self.copy()

        if convert_integer and all(
            isinstance(v, int) and not isinstance(v, bool) for v in non_none
        ):
            dtype = "int64"
        elif convert_boolean and all(isinstance(v, bool) for v in non_none):
            dtype = "bool"
        elif convert_floating and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_none
        ):
            dtype = "float64"
        else:
            dtype = self._dtype_str

        return Series(list(vals), name=self.name, dtype=dtype, index=self._index)

    def asfreq(self, freq, method=None, how: str = "end", normalize: bool = False):
        """将时间序列转换为指定频率。"""
        # 简化实现：直接返回副本
        return self.copy()

    def tz_localize(self, tz, ambiguous: str = "raise", nonexistent: str = "raise"):
        """本地化时区（同时本地化值和索引）。"""
        from zoneinfo import ZoneInfo

        tz_obj = ZoneInfo(tz) if isinstance(tz, str) else tz
        vals = []
        for v in self.values:
            if v is None:
                vals.append(None)
            elif hasattr(v, "replace"):
                vals.append(v.replace(tzinfo=tz_obj))
            else:
                vals.append(v)
        # 同时本地化索引
        new_index = []
        for idx in self._index:
            if hasattr(idx, "replace"):
                new_index.append(idx.replace(tzinfo=tz_obj))
            else:
                new_index.append(idx)
        result = Series(vals, name=self.name, dtype=self._dtype_str, index=new_index)
        result._freq = self._freq
        return result

    def tz_convert(self, tz):
        """转换时区（同时转换值和索引）。"""
        from zoneinfo import ZoneInfo

        tz_obj = ZoneInfo(tz) if isinstance(tz, str) else tz
        vals = []
        for v in self.values:
            if v is None:
                vals.append(None)
            elif hasattr(v, "astimezone"):
                vals.append(v.astimezone(tz_obj))
            elif hasattr(v, "replace"):
                vals.append(v.replace(tzinfo=tz_obj))
            else:
                vals.append(v)
        # 同时转换索引
        new_index = []
        for idx in self._index:
            if hasattr(idx, "astimezone"):
                new_index.append(idx.astimezone(tz_obj))
            elif hasattr(idx, "replace"):
                new_index.append(idx.replace(tzinfo=tz_obj))
            else:
                new_index.append(idx)
        result = Series(vals, name=self.name, dtype=self._dtype_str, index=new_index)
        result._freq = self._freq
        return result

    # ---------- v2.1.0: 属性扩展 ----------

    @property
    def array(self):
        """返回底层值的 rsnumpy 数组（显示格式对齐 pandas NumpyExtensionArray）。"""
        import rsnumpy as rnp

        return _ExtensionArray(rnp.array(self.values), self._dtype_str)

    @property
    def flags(self):
        """返回标志字典。"""
        return {"allows_duplicate_labels": True}

    @property
    def sparse(self):
        """稀疏访问器（仅对稀疏 dtype 有效，这里返回 NotImplementedError）。"""
        raise NotImplementedError("Series.sparse only available for SparseDtype")

    def _format_repr(self) -> str:
        # 辅助函数：格式化 datetime 值
        from datetime import datetime

        def _fmt_val(v):
            """格式化单个值，datetime 去除 00:00:00，用空格替代 'T'。"""
            if isinstance(v, datetime):
                # 如果时间部分为 0，只显示日期
                if (
                    v.hour == 0
                    and v.minute == 0
                    and v.second == 0
                    and v.microsecond == 0
                ):
                    return v.strftime("%Y-%m-%d")
                # 用空格替代 ISO 默认的 'T'，与 pandas 显示一致
                return v.isoformat().replace("T", " ", 1)
            if v is None:
                return "NaN"
            if isinstance(v, float) and v != v:
                return "NaN"
            return str(v)

        def _format_floats_precise(values, precision=6):
            """按 pandas 规则格式化浮点列表：每列统一精度。

            规则：先将所有有效值格式化为 precision 位小数，
            然后确定所需最大精度（去掉末尾零后），统一应用。
            若所有值均为整数，显示 1 位小数。
            不添加前导空格，对齐在 _format_repr 中处理。
            """
            # 分离有效浮点数和无效值
            valid_floats = []
            valid_indices = []
            for i, v in enumerate(values):
                if v is None:
                    continue
                if isinstance(v, float) and v != v:  # NaN
                    continue
                try:
                    fv = float(v)
                    valid_floats.append(fv)
                    valid_indices.append(i)
                except (ValueError, TypeError):
                    pass

            if not valid_floats:
                return [
                    (
                        "NaN"
                        if (v is None or (isinstance(v, float) and v != v))
                        else str(v)
                    )
                    for v in values
                ]

            # 格式化为 6 位小数，找出最大精度
            formatted_all = [f"{fv:.{precision}f}" for fv in valid_floats]
            # 对每个值去掉末尾零，确定所需精度
            decimal_counts = []
            for fmt in formatted_all:
                if "." in fmt:
                    stripped = fmt.rstrip("0")
                    if stripped.endswith("."):
                        decimal_counts.append(1)
                    else:
                        decimal_counts.append(
                            len(stripped.split(".")[1]) if "." in stripped else 0
                        )
                else:
                    decimal_counts.append(1)

            # 所需最大精度（至少 1 位）
            max_decimals = max(max(decimal_counts), 1)

            # 重新用统一精度格式化
            precise_format = f".{max_decimals}f"
            result = [f"{fv:{precise_format}}" for fv in valid_floats]

            # 构建完整结果
            full_result = list(values)
            for i, idx in enumerate(valid_indices):
                full_result[idx] = result[i]

            # 替换无效值为 "NaN"
            for i, v in enumerate(full_result):
                if v is None:
                    full_result[i] = "NaN"
                elif isinstance(v, float) and v != v:
                    full_result[i] = "NaN"
                elif not isinstance(v, str):
                    try:
                        fv = float(v)
                        if fv != fv:
                            full_result[i] = "NaN"
                    except (ValueError, TypeError):
                        full_result[i] = str(v)

            return full_result

        # 字符串化每个值，float64 类型显示合适精度（对齐 pandas 行为）
        dtype_str = self._dtype_str
        if dtype_str in ("float64", "float32", "float16", "float"):
            strs = _format_floats_precise(list(self.values))
            # 处理极大/极小值使用科学计数法
            for i, v in enumerate(self.values):
                if v is None:
                    continue
                if isinstance(v, float) and v != v:
                    continue
                try:
                    fv = float(v)
                    if abs(fv) >= 1e15 or (fv != 0 and abs(fv) < 0.001):
                        strs[i] = f"{fv:.6g}"
                except (ValueError, TypeError):
                    pass
        elif dtype_str == "int64":
            strs = [str(int(v)) if v is not None else "NaN" for v in self.values]
        elif dtype_str == "object":
            # object dtype: 混合类型，检测是否有数值类型
            raw_values = list(self.values)
            # 检查是否有数值（int/float）值
            has_numeric = any(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and v is not None
                for v in raw_values
            )
            if has_numeric:
                # 有数值: 对数值使用浮点格式化，其他保持原样
                strs = []
                for v in raw_values:
                    if v is None or (isinstance(v, float) and v != v):
                        strs.append("NaN")
                    elif isinstance(v, (int, float)) and not isinstance(v, bool):
                        # 数值: 使用浮点格式
                        strs.append(f"{float(v):.1f}")
                    else:
                        strs.append(_fmt_val(v))
            else:
                strs = [_fmt_val(v) for v in raw_values]
        else:
            strs = [_fmt_val(v) for v in self.values]

        n = len(strs)

        # 准备索引字符串（处理 MultiIndex tuple 格式和 datetime）
        idx_strs = (
            [
                (
                    "  ".join(_fmt_val(v) for v in i)
                    if isinstance(i, tuple)
                    else _fmt_val(i)
                )
                for i in self._index
            ]
            if self._index is not None
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
        idx_width = max((len(s) for s in idx_strs), default=1)

        # 值列宽度：基于整数部分的最大位数对齐（对齐 pandas 行为）
        non_ellipsis_strs = [s for s in strs if s != "..."]
        if non_ellipsis_strs:
            max_str_len = max(len(s) for s in non_ellipsis_strs)
            max_int_digits = 0
            max_dec_digits = 0
            for s in non_ellipsis_strs:
                # 处理负数和科学计数法
                clean_s = s.lstrip("-")
                if "." in clean_s:
                    int_part, dec_part = clean_s.split(".", 1)
                    max_int_digits = max(max_int_digits, len(int_part))
                    max_dec_digits = max(max_dec_digits, len(dec_part))
                else:
                    max_int_digits = max(max_int_digits, len(clean_s))
            # 对齐宽度：max(最长字符串长度, 整数部分+小数部分+1(负号对齐)+1(小数点))
            val_width = max(max_str_len, max_int_digits + max_dec_digits + 1)
            if max_dec_digits > 0:
                val_width = max(
                    val_width, max_int_digits + max_dec_digits + 2
                )  # +2: 小数点 + 前导空格
        else:
            val_width = 1

        # 右对齐值字符串
        strs_aligned = [s.rjust(val_width) if s != "..." else s for s in strs]

        # 使用列表推导式构建 lines（pandas 使用 3 个空格分隔索引和值）
        lines = [
            " .." if s == "..." else f"{idx_s:>{idx_width}}   {s}"
            for s, idx_s in zip(strs_aligned, idx_strs)
        ]

        body = "\n".join(lines)

        # 确定显示用的 dtype 名称
        display_dtype = dtype_str
        # 如果是 category 类型，添加 Categories 信息
        categories_info = ""
        if display_dtype == "category" or dtype_str == "category":
            try:
                cat_accessor = self.cat
                cats = cat_accessor.categories
                cat_str = ", ".join(repr(c) for c in cats)
                categories_info = f"\nCategories ({len(cats)}, str): [{cat_str}]"
            except (AttributeError, Exception):
                pass

        # 频率信息（对齐 pandas: Freq 放在 Name 之前，同一行）
        freq_str = ""
        if self._freq is not None:
            freq_str = f"Freq: {self._freq}, "

        # 对齐 pandas: name 为 None 时不显示 Name 行
        if self.name is not None and self.name != "":
            return f"{body}\n{freq_str}Name: {self.name}, dtype: {display_dtype}{categories_info}"
        else:
            return f"{body}\n{freq_str}dtype: {display_dtype}{categories_info}"

    # ----------------------------------------------------------------
    # 扩展功能（pandas 之外的增强）
    # ----------------------------------------------------------------

    def infer_type(self) -> str:
        """智能推断 Series 的数据类型。

        Returns:
            str: 推断的类型名（'bool' / 'int64' / 'float64' / 'datetime' / 'string' / 'object'）

        Examples:
            >>> Series([1, 2, 3]).infer_type()
            'int64'
            >>> Series([1.0, 2.0]).infer_type()
            'float64'
            >>> Series(['a', 'b']).infer_type()
            'string'
        """
        values = list(self.values)
        non_null = [v for v in values if v is not None]
        if not non_null:
            return "object"

        # 利用 all() + 生成器表达式按优先级判断类型
        if all(isinstance(v, bool) for v in non_null):
            return "bool"
        if all(isinstance(v, int) and not isinstance(v, bool) for v in non_null):
            return "int64"
        if all(isinstance(v, (int, float)) for v in non_null):
            return "float64"
        # 尝试日期解析
        import datetime

        def _is_date(v) -> bool:
            """判断 v 是否为日期类型。"""
            return isinstance(v, (datetime.datetime, datetime.date))

        if all(_is_date(v) for v in non_null):
            return "datetime"
        return "string"

    def describe_full(self) -> Dict[str, Any]:
        """扩展描述统计（包含偏度/峰度/四分位间距/变异系数）。

        Returns:
            Dict[str, Any]: 统计指标字典

        Examples:
            >>> s = Series([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> stats = s.describe_full()
            >>> stats['mean']
            3.0
        """
        values = [v for v in self.values if v is not None]
        if not values:
            return {}
        if not all(isinstance(v, (int, float)) for v in values):
            raise TypeError("describe_full 仅支持数值列")

        n = len(values)
        mean_v = sum(values) / n
        # 样本方差（ddof=1）
        var_v = sum((v - mean_v) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
        std_v = var_v**0.5

        sorted_vals = sorted(values)
        q1 = sorted_vals[n // 4]
        q3 = sorted_vals[3 * n // 4]
        median = sorted_vals[n // 2]
        iqr = q3 - q1

        # 偏度（Skewness）
        if std_v > 0:
            skew = sum((v - mean_v) ** 3 for v in values) / (n * std_v**3)
        else:
            skew = 0.0

        # 峰度（Kurtosis）- 超额峰度
        if std_v > 0:
            kurt = sum((v - mean_v) ** 4 for v in values) / (n * std_v**4) - 3
        else:
            kurt = 0.0

        # 变异系数
        cv = std_v / mean_v if mean_v != 0 else None

        return {
            "count": n,
            "mean": mean_v,
            "std": std_v,
            "var": var_v,
            "min": min(values),
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": max(values),
            "iqr": iqr,
            "skew": skew,
            "kurt": kurt,
            "cv": cv,
        }

    def moving_average(self, window: int) -> "Series":
        """简单移动平均（SMA）。

        Parameters:
            window: 窗口大小

        Returns:
            Series: 移动平均结果，前 window-1 个位置为 None

        Examples:
            >>> s = Series([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> ma = s.moving_average(3)
            >>> ma.values[0] is None
            True
            >>> ma.values[2]
            2.0
        """
        if window < 1:
            raise ValueError("window 必须 >= 1")
        values = list(self.values)
        n = len(values)

        # 使用列表推导式计算每个位置的窗口均值
        def _ma_at(i: int) -> Optional[float]:
            """计算第 i 位的移动平均值。"""
            if i < window - 1:
                return None
            win = values[i - window + 1 : i + 1]  # noqa
            non_null = [v for v in win if v is not None]
            if not non_null:
                return None
            return sum(non_null) / len(non_null)

        result = [_ma_at(i) for i in range(n)]
        return Series(result, name=self.name, index=self._index)

    def detect_encoding(self) -> str:
        """检测 Series 中字符串的编码。

        通过采样非空字符串值，判断是否包含 ASCII / UTF-8 / GBK 等编码特征。
        底层使用 chardet 库（若安装），否则使用启发式检测。

        Returns:
            str: 检测到的编码名（'ascii' / 'utf-8' / 'gbk' / 'unknown'）

        Examples:
            >>> Series(['hello', 'world']).detect_encoding()
            'ascii'
            >>> Series(['你好', '世界']).detect_encoding()
            'utf-8'
        """
        # 收集非空字符串值
        str_vals = [
            v
            for v in self.values
            if v is not None and isinstance(v, str) and len(v) > 0
        ]
        if not str_vals:
            return "unknown"

        # 优先使用 chardet（若安装）
        try:
            import chardet

            sample = " ".join(str_vals[:100]).encode("utf-8", errors="replace")
            result = chardet.detect(sample)
            encoding = result.get("encoding", None)
            if encoding:
                return encoding.lower()
        except ImportError:
            pass

        # 启发式检测：检查是否纯 ASCII
        all_ascii = all(v.isascii() for v in str_vals)
        if all_ascii:
            return "ascii"

        # 尝试 UTF-8 解码
        try:
            for v in str_vals:
                v.encode("utf-8")
            return "utf-8"
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

        # 尝试 GBK 解码
        try:
            for v in str_vals:
                v.encode("gbk")
            return "gbk"
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

        return "unknown"


# ============================================================================
# 重新导出辅助类（向后兼容）
# 这些类已迁移到子模块，但 series.py 仍重新导出以保持向后兼容
# ============================================================================
from ._internal._series_helpers import (  # noqa: F401
    _AlignmentResult,
    _DtypeScalar,
    _ExtensionArray,
    _PySeries_filter,
    _dtype_to_str,
    _format_timedelta,
    _infer_dtype,
    _is_missing,
    _is_range_index,
    _to_python_list,
    _to_python_list_and_index,
)
from .accessors.cat import CatAccessor  # noqa: F401
from .accessors.datetime import DatetimeAccessor  # noqa: F401
from .accessors.string import StringAccessor  # noqa: F401
from .groupby.series_groupby import SeriesGroupBy  # noqa: F401
from .indexing.series_indexers import (
    _IatIndexer,
    _ILocIndexer,
    _LocIndexer,
)  # noqa: F401
from .window.ewm import EWM  # noqa: F401
from .window.expanding import Expanding  # noqa: F401
from .window.resampler import Resampler  # noqa: F401
from .window.rolling import Rolling  # noqa: F401
