"""datetime 工具函数。"""

from __future__ import annotations
from datetime import datetime, date, timedelta
from typing import Optional, Union

from .series import Series


# ---------------------------------------------------------------------------
# 频率 -> timedelta 映射
# ---------------------------------------------------------------------------

_FREQ_MAP = {
    "D": timedelta(days=1),
    "W": timedelta(weeks=1),
    "H": timedelta(hours=1),
    "h": timedelta(hours=1),
    "M": timedelta(days=30),
    "Y": timedelta(days=365),
    "S": timedelta(seconds=1),
    "T": timedelta(minutes=1),
    "min": timedelta(minutes=1),
}


def _freq_to_timedelta(freq: str) -> timedelta:
    if freq in _FREQ_MAP:
        return _FREQ_MAP[freq]
    raise ValueError(f"unsupported freq: {freq!r}")


# ---------------------------------------------------------------------------
# 日期解析
# ---------------------------------------------------------------------------

def _parse_iso(s: str) -> datetime:
    """解析常见日期格式字符串。"""
    if not isinstance(s, str):
        raise TypeError(f"expected str, got {type(s).__name__}")
    s = s.strip()
    if not s:
        raise ValueError("empty date string")
    # 尝试常见格式 (顺序: 从精确到宽泛)
    fmts = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%H:%M:%S",
        "%H:%M",
    ]
    last_err: Optional[Exception] = None
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError as e:
            last_err = e
            continue
    raise ValueError(f"cannot parse date string: {s!r}") from last_err


def _to_iso(v) -> Optional[str]:
    """将 datetime/date/None 转换为 ISO 字符串。"""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day).isoformat()
    return None


# ---------------------------------------------------------------------------
# DatetimeSeries: 包装 Series，提供 datetime 语义
# ---------------------------------------------------------------------------

class DatetimeSeries:
    """datetime 类型的 Series 包装类。

    内部使用 ISO 字符串存储以兼容 Rust 端 (Rust 端无 datetime 类型)。
    暴露的 values 为 datetime 对象数组。

    Examples:
        >>> s = to_datetime(['2024-01-01', '2024-01-02'])
        >>> s.values
        [datetime.datetime(2024, 1, 1, 0, 0), datetime.datetime(2024, 1, 2, 0, 0)]
        >>> s.dt.year
        [2024, 2024]
    """

    def __init__(self, values: list, name: Optional[str] = None, index: Optional[list] = None):
        # 内部存储: ISO 字符串列表
        iso_values = [_to_iso(v) for v in values]
        self._inner: Series = Series(iso_values, name=name, index=index, dtype="object")
        self._raw_values: list = list(values)  # 缓存 datetime 对象
        self.name: Optional[str] = name
        self._index: Optional[list] = index

    @property
    def values(self) -> list:
        """返回 datetime 对象列表。"""
        # 优先使用缓存；如果 inner 被修改，重新解析
        if len(self._raw_values) == len(self._inner.values):
            return list(self._raw_values)
        return [_parse_iso(s) if s is not None else None for s in self._inner.values]

    @property
    def dtype(self) -> str:
        return "datetime64[ns]"

    @property
    def shape(self) -> tuple:
        return self._inner.shape

    @property
    def size(self) -> int:
        return self._inner.size

    @property
    def empty(self) -> bool:
        return self._inner.empty

    def __len__(self) -> int:
        return len(self._inner)

    def __repr__(self) -> str:
        return f"DatetimeSeries({self.values!r}, name={self.name!r})"

    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, key):
        return self.values[key]

    def head(self, n: int = 5) -> "DatetimeSeries":
        return DatetimeSeries(self.values[:n], name=self.name, index=self._index[:n] if self._index else None)

    def tail(self, n: int = 5) -> "DatetimeSeries":
        return DatetimeSeries(self.values[-n:] if n > 0 else [], name=self.name, index=self._index[-n:] if self._index else None)

    @property
    def dt(self):
        """datetime 访问器。"""
        return DatetimeAccessor(self)


class DatetimeAccessor:
    """Series.dt 访问器。"""

    def __init__(self, series: DatetimeSeries):
        self._s = series

    def _wrap_series(self, values: list, name: Optional[str] = None) -> Series:
        return Series(values, name=name or self._s.name, index=self._s._index)

    @property
    def year(self) -> Series:
        return self._wrap_series([v.year if v is not None else None for v in self._s.values])

    @property
    def month(self) -> Series:
        return self._wrap_series([v.month if v is not None else None for v in self._s.values])

    @property
    def day(self) -> Series:
        return self._wrap_series([v.day if v is not None else None for v in self._s.values])

    @property
    def hour(self) -> Series:
        return self._wrap_series([v.hour if v is not None else None for v in self._s.values])

    @property
    def minute(self) -> Series:
        return self._wrap_series([v.minute if v is not None else None for v in self._s.values])

    @property
    def second(self) -> Series:
        return self._wrap_series([v.second if v is not None else None for v in self._s.values])

    @property
    def weekday(self) -> Series:
        """返回星期几 (0=周一, 6=周日)，与 pandas 一致。"""
        return self._wrap_series([v.weekday() if v is not None else None for v in self._s.values])

    @property
    def dayofweek(self) -> Series:
        return self.weekday

    @property
    def day_name(self) -> Series:
        """返回星期几的名称。"""
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return self._wrap_series([names[v.weekday()] if v is not None else None for v in self._s.values])

    @property
    def month_name(self) -> Series:
        names = ["", "January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"]
        return self._wrap_series([names[v.month] if v is not None else None for v in self._s.values])

    @property
    def date(self) -> Series:
        """返回 date 部分 (Python date 对象数组)。"""
        return self._wrap_series([v.date() if v is not None else None for v in self._s.values])

    def strftime(self, fmt: str) -> Series:
        """格式化为字符串。"""
        return self._wrap_series([v.strftime(fmt) if v is not None else None for v in self._s.values])


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def to_datetime(
    arg,
    format: Optional[str] = None,
    errors: str = "raise",
) -> Union[DatetimeSeries, datetime]:
    """将输入转换为 datetime。

    :param arg: list[str]/str/Series/int/float/datetime
    :param format: 显式日期格式 (如 '%Y-%m-%d')
    :param errors: 'raise' / 'coerce' / 'ignore'
    :return: DatetimeSeries (如果 arg 是序列) 或单个 datetime
    """
    # 单值情况
    if isinstance(arg, (str, int, float, datetime, date)) and not (
        isinstance(arg, str) and "," in arg
    ):
        # 标量输入 -> 返回单个 datetime
        try:
            if isinstance(arg, datetime):
                return arg
            if isinstance(arg, date):
                return datetime(arg.year, arg.month, arg.day)
            if isinstance(arg, (int, float)):
                return datetime.fromtimestamp(float(arg))
            if isinstance(arg, str):
                if format:
                    return datetime.strptime(arg, format)
                return _parse_iso(arg)
        except (ValueError, TypeError):
            if errors == "coerce":
                return None
            if errors == "ignore":
                return arg
            raise

    # 序列输入 -> DatetimeSeries
    if isinstance(arg, Series):
        raw_values = arg.values
    elif isinstance(arg, (list, tuple)):
        raw_values = list(arg)
    elif hasattr(arg, '__iter__') and not isinstance(arg, str):
        raw_values = list(arg)
    else:
        raw_values = [arg]

    out: list = []
    for v in raw_values:
        if v is None:
            out.append(None)
            continue
        try:
            if isinstance(v, datetime):
                out.append(v)
            elif isinstance(v, date):
                out.append(datetime(v.year, v.month, v.day))
            elif isinstance(v, (int, float)):
                out.append(datetime.fromtimestamp(float(v)))
            elif isinstance(v, str):
                if format:
                    out.append(datetime.strptime(v, format))
                else:
                    out.append(_parse_iso(v))
            else:
                if errors == "coerce":
                    out.append(None)
                elif errors == "ignore":
                    out.append(v)
                else:
                    raise ValueError(f"cannot convert {v!r} to datetime")
        except (ValueError, TypeError, OverflowError):
            if errors == "coerce":
                out.append(None)
            elif errors == "ignore":
                out.append(v)
            else:
                raise

    return DatetimeSeries(out, name=None)


def date_range(
    start: Union[str, datetime],
    end: Optional[Union[str, datetime]] = None,
    periods: Optional[int] = None,
    freq: str = "D",
) -> DatetimeSeries:
    """生成日期范围。

    :param start: 起始日期 (str 或 datetime)
    :param end: 结束日期 (str 或 datetime, 与 periods 二选一)
    :param periods: 周期数 (与 end 二选一)
    :param freq: 频率 ('D'日, 'W'周, 'M'月, 'Y'年, 'H'时)
    :return: DatetimeSeries
    """
    # 解析 start
    if isinstance(start, str):
        start_dt = _parse_iso(start)
    elif isinstance(start, datetime):
        start_dt = start
    elif isinstance(start, date):
        start_dt = datetime(start.year, start.month, start.day)
    else:
        raise TypeError(f"start must be str or datetime, got {type(start).__name__}")

    # 解析 end
    end_dt: Optional[datetime] = None
    if end is not None:
        if isinstance(end, str):
            end_dt = _parse_iso(end)
        elif isinstance(end, datetime):
            end_dt = end
        elif isinstance(end, date):
            end_dt = datetime(end.year, end.month, end.day)
        else:
            raise TypeError(f"end must be str or datetime, got {type(end).__name__}")

    step = _freq_to_timedelta(freq)

    if periods is not None:
        if periods < 0:
            raise ValueError("periods must be non-negative")
        n = periods
    else:
        if end_dt is None:
            raise ValueError("end or periods must be specified")
        if end_dt < start_dt:
            raise ValueError("end must be >= start")
        n = int((end_dt - start_dt) / step) + 1

    out = [start_dt + step * i for i in range(n)]
    return DatetimeSeries(out, name=None)


def to_timedelta(arg, unit=None):
    """将输入转换为 timedelta。"""
    from datetime import timedelta as td
    if isinstance(arg, td):
        return arg
    if isinstance(arg, (int, float)):
        if unit is None:
            return td(seconds=float(arg))
        unit = unit.lower()
        if unit in ("s", "sec", "seconds"):
            return td(seconds=float(arg))
        elif unit in ("ms", "milli", "milliseconds"):
            return td(milliseconds=float(arg))
        elif unit in ("us", "micro", "microseconds"):
            return td(microseconds=float(arg))
        elif unit in ("ns", "nano", "nanoseconds"):
            return td(microseconds=float(arg) / 1000)
        elif unit in ("m", "min", "minutes"):
            return td(minutes=float(arg))
        elif unit in ("h", "hour", "hours"):
            return td(hours=float(arg))
        elif unit in ("d", "day", "days"):
            return td(days=float(arg))
        else:
            raise ValueError(f"unsupported unit: {unit}")
    if isinstance(arg, str):
        return td(seconds=float(arg))
    if isinstance(arg, (list, tuple)):
        return [to_timedelta(x, unit) for x in arg]
    raise TypeError(f"cannot convert {type(arg).__name__} to timedelta")


def timedelta_range(start=None, end=None, periods=None, freq="D"):
    """生成 timedelta 范围。"""
    from datetime import timedelta as td
    if start is None:
        start = td(0)
    elif isinstance(start, (int, float)):
        start = to_timedelta(start)
    elif isinstance(start, str):
        start = to_timedelta(start)
    step = _freq_to_timedelta(freq)
    if periods is not None:
        n = periods
    else:
        if end is None:
            raise ValueError("end or periods must be specified")
        if isinstance(end, (int, float)):
            end = to_timedelta(end)
        elif isinstance(end, str):
            end = to_timedelta(end)
        n = int((end - start) / step) + 1
    out = [start + step * i for i in range(n)]
    return Series(out, name=None)


def period_range(start=None, periods=None, freq="M"):
    """生成周期范围。"""
    if start is None:
        start = datetime.now()
    elif isinstance(start, str):
        start = _parse_iso(start)
    elif isinstance(start, date):
        start = datetime(start.year, start.month, start.day)
    if periods is None:
        periods = 12
    out = []
    for i in range(periods):
        if freq == "M":
            year, month = start.year, start.month + i
            while month > 12:
                year += 1
                month -= 12
            out.append(datetime(year, month, 1))
        elif freq == "Y":
            out.append(datetime(start.year + i, start.month, start.day))
        elif freq == "Q":
            year, month = start.year, start.month + i * 3
            while month > 12:
                year += 1
                month -= 12
            out.append(datetime(year, month, 1))
        else:
            step = _freq_to_timedelta(freq)
            out.append(start + step * i)
    return DatetimeSeries(out, name=None)


def bdate_range(start=None, end=None, periods=None):
    """生成工作日日期范围。"""
    if start is None:
        start = datetime.now()
    elif isinstance(start, str):
        start = _parse_iso(start)
    elif isinstance(start, date):
        start = datetime(start.year, start.month, start.day)
    end_dt = None
    if end is not None:
        if isinstance(end, str):
            end_dt = _parse_iso(end)
        elif isinstance(end, datetime):
            end_dt = end
        elif isinstance(end, date):
            end_dt = datetime(end.year, end.month, end.day)
    if periods is None:
        if end_dt is None:
            raise ValueError("end or periods must be specified")
        periods = ((end_dt - start).days // 7) * 5 + 3
    out = []
    current = start
    while len(out) < periods:
        if current.weekday() < 5:
            out.append(current)
        current += timedelta(days=1)
        if end_dt is not None and current > end_dt:
            break
    return DatetimeSeries(out, name=None)


def infer_freq(index):
    """推断频率 (简化版)。"""
    if not index:
        return None
    if isinstance(index[0], datetime):
        deltas = []
        for i in range(1, len(index)):
            delta = (index[i] - index[i-1]).days
            if delta > 0:
                deltas.append(delta)
        if not deltas:
            return None
        if all(d == 1 for d in deltas):
            return "D"
        if all(d == 7 for d in deltas):
            return "W"
        if all(d in (28, 29, 30, 31) for d in deltas):
            return "M"
        if all(d in (365, 366) for d in deltas):
            return "Y"
    return None
