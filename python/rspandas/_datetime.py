"""datetime 工具函数。"""

from __future__ import annotations

import calendar
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional, Union

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
    "s": timedelta(seconds=1),
    "T": timedelta(minutes=1),
    "min": timedelta(minutes=1),
    "m": timedelta(minutes=1),
    "ms": timedelta(milliseconds=1),
    "us": timedelta(microseconds=1),
    "ns": timedelta(microseconds=1),  # 简化：ns 用 microseconds 近似
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
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%Y%m%d%H%M%S",
        "%Y%m%d%H%M%S.%f",
        "%Y%m%d",
        "%Y-%m",
        "%Y/%m",
        "%Y",
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

    def __init__(
        self,
        values: list,
        name: Optional[str] = None,
        index: Optional[list] = None,
        freq: Optional[str] = None,
        tz: Any = None,
    ):
        # 内部存储: ISO 字符串列表
        iso_values = [_to_iso(v) for v in values]
        self._inner: Series = Series(iso_values, name=name, index=index, dtype="object")
        self._raw_values: list = list(values)  # 缓存 datetime 对象
        self.name: Optional[str] = name
        self._index: Optional[list] = index
        self._freq: Optional[str] = freq
        self._tz = tz

    @property
    def freq(self) -> Optional[str]:
        return self._freq

    @property
    def tz(self):
        """返回 DatetimeSeries 绑定的时区对象。None 表示 naive 时间。"""
        if self._tz is not None:
            return self._tz
        vals = self.values
        if vals and isinstance(vals[0], datetime) and vals[0].tzinfo is not None:
            return vals[0].tzinfo
        return None

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
    def array(self):
        """返回底层 rsnumpy ndarray（与 pandas Index.array 行为一致）。

        由于 rsnumpy 底层仅支持 f64，此处返回 ISO 字符串数组，
        调用方如需 datetime 对象应使用 .values 属性。
        """
        import rsnumpy as rnp

        return rnp.array(self._inner.values)

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
        # 对齐 pandas DatetimeIndex 格式
        date_strs = []
        for v in self.values:
            if isinstance(v, datetime):
                if (
                    v.hour == 0
                    and v.minute == 0
                    and v.second == 0
                    and v.microsecond == 0
                ):
                    date_strs.append(repr(v.strftime("%Y-%m-%d")))
                else:
                    date_strs.append(repr(str(v)))
            else:
                date_strs.append(repr(v))
        # 多行显示（每行最多 4 个值，对齐 pandas 行为）
        items_per_line = 4
        name = f", name='{self.name}'" if self.name else ""
        freq_str = f", freq='{self._freq}'" if self._freq else ""
        if len(date_strs) <= items_per_line:
            joined = ", ".join(date_strs)
            return f"DatetimeIndex([{joined}], dtype='datetime64[us]'{freq_str}{name})"
        # 多行模式
        lines = []
        indent = " " * 15  # 对齐 pandas 的缩进
        n = len(date_strs)
        for i in range(0, n, items_per_line):
            chunk = date_strs[i : i + items_per_line]
            joined = ", ".join(chunk)
            if i == 0:
                line = "DatetimeIndex([" + joined + ","
            elif i + items_per_line >= n:
                # 最后一行：直接闭合括号
                line = indent + joined + "],"
            else:
                line = indent + joined + ","
            lines.append(line)
        # 最后一行：dtype/freq/name 元数据
        close_indent = " " * 14
        lines.append(close_indent + "dtype='datetime64[us]'" + freq_str + name + ")")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, key):
        return self.values[key]

    def __add__(self, other):
        """支持 DatetimeSeries + offset/datetime/timedelta。"""
        if hasattr(other, "__radd__") and not isinstance(
            other, (datetime, timedelta, int, float)
        ):
            # offset 类型：对每个 datetime 元素应用 offset
            new_values = [
                other.__radd__(v) if v is not None else None for v in self.values
            ]
        elif isinstance(other, (datetime, timedelta, int, float)):
            new_values = [v + other if v is not None else None for v in self.values]
        else:
            return NotImplemented
        return DatetimeSeries(new_values, name=self.name, index=self._index)

    def __radd__(self, other):
        """支持 other + DatetimeSeries。"""
        return self.__add__(other)

    def head(self, n: int = 5) -> "DatetimeSeries":
        return DatetimeSeries(
            self.values[:n],
            name=self.name,
            index=self._index[:n] if self._index else None,
            freq=self._freq,
            tz=self._tz,
        )

    def tail(self, n: int = 5) -> "DatetimeSeries":
        return DatetimeSeries(
            self.values[-n:] if n > 0 else [],
            name=self.name,
            index=self._index[-n:] if self._index else None,
            tz=self._tz,
        )

    def to_numpy(self, dtype=None, copy: bool = True, na_value: Any = None):
        """转换为 rsnumpy array（与 pandas DatetimeIndex.to_numpy 行为对齐）。

        - `dtype=object` 或 `dtype='object'`: 返回 datetime 对象数组 (object dtype)
        - `dtype=datetime64[ns]` / `dtype='datetime64[ns]'`: 返回 int64 纳秒纪元数组
        - 默认（dtype=None）: 同 'datetime64[ns]' 语义，返回纳秒纪元 f64/int64 数组
        """
        import rsnumpy as rnp

        vals = self.values
        # 处理 na_value
        if na_value is not None:
            vals = [na_value if v is None else v for v in vals]
        if dtype is None:
            dtype_str = "datetime64[ns]"
        elif isinstance(dtype, str):
            dtype_str = dtype.lower().replace(" ", "")
        else:
            dtype_str = str(dtype).lower()
        if dtype_str in ("object", "o"):
            # rsnumpy 不直接支持对象数组，退回 list 包装后 rnp.array(object dtype)
            return rnp.array(list(vals))
        # datetime64[ns] (或 datetime64[us]/[ms] 简化为 ns)
        if dtype_str.startswith("datetime64"):
            # 转换为纳秒纪元 (int)
            epoch = datetime(1970, 1, 1, tzinfo=vals[0].tzinfo if vals and isinstance(vals[0], datetime) and vals[0].tzinfo is not None else None)
            nano_vals: list = []
            for v in vals:
                if v is None:
                    nano_vals.append(float("nan"))
                elif isinstance(v, datetime):
                    # 如果 aware 且要求转换结果带 UTC 偏移，这里保持原样做差
                    delta = v - epoch
                    ns = (
                        delta.days * 86_400_000_000_000
                        + delta.seconds * 1_000_000_000
                        + delta.microseconds * 1000
                    )
                    nano_vals.append(ns)
                else:
                    nano_vals.append(float("nan"))
            return rnp.array(nano_vals)
        # 其他 dtype 直接走 rnp.array
        return rnp.array(list(vals), dtype=dtype)

    def copy(self) -> "DatetimeSeries":
        """返回自身副本。"""
        return DatetimeSeries(
            list(self.values),
            name=self.name,
            index=list(self._index) if self._index else None,
            freq=self._freq,
            tz=self._tz,
        )

    @property
    def dt(self):
        """datetime 访问器。"""
        return DatetimeAccessor(self)


class DatetimeAccessor:
    """Series.dt 访问器。"""

    def __init__(self, series: DatetimeSeries):
        self._s = series

    def _wrap_series(self, values: list, name: Optional[str] = None) -> Series:
        # 将 datetime/date/time 对象转换为 ISO 字符串
        converted = []
        for v in values:
            if v is None:
                converted.append(None)
            elif isinstance(v, (datetime, date, time)):
                converted.append(v.isoformat() if hasattr(v, "isoformat") else str(v))
            elif isinstance(v, timedelta):
                converted.append(v.total_seconds())
            else:
                converted.append(v)
        return Series(converted, name=name or self._s.name, index=self._s._index)

    @property
    def year(self) -> Series:
        return self._wrap_series(
            [v.year if v is not None else None for v in self._s.values]
        )

    @property
    def month(self) -> Series:
        return self._wrap_series(
            [v.month if v is not None else None for v in self._s.values]
        )

    @property
    def day(self) -> Series:
        return self._wrap_series(
            [v.day if v is not None else None for v in self._s.values]
        )

    @property
    def hour(self) -> Series:
        return self._wrap_series(
            [v.hour if v is not None else None for v in self._s.values]
        )

    @property
    def minute(self) -> Series:
        return self._wrap_series(
            [v.minute if v is not None else None for v in self._s.values]
        )

    @property
    def second(self) -> Series:
        return self._wrap_series(
            [v.second if v is not None else None for v in self._s.values]
        )

    @property
    def weekday(self) -> Series:
        """返回星期几 (0=周一, 6=周日)，与 pandas 一致。"""
        return self._wrap_series(
            [v.weekday() if v is not None else None for v in self._s.values]
        )

    @property
    def dayofweek(self) -> Series:
        return self.weekday

    @property
    def day_name(self) -> Series:
        """返回星期几的名称。"""
        names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        return self._wrap_series(
            [names[v.weekday()] if v is not None else None for v in self._s.values]
        )

    @property
    def month_name(self) -> Series:
        names = [
            "",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        return self._wrap_series(
            [names[v.month] if v is not None else None for v in self._s.values]
        )

    @property
    def date(self) -> Series:
        """返回 date 部分 (Python date 对象数组)。"""
        return self._wrap_series(
            [v.date() if v is not None else None for v in self._s.values]
        )

    def strftime(self, fmt: str) -> Series:
        """格式化为字符串。"""
        return self._wrap_series(
            [v.strftime(fmt) if v is not None else None for v in self._s.values]
        )

    @property
    def microsecond(self) -> Series:
        return self._wrap_series(
            [v.microsecond if v is not None else None for v in self._s.values]
        )

    @property
    def dayofyear(self) -> Series:
        return self._wrap_series(
            [v.timetuple().tm_yday if v is not None else None for v in self._s.values]
        )

    @property
    def quarter(self) -> Series:
        return self._wrap_series(
            [(v.month - 1) // 3 + 1 if v is not None else None for v in self._s.values]
        )

    @property
    def is_month_start(self) -> Series:
        return self._wrap_series(
            [v.day == 1 if v is not None else None for v in self._s.values]
        )

    @property
    def is_month_end(self) -> Series:
        return self._wrap_series(
            [
                (
                    v.day == calendar.monthrange(v.year, v.month)[1]
                    if v is not None
                    else None
                )
                for v in self._s.values
            ]
        )

    @property
    def is_year_start(self) -> Series:
        return self._wrap_series(
            [
                (v.month == 1 and v.day == 1) if v is not None else None
                for v in self._s.values
            ]
        )

    @property
    def is_year_end(self) -> Series:
        return self._wrap_series(
            [
                (v.month == 12 and v.day == 31) if v is not None else None
                for v in self._s.values
            ]
        )

    @property
    def is_leap_year(self) -> Series:
        return self._wrap_series(
            [calendar.isleap(v.year) if v is not None else None for v in self._s.values]
        )

    @property
    def days_in_month(self) -> Series:
        return self._wrap_series(
            [
                calendar.monthrange(v.year, v.month)[1] if v is not None else None
                for v in self._s.values
            ]
        )

    def to_pydatetime(self) -> list:
        """返回 Python datetime 对象列表。"""
        return list(self._s.values)

    # ---------- v2.0.0: 扩展 dt 访问器 ----------

    @property
    def tz(self):
        """返回时区信息 (如果存在)。"""
        for v in self._s.values:
            if v is not None and hasattr(v, "tzinfo") and v.tzinfo is not None:
                return v.tzinfo
        return None

    def floor(self, freq: str) -> Series:
        """将 datetime 向下舍入到指定频率。

        :param freq: 频率字符串 ('D'/'H'/'M'/'S' 等)
        """
        freq = freq.strip().upper()
        if freq not in ("D", "H", "M", "T", "min", "S"):
            raise ValueError(f"unsupported freq: {freq!r}")

        def _floor(dt):
            if dt is None:
                return None
            if freq == "D":
                return dt.replace(hour=0, minute=0, second=0, microsecond=0)
            elif freq == "H":
                return dt.replace(minute=0, second=0, microsecond=0)
            elif freq == "M" or freq == "T" or freq == "min":
                return dt.replace(second=0, microsecond=0)
            elif freq == "S":
                return dt.replace(microsecond=0)
            return dt

        return self._wrap_series([_floor(v) for v in self._s.values])

    def ceil(self, freq: str) -> Series:
        """将 datetime 向上舍入到指定频率。

        :param freq: 频率字符串 ('D'/'H'/'M'/'S' 等)
        """
        freq = freq.strip().upper()
        if freq not in ("D", "H", "M", "T", "min", "S"):
            raise ValueError(f"unsupported freq: {freq!r}")

        def _ceil(dt):
            if dt is None:
                return None
            floored = None
            if freq == "D":
                floored = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            elif freq == "H":
                floored = dt.replace(minute=0, second=0, microsecond=0)
            elif freq == "M" or freq == "T" or freq == "min":
                floored = dt.replace(second=0, microsecond=0)
            elif freq == "S":
                floored = dt.replace(microsecond=0)
            if floored == dt:
                return dt
            if freq == "D":
                return floored + timedelta(days=1)
            elif freq == "H":
                return floored + timedelta(hours=1)
            elif freq == "M" or freq == "T" or freq == "min":
                return floored + timedelta(minutes=1)
            elif freq == "S":
                return floored + timedelta(seconds=1)
            return dt

        return self._wrap_series([_ceil(v) for v in self._s.values])

    def round(self, freq: str) -> Series:
        """将 datetime 四舍五入到指定频率。

        :param freq: 频率字符串 ('D'/'H'/'M'/'S' 等)
        """
        freq = freq.strip().upper()
        if freq not in ("D", "H", "M", "T", "min", "S"):
            raise ValueError(f"unsupported freq: {freq!r}")

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
            elif freq == "M" or freq == "T" or freq == "min":
                floored = dt.replace(second=0, microsecond=0)
                next_tick = floored + timedelta(minutes=1)
            elif freq == "S":
                floored = dt.replace(microsecond=0)
                next_tick = floored + timedelta(seconds=1)
            if dt - floored < next_tick - dt:
                return floored
            return next_tick

        return self._wrap_series([_round(v) for v in self._s.values])

    @property
    def time(self) -> Series:
        """返回 datetime 的时间部分 (datetime.time 对象)。"""
        return self._wrap_series(
            [v.time() if v is not None else None for v in self._s.values]
        )

    @property
    def total_seconds(self) -> Series:
        """返回 timedelta 的总秒数。"""
        return self._wrap_series(
            [
                (
                    v.total_seconds()
                    if v is not None and hasattr(v, "total_seconds")
                    else None
                )
                for v in self._s.values
            ]
        )


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
    elif hasattr(arg, "__iter__") and not isinstance(arg, str):
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


def _normalize_tz(tz: Any) -> Any:
    """解析时区字符串或对象，返回时区对象（若无则返回 None）。

    优先级:
    1) datetime.timezone / zoneinfo.ZoneInfo / pytz 等 tzinfo 对象
    2) 字符串名称：通过 zoneinfo (Python 3.9+) 或 pytz 查找，最后回退到 datetime.timezone 的 UTC/固定偏移
    """
    if tz is None or tz is False:
        return None
    # 已经是 tzinfo
    if hasattr(tz, "utcoffset"):
        return tz
    if isinstance(tz, str):
        name = tz
        # 1) zoneinfo
        try:
            from zoneinfo import ZoneInfo

            return ZoneInfo(name)
        except Exception:
            pass
        # 2) pytz
        try:
            import pytz

            return pytz.timezone(name)
        except Exception:
            pass
        # 3) 常见别名 -> UTC 偏移
        aliases = {
            "UTC": 0,
            "GMT": 0,
            "CET": 60,  # UTC+1
            "CEST": 120,
            "EET": 120,
            "EEST": 180,
            "JST": 540,
            "CST": 480,  # 中国标准时间
            "EST": -300,
            "EDT": -240,
            "PST": -480,
            "PDT": -420,
            "MST": -420,
            "MDT": -360,
        }
        if name.upper() in aliases:
            mins = aliases[name.upper()]
            return timezone(timedelta(minutes=mins), name=name)
        # 4) 形如 "UTC+3" / "GMT-5" 等固定偏移
        if name.upper().startswith(("UTC", "GMT")):
            import re

            m = re.fullmatch(
                r"(?i)(?:UTC|GMT)([+-]?)(\d{1,2})(?::(\d{2}))?", name[len(name) - len(name) + 3 :]
                if False
                else name[3:] if name.upper().startswith(("UTC", "GMT")) else ""
            )
            # 简单实现：手动解析
            tail = name[3:]
            if not tail:
                return timezone.utc
            sign = 1
            if tail.startswith("+"):
                tail = tail[1:]
            elif tail.startswith("-"):
                sign = -1
                tail = tail[1:]
            if ":" in tail:
                hh, mm = tail.split(":")
                mins = sign * (int(hh) * 60 + int(mm))
            else:
                mins = sign * (int(tail) * 60)
            return timezone(timedelta(minutes=mins), name=name)
        raise ValueError(f"unknown timezone: {tz!r}")
    raise TypeError(f"tz must be str or tzinfo, got {type(tz).__name__}")


def _localize(dt: datetime, tz: Any) -> datetime:
    """把 naive datetime 绑定为带 tz 的 datetime；aware 则进行转换。"""
    if tz is None:
        return dt
    if dt.tzinfo is None:
        # pytz 推荐使用 localize
        if hasattr(tz, "localize"):
            return tz.localize(dt)
        return dt.replace(tzinfo=tz)
    # aware -> 转换
    return dt.astimezone(tz)


def date_range(
    start: Union[str, datetime],
    end: Optional[Union[str, datetime]] = None,
    periods: Optional[int] = None,
    freq: str = "D",
    tz: Any = None,
    normalize: bool = False,
    name: Optional[str] = None,
    inclusive: str = "both",
    unit: Optional[str] = None,
) -> DatetimeSeries:
    """生成日期范围。

    :param start: 起始日期 (str 或 datetime)
    :param end: 结束日期 (str 或 datetime, 与 periods 二选一)
    :param periods: 周期数 (与 end 二选一)
    :param freq: 频率 ('D'日, 'W'周, 'M'月, 'Y'年, 'H'时)
    :param tz: 时区 (str 或 tzinfo, 如 'CET' / 'Asia/Shanghai' / UTC)
    :param normalize: 是否把 start/end 归一到午夜 (默认 False)
    :param name: 返回的 DatetimeSeries 的 name
    :param inclusive: 区间包含性 ('both'/'left'/'right'/'neither'), 默认 'both'
    :param unit: str (保留参数, 目前忽略, 与 pandas 签名对齐)
    :return: DatetimeSeries
    """
    tzobj = _normalize_tz(tz)

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

    if normalize:
        start_dt = datetime(start_dt.year, start_dt.month, start_dt.day)
        if end_dt is not None:
            end_dt = datetime(end_dt.year, end_dt.month, end_dt.day)

    # 应用时区 (naive -> localize; aware -> convert)
    start_dt = _localize(start_dt, tzobj)
    if end_dt is not None:
        end_dt = _localize(end_dt, tzobj)

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

    # inclusive 处理
    if inclusive == "neither":
        out = out[1:-1] if len(out) >= 2 else []
    elif inclusive == "left":
        out = out[:-1] if out else out
    elif inclusive == "right":
        out = out[1:] if out else out
    elif inclusive != "both":
        raise ValueError(
            f"inclusive must be one of 'both', 'left', 'right', 'neither', got {inclusive!r}"
        )

    return DatetimeSeries(out, name=name, freq=freq, tz=tzobj)


def to_timedelta(arg, unit=None):
    """将输入转换为 timedelta。"""
    if isinstance(arg, timedelta):
        return arg
    if isinstance(arg, (int, float)):
        if unit is None:
            return timedelta(seconds=float(arg))
        unit = unit.lower()
        if unit in ("s", "sec", "seconds"):
            return timedelta(seconds=float(arg))
        elif unit in ("ms", "milli", "milliseconds"):
            return timedelta(milliseconds=float(arg))
        elif unit in ("us", "micro", "microseconds"):
            return timedelta(microseconds=float(arg))
        elif unit in ("ns", "nano", "nanoseconds"):
            return timedelta(microseconds=float(arg) / 1000)
        elif unit in ("m", "min", "minutes"):
            return timedelta(minutes=float(arg))
        elif unit in ("h", "hour", "hours"):
            return timedelta(hours=float(arg))
        elif unit in ("d", "day", "days"):
            return timedelta(days=float(arg))
        else:
            raise ValueError(f"unsupported unit: {unit}")
    if isinstance(arg, str):
        return timedelta(seconds=float(arg))
    if isinstance(arg, (list, tuple)):
        return [to_timedelta(x, unit) for x in arg]
    raise TypeError(f"cannot convert {type(arg).__name__} to timedelta")


def timedelta_range(start=None, end=None, periods=None, freq="D"):
    """生成 timedelta 范围。"""
    if start is None:
        start = timedelta(0)
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
            delta = (index[i] - index[i - 1]).days
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
