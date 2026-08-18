"""标量类型：Timestamp, Timedelta, Period, Interval, Categorical, DateOffset, _NaT, _NA

由 __init__.py 拆分而来。
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from ._datetime import DatetimeSeries, to_datetime, to_timedelta


class Timestamp:
    """时间戳（简化版，包装 datetime.datetime）。"""

    def __init__(self, *args, **kwargs):
        import datetime

        if len(args) == 1 and isinstance(args[0], str):
            # 简化：只支持 ISO 格式
            self._dt = datetime.datetime.fromisoformat(args[0])
        elif len(args) == 1 and isinstance(args[0], datetime.datetime):
            self._dt = args[0]
        else:
            self._dt = datetime.datetime(*args, **kwargs)

    def __repr__(self) -> str:
        # 用空格替代 ISO 默认的 'T'，与 pandas 显示一致
        return f"Timestamp('{self._dt.isoformat().replace('T', ' ', 1)}')"

    def __str__(self) -> str:
        # 用空格替代 ISO 默认的 'T'，与 pandas 显示一致
        return self._dt.isoformat().replace("T", " ", 1)

    def __eq__(self, other):
        if isinstance(other, Timestamp):
            return self._dt == other._dt
        return self._dt == other

    def __lt__(self, other):
        if isinstance(other, Timestamp):
            return self._dt < other._dt
        return self._dt < other

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return not (self <= other)

    def __ge__(self, other):
        return not (self < other)

    @property
    def year(self):
        return self._dt.year

    @property
    def month(self):
        return self._dt.month

    @property
    def day(self):
        return self._dt.day

    @property
    def hour(self):
        return self._dt.hour

    @property
    def minute(self):
        return self._dt.minute

    @property
    def second(self):
        return self._dt.second


class Timedelta:
    """时间差（简化版，包装 datetime.timedelta）。"""

    def __init__(self, *args, **kwargs):
        import datetime
        import re

        if len(args) == 1 and isinstance(args[0], str):
            # 简化：只支持 "X days HH:MM:SS" 类似格式
            s = args[0].strip()
            parts = s.split()
            days = 0
            time_part = s
            if len(parts) > 1 and parts[1] in ("days", "day"):
                days = int(parts[0])
                time_part = parts[2] if len(parts) > 2 else "0:0:0"
            elif "day" in s:
                # "1day" 或 "1 day" 格式 - 使用正则提取数字
                m = re.search(r"(\d+)\s*day", s)
                if m:
                    days = int(m.group(1))
                time_part = "0:0:0"
            tparts = time_part.split(":")
            hours = int(tparts[0]) if len(tparts) > 0 else 0
            minutes = int(tparts[1]) if len(tparts) > 1 else 0
            seconds = int(tparts[2]) if len(tparts) > 2 else 0
            self._td = datetime.timedelta(
                days=days, hours=hours, minutes=minutes, seconds=seconds
            )
        else:
            self._td = datetime.timedelta(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Timedelta('{self._td}')"

    def __str__(self) -> str:
        return str(self._td)

    def __eq__(self, other):
        if isinstance(other, Timedelta):
            return self._td == other._td
        return self._td == other

    @property
    def days(self):
        return self._td.days

    @property
    def seconds(self):
        return self._td.seconds

    @property
    def total_seconds(self):
        return self._td.total_seconds()


class Period:
    """周期类型（简化版）。"""

    def __init__(self, value=None, freq=None):
        self._value = value
        self._freq = freq

    def __repr__(self) -> str:
        return f"Period('{self._value}', freq='{self._freq}')"

    def __str__(self) -> str:
        return str(self._value)

    @property
    def freq(self):
        return self._freq


class Interval:
    """区间类型。"""

    def __init__(self, left, right, closed: str = "right"):
        self.left = left
        self.right = right
        self.closed = closed

    def __repr__(self) -> str:
        return f"Interval({self.left}, {self.right}, closed='{self.closed}')"

    def __contains__(self, item):
        if self.closed == "right":
            return self.left <= item < self.right
        elif self.closed == "left":
            return self.left < item <= self.right
        elif self.closed == "both":
            return self.left <= item <= self.right
        else:  # neither
            return self.left < item < self.right


class Categorical:
    """分类类型（简化版）。"""

    def __init__(self, values, categories=None, ordered=False):
        self._values = list(values)
        self._categories = (
            list(categories) if categories is not None else sorted(set(values))
        )
        self._ordered = ordered

    @property
    def categories(self):
        return self._categories

    @property
    def ordered(self):
        return self._ordered

    def __repr__(self) -> str:
        return f"Categorical({self._values})"

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


class DateOffset:
    """日期偏移量（简化版）。"""

    def __init__(
        self,
        days=0,
        hours=0,
        minutes=0,
        seconds=0,
        weeks=0,
        months=0,
        years=0,
    ):
        self.days = days
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
        self.weeks = weeks
        self.months = months
        self.years = years

    def __repr__(self) -> str:
        return (
            f"DateOffset(days={self.days}, hours={self.hours}, "
            f"months={self.months}, years={self.years})"
        )

    def __add__(self, other):
        import datetime

        if isinstance(other, datetime.datetime):
            # 简化：只处理 days/hours/minutes/seconds
            return other + datetime.timedelta(
                days=self.days + self.weeks * 7,
                hours=self.hours,
                minutes=self.minutes,
                seconds=self.seconds,
            )
        return NotImplemented


# ============================================================================
# 缺失值常量
# ============================================================================


class _NaT:
    """Not a Time（时间类型的缺失值）。"""

    def __repr__(self) -> str:
        return "NaT"

    def __bool__(self) -> bool:
        return False

    def __eq__(self, other):
        return isinstance(other, _NaT)

    def __ne__(self, other):
        return not isinstance(other, _NaT)

    def __hash__(self):
        return hash("NaT")


class _NA:
    """pandas.NA（缺失值常量）。"""

    def __repr__(self) -> str:
        return "<NA>"

    def __bool__(self) -> bool:
        return False

    def __eq__(self, other):
        return isinstance(other, _NA)

    def __ne__(self, other):
        return not isinstance(other, _NA)

    def __hash__(self):
        return hash("NA")


NaT = _NaT()
NA = _NA()
