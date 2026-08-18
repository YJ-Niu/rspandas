"""周期：period_range

由 rspandas/_datetime.py 拆分而来。
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional, Union

from ..series import Series
from ._common import _parse_iso, _freq_to_timedelta


def period_range(start=None, end=None, periods=None, freq="D", name=None):
    """生成周期范围。

    与 pandas 一致，默认 freq='D'（当 start 是日期字符串时）。
    返回 PeriodIndex，dtype 显示为 period[freq]。

    :param start: 起始日期 (str/datetime/date)
    :param end: 结束日期 (与 periods 二选一)
    :param periods: 周期数 (与 end 二选一)
    :param freq: 频率 ('D'/'M'/'Q'/'Y'/'H' 等)
    :param name: 索引名称
    """
    from ..indexes import PeriodIndex

    if start is None:
        start = datetime.now()
    elif isinstance(start, str):
        start = _parse_iso(start)
    elif isinstance(start, date) and not isinstance(start, datetime):
        start = datetime(start.year, start.month, start.day)

    if periods is None and end is None:
        periods = 12

    # 如果指定了 end，计算 periods
    if periods is None and end is not None:
        if isinstance(end, str):
            end_dt = _parse_iso(end)
        elif isinstance(end, date) and not isinstance(end, datetime):
            end_dt = datetime(end.year, end.month, end.day)
        else:
            end_dt = end
        step = _freq_to_timedelta(freq)
        periods = int((end_dt - start) / step) + 1

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

    return PeriodIndex(out, freq=freq, name=name)
