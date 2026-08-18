"""时间差：to_timedelta / timedelta_range

由 rspandas/_datetime.py 拆分而来。
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional, Union

from ..series import Series
from ._common import _parse_timedelta_str, _freq_to_timedelta


def to_timedelta(arg, unit=None, errors: str = "raise"):
    """将输入转换为 timedelta。

    :param arg: 标量/list/Series
    :param unit: 数值输入的单位（'s'/'ms'/'us'/'ns'/'m'/'h'/'d'）
    :param errors: 'raise' / 'coerce' / 'ignore'
    """
    # Series 输入：逐列转换并返回 Series
    if isinstance(arg, Series):
        from ..dataframe import _convert_to_basic

        out = []
        for v in arg.values:
            try:
                td = to_timedelta(v, unit, errors)
                # 将 timedelta 转换为 pandas 兼容的字符串格式
                out.append(_convert_to_basic(td) if td is not None else None)
            except (ValueError, TypeError):
                if errors == "coerce":
                    out.append(None)
                elif errors == "ignore":
                    out.append(v)
                else:
                    raise
        return Series(out, name=arg.name)
    if isinstance(arg, timedelta):
        return arg
    if isinstance(arg, (int, float)):
        if unit is None:
            return timedelta(seconds=float(arg))
        unit_lower = unit.lower()
        if unit_lower in ("s", "sec", "seconds"):
            return timedelta(seconds=float(arg))
        elif unit_lower in ("ms", "milli", "milliseconds"):
            return timedelta(milliseconds=float(arg))
        elif unit_lower in ("us", "micro", "microseconds"):
            return timedelta(microseconds=float(arg))
        elif unit_lower in ("ns", "nano", "nanoseconds"):
            return timedelta(microseconds=float(arg) / 1000)
        elif unit_lower in ("m", "min", "minutes"):
            return timedelta(minutes=float(arg))
        elif unit_lower in ("h", "hour", "hours"):
            return timedelta(hours=float(arg))
        elif unit_lower in ("d", "day", "days"):
            return timedelta(days=float(arg))
        else:
            raise ValueError(f"unsupported unit: {unit}")
    if isinstance(arg, str):
        try:
            return _parse_timedelta_str(arg)
        except (ValueError, TypeError):
            if errors == "coerce":
                # pandas: errors='coerce' 时无法解析的元素变为 NaT（用 None 表示）
                return None
            if errors == "ignore":
                return arg
            raise
    # rspandas.Timedelta 对象
    if hasattr(arg, "_td") and isinstance(getattr(arg, "_td", None), timedelta):
        return arg._td
    if isinstance(arg, (list, tuple)):
        from ..indexes import TimedeltaIndex

        out = []
        for x in arg:
            try:
                out.append(to_timedelta(x, unit, errors))
            except (ValueError, TypeError):
                if errors == "coerce":
                    out.append(None)
                elif errors == "ignore":
                    out.append(x)
                else:
                    raise
        return TimedeltaIndex(out)
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
    # 返回带 timedelta 缓存的 Series
    s = Series(out, name=None)
    s._td_values = list(out)
    s._dtype_str = "timedelta64[us]"
    return s
