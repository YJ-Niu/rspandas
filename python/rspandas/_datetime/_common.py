"""datetime 内部辅助函数

由 rspandas/_datetime.py 拆分而来。
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional, Union

from ..series import Series

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
        "%Y%m%d %H:%M:%S.%f",
        "%Y%m%d %H:%M:%S",
        "%Y%m%d %H:%M",
        "%Y%m%dT%H:%M:%S.%f",
        "%Y%m%dT%H:%M:%S",
        "%Y%m%dT%H%M%S",
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
    # 尝试解析特殊关键字（对齐 pandas）：now/today 等
    lower_s = s.lower().strip()
    if lower_s in ("now", "today"):
        return datetime.now()
    if lower_s == "yesterday":
        return datetime.now() - timedelta(days=1)
    if lower_s == "tomorrow":
        return datetime.now() + timedelta(days=1)
    raise ValueError(f"cannot parse date string: {s!r}") from last_err


def _to_iso(v) -> Optional[str]:
    """将 datetime/date/None 转换为 ISO 字符串。

    与 pandas 一致：
    - 日期与时间之间使用空格分隔（而非 ISO 默认的 'T'）。
    - naive datetime 当时间部分全为 0 时，只显示日期部分。
    - aware datetime 始终显示完整时间（含时区偏移）。
    """
    if v is None:
        return None
    if isinstance(v, datetime):
        # naive datetime 且时间为 00:00:00 -> 只显示日期
        if (
            v.tzinfo is None
            and v.hour == 0
            and v.minute == 0
            and v.second == 0
            and v.microsecond == 0
        ):
            return v.strftime("%Y-%m-%d")
        # 用空格替代 ISO 默认的 'T'，与 pandas 显示一致
        s = v.isoformat()
        if "T" in s:
            s = s.replace("T", " ", 1)
        return s
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day).isoformat()
    return None


# ---------------------------------------------------------------------------
# DatetimeSeries: 包装 Series，提供 datetime 语义
# ---------------------------------------------------------------------------


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


def _parse_timedelta_str(s: str) -> timedelta:
    """解析 timedelta 字符串。

    支持的格式（与 pandas 兼容）：
    - "1 day 00:00:05" / "1 days 00:00:05"
    - "1 day" / "2 days" / "3 hours" / "5 minutes" / "10 seconds"
    - "00:00:05" (HH:MM:SS)
    - "1 days 00:00:00.000005" (带微秒)
    - "1D" / "2H" / "3min" / "4s" (缩写)
    """
    import re

    s = s.strip()

    # 格式 1: "N day(s) [HH:MM:SS[.ffffff]]" 或 "[N day(s)] HH:MM:SS"
    # 先提取 "N day(s)" 部分
    day_match = re.match(r"^(\d+)\s+days?\s*(.*)$", s, re.IGNORECASE)
    days = 0
    rest = s
    if day_match:
        days = int(day_match.group(1))
        rest = day_match.group(2).strip()
    elif s.lower().endswith(("day", "days")):
        m = re.match(r"^(\d+)\s+days?$", s, re.IGNORECASE)
        if m:
            return timedelta(days=int(m.group(1)))

    # 如果剩余部分是 HH:MM:SS 格式
    if rest:
        time_match = re.match(r"^(\d+):(\d+):(\d+)(?:\.(\d+))?$", rest)
        if time_match:
            h = int(time_match.group(1))
            m = int(time_match.group(2))
            sec = int(time_match.group(3))
            us = 0
            if time_match.group(4):
                frac = time_match.group(4)
                # 补齐到 6 位微秒
                frac = (frac + "000000")[:6]
                us = int(frac)
            return timedelta(
                days=days, hours=h, minutes=m, seconds=sec, microseconds=us
            )
        # 如果只有 days 部分
        if days > 0:
            return timedelta(days=days)

    # 格式 2: 单位缩写 "1D" / "2H" / "3min" / "4s" / "5ms" / "6us"
    unit_match = re.match(
        r"^(\d+(?:\.\d+)?)\s*(ns|us|µs|ms|s|min|minutes|h|hour|hours|D|day|days|H|M|S)\b",
        s,
        re.IGNORECASE,
    )
    if unit_match:
        val = float(unit_match.group(1))
        u = unit_match.group(2).lower()
        if u in ("d", "day", "days"):
            return timedelta(days=val)
        elif u in ("h", "hour", "hours"):
            return timedelta(hours=val)
        elif u in ("min", "minutes", "m"):
            return timedelta(minutes=val)
        elif u in ("s",):
            return timedelta(seconds=val)
        elif u in ("ms",):
            return timedelta(milliseconds=val)
        elif u in ("us", "µs"):
            return timedelta(microseconds=val)
        elif u in ("ns",):
            return timedelta(microseconds=val / 1000)

    # 格式 3: 纯数字字符串（默认秒）
    try:
        return timedelta(seconds=float(s))
    except ValueError:
        pass

    raise ValueError(f"could not convert string to timedelta: {s!r}")
