"""datetime 子包：日期时间相关函数。

按功能拆分：
- :mod:`._common`: 内部辅助（_parse_iso / _FREQ_MAP 等）
- :mod:`.datetime`: DatetimeSeries / DatetimeAccessor / to_datetime / date_range / bdate_range / infer_freq
- :mod:`.timedelta`: to_timedelta / timedelta_range
- :mod:`.period`: period_range

向后兼容：``from rspandas._datetime import to_datetime`` 仍可用。
"""

from __future__ import annotations

from ._common import (  # noqa: F401
    _freq_to_timedelta,
    _localize,
    _normalize_tz,
    _parse_iso,
    _parse_timedelta_str,
    _to_iso,
)
from .datetime import (
    DatetimeAccessor,
    DatetimeSeries,
    bdate_range,
    date_range,
    infer_freq,
    to_datetime,
)
from .period import period_range
from .timedelta import timedelta_range, to_timedelta

__all__ = [
    "DatetimeSeries",
    "DatetimeAccessor",
    "to_datetime",
    "date_range",
    "bdate_range",
    "infer_freq",
    "to_timedelta",
    "timedelta_range",
    "period_range",
    "_parse_iso",
]
