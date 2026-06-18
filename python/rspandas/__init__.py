"""rspandas: pandas-like library built on Rust.

A drop-in pandas-like API where the heavy lifting is done in Rust.
"""

from .series import Series
from .dataframe import DataFrame
from .datetime import (
    to_datetime,
    date_range,
    to_timedelta,
    timedelta_range,
    period_range,
    bdate_range,
    infer_freq,
    DatetimeSeries,
)
from .rspandas import _Series, _DataFrame  # 重新导出 Rust 类型，供内部使用


__version__ = "1.0.0"
__all__ = [
    "Series",
    "DataFrame",
    "to_datetime",
    "date_range",
    "to_timedelta",
    "timedelta_range",
    "period_range",
    "bdate_range",
    "infer_freq",
    "DatetimeSeries",
    "_Series",
    "_DataFrame",
    "__version__",
]
