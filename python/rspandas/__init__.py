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
from .io import (
    read_json,
    to_json,
    read_excel,
    to_excel,
    read_parquet,
    to_parquet,
    read_pickle,
    to_pickle,
    read_sql,
    to_sql,
)
from .rspandas import _Series, _DataFrame  # 重新导出 Rust 类型，供内部使用
from .rspandas import factorize as _factorize  # Rust 端 factorize


def factorize(values):
    """对值进行编码，返回 (codes, categories)。

    Examples:
        >>> import rspandas as rpd
        >>> codes, cats = rpd.factorize(['a', 'b', 'a', 'c', 'b'])
        >>> list(codes)
        [0, 1, 0, 2, 1]
        >>> list(cats)
        ['a', 'b', 'c']
    """
    return _factorize(values)


__version__ = "1.2.0"
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
    "factorize",
    "read_json",
    "to_json",
    "read_excel",
    "to_excel",
    "read_parquet",
    "to_parquet",
    "read_pickle",
    "to_pickle",
    "read_sql",
    "to_sql",
    "_Series",
    "_DataFrame",
    "__version__",
]
