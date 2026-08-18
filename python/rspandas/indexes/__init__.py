"""索引类型与分箱工具子包。

按功能聚合：
- ``base``: ``Index`` / ``RangeIndex`` / ``MultiIndex`` / ``IntervalIndex``
  / ``CategoricalIndex`` / ``DatetimeIndex`` / ``TimedeltaIndex`` / ``PeriodIndex``
- ``binning``: ``cut`` / ``qcut`` / ``crosstab`` / ``get_dummies``

向后兼容：``rspandas.indexes.Series`` / ``rspandas.indexes.rspandas_DataFrame``
仍然可访问，供老代码使用。
"""

from __future__ import annotations

from ..rspandas import _DataFrame as rspandas_DataFrame  # type: ignore  # noqa: F401
from ..series import Series  # noqa: F401
from .base import (
    CategoricalIndex,
    DatetimeIndex,
    Index,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    TimedeltaIndex,
)
from .binning import crosstab, cut, get_dummies, qcut

__all__ = [
    "Index",
    "RangeIndex",
    "MultiIndex",
    "IntervalIndex",
    "CategoricalIndex",
    "DatetimeIndex",
    "TimedeltaIndex",
    "PeriodIndex",
    "cut",
    "qcut",
    "crosstab",
    "get_dummies",
    "Series",
    "rspandas_DataFrame",
]
