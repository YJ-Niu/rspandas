"""rspandas: pandas-like library built on Rust.

A drop-in pandas-like API where the heavy lifting is done in Rust.
"""

from typing import Any, Dict

from .series import Series
from .dataframe import DataFrame
from ._datetime import (
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
    read_feather,
    to_feather,
    read_pickle,
    to_pickle,
    read_sql,
    to_sql,
)
from .indexes import (
    Index,
    RangeIndex,
    MultiIndex,
    get_dummies,
    cut,
    qcut,
    crosstab,
)
from .rspandas import _Series, _DataFrame  # 重新导出 Rust 类型，供内部使用
from .rspandas import factorize as _factorize  # Rust 端 factorize
from . import offsets


# ---------------------------------------------------------------------------
# 全局选项配置
# ---------------------------------------------------------------------------

_options: Dict[str, Any] = {
    "display.max_rows": 60,
    "display.max_columns": 20,
    "display.width": 80,
    "display.precision": 6,
    "mode.chained_assignment": "warn",
}


def set_option(pat: str, value: Any) -> None:
    """设置全局选项。

    :param pat: 选项名 (如 'display.max_rows')
    :param value: 选项值
    """
    if pat in _options:
        _options[pat] = value
    else:
        raise ValueError(f"Unknown option: {pat!r}")


def get_option(pat: str) -> Any:
    """获取全局选项。

    :param pat: 选项名 (如 'display.max_rows')
    """
    if pat in _options:
        return _options[pat]
    raise ValueError(f"Unknown option: {pat!r}")


def reset_option(pat: str) -> None:
    """重置选项为默认值。

    :param pat: 选项名 (如 'display.max_rows')
    """
    _defaults = {
        "display.max_rows": 60,
        "display.max_columns": 20,
        "display.width": 80,
        "display.precision": 6,
        "mode.chained_assignment": "warn",
    }
    if pat in _defaults:
        _options[pat] = _defaults[pat]
    elif pat == "all":
        for k, v in _defaults.items():
            _options[k] = v
    else:
        raise ValueError(f"Unknown option: {pat!r}")


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


def to_numeric(arg, errors: str = "raise", downcast=None):
    """将参数转换为数值类型。

    :param arg: list / Series / 标量
    :param errors: 'raise' / 'coerce' / 'ignore'
    :param downcast: None / 'integer' / 'signed' / 'unsigned' / 'float'
    :return: 数值 Series 或标量

    Examples:
        >>> to_numeric(['1', '2', '3'])
        [1, 2, 3]
        >>> to_numeric(['1', 'x', '3'], errors='coerce')
        [1, None, 3]
    """
    from .series import Series as _Series

    if isinstance(arg, _Series):
        values = list(arg.values)
    elif isinstance(arg, (list, tuple)):
        values = list(arg)
    else:
        # 标量
        try:
            v = float(arg)
            if v == int(v) and not isinstance(arg, bool):
                return int(v)
            return v
        except (ValueError, TypeError):
            if errors == "coerce":
                return None
            if errors == "ignore":
                return arg
            raise ValueError(f"Unable to parse string {arg!r}")

    result = []
    for v in values:
        if v is None:
            result.append(None)
            continue
        try:
            if isinstance(v, (int, float)):
                result.append(v)
            elif isinstance(v, bool):
                result.append(int(v))
            elif isinstance(v, str):
                v = v.strip()
                if v == "":
                    result.append(None if errors == "coerce" else v)
                else:
                    val = float(v)
                    if val == int(val) and "." not in v and "e" not in v.lower():
                        result.append(int(val))
                    else:
                        result.append(val)
            else:
                try:
                    result.append(float(v))
                except (ValueError, TypeError):
                    if errors == "coerce":
                        result.append(None)
                    elif errors == "ignore":
                        result.append(v)
                    else:
                        raise ValueError(f"Unable to parse {v!r}")
        except (ValueError, TypeError):
            if errors == "coerce":
                result.append(None)
            elif errors == "ignore":
                result.append(v)
            else:
                raise

    if downcast:
        # 简化版 downcast
        if downcast in ("integer", "signed", "unsigned"):
            if all(v is None or isinstance(v, int) for v in result):
                pass
            else:
                result = [int(v) if v is not None and v == int(v) else v for v in result]

    return _Series(result, name=None)


__version__ = "2.0.1"
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
    "read_feather",
    "to_feather",
    "read_pickle",
    "to_pickle",
    "read_sql",
    "to_sql",
    "Index",
    "RangeIndex",
    "MultiIndex",
    "get_dummies",
    "cut",
    "qcut",
    "crosstab",
    "offsets",
    "set_option",
    "get_option",
    "reset_option",
    "to_numeric",
    "_Series",
    "_DataFrame",
    "__version__",
]
