"""rspandas: pandas-like library built on Rust.

A drop-in pandas-like API where the heavy lifting is done in Rust.
"""

from . import offsets
from ._datetime import (
    bdate_range,
    date_range,
    DatetimeSeries,
    infer_freq,
    period_range,
    timedelta_range,
    to_datetime,
    to_timedelta,
)
from .dataframe import DataFrame
from .indexes import crosstab, cut, get_dummies, Index, MultiIndex, qcut, RangeIndex
from .io import (
    ExcelWriter,
    read_excel,
    read_feather,
    read_json,
    read_parquet,
    read_pickle,
    read_sql,
    to_excel,
    to_feather,
    to_json,
    to_parquet,
    to_pickle,
    to_sql,
)
from .rspandas import _DataFrame, _Series  # 重新导出 Rust 类型，供内部使用
from .rspandas import factorize as _factorize  # Rust 端 factorize
from .series import Series
from typing import Any, Dict

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
                result = [
                    int(v) if v is not None and v == int(v) else v for v in result
                ]

    return _Series(result, name=None)


def merge(
    left,
    right,
    how: str = "inner",
    on=None,
    left_on=None,
    right_on=None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes=("_x", "_y"),
) -> "DataFrame":
    """合并两个 DataFrame。

    :param left: 左侧 DataFrame
    :param right: 右侧 DataFrame
    :param how: 合并方式 ('inner'/'outer'/'left'/'right')
    :param on: 合并键列
    :param left_on: 左侧合并键
    :param right_on: 右侧合并键
    :param left_index: 是否使用左侧索引
    :param right_index: 是否使用右侧索引
    :param sort: 是否排序
    :param suffixes: 重复列的后缀
    """
    if not isinstance(left, DataFrame) or not isinstance(right, DataFrame):
        raise TypeError("merge requires DataFrame inputs")
    return left.merge(
        right,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        sort=sort,
        suffixes=suffixes,
    )


def concat(
    objs, axis: int = 0, join: str = "outer", ignore_index: bool = False
) -> "Series":
    """拼接多个 Series 或 DataFrame。

    :param objs: 要拼接的对象列表
    :param axis: 拼接轴 (0=纵向, 1=横向)
    :param join: 连接方式 ('outer'/'inner')
    :param ignore_index: 是否忽略索引
    """
    from .series import Series as _Series

    if not objs:
        return _Series([])

    if all(isinstance(o, _Series) for o in objs):
        all_values = []
        all_index = []
        for s in objs:
            all_values.extend(list(s.values))
            if not ignore_index:
                idx = s._index if s._index else list(range(len(s)))
                all_index.extend(idx)
        return _Series(all_values, index=all_index if not ignore_index else None)

    if all(isinstance(o, DataFrame) for o in objs):
        all_data = {}
        all_columns = set()
        for df in objs:
            all_columns.update(df._columns)
        for col in all_columns:
            col_values = []
            for df in objs:
                if col in df._columns:
                    col_values.extend(list(df._inner.get_column(col).values))
                else:
                    col_values.extend([None] * df._nrows)
            all_data[col] = col_values
        return DataFrame(all_data)

    raise TypeError("concat requires all objects of the same type")


def isnull(obj):
    """检测缺失值 (None)。"""
    if isinstance(obj, _Series):
        return obj.isnull()
    elif isinstance(obj, DataFrame):
        return obj.isnull()
    elif isinstance(obj, list):
        return [v is None for v in obj]
    else:
        return obj is None


def notnull(obj):
    """检测非缺失值。"""
    if isinstance(obj, _Series):
        return obj.notnull()
    elif isinstance(obj, DataFrame):
        return obj.notnull()
    elif isinstance(obj, list):
        return [v is not None for v in obj]
    else:
        return obj is not None


# isna / notna 别名
isna = isnull
notna = notnull


def unique(values):
    """返回唯一值。"""
    if isinstance(values, _Series):
        return values.unique()
    elif isinstance(values, list):
        seen = set()
        result = []
        for v in values:
            if v not in seen:
                seen.add(v)
                result.append(v)
        return result
    raise TypeError("unique requires Series or list")


def value_counts(
    values, normalize: bool = False, sort: bool = True, ascending: bool = False
) -> _Series:
    """统计值出现次数。"""
    if isinstance(values, _Series):
        return values.value_counts()
    elif isinstance(values, list):
        s = _Series(values)
        return s.value_counts()
    raise TypeError("value_counts requires Series or list")


__version__ = "2.0.6"
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
    "ExcelWriter",
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
    "merge",
    "concat",
    "isnull",
    "notnull",
    "isna",
    "notna",
    "unique",
    "value_counts",
    "_Series",
    "_DataFrame",
    "__version__",
]
