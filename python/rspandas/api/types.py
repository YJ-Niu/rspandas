"""rspandas.api.types - 类型判断函数。

提供与 pandas.api.types 兼容的类型检查函数。
"""


def is_numeric_dtype(arr_or_dtype) -> bool:
    """检查是否为数值类型 (int64 / float64)。

    :param arr_or_dtype: 数组、Series 或 dtype 字符串
    """
    dtype = _extract_dtype(arr_or_dtype)
    return dtype in ("int64", "float64")


def is_string_dtype(arr_or_dtype) -> bool:
    """检查是否为字符串类型 (object)。

    :param arr_or_dtype: 数组、Series 或 dtype 字符串
    """
    dtype = _extract_dtype(arr_or_dtype)
    return dtype == "object"


def is_bool_dtype(arr_or_dtype) -> bool:
    """检查是否为布尔类型。

    :param arr_or_dtype: 数组、Series 或 dtype 字符串
    """
    dtype = _extract_dtype(arr_or_dtype)
    return dtype == "bool"


def is_integer_dtype(arr_or_dtype) -> bool:
    """检查是否为整数类型。

    :param arr_or_dtype: 数组、Series 或 dtype 字符串
    """
    dtype = _extract_dtype(arr_or_dtype)
    return dtype == "int64"


def is_float_dtype(arr_or_dtype) -> bool:
    """检查是否为浮点类型。

    :param arr_or_dtype: 数组、Series 或 dtype 字符串
    """
    dtype = _extract_dtype(arr_or_dtype)
    return dtype == "float64"


def is_categorical_dtype(arr_or_dtype) -> bool:
    """检查是否为分类类型。

    :param arr_or_dtype: 数组、Series 或 dtype 字符串
    """
    dtype = _extract_dtype(arr_or_dtype)
    return dtype == "category"


def is_datetime64_any_dtype(arr_or_dtype) -> bool:
    """检查是否为 datetime 类型。

    :param arr_or_dtype: 数组、Series 或 dtype 字符串
    """
    dtype = _extract_dtype(arr_or_dtype)
    return dtype == "datetime64"


def is_timedelta64_dtype(arr_or_dtype) -> bool:
    """检查是否为 timedelta 类型。

    :param arr_or_dtype: 数组、Series 或 dtype 字符串
    """
    dtype = _extract_dtype(arr_or_dtype)
    return dtype == "timedelta64"


def is_dict_like(obj) -> bool:
    """检查是否为 dict-like 对象。

    :param obj: 任意对象
    """
    return isinstance(obj, dict) or hasattr(obj, "keys") and hasattr(obj, "__getitem__")


def is_list_like(obj) -> bool:
    """检查是否为 list-like 对象。

    :param obj: 任意对象
    """
    if isinstance(obj, (str, bytes)):
        return False
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _extract_dtype(arr_or_dtype) -> str:
    """从各种输入中提取 dtype 字符串。"""
    if isinstance(arr_or_dtype, str):
        return arr_or_dtype
    if hasattr(arr_or_dtype, "dtype"):
        d = arr_or_dtype.dtype
        return d if isinstance(d, str) else str(d)
    if hasattr(arr_or_dtype, "dtypes"):
        return arr_or_dtype.dtypes
    raise TypeError(f"Cannot extract dtype from {type(arr_or_dtype).__name__}")