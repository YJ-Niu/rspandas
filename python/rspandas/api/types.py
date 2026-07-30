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


def is_object_dtype(arr_or_dtype) -> bool:
    """检查是否为对象类型 (object)。"""
    dtype = _extract_dtype(arr_or_dtype)
    return dtype == "object"


def is_complex_dtype(arr_or_dtype) -> bool:
    """检查是否为复数类型 (complex64/complex128)。"""
    dtype = _extract_dtype(arr_or_dtype)
    return dtype in ("complex64", "complex128")


def is_unsigned_integer_dtype(arr_or_dtype) -> bool:
    """检查是否为无符号整数类型 (uint8/uint16/uint32/uint64)。"""
    dtype = _extract_dtype(arr_or_dtype)
    return dtype in ("uint8", "uint16", "uint32", "uint64")


def is_signed_integer_dtype(arr_or_dtype) -> bool:
    """检查是否为有符号整数类型 (int8/int16/int32/int64)。"""
    dtype = _extract_dtype(arr_or_dtype)
    return dtype in ("int8", "int16", "int32", "int64")


def is_extension_type(arr_or_dtype) -> bool:
    """检查是否为扩展类型 (category/datetime64/timedelta64)。"""
    dtype = _extract_dtype(arr_or_dtype)
    return dtype in ("category", "datetime64", "timedelta64")


def is_interval_dtype(arr_or_dtype) -> bool:
    """检查是否为区间类型 (Interval)。"""
    dtype = _extract_dtype(arr_or_dtype)
    return dtype == "interval"


def is_period_dtype(arr_or_dtype) -> bool:
    """检查是否为周期类型 (Period)。"""
    dtype = _extract_dtype(arr_or_dtype)
    return dtype in ("period", "Period")


def is_sparse(obj) -> bool:
    """检查是否为稀疏对象 (SparseDtype/SparseArray)。"""
    if hasattr(obj, "dtype") and isinstance(obj.dtype, str):
        return obj.dtype.startswith("Sparse")
    if isinstance(obj, str):
        return obj.startswith("Sparse")
    return getattr(obj, "_sub_type", None) == "Sparse" or "Sparse" in type(obj).__name__


def is_re(obj) -> bool:
    """检查是否为正则表达式对象 (re.Pattern)。"""
    import re

    return isinstance(obj, re.Pattern)


def is_scalar(obj) -> bool:
    """检查是否为标量值。"""
    if obj is None:
        return True
    if isinstance(obj, (int, float, complex, bool, str, bytes)):
        return True
    import datetime

    if isinstance(obj, (datetime.date, datetime.datetime, datetime.timedelta)):
        return True
    # 拦截 numpy / rsnumpy 的 0 维数组
    if hasattr(obj, "ndim") and obj.ndim == 0:
        return True
    return False


def is_number(obj) -> bool:
    """检查是否为数字 (int/float/complex，排除 bool)。"""
    if isinstance(obj, bool):
        return False
    return isinstance(obj, (int, float, complex))


def is_iterable(obj) -> bool:
    """检查是否为可迭代对象 (排除 str/bytes)。"""
    if isinstance(obj, (str, bytes)):
        return False
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def is_file_like(obj) -> bool:
    """检查是否为文件类对象 (有 read/write 方法之一)。"""
    return hasattr(obj, "read") or hasattr(obj, "write")


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
