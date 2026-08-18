"""rsnumpy 函数包装器

将 rsnumpy 的数学函数包装为对 Series/DataFrame 返回正确类型。
由 __init__.py 拆分而来。
"""

from __future__ import annotations


def _wrap_rsnumpy_functions():
    """包装 rsnumpy 的数学函数，使其对 Series/DataFrame 返回正确类型。"""
    import rsnumpy as _rnp
    from .series import Series as _Series
    from .dataframe import DataFrame as _DataFrame

    def _is_series_or_df(obj):
        """检查对象是否为 Series 或 DataFrame。"""
        return isinstance(obj, _Series) or isinstance(obj, _DataFrame)

    def _ensure_rsnumpy_array(obj):
        """将 Series/DataFrame 转换为 rsnumpy ndarray。"""
        if isinstance(obj, _Series):
            return obj.to_numpy()
        if isinstance(obj, _DataFrame):
            return obj.to_numpy()
        return obj

    def _apply_unary_ufunc(func_name):
        """创建一元函数的包装器。"""
        original_func = getattr(_rnp, func_name)

        def _to_scalar(val):
            """将 rsnumpy 0-dim ndarray 转为 Python 标量。"""
            if hasattr(val, "item"):
                return val.item()
            if hasattr(val, "tolist"):
                return val.tolist()
            return val

        def wrapper(x):
            if isinstance(x, _Series):
                values = list(x.values)
                result_values = []
                for v in values:
                    if v is None:
                        result_values.append(None)
                    else:
                        try:
                            raw = original_func(v)
                            result_values.append(_to_scalar(raw))
                        except (TypeError, ValueError):
                            result_values.append(None)
                # 推断 dtype
                dtype = x._dtype_str
                non_null = [v for v in result_values if v is not None]
                if non_null:
                    if all(isinstance(v, bool) for v in non_null):
                        dtype = "bool"
                    elif all(isinstance(v, int) for v in non_null):
                        dtype = "int64"
                    else:
                        dtype = "float64"
                return _Series(result_values, name=x.name, dtype=dtype, index=x._index)
            elif isinstance(x, _DataFrame):
                # 对 DataFrame 逐列应用
                new_data = {}
                for col_name in x._columns:
                    col_series = x[col_name]
                    new_data[col_name] = wrapper(col_series).values
                return _DataFrame(new_data, index=x._index)
            else:
                return original_func(x)

        wrapper.__name__ = func_name
        return wrapper

    def _apply_binary_ufunc(func_name):
        """创建二元函数的包装器（支持 remainder, maximum 等）。"""
        original_func = getattr(_rnp, func_name)

        def _to_scalar(val):
            """将 rsnumpy 0-dim ndarray 转为 Python 标量。"""
            if hasattr(val, "item"):
                return val.item()
            if hasattr(val, "tolist"):
                return val.tolist()
            return val

        def _infer_dtype(result_vals, original_dtypes=None):
            """推断结果的 dtype，并转换值为正确类型。

            :param original_dtypes: 原始操作数的 dtype 集合，用于判断是否应保持 float 类型
            """
            non_null = [v for v in result_vals if v is not None]
            has_nan = any(v is None for v in result_vals)
            if not non_null:
                return "float64", result_vals
            # 如果有 NaN/None，必须用 float64（因为 NaN 是浮点数）
            if has_nan:
                return "float64", result_vals
            if all(isinstance(v, bool) for v in non_null):
                return "bool", result_vals
            if all(isinstance(v, int) for v in non_null):
                return "int64", result_vals
            if all(isinstance(v, float) and v == int(v) for v in non_null):
                # 如果原始操作数中有 float 类型，保持 float 类型
                if original_dtypes and any(
                    d in ("float64", "float32", "float16", "float")
                    for d in original_dtypes
                ):
                    return "float64", result_vals
                # 整数值的浮点数，转换为整数
                converted = [
                    int(v) if isinstance(v, float) and v == int(v) else v
                    for v in result_vals
                ]
                return "int64", converted
            return "float64", result_vals

        def _to_list(val):
            """将各种输入转为 Python 列表。"""
            if isinstance(val, _Series):
                return list(val.values), (
                    list(val._index)
                    if val._index is not None
                    else list(range(len(val)))
                )
            elif isinstance(val, _DataFrame):
                return val.to_numpy().tolist(), (
                    list(val._index)
                    if val._index is not None
                    else list(range(len(val)))
                )
            elif hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                try:
                    return [v for v in val], None
                except TypeError:
                    return val, None
            else:
                return val, None

        def wrapper(x, y):
            x_is_custom = isinstance(x, (_Series, _DataFrame))
            y_is_custom = isinstance(y, (_Series, _DataFrame))

            if x_is_custom or y_is_custom:
                # 将输入转为列表
                x_vals, x_idx = _to_list(x)
                y_vals, y_idx = _to_list(y)

                # 如果 x 是 Series/DataFrame，_to_list 会正确处理
                # 但如果 x 是 Index 或其他 array-like，x_idx 为 None

                # 决定结果类型和对齐方式
                if (
                    x_is_custom
                    and y_is_custom
                    and isinstance(x, _Series)
                    and isinstance(y, _Series)
                ):
                    # Series + Series: 按索引对齐
                    if x_idx is None:
                        x_idx = list(range(len(x_vals)))
                    if y_idx is None:
                        y_idx = list(range(len(y_vals)))
                    # 计算并集索引
                    union_idx = list(x_idx)
                    seen = set(x_idx)
                    for idx in y_idx:
                        if idx not in seen:
                            seen.add(idx)
                            union_idx.append(idx)
                    # 排序并集以匹配 pandas 行为
                    try:
                        union_idx = sorted(union_idx)
                    except TypeError:
                        pass
                    # 构建映射
                    x_map = dict(zip(x_idx, x_vals))
                    y_map = dict(zip(y_idx, y_vals))
                    # 逐元素计算
                    result_vals = []
                    for idx in union_idx:
                        xv = x_map.get(idx)
                        yv = y_map.get(idx)
                        if xv is None or yv is None:
                            result_vals.append(None)
                        else:
                            try:
                                raw = original_func(xv, yv)
                                result_vals.append(_to_scalar(raw))
                            except (TypeError, ValueError):
                                result_vals.append(None)
                    x_dtype = getattr(x, "dtype", None) if x_is_custom else None
                    y_dtype = getattr(y, "dtype", None) if y_is_custom else None
                    dtype, result_vals = _infer_dtype(
                        result_vals, original_dtypes=[x_dtype, y_dtype]
                    )
                    return _Series(
                        result_vals,
                        dtype=dtype,
                        index=union_idx,
                        name=getattr(x, "name", None),
                    )

                elif x_is_custom and isinstance(x, _Series):
                    # Series + 其他（Index, array-like, scalar）
                    if isinstance(y_vals, list) and len(y_vals) == len(x_vals):
                        # 逐元素计算
                        result_vals = []
                        for i, xv in enumerate(x_vals):
                            yv = y_vals[i] if i < len(y_vals) else None
                            if xv is None or yv is None:
                                result_vals.append(None)
                            else:
                                try:
                                    raw = original_func(xv, yv)
                                    result_vals.append(_to_scalar(raw))
                                except (TypeError, ValueError):
                                    result_vals.append(None)
                    else:
                        # scalar
                        result_vals = []
                        for xv in x_vals:
                            if xv is None or y_vals is None:
                                result_vals.append(None)
                            else:
                                try:
                                    raw = original_func(xv, y_vals)
                                    result_vals.append(_to_scalar(raw))
                                except (TypeError, ValueError):
                                    result_vals.append(None)
                    dtype, result_vals = _infer_dtype(
                        result_vals, original_dtypes=[getattr(x, "dtype", None)]
                    )
                    return _Series(
                        result_vals, dtype=dtype, name=x.name, index=x._index
                    )

                elif y_is_custom and isinstance(y, _Series):
                    # 其他 + Series
                    if isinstance(x_vals, list) and len(x_vals) == len(y_vals):
                        # 逐元素计算
                        result_vals = []
                        for i, yv in enumerate(y_vals):
                            xv = x_vals[i] if i < len(x_vals) else None
                            if xv is None or yv is None:
                                result_vals.append(None)
                            else:
                                try:
                                    raw = original_func(xv, yv)
                                    result_vals.append(_to_scalar(raw))
                                except (TypeError, ValueError):
                                    result_vals.append(None)
                    else:
                        # scalar
                        result_vals = []
                        for yv in y_vals:
                            if x_vals is None or yv is None:
                                result_vals.append(None)
                            else:
                                try:
                                    raw = original_func(x_vals, yv)
                                    result_vals.append(_to_scalar(raw))
                                except (TypeError, ValueError):
                                    result_vals.append(None)
                    dtype, result_vals = _infer_dtype(
                        result_vals, original_dtypes=[getattr(y, "dtype", None)]
                    )
                    return _Series(
                        result_vals, dtype=dtype, name=y.name, index=y._index
                    )

            return original_func(x, y)

        wrapper.__name__ = func_name
        return wrapper

    # 包装常用的一元函数
    for fname in [
        "exp",
        "log",
        "log10",
        "log2",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "abs",
        "floor",
        "ceil",
        "negative",
        "positive",
    ]:
        if hasattr(_rnp, fname):
            setattr(_rnp, fname, _apply_unary_ufunc(fname))

    # 包装常用的二元函数
    for fname in [
        "remainder",
        "maximum",
        "minimum",
        "power",
        "add",
        "subtract",
        "multiply",
        "divide",
        "mod",
    ]:
        if hasattr(_rnp, fname):
            setattr(_rnp, fname, _apply_binary_ufunc(fname))

    # 包装 asarray 使其对 Series/DataFrame 返回 rsnumpy ndarray（显示格式对齐 pandas）
    original_asarray = _rnp.asarray if hasattr(_rnp, "asarray") else None
    if original_asarray is not None:

        def _wrap_asarray(a, *args, **kwargs):
            if isinstance(a, _Series):
                # 返回 rsnumpy ndarray 以获得正确的显示格式
                # 将 None 替换为 NaN (rsnumpy 不支持 None)
                vals = list(a.values)
                vals = [float("nan") if v is None else v for v in vals]
                return _rnp.array(vals, *args, **kwargs)
            if isinstance(a, _DataFrame):
                # 将 DataFrame 转换为 rsnumpy 二维数组
                cols = list(a._columns)
                data = [
                    [
                        (
                            float("nan")
                            if a._inner.get_column(c).values[i] is None
                            else a._inner.get_column(c).values[i]
                        )
                        for c in cols
                    ]
                    for i in range(a._nrows)
                ]
                return _rnp.array(data, *args, **kwargs)
            return original_asarray(a, *args, **kwargs)

        _rnp.asarray = _wrap_asarray

    # 包装聚合函数 (mean/sum/std/var/min/max/median/prod)
    # 使 Series 输入时默认排除 NA (与 pandas 行为一致):
    #   np.mean(series)            -> 排除 NA (委托给 series.mean(skipna=True))
    #   np.mean(series.to_numpy()) -> 不排除 NA (走 rsnumpy 原生路径)
    _reduction_map = {
        "mean": "mean",
        "sum": "sum",
        "std": "std",
        "var": "var",
        "min": "min",
        "max": "max",
        "amin": "min",
        "amax": "max",
        "median": "median",
        "prod": "prod",
        "product": "prod",
    }
    for fname, method_name in _reduction_map.items():
        if not hasattr(_rnp, fname):
            continue
        _original = getattr(_rnp, fname)

        def _make_reduction(_fn, _mn, _orig):
            def wrapper(a, *args, **kwargs):
                if isinstance(a, _Series):
                    method = getattr(a, _mn)
                    if _fn in ("std", "var"):
                        # pandas 默认 ddof=1, numpy 默认 ddof=0
                        # np.std(series) 采用 pandas 语义 (ddof=1)
                        ddof = kwargs.get("ddof", None)
                        if ddof is None and len(args) > 3:
                            ddof = args[3]
                        if ddof is None:
                            ddof = 1
                        return method(skipna=True, ddof=ddof)
                    return method()
                if isinstance(a, _DataFrame):
                    from .series import Series as _SSeries

                    target_cols = [
                        c
                        for c in a._columns
                        if a._inner.get_column(c).dtype in ("int64", "float64")
                    ]
                    return _SSeries(
                        {
                            c: wrapper(a._get_column_as_series(c), *args, **kwargs)
                            for c in target_cols
                        }
                    )
                return _orig(a, *args, **kwargs)

            wrapper.__name__ = _fn
            return wrapper

        setattr(_rnp, fname, _make_reduction(fname, method_name, _original))


# 在模块加载时执行包装
try:
    _wrap_rsnumpy_functions()
except Exception:
    pass  # rsnumpy 未安装时静默忽略
