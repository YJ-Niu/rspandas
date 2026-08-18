"""EWM 指数加权

由 rspandas/series.py 拆分而来。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from ..dataframe import DataFrame
    from ..series import Series
from typing import Any, Optional, Union


class EWM:
    """指数加权移动窗口。

    EWM 通过衰减因子 alpha 对历史值进行加权平均，越近的值权重越大。

    Parameters
    ----------
    series : Series
        输入序列。
    alpha : float, optional
        平滑因子 (0 < alpha <= 1)。
    span : int, optional
        N 日跨度，alpha = 2/(span+1)。
    halflife : float, optional
        半衰期，alpha = 1 - exp(-log(2)/halflife)。
    com : float, optional
        质心，alpha = 1/(com+1)。
    adjust : bool, default True
        是否使用调整因子（除以权重和）。

    Examples:
        >>> s = Series([1, 2, 3, 4, 5])
        >>> s.ewm(span=3).mean().values
        [1.0, 1.5, 2.25, 3.125, 4.0625]
    """

    def __init__(
        self,
        series,
        alpha: Optional[float] = None,
        span: Optional[int] = None,
        halflife: Optional[float] = None,
        com: Optional[float] = None,
        adjust: bool = True,
    ):
        import math

        self._s = series
        self._adjust = adjust

        # 解析 alpha
        if alpha is not None:
            if not (0 < alpha <= 1):
                raise ValueError("alpha must be in (0, 1]")
            self._alpha = alpha
        elif span is not None:
            if span < 1:
                raise ValueError("span must be >= 1")
            self._alpha = 2.0 / (span + 1)
        elif halflife is not None:
            if halflife <= 0:
                raise ValueError("halflife must be > 0")
            self._alpha = 1.0 - math.exp(-math.log(2) / halflife)
        elif com is not None:
            if com < 0:
                raise ValueError("com must be >= 0")
            self._alpha = 1.0 / (com + 1)
        else:
            raise ValueError("Must provide one of: alpha, span, halflife, com")

    def _get_weights(self, n: int) -> list:
        """计算衰减权重 (w_k = (1-alpha)^(n-1-k) for k=0..n-1)。"""
        # 使用列表推导式替代显式 for 循环
        alpha = self._alpha
        return [(1 - alpha) ** (n - 1 - k) for k in range(n)]

    def mean(self):
        """指数加权均值。"""
        # 优先调用 Rust 层（仅支持非调整版 adjust=False，使用递推公式）
        if not self._adjust:
            try:
                result = self._s._inner.ewm_mean(self._alpha, 0)
                return Series(result, name=self._s.name, index=self._s._index)
            except Exception:
                pass

        # 回退到 Python 实现
        values = self._s.values
        n = len(values)

        def _mean_at(i):
            """计算第 i 位的指数加权均值。"""
            win = values[: i + 1]
            non_null = [(k, v) for k, v in enumerate(win) if v is not None]
            if not non_null:
                return None
            if self._adjust:
                # 调整版: sum(w_k * x_k) / sum(w_k)
                weights = self._get_weights(i + 1)
                num = sum(weights[k] * v for k, v in non_null)
                den = sum(weights[k] for k, _ in non_null)
                return num / den if den > 0 else None
            # 非调整版: 递推公式
            return self._mean_recursive(non_null)

        # 使用列表推导式替代显式 for 循环
        out = [_mean_at(i) for i in range(n)]
        return Series(out, name=self._s.name, index=self._s._index)

    def _mean_recursive(self, non_null):
        """非调整版 EWM 均值的递推计算。"""
        result = None
        alpha = self._alpha
        for _, v in non_null:
            if result is None:
                result = float(v)
            else:
                result = alpha * float(v) + (1 - alpha) * result
        return result

    def std(self):
        """指数加权标准差 (调整版)。"""
        values = self._s.values
        n = len(values)

        if self._adjust:

            def _std_at(i):
                win = values[: i + 1]
                non_null = [(k, v) for k, v in enumerate(win) if v is not None]
                if len(non_null) < 2:
                    return None
                weights = self._get_weights(i + 1)
                w_k = [weights[k] for k, _ in non_null]
                vals = [v for _, v in non_null]
                w_sum = sum(w_k)
                mean = sum(w * v for w, v in zip(w_k, vals)) / w_sum
                # 偏差修正: 除以 w_sum (总体), 或除以 w_sum - sum(w_k^2)/w_sum (样本)
                var = sum(w * (v - mean) ** 2 for w, v in zip(w_k, vals)) / w_sum
                return var**0.5

            # 使用列表推导式替代显式 for 循环
            out = [_std_at(i) for i in range(n)]
        else:
            # 非调整版: 使用递推公式（state-dependent，保留循环）
            out = [None] * n
            mean = None
            s2 = None
            alpha = self._alpha
            for i in range(n):
                v = values[i]
                if v is None:
                    continue

                if mean is None:
                    mean = v
                    s2 = 0.0
                else:
                    old_mean = mean
                    mean = alpha * v + (1 - alpha) * old_mean
                    s2 = (1 - alpha) * (s2 + alpha * (v - old_mean) ** 2)

                if s2 is not None and s2 >= 0:
                    out[i] = s2**0.5

        return Series(out, name=self._s.name, index=self._s._index)

    def var(self):
        """指数加权方差 (调整版)。"""
        values = self._s.values
        n = len(values)

        if self._adjust:

            def _var_at(i):
                win = values[: i + 1]
                non_null = [(k, v) for k, v in enumerate(win) if v is not None]
                if len(non_null) < 2:
                    return None
                weights = self._get_weights(i + 1)
                w_k = [weights[k] for k, _ in non_null]
                vals = [v for _, v in non_null]
                w_sum = sum(w_k)
                mean = sum(w * v for w, v in zip(w_k, vals)) / w_sum
                var = sum(w * (v - mean) ** 2 for w, v in zip(w_k, vals)) / w_sum
                return var

            # 使用列表推导式替代显式 for 循环
            out = [_var_at(i) for i in range(n)]
        else:
            mean = None
            s2 = None
            alpha = self._alpha
            for i in range(n):
                v = values[i]
                if v is None:
                    continue

                if mean is None:
                    mean = v
                    s2 = 0.0
                else:
                    old_mean = mean
                    mean = alpha * v + (1 - alpha) * old_mean
                    s2 = (1 - alpha) * (s2 + alpha * (v - old_mean) ** 2)

                if s2 is not None:
                    out[i] = s2

        return Series(out, name=self._s.name, index=self._s._index)

    # ---------- v2.0.0: corr / cov ----------

    def corr(self, other):
        """指数加权相关系数。

        :param other: 另一个 Series
        """
        cov = self.cov(other)
        var_a = self.var()
        var_b = type(self)(
            other,
            alpha=self._alpha,
            adjust=self._adjust,
        ).var()
        out = []
        for i in range(len(var_a)):
            if var_a[i] is None or var_b[i] is None or cov[i] is None:
                out.append(None)
            elif var_a[i] == 0 or var_b[i] == 0:
                out.append(None)
            else:
                out.append(cov[i] / ((var_a[i] * var_b[i]) ** 0.5))
        return Series(out, name=self._s.name, index=self._s._index)

    def cov(self, other):
        """指数加权协方差。

        :param other: 另一个 Series
        """
        values_a = self._s.values
        values_b = other.values
        if len(values_a) != len(values_b):
            raise ValueError("lengths must match")
        n = len(values_a)
        out = [None] * n

        if self._adjust:
            for i in range(n):
                wa = values_a[: i + 1]
                wb = values_b[: i + 1]
                pairs = [
                    (k, a, b)
                    for k, (a, b) in enumerate(zip(wa, wb))
                    if a is not None and b is not None
                ]
                if len(pairs) < 2:
                    continue
                weights = self._get_weights(i + 1)
                w_k = [weights[k] for k, _, _ in pairs]
                vals_a = [a for _, a, _ in pairs]
                vals_b = [b for _, _, b in pairs]
                w_sum = sum(w_k)
                ma = sum(w * a for w, a in zip(w_k, vals_a)) / w_sum
                mb = sum(w * b for w, b in zip(w_k, vals_b)) / w_sum
                cov_val = (
                    sum(w * (a - ma) * (b - mb) for w, a, b in zip(w_k, vals_a, vals_b))
                    / w_sum
                )
                out[i] = cov_val
        else:
            # 非调整版: 递推公式
            ma = None
            mb = None
            cov_val = None
            alpha = self._alpha
            for i in range(n):
                a = values_a[i]
                b = values_b[i]
                if a is None or b is None:
                    continue
                if ma is None:
                    ma = float(a)
                    mb = float(b)
                    cov_val = 0.0
                else:
                    old_ma = ma
                    old_mb = mb
                    ma = alpha * float(a) + (1 - alpha) * old_ma
                    mb = alpha * float(b) + (1 - alpha) * old_mb
                    cov_val = (1 - alpha) * (
                        cov_val + alpha * (a - old_ma) * (b - old_mb)
                    )
                if cov_val is not None:
                    out[i] = cov_val

        return Series(out, name=self._s.name, index=self._s._index)


def __getattr__(name: str):
    """模块级延迟导入，避免与 dataframe.py / series.py 的循环导入。"""
    if name == "DataFrame":
        from ..dataframe import DataFrame as _DF

        return _DF
    if name == "Series":
        from ..series import Series as _S

        return _S
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# 懒加载代理：避免循环导入
# ---------------------------------------------------------------------------
import importlib as _importlib


class _LazyProxy:
    """延迟导入代理，支持 __call__（构造）和 __getattr__（属性访问）。"""

    def __init__(self, mod_path: str, attr_name: str):
        self._mod_path = mod_path
        self._attr_name = attr_name
        self._obj = None

    def _load(self):
        if self._obj is None:
            mod = _importlib.import_module(self._mod_path)
            self._obj = getattr(mod, self._attr_name)
        return self._obj

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __instancecheck__(self, instance):
        return isinstance(instance, self._load())


# 安装懒加载代理
if "Series" not in globals() or not callable(globals().get("Series", None)):
    Series = _LazyProxy("rspandas.series", "Series")
if "DataFrame" not in globals() or not callable(globals().get("DataFrame", None)):
    DataFrame = _LazyProxy("rspandas.dataframe", "DataFrame")
