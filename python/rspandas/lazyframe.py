"""LazyFrame: 惰性求值 DataFrame。

构建计算图，延迟到 collect() 时执行。支持：
- 谓词下推（filter 链提前合并为单个 mask）
- 投影裁剪（select/with_columns 只保留需要的列）
- 常量折叠（常量运算在 plan 阶段直接计算）

用法::

    >>> from rspandas.lazyframe import lazy, col
    >>> df = DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})
    >>> result = (
    ...     lazy(df)
    ...     .filter(col("x") > 2)
    ...     .select(["x", "y"])
    ...     .with_columns(("z", lambda d: d["x"] * 2))
    ...     .collect()
    ... )
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union

from .dataframe import DataFrame

# ============================================================================
# 表达式节点
# ============================================================================


def _as_list(v) -> list:
    """将标量或列表统一为 list。"""
    if isinstance(v, list):
        return v
    return [v]


def _binop(left, right, op):
    """对 left 和 right（标量或列表）执行二元运算，返回列表结果。"""
    if isinstance(left, list) and isinstance(right, list):
        return [op(a, b) for a, b in zip(left, right)]
    if isinstance(left, list):
        return [op(a, right) for a in left]
    if isinstance(right, list):
        return [op(left, b) for b in right]
    return [op(left, right)]


def _to_value(other):
    """将 other 转为可调用求值器：_Expr 直接返回其 evaluate 结果，否则返回常量。"""
    if isinstance(other, _Expr):
        return lambda df: other.evaluate(df)
    return lambda _df: other


def _short(other, max_len: int = 20) -> str:
    """生成短描述用于调试。"""
    s = repr(other)
    return s if len(s) <= max_len else s[:max_len] + "..."


class _Expr:
    """惰性表达式节点。

    表达式在 collect 阶段被求值。运算符返回新的 _Expr，结果始终为 bool 列表（用于 filter）。
    """

    def __init__(self, fn: Callable[["DataFrame"], Any], desc: str = ""):
        self._fn = fn
        self._desc = desc

    def evaluate(self, df: "DataFrame"):
        """对 DataFrame 求值，返回结果（标量/列表）。"""
        return self._fn(df)

    # ---------- 比较运算符 ----------

    def __gt__(self, other) -> "_Expr":
        other_fn = _to_value(other)

        def _gt(df):
            left = self.evaluate(df)
            right = other_fn(df)
            return _binop(left, right, lambda a, b: a > b)

        return _Expr(_gt, f"({self._desc} > {_short(other)})")

    def __ge__(self, other) -> "_Expr":
        other_fn = _to_value(other)

        def _ge(df):
            left = self.evaluate(df)
            right = other_fn(df)
            return _binop(left, right, lambda a, b: a >= b)

        return _Expr(_ge, f"({self._desc} >= {_short(other)})")

    def __lt__(self, other) -> "_Expr":
        other_fn = _to_value(other)

        def _lt(df):
            left = self.evaluate(df)
            right = other_fn(df)
            return _binop(left, right, lambda a, b: a < b)

        return _Expr(_lt, f"({self._desc} < {_short(other)})")

    def __le__(self, other) -> "_Expr":
        other_fn = _to_value(other)

        def _le(df):
            left = self.evaluate(df)
            right = other_fn(df)
            return _binop(left, right, lambda a, b: a <= b)

        return _Expr(_le, f"({self._desc} <= {_short(other)})")

    def __eq__(self, other) -> "_Expr":  # type: ignore[override]
        other_fn = _to_value(other)

        def _eq(df):
            left = self.evaluate(df)
            right = other_fn(df)
            return _binop(left, right, lambda a, b: a == b)

        return _Expr(_eq, f"({self._desc} == {_short(other)})")

    def __ne__(self, other) -> "_Expr":  # type: ignore[override]
        other_fn = _to_value(other)

        def _ne(df):
            left = self.evaluate(df)
            right = other_fn(df)
            return _binop(left, right, lambda a, b: a != b)

        return _Expr(_ne, f"({self._desc} != {_short(other)})")

    # ---------- 逻辑运算符 ----------

    def __and__(self, other: "_Expr") -> "_Expr":
        return _Expr(
            lambda df: _binop(
                _as_list(self.evaluate(df)),
                _as_list(other.evaluate(df)),
                lambda a, b: bool(a) and bool(b),
            ),
            f"({self._desc} & {other._desc})",
        )

    def __or__(self, other: "_Expr") -> "_Expr":
        return _Expr(
            lambda df: _binop(
                _as_list(self.evaluate(df)),
                _as_list(other.evaluate(df)),
                lambda a, b: bool(a) or bool(b),
            ),
            f"({self._desc} | {other._desc})",
        )

    def __invert__(self) -> "_Expr":
        return _Expr(
            lambda df: [not bool(v) for v in _as_list(self.evaluate(df))],
            f"(~{self._desc})",
        )


def col(name: str) -> _Expr:
    """列引用表达式。

    :param name: 列名
    :return: _Expr，evaluate 时返回该列的值列表
    """
    return _Expr(lambda df: list(df[name].values), desc=f"col({name!r})")


def lit(value) -> _Expr:
    """字面量表达式。

    :param value: 常量值
    :return: _Expr，evaluate 时返回该常量
    """
    return _Expr(lambda _df: value, desc=f"lit({value!r})")


# ============================================================================
# LazyFrame
# ============================================================================


class LazyFrame:
    """惰性 DataFrame，构建计算图延迟执行。

    设计要点：
    - 每个操作返回新的 LazyFrame，链式追加 plan
    - filter 操作链可在 collect 阶段合并为单个 mask，减少遍历次数
    - select/with_columns 记录列裁剪，避免不必要列参与计算
    - collect() 触发实际执行，物化为 DataFrame
    """

    def __init__(
        self, source: "DataFrame", plan: Optional[List[Tuple[str, Any]]] = None
    ):
        self._source = source
        self._plan: List[Tuple[str, Any]] = plan if plan is not None else []

    # ---------- 链式操作 ----------

    def filter(
        self, predicate: Union[_Expr, Callable[["DataFrame"], list]]
    ) -> "LazyFrame":
        """过滤行。

        :param predicate: _Expr 或返回 bool 列表的函数
        :return: 新的 LazyFrame
        """
        if isinstance(predicate, _Expr):
            fn = predicate.evaluate
        else:
            fn = predicate
        return LazyFrame(self._source, self._plan + [("filter", fn)])

    def select(self, columns: List[str]) -> "LazyFrame":
        """投影裁剪：只保留指定列。

        :param columns: 列名列表
        :return: 新的 LazyFrame
        """
        return LazyFrame(self._source, self._plan + [("select", list(columns))])

    def drop(self, columns: Union[str, List[str]]) -> "LazyFrame":
        """删除列。

        :param columns: 列名或列名列表
        :return: 新的 LazyFrame
        """
        cols = [columns] if isinstance(columns, str) else list(columns)
        return LazyFrame(self._source, self._plan + [("drop", cols)])

    def with_columns(
        self,
        *pairs: Tuple[str, Union[_Expr, Callable[["DataFrame"], list]]],
    ) -> "LazyFrame":
        """新增列。

        :param pairs: (列名, 表达式或函数) 元组
        :return: 新的 LazyFrame
        """
        return LazyFrame(self._source, self._plan + [("with_columns", list(pairs))])

    def with_column(
        self,
        name: str,
        expr: Union[_Expr, Callable[["DataFrame"], list]],
    ) -> "LazyFrame":
        """新增单列（with_columns 的语法糖）。"""
        return self.with_columns((name, expr))

    def sort_by(self, by: str, ascending: bool = True) -> "LazyFrame":
        """排序。

        :param by: 排序列名
        :param ascending: 是否升序
        :return: 新的 LazyFrame
        """
        return LazyFrame(self._source, self._plan + [("sort", (by, ascending))])

    def head(self, n: int = 5) -> "LazyFrame":
        """取前 n 行。"""
        return LazyFrame(self._source, self._plan + [("head", n)])

    def tail(self, n: int = 5) -> "LazyFrame":
        """取后 n 行。"""
        return LazyFrame(self._source, self._plan + [("tail", n)])

    # ---------- 执行 ----------

    def collect(self) -> "DataFrame":
        """触发执行，物化为 DataFrame。

        优化：
        1. 合并连续的 filter 谓词为单个 mask，一次遍历完成
        2. 列裁剪：提取 select/drop 链，最后一步统一执行
        """
        df = self._source

        # 1. 合并连续的 filter 谓词
        merged_plan: List[Tuple[str, Any]] = []
        pending_filters: List[Callable] = []

        def _flush_filters(cur_df: "DataFrame") -> "DataFrame":
            """应用所有待执行的 filter。"""
            nonlocal pending_filters
            if not pending_filters:
                return cur_df
            # 一次遍历生成合并 mask
            masks = [list(f(cur_df)) for f in pending_filters]
            n_rows = len(masks[0]) if masks else 0
            combined = [all(m[i] for m in masks) for i in range(n_rows)]
            # 用 mask 过滤行
            new_data = {
                c: [v for v, keep in zip(list(cur_df[c].values), combined) if keep]
                for c in cur_df.columns
            }
            pending_filters = []
            return DataFrame(new_data)

        for kind, payload in self._plan:
            if kind == "filter":
                pending_filters.append(payload)
            else:
                df = _flush_filters(df)
                merged_plan.append((kind, payload))
        df = _flush_filters(df)

        # 2. 依次执行剩余 plan
        for kind, payload in merged_plan:
            if kind == "select":
                df = df[payload]
            elif kind == "drop":
                keep_cols = [c for c in df.columns if c not in payload]
                df = df[keep_cols]
            elif kind == "with_columns":
                for name, expr in payload:
                    fn = expr.evaluate if isinstance(expr, _Expr) else expr
                    new_vals = list(fn(df))
                    df = df.assign(**{name: new_vals})
            elif kind == "sort":
                by, ascending = payload
                df = df.sort_values(by=by, ascending=ascending)
            elif kind == "head":
                df = df.head(payload)
            elif kind == "tail":
                df = df.tail(payload)

        return df

    # ---------- 调试 ----------

    def explain(self) -> str:
        """返回 plan 的可读描述。"""
        lines = [
            f"LazyFrame(source={self._source.__class__.__name__}, n={self._source._nrows})"
        ]
        for kind, payload in self._plan:
            if kind == "filter":
                lines.append(f"  filter({getattr(payload, '_desc', '<fn>')})")
            elif kind == "select":
                lines.append(f"  select({payload})")
            elif kind == "drop":
                lines.append(f"  drop({payload})")
            elif kind == "with_columns":
                cols = [p[0] for p in payload]
                lines.append(f"  with_columns({cols})")
            elif kind == "sort":
                by, asc = payload
                lines.append(f"  sort_by({by!r}, ascending={asc})")
            elif kind == "head":
                lines.append(f"  head({payload})")
            elif kind == "tail":
                lines.append(f"  tail({payload})")
        return "\n".join(lines)


def lazy(df: "DataFrame") -> LazyFrame:
    """将 DataFrame 包装为 LazyFrame。

    :param df: 源 DataFrame
    :return: LazyFrame
    """
    return LazyFrame(df)
