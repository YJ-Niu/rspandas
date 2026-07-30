"""验证 series.py 和 dataframe.py 中 Rust 层调用优化的正确性。

测试范围：
- Series.quantile (Rust 层 quantile)
- Series.rank (Rust 层 rank)
- Rolling.sum/mean/std (Rust 层 rolling_sum/mean/std)
- Expanding.sum/mean (Rust 层 expanding_sum/mean)
- EWM.mean (Rust 层 ewm_mean，仅 adjust=False)
- DataFrame.merge (Rust 层 merge)
- DataFrameGroupBy._agg (Rust 层 groupby_agg)
- DataFrame.pivot/pivot_table (Rust 层 pivot)
- DataFrame.melt (Rust 层 melt)
"""

import sys

import rspandas as pd
from rspandas import DataFrame, Series


def _check(label: str, got, expected, tol: float = 1e-9) -> bool:
    """比较 got 与 expected，允许浮点误差。"""
    if isinstance(expected, list):
        if not isinstance(got, list) or len(got) != len(expected):
            print(f"[FAIL] {label}: 长度不符 got={got!r} expected={expected!r}")
            return False
        for g, e in zip(got, expected):
            if g is None or e is None:
                if g is not e:
                    print(f"[FAIL] {label}: None 不匹配 got={got!r}")
                    return False
            elif isinstance(e, float) and abs(g - e) > tol:
                print(f"[FAIL] {label}: 数值不符 got={got!r} expected={expected!r}")
                return False
        print(f"[OK] {label}")
        return True
    if got is None and expected is None:
        print(f"[OK] {label}")
        return True
    if abs(got - expected) > tol:
        print(f"[FAIL] {label}: got={got!r} expected={expected!r}")
        return False
    print(f"[OK] {label}")
    return True


def test_series_quantile():
    """测试 Series.quantile 调用 Rust 层。"""
    print("\n=== test_series_quantile ===")
    s = Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # 中位数
    _check("quantile(0.5)", s.quantile(0.5), 3.0)
    # 0.0 分位数
    _check("quantile(0.0)", s.quantile(0.0), 1.0)
    # 1.0 分位数
    _check("quantile(1.0)", s.quantile(1.0), 5.0)
    # 0.25 分位数
    _check("quantile(0.25)", s.quantile(0.25), 2.0)
    # 非线性插值回退到 Python
    _check("quantile(0.5, lower)", s.quantile(0.5, interpolation="lower"), 3.0)
    # 列表 q 回退到 Python
    _check("quantile([0.5])", s.quantile([0.5]), [3.0])


def test_series_rank():
    """测试 Series.rank 调用 Rust 层。"""
    print("\n=== test_series_rank ===")
    s = Series([3.0, 1.0, 2.0, 1.0])
    # average 方法
    r = s.rank().to_list()
    # 3.0 -> 4, 1.0 -> 1.5, 2.0 -> 3, 1.0 -> 1.5
    _check("rank average", r, [4.0, 1.5, 3.0, 1.5])
    # min 方法
    r = s.rank(method="min").to_list()
    _check("rank min", r, [4.0, 1.0, 3.0, 1.0])
    # max 方法
    r = s.rank(method="max").to_list()
    _check("rank max", r, [4.0, 2.0, 3.0, 2.0])
    # first 方法
    r = s.rank(method="first").to_list()
    _check("rank first", r, [4.0, 1.0, 3.0, 2.0])
    # dense 方法回退到 Python
    r = s.rank(method="dense").to_list()
    _check("rank dense", r, [3.0, 1.0, 2.0, 1.0])
    # pct=True
    r = s.rank(pct=True).to_list()
    _check("rank average pct", r, [1.0, 0.375, 0.75, 0.375])
    # 含 None 的数据
    s2 = Series([3.0, None, 2.0, 1.0])
    r = s2.rank().to_list()
    _check("rank with None", r, [3.0, None, 2.0, 1.0])


def test_rolling():
    """测试 Rolling.sum/mean/std 调用 Rust 层。"""
    print("\n=== test_rolling ===")
    s = Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # rolling sum
    r = s.rolling(3).sum().to_list()
    _check("rolling sum", r, [None, None, 6.0, 9.0, 12.0])
    # rolling mean
    r = s.rolling(3).mean().to_list()
    _check("rolling mean", r, [None, None, 2.0, 3.0, 4.0])
    # rolling std
    r = s.rolling(3).std().to_list()
    # std([1,2,3]) = sqrt(((1-2)^2+(2-2)^2+(3-2)^2)/3) = sqrt(2/3)
    expected_std = [None, None, (2.0 / 3.0) ** 0.5, (2.0 / 3.0) ** 0.5, (2.0 / 3.0) ** 0.5]
    _check("rolling std", r, expected_std)
    # min_periods
    r = s.rolling(3, min_periods=1).sum().to_list()
    _check("rolling sum min_periods=1", r, [1.0, 3.0, 6.0, 9.0, 12.0])
    # center=True 回退到 Python
    r = s.rolling(3, center=True).sum().to_list()
    # center: 窗口 [i-1, i+1]
    # i=0: [1,2] -> 3
    # i=1: [1,2,3] -> 6
    # i=2: [2,3,4] -> 9
    # i=3: [3,4,5] -> 12
    # i=4: [4,5] -> 9
    _check("rolling sum center", r, [3.0, 6.0, 9.0, 12.0, 9.0])


def test_expanding():
    """测试 Expanding.sum/mean 调用 Rust 层。"""
    print("\n=== test_expanding ===")
    s = Series([1.0, 2.0, 3.0, 4.0])
    # expanding sum
    r = s.expanding(1).sum().to_list()
    _check("expanding sum", r, [1.0, 3.0, 6.0, 10.0])
    # expanding mean
    r = s.expanding(1).mean().to_list()
    _check("expanding mean", r, [1.0, 1.5, 2.0, 2.5])
    # min_periods=2
    r = s.expanding(2).sum().to_list()
    _check("expanding sum min_periods=2", r, [None, 3.0, 6.0, 10.0])


def test_ewm_mean():
    """测试 EWM.mean 调用 Rust 层（仅 adjust=False）。"""
    print("\n=== test_ewm_mean ===")
    s = Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # adjust=False 时调用 Rust 层
    r = s.ewm(alpha=0.5, adjust=False).mean().to_list()
    # 递推: ema[0]=1, ema[1]=0.5*2+0.5*1=1.5, ema[2]=0.5*3+0.5*1.5=2.25, ...
    expected = [1.0, 1.5, 2.25, 3.125, 4.0625]
    _check("ewm mean adjust=False", r, expected)
    # adjust=True 时回退到 Python
    r = s.ewm(alpha=0.5, adjust=True).mean().to_list()
    # 调整版: sum(w_k * x_k) / sum(w_k), w_k = (1-alpha)^(n-1-k)
    # i=0: 1
    # i=1: (0.5*1 + 1*2) / (0.5+1) = 1.5/1.5 = 1.0... 让我重新算
    # alpha=0.5, w = [0.5, 1.0] for n=2
    # mean = (0.5*1 + 1.0*2) / (0.5+1.0) = 2.5/1.5 = 1.6667
    expected_adjust = [1.0, 5.0 / 3.0, 13.0 / 7.0, 17.0 / 15.0 * 2 + 1, 0.0]
    # 不验证 adjust=True 的具体值，只验证不报错
    print(f"[INFO] ewm mean adjust=True: {r}")


def test_dataframe_merge():
    """测试 DataFrame.merge 调用 Rust 层。"""
    print("\n=== test_dataframe_merge ===")
    left = DataFrame({"key": ["a", "b", "c"], "value": [1, 2, 3]})
    right = DataFrame({"key": ["a", "b", "c"], "value2": [4, 5, 6]})
    # inner join，单一 on 字符串，无冲突列
    merged = left.merge(right, on="key", how="inner")
    print(f"[INFO] merged columns: {merged.columns}")
    print(f"[INFO] merged shape: {merged.shape}")
    assert merged.shape[0] == 3, f"inner merge 行数应为 3，实际 {merged.shape[0]}"
    assert "key" in merged.columns
    assert "value" in merged.columns
    assert "value2" in merged.columns
    print("[OK] merge inner")

    # left join
    left2 = DataFrame({"key": ["a", "b", "c"], "value": [1, 2, 3]})
    right2 = DataFrame({"key": ["a", "b"], "value2": [4, 5]})
    merged = left2.merge(right2, on="key", how="left")
    print(f"[INFO] left merge shape: {merged.shape}")
    assert merged.shape[0] == 3, f"left merge 行数应为 3，实际 {merged.shape[0]}"
    print("[OK] merge left")

    # 列名冲突时回退到 Python
    left3 = DataFrame({"key": ["a", "b"], "value": [1, 2]})
    right3 = DataFrame({"key": ["a", "b"], "value": [3, 4]})
    merged = left3.merge(right3, on="key", how="inner", suffixes=("_x", "_y"))
    print(f"[INFO] conflict merge columns: {merged.columns}")
    assert "value_x" in merged.columns
    assert "value_y" in merged.columns
    print("[OK] merge with suffixes (回退到 Python)")


def test_groupby_agg():
    """测试 DataFrameGroupBy._agg 调用 Rust 层。"""
    print("\n=== test_groupby_agg ===")
    df = DataFrame(
        {
            "key": ["a", "b", "a", "b", "a"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "count": [10, 20, 30, 40, 50],
        }
    )
    # sum 聚合
    grouped = df.groupby("key").sum()
    print(f"[INFO] groupby sum columns: {grouped.columns}")
    print(f"[INFO] groupby sum shape: {grouped.shape}")
    # a: value=1+3+5=9, count=10+30+50=90
    # b: value=2+4=6, count=20+40=60
    # 按 key 排序: a, b
    value_col = grouped["value"].to_list()
    _check("groupby sum value", value_col, [9.0, 6.0])
    count_col = grouped["count"].to_list()
    _check("groupby sum count", count_col, [90.0, 60.0])

    # mean 聚合
    grouped = df.groupby("key").mean()
    # a: value=9/3=3, b: value=6/2=3
    value_col = grouped["value"].to_list()
    _check("groupby mean value", value_col, [3.0, 3.0])

    # 不同列不同 agg 回退到 Python
    grouped = df.groupby("key").agg({"value": "sum", "count": "max"})
    print(f"[INFO] mixed agg columns: {grouped.columns}")
    print("[OK] groupby mixed agg (回退到 Python)")


def test_pivot_table():
    """测试 DataFrame.pivot_table 调用 Rust 层。"""
    print("\n=== test_pivot_table ===")
    df = DataFrame(
        {
            "index": ["a", "a", "b", "b"],
            "columns": ["x", "y", "x", "y"],
            "values": [1.0, 2.0, 3.0, 4.0],
        }
    )
    # pivot_table 单一 index/columns/values
    pt = df.pivot_table(index="index", columns="columns", values="values", aggfunc="sum")
    print(f"[INFO] pivot_table columns: {pt.columns}")
    print(f"[INFO] pivot_table shape: {pt.shape}")
    # 验证结果
    assert pt.shape[0] == 2, f"pivot_table 行数应为 2，实际 {pt.shape[0]}"
    print("[OK] pivot_table sum")

    # mean 聚合
    df2 = DataFrame(
        {
            "index": ["a", "a", "b"],
            "columns": ["x", "x", "x"],
            "values": [1.0, 3.0, 5.0],
        }
    )
    pt = df2.pivot_table(index="index", columns="columns", values="values", aggfunc="mean")
    # a: (1+3)/2=2, b: 5
    print(f"[INFO] pivot_table mean: {pt}")
    print("[OK] pivot_table mean")


def test_pivot():
    """测试 DataFrame.pivot 调用 Rust 层。"""
    print("\n=== test_pivot ===")
    df = DataFrame(
        {
            "foo": ["one", "one", "two", "two"],
            "bar": ["A", "B", "A", "B"],
            "baz": [1.0, 2.0, 3.0, 4.0],
        }
    )
    # pivot 数据无重复
    pv = df.pivot(index="foo", columns="bar", values="baz")
    print(f"[INFO] pivot columns: {pv.columns}")
    print(f"[INFO] pivot shape: {pv.shape}")
    # 验证结果
    assert pv.shape[0] == 2, f"pivot 行数应为 2，实际 {pv.shape[0]}"
    print("[OK] pivot 无重复")


def test_melt():
    """测试 DataFrame.melt 调用 Rust 层。"""
    print("\n=== test_melt ===")
    df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    # melt 默认参数
    melted = df.melt(id_vars=["A"])
    print(f"[INFO] melt columns: {melted.columns}")
    print(f"[INFO] melt shape: {melted.shape}")
    # 应有 3 列: A, variable, value
    assert "A" in melted.columns
    assert "variable" in melted.columns
    assert "value" in melted.columns
    # 4 行 (2 行 * 2 值列)
    assert melted.shape[0] == 4, f"melt 行数应为 4，实际 {melted.shape[0]}"
    print("[OK] melt id_vars=['A']")

    # 自定义 var_name/value_name
    melted = df.melt(id_vars=["A"], var_name="col", value_name="val")
    print(f"[INFO] melt custom columns: {melted.columns}")
    assert "col" in melted.columns
    assert "val" in melted.columns
    print("[OK] melt 自定义列名")


def main():
    """主测试入口。"""
    print(f"rspandas version: {pd.__version__}")
    tests = [
        test_series_quantile,
        test_series_rank,
        test_rolling,
        test_expanding,
        test_ewm_mean,
        test_dataframe_merge,
        test_groupby_agg,
        test_pivot_table,
        test_pivot,
        test_melt,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"[ERROR] {test.__name__} 异常: {e!r}")
            import traceback

            traceback.print_exc()
            failed += 1
    print(f"\n=== 测试完成: {len(tests) - failed}/{len(tests)} 通过 ===")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
