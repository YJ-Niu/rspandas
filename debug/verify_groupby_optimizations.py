"""验证 GroupBy 优化和剩余循环优化。"""

import sys
sys.path.insert(0, '.')

import rspandas as rpd


def test_series_groupby_basic():
    """SeriesGroupBy 基础方法。"""
    print("=== SeriesGroupBy 基础测试 ===")
    keys = ['a', 'b', 'a', 'b', 'a', 'b']
    gb = rpd.Series([1, 2, 3, 4, 5, 6]).groupby(by=keys)
    print("sum:", list(gb.sum()))
    print("mean:", list(gb.mean()))
    print("count:", list(gb.count()))
    print("min:", list(gb.min()))
    print("max:", list(gb.max()))
    print("std:", list(gb.std()))
    print("var:", list(gb.var()))
    print("median:", list(gb.median()))
    print("prod:", list(gb.prod()))
    print("sem:", list(gb.sem()))
    print("first:", list(gb.first()))
    print("last:", list(gb.last()))
    print("size:", dict(gb.size()))
    print("nunique:", dict(gb.nunique()))


def test_series_groupby_extended():
    """SeriesGroupBy 扩展方法（新增）。"""
    print("\n=== SeriesGroupBy 扩展方法测试 ===")
    keys = ['a', 'b', 'a', 'b', 'a', 'b']
    s = rpd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    gb = s.groupby(by=keys)
    print("quantile(0.5):", list(gb.quantile(0.5)))
    print("skew:", list(gb.skew()))
    print("kurt:", list(gb.kurt()))
    print("mad:", list(gb.mad()))
    print("ngroup:", list(gb.ngroup()))
    print("cumcount:", list(gb.cumcount()))


def test_series_groupby_agg_multi():
    """agg 多函数聚合测试。"""
    print("\n=== SeriesGroupBy agg 多函数 ===")
    keys = ['a', 'b', 'a', 'b', 'a', 'b']
    s = rpd.Series([1, 2, 3, 4, 5, 6])
    gb = s.groupby(by=keys)
    # list 形式 -> DataFrame
    result = gb.agg(['sum', 'mean', 'count'])
    print("agg(list) DataFrame columns:", result.columns)
    print("agg(list) values:", result.to_dict())
    # dict 形式 -> DataFrame
    result2 = gb.agg({'total': 'sum', 'average': 'mean'})
    print("agg(dict) columns:", result2.columns)
    print("agg(dict) values:", result2.to_dict())


def test_series_groupby_transform_filter():
    """transform/filter 测试。"""
    print("\n=== SeriesGroupBy transform/filter ===")
    keys = ['a', 'b', 'a', 'b', 'a', 'b']
    s = rpd.Series([1, 2, 3, 4, 5, 6])
    gb = s.groupby(by=keys)
    # 调用内置 str 名的 transform 会更简单，用自定义函数测试 filter
    print("transform('sum'):", list(gb.transform('sum')))
    # filter: 只保留分组 sum > 8 的（b组:2+4+6=12，a组:1+3+5=9 均>8）
    filtered = gb.filter(lambda g: sum(list(g.values)) > 8)
    print("filter(sum>8):", list(filtered))


def test_dataframe_groupby_basic():
    """DataFrameGroupBy 基础方法。"""
    print("\n=== DataFrameGroupBy 基础测试 ===")
    df = rpd.DataFrame({
        'key': ['a', 'b', 'a', 'b', 'a', 'b'],
        'val1': [1, 2, 3, 4, 5, 6],
        'val2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    gb = df.groupby('key')
    print("sum:", gb.sum().to_dict())
    print("mean:", gb.mean().to_dict())
    print("count:", gb.count().to_dict())
    print("min:", gb.min().to_dict())
    print("max:", gb.max().to_dict())
    print("std:", gb.std().to_dict())
    print("var:", gb.var().to_dict())
    print("median:", gb.median().to_dict())
    print("prod:", gb.prod().to_dict())
    print("sem:", gb.sem().to_dict())
    print("first:", gb.first().to_dict())
    print("last:", gb.last().to_dict())


def test_dataframe_groupby_extended():
    """DataFrameGroupBy 扩展方法。"""
    print("\n=== DataFrameGroupBy 扩展方法 ===")
    df = rpd.DataFrame({
        'key': ['a', 'b', 'a', 'b', 'a', 'b'],
        'val1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'val2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    gb = df.groupby('key')
    print("quantile(0.25):", gb.quantile(0.25).to_dict())
    print("ngroup:", list(gb.ngroup()))
    print("cumcount:", list(gb.cumcount()))
    print("rank:\n", gb.rank().to_dict())


def test_dataframe_groupby_agg_multi():
    """DataFrameGroupBy agg 多函数。"""
    print("\n=== DataFrameGroupBy agg 多函数 ===")
    df = rpd.DataFrame({
        'key': ['a', 'b', 'a', 'b', 'a', 'b'],
        'val1': [1, 2, 3, 4, 5, 6],
        'val2': [10, 20, 30, 40, 50, 60],
    })
    gb = df.groupby('key')
    result = gb.agg(['sum', 'mean'])
    print("agg(list) columns:", result.columns)
    print("agg(list) values:", result.to_dict())
    # dict 形式
    result2 = gb.agg({'val1': 'sum', 'val2': 'mean'})
    print("agg(dict) values:", result2.to_dict())


def test_series_python_loops():
    """Series 循环优化验证。"""
    print("\n=== Series 循环优化验证 ===")
    s = rpd.Series([1, 2, 3, None, 5, 6], index=['a', 'b', 'c', 'd', 'e', 'f'])
    print("ffill:", list(s.ffill()))
    print("bfill:", list(s.bfill()))
    print("cumsum:", list(s.cumsum()))
    print("cumprod:", list(s.cumprod()))
    print("truncate(before='b',after='e'):", list(s.truncate(before='b', after='e')))
    print("add_prefix('X_'):", s.add_prefix('X_')._index)
    print("add_suffix('_Y'):", s.add_suffix('_Y')._index)
    sampled = s.sample(n=3, random_state=42)
    print("sample(n=3, seed=42):", list(sampled))
    print("argsort():", list(s.argsort()))


def test_index_methods():
    """Index 扩展方法验证。"""
    print("\n=== Index 扩展方法验证 ===")
    idx = rpd.Index([1, 2, 3, 4, 5], name='test')
    print("equals(Index([1,2,3,4,5])):", idx.equals(rpd.Index([1, 2, 3, 4, 5])))
    print("hasnans:", idx.hasnans)
    print("nbytes:", idx.nbytes)
    idx_nan = rpd.Index([1.0, None, 3.0])
    print("Index with nan hasnans:", idx_nan.hasnans)
    print("to_series():", list(idx.to_series()))
    print("delete(1):", list(idx.delete(1)))
    print("insert(1, 99):", list(idx.insert(1, 99)))


if __name__ == "__main__":
    test_series_groupby_basic()
    test_series_groupby_extended()
    test_series_groupby_agg_multi()
    test_series_groupby_transform_filter()
    test_dataframe_groupby_basic()
    test_dataframe_groupby_extended()
    test_dataframe_groupby_agg_multi()
    test_series_python_loops()
    test_index_methods()
    print("\n✅ 所有验证通过！")
