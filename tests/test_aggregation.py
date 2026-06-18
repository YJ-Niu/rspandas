"""聚合/运算集成测试。"""

import math

import pytest

import rspandas as rpd


# ---------------------------------------------------------------------------
# 数值聚合
# ---------------------------------------------------------------------------

def test_series_sum_int():
    assert rpd.Series([1, 2, 3, 4]).sum() == 10


def test_series_sum_float():
    assert rpd.Series([1.0, 2.0, 3.0]).sum() == 6.0


def test_series_sum_bool_is_count_true():
    s = rpd.Series([True, False, True, True])
    assert s.sum() == 3


def test_series_mean_int_promotes_to_float():
    s = rpd.Series([1, 2, 3])
    assert s.mean() == 2.0


def test_series_min_max_int():
    s = rpd.Series([3, 1, 4, 1, 5, 9, 2, 6])
    assert s.min() == 1
    assert s.max() == 9


def test_series_min_max_string():
    s = rpd.Series(['c', 'a', 'b'])
    assert s.min() == 'a'
    assert s.max() == 'c'


def test_series_count_excludes_null():
    s = rpd.Series([1, None, 2, None, 3])
    assert s.count() == 3


def test_series_std_var():
    s = rpd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # pandas 总体方差 = 2.0
    assert math.isclose(s.var(), 2.0, rel_tol=1e-9)
    assert math.isclose(s.std(), math.sqrt(2.0), rel_tol=1e-9)


def test_series_median_odd():
    s = rpd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s.median() == 3.0


def test_series_median_even():
    s = rpd.Series([1.0, 2.0, 3.0, 4.0])
    assert s.median() == 2.5


def test_series_aggregation_empty_returns_none():
    s = rpd.Series([])
    assert s.sum() is None
    assert s.mean() is None


def test_series_aggregation_all_null():
    s = rpd.Series([None, None, None])
    assert s.sum() is None
    assert s.mean() is None
    assert s.count() == 0


# ---------------------------------------------------------------------------
# DataFrame 级聚合
# ---------------------------------------------------------------------------

def test_df_column_sum_filtered():
    df = rpd.DataFrame({'a': [1, 2, 3, 4, 5]})
    assert df[df['a'] > 2]['a'].sum() == 12  # 3+4+5


def test_df_describe_count_column():
    df = rpd.DataFrame({'a': [1, None, 3, 4, None]})
    desc = df.describe()
    # 'count' 行 (其实是每列一个count)
    counts = list(desc['count'].values)
    assert counts == [3]


def test_df_describe_with_multiple_numeric_cols():
    df = rpd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
    desc = df.describe()
    # 6 stats + 1 unnamed index col = 7
    assert desc.shape == (2, 7)
    means = list(desc['mean'].values)
    assert means == [2.0, 20.0]


# ---------------------------------------------------------------------------
# 缺失值处理
# ---------------------------------------------------------------------------

def test_series_count_with_null():
    s = rpd.Series([1.0, None, 2.0, None, 3.0])
    assert s.count() == 3


def test_df_filter_with_null_in_data():
    df = rpd.DataFrame({'a': [1, None, 3, None, 5]})
    # a > 2: 索引 2, 4 -> True
    mask = df['a'] > 2
    f = df[mask]
    # None > 2 -> 在 Rust 端返回 False，所以过滤后剩 [3, 5]
    assert list(f['a'].values) == [3, 5]


# ---------------------------------------------------------------------------
# 字符串聚合
# ---------------------------------------------------------------------------

def test_string_series_count():
    s = rpd.Series(['a', 'b', 'c'])
    assert s.count() == 3


def test_string_series_min_max():
    s = rpd.Series(['banana', 'apple', 'cherry'])
    assert s.min() == 'apple'
    assert s.max() == 'cherry'
