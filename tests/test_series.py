"""Series 集成测试。"""

import math

import pytest

import rspandas as rpd


# ---------------------------------------------------------------------------
# 构造
# ---------------------------------------------------------------------------

def test_series_from_int_list():
    s = rpd.Series([1, 2, 3], name='a')
    assert s.shape == (3,)
    assert s.dtype == 'int64'
    assert s.name == 'a'
    assert s.values == [1, 2, 3]


def test_series_from_float_list():
    s = rpd.Series([1.0, 2.0, 3.0])
    assert s.dtype == 'float64'
    assert s.values == [1.0, 2.0, 3.0]


def test_series_from_bool_list():
    s = rpd.Series([True, False, True])
    assert s.dtype == 'bool'
    assert s.values == [True, False, True]


def test_series_from_str_list():
    s = rpd.Series(['x', 'y', 'z'])
    assert s.dtype == 'object'
    assert s.values == ['x', 'y', 'z']


def test_series_with_none():
    s = rpd.Series([1, None, 3])
    assert s.values == [1, None, 3]
    assert s.count() == 2


def test_series_empty():
    s = rpd.Series([])
    assert s.empty
    assert len(s) == 0
    assert s.shape == (0,)


# ---------------------------------------------------------------------------
# 属性
# ---------------------------------------------------------------------------

def test_series_size_nbytes_ndim():
    s = rpd.Series([1, 2, 3])
    assert s.size == 3
    assert s.ndim == 1
    assert s.nbytes > 0


def test_series_name_setter():
    s = rpd.Series([1, 2, 3], name='x')
    s.name = 'y'
    assert s.name == 'y'


def test_series_index():
    s = rpd.Series([1, 2, 3])
    assert list(s.index) == [0, 1, 2]


# ---------------------------------------------------------------------------
# 子集
# ---------------------------------------------------------------------------

def test_series_head():
    s = rpd.Series([1, 2, 3, 4, 5])
    h = s.head(3)
    assert list(h.values) == [1, 2, 3]


def test_series_tail():
    s = rpd.Series([1, 2, 3, 4, 5])
    t = s.tail(2)
    assert list(t.values) == [4, 5]


def test_series_head_more_than_len():
    s = rpd.Series([1, 2])
    h = s.head(10)
    assert list(h.values) == [1, 2]


# ---------------------------------------------------------------------------
# 索引
# ---------------------------------------------------------------------------

def test_series_getitem_int():
    s = rpd.Series([10, 20, 30])
    assert s[0] == 10
    assert s[2] == 30
    assert s[-1] == 30


def test_series_getitem_slice():
    s = rpd.Series([10, 20, 30, 40])
    sub = s[1:3]
    assert list(sub.values) == [20, 30]


def test_series_getitem_out_of_range():
    s = rpd.Series([1, 2, 3])
    with pytest.raises(IndexError):
        _ = s[5]


def test_series_iter():
    s = rpd.Series([1, 2, 3])
    assert list(iter(s)) == [1, 2, 3]


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------

def test_series_sum_int():
    s = rpd.Series([1, 2, 3])
    assert s.sum() == 6


def test_series_sum_float():
    s = rpd.Series([1.0, 2.0, 3.0])
    assert s.sum() == 6.0


def test_series_sum_with_null():
    s = rpd.Series([1, None, 3])
    assert s.sum() == 4


def test_series_mean():
    s = rpd.Series([1.0, 2.0, 3.0])
    assert s.mean() == 2.0


def test_series_mean_with_null():
    s = rpd.Series([1, None, 3])
    assert s.mean() == 2.0


def test_series_min_max():
    s = rpd.Series([3, 1, 2])
    assert s.min() == 1
    assert s.max() == 3


def test_series_count():
    s = rpd.Series([1, None, 3, None])
    assert s.count() == 2


def test_series_std_var():
    s = rpd.Series([1.0, 2.0, 3.0])
    assert math.isclose(s.std() ** 2, s.var(), rel_tol=1e-9)


def test_series_median():
    s = rpd.Series([1, 3, 2])
    assert s.median() == 2


def test_series_describe():
    s = rpd.Series([1.0, 2.0, 3.0])
    desc = s.describe()
    assert isinstance(desc, rpd.Series)
    assert desc.shape == (6,)


# ---------------------------------------------------------------------------
# 比较 -> bool Series
# ---------------------------------------------------------------------------

def test_series_eq_scalar():
    s = rpd.Series([1, 2, 3])
    mask = s == 2
    assert isinstance(mask, rpd.Series)
    assert mask.dtype == 'bool'
    assert list(mask.values) == [False, True, False]


def test_series_gt_scalar():
    s = rpd.Series([1, 2, 3])
    mask = s > 1
    assert list(mask.values) == [False, True, True]


def test_series_lt_scalar():
    s = rpd.Series([1, 2, 3])
    mask = s < 2
    assert list(mask.values) == [True, False, False]


def test_series_ne_scalar():
    s = rpd.Series([1, 2, 3])
    mask = s != 2
    assert list(mask.values) == [True, False, True]


# ---------------------------------------------------------------------------
# 过滤
# ---------------------------------------------------------------------------

def test_series_filter_mask():
    s = rpd.Series([10, 20, 30, 40])
    f = s.filter([True, False, True, False])
    assert list(f.values) == [10, 30]


def test_series_bool_indexing():
    s = rpd.Series([10, 20, 30])
    f = s[[True, False, True]]
    assert list(f.values) == [10, 30]


# ---------------------------------------------------------------------------
# 类型转换
# ---------------------------------------------------------------------------

def test_series_astype_to_float():
    s = rpd.Series([1, 2, 3])
    f = s.astype('float64')
    assert f.dtype == 'float64'
    assert f.values == [1.0, 2.0, 3.0]


def test_series_astype_to_str():
    s = rpd.Series([1, 2, 3])
    f = s.astype('object')
    assert f.dtype == 'object'
    assert f.values == ['1', '2', '3']


# ---------------------------------------------------------------------------
# 缺失值 (v0.2.0)
# ---------------------------------------------------------------------------

def test_series_isnull():
    s = rpd.Series([1, None, 3])
    m = s.isnull()
    assert m.dtype == 'bool'
    assert list(m.values) == [False, True, False]


def test_series_notnull():
    s = rpd.Series([1, None, 3])
    m = s.notnull()
    assert m.dtype == 'bool'
    assert list(m.values) == [True, False, True]


def test_series_dropna():
    s = rpd.Series([1, None, 3, None, 5])
    d = s.dropna()
    assert list(d.values) == [1, 3, 5]


def test_series_dropna_no_null():
    s = rpd.Series([1, 2, 3])
    assert list(s.dropna().values) == [1, 2, 3]


def test_series_fillna_int():
    s = rpd.Series([1, None, 3])
    f = s.fillna(0)
    assert list(f.values) == [1, 0, 3]


def test_series_fillna_string():
    s = rpd.Series(['a', None, 'c'])
    f = s.fillna('x')
    assert list(f.values) == ['a', 'x', 'c']


# ---------------------------------------------------------------------------
# 唯一值 (v0.2.0)
# ---------------------------------------------------------------------------

def test_series_unique_int():
    s = rpd.Series([1, 2, 2, 3, 1, 4])
    u = s.unique()
    assert list(u.values) == [1, 2, 3, 4]


def test_series_unique_string():
    s = rpd.Series(['a', 'b', 'a', 'c'])
    u = s.unique()
    assert list(u.values) == ['a', 'b', 'c']


def test_series_nunique():
    s = rpd.Series([1, 2, 2, 3, 1])
    assert s.nunique() == 3


def test_series_nunique_with_null():
    s = rpd.Series([1, None, 2, None, 3])
    assert s.nunique() == 3


def test_series_value_counts():
    s = rpd.Series(['a', 'b', 'a', 'c', 'b', 'a'])
    vc = s.value_counts()
    # 计数: a=3, b=2, c=1
    assert vc.shape == (3,)
    assert list(vc.values) == [3, 2, 1]


# ---------------------------------------------------------------------------
# 自定义 index
# ---------------------------------------------------------------------------

def test_series_custom_string_index():
    s = rpd.Series([10, 20, 30], index=['a', 'b', 'c'], name='nums')
    assert list(s.values) == [10, 20, 30]
    assert list(s.index) == ['a', 'b', 'c']


def test_series_getitem_str_label():
    s = rpd.Series([10, 20, 30], index=['a', 'b', 'c'])
    assert s['b'] == 20
    assert s['a'] == 10


def test_series_getitem_str_label_missing():
    s = rpd.Series([10, 20, 30], index=['a', 'b', 'c'])
    with pytest.raises(KeyError):
        _ = s['z']


def test_series_custom_int_index():
    """自定义整数 index 时，整数 key 应按 label 查找。"""
    s = rpd.Series([1, 2, 3], index=[100, 200, 300])
    assert s[200] == 2
    assert s[100] == 1


def test_series_range_index_uses_position():
    """RangeIndex 时整数 key 按位置查找。"""
    s = rpd.Series([1, 2, 3])
    assert s[0] == 1
    assert s[2] == 3


def test_series_index_in_repr():
    s = rpd.Series([10, 20, 30], index=['a', 'b', 'c'], name='nums')
    rep = repr(s)
    assert 'a' in rep
    assert 'b' in rep
    assert 'c' in rep
    assert 'nums' in rep


def test_series_index_property():
    s = rpd.Series([1, 2, 3], index=['x', 'y', 'z'])
    assert list(s.index) == ['x', 'y', 'z']


def test_series_default_index():
    s = rpd.Series([1, 2, 3])
    assert list(s.index) == [0, 1, 2]
