"""DataFrame 集成测试。"""

import pytest

import rspandas as rpd


# ---------------------------------------------------------------------------
# 构造
# ---------------------------------------------------------------------------

def test_dataframe_from_dict():
    df = rpd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    assert df.shape == (3, 2)
    assert df.columns == ['a', 'b']


def test_dataframe_from_list_of_dict():
    df = rpd.DataFrame([{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}])
    assert df.shape == (2, 2)
    assert set(df.columns) == {'a', 'b'}


def test_dataframe_from_list_of_list():
    df = rpd.DataFrame([[1, 'x'], [2, 'y']], columns=['a', 'b'])
    assert df.shape == (2, 2)
    assert df.columns == ['a', 'b']


def test_dataframe_with_none():
    df = rpd.DataFrame({'a': [1, None, 3]})
    assert df['a'].values == [1, None, 3]


def test_dataframe_empty():
    df = rpd.DataFrame({})
    assert df.empty


def test_dataframe_shape_mismatch():
    with pytest.raises(ValueError):
        rpd.DataFrame({'a': [1, 2], 'b': [1]})


# ---------------------------------------------------------------------------
# 属性
# ---------------------------------------------------------------------------

def test_dataframe_dtypes():
    df = rpd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    dtypes = df.dtypes
    assert dtypes == {'a': 'int64', 'b': 'object'}


def test_dataframe_columns_setter():
    df = rpd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    df.columns = ['c1', 'c2']
    assert df.columns == ['c1', 'c2']


def test_dataframe_columns_setter_length_mismatch():
    df = rpd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    with pytest.raises(ValueError):
        df.columns = ['c1']


def test_dataframe_index():
    df = rpd.DataFrame({'a': [1, 2, 3]})
    assert list(df.index) == [0, 1, 2]


def test_dataframe_size_ndim():
    df = rpd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    assert df.size == 4
    assert df.ndim == 2


def test_dataframe_contains():
    df = rpd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    assert 'a' in df
    assert 'c' not in df


# ---------------------------------------------------------------------------
# 子集
# ---------------------------------------------------------------------------

def test_dataframe_getitem_str():
    df = rpd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    s = df['a']
    assert isinstance(s, rpd.Series)
    assert list(s.values) == [1, 2, 3]


def test_dataframe_getitem_list_of_str():
    df = rpd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
    sub = df[['a', 'c']]
    assert sub.shape == (2, 2)
    assert sub.columns == ['a', 'c']


def test_dataframe_getitem_int():
    df = rpd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    row = df[0]
    assert row == {'a': 1, 'b': 'x'}


def test_dataframe_getitem_slice():
    df = rpd.DataFrame({'a': [1, 2, 3, 4]})
    sub = df[1:3]
    assert sub.shape == (2, 1)
    assert list(sub['a'].values) == [2, 3]


def test_dataframe_getitem_bool_mask():
    df = rpd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    mask = df['a'] > 1
    f = df[mask]
    assert f.shape == (2, 2)
    assert list(f['a'].values) == [2, 3]


def test_dataframe_head_tail():
    df = rpd.DataFrame({'a': list(range(10))})
    h = df.head(3)
    t = df.tail(2)
    assert h.shape == (3, 1)
    assert t.shape == (2, 1)
    assert list(t['a'].values) == [8, 9]


def test_dataframe_filter_rows():
    df = rpd.DataFrame({'a': [1, 2, 3, 4]})
    f = df.filter_rows([True, False, True, False])
    assert list(f['a'].values) == [1, 3]


# ---------------------------------------------------------------------------
# 列操作
# ---------------------------------------------------------------------------

def test_dataframe_setitem_new_col():
    df = rpd.DataFrame({'a': [1, 2, 3]})
    df['b'] = [10, 20, 30]
    assert df.shape == (3, 2)
    assert 'b' in df


def test_dataframe_setitem_update_col():
    df = rpd.DataFrame({'a': [1, 2, 3]})
    df['a'] = [10, 20, 30]
    assert list(df['a'].values) == [10, 20, 30]


def test_dataframe_setitem_wrong_length():
    df = rpd.DataFrame({'a': [1, 2, 3]})
    with pytest.raises(ValueError):
        df['b'] = [1, 2]


# ---------------------------------------------------------------------------
# 概览
# ---------------------------------------------------------------------------

def test_dataframe_info(capsys):
    df = rpd.DataFrame({'a': [1, 2, 3], 'b': ['x', None, 'z']})
    df.info()
    captured = capsys.readouterr()
    assert 'DataFrame' in captured.out
    assert 'a' in captured.out
    assert 'b' in captured.out


def test_dataframe_describe():
    df = rpd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    desc = df.describe()
    assert isinstance(desc, rpd.DataFrame)
    # 6 stats + 1 unnamed index column = 7 columns
    assert desc.shape == (2, 7)


def test_dataframe_values():
    df = rpd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    vals = df.values
    assert vals == [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}]


# ---------------------------------------------------------------------------
# 缺失值 (v0.2.0)
# ---------------------------------------------------------------------------

def test_dataframe_dropna():
    df = rpd.DataFrame({'a': [1, None, 3, 4], 'b': [10, 20, None, 40]})
    d = df.dropna()
    # 行 0, 3 不含 None
    assert d.shape == (2, 2)
    assert list(d['a'].values) == [1, 4]
    assert list(d['b'].values) == [10, 40]


def test_dataframe_dropna_no_null():
    df = rpd.DataFrame({'a': [1, 2, 3]})
    d = df.dropna()
    assert d.shape == (3, 1)


def test_dataframe_fillna_scalar():
    """标量 fillna 只对匹配 dtype 的列生效，不匹配的列保持原样。"""
    df = rpd.DataFrame({'a': [1, None, 3]})
    f = df.fillna(0)
    assert f.shape == (3, 1)
    assert list(f['a'].values) == [1, 0, 3]


def test_dataframe_fillna_dict():
    df = rpd.DataFrame({'a': [1, None, 3], 'b': [10, 20, None]})
    f = df.fillna({'a': 99, 'b': 0})
    assert list(f['a'].values) == [1, 99, 3]
    assert list(f['b'].values) == [10, 20, 0]


def test_dataframe_fillna_dict_partial():
    """只指定部分列，未指定列保持原样。"""
    df = rpd.DataFrame({'a': [1, None, 3], 'b': [10, None, 30]})
    f = df.fillna({'a': 0})
    assert list(f['a'].values) == [1, 0, 3]
    assert list(f['b'].values) == [10, None, 30]


# ---------------------------------------------------------------------------
# loc / iloc 索引器 (v0.2.0)
# ---------------------------------------------------------------------------

def test_loc_single_row():
    df = rpd.DataFrame({'a': [10, 20, 30], 'b': ['x', 'y', 'z']})
    row = df.loc[1]
    assert row.shape == (1, 2)
    assert list(row['a'].values) == [20]
    assert list(row['b'].values) == ['y']


def test_loc_slice_inclusive():
    """loc 切片是双闭区间。"""
    df = rpd.DataFrame({'a': [10, 20, 30, 40, 50]})
    sub = df.loc[1:3]
    assert list(sub['a'].values) == [20, 30, 40]


def test_loc_fancy_index():
    df = rpd.DataFrame({'a': [10, 20, 30, 40, 50]})
    sub = df.loc[[0, 2, 4]]
    assert list(sub['a'].values) == [10, 30, 50]


def test_loc_bool_mask():
    df = rpd.DataFrame({'a': [10, 20, 30, 40, 50]})
    f = df.loc[df['a'] > 20]
    assert list(f['a'].values) == [30, 40, 50]


def test_loc_column_str():
    df = rpd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    s = df.loc[:, 'a']
    assert isinstance(s, rpd.Series)
    assert list(s.values) == [1, 2, 3]


def test_loc_column_list():
    df = rpd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    sub = df.loc[:, ['a']]
    assert sub.shape == (3, 1)
    assert sub.columns == ['a']


def test_loc_row_and_col():
    df = rpd.DataFrame({'a': [10, 20, 30, 40], 'b': [1, 2, 3, 4]})
    sub = df.loc[1:2, 'a']
    assert isinstance(sub, rpd.Series)
    assert list(sub.values) == [20, 30]


def test_loc_fancy_row_and_col():
    df = rpd.DataFrame({'a': [10, 20, 30], 'b': [1, 2, 3], 'c': ['x', 'y', 'z']})
    sub = df.loc[[0, 2], ['a', 'b']]
    assert sub.shape == (2, 2)
    assert list(sub['a'].values) == [10, 30]
    assert list(sub['b'].values) == [1, 3]


def test_iloc_single_row():
    df = rpd.DataFrame({'a': [10, 20, 30]})
    row = df.iloc[1]
    assert list(row['a'].values) == [20]


def test_iloc_slice_exclusive():
    """iloc 切片是右开区间 (Python 风格)。"""
    df = rpd.DataFrame({'a': [10, 20, 30, 40, 50]})
    sub = df.iloc[1:3]
    assert list(sub['a'].values) == [20, 30]


def test_iloc_fancy_index():
    df = rpd.DataFrame({'a': [10, 20, 30, 40, 50]})
    sub = df.iloc[[0, 2, 4]]
    assert list(sub['a'].values) == [10, 30, 50]


def test_iloc_negative_index():
    df = rpd.DataFrame({'a': [10, 20, 30]})
    assert list(df.iloc[-1]['a'].values) == [30]
    assert list(df.iloc[-2]['a'].values) == [20]


def test_iloc_column_int():
    df = rpd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    s = df.iloc[:, 0]
    assert isinstance(s, rpd.Series)
    assert list(s.values) == [1, 2, 3]


def test_iloc_column_slice():
    df = rpd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
    sub = df.iloc[:, 0:2]
    assert sub.shape == (2, 2)
    assert sub.columns == ['a', 'b']


def test_iloc_row_and_col():
    df = rpd.DataFrame({'a': [10, 20, 30], 'b': [1, 2, 3]})
    sub = df.iloc[0:2, 0:1]
    assert sub.shape == (2, 1)
    assert sub.columns == ['a']
    assert list(sub['a'].values) == [10, 20]


def test_iloc_fancy_row_and_col():
    df = rpd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    sub = df.iloc[[0, 2], [1]]
    assert sub.shape == (2, 1)
    assert list(sub['b'].values) == [4, 6]
