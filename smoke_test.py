"""Smoke test for rspandas v0.1.0 MVP."""
import sys
sys.path.insert(0, '/Users/user/Desktop/rust_project/rspandas/python')

import rspandas as rpd

# 1. Series 构造
s = rpd.Series([1, 2, 3], name='a')
print('=== Series basic ===')
print('shape:', s.shape, 'dtype:', s.dtype, 'sum:', s.sum(), 'mean:', s.mean())
print('values:', s.values)
print('repr:', repr(s))
print()

# 2. Series 缺失值
s2 = rpd.Series([1, None, 3])
print('=== Series with None ===')
print('sum:', s2.sum(), 'count:', s2.count(), 'mean:', s2.mean())
print()

# 3. DataFrame 构造
df = rpd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
print('=== DataFrame ===')
print('shape:', df.shape, 'cols:', df.columns, 'dtypes:', df.dtypes)
print('df[a]:', repr(df['a']))
print('---repr---')
print(repr(df))
print('---repr head(2)---')
print(repr(df.head(2)))
print('---repr tail(1)---')
print(repr(df.tail(1)))
print()

# 4. info & describe
df.info()
print('---describe---')
print(repr(df.describe()))
print()

# 5. 过滤
print('=== filter (a > 1) ===')
mask = df['a'] > 1
print('mask:', mask)
print('filtered:', repr(df[mask]))
