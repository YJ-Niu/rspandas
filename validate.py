"""plan.txt 中的 10 个验证用例。"""
import sys
sys.path.insert(0, '/Users/user/Desktop/rust_project/rspandas/python')

import rspandas as rpd

print("=== 1. Series 构造 ===")
s = rpd.Series([1, 2, 3], name='a')
assert s.shape == (3,), f"shape={s.shape}"
assert s.dtype == 'int64', f"dtype={s.dtype}"
print(f"OK: shape={s.shape}, dtype={s.dtype}")

print("=== 2. Series 聚合 ===")
s = rpd.Series([1.0, 2.0, 3.0])
m = s.mean()
assert abs(m - 2.0) < 1e-9, f"mean={m}"
print(f"OK: mean={m}")

print("=== 3. DataFrame 构造 ===")
df = rpd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
assert df.shape == (3, 2), f"shape={df.shape}"
print(f"OK: shape={df.shape}")

print("=== 4. df['a'] -> Series ===")
col = df['a']
assert isinstance(col, rpd.Series)
assert col.sum() == 6
print(f"OK: type={type(col).__name__}, sum={col.sum()}")

print("=== 5. df.head(2) ===")
h = df.head(2)
assert h.shape == (2, 2), f"shape={h.shape}"
print(f"OK: shape={h.shape}")

print("=== 6. df.tail(1) ===")
t = df.tail(1)
assert t.shape == (1, 2), f"shape={t.shape}"
print(f"OK: shape={t.shape}")

print("=== 7. df.describe() ===")
desc = df.describe()
print(f"OK: describe shape={desc.shape}")
print(desc)

print("=== 8. df.info() ===")
df.info()
print("OK")

print("=== 9. Series 缺失值 ===")
s = rpd.Series([1, None, 3])
assert s.sum() == 4, f"sum={s.sum()}"
assert s.count() == 2, f"count={s.count()}"
print(f"OK: sum={s.sum()}, count={s.count()}")

print("=== 10. DataFrame 过滤 ===")
df = rpd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
filt = df[df['a'] > 1]
assert filt.shape == (2, 2), f"shape={filt.shape}"
print(f"OK: shape={filt.shape}")
print(filt)

print("\n*** ALL 10 CASES PASSED ***")
