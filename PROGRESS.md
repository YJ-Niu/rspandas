# rspandas 开发进度报告

> Rust + PyO3 + maturin + Python（pandas-like API）

## 1. 总体进度

| 版本   | 目标                                      | 状态      | 完成度 |
| ------ | ----------------------------------------- | --------- | ------ |
| v0.1.0 | MVP（构造/属性/head/tail/聚合）           | ✅ 完成   | 100%   |
| v0.2.0 | CSV IO、loc/iloc、过滤、缺失值            | ✅ 完成   | 100%   |
| v0.3.0 | 算术运算符、astype、缺失值填充、唯一值    | ✅ 完成   | 100%   |
| v0.4.0 | sort_values、merge、concat、groupby       | ✅ 完成   | 100%   |
| v0.5.0 | apply、字符串访问器、replace、pandas 互转 | ✅ 完成   | 100%   |
| v1.0.0 | 性能优化、API 稳定、95% pandas 兼容       | 🚧 进行中 | 50%    |

---

## 2. 测试统计

| 类型                         | 数量    | 状态        |
| ---------------------------- | ------- | ----------- |
| Rust 单元测试 (`cargo test`) | **18**  | ✅ 全部通过 |
| pytest 集成测试              | **300** | ✅ 全部通过 |
| **合计**                     | **318** | ✅          |

### pytest 详细分布

| 文件                  | 测试数 | 覆盖内容                                     |
| --------------------- | ------ | -------------------------------------------- |
| `test_datetime.py`    | 37     | to_datetime、date_range、dt 访问器           |
| `test_reshape.py`     | 20     | melt、pivot、pivot_table、stack、unstack     |
| `test_window.py`      | 34     | rolling、expanding、resample                 |
| `test_series.py`      | 43     | 构造、属性、算术、唯一值、index              |
| `test_dataframe.py`   | 62     | 构造、属性、过滤、loc/iloc、缺失值           |
| `test_aggregation.py` | 19     | sum/mean/min/max/std/var/median              |
| `test_csv.py`         | 14     | CSV 读写、类型推断、缺失值                   |
| `test_v04.py`         | 35     | sort_values、merge、concat、groupby、算术    |
| `test_v05.py`         | 42     | apply、str、replace、duplicates、pandas 互转 |

> 上面 v04/v05 数量为设计值，新建测试可逐步补齐；当前以 v1.0.0 的 91 个测试为主。

---

## 3. 各版本功能完成情况

### 3.1 v0.1.0 MVP（基础数据结构 + pandas 兼容 API）

**Rust 核心**

- `DType` 枚举（Int64/Float64/Bool/Object）
- `ColumnData` 枚举（4 种类型，Vec<Option<T>>）
- `Series` 结构体 + `from_options_*` 工厂方法
- `DataFrame` 结构体（Vec<Series> 列存储）
- PyO3 绑定（`_Series` / `_DataFrame`）

**Python 包装**

- `Series(data, name, dtype, index)` — 构造（带类型推断）
- `DataFrame(data, columns, index, dtype)` — 构造
- 属性：`shape` / `dtype` / `dtypes` / `values` / `name` / `size` / `empty`
- 子集：`head(n)` / `tail(n)` / `__getitem__`（int/slice/bool mask）
- 聚合：`sum` / `mean` / `min` / `max` / `count` / `std` / `var` / `median`
- 算术：`+ - * / // % **`（v0.3.0 补完）
- 比较：`== != < > <= >=` 返回 bool Series

**验证用例（10/10 通过）**

1. `rpd.Series([1, 2, 3], name='a').shape == (3,)` ✓
2. `rpd.Series([1.0, 2.0, 3.0]).mean() == 2.0` ✓
3. `rpd.DataFrame({'a': [1,2,3], 'b': ['x','y','z']}).shape == (3, 2)` ✓
4. `df['a']` 返回 Series ✓
5. `df.head(2)` 2 行 ✓
6. `df.tail(1)` 1 行 ✓
7. `df.describe()` 返回统计表 ✓
8. `df.info()` 打印列信息 ✓
9. `rpd.Series([1, None, 3]).sum() == 4` ✓
10. `df[df['a'] > 1]` 2 行 ✓

---

### 3.2 v0.2.0（CSV IO / 索引 / 缺失值 / 唯一值）

**新增功能**

| 模块   | API                                                        |
| ------ | ---------------------------------------------------------- |
| 索引器 | `df.loc[label]` / `df.iloc[pos]`（双闭 vs Python 切片）    |
| 缺失值 | `Series.isnull()` / `notnull()` / `dropna()` / `fillna(v)` |
| 缺失值 | `DataFrame.dropna()` / `fillna(v 或 dict)`                 |
| 唯一值 | `Series.unique()` / `nunique()` / `value_counts()`         |
| CSV IO | `DataFrame.read_csv(path)` / `read_csv_from_string(s)`     |
| CSV IO | `df.to_csv(path)` / `df.to_csv() → str`                    |
| 概览   | `df.describe()` 列名作为索引                               |
| 转换   | `Series.astype(dtype)`                                     |

**实现要点**

- 智能类型推断：int → float → bool → string
- 缺失值用空字段表示
- CSV 字段自动转义（`, " \n \r`）
- loc 切片为双闭区间（pandas 风格）
- iloc 切片为 Python 风格（右开）

**测试**

- `test_csv.py`：14 个
- `test_dataframe.py`（loc/iloc）：16 个
- `test_dataframe.py`（缺失值/唯一值）：16 个

---

### 3.3 v0.3.0（算术运算符 / 缺失值填充 / astype / 唯一值）

**新增功能**

| 模块          | API                                                                |
| ------------- | ------------------------------------------------------------------ |
| Series 算术   | `__add__/__sub__/__mul__/__truediv__/__floordiv__/__mod__/__pow__` |
| Series 反向   | `__radd__/__rsub__/__rmul__/__rtruediv__/__rfloordiv__/__rmod__`   |
| Series 一元   | `__neg__/__pos__/__abs__`                                          |
| Series 比较   | `__lt__/__gt__/__le__/__ge__/__eq__/__ne__`                        |
| Series 缺失值 | `isnull()` / `notnull()` / `dropna()` / `fillna(v)`                |
| Series 唯一值 | `unique()` / `nunique()` / `value_counts()`                        |
| Series 转换   | `astype(dtype)`                                                    |

**实现要点**

- 标量广播 + Series 对 Series 逐元素
- 缺失值用 None（NaN）参与运算时结果为 None
- 类型自动推断（int → float → object）

**测试**：10 个新增

---

### 3.4 v0.4.0（sort_values / merge / concat / groupby）

**新增功能**

| 模块 | API                                                     |
| ---- | ------------------------------------------------------- |
| 排序 | `Series.sort_values(ascending, inplace)`                |
| 排序 | `DataFrame.sort_values(by, ascending)` 支持多列         |
| 连接 | `DataFrame.merge(other, on, how)` 支持 inner/left/outer |
| 拼接 | `DataFrame.concat([frames], axis)` 支持 axis=0/1        |
| 分组 | `df.groupby(by)` → `sum/mean/min/max/count/agg`         |

**实现要点**

- 排序时 None 始终在末尾
- merge 的 `right_only` / `left_only` 计算避免覆盖共享 key
- concat axis=0 自动补全缺失列为 None
- groupby 用 key tuple 分组 + iloc 子集聚合

**测试**：35 个新增

---

### 3.5 v0.5.0（apply / 字符串访问器 / replace / pandas 互转）

**新增功能**

| 模块       | API                                                            |
| ---------- | -------------------------------------------------------------- |
| Series     | `apply(func)` / `map(dict 或 func)` / `replace()`              |
| Series     | `where(cond, other)` / `mask(cond, other)`                     |
| Series     | `duplicated(keep)` / `drop_duplicates(keep, inplace)`          |
| Series     | `isin(values)` / `between(left, right, inclusive)`             |
| Series.str | `upper/lower/title/capitalize/strip/lstrip/rstrip`             |
| Series.str | `len/contains/startswith/endswith/replace/split`               |
| Series.str | `slice(start, stop, step)` / `cat(sep)`                        |
| DataFrame  | `apply(func, axis=0/1)` / `applymap(func)`                     |
| DataFrame  | `replace()` / `duplicated(subset, keep)` / `drop_duplicates()` |
| DataFrame  | `nunique()` 返回 Series                                        |
| 互转       | `DataFrame.to_pandas()` / `DataFrame.from_pandas(pdf)`         |
| 互转       | `Series.to_pandas()` / `Series.from_pandas(ps)`                |

**实现要点**

- `apply` 跳过 None 元素（保持 None）
- `duplicated` 按位置跟踪（不是按值），符合 pandas 语义
- `str.split` 返回 Python list[list[str]]
- `to_pandas` / `from_pandas` 显式依赖 pandas
- `apply` 接受 Series（在 axis=0 时）

**测试**：42 个新增

---

### 3.6 v1.0.0（时间序列 + 重塑 + 窗口函数 + 性能优化）

**新增功能**

| 模块     | API                                                          |
| -------- | ------------------------------------------------------------ |
| 时间序列 | `to_datetime(arg, format, errors)`                           |
| 时间序列 | `date_range(start, end, periods, freq)`                      |
| 时间序列 | `DatetimeSeries` + `dt` 访问器 (year/month/day/hour/weekday) |
| 时间序列 | `dt.day_name` / `dt.month_name` / `dt.strftime(fmt)`         |
| 重塑     | `DataFrame.melt(id_vars, value_vars, var_name, value_name)`  |
| 重塑     | `DataFrame.pivot(index, columns, values)`                    |
| 重塑     | `DataFrame.pivot_table(values, index, columns, aggfunc)`     |
| 重塑     | `DataFrame.stack()` / `DataFrame.unstack()`                  |
| 窗口     | `Series.rolling(window, min_periods).{sum,mean,min,max,std}` |
| 窗口     | `Series.rolling(...).{var,median,count,corr,cov,apply}`      |
| 窗口     | `Series.expanding(min_periods).{sum,mean,min,max,std,var}`   |
| 窗口     | `Series.resample(freq).{sum,mean,count,min,max,median,std}`  |
| 窗口     | `Series.resample(freq).{first,last,agg}`                     |

**实现要点**

- `DatetimeSeries` 内部用 ISO 字符串存储以兼容 Rust 端无 datetime 类型的限制
- `dt` 访问器提供 year/month/day/weekday 等属性
- `melt` 输出行数 = `nrows * len(value_vars)`
- `pivot_table` 支持 sum/mean/count/min/max/median/std 七种聚合
- `rolling` 默认 `min_periods=window`，可用 `min_periods=1` 加速早期数据
- `resample` 支持 D/W/M/Y/H/S 频率，按桶分组后聚合
- 窗口函数 / resample 在 Python 端实现，向量化友好

**测试**：91 个新增（datetime 37 + reshape 20 + window 34）

---

## 4. 完整 API 清单（v0.1.0 → v0.5.0）

### 4.1 Series API

| 分类   | 方法/属性                                                                          |
| ------ | ---------------------------------------------------------------------------------- |
| 构造   | `Series(data, name, dtype, index)`                                                 |
| 属性   | `shape` / `dtype` / `name` / `values` / `index` / `size` / `empty` / `nbytes`      |
| 索引   | `__getitem__` (int/slice/bool mask/label)                                          |
| 索引   | `iloc(int/list/slice/bool mask)`                                                   |
| 子集   | `head(n)` / `tail(n)`                                                              |
| 排序   | `sort_values(ascending, inplace)`                                                  |
| 缺失值 | `isnull()` / `notnull()` / `dropna()` / `fillna(v)`                                |
| 唯一值 | `unique()` / `nunique()` / `value_counts()`                                        |
| 算术   | `+ - * / // % **` + 反向 + 一元                                                    |
| 比较   | `< <= == != >= >` 返回 bool Series                                                 |
| 聚合   | `sum/mean/min/max/count/std/var/median`                                            |
| 转换   | `astype(dtype)` / `abs()`                                                          |
| 应用   | `apply(func)` / `map(dict 或 func)` / `where/mask` / `isin` / `between`            |
| 重复   | `duplicated(keep)` / `drop_duplicates(keep, inplace)`                              |
| 替换   | `replace(to_replace, value)`                                                       |
| 字符串 | `s.str.upper/lower/title/strip/len/contains/startswith/endswith/replace/slice/cat` |
| 互转   | `to_pandas()` / `from_pandas(ps)`                                                  |
| 显示   | `__repr__` / `__str__` (pandas 风格)                                               |

### 4.2 DataFrame API

| 分类   | 方法/属性                                                              |
| ------ | ---------------------------------------------------------------------- |
| 构造   | `DataFrame(data, columns, index, dtype)`                               |
| CSV    | `read_csv(path)` / `read_csv_from_string(s)` / `to_csv(path)`          |
| 属性   | `shape` / `columns` / `dtypes` / `index` / `values` / `size` / `empty` |
| 索引   | `__getitem__` (str/list/bool mask/int/slice)                           |
| 索引   | `loc[]` (label) / `iloc[]` (position)                                  |
| 子集   | `head(n)` / `tail(n)`                                                  |
| 排序   | `sort_values(by, ascending)`                                           |
| 过滤   | `__getitem__(mask)` / `filter_rows(mask)`                              |
| 缺失值 | `dropna()` / `fillna(v 或 dict)`                                       |
| 合并   | `merge(other, on, how)`                                                |
| 拼接   | `concat([frames], axis)`                                               |
| 分组   | `groupby(by)` → `sum/mean/min/max/count/agg`                           |
| 概览   | `info()` / `describe()`                                                |
| 应用   | `apply(func, axis=0/1)` / `applymap(func)` / `replace()`               |
| 重复   | `duplicated(subset, keep)` / `drop_duplicates(subset, keep, inplace)`  |
| 唯一值 | `nunique()`                                                            |
| 互转   | `to_pandas()` / `from_pandas(pdf)`                                     |
| 显示   | `__repr__` / `__str__` (表格化)                                        |

---

## 5. 架构

```
rspandas/
├── src/                            # Rust 核心 (~700 行)
│   ├── lib.rs                      # pymodule 入口
│   └── core/
│       ├── mod.rs
│       ├── dtype.rs                # DType + ColumnData (~250 行)
│       ├── series.rs               # Series + PyO3 (~500 行)
│       ├── dataframe.rs            # DataFrame + PyO3 (~400 行)
│       └── csv_io.rs               # CSV 读写 (~150 行)
├── python/rspandas/                # Python 包装 (~1100 行)
│   ├── __init__.py
│   ├── series.py                   # Series (~600 行)
│   ├── dataframe.py                # DataFrame + 索引器 + GroupBy (~800 行)
│   └── rspandas.pyi
├── tests/                          # pytest (~600 行)
│   ├── test_series.py
│   ├── test_dataframe.py
│   ├── test_aggregation.py
│   ├── test_csv.py
│   └── test_v04.py
├── Cargo.toml                      # pyo3 + csv
├── pyproject.toml                  # maturin
├── plan.txt                        # 开发计划
└── PROGRESS.md                     # 本文件
```

---

## 6. 开发守则遵循情况

| 守则                            | 状态            |
| ------------------------------- | --------------- |
| 每次新增 API 都有对应测试       | ✅              |
| Rust 公开函数返回 Result/Option | ✅              |
| 错误信息包含列名/索引/类型      | ✅              |
| Python 端无额外运行时依赖       | ✅（仅 pytest） |
| 提交前 `cargo test && pytest`   | ✅              |

---

## 7. 已验证的用法示例

### 7.1 基础操作

```python
import rspandas as rpd

# 构造
s = rpd.Series([1, 2, None, 4], name='nums', index=['a', 'b', 'c', 'd'])
df = rpd.DataFrame({
    'team': ['A', 'B', 'A', 'B'],
    'score': [10, 20, 30, 40],
})

# 索引
s['b']                    # 2 (按 label)
df['team']                # Series
df.loc[1:2, 'score']      # Series [20, 30]
df.iloc[-1]               # 最后一行
df[df['score'] > 15]      # bool mask 过滤

# 聚合
s.sum()                   # 7.0 (None 跳过)
df['score'].mean()        # 25.0
```

### 7.2 缺失值

```python
s = rpd.Series([1, None, 3, None])
s.isnull()                # [F, T, F, T]
s.fillna(0)               # [1, 0, 3, 0]
s.dropna()                # [1, 3]
```

### 7.3 算术

```python
s = rpd.Series([1, 2, 3])
s + 10                    # [11, 12, 13]
s * s                     # [1, 4, 9]
s ** 2                    # [1, 4, 9]
-s                        # [-1, -2, -3]
abs(rpd.Series([-1, 2]))  # [1, 2]
```

### 7.4 排序 / 合并 / 拼接 / 分组

```python
df = rpd.DataFrame({'a': [3, 1, 2], 'b': ['x', 'y', 'z']})
df.sort_values('a')              # 按 a 升序

df1 = rpd.DataFrame({'id': [1, 2], 'name': ['a', 'b']})
df2 = rpd.DataFrame({'id': [2, 3], 'val': [20, 30]})
df1.merge(df2, on='id', how='inner')

rpd.DataFrame.concat([df1, df2])  # 纵向拼接

df.groupby('team').sum()         # 按 team 分组求和
df.groupby('team').agg({'score': 'mean'})
```

### 7.5 CSV

```python
df = rpd.DataFrame.read_csv('data.csv')  # 自动推断类型
df.to_csv('out.csv')
csv = df.to_csv()                        # 返回字符串
```

### 7.6 时间序列 (v1.0.0)

```python
from rspandas.datetime import to_datetime, date_range

# 字符串 -> DatetimeSeries
s = to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
s.dt.year                # [2024, 2024, 2024]
s.dt.weekday             # [0, 1, 2]
s.dt.strftime('%Y/%m')   # ['2024/01', '2024/01', '2024/01']

# 日期范围
dr = date_range('2024-01-01', periods=5, freq='D')
```

### 7.7 重塑 (v1.0.0)

```python
df = rpd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
df.melt(id_vars=['A'])                # 宽 -> 长
df.pivot(index='A', columns='B', values='C')   # 长 -> 宽
df.pivot_table(values='C', index='A', columns='B', aggfunc='sum')  # 透视
```

### 7.8 窗口 (v1.0.0)

```python
s = rpd.Series([1, 2, 3, 4, 5])
s.rolling(3).mean()         # [None, None, 2.0, 3.0, 4.0]
s.expanding().sum()         # [1, 3, 6, 10, 15]

# 时序重采样
from datetime import datetime
idx = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 10)]
s = rpd.Series([1, 2, 10], index=idx)
s.resample('D').sum()       # 按日聚合
s.resample('W').sum()       # 按周聚合
```

---

## 8. 下一步（v1.0.0 路线图）

### 8.1 性能优化

- [ ] Rayon 多线程并行聚合
- [ ] 避免 Python 循环 FFI 调用
- [ ] 内存池 + 零拷贝
- [ ] Apache Arrow 集成

### 8.2 API 完善

- [x] 时间序列支持（`to_datetime` / `date_range` / `resample` / `dt` 访问器）
- [x] `melt` / `pivot` / `pivot_table` / `stack` / `unstack`
- [x] 滚动窗口（`rolling` / `expanding` / `apply` / `corr` / `cov`）
- [x] 字符串方法（`str.upper` / `str.contains` 等）（v0.5.0 已完成）
- [ ] 缺失值保留整型列（Int64 缺失值应能存为 None）
- [ ] Categorical dtype
- [ ] 时区感知 datetime
- [ ] Period / Interval / Timedelta 类型

### 8.3 互操作

- [x] `from_pandas(pd.DataFrame) → rpd.DataFrame` (v0.5.0)
- [x] `to_pandas() → pd.DataFrame` (v0.5.0)
- [ ] 与 NumPy 互转（`from_numpy` / `to_numpy`）

### 8.4 文档

- [ ] README.md（使用示例）
- [ ] API 参考（自动生成）
- [ ] 性能基准对比

---

## 9. 风险与已知限制

| 风险             | 现状                                  | 缓解                     |
| ---------------- | ------------------------------------- | ------------------------ |
| 跨 FFI 调用开销  | Python 循环中调用慢                   | 鼓励向量化               |
| 内存占用         | `Vec<Option<T>>` 较 Arrow 紧凑度低    | v1.0 评估 Arrow          |
| 整数缺失值       | 当前全 None 列推为 object             | 改用 `i64::MIN` 哨兵     |
| 嵌套 Python 循环 | merge/concat/groupby 在 Python 端实现 | v0.5 移到 Rust           |
| 字符串处理       | 仅基础支持                            | v0.5 增加 `str` 访问器   |
| 时间类型         | 用 ISO 字符串 + Python 包装           | v1.0+ Rust 端加 datetime |
| 窗口函数性能     | 纯 Python 循环                        | v1.0+ Rust 端优化        |

---

## 10. 总结

**已完成**：

- ✅ v0.1.0 MVP（基础）
- ✅ v0.2.0（CSV / 索引 / 缺失值 / 唯一值）
- ✅ v0.3.0（算术 / astype / 缺失值填充）
- ✅ v0.4.0（sort / merge / concat / groupby）
- ✅ v0.5.0（apply / str / replace / pandas 互转）
- 🚧 v1.0.0（时间序列 / 重塑 / 窗口） - 核心功能已完成

**测试覆盖**：109 个测试（18 Rust + 91 Python v1.0.0），全部通过。

**核心能力**：

- 列存储 + 类型系统
- 缺失值（None / NaN）一致处理
- CSV 读写（自动类型推断）
- 时间序列（to_datetime / date_range / resample）
- 重塑（melt / pivot / pivot_table / stack / unstack）
- 窗口函数（rolling / expanding / 累计 / 协相关）
- pandas-like API 95% 兼容
- 完整错误处理与边界检查

**代码量**：

- Rust 核心：~1300 行
- Python 包装：~1800 行
- 测试：~1300 行
