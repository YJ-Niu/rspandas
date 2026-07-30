# Release Notes

## v2.0.6 (2026-07-30)

### 新增功能

#### Series 新增方法

- `copy()` — 复制 Series
- `drop()` — 删除指定索引元素
- `dropna()` — 删除缺失值（支持 `inplace`, `subset` 参数）
- `isna()` / `notna()` — 缺失值检测
- `nlargest()` / `nsmallest()` — 获取最大/最小 N 个元素
- `droplevel()` — 删除多级索引级别
- `reindex()` / `reindex_like()` — 重新索引
- `sort_index()` — 按索引排序
- `sem()` — 标准误差
- `tolist()` — 转换为 Python 列表
- `clip()` — 数值裁剪
- `compare()` — 比较 Series
- `transform()` — 变换操作
- `compress()` — 压缩
- `swaplevel()` — 交换索引级别
- `rename()` / `rename_axis()` — 重命名
- `ffill()` / `bfill()` / `interpolate()` — 缺失值填充
- `prod()` — 乘积
- `dot()` — 点积
- `autocorr()` — 自相关系数
- `round()` — 四舍五入
- `reset_index()` — 重置索引
- `pop()` — 弹出元素
- `keys()` / `items()` — 迭代方法
- `first_valid_index()` / `last_valid_index()` — 有效索引
- `truncate()` — 截断
- `add_prefix()` / `add_suffix()` — 添加前缀/后缀
- `squeeze()` — 压缩维度
- `sample()` — 随机采样
- `argsort()` — 排序索引
- 支持 `dt` 访问器（DatetimeAccessor）
- 支持反向算术运算符（`__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__` 等）

#### DataFrame 新增方法

- `map()` — 元素级映射
- `abs()` — 绝对值
- `copy()` — 复制 DataFrame
- `isna()` / `notna()` — 缺失值检测
- `nlargest()` / `nsmallest()` — 获取最大/最小 N 行
- `corr()` / `cov()` / `corrwith()` — 相关性/协方差计算
- `sort_index()` — 按索引排序
- `reindex()` / `reindex_like()` — 重新索引
- `T` — 转置属性
- `axes` / `nbytes` / `style` — 新属性
- 统计方法：`sum()`, `mean()`, `min()`, `max()`, `count()`, `std()`, `var()`, `median()`, `any()`, `all()`, `sem()`
- 转换方法：`to_dict()`, `to_string()`, `to_html()`, `to_latex()`, `to_markdown()`, `itertuples()`, `to_records()`
- 操作方法：`droplevel()`, `swaplevel()`, `join()`, `merge()`（完整参数支持）
- `ffill()` / `bfill()` / `interpolate()` — 缺失值填充
- `prod()` — 乘积
- `round()` — 四舍五入
- `dot()` — 矩阵点积
- `items()` / `iterrows()` / `keys()` — 迭代方法
- `sample()` — 随机采样
- `squeeze()` — 压缩维度
- `align()` — 对齐
- `combine_first()` — 合并
- `update()` — 更新
- `add_prefix()` / `add_suffix()` — 添加前缀/后缀
- 支持 `dtypes` 属性

#### 顶层函数

- `merge()` — DataFrame 合并（完整参数支持）
- `concat()` — DataFrame 拼接
- `isnull()` / `notnull()` / `isna()` / `notna()` — 缺失值检测
- `unique()` — 唯一值
- `value_counts()` — 值计数

### 修复问题

- **修复 `DataFrame.sort_values()` 的 `na_position` 参数行为反转问题**：`na_position='first'` 和 `na_position='last'` 的行为现在与 pandas 一致
- **修复 `rsnumpy.ndarray` 兼容性问题**：
  - 0 维数组 `abs()` 现在返回 Python 标量，解决 `math.log10()` 类型错误
  - DataFrame 构造函数现在正确处理 0D、1D、2D `rsnumpy.ndarray` 输入
- **修复 `DataFrameGroupBy` 的 `as_index=True` 行为**：聚合结果现在正确设置分组列为索引
- **修复 Series 反向运算符实现**：`__rsub__`, `__rtruediv__` 等现在正确处理类型转换

### 性能优化

- **优化 Python 循环性能**：
  - Series：`shift()`, `diff()`, `pct_change()`, `prod()`, `cumsum()`, `cumprod()`, `cummax()`, `cummin()` 使用切片和 `itertools.accumulate` 优化
  - DataFrame：`cumsum()`, `cumprod()`, `cummax()`, `cummin()`, `diff()`, `pct_change()`, `ffill()`, `bfill()`, `interpolate()`, `prod()` 使用列表推导和 `math.prod` 优化

### 构建与开发

- **新增 flake8 检查**：`build_wheel.sh` 现在包含 `uv run flake8 python/ --max-line-length=500 --extend-ignore=E203` 检查
- **Python 版本要求更新**：从 3.9 提升到 3.10
- **Rust edition 更新**：使用 Rust 2024 edition
- **添加 `black` 格式化检查**：CI 执行 `black --check --target-version py313 python/`

### API 兼容性改进

- 所有新增方法均与 pandas API 签名和默认值保持一致
- `DataFrame.drop()` 支持 `index`, `columns`, `level`, `inplace`, `errors` 参数
- `DataFrame.rename()` 支持 `index`, `columns`, `axis`, `copy`, `inplace`, `level`, `errors` 参数
- `DataFrame.reset_index()` 支持 `level`, `inplace`, `col_level`, `col_fill` 参数
- `Series.drop()` 支持 `index`, `columns`, `level`, `inplace` 参数
- `Series.dropna()` 支持 `inplace`, `subset` 参数

---

## 历史版本

### v2.0.5 及更早版本

- 初始版本，提供基础 Series、DataFrame、GroupBy、I/O 功能
- CSV/Excel 原生 Rust 读写支持
- 时间序列处理（datetime、timedelta、period）
- 索引类型（Index、RangeIndex、MultiIndex）
- 滚动窗口、扩展窗口、指数加权移动平均
- 采样器（Resampler）