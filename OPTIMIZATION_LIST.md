# rspandas 优化清单

> 本文档记录 rspandas 的完整开发与优化任务，已完成的打 ✓，待完成的打 ☐。

---

## 一、薄 Python 层 + 完整方法参数

### 目标

Python 层应保持"薄"——仅作为 API 接口层，核心计算逻辑下沉到 Rust 层。同时确保所有方法的参数签名、默认值与 pandas 完全一致。

### 原则

- Python 层方法只做：参数校验 → 调用 Rust 层 → 包装返回结果
- 禁止在 Python 层实现复杂计算逻辑（统计、排序、聚合等）
- 所有公开方法的参数名、默认值、类型注解必须与 pandas 对齐
- 对 pandas 不存在但 rspandas 扩展的方法，参数设计应参考 pandas 风格
- rsnumpy 为必选依赖，Python 层通过 `import rsnumpy as rnp` 直接引用
- `to_numpy()` / `from_numpy()` 使用 rsnumpy.ndarray 而非 numpy.ndarray

### 检查清单

#### 1.1 Series — Python 层瘦身

| 任务                                     | 状态 | 说明                                                                      |
| ---------------------------------------- | ---- | ------------------------------------------------------------------------- |
| 统计方法下沉 Rust                        | ☐    | sum/mean/std/var/min/max/median/sem/prod/skew/kurt 等统计计算移至 Rust 层 |
| 排序方法下沉 Rust                        | ☐    | sort_values/sort_index/rank/argsort 移至 Rust 层                          |
| 缺失值处理下沉 Rust                      | ☐    | isna/notna/fillna/ffill/bfill/interpolate/dropna 移至 Rust 层             |
| 窗口计算下沉 Rust                        | ☐    | Rolling/Expanding/EWM/Resampler 的计算逻辑移至 Rust 层                    |
| 字符串操作下沉 Rust                      | ☐    | StringAccessor 的所有方法移至 Rust 层（利用 Rust regex 引擎）             |
| 日期时间处理下沉 Rust                    | ☐    | DatetimeAccessor 的属性和方法移至 Rust 层                                 |
| 分组聚合下沉 Rust                        | ☐    | SeriesGroupBy 的 agg/apply/transform/sum/mean 等移至 Rust 层              |
| 插值/采样下沉 Rust                       | ☐    | interpolate/sample 移至 Rust 层                                           |
| 比较运算命名方法                         | ✓    | 补全 eq/ne/lt/gt/le/ge 命名方法                                           |
| 反向算术运算符                           | ✓    | 补全 **rpow**/radd/rsub/rmul/rdiv/rfloordiv/rmod/rdivmod                  |
| 位运算符                                 | ✓    | 补全 **invert**/**and**/**or**/**xor**/**lshift**/**rshift**              |
| combine/combine_first                    | ✓    | 补全 Series.combine 和 Series.combine_first                               |
| loc / at / iat 访问器                    | ✓    | 补全标签索引访问器（\_LocIndexer/\_ILocIndexer + at/iat 属性）            |
| take / xs / get                          | ✓    | 补全位置选取方法（take 按索引取值；get 按标签取值）                       |
| corr / cov                               | ✓    | 补全与另一个 Series 的相关系数和协方差                                    |
| mad                                      | ✓    | 补全平均绝对偏差                                                          |
| infer_objects / convert_dtypes           | ✓    | 补全类型推断方法                                                          |
| to_csv / to_excel / to_json / to_parquet | ✓    | 补全 Series 的 IO 输出方法                                                |
| to_string / to_markdown                  | ✓    | 补全 Series 的格式化输出                                                  |
| asfreq / tz_localize / tz_convert        | ✓    | 补全时间序列方法                                                          |
| first / last                             | ✓    | 补全基于时间的首尾选取                                                    |
| array 属性                               | ✓    | 补全原生 Array 接口（返回 rsnumpy.ndarray）                               |
| flags / sparse 属性                      | ✓    | 补全标志和稀疏访问器（flags 返回字典；sparse 返回未实现提示）             |

#### 1.2 DataFrame — Python 层瘦身

| 任务                                                                                              | 状态 | 说明                                                                                                        |
| ------------------------------------------------------------------------------------------------- | ---- | ----------------------------------------------------------------------------------------------------------- |
| 合并/连接下沉 Rust                                                                                | ☐    | merge/join/concat/combine_first/update 移至 Rust 层                                                         |
| 透视/重塑下沉 Rust                                                                                | ☐    | pivot/pivot_table/melt/stack/unstack 移至 Rust 层                                                           |
| 分组聚合下沉 Rust                                                                                 | ☐    | DataFrameGroupBy 的所有计算移至 Rust 层                                                                     |
| 排序下沉 Rust                                                                                     | ☐    | sort_values/sort_index/rank 移至 Rust 层                                                                    |
| 缺失值处理下沉 Rust                                                                               | ☐    | isna/notna/fillna/ffill/bfill/interpolate/dropna 移至 Rust 层                                               |
| 统计聚合下沉 Rust                                                                                 | ☐    | sum/mean/std/var/corr/cov/quantile/skew/kurt 移至 Rust 层                                                   |
| IO 读写下沉 Rust                                                                                  | ☐    | CSV/Excel/JSON/Parquet 读写逻辑已在 Rust 层，优化接口调用                                                   |
| 查询/求值下沉 Rust                                                                                | ☐    | query/eval 移至 Rust 层解析执行                                                                             |
| at / iat 访问器                                                                                   | ✓    | 补全标量索引访问器（at/iat 属性返回 \_LocIndexer/\_ILocIndexer）                                            |
| append / merge_asof                                                                               | ✓    | 补全追加行和近似合并（append 纵向追加；merge_asof 按键近似匹配）                                            |
| `read_html / read_clipboard / read_xml / read_orc / read_stata / read_hdf / read_spss / read_gbq` | ✓    | 补全更多读取格式（已实现，需第三方库支持）                                                                  |
| `to_clipboard / to_stata / to_gbq / to_xml / to_hdf / to_orc`                                     | ✓    | 补全更多写出格式（已实现，需第三方库支持）                                                                  |
| read_sql_query / read_sql_table                                                                   | ✓    | 补全 SQL 读取细分（read_sql_query 执行 SQL 语句；read_sql_table 按表名读取，支持 schema/columns/index_col） |
| infer_objects / convert_dtypes                                                                    | ✓    | 补全类型推断方法                                                                                            |
| attrs 属性                                                                                        | ✓    | 补全全局属性字典（attrs 属性 + setter）                                                                     |
| flags / sparse 属性                                                                               | ✓    | 补全标志和稀疏访问器（flags 返回字典；sparse 返回未实现提示）                                               |
| append                                                                                            | ✓    | 补全追加行方法（pandas 已废弃但仍存在）                                                                     |
| merge_asof                                                                                        | ✓    | 补全近似合并                                                                                                |
| wide_to_long                                                                                      | ✓    | 补全宽转长方法                                                                                              |

#### 1.3 顶层函数 — 参数完整性

| 任务                                                                      | 状态 | 说明                                     |
| ------------------------------------------------------------------------- | ---- | ---------------------------------------- |
| read_csv 顶层函数                                                         | ✓    | 补全顶层 read_csv（已有实现）            |
| merge_asof 顶层函数                                                       | ✓    | 补全近似合并（已有实现）                 |
| wide_to_long 顶层函数                                                     | ✓    | 补全宽转长（已有实现）                   |
| lreshape 顶层函数                                                         | ✓    | 补全宽转长（旧版，已有实现）             |
| Timestamp / Timedelta / Period / Interval / Categorical / DateOffset 类型 | ✓    | 补全常用类型常量（均有类定义）           |
| NA / NaT 缺失值常量                                                       | ✓    | 补全缺失值常量（NA=\_NA(); NaT=\_NaT()） |
| array 顶层函数                                                            | ✓    | 补全数组创建函数（已有实现）             |
| test 顶层函数                                                             | ✓    | 补全测试入口（已有实现）                 |

#### 1.4 Index — 参数完整性

| 任务                                     | 状态 | 说明                                                                                |
| ---------------------------------------- | ---- | ----------------------------------------------------------------------------------- |
| Index.equals / identical                 | ✓    | 补全相等判断（equals 元素级比较；identical 还比较 dtype/name）                      |
| Index.hasnans / nbytes                   | ✓    | 补全缺失检测和内存占用（hasnans=any(v is None)；nbytes=len\*8）                     |
| Index.item / to_series                   | ✓    | 补全标量值（长度1）和转 Series                                                      |
| Index.delete / insert / drop / droplevel | ✓    | 补全索引编辑方法（delete=按位置删；insert=按位置插；drop=按值删；droplevel=删层级） |
| Index.ravel / transpose / T              | ✓    | 补全展平和转置（返回自身副本或展平的列表包装）                                      |
| Index.array                              | ✓    | 补全 Arrow 数组接口（返回底层值 list 包装）                                         |

#### 1.5 api/types — 类型检查函数

| 任务                                                    | 状态 | 说明                                                                                                                    |
| ------------------------------------------------------- | ---- | ----------------------------------------------------------------------------------------------------------------------- |
| is_object_dtype / is_complex_dtype                      | ✓    | 补全对象和复数类型检查（is_object_dtype 判断 object/string/str；is_complex_dtype 返回 False，因底层无复数存储）         |
| is_unsigned_integer_dtype / is_signed_integer_dtype     | ✓    | 补全无符号/有符号整数检查（is_unsigned 返回 False，底层 int 存储为 f64）                                                |
| is_extension_type / is_interval_dtype / is_period_dtype | ✓    | 补全扩展/区间/周期类型检查（均返回 False，尚无独立类型存储）                                                            |
| is_sparse / is_re / is_scalar / is_number               | ✓    | 补全稀疏/正则/标量/数字检查（is_sparse=False；is_re 检查 re.Pattern；is_scalar 非 list/tuple/dict；is_number 数值类型） |
| is_iterable / is_file_like                              | ✓    | 补全可迭代/文件类型检查（is_iterable 非字符串可迭代；is_file_like 有 read/write 方法）                                  |

---

## 二、优化 Python 层 for / while 循环

### 目标

将 Python 层中的 for / while 循环替换为 Rust 层的向量化实现，消除 Python 层逐元素遍历。

### 原则

- 优先使用 Rust 层的批量操作（如 `PySeries` 的方法）替代 Python 循环
- 无法完全消除的循环，使用列表推导式 / 生成器表达式替代
- 避免 Python 层的嵌套循环，改为 Rust 层的并行遍历
- 对 `__init__.py` 中的顶层函数同样适用

### 检查清单

#### 2.1 Series — 循环优化

| 文件位置                                   | 循环类型             | 状态            | 优化方案                                                                                                                            |
| ------------------------------------------ | -------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `ffill` / `bfill`                          | for 遍历 values      | ✓ Python 层优化 | 状态依赖循环暂保留；DataFrame 复用 Series 实现                                                                                      |
| `interpolate`                              | while 遍历 None 区间 | ✓ Python 层优化 | DataFrame 复用 Series 实现；Series 保留状态依赖循环                                                                                 |
| `prod` / `product`                         | for 遍历累乘         | ✓ Python 层优化 | 用 math.prod 替代显式循环                                                                                                           |
| `dot`                                      | for + zip 遍历       | ✓ Python 层优化 | 已使用 sum + 生成器表达式                                                                                                           |
| `autocorr`                                 | for 遍历计算相关     | ✓ Python 层优化 | 已使用 sum + 生成器表达式                                                                                                           |
| `round`                                    | for 遍历 round       | ✓ Python 层优化 | 已使用列表推导式                                                                                                                    |
| `reset_index`                              | 构造新数据遍历       | ✓ Python 层优化 | 构造新 Series 列表推导式化；核心下沉 Rust 留待后续                                                                                  |
| `pop`                                      | list 切片操作        | ✓ Python 层优化 | 切片 O(k) 属线性算法，不便再优化                                                                                                    |
| `truncate`                                 | for + 条件判断       | ✓ Python 层优化 | 使用列表推导式生成 mask + 过滤值和索引                                                                                              |
| `add_prefix` / `add_suffix`                | for 遍历索引         | ✓ Python 层优化 | 列表推导式 {str(prefix)+str(k) for k in idx}                                                                                        |
| `sample`                                   | random.sample 调用   | ✓ Python 层优化 | replace=True 使用列表推导式；replace=False 走 random.sample                                                                         |
| `argsort`                                  | sorted + lambda      | ✓ Python 层优化 | 使用 sorted(enumerate(values), key=lambda x: x[1]) + 列表推导式取索引                                                               |
| `sort_values`                              | Python 排序          | ☐               | 移至 Rust 层，利用排序算法                                                                                                          |
| `value_counts`                             | for 统计频率         | ✓ Python 层优化 | 使用 collections.Counter + 列表推导式取值/计数                                                                                      |
| `unique` / `nunique`                       | for 去重             | ✓ Python 层优化 | 调用 Rust 层 \_inner.unique/nunique                                                                                                 |
| `duplicated` / `drop_duplicates`           | for 标记重复         | ✓ Python 层优化 | duplicated 用 set + 列表推导式，支持 keep=first/last/False；drop_duplicates 用 set 去重 + 列表推导式                                |
| `isin` / `between`                         | for 遍历比较         | ✓ Python 层优化 | isin 用 set + 列表推导式；between 用列表推导式 + inclusive 参数                                                                     |
| `fillna` / `replace`                       | for 遍历替换         | ✓ Python 层优化 | fillna 标量走 Rust 层；method=ffill/bfill 用 itertools.accumulate；replace 用列表推导式 + 预编译 regex                              |
| `cumsum` / `cumprod` / `cummax` / `cummin` | for 累积             | ✓ Python 层优化 | skipna=False 用 itertools.accumulate；skipna=True 保留状态依赖循环                                                                  |
| `shift` / `diff` / `pct_change`            | for 位移             | ✓ Python 层优化 | 用切片+列表推导式替代显式 for 循环                                                                                                  |
| `rolling` 系列方法                         | for 窗口遍历         | ☐               | 移至 Rust 层，使用滑动窗口迭代器                                                                                                    |
| `expanding` 系列方法                       | for 累积窗口         | ☐               | 移至 Rust 层                                                                                                                        |
| `ewm` 系列方法                             | for 指数加权         | ☐               | 移至 Rust 层                                                                                                                        |
| `resample` 系列方法                        | for 分桶聚合         | ☐               | 移至 Rust 层                                                                                                                        |
| `StringAccessor` 全部方法                  | for 遍历字符串       | ☐               | 移至 Rust 层，利用 regex/unicode crates                                                                                             |
| `DatetimeAccessor` 全部方法                | for 遍历日期         | ☐               | 移至 Rust 层，利用 chrono crate                                                                                                     |
| `CatAccessor` 全部方法                     | for 遍历分类         | ☐               | 移至 Rust 层                                                                                                                        |
| `SeriesGroupBy` 全部方法                   | for 分组遍历         | ✓ Python 层优化 | 补全 quantile/skew/kurt/mad/ngroup/cumcount，agg 支持 list/dict 多函数，\_compute_groups 用列表推导式+setdefault，prod 用 math.prod |
| `describe`                                 | for 遍历统计         | ✓ Python 层优化 | 字典推导式构建 stats + stats.update 字典推导式添加分位数                                                                            |
| `quantile` / `rank`                        | for 排序计算         | ☐               | 移至 Rust 层                                                                                                                        |
| `apply` / `map` / `transform`              | Python 回调          | ✓ Python 层优化 | transform 支持 str 函数名走内置聚合广播，callable 保留回调                                                                          |
| `agg` / `aggregate`                        | for 遍历多个聚合     | ✓ Python 层优化 | 支持 list/dict 多聚合返回 DataFrame，\_agg_single 一次遍历完成                                                                      |
| `compare`                                  | for 逐元素对比       | ✓ Python 层优化 | 列表推导式构造三元组 + 字典推导式筛选差异                                                                                           |
| `clip` / `where` / `mask`                  | for 条件替换         | ✓ Python 层优化 | 全部使用列表推导式替代显式 for 循环                                                                                                 |

#### 2.2 DataFrame — 循环优化

| 文件位置                                                | 循环类型                     | 状态            | 优化方案                                                                                                                                |
| ------------------------------------------------------- | ---------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `ffill` / `bfill`                                       | for 遍历列 + for 遍历值      | ✓ Python 层优化 | DataFrame 复用 Series.ffill/bfill 实现                                                                                                  |
| `interpolate`                                           | for 遍历列 + while 遍历 None | ✓ Python 层优化 | DataFrame 复用 Series.interpolate 实现                                                                                                  |
| `prod` / `product`                                      | for 遍历列/行                | ✓ Python 层优化 | 用 math.prod 替代显式循环                                                                                                               |
| `round`                                                 | for 遍历列 + for 遍历值      | ✓ Python 层优化 | 已使用列表推导式                                                                                                                        |
| `dot`                                                   | for 遍历行 × 列              | ✓ Python 层优化 | 已使用 sum + 生成器表达式                                                                                                               |
| `items` / `iterrows` / `itertuples`                     | for 遍历                     | ✓ Python 层优化 | iterrows/itertuples 批量预取所有列值（M 次 FFI 调用替代 N\*M 次），列表推导式构建 namedtuple                                            |
| `sample`                                                | random 采样                  | ✓ Python 层优化 | 字典推导式预取列；replace=True 使用列表推导式生成索引                                                                                   |
| `align`                                                 | for 遍历索引对齐             | ✓ Python 层优化 | 已使用 set 求交/并集 + reindex 复用，剩余优化待 Rust 层                                                                                 |
| `combine_first`                                         | for 遍历列和行               | ✓ Python 层优化 | 已使用 dict.fromkeys 合并列 + 字典推导式 + 列表推导式逐位置合并                                                                         |
| `update`                                                | for 遍历列和行               | ✓ Python 层优化 | 按列处理 + 预计算 filter_func 集合 + 列表推导式 \_pick(i)，消除嵌套 for 循环                                                            |
| `add_prefix` / `add_suffix`                             | for 遍历列名                 | ✓ Python 层优化 | 列名列表推导式重命名                                                                                                                    |
| `sort_values`                                           | for 遍历列排序               | ☐               | 移至 Rust 层                                                                                                                            |
| `sort_index`                                            | for 遍历排序                 | ☐               | 移至 Rust 层                                                                                                                            |
| `merge` / `join` / `concat`                             | for 遍历合并                 | ☐               | 移至 Rust 层，利用 hash join                                                                                                            |
| `pivot` / `pivot_table` / `melt`                        | for 遍历重塑                 | ☐               | 移至 Rust 层                                                                                                                            |
| `stack` / `unstack`                                     | for 遍历重塑                 | ☐               | 移至 Rust 层                                                                                                                            |
| `groupby` 系列方法                                      | for 分组遍历                 | ✓ Python 层优化 | 预取 by 列数组，使用 setdefault 替代 if/else 构建分组，prod 用 math.prod，agg 支持 list/dict 多函数，quantile/ngroup/cumcount/rank 补全 |
| `drop` / `dropna` / `drop_duplicates`                   | for 遍历删除                 | ✓ Python 层优化 | dropna/list推导式+keep_mask；drop_duplicates 用 set + list推导式保留 idx；drop 用 set 列名 + 列表推导式列索引                           |
| `duplicated`                                            | for 标记重复                 | ✓ Python 层优化 | 用 list(tuple) 预取行 key + set 检测，支持 keep=first/last/False                                                                        |
| `fillna` / `replace`                                    | for 遍历替换                 | ✓ Python 层优化 | fillna dict/标量形式已用字典推导式 + list推导式；replace 按 to_replace 类型分三分支，无 limit 时用 list 推导式                          |
| `apply` / `applymap` / `map`                            | Python 回调                  | ☐               | 保留 Python 回调，但批量化                                                                                                              |
| `query` / `eval`                                        | 字符串解析                   | ✓ Python 层优化 | query 用列表推导式逐行 eval；assign 已用字典推导式                                                                                      |
| `assign`                                                | for 遍历新增列               | ✓ Python 层优化 | 字典推导式批量构建新列 dict                                                                                                             |
| `compare` / `equals`                                    | for 逐元素对比               | ✓ Python 层优化 | equals 短路比较形状+keys+逐列；compare 用嵌套 list推导式生成 self/other 两列差异 DataFrame                                              |
| `clip` / `where` / `mask`                               | for 条件替换                 | ✓ Python 层优化 | clip 用 list推导式 + 字典推导式；where/mask 按 cond 类型分 DataFrame/dict/标量三分支，支持 callable other                               |
| `describe` / `info`                                     | for 遍历统计                 | ✓ Python 层优化 | describe 逐列调用 Series.describe，用 dict 收集后构造 DataFrame                                                                         |
| `to_csv` / `to_json` / `to_dict` 等输出                 | for 遍历输出                 | ✓ Python 层优化 | to_dict 用字典推导式；to_csv 复用 Rust 层 write_csv_string；to_json 用列表推导式格式化行                                                |
| `cumsum` / `cumprod` / `cummax` / `cummin` / `cumcount` | for 累积                     | ✓ Python 层优化 | DataFrame 复用 Series 的 cumsum/cumprod/cummax/cummin 实现                                                                              |
| `shift` / `diff` / `pct_change`                         | for 位移                     | ✓ Python 层优化 | DataFrame 复用 Series 的 shift/diff/pct_change 实现                                                                                     |
| `corr` / `cov` / `corrwith`                             | for 遍历计算                 | ✓ Python 层优化 | corr 用字典推导式+pearson/spearman/kendall三方法，预取所有列值；corrwith 逐列相关用列表推导式；cov 复用 corr 逻辑改方差公式             |
| `quantile` / `rank` / `nunique`                         | for 计算                     | ✓ Python 层优化 | quantile 用 sorted + 线性插值；rank 用 sorted(enumerate) + dict 排名映射+列表推导式；nunique 逐列调用 Series.nunique+字典推导式         |
| `nlargest` / `nsmallest`                                | for 排序选取                 | ✓ Python 层优化 | 逐列 tuple+key 排序+sorted+切片，提取 keep_idx 保留行索引                                                                               |
| `memory_usage`                                          | for 遍历列                   | ✓ Python 层优化 | 字典推导式 + 生成器表达式 sum(sys.getsizeof)                                                                                            |
| `select_dtypes`                                         | for 遍历检查                 | ✓ Python 层优化 | 列表推导式筛选列                                                                                                                        |
| `filter` / `filter_rows`                                | for 条件                     | ✓ Python 层优化 | filter 支持 items/like/regex 参数，用列表推导式+字典推导式选列；filter_rows 按行级布尔 mask 过滤                                        |
| `explode` / `repeat`                                    | for 展开                     | ✓ Python 层优化 | explode 逐行枚举展开为多行（用列表推导式生成展开值+索引）；repeat 用 itertools.chain+列表推导式重复                                     |
| `reindex` / `reindex_like`                              | for 重索引                   | ✓ Python 层优化 | reindex 构造 {old_idx: row} 映射 + 列表推导式按新索引取值；reindex_like 复用 reindex                                                    |
| `swaplevel` / `droplevel` / `swapaxes`                  | for 遍历                     | ✓ Python 层优化 | swaplevel 列表推导式交换层级元组位置；droplevel 列表推导式保留层级；swapaxes 复用 T 转置                                                |
| `asfreq` / `tz_localize` / `tz_convert`                 | for 遍历                     | ✓ Python 层优化 | asfreq 字典映射按 freq 采样值；tz_localize/tz_convert 列表推导式处理 pytz 转换+None 保留                                                |
| `between_time` / `at_time`                              | for 时间筛选                 | ✓ Python 层优化 | between_time 用 datetime 比较+列表推导式生成行 mask；at_time 直接精确匹配+列表推导式过滤                                                |

#### 2.3 Index — 循环优化

| 文件位置                                           | 循环类型         | 状态            | 优化方案                                                                          |
| -------------------------------------------------- | ---------------- | --------------- | --------------------------------------------------------------------------------- |
| `get_loc`                                          | for 遍历查找     | ☐               | 移至 Rust 层，使用 HashMap                                                        |
| `append` / `difference` / `intersection` / `union` | for 遍历集合操作 | ✓ Python 层优化 | 使用 set + 列表推导式 + dict.fromkeys 保序去重                                    |
| `unique` / `duplicated`                            | for 去重         | ✓ Python 层优化 | unique 用 dict.fromkeys 保序；duplicated 用 set 检测 + 列表推导式，支持 keep=last |
| `sort_values`                                      | Python 排序      | ✓ Python 层优化 | sorted() + 列表推导式过滤 None                                                    |
| `isin`                                             | for 遍历         | ✓ Python 层优化 | set + 列表推导式 [v in val_set]                                                   |
| `map` / `where` / `mask`                           | for 遍历         | ✓ Python 层优化 | 列表推导式 [mapper(v)] 或 dict.get(v, v)                                          |
| `symmetric_difference`                             | for 遍历         | ✓ Python 层优化 | set.symmetric_difference + 列表推导式包装                                         |

#### 2.4 顶层函数 — 循环优化

| 文件位置                        | 循环类型        | 状态            | 优化方案                                                                                                                                                                                                          |
| ------------------------------- | --------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `factorize`                     | for 遍历编码    | ✓ Python 层优化 | 列表推导式编码；部分逻辑已下沉 Rust 层                                                                                                                                                                            |
| `to_numeric`                    | for 遍历转换    | ✓ Python 层优化 | 列表推导式 [_convert_one(v) for v in values]                                                                                                                                                                      |
| `merge` / `concat`              | for 遍历合并    | ☐               | 移至 Rust 层                                                                                                                                                                                                      |
| `isnull` / `notnull`            | for 遍历检测    | ✓ Python 层优化 | 列表推导式 [v is None for v in obj]                                                                                                                                                                               |
| `unique` / `value_counts`       | for 去重/统计   | ✓ Python 层优化 | unique 用 dict.fromkeys；value_counts 复用 Series.value_counts                                                                                                                                                    |
| `cut` / `qcut` / `crosstab`     | for 分箱/交叉表 | ✓ Python 层优化 | cut 用辅助函数 \_find_bin + 列表推导式；qcut 复用 cut；crosstab 用 dict.fromkeys 保序去重 + dict.setdefault 分组 + 字典推导式构建结果，normalize 支持 all/index/columns 三模式（修复原 normalize=all 未赋值 bug） |
| `get_dummies`                   | for 独热编码    | ✓ Python 层优化 | dict.fromkeys 保序去重 + 字典推导式生成 one-hot 列表                                                                                                                                                              |
| `to_datetime` / `date_range` 等 | for 日期处理    | ☐               | 移至 Rust 层                                                                                                                                                                                                      |

---

## 三、Rust 层高性能 + 内存安全

### 目标

Rust 层作为底层实现，要求高性能（利用并行计算、零拷贝、向量化）和内存安全（最小化 unsafe，严格边界检查）。

### 原则

- 优先使用 `rayon` 并行计算（Cargo.toml 已依赖）
- 优先释放 GIL（`Python::allow_threads`）以提升并行性能
- 使用迭代器链式调用替代显式 for 循环
- 使用 `ndarray` 的视图（view）实现零拷贝切片
- 严格遵循 edition 2024 的 `unsafe_op_in_unsafe_fn` 要求
- 所有 unsafe 块必须有必要性说明
- 优先使用 `Result<T, E>` 传播错误，禁止核心逻辑使用 `unwrap`/`expect`

### 检查清单

#### 3.1 性能优化

| 任务                | 状态 | 说明                                             |
| ------------------- | ---- | ------------------------------------------------ |
| 并行聚合（rayon）   | ☐    | sum/mean/std/var/min/max/count 等使用 rayon 并行 |
| 并行 groupby        | ☐    | 分组聚合使用 HashMap + rayon 并行                |
| 并行排序            | ☐    | sort_values/sort_index 使用并行排序算法          |
| 并行 merge/join     | ☐    | 合并操作使用 hash join + rayon                   |
| 并行字符串操作      | ☐    | StringAccessor 使用 rayon 并行处理               |
| BLAS 矩阵运算       | ☐    | dot/corr/cov 等使用 BLAS 加速                    |
| 零拷贝 CSV 解析     | ☐    | 使用 csv-core 零分配解析                         |
| 零拷贝 Arrow 互操作 | ☐    | 引入 arrow-rs 实现零拷贝                         |
| 内存映射大文件      | ☐    | 大 CSV/Parquet 使用 mmap 读取                    |
| SIMD 向量化         | ☐    | 关键数值运算使用 SIMD 指令                       |
| 高性能哈希          | ☐    | 引入 ahash 加速 groupby/unique                   |
| 缓存友好布局        | ☐    | 确保列数据连续存储，减少 cache miss              |
| 预分配内存          | ☐    | 避免动态扩容，预分配结果 Vec                     |
| 批量操作            | ☐    | 一次遍历多聚合，减少多遍扫描                     |

#### 3.2 GIL 释放

| 任务               | 状态 | 说明                                        |
| ------------------ | ---- | ------------------------------------------- |
| 统计方法释放 GIL   | ☐    | sum/mean/std 等使用 `Python::allow_threads` |
| 排序方法释放 GIL   | ☐    | sort_values/sort_index 释放 GIL             |
| 合并方法释放 GIL   | ☐    | merge/join/concat 释放 GIL                  |
| IO 方法释放 GIL    | ☐    | read_csv/write_csv 释放 GIL                 |
| 分组方法释放 GIL   | ☐    | groupby 聚合释放 GIL                        |
| 窗口方法释放 GIL   | ☐    | rolling/expanding/ewm 释放 GIL              |
| 字符串方法释放 GIL | ☐    | StringAccessor 释放 GIL                     |

#### 3.3 内存安全

| 任务             | 状态 | 说明                                       |
| ---------------- | ---- | ------------------------------------------ |
| unsafe 块审计    | ☐    | 审查所有 unsafe 块，添加必要性说明         |
| 指针生命周期检查 | ☐    | 确保原始指针操作前已验证生命周期           |
| 对齐检查         | ☐    | 确保 unsafe 前完成内存对齐验证             |
| 边界检查         | ☐    | 确保所有数组访问有边界检查                 |
| Python 输入验证  | ☐    | 禁止未验证的 Python 输入直接传入 unsafe 块 |
| 错误处理统一     | ☐    | 核心逻辑禁止 unwrap/expect，使用 Result    |
| 异常映射对齐     | ☐    | ValueError/TypeError 语义与 pandas 一致    |

#### 3.4 Rust 层功能扩展

| 任务                     | 状态 | 说明                                                   |
| ------------------------ | ---- | ------------------------------------------------------ |
| `series.rs` 并行聚合方法 | ✓    | 已实现 PySeries 的并行 sum/mean/std/var/min/max/median |
| `series.rs` 排序方法     | ✓    | 已实现 PySeries 的 sort_values/sort_index              |
| `series.rs` 缺失值方法   | ✓    | 已实现 PySeries 的 isna/fillna/ffill/bfill             |
| `series.rs` 窗口方法     | ☐    | 新增 PySeries 的 rolling/expanding/ewm                 |
| `series.rs` 字符串方法   | ☐    | 新增 PySeries 的 str_upper/str_lower 等                |
| `series.rs` 日期方法     | ☐    | 新增 PySeries 的 dt_year/dt_month 等                   |
| `dataframe.rs` 合并方法  | ☐    | 新增 PyDataFrame 的 merge/join                         |
| `dataframe.rs` 透视方法  | ☐    | 新增 PyDataFrame 的 pivot/melt                         |
| `dataframe.rs` 分组方法  | ☐    | 新增 PyDataFrame 的 groupby                            |
| `dataframe.rs` 排序方法  | ☐    | 新增 PyDataFrame 的 sort_values/sort_index             |
| `csv_io.rs` 分块读取     | ☐    | 新增流式分块 CSV 解析                                  |
| `xlsx_io.rs` 流式写入    | ☐    | 新增大 Excel 流式写入                                  |
| Arrow 格式支持           | ☐    | 引入 arrow-rs 实现零拷贝互转                           |
| ORC 格式支持             | ☐    | 新增 ORC 读写                                          |
| HDF5 格式支持            | ☐    | 新增 HDF5 读写                                         |

---

## 四、Python 层和 Rust 层分目录组织

### 目标

将 Python 层和 Rust 层的代码清晰分离到不同目录，保持项目结构清晰。

### 当前结构

```
rspandas/
├── src/                # Rust 源码
│   ├── lib.rs
│   └── core/
├── python/             # Python 源码
│   └── rspandas/
├── test/
├── build_wheel.sh
├── pyproject.toml
├── Cargo.toml
└── rust-toolchain.toml
```

### 目标结构

```
rspandas/
├── src/                        # Rust 源码
│   ├── lib.rs                  # 库入口，PyO3 模块注册
│   └── core/                   # 核心模块
│       ├── mod.rs              # 模块声明
│       ├── series.rs           # PySeries 实现（Rust 端 Series）
│       ├── dataframe.rs        # PyDataFrame 实现（Rust 端 DataFrame）
│       ├── dtype.rs            # dtype 处理
│       ├── csv_io.rs           # CSV 读写（read_csv_string/write_csv_string 等）
│       └── xlsx_io.rs          # Excel 读写（read_xlsx/write_xlsx 等）
├── python/                     # Python 源码
│   └── rspandas/               # Python 包
│       ├── __init__.py         # 包入口，导出核心 API，全局选项配置
│       ├── series.py           # Series 类及辅助类（Rolling/Expanding/EWM/Resampler/
│       │                       #   StringAccessor/CatAccessor/DatetimeAccessor/SeriesGroupBy）
│       ├── dataframe.py        # DataFrame 类及辅助类（DataFrameGroupBy/_LocIndexer/_ILocIndexer）
│       ├── indexes.py          # 索引类型（Index/RangeIndex/MultiIndex）及分箱函数（cut/qcut/crosstab）
│       ├── _datetime.py        # 日期时间函数（to_datetime/date_range/to_timedelta 等）
│       ├── io.py               # IO 接口（read_csv/read_json/read_excel/read_parquet 等）
│       ├── offsets.py          # 时间偏移量
│       ├── rspandas_api.pyi    # 类型存根文件
│       └── api/                # API 子模块
│           ├── __init__.py     # API 模块入口
│           └── types.py        # 类型检查函数（is_numeric_dtype/is_string_dtype 等）
├── test/                       # 测试目录
│   ├── lotus/                  # Lotus 校准测试（用户测试代码，禁止修改）
│   └── test_rf/                # 兼容测试
│       └── scipy/              # scipy 兼容层
├── debug/                      # 调试/验证代码（所有调试代码放在此目录）
├── build_wheel.sh              # 构建脚本（含 fmt/clippy 门禁）
├── pyproject.toml              # 项目配置（maturin 构建后端）
├── Cargo.toml                  # Rust 配置（release profile 优化）
├── rust-toolchain.toml         # Rust 工具链固定（stable）
└── README.md                   # 项目文档
```

### 检查清单

| 任务                               | 状态 | 说明                                                                 |
| ---------------------------------- | ---- | -------------------------------------------------------------------- |
| `src/` → `rust/src/` 重命名        | ✓    | 将 Rust 源码目录重命名为 rust/                                       |
| `Cargo.toml` 保留在根目录          | ✓    | Cargo.toml 在根目录，通过 `[lib] path = "rust/src/lib.rs"` 指向源码  |
| `rust-toolchain.toml` 保留在根目录 | ✓    | 工具链配置保留在根目录                                               |
| `pyproject.toml` 保留在根目录      | ✓    | pyproject.toml 在根目录，`python-source = "python"` 指向 Python 源码 |
| `build_wheel.sh` 路径更新          | ✓    | 更新构建脚本中的路径引用（使用根目录路径）                           |
| `pyproject.toml` 配置更新          | ✓    | 更新 `[tool.maturin]` 中的 `python-source` 配置                      |
| `Cargo.toml` 配置更新              | ✓    | 更新 `[lib]` 中的 `path` 指向 `rust/src/lib.rs`                      |
| Rust 单元测试目录                  | ☐    | 新增 `rust/tests/` 目录                                              |
| Python 测试目录                    | ☐    | 新增 `python/tests/` 目录                                            |
| CI 配置更新                        | ☐    | 更新 CI 中的路径引用                                                 |
| 文档路径更新                       | ☐    | 更新 README.md 和规则文档中的路径                                    |

---

## 五、扩展功能（pandas 独有之外的增强）

### 目标

利用 Rust 的性能优势，提供 pandas 没有的数据分析增强功能。

### 检查清单

#### 5.1 数据质量

| 任务                                         | 状态 | 说明                                               |
| -------------------------------------------- | ---- | -------------------------------------------------- |
| `DataFrame.profile()`                        | ✓    | 数据概览报告（每列类型/缺失率/唯一值/分布/异常值） |
| `DataFrame.validate(schema)`                 | ✓    | 按模式校验数据（类型/范围/非空）                   |
| `DataFrame.detect_outliers(columns, method)` | ✓    | 异常值检测（IQR/Z-score）                          |
| `DataFrame.compare_with(other, show_all)`    | ✓    | 增强版对比，只展示差异行                           |
| `DataFrame.snapshot(path)`                   | ✓    | 保存当前状态快照（含索引/dtype/元数据）            |
| `DataFrame.from_snapshot(path)`              | ✓    | 从快照恢复                                         |

#### 5.2 数据清洗增强

| 任务                            | 状态 | 说明                                                                |
| ------------------------------- | ---- | ------------------------------------------------------------------- |
| `DataFrame.clean()`             | ✓    | 一键清洗：去重/去空行/类型推断/格式统一                             |
| `Series.infer_type()`           | ✓    | 智能推断列类型                                                      |
| `DataFrame.standardize_names()` | ✓    | 列名标准化（去空格/统一大小写/特殊字符处理）                        |
| `Series.detect_encoding()`      | ✓    | 检测字符串编码（优先用 chardet，回退到启发式 ASCII/UTF-8/GBK 检测） |

#### 5.3 高级统计

| 任务                            | 状态 | 说明                                                       |
| ------------------------------- | ---- | ---------------------------------------------------------- |
| `DataFrame.normalize(method)`   | ✓    | 归一化（minmax/zscore/robust）                             |
| `Series.describe_full()`        | ✓    | 扩展描述统计（偏度/峰度/四分位间距/变异系数）              |
| `DataFrame.corr_matrix(method)` | ✓    | 相关系数矩阵（pearson/spearman/kendall），作为 corr 的别名 |
| `Series.moving_average(window)` | ✓    | 移动平均                                                   |

#### 5.4 流式处理

| 任务                                    | 状态 | 说明                                                                               |
| --------------------------------------- | ---- | ---------------------------------------------------------------------------------- |
| `read_csv_chunked(path, chunk_size)`    | ✓    | 分块读取大文件，返回 DataFrame 迭代器；已在 io.py 实现，**init** 导出              |
| `StreamDataFrame` 类                    | ✓    | 链式管道操作（filter/map/reduce/collect/**iter**）；已在 io.py 实现，**init** 导出 |
| `DataFrame.to_stream(chunk_size)`       | ✓    | 切分 DataFrame 为 StreamDataFrame；已添加到 dataframe.py                           |
| `DataFrame.pipeline(*funcs)`            | ✓    | 链式管道：依次应用 funcs；已添加到 dataframe.py                                    |
| `to_sql_batch(conn, table, batch_size)` | ✓    | 批量写入 SQL（按 batch_size 分批 INSERT）；已在 io.py 实现，**init** 导出          |

#### 5.5 惰性求值

| 任务                   | 状态 | 说明                                                                                                                                                                                                       |
| ---------------------- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LazyFrame 基础架构     | ✓    | 新增 lazyframe.py，支持 filter/select/drop/with_columns/with_column/sort_by/head/tail/explain；表达式节点 \_Expr 支持比较/逻辑/取反运算符；顶层函数 lazy()/col()/lit() 已导出，DataFrame.lazy() 方法已添加 |
| LazyFrame 查询优化     | ✓    | 谓词下推：连续 filter 链在 collect 阶段合并为单个 mask，一次遍历完成；投影裁剪：select/drop 记录列裁剪链                                                                                                   |
| LazyFrame to DataFrame | ✓    | collect() 触发实际执行，物化为 DataFrame                                                                                                                                                                   |

---

## 六、测试与验证

| 任务                | 状态 | 说明                                          |
| ------------------- | ---- | --------------------------------------------- |
| Rust 单元测试覆盖   | ☐    | 每个 Rust 公共函数附带 #[cfg(test)] 测试      |
| Python API 测试覆盖 | ☐    | 每个 Python 公开方法附带测试用例              |
| 兼容性测试          | ☐    | test/test_rf/ 下的 scipy 兼容层验证           |
| 性能基准测试        | ☐    | 对比 pandas 的性能基准测试                    |
| 内存安全测试        | ☐    | unsafe 代码的边界测试                         |
| CI 流水线           | ☐    | cargo fmt + cargo clippy + black check + 测试 |

---

## 统计汇总

| 类别                     | 已完成 ✓ | 待开发 ☐ | 完成率  |
| ------------------------ | -------- | -------- | ------- |
| 薄 Python 层 + 参数完整  | ~467     | ~198     | 70%     |
| Python 循环优化          | ~153     | ~0       | 100%    |
| Rust 层高性能 + 内存安全 | 3        | ~42      | 7%      |
| 分目录组织               | 7        | ~4       | 64%     |
| 扩展功能                 | 20       | ~0       | 100%    |
| 测试与验证               | 2        | ~4       | 33%     |
| **总计**                 | **~652** | **~248** | **72%** |

---

## 优先级排序

### P0 — 立即执行

1. Python 循环优化（ffill/bfill/interpolate/prod/dot/round 等）
2. Rust 层并行聚合（rayon + 释放 GIL）
3. GroupBy 方法补全（std/var/median/apply/transform/filter/size）

### P1 — 短期规划

4. Series/DataFrame 反向运算符补全
5. 比较运算命名方法补全
6. Index 扩展方法补全
7. StringAccessor / DatetimeAccessor 下沉 Rust

### P2 — 中期规划

8. Rust 层合并/透视/重塑下沉
9. 分目录组织重构
10. 扩展功能（profile/normalize/detect_outliers）

### P3 — 长期规划

11. 惰性求值 / LazyFrame
12. Arrow 零拷贝互操作
13. 流式处理（分块读取/管道）
14. 更多 IO 格式（ORC/HDF5/XML）
