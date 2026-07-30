# rspandas

**pandas-compatible API, Rust-powered performance**

rspandas 是一个高性能的 pandas 兼容数据分析库，使用 Rust 构建核心计算引擎。提供与 pandas 一致的 API 接口，同时获得接近原生的执行性能。

## 特性

- **95%+ pandas API 覆盖** — Series、DataFrame、GroupBy、Rolling/Expanding/EWM、Resampler、时间序列
- **Rust 2024 核心** — 列式存储、向量化计算、Rayon 并行迭代器、LTO 优化
- **跨平台 wheel** — 预编译二进制支持 Linux (x86_64/arm64)、macOS (Intel/Apple Silicon)、Windows (x64/x86)
- **最小依赖** — 运行时仅需 `rsnumpy`，无需 NumPy/PyArrow
- **丰富 I/O** — CSV、Excel（原生 Rust calamine + rust_xlsxwriter）、JSON、Parquet、SQL、Pickle、Feather
- **完整类型系统** — int64、float64、bool、string、category、datetime、timedelta、period

## 安装

```bash
pip install rspandas
```

要求 Python >= 3.9。

## 快速开始

```python
import rspandas as rpd

# 创建 DataFrame
df = rpd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "score": [88.5, 92.0, 79.3],
})

# 基础操作
print(df.shape)           # (3, 3)
print(df.dtypes)          # 数据类型
print(df.describe())      # 统计摘要

# 过滤和排序
df[df["age"] > 26]
df.sort_values("score", ascending=False)

# 分组聚合
df.groupby("name").sum()
df.groupby("name").agg({"score": "mean"})

# I/O 操作
df.to_csv("output.csv")
df.to_excel("output.xlsx")
df.to_parquet("output.parquet")
```

## 依赖

- Python >= 3.9
- `rsnumpy` — 零拷贝 ndarray 操作

## 许可证

MIT
