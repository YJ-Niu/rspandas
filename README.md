# rspandas

A pandas-like library built on Rust — familiar pandas API, Rust-powered performance.

```python
import rspandas as rpd

df = rpd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
print(df.describe())
print(df.groupby("a").sum())
```

## Features

- **95%+ pandas API compatibility** — drop-in replacement for most use cases
- **Rust backend** — columnar storage, vectorized operations, near-native performance
- **Zero runtime dependencies** — pure Python + compiled Rust extension
- **Rich I/O** — CSV, Excel, Parquet, JSON, SQL, Pickle, Feather
- **Full type system** — int64, float64, bool, string, category, datetime, timedelta, period
- **Comprehensive test suite** — 950+ tests covering all major APIs

## Installation

Requires Python >= 3.9.

```bash
pip install -e .
```

Or build from source:

```bash
maturin build --release
pip install target/wheels/rspandas-*.whl
```

## Quick Start

### Series

```python
import rspandas as rpd

# Create
s = rpd.Series([1, 2, 3, 4, 5], name="values")

# Basic ops
s.head(3)          # first 3 rows
s.sum()            # 15
s.mean()           # 3.0
s.std()            # ~1.58
s.describe()       # summary stats

# Missing values
s2 = rpd.Series([1, None, 3])
s2.fillna(0)       # replace None with 0
s2.dropna()        # remove None rows

# String operations
s3 = rpd.Series(["hello", "world"])
s3.str.upper()     # ["HELLO", "WORLD"]
s3.str.contains("ell")  # [True, False]
```

### DataFrame

```python
import rspandas as rpd

# Create
df = rpd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "score": [88.5, 92.0, 79.3],
})

# Properties
df.shape           # (3, 3)
df.columns         # ["name", "age", "score"]
df.dtypes          # {"name": "object", "age": "int64", "score": "float64"}

# Subsetting
df.head(2)         # first 2 rows
df["age"]          # Series
df[["name", "score"]]  # DataFrame with selected columns

# Filtering
df[df["age"] > 26]  # rows where age > 26

# Sorting
df.sort_values("score", ascending=False)
```

### GroupBy

```python
df = rpd.DataFrame({
    "team": ["A", "A", "B", "B", "C"],
    "score": [10, 20, 30, 40, 50],
})

df.groupby("team").sum()      # sum by group
df.groupby("team").mean()     # mean by group
df.groupby("team").agg("std") # std by group
df.groupby("team").cumcount() # cumulative count
df.groupby("team").rank()     # rank within group
```

### Window Functions

```python
s = rpd.Series([1, 2, 3, 4, 5])

s.rolling(3).mean()    # [None, None, 2.0, 3.0, 4.0]
s.rolling(3).sum()     # [None, None, 6.0, 9.0, 12.0]
s.rolling(3).std()     # rolling std

s.expanding().mean()   # expanding window
s.expanding().sum()    # [1.0, 3.0, 6.0, 10.0, 15.0]

s.ewm(span=3).mean()   # exponentially weighted
```

### Time Series

```python
import rspandas as rpd

dates = rpd.date_range("2024-01-01", periods=5, freq="D")
ts = rpd.Series([1, 2, 3, 4, 5], index=dates)

ts.shift(1)            # lag
ts.diff()              # difference
ts.pct_change()        # percent change
ts.cumsum()            # [1, 3, 6, 10, 15]

# DatetimeSeries
ds = rpd.to_datetime(["2024-01-15", "2024-06-15"])
ds.dt.year             # [2024, 2024]
ds.dt.month            # [1, 6]
ds.dt.dayofweek        # [0, 5]  (Monday=0)
ds.dt.month_name       # ["January", "June"]
```

### Reshape

```python
# Melt (wide to long)
df = rpd.DataFrame({"a": [1, 2], "b": [3, 4]})
df.melt(id_vars=["a"])

# Pivot (long to wide)
df = rpd.DataFrame({
    "x": ["a", "a", "b"],
    "y": ["p", "q", "p"],
    "v": [1, 2, 3],
})
df.pivot(index="x", columns="y", values="v")

# Stack / Unstack
df.stack()
df.transpose()  # or df.T
```

### I/O

```python
import rspandas as rpd

# CSV
df = rpd.read_csv("data.csv")
df.to_csv("output.csv")

# Excel
df = rpd.read_excel("data.xlsx")
df.to_excel("output.xlsx")

# Parquet
df = rpd.read_parquet("data.parquet")
df.to_parquet("output.parquet")

# JSON
df = rpd.read_json("data.json")
df.to_json("output.json")

# SQL
df = rpd.read_sql("sqlite:///db.sqlite", "SELECT * FROM table")
df.to_sql("sqlite:///db.sqlite", "table_name")

# Pickle
df = rpd.read_pickle("data.pkl")
df.to_pickle("output.pkl")
```

### Merge & Concat

```python
df1 = rpd.DataFrame({"key": ["a", "b", "c"], "v1": [1, 2, 3]})
df2 = rpd.DataFrame({"key": ["b", "c", "d"], "v2": [4, 5, 6]})

df1.merge(df2, on="key", how="inner")  # inner join
df1.merge(df2, on="key", how="left")   # left join
df1.merge(df2, on="key", how="outer")  # outer join

rpd.concat([df1, df2], axis=0)  # row-wise
rpd.concat([df1, df2], axis=1)  # column-wise
```

### Interop

```python
# pandas
df.to_pandas()
rpd.DataFrame.from_pandas(pdf)

# numpy
s.to_numpy()
rpd.Series.from_numpy(arr)

# pyarrow
df.to_arrow()
rpd.DataFrame.from_arrow(table)
```

### Utilities

```python
rpd.factorize(["a", "b", "a", "c"])  # (codes, categories)
rpd.to_numeric(["1", "2", "x"], errors="coerce")  # [1, 2, None]
rpd.get_dummies(df)
rpd.cut(s, bins=[0, 10, 20, 30])
rpd.qcut(s, q=4)
rpd.crosstab(df["a"], df["b"])
```

### Options

```python
rpd.set_option("display.max_rows", 100)
rpd.set_option("display.max_columns", 50)
rpd.get_option("display.width")  # 80
rpd.reset_option("all")
```

## API Coverage

### Top-level Functions

`read_csv`, `to_csv`, `read_excel`, `to_excel`, `read_parquet`, `to_parquet`, `read_json`, `to_json`, `read_sql`, `to_sql`, `read_pickle`, `to_pickle`, `read_feather`, `to_feather`, `concat`, `merge`, `get_dummies`, `cut`, `qcut`, `crosstab`, `factorize`, `to_datetime`, `to_timedelta`, `to_numeric`, `date_range`, `timedelta_range`, `period_range`, `bdate_range`, `infer_freq`, `set_option`, `get_option`, `reset_option`

### Series

Properties: `shape`, `dtype`, `name`, `values`, `index`, `size`, `empty`

Accessors: `.str`, `.dt`, `.cat`

Methods: `head`, `tail`, `sum`, `mean`, `min`, `max`, `count`, `std`, `var`, `median`, `describe`, `isnull`, `notnull`, `dropna`, `fillna`, `unique`, `nunique`, `value_counts`, `astype`, `sort_values`, `sort_index`, `apply`, `map`, `replace`, `where`, `mask`, `duplicated`, `drop_duplicates`, `isin`, `between`, `rolling`, `expanding`, `ewm`, `resample`, `shift`, `diff`, `pct_change`, `cumsum`, `cumprod`, `cummax`, `cummin`, `rank`, `quantile`, `argmax`, `argmin`, `idxmax`, `idxmin`, `explode`, `repeat`, `to_list`, `to_numpy`, `to_dict`, `to_frame`, `to_pandas`, `from_pandas`, `mode`, `skew`, `kurt`, `sem`, `abs`, `round`, `clip`, `rename`, `rename_axis`, `iloc`, `loc`

### DataFrame

Properties: `shape`, `columns`, `dtypes`, `index`, `values`, `size`, `empty`, `T`

Methods: `head`, `tail`, `describe`, `info`, `dropna`, `fillna`, `merge`, `concat`, `groupby`, `apply`, `applymap`, `sort_values`, `sort_index`, `sort_columns`, `set_index`, `reset_index`, `reindex`, `drop`, `rename`, `rename_axis`, `replace`, `duplicated`, `drop_duplicates`, `melt`, `pivot`, `pivot_table`, `stack`, `unstack`, `transpose`, `swapaxes`, `rolling`, `expanding`, `ewm`, `resample`, `shift`, `diff`, `pct_change`, `cumsum`, `cumprod`, `cummax`, `cummin`, `rank`, `quantile`, `mode`, `skew`, `kurt`, `idxmax`, `idxmin`, `clip`, `astype`, `select_dtypes`, `filter`, `assign`, `eval`, `query`, `pipe`, `transform`, `take`, `xs`, `get`, `compare`, `equals`, `copy`, `pop`, `insert`, `first`, `last`, `truncate`, `asfreq`, `tz_localize`, `tz_convert`, `between_time`, `at_time`, `first_valid_index`, `last_valid_index`, `nunique`, `memory_usage`, `cumcount`, `to_pandas`, `from_pandas`, `to_numpy`, `from_numpy`, `to_arrow`, `from_arrow`, `iloc`, `loc`

### String Accessor

`lower`, `upper`, `title`, `capitalize`, `swapcase`, `casefold`, `strip`, `lstrip`, `rstrip`, `len`, `contains`, `startswith`, `endswith`, `replace`, `split`, `rsplit`, `slice`, `cat`, `find`, `rfind`, `findall`, `match`, `fullmatch`, `extract`, `extractall`, `partition`, `rpartition`, `wrap`, `zfill`, `pad`, `isalnum`, `isalpha`, `isdigit`, `islower`, `isupper`, `isspace`, `istitle`, `encode`, `decode`, `get`, `count`, `ljust`, `rjust`, `center`, `slice_replace`, `get_dummies`

### Datetime Accessor

`year`, `month`, `day`, `hour`, `minute`, `second`, `microsecond`, `dayofweek`, `dayofyear`, `quarter`, `is_month_start`, `is_month_end`, `is_year_start`, `is_year_end`, `is_leap_year`, `days_in_month`, `day_name`, `month_name`, `strftime`, `to_pydatetime`

### Index Types

`Index`, `RangeIndex`, `MultiIndex`, `IntervalIndex`, `DatetimeIndex`, `TimedeltaIndex`, `PeriodIndex`, `CategoricalIndex`

## Architecture

```
User Code (Python)
    |
    |  import rspandas as rpd
    v
+----------------------------------------+
|  python/rspandas/                      |  <-- Python API layer
|  series.py / dataframe.py / ...        |      (pandas-compatible signatures)
+----------------------------------------+
    | PyO3 FFI
    v
+----------------------------------------+
|  rspandas._rust (compiled .so/.dylib)  |  <-- Native module
+----------------------------------------+
|  PySeries / PyDataFrame (#[pyclass])   |
+----------------------------------------+
|  Series / DataFrame (Rust structs)     |  <-- Rust core
+----------------------------------------+
|  ColumnData (Int/Float/Bool/String)    |
+----------------------------------------+
```

- **Python layer**: parameter validation, type inference, formatted display, API compatibility
- **Rust core**: columnar storage, vectorized computation, filtering, aggregation
- **PyO3 bridge**: zero-copy Python/Rust type conversion

## Development

```bash
# Install in dev mode
pip install -e .

# Run tests
pytest tests/        # 950+ Python tests
cargo test           # Rust unit tests

# Lint
cargo clippy
cargo fmt
```

## Requirements

- Python >= 3.9
- Rust toolchain (stable)
- maturin >= 1.7

## License

MIT
