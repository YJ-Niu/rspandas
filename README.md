<p align="center">
  <h1 align="center">rspandas</h1>
  <p align="center">
    <strong>pandas-compatible API, Rust-powered performance</strong>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/rspandas?label=PyPI" alt="PyPI">
  <img src="https://img.shields.io/pypi/pyversions/rspandas" alt="Python">
  <img src="https://img.shields.io/github/actions/workflow/status/USERNAME/rspandas/publish.yml?label=CI" alt="CI">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</p>

---

**rspandas** is a drop-in pandas replacement with a Rust backend. Write the same pandas code you know—filtering, grouping, window functions, reshaping—but get near-native performance thanks to columnar storage, vectorized operations, and multi-threaded parallelism via [Rayon](https://github.com/rayon-rs/rayon).

```python
import rspandas as rpd

df = rpd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
print(df.describe())
print(df.groupby("a").sum())
```

## Highlights

- **95%+ pandas API coverage** — Series, DataFrame, GroupBy, window functions, reshaping, time series
- **Rust core** — columnar storage, vectorized computation, Rayon parallel iterators
- **Multi-platform wheels** — pre-built binaries for Linux (x86_64 / arm64), macOS (Intel / Apple Silicon), Windows (x64 / x86)
- **Zero required Python dependencies** — one compiled extension, no NumPy/PyArrow required at runtime
- **Rich I/O** — CSV, Excel (native Rust), JSON, Parquet, SQL, Pickle, Feather
- **Full type system** — int64, float64, bool, string, category, datetime, timedelta, period
- **950+ tests** — comprehensive pytest suite validating pandas compatibility

## Installation

Requires **Python >= 3.9**.

```bash
pip install rspandas
```

Or build from source:

```bash
pip install maturin
maturin build --release
pip install target/wheels/rspandas-*.whl
```

## Quick Start

### Series

```python
import rspandas as rpd

s = rpd.Series([1, 2, 3, 4, 5], name="values")
s.head(3)           # first 3 rows
s.sum()             # 15
s.mean()            # 3.0
s.std()             # ~1.58
s.describe()        # summary stats

# Missing values
s2 = rpd.Series([1, None, 3])
s2.fillna(0)        # replace None → 0
s2.dropna()         # remove None rows

# String operations
s3 = rpd.Series(["hello", "world"])
s3.str.upper()      # ["HELLO", "WORLD"]
s3.str.contains("ell")  # [True, False]
```

### DataFrame

```python
df = rpd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age":  [25, 30, 35],
    "score": [88.5, 92.0, 79.3],
})

df.shape            # (3, 3)
df.dtypes           # {"name": "object", "age": "int64", "score": "float64"}
df.head(2)
df["age"]           # Series
df[df["age"] > 26]  # filter rows
df.sort_values("score", ascending=False)
```

### GroupBy

```python
df = rpd.DataFrame({"team": ["A", "A", "B", "B", "C"], "score": [10, 20, 30, 40, 50]})

df.groupby("team").sum()       # sum by group
df.groupby("team").mean()      # mean by group
df.groupby("team").agg("std")  # std by group
df.groupby("team").rank()      # rank within group
```

### Window Functions

```python
s = rpd.Series([1, 2, 3, 4, 5])

s.rolling(3).mean()    # [NaN, NaN, 2.0, 3.0, 4.0]
s.rolling(3).sum()     # [NaN, NaN, 6.0, 9.0, 12.0]
s.expanding().sum()    # [1.0, 3.0, 6.0, 10.0, 15.0]
s.ewm(span=3).mean()   # exponentially weighted
```

### Time Series

```python
dates = rpd.date_range("2024-01-01", periods=5, freq="D")
ts = rpd.Series([1, 2, 3, 4, 5], index=dates)

ts.shift(1)       # lag
ts.diff()         # difference
ts.pct_change()   # percent change
ts.cumsum()       # [1, 3, 6, 10, 15]

# DatetimeSeries
ds = rpd.to_datetime(["2024-01-15", "2024-06-15"])
ds.dt.year        # [2024, 2024]
ds.dt.month_name  # ["January", "June"]
```

### I/O

```python
# CSV (native Rust)
df = rpd.read_csv("data.csv")
df.to_csv("output.csv")

# Excel (native Rust via calamine + rust_xlsxwriter)
df = rpd.read_excel("data.xlsx")
df.to_excel("output.xlsx")

# JSON
df = rpd.read_json("data.json")
df.to_json("output.json")

# Parquet / Feather (requires pyarrow)
df = rpd.read_parquet("data.parquet")
df.to_parquet("output.parquet")
df = rpd.read_feather("data.feather")

# SQL (requires sqlalchemy)
df = rpd.read_sql("SELECT * FROM table", engine)
df.to_sql("table_name", engine)

# Pickle
df = rpd.read_pickle("data.pkl")
df.to_pickle("output.pkl")
```

### Merge & Reshape

```python
df1 = rpd.DataFrame({"key": ["a", "b", "c"], "v1": [1, 2, 3]})
df2 = rpd.DataFrame({"key": ["b", "c", "d"], "v2": [4, 5, 6]})

df1.merge(df2, on="key", how="inner")   # inner join
df1.merge(df2, on="key", how="left")    # left join
rpd.concat([df1, df2], axis=0)          # row-wise

# Melt / Pivot
df.melt(id_vars=["a"])
df.pivot(index="x", columns="y", values="v")
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
rpd.factorize(["a", "b", "a", "c"])          # (codes, categories)
rpd.to_numeric(["1", "2", "x"], errors="coerce")  # [1, 2, None]
rpd.get_dummies(df)
rpd.cut(s, bins=[0, 10, 20, 30])
rpd.crosstab(df["a"], df["b"])
```

## Performance

rspandas leverages multi-threaded parallelism via [Rayon](https://github.com/rayon-rs/rayon) across all heavy operations:

| Operation         | Parallelized methods                                                     |
| ----------------- | ------------------------------------------------------------------------ |
| Aggregation       | `sum`, `mean`, `min`, `max`, `std`, `var`, `nunique`, `any`, `all`       |
| Filtering         | `filter`, `dropna`, `isnull`, `notnull`, `fillna`                        |
| I/O               | CSV type inference & parsing, XLSX column conversion                     |
| DataFrame         | `head`, `tail`, `filter_rows`, `dropna_rows`, `fillna_rows`, `to_string` |
| String conversion | `to_string_vec`, `columns_to_string`                                     |

The Rust core uses columnar `Vec<Option<T>>` storage with `opt-level=3` and `lto=true` in release builds.

## Architecture

```
User Code (Python)
    │  import rspandas as rpd
    ▼
┌──────────────────────────────────────┐
│  python/rspandas/                    │  ← Python API layer
│  series.py / dataframe.py / ...      │     (pandas-compatible signatures)
└────────────────┬─────────────────────┘
                 │  PyO3 FFI
                 ▼
┌──────────────────────────────────────┐
│  rspandas._rust  (compiled .so/.dylib)│  ← Native module
├──────────────────────────────────────┤
│  PySeries / PyDataFrame  (#[pyclass])│
├──────────────────────────────────────┤
│  Series / DataFrame  (Rust structs)  │  ← Rust core
├──────────────────────────────────────┤
│  ColumnData  (Int/Float/Bool/String/…)│
└──────────────────────────────────────┘
```

- **Python layer**: parameter validation, type inference, display formatting, API compatibility
- **Rust core**: columnar storage, vectorized computation, parallel filtering & aggregation
- **PyO3 bridge**: zero-copy Python/Rust type conversion

## API Coverage

### Top-level Functions

`read_csv`, `to_csv`, `read_excel`, `to_excel`, `read_parquet`, `to_parquet`, `read_json`, `to_json`, `read_sql`, `to_sql`, `read_pickle`, `to_pickle`, `read_feather`, `to_feather`, `concat`, `merge`, `get_dummies`, `cut`, `qcut`, `crosstab`, `factorize`, `to_datetime`, `to_timedelta`, `to_numeric`, `date_range`, `timedelta_range`, `period_range`, `bdate_range`, `infer_freq`, `set_option`, `get_option`, `reset_option`

### Series

Properties: `shape`, `dtype`, `name`, `values`, `index`, `size`, `empty`, `nbytes`

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
- [maturin](https://github.com/PyO3/maturin) >= 1.7

## License

MIT
