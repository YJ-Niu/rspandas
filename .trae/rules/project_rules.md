# rspandas 项目开发规则

## 项目概述

rspandas 是一个使用 **Rust + PyO3** 开发的高性能 Python 数据分析库，提供与 **pandas** 兼容的 API 接口。

- **Rust 层** (`src/`): 负责底层核心实现，包括 `PySeries` 和 `PyDataFrame` 两个核心类，以及 CSV/Excel IO 操作。
- **Python 层** (`python/rspandas/`): 负责公开 API 接口，方法参数默认值要全面，代理调用 Rust 底层实现。
- **构建工具**: 使用 `maturin` 构建 wheel，通过 `build_wheel.sh` 脚本构建并安装到 `.venv`。
- **依赖关系**: rspandas 与 rsnumpy 协同工作——rsnumpy 提供底层 ndarray 支持，rspandas 在其上构建 Series/DataFrame。rsnumpy 是 rspandas 的**必选依赖**（在 `pyproject.toml` 的 `dependencies` 中声明），Python 层通过 `import rsnumpy as rnp` 直接引用。

## 开发环境

- Python 虚拟环境: `.venv` (使用 uv 创建)
- Python 版本要求: `>=3.10` (pyproject.toml 声明)
- 构建命令: `./build_wheel.sh` (默认 release 模式，支持 `--debug` 参数)
- Rust 工具链: 由 `rust-toolchain.toml` 固定为 stable channel
- Rust edition: 2024 (在 `Cargo.toml` 中声明)

### Rust 依赖

| 依赖              | 版本   | 用途        |
| ----------------- | ------ | ----------- |
| `pyo3`            | 0.29.0 | Python 绑定 |
| `csv`             | 1.4.0  | CSV 读写    |
| `rayon`           | 1.12.0 | 并行计算    |
| `calamine`        | 0.36.1 | Excel 读取  |
| `rust_xlsxwriter` | 0.96.0 | Excel 写入  |

### 构建优化

`Cargo.toml` 的 release profile 配置：

- `opt-level = 3` / `lto = "fat"` / `codegen-units = 1` / `strip = "symbols"` / `panic = "abort"`

## 关键约定

### 1. Debug 目录

- 所有调试、分析、验证、测试代码及生成的结果必须放在 `/Users/user/Desktop/rust_project/rspandas/debug/` 目录中
- 禁止在项目根目录或其他位置创建零散的测试文件
- 调试时，添加一些 debug 信息，如打印数组形状、dtype、内存布局等，也要判断数据是否正确加载

### 2. 代码修改范围

- **可以修改**: `rspandas` 项目内的代码 (`src/`, `python/rspandas/`, `pyproject.toml`, `Cargo.toml`, `test/test_rf/` 等)
- **禁止修改**: 用户测试代码（如 `test/lotus/`）、`.venv` 环境中的任何代码
- 修改后使用 `build_wheel.sh` 构建并安装到 `.venv` 进行测试

### 3. 修复优先级

- 优先在 **Rust 层** (`src/`) 实现底层修复
- 然后在 **Python 层** (`python/rspandas/`) 补全和优化方法接口
- 确保 Python 接口的默认参数全面，与 pandas 行为一致

### 4. 依赖库处理

- **禁止**安装 `numpy` 或 `pandas` 或其他第三方 Python 数值库（如 `scipy` 的核心功能）
- 项目自身即为 pandas 兼容库，应完善自身实现而非引入 pandas
- 如需使用 pandas 来测试或验证功能与 rspandas 进行对比，验证完后，及时删除或注释掉相关代码，确保 rspandas 里没有引用的 pandas 代码

### 5. 代码风格

- Rust 代码: 遵循 `rustfmt` 和 `clippy` 规范（构建脚本会自动检查）
- Python 代码: 遵循 PEP 8 规范，使用 `black` 进行格式化（目标版本 `py313`）
- 提交前确保以下检查全部通过：
  - `cargo fmt --all -- --check`
  - `cargo clippy --all-targets -- -D warnings`
  - `black --check --target-version py313 python/`
- edition 2024 要求：`unsafe fn` 内的 unsafe 操作仍须显式 `unsafe {}` 块（unsafe_op_in_unsafe_fn）
- **注意**: `build_wheel.sh` 仅包含 `cargo fmt` 和 `cargo clippy` 检查；`black` 检查在 CI 中执行，本地修改 Python 文件后需手动运行 `black --target-version py313 python/` 格式化

### 6. 构建与测试

- `build_wheel.sh` 执行流程：
  1. 从 `Cargo.toml` 读取版本号，同步到 `pyproject.toml` 和 `__init__.py`
  2. 运行 `cargo fmt --all -- --check`（失败则中止）
  3. 运行 `cargo clippy --all-targets -- -D warnings`（失败则中止）
  4. 使用 `maturin build` 构建 wheel（优先使用 PATH 上的 maturin，否则通过 Python 模块调用）
  5. 安装到 `.venv`（优先使用 `uv pip install`，否则回退到 `pip install`）
- 每次修改后，立即执行 `./build_wheel.sh` 重新构建并安装
- 运行项目测试确保修复有效，若测试失败，继续修改直到通过

### 7. 长期优化原则（非强制）

- 当发现性能瓶颈来自 Python 层循环时，应评估将其迁移到 Rust 实现的可行性，但**本次修复不必强制执行**。
- 优先释放 GIL（`Python::allow_threads`）以提升并行计算性能。
- `Cargo.toml` 已启用 `rayon` 并行计算库，可在 Rust 层利用多核加速。

### 8. 缓冲协议架构（rsnumpy 层）

rspandas 依赖 rsnumpy 提供底层 ndarray 支持。rsnumpy 通过双层缓冲协议实现零拷贝数据共享，同时保证 dtype 精度：

- **内层 Rust 层** (`_core.ndarray`)：实现 PEP 3118 的 `__getbuffer__`/`__releasebuffer__`，直接暴露底层 `Array<f64, IxDyn>` 的连续内存，支持 shape/strides/format='d'/readonly。

- **外层 Python 层** (`rsnumpy.ndarray`)：实现 PEP 688 的 `__buffer__`/`__release_buffer__`（Python 3.12+），转发到内层 `_array` 的缓冲协议。**仅对 float64 dtype 暴露缓冲**——因为底层存储恒为 f64，若对 int/bool 等 dtype 也暴露 f64 缓冲，numpy 会优先按缓冲解读导致 dtype 失真。

- **回退路径**：int/bool/datetime64 等 dtype 不暴露缓冲，消费方自动回退到 `__array_interface__`（含 bytes 副本的 dtype 精确表示）。

- **兼容性**：Python 3.11 及更早版本不识别 `__buffer__`，`memoryview()` 会失败并自动回退到 `__array_interface__` bytes，行为安全。

### 9. 已知限制

- **f64 唯一存储**：rsnumpy 底层数组恒为 `Array<f64, IxDyn>`，int64/bool/datetime64 等 dtype 仅在 Python 层通过 `_dtype` 追踪。
- **int64 精度丢失**：大于 `2^53` 的整数会因 f64 存储限制而丢失精度，即使 `typestr` 报 `<i8`。
- **datetime64/timedelta64**：内部存储为 f64 纪元值，`dtype.kind` 返回 `'f'`，日期零拷贝需未来引入独立整型存储。
- **rsnumpy 必选依赖**：rspandas 的 `pyproject.toml` 声明 `dependencies = ["rsnumpy"]`，Python 层通过 `import rsnumpy as rnp` 直接引用，使用 `isinstance(data, rnp.ndarray)` 进行类型检测。

### 10. 测试规范

- **新增功能必须附带测试**：Rust 层新增公共函数应在其模块内增加 `#[cfg(test)]` 单元测试；Python 层新增 API 应在 `debug/` 或 `test/` 目录补充对应测试用例。
- **测试数据隔离**：测试用例使用的临时文件、随机种子必须固定或清理，避免跨测试污染。
- **兼容性测试**：若修改涉及与 numpy/pandas 的交互行为（如 `__array_ufunc__`、`__array_function__`），需同步验证 `test/test_rf/` 下的 scipy 兼容层是否仍通过。
- **禁止在测试代码中引入 pandas 依赖**：功能验证应优先使用 rspandas 自身 API，若必须与 pandas 对比，对比完成后及时删除或注释相关引入代码。

### 11. 错误处理与异常映射

- **Rust 层错误处理**：优先使用 `Result<T, E>` 传播错误，避免在核心计算逻辑中使用 `unwrap` 或 `expect`（测试代码除外）。
- **Python 异常映射**：Rust 层通过 PyO3 抛出的异常应与 pandas/numpy 的异常类型语义一致（如形状不匹配应映射为 `ValueError`，类型错误映射为 `TypeError`）。
- **错误信息清晰性**：异常消息应包含导致错误的关键上下文（如期望的 shape、实际的 shape、涉及的 dtype）。

### 12. Unsafe 代码与内存安全

- **最小化 unsafe 使用**：unsafe 代码块必须有明确的必要性说明（如调用 C BLAS、实现缓冲协议、与 Python C-API 交互）。
- **unsafe 代码审查**：新增 unsafe 代码必须通过 `cargo clippy` 的 `unsafe_op_in_unsafe_fn` 检查（edition 2024 强制要求）。
- **内存安全边界**：在 Rust 层直接操作原始指针时，必须确保指针生命周期、对齐要求和边界检查在调用 unsafe 之前已完成，禁止将未验证的 Python 输入直接传入 unsafe 块。

### 13. API 兼容性

- **pandas API 对齐**：Python 层公开方法的签名、参数名、默认值应与 pandas 对应方法保持一致。若因底层限制无法完全一致，需在注释中明确标注差异及原因。
- **行为一致性**：运算结果（如 `NaN` 传播规则、空数组的聚合行为、索引方式）应尽可能与 pandas/numpy 默认行为一致，避免用户迁移时出现意外差异。
- **破坏性变更管控**：若必须修改公开 API 的行为或签名，需评估对现有测试及兼容层的影响，并优先提供过渡方案或明确文档说明。

## 项目结构

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

## Rust 模块注册

`lib.rs` 通过 `#[pymodule]` 注册以下符号：

| 类型/函数          | 来源模块          | 说明                    |
| ------------------ | ----------------- | ----------------------- |
| `PySeries`         | `core::series`    | Rust 端 Series 类       |
| `PyDataFrame`      | `core::dataframe` | Rust 端 DataFrame 类    |
| `read_csv_string`  | `core::csv_io`    | 从字符串读取 CSV        |
| `write_csv_string` | `core::csv_io`    | 写入 CSV 为字符串       |
| `read_csv_path`    | `core::csv_io`    | 从文件路径读取 CSV      |
| `write_csv_path`   | `core::csv_io`    | 写入 CSV 到文件路径     |
| `factorize`        | `core::series`    | 因子编码                |
| `read_xlsx`        | `core::xlsx_io`   | 读取 Excel 文件         |
| `write_xlsx`       | `core::xlsx_io`   | 写入 Excel 文件         |
| `write_xlsx_multi` | `core::xlsx_io`   | 写入多 sheet Excel      |
| `xlsx_sheet_names` | `core::xlsx_io`   | 获取 Excel sheet 名列表 |
