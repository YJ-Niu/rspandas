//! rspandas 核心模块
//!
//! 按功能拆分为子模块：
//! - :mod:`.dtype`：数据类型 (DType / ColumnData)
//! - :mod:`.series`：Series 实现 (Rust 端 + PyO3 绑定)
//! - :mod:`.dataframe`：DataFrame 实现 (Rust 端 + PyO3 绑定)
//! - :mod:`.io`：按文件格式拆分的读写实现 (csv / excel / parquet / arrow)

pub mod dataframe;
pub mod dtype;
pub mod io;
pub mod series;
