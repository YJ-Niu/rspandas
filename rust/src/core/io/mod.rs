//! IO 子模块：按文件格式拆分的读写实现
//!
//! - :mod:`.csv`：CSV 读写
//! - :mod:`.excel`：Excel 读写
//! - :mod:`.parquet`：Parquet 读写
//! - :mod:`.arrow`：Arrow IPC (Feather) 读写

pub mod arrow;
pub mod csv;
pub mod excel;
pub mod parquet;
