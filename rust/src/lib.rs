pub mod core;

use crate::core::dataframe::PyDataFrame;
use crate::core::series::PySeries;
use pyo3::prelude::*;

#[pymodule]
fn rspandas(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySeries>()?;
    m.add_class::<PyDataFrame>()?;
    // CSV 读写
    m.add_function(wrap_pyfunction!(crate::core::io::csv::read_csv_string, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::io::csv::write_csv_string, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::io::csv::read_csv_path, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::io::csv::write_csv_path, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::io::csv::read_csv_chunks, m)?)?;
    // factorize 编码
    m.add_function(wrap_pyfunction!(crate::core::series::factorize, m)?)?;
    // Excel 读写
    m.add_function(wrap_pyfunction!(crate::core::io::excel::read_xlsx, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::io::excel::write_xlsx, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::core::io::excel::write_xlsx_multi,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::core::io::excel::xlsx_sheet_names,
        m
    )?)?;
    // Parquet / Feather (Arrow IPC) 读写 —— 基于 Rust arrow/parquet crate，无需 pyarrow
    m.add_function(wrap_pyfunction!(crate::core::io::parquet::read_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::core::io::parquet::write_parquet,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::core::io::arrow::read_feather, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::io::arrow::write_feather, m)?)?;
    // Arrow IPC bytes 桥接 —— 用于 to_arrow()/from_arrow()，替代 Python 层 list 中转
    m.add_function(wrap_pyfunction!(
        crate::core::io::arrow::to_arrow_ipc_bytes,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::core::io::arrow::from_arrow_ipc_bytes,
        m
    )?)?;
    Ok(())
}
