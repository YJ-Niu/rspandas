pub mod core;

use crate::core::dataframe::PyDataFrame;
use crate::core::series::PySeries;
use pyo3::prelude::*;

#[pymodule]
fn rspandas(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySeries>()?;
    m.add_class::<PyDataFrame>()?;
    m.add_function(wrap_pyfunction!(crate::core::csv_io::read_csv_string, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::csv_io::write_csv_string, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::csv_io::read_csv_path, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::csv_io::write_csv_path, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::csv_io::read_csv_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::series::factorize, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::xlsx_io::read_xlsx, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::xlsx_io::write_xlsx, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::xlsx_io::write_xlsx_multi, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::xlsx_io::xlsx_sheet_names, m)?)?;
    // Parquet / Feather (Arrow IPC) 读写 —— 基于 Rust arrow/parquet crate，无需 pyarrow
    m.add_function(wrap_pyfunction!(crate::core::parquet_io::read_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::parquet_io::write_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::core::arrow_ipc_io::read_feather,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::core::arrow_ipc_io::write_feather,
        m
    )?)?;
    Ok(())
}
