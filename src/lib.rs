pub mod core;

use pyo3::prelude::*;
use crate::core::series::PySeries;
use crate::core::dataframe::PyDataFrame;

#[pymodule]
fn rspandas(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySeries>()?;
    m.add_class::<PyDataFrame>()?;
    m.add_function(wrap_pyfunction!(crate::core::csv_io::read_csv_string, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::csv_io::write_csv_string, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::csv_io::read_csv_path, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::csv_io::write_csv_path, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::series::factorize, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::xlsx_io::read_xlsx, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::xlsx_io::write_xlsx, m)?)?;
    m.add_function(wrap_pyfunction!(crate::core::xlsx_io::xlsx_sheet_names, m)?)?;
    Ok(())
}
