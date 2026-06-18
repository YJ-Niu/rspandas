pub mod core;

use pyo3::prelude::*;
use crate::core::series::PySeries;
use crate::core::dataframe::PyDataFrame;

#[pymodule]
fn rspandas(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySeries>()?;
    m.add_class::<PyDataFrame>()?;
    Ok(())
}
