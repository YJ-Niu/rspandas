//! DataFrame: 列存储的多列数据结构 + PyO3 绑定

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use pyo3::types::PyAnyMethods;
use pyo3::IntoPyObject;

use super::series::{PySeries, Series};
use super::dtype::DType;

#[derive(Debug, Clone)]
pub struct DataFrame {
    pub columns: Vec<String>,
    pub data: Vec<Series>,
}

impl DataFrame {
    pub fn new_empty() -> Self {
        Self { columns: Vec::new(), data: Vec::new() }
    }

    pub fn from_series(columns: Vec<String>, data: Vec<Series>) -> PyResult<Self> {
        if columns.len() != data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "columns len {} != data len {}", columns.len(), data.len()
            )));
        }
        // 校验列名去重
        let mut seen = std::collections::HashSet::new();
        for c in &columns {
            if !seen.insert(c.clone()) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "duplicate column name: {}", c
                )));
            }
        }
        // 校验每列长度一致
        if let Some(first) = data.first() {
            let n = first.len();
            for (i, s) in data.iter().enumerate() {
                if s.len() != n {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "column '{}' length {} != row length {}",
                        columns[i], s.len(), n
                    )));
                }
            }
        }
        Ok(Self { columns, data })
    }

    pub fn nrows(&self) -> usize {
        self.data.first().map(|s| s.len()).unwrap_or(0)
    }
    pub fn ncols(&self) -> usize { self.columns.len() }
    pub fn shape(&self) -> (usize, usize) { (self.nrows(), self.ncols()) }
    pub fn column_names(&self) -> &[String] { &self.columns }

    pub fn dtypes(&self) -> Vec<(&str, &'static str)> {
        self.data.iter().zip(self.columns.iter())
            .map(|(s, c)| (c.as_str(), s.dtype_name()))
            .collect()
    }

    pub fn get_column(&self, name: &str) -> Option<&Series> {
        self.columns.iter().position(|c| c == name)
            .and_then(|i| self.data.get(i))
    }

    pub fn get_column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }

    pub fn get_column_at(&self, idx: usize) -> Option<&Series> {
        self.data.get(idx)
    }

    pub fn head(&self, n: usize) -> DataFrame {
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.head(n)).collect();
        DataFrame { columns: self.columns.clone(), data: n_data }
    }

    pub fn tail(&self, n: usize) -> DataFrame {
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.tail(n)).collect();
        DataFrame { columns: self.columns.clone(), data: n_data }
    }

    pub fn filter_rows(&self, mask: &[bool]) -> PyResult<DataFrame> {
        if mask.len() != self.nrows() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "mask length {} != nrows {}", mask.len(), self.nrows()
            )));
        }
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.filter(mask)).collect();
        Ok(DataFrame { columns: self.columns.clone(), data: n_data })
    }

    /// 删除任意一列为 None 的行 (axis=0)
    pub fn dropna_rows(&self) -> DataFrame {
        if self.nrows() == 0 {
            return self.clone();
        }
        let nrows = self.nrows();
        // 并行计算每列的非空 mask，然后合并 (任意列 None 则整行删除)
        let keep: Vec<bool> = (0..nrows)
            .into_par_iter()
            .map(|i| self.data.iter().all(|s| {
                match &s.data {
                    super::dtype::ColumnData::Int(v) => v[i].is_some(),
                    super::dtype::ColumnData::Float(v) => v[i].is_some(),
                    super::dtype::ColumnData::Bool(v) => v[i].is_some(),
                    super::dtype::ColumnData::String(v) => v[i].is_some(),
                    super::dtype::ColumnData::Categorical(c) => c.codes[i].is_some(),
                }
            }))
            .collect();
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.filter(&keep)).collect();
        DataFrame { columns: self.columns.clone(), data: n_data }
    }

    /// 填充整个 DataFrame 中所有列的 None 值
    pub fn fillna_rows(&self, fill_dict: &std::collections::HashMap<String, FillValue>) -> PyResult<DataFrame> {
        let n_data: Vec<Series> = self.columns.par_iter().zip(self.data.par_iter()).map(|(col, series)| {
            if let Some(v) = fill_dict.get(col) {
                match (v, series.dtype()) {
                    (FillValue::Int(x), DType::Int64) => series.fillna_i64(*x),
                    (FillValue::Float(x), DType::Float64) => series.fillna_f64(*x),
                    (FillValue::Bool(x), DType::Bool) => series.fillna_bool(*x),
                    (FillValue::String(x), DType::Object) => series.fillna_string(x),
                    _ => series.clone(),
                }
            } else {
                series.clone()
            }
        }).collect();
        Ok(DataFrame { columns: self.columns.clone(), data: n_data })
    }
}

/// DataFrame fillna 用的填充值类型
#[derive(Debug, Clone)]
pub enum FillValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

// =====================================================================
// PyO3 绑定
// =====================================================================

#[pyclass(name = "_DataFrame", module = "rspandas", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyDataFrame {
    pub inner: DataFrame,
}

#[pymethods]
impl PyDataFrame {
    /// 构造: 接受 columns (list[str]) 和 series (list[_Series])
    #[new]
    fn new(columns: Vec<String>, series: Vec<PySeries>) -> PyResult<Self> {
        let data: Vec<Series> = series.into_iter().map(|s| s.inner).collect();
        let inner = DataFrame::from_series(columns, data)?;
        Ok(PyDataFrame { inner })
    }

    // ---------- 属性 ----------

    #[getter]
    fn nrows(&self) -> usize { self.inner.nrows() }
    #[getter]
    fn ncols(&self) -> usize { self.inner.ncols() }
    #[getter]
    fn shape(&self) -> (usize, usize) { self.inner.shape() }
    #[getter]
    fn size(&self) -> usize {
        self.inner.nrows() * self.inner.ncols()
    }
    #[getter]
    fn empty(&self) -> bool { self.inner.nrows() == 0 }

    #[getter]
    fn columns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        Ok(PyList::new(py, self.inner.columns.iter().map(|s| s.as_str()))?)
    }

    #[getter]
    fn dtypes<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let d = PyDict::new(py);
        for (name, dt) in self.inner.dtypes() {
            d.set_item(name, dt).unwrap();
        }
        d
    }

    // ---------- 子集 ----------

    /// 按列名取列 -> _Series
    fn get_column(&self, name: &str) -> PyResult<PySeries> {
        match self.inner.get_column(name) {
            Some(s) => Ok(PySeries { inner: s.clone() }),
            None => Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "column not found: {}", name
            ))),
        }
    }

    /// 按索引取列 -> _Series
    fn get_column_at(&self, idx: usize) -> PyResult<PySeries> {
        match self.inner.get_column_at(idx) {
            Some(s) => Ok(PySeries { inner: s.clone() }),
            None => Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "column index out of range: {}", idx
            ))),
        }
    }

    /// 列名 -> 索引
    fn column_index(&self, name: &str) -> Option<usize> {
        self.inner.get_column_index(name)
    }

    fn head(&self, n: usize) -> Self {
        PyDataFrame { inner: self.inner.head(n) }
    }
    fn tail(&self, n: usize) -> Self {
        PyDataFrame { inner: self.inner.tail(n) }
    }
    fn filter_rows(&self, mask: Vec<bool>) -> PyResult<Self> {
        Ok(PyDataFrame { inner: self.inner.filter_rows(&mask)? })
    }

    // ---------- 缺失值 ----------

    fn dropna(&self) -> Self {
        PyDataFrame { inner: self.inner.dropna_rows() }
    }

    /// fillna: 接受 dict {col_name: value}，只填充指定列
    fn fillna(&self, fill_dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut converted = std::collections::HashMap::new();
        for (key, val) in fill_dict.iter() {
            let col: String = key.extract()?;
            // 优先尝试 bool，再 int，再 float，最后 string
            if let Ok(b) = val.extract::<bool>() {
                converted.insert(col, FillValue::Bool(b));
            } else if let Ok(i) = val.extract::<i64>() {
                converted.insert(col, FillValue::Int(i));
            } else if let Ok(f) = val.extract::<f64>() {
                converted.insert(col, FillValue::Float(f));
            } else if let Ok(s) = val.extract::<String>() {
                converted.insert(col, FillValue::String(s));
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "unsupported fill value type for column '{}'", col
                )));
            }
        }
        Ok(PyDataFrame { inner: self.inner.fillna_rows(&converted)? })
    }

    // ---------- 显示辅助 ----------

    /// 逐行构造 dict 列表 (用于 Python 端显示)
    fn to_rows<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let rows = PyList::empty(py);
        let nrows = self.inner.nrows();
        for i in 0..nrows {
            let row = PyDict::new(py);
            for (col_name, series) in self.inner.columns.iter().zip(self.inner.data.iter()) {
                let val: pyo3::Py<pyo3::PyAny> = match &series.data {
                    super::dtype::ColumnData::Int(v) => match v.get(i) {
                        Some(Some(n)) => (*n).into_pyobject(py).unwrap().into_any().unbind(),
                        _ => py.None(),
                    },
                    super::dtype::ColumnData::Float(v) => match v.get(i) {
                        Some(Some(n)) => (*n).into_pyobject(py).unwrap().into_any().unbind(),
                        _ => py.None(),
                    },
                    super::dtype::ColumnData::Bool(v) => match v.get(i) {
                        Some(Some(n)) => (*n).into_pyobject(py).unwrap().as_any().clone().unbind(),
                        _ => py.None(),
                    },
                    super::dtype::ColumnData::String(v) => match v.get(i) {
                        Some(Some(s)) => s.clone().into_pyobject(py).unwrap().into_any().unbind(),
                        _ => py.None(),
                    },
                    super::dtype::ColumnData::Categorical(c) => match c.codes.get(i) {
                        Some(Some(code_idx)) => {
                            let cat_str = c.categories.get(*code_idx as usize)
                                .cloned()
                                .unwrap_or_else(|| "NaN".to_string());
                            cat_str.into_pyobject(py).unwrap().into_any().unbind()
                        }
                        _ => py.None(),
                    },
                };
                row.set_item(col_name, val).unwrap();
            }
            rows.append(row).unwrap();
        }
        rows
    }

    /// 每列的 string 列表 (用于显示)
    fn columns_to_string<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let d = PyDict::new(py);
        let pairs: Vec<(&str, Vec<String>)> = self.inner.columns.par_iter()
            .zip(self.inner.data.par_iter())
            .map(|(col_name, series)| {
                (col_name.as_str(), series.to_string_vec())
            })
            .collect();
        for (col_name, svec) in pairs {
            let pylist: Bound<'_, PyList> = PyList::new(py, svec.iter().map(|s| s.as_str())).unwrap();
            d.set_item(col_name, pylist).unwrap();
        }
        d
    }

    /// 各列 dtype 的 dict
    fn dtypes_dict<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        self.dtypes(py)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataframe_basic() {
        let s1 = Series::new_int(Some("a".to_string()), vec![Some(1), Some(2), Some(3)]);
        let s2 = Series::new_string(Some("b".to_string()), vec![Some("x".to_string()), Some("y".to_string()), Some("z".to_string())]);
        let df = DataFrame::from_series(
            vec!["a".to_string(), "b".to_string()],
            vec![s1, s2],
        ).unwrap();
        assert_eq!(df.shape(), (3, 2));
        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 2);
    }

    #[test]
    fn test_dataframe_head_tail() {
        let s1 = Series::new_int(None, vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        let df = DataFrame::from_series(vec!["a".to_string()], vec![s1]).unwrap();
        assert_eq!(df.head(2).nrows(), 2);
        assert_eq!(df.tail(2).nrows(), 2);
    }

    #[test]
    fn test_dataframe_filter() {
        let s1 = Series::new_int(None, vec![Some(1), Some(2), Some(3), Some(4)]);
        let df = DataFrame::from_series(vec!["a".to_string()], vec![s1]).unwrap();
        let filtered = df.filter_rows(&[true, false, true, false]).unwrap();
        assert_eq!(filtered.nrows(), 2);
    }

    #[test]
    fn test_dataframe_duplicate_col() {
        let s1 = Series::new_int(None, vec![Some(1)]);
        let s2 = Series::new_int(None, vec![Some(2)]);
        let r = DataFrame::from_series(
            vec!["a".to_string(), "a".to_string()],
            vec![s1, s2],
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_dataframe_shape_mismatch() {
        let s1 = Series::new_int(None, vec![Some(1), Some(2)]);
        let s2 = Series::new_int(None, vec![Some(3)]);
        let r = DataFrame::from_series(
            vec!["a".to_string(), "b".to_string()],
            vec![s1, s2],
        );
        assert!(r.is_err());
    }

    // 防止 DType 未使用警告
    #[test]
    fn test_dtype_compile() {
        let _ = DType::Int64;
    }
}
