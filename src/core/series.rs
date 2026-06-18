//! Series: 单列数据结构 + PyO3 绑定
//!
//! PyO3 0.29 API: PyAnyMethods trait 提供 downcast/is_instance_of 等

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBoolMethods, PyFloat, PyInt, PyList, PyString};

use super::dtype::{ColumnData, DType};

/// Series: 带名字的单列
#[derive(Debug, Clone)]
pub struct Series {
    pub name: Option<String>,
    pub data: ColumnData,
}

impl Series {
    // ---------- 构造器 ----------

    pub fn new_int(name: Option<String>, v: Vec<Option<i64>>) -> Self {
        Self { name, data: ColumnData::Int(v) }
    }
    pub fn new_float(name: Option<String>, v: Vec<Option<f64>>) -> Self {
        Self { name, data: ColumnData::Float(v) }
    }
    pub fn new_bool(name: Option<String>, v: Vec<Option<bool>>) -> Self {
        Self { name, data: ColumnData::Bool(v) }
    }
    pub fn new_string(name: Option<String>, v: Vec<Option<String>>) -> Self {
        Self { name, data: ColumnData::String(v) }
    }

    pub fn new_null(name: Option<String>, dtype: DType, len: usize) -> Self {
        let data = match dtype {
            DType::Int64 => ColumnData::Int(vec![None; len]),
            DType::Float64 => ColumnData::Float(vec![None; len]),
            DType::Bool => ColumnData::Bool(vec![None; len]),
            DType::Object => ColumnData::String(vec![None; len]),
        };
        Self { name, data }
    }

    // ---------- 属性 ----------

    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn shape(&self) -> (usize,) { (self.data.len(),) }
    pub fn dtype(&self) -> DType { self.data.dtype() }
    pub fn dtype_name(&self) -> &'static str { self.data.dtype_name() }
    pub fn name(&self) -> Option<&str> { self.name.as_deref() }
    pub fn set_name(&mut self, name: Option<String>) { self.name = name; }
    pub fn nbytes(&self) -> usize {
        match &self.data {
            ColumnData::Int(v) => v.len() * std::mem::size_of::<Option<i64>>(),
            ColumnData::Float(v) => v.len() * std::mem::size_of::<Option<f64>>(),
            ColumnData::Bool(v) => v.len() * std::mem::size_of::<Option<bool>>(),
            ColumnData::String(v) => v
                .iter()
                .map(|s| s.as_ref().map(|x| x.len()).unwrap_or(0))
                .sum::<usize>()
                + v.len() * std::mem::size_of::<Option<String>>(),
        }
    }

    // ---------- 切片 ----------

    pub fn slice(&self, start: usize, end: usize) -> Series {
        Self { name: self.name.clone(), data: self.data.slice(start, end) }
    }
    pub fn head(&self, n: usize) -> Series {
        Self { name: self.name.clone(), data: self.data.head(n) }
    }
    pub fn tail(&self, n: usize) -> Series {
        Self { name: self.name.clone(), data: self.data.tail(n) }
    }
    pub fn filter(&self, mask: &[bool]) -> Series {
        Self { name: self.name.clone(), data: self.data.filter(mask) }
    }

    // ---------- 比较 (返回 mask) ----------

    pub fn eq_scalar_i64(&self, v: i64) -> Vec<bool> {
        match &self.data {
            ColumnData::Int(col) => col.iter().map(|x| matches!(x, Some(x) if *x == v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn eq_scalar_f64(&self, v: f64) -> Vec<bool> {
        match &self.data {
            ColumnData::Float(col) => col.iter().map(|x| matches!(x, Some(x) if *x == v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn eq_scalar_bool(&self, v: bool) -> Vec<bool> {
        match &self.data {
            ColumnData::Bool(col) => col.iter().map(|x| matches!(x, Some(x) if *x == v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn eq_scalar_str(&self, v: &str) -> Vec<bool> {
        match &self.data {
            ColumnData::String(col) => col.iter().map(|x| matches!(x, Some(x) if x == v)).collect(),
            _ => vec![false; self.len()],
        }
    }

    pub fn gt_scalar_i64(&self, v: i64) -> Vec<bool> {
        match &self.data {
            ColumnData::Int(col) => col.iter().map(|x| matches!(x, Some(x) if *x > v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn gt_scalar_f64(&self, v: f64) -> Vec<bool> {
        match &self.data {
            ColumnData::Float(col) => col.iter().map(|x| matches!(x, Some(x) if *x > v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn lt_scalar_i64(&self, v: i64) -> Vec<bool> {
        match &self.data {
            ColumnData::Int(col) => col.iter().map(|x| matches!(x, Some(x) if *x < v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn lt_scalar_f64(&self, v: f64) -> Vec<bool> {
        match &self.data {
            ColumnData::Float(col) => col.iter().map(|x| matches!(x, Some(x) if *x < v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn ge_scalar_i64(&self, v: i64) -> Vec<bool> {
        match &self.data {
            ColumnData::Int(col) => col.iter().map(|x| matches!(x, Some(x) if *x >= v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn ge_scalar_f64(&self, v: f64) -> Vec<bool> {
        match &self.data {
            ColumnData::Float(col) => col.iter().map(|x| matches!(x, Some(x) if *x >= v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn le_scalar_i64(&self, v: i64) -> Vec<bool> {
        match &self.data {
            ColumnData::Int(col) => col.iter().map(|x| matches!(x, Some(x) if *x <= v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn le_scalar_f64(&self, v: f64) -> Vec<bool> {
        match &self.data {
            ColumnData::Float(col) => col.iter().map(|x| matches!(x, Some(x) if *x <= v)).collect(),
            _ => vec![false; self.len()],
        }
    }

    // ---------- 聚合 ----------

    pub fn count(&self) -> usize { self.data.count_non_null() }

    pub fn sum_i64(&self) -> Option<i64> {
        if let ColumnData::Int(v) = &self.data {
            Some(v.iter().filter_map(|x| *x).sum())
        } else { None }
    }
    pub fn sum_f64(&self) -> Option<f64> {
        if let ColumnData::Float(v) = &self.data {
            Some(v.iter().filter_map(|x| *x).sum())
        } else { None }
    }
    pub fn sum_bool(&self) -> usize {
        if let ColumnData::Bool(v) = &self.data {
            v.iter().filter(|x| matches!(x, Some(true))).count()
        } else { 0 }
    }

    pub fn mean(&self) -> Option<f64> {
        let cnt = self.count();
        if cnt == 0 { return None; }
        match &self.data {
            ColumnData::Int(v) => {
                let s: i64 = v.iter().filter_map(|x| *x).sum();
                Some(s as f64 / cnt as f64)
            }
            ColumnData::Float(v) => {
                let s: f64 = v.iter().filter_map(|x| *x).sum();
                Some(s / cnt as f64)
            }
            _ => None,
        }
    }

    pub fn min_i64(&self) -> Option<i64> {
        if let ColumnData::Int(v) = &self.data {
            v.iter().filter_map(|x| *x).min()
        } else { None }
    }
    pub fn min_f64(&self) -> Option<f64> {
        if let ColumnData::Float(v) = &self.data {
            v.iter().filter_map(|x| *x).reduce(|a, b| a.min(b))
        } else { None }
    }
    pub fn min_str(&self) -> Option<String> {
        if let ColumnData::String(v) = &self.data {
            v.iter().filter_map(|x| x.clone()).min()
        } else { None }
    }
    pub fn max_i64(&self) -> Option<i64> {
        if let ColumnData::Int(v) = &self.data {
            v.iter().filter_map(|x| *x).max()
        } else { None }
    }
    pub fn max_f64(&self) -> Option<f64> {
        if let ColumnData::Float(v) = &self.data {
            v.iter().filter_map(|x| *x).reduce(|a, b| a.max(b))
        } else { None }
    }
    pub fn max_str(&self) -> Option<String> {
        if let ColumnData::String(v) = &self.data {
            v.iter().filter_map(|x| x.clone()).max()
        } else { None }
    }

    pub fn std(&self) -> Option<f64> { self.var().map(|v| v.sqrt()) }
    pub fn var(&self) -> Option<f64> {
        let m = self.mean()?;
        let cnt = self.count();
        if cnt == 0 { return None; }
        let s = match &self.data {
            ColumnData::Int(v) => v
                .iter().filter_map(|x| *x)
                .map(|x| (x as f64 - m).powi(2))
                .sum::<f64>(),
            ColumnData::Float(v) => v
                .iter().filter_map(|x| *x)
                .map(|x| (x - m).powi(2))
                .sum::<f64>(),
            _ => return None,
        };
        Some(s / cnt as f64)
    }
    pub fn median(&self) -> Option<f64> {
        let mut vs: Vec<f64> = match &self.data {
            ColumnData::Int(v) => v.iter().filter_map(|x| *x).map(|x| x as f64).collect(),
            ColumnData::Float(v) => v.iter().filter_map(|x| *x).collect(),
            _ => return None,
        };
        if vs.is_empty() { return None; }
        vs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = vs.len();
        if n % 2 == 1 { Some(vs[n / 2]) }
        else { Some((vs[n / 2 - 1] + vs[n / 2]) / 2.0) }
    }

    pub fn any(&self) -> Option<bool> {
        if let ColumnData::Bool(v) = &self.data {
            Some(v.iter().any(|x| matches!(x, Some(true))))
        } else { None }
    }
    pub fn all(&self) -> Option<bool> {
        if let ColumnData::Bool(v) = &self.data {
            let mut saw_true = false;
            for x in v {
                match x {
                    Some(true) => saw_true = true,
                    Some(false) => return Some(false),
                    None => {}
                }
            }
            Some(saw_true)
        } else { None }
    }

    // ---------- 缺失值 ----------

    pub fn isnull(&self) -> Vec<bool> { self.data.isnull() }
    pub fn notnull(&self) -> Vec<bool> { self.data.notnull() }

    pub fn dropna(&self) -> Series {
        Self { name: self.name.clone(), data: self.data.dropna() }
    }

    pub fn fillna_i64(&self, v: i64) -> Series {
        Self { name: self.name.clone(), data: self.data.fillna_i64(v) }
    }
    pub fn fillna_f64(&self, v: f64) -> Series {
        Self { name: self.name.clone(), data: self.data.fillna_f64(v) }
    }
    pub fn fillna_bool(&self, v: bool) -> Series {
        Self { name: self.name.clone(), data: self.data.fillna_bool(v) }
    }
    pub fn fillna_string(&self, v: &str) -> Series {
        Self { name: self.name.clone(), data: self.data.fillna_string(v) }
    }

    // ---------- 唯一值 ----------

    pub fn unique(&self) -> Series {
        Self { name: self.name.clone(), data: self.data.unique() }
    }

    pub fn nunique(&self) -> usize {
        match &self.data {
            ColumnData::Int(v) => v.iter().filter_map(|x| *x).collect::<std::collections::HashSet<_>>().len(),
            ColumnData::Float(v) => {
                let mut seen: Vec<f64> = Vec::new();
                for x in v.iter().filter_map(|x| *x) {
                    if !seen.iter().any(|y| y == &x) {
                        seen.push(x);
                    }
                }
                seen.len()
            }
            ColumnData::Bool(v) => v.iter().filter_map(|x| *x).collect::<std::collections::HashSet<_>>().len(),
            ColumnData::String(v) => v.iter().filter_map(|x| x.clone()).collect::<std::collections::HashSet<_>>().len(),
        }
    }

    /// 转换为字符串列表 (None -> "NaN") - 给 DataFrame 显示用
    pub fn to_string_vec(&self) -> Vec<String> {
        match &self.data {
            ColumnData::Int(v) => v
                .iter()
                .map(|x| match x { Some(n) => n.to_string(), None => "NaN".to_string() })
                .collect(),
            ColumnData::Float(v) => v
                .iter()
                .map(|x| match x { Some(n) => n.to_string(), None => "NaN".to_string() })
                .collect(),
            ColumnData::Bool(v) => v
                .iter()
                .map(|x| match x { Some(b) => b.to_string(), None => "NaN".to_string() })
                .collect(),
            ColumnData::String(v) => v
                .iter()
                .map(|x| match x { Some(s) => s.clone(), None => "NaN".to_string() })
                .collect(),
        }
    }
}

// =====================================================================
// PyO3 绑定
// =====================================================================

/// Python 端 _Series，包装 Rust Series
#[pyclass(name = "_Series", module = "rspandas._rust")]
#[derive(Debug, Clone)]
pub struct PySeries {
    pub inner: Series,
}

#[pymethods]
impl PySeries {
    /// 构造: data 必须是 list，每个元素是 None/bool/int/float/str
    #[new]
    fn new(data: &Bound<'_, PyAny>, name: Option<String>) -> PyResult<Self> {
        use pyo3::types::PyAnyMethods;

        let py = data.py();
        let pylist: &Bound<'_, PyList> = data.cast::<PyList>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Series data must be a list")
        })?;

        // 类型推断: bool -> int -> float -> str (按"最宽"覆盖)
        let mut all_bool = true;
        let mut all_int = true;
        let mut all_float = true;
        let mut all_str = true;
        let mut any_non_null = false;

        for item in pylist.iter() {
            if item.is_none() { continue; }
            any_non_null = true;
            if !item.is_instance_of::<PyBool>() { all_bool = false; }
            if !item.is_instance_of::<PyInt>() { all_int = false; }
            if !item.is_instance_of::<PyFloat>() { all_float = false; }
            if !item.is_instance_of::<PyString>() { all_str = false; }
        }

        // 全 None 时默认 object (避免误判为 bool)
        let dtype = if !any_non_null { DType::Object }
                    else if all_bool { DType::Bool }
                    else if all_int { DType::Int64 }
                    else if all_float { DType::Float64 }
                    else if all_str { DType::Object }
                    else { DType::Object };

        let inner = match dtype {
            DType::Bool => {
                let mut v: Vec<Option<bool>> = Vec::with_capacity(pylist.len());
                for item in pylist.iter() {
                    if item.is_none() { v.push(None); }
                    else if let Ok(b) = item.cast::<PyBool>() {
                        v.push(Some(b.is_true()));
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err("type mismatch (bool)"));
                    }
                }
                Series::new_bool(name, v)
            }
            DType::Int64 => {
                let mut v: Vec<Option<i64>> = Vec::with_capacity(pylist.len());
                for item in pylist.iter() {
                    if item.is_none() { v.push(None); }
                    else if let Ok(i) = item.cast::<PyInt>() {
                        v.push(Some(i.extract::<i64>()?));
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err("type mismatch (int)"));
                    }
                }
                Series::new_int(name, v)
            }
            DType::Float64 => {
                let mut v: Vec<Option<f64>> = Vec::with_capacity(pylist.len());
                for item in pylist.iter() {
                    if item.is_none() { v.push(None); }
                    else if let Ok(f) = item.cast::<PyFloat>() {
                        v.push(Some(f.extract::<f64>()?));
                    } else if let Ok(i) = item.cast::<PyInt>() {
                        v.push(Some(i.extract::<i64>()? as f64));
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err("type mismatch (float)"));
                    }
                }
                Series::new_float(name, v)
            }
            DType::Object => {
                let mut v: Vec<Option<String>> = Vec::with_capacity(pylist.len());
                for item in pylist.iter() {
                    if item.is_none() { v.push(None); }
                    else if let Ok(s) = item.cast::<PyString>() {
                        v.push(Some(s.extract::<String>()?));
                    } else if let Ok(b) = item.cast::<PyBool>() {
                        v.push(Some(b.extract::<bool>()?.to_string()));
                    } else if let Ok(i) = item.cast::<PyInt>() {
                        v.push(Some(i.extract::<i64>()?.to_string()));
                    } else if let Ok(f) = item.cast::<PyFloat>() {
                        v.push(Some(f.extract::<f64>()?.to_string()));
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err("unsupported type"));
                    }
                }
                Series::new_string(name, v)
            }
        };

        Ok(PySeries { inner })
    }

    // ---------- 属性 ----------

    #[getter]
    fn name(&self) -> Option<&str> { self.inner.name() }
    #[setter]
    fn set_name(&mut self, value: Option<String>) { self.inner.set_name(value); }
    #[getter]
    fn dtype(&self) -> &'static str { self.inner.dtype_name() }
    #[getter]
    fn shape(&self) -> (usize,) { self.inner.shape() }
    #[getter]
    fn size(&self) -> usize { self.inner.len() }
    #[getter]
    fn empty(&self) -> bool { self.inner.is_empty() }
    #[getter]
    fn nbytes(&self) -> usize { self.inner.nbytes() }

    /// 原始 list (None -> Python None)
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        self.inner.data.to_py_list(py)
    }

    // ---------- 切片 / 过滤 ----------

    fn head(&self, n: usize) -> Self {
        PySeries { inner: self.inner.head(n) }
    }
    fn tail(&self, n: usize) -> Self {
        PySeries { inner: self.inner.tail(n) }
    }
    fn filter(&self, mask: Vec<bool>) -> PyResult<Self> {
        if mask.len() != self.inner.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "mask length {} != series length {}",
                mask.len(), self.inner.len()
            )));
        }
        Ok(PySeries { inner: self.inner.filter(&mask) })
    }

    // ---------- 比较 (返回 Python list[bool]) ----------

    fn eq_scalar<'py>(&self, py: Python<'py>, value: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyList>> {
        use pyo3::types::PyAnyMethods;
        let mask = if let Ok(i) = value.cast::<PyInt>() {
            self.inner.eq_scalar_i64(i.extract::<i64>()?)
        } else if let Ok(f) = value.cast::<PyFloat>() {
            self.inner.eq_scalar_f64(f.extract::<f64>()?)
        } else if let Ok(b) = value.cast::<PyBool>() {
            self.inner.eq_scalar_bool(b.is_true())
        } else if let Ok(s) = value.cast::<PyString>() {
            self.inner.eq_scalar_str(&s.extract::<String>()?)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("value type not supported"));
        };
        Ok(PyList::new(py, mask.iter().map(|x| *x))?)
    }

    fn gt_scalar<'py>(&self, py: Python<'py>, value: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyList>> {
        use pyo3::types::PyAnyMethods;
        let mask = if let Ok(i) = value.cast::<PyInt>() {
            self.inner.gt_scalar_i64(i.extract::<i64>()?)
        } else if let Ok(f) = value.cast::<PyFloat>() {
            self.inner.gt_scalar_f64(f.extract::<f64>()?)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("gt only supports int/float"));
        };
        Ok(PyList::new(py, mask.iter().map(|x| *x))?)
    }

    fn lt_scalar<'py>(&self, py: Python<'py>, value: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyList>> {
        use pyo3::types::PyAnyMethods;
        let mask = if let Ok(i) = value.cast::<PyInt>() {
            self.inner.lt_scalar_i64(i.extract::<i64>()?)
        } else if let Ok(f) = value.cast::<PyFloat>() {
            self.inner.lt_scalar_f64(f.extract::<f64>()?)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("lt only supports int/float"));
        };
        Ok(PyList::new(py, mask.iter().map(|x| *x))?)
    }

    fn ge_scalar<'py>(&self, py: Python<'py>, value: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyList>> {
        use pyo3::types::PyAnyMethods;
        let mask = if let Ok(i) = value.cast::<PyInt>() {
            self.inner.ge_scalar_i64(i.extract::<i64>()?)
        } else if let Ok(f) = value.cast::<PyFloat>() {
            self.inner.ge_scalar_f64(f.extract::<f64>()?)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("ge only supports int/float"));
        };
        Ok(PyList::new(py, mask.iter().map(|x| *x))?)
    }

    fn le_scalar<'py>(&self, py: Python<'py>, value: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyList>> {
        use pyo3::types::PyAnyMethods;
        let mask = if let Ok(i) = value.cast::<PyInt>() {
            self.inner.le_scalar_i64(i.extract::<i64>()?)
        } else if let Ok(f) = value.cast::<PyFloat>() {
            self.inner.le_scalar_f64(f.extract::<f64>()?)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("le only supports int/float"));
        };
        Ok(PyList::new(py, mask.iter().map(|x| *x))?)
    }

    // ---------- 聚合 ----------

    fn count(&self) -> usize { self.inner.count() }

    fn sum<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::IntoPyObject;
        use pyo3::types::PyAnyMethods;
        match self.inner.dtype() {
            DType::Int64 => match self.inner.sum_i64() {
                Some(v) => Ok(v.into_pyobject(py)?.into_any()),
                None => Ok(py.None().into_bound(py)),
            },
            DType::Float64 => match self.inner.sum_f64() {
                Some(v) => Ok(v.into_pyobject(py)?.into_any()),
                None => Ok(py.None().into_bound(py)),
            },
            DType::Bool => {
                let v = self.inner.sum_bool();
                Ok(v.into_pyobject(py)?.into_any())
            }
            DType::Object => Ok(py.None().into_bound(py)),
        }
    }

    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::IntoPyObject;
        use pyo3::types::PyAnyMethods;
        match self.inner.mean() {
            Some(v) => Ok(v.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn min<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::IntoPyObject;
        use pyo3::types::PyAnyMethods;
        match self.inner.dtype() {
            DType::Int64 => match self.inner.min_i64() {
                Some(v) => Ok(v.into_pyobject(py)?.into_any()),
                None => Ok(py.None().into_bound(py)),
            },
            DType::Float64 => match self.inner.min_f64() {
                Some(v) => Ok(v.into_pyobject(py)?.into_any()),
                None => Ok(py.None().into_bound(py)),
            },
            DType::Object => match self.inner.min_str() {
                Some(v) => Ok(v.into_pyobject(py)?.into_any()),
                None => Ok(py.None().into_bound(py)),
            },
            DType::Bool => Ok(py.None().into_bound(py)),
        }
    }

    fn max<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::IntoPyObject;
        use pyo3::types::PyAnyMethods;
        match self.inner.dtype() {
            DType::Int64 => match self.inner.max_i64() {
                Some(v) => Ok(v.into_pyobject(py)?.into_any()),
                None => Ok(py.None().into_bound(py)),
            },
            DType::Float64 => match self.inner.max_f64() {
                Some(v) => Ok(v.into_pyobject(py)?.into_any()),
                None => Ok(py.None().into_bound(py)),
            },
            DType::Object => match self.inner.max_str() {
                Some(v) => Ok(v.into_pyobject(py)?.into_any()),
                None => Ok(py.None().into_bound(py)),
            },
            DType::Bool => Ok(py.None().into_bound(py)),
        }
    }

    fn std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::IntoPyObject;
        use pyo3::types::PyAnyMethods;
        match self.inner.std() {
            Some(v) => Ok(v.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn var<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::IntoPyObject;
        use pyo3::types::PyAnyMethods;
        match self.inner.var() {
            Some(v) => Ok(v.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn median<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::IntoPyObject;
        use pyo3::types::PyAnyMethods;
        match self.inner.median() {
            Some(v) => Ok(v.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn any<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::IntoPyObject;
        use pyo3::types::PyAnyMethods;
        match self.inner.any() {
            Some(v) => Ok(v.into_pyobject(py)?.as_any().clone()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn all<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::IntoPyObject;
        use pyo3::types::PyAnyMethods;
        match self.inner.all() {
            Some(v) => Ok(v.into_pyobject(py)?.as_any().clone()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    // ---------- 缺失值 ----------

    fn isnull<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        Ok(PyList::new(py, self.inner.isnull().iter().map(|x| *x))?)
    }

    fn notnull<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        Ok(PyList::new(py, self.inner.notnull().iter().map(|x| *x))?)
    }

    fn dropna(&self) -> Self {
        PySeries { inner: self.inner.dropna() }
    }

    /// 填充缺失值 (根据 dtype 自动选择)
    fn fillna<'py>(&self, py: Python<'py>, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        use pyo3::types::PyAnyMethods;
        let inner = match self.inner.dtype() {
            DType::Int64 => {
                let v: i64 = value.extract::<i64>()?;
                self.inner.fillna_i64(v)
            }
            DType::Float64 => {
                let v: f64 = value.extract::<f64>()?;
                self.inner.fillna_f64(v)
            }
            DType::Bool => {
                let v: bool = value.extract::<bool>()?;
                self.inner.fillna_bool(v)
            }
            DType::Object => {
                let v: String = value.extract::<String>()?;
                self.inner.fillna_string(&v)
            }
        };
        // suppress unused import warning
        let _ = py;
        Ok(PySeries { inner })
    }

    // ---------- 唯一值 ----------

    fn unique(&self) -> Self {
        PySeries { inner: self.inner.unique() }
    }

    fn nunique(&self) -> usize {
        self.inner.nunique()
    }

    /// value_counts: 返回 (value, count) 两条 Series
    fn value_counts<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
        use std::collections::HashMap;
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        for s in self.inner.to_string_vec() {
            // NaN 跳过
            if s == "NaN" { continue; }
            if let std::collections::hash_map::Entry::Vacant(e) = counts.entry(s.clone()) {
                order.push(s.clone());
                e.insert(0);
            }
            *counts.get_mut(&s).unwrap() += 1;
        }
        let values: Vec<&str> = order.iter().map(|s| s.as_str()).collect();
        let cnts: Vec<usize> = order.iter().map(|s| counts[s]).collect();
        Ok((
            PyList::new(py, values)?,
            PyList::new(py, cnts)?,
        ))
    }

    // ---------- 显示辅助 ----------

    /// 转换为字符串列表 (None -> "NaN")
    fn to_string_vec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let svec = self.inner.to_string_vec();
        Ok(PyList::new(py, svec.iter().map(|s| s.as_str()))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_series_basic() {
        let s = Series::new_int(Some("a".to_string()), vec![Some(1), Some(2), Some(3)]);
        assert_eq!(s.len(), 3);
        assert_eq!(s.dtype(), DType::Int64);
        assert_eq!(s.dtype_name(), "int64");
        assert_eq!(s.name(), Some("a"));
    }

    #[test]
    fn test_series_sum_mean() {
        let s = Series::new_float(None, vec![Some(1.0), Some(2.0), Some(3.0)]);
        assert_eq!(s.sum_f64(), Some(6.0));
        assert_eq!(s.mean(), Some(2.0));
        assert_eq!(s.min_f64(), Some(1.0));
        assert_eq!(s.max_f64(), Some(3.0));
    }

    #[test]
    fn test_series_with_null() {
        let s = Series::new_int(None, vec![Some(1), None, Some(3)]);
        assert_eq!(s.count(), 2);
        assert_eq!(s.sum_i64(), Some(4));
        assert_eq!(s.mean(), Some(2.0));
    }

    #[test]
    fn test_series_filter() {
        let s = Series::new_int(None, vec![Some(1), Some(2), Some(3)]);
        let filtered = s.filter(&[true, false, true]);
        assert_eq!(filtered.len(), 2);
    }
}
