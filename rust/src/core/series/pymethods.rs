//! Series 的 PyO3 绑定：将 Rust 端 Series 方法暴露给 Python。
//!
//! 该模块包含单个 `#[pymethods] impl PySeries` 块，涵盖构造、属性、切片、
//! 比较、聚合、缺失值、Categorical 访问器、排序、字符串方法、窗口/排名、
//! 日期时间、插值/采样/重采样、分组聚合与表达式过滤等全部 Python 可见方法。
//!
//! 所有计算密集型方法均通过 `py.detach` 释放 GIL 后再委托给 Rust 端 :struct:`Series`。

use pyo3::IntoPyObject;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pyo3::types::{PyBool, PyBoolMethods, PyFloat, PyInt, PyList, PyString};

use crate::core::dtype::{CategoricalData, ColumnData, DType};
use crate::core::series::{AggResult, PySeries};

#[pymethods]
impl PySeries {
    /// 构造: data 必须是 list，每个元素是 None/bool/int/float/str
    #[new]
    #[pyo3(signature = (data, name=None, dtype=None))]
    fn new(data: &Bound<'_, PyAny>, name: Option<String>, dtype: Option<&str>) -> PyResult<Self> {
        let pylist: &Bound<'_, PyList> = data
            .cast::<PyList>()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("Series data must be a list"))?;

        // 如果指定了 dtype，使用指定的类型
        if let Some(dt_str) = dtype {
            let dt = DType::parse(dt_str).unwrap_or(DType::Object);
            return Self::new_with_dtype(pylist, name, dt);
        }

        // 类型推断: bool -> int -> float -> str (按"最宽"覆盖)
        let mut all_bool = true;
        let mut all_int = true;
        let mut all_float = true;
        let mut all_int_or_float = true;
        let mut all_numeric = true;
        let mut any_non_null = false;
        let mut has_none = false;

        for item in pylist.iter() {
            if item.is_none() {
                has_none = true;
                continue;
            }
            any_non_null = true;
            if !item.is_instance_of::<PyBool>() {
                all_bool = false;
            }
            if !item.is_instance_of::<PyInt>() {
                all_int = false;
            }
            if !item.is_instance_of::<PyFloat>() {
                all_float = false;
            }
            if !item.is_instance_of::<PyInt>() && !item.is_instance_of::<PyFloat>() {
                all_int_or_float = false;
            }
            if !item.is_instance_of::<PyBool>()
                && !item.is_instance_of::<PyInt>()
                && !item.is_instance_of::<PyFloat>()
            {
                all_numeric = false;
            }
        }

        // 全 None 时默认 object (避免误判为 bool)
        // 有 None 值时，整数和布尔类型提升为 float（NaN 需要浮点存储）
        let dtype = if !any_non_null {
            DType::Object
        } else if all_bool {
            if has_none {
                DType::Float64
            } else {
                DType::Bool
            }
        } else if all_int {
            if has_none {
                DType::Float64
            } else {
                DType::Int64
            }
        } else if all_float || all_int_or_float || all_numeric {
            DType::Float64
        } else {
            DType::Object
        };

        let inner = match dtype {
            DType::Bool => {
                let mut v: Vec<Option<bool>> = Vec::with_capacity(pylist.len());
                for item in pylist.iter() {
                    if item.is_none() {
                        v.push(None);
                    } else if let Ok(b) = item.cast::<PyBool>() {
                        v.push(Some(b.is_true()));
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "type mismatch (bool)",
                        ));
                    }
                }
                crate::core::series::Series::new_bool(name, v)
            }
            DType::Int64 => {
                let mut v: Vec<Option<i64>> = Vec::with_capacity(pylist.len());
                for item in pylist.iter() {
                    if item.is_none() {
                        v.push(None);
                    } else if let Ok(i) = item.cast::<PyInt>() {
                        v.push(Some(i.extract::<i64>()?));
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "type mismatch (int)",
                        ));
                    }
                }
                crate::core::series::Series::new_int(name, v)
            }
            DType::Float64 => {
                let mut v: Vec<Option<f64>> = Vec::with_capacity(pylist.len());
                for item in pylist.iter() {
                    if item.is_none() {
                        v.push(None);
                    } else if let Ok(f) = item.cast::<PyFloat>() {
                        v.push(Some(f.extract::<f64>()?));
                    } else if let Ok(i) = item.cast::<PyInt>() {
                        v.push(Some(i.extract::<i64>()? as f64));
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "type mismatch (float)",
                        ));
                    }
                }
                crate::core::series::Series::new_float(name, v)
            }
            DType::Object => {
                let mut v: Vec<Option<String>> = Vec::with_capacity(pylist.len());
                for item in pylist.iter() {
                    if item.is_none() {
                        v.push(None);
                    } else if let Ok(s) = item.cast::<PyString>() {
                        v.push(Some(s.extract::<String>()?));
                    } else if let Ok(b) = item.cast::<PyBool>() {
                        v.push(Some(b.extract::<bool>()?.to_string()));
                    } else if let Ok(i) = item.cast::<PyInt>() {
                        v.push(Some(i.extract::<i64>()?.to_string()));
                    } else if let Ok(f) = item.cast::<PyFloat>() {
                        let fv = f.extract::<f64>()?;
                        if fv.is_nan() {
                            // NaN 在 object dtype 中存储为 None, 便于缺失值检测
                            v.push(None);
                        } else {
                            v.push(Some(fv.to_string()));
                        }
                    } else {
                        // 其他类型 (如 list/dict) 使用 str() 转为字符串
                        let s = item.str()?;
                        v.push(Some(s.extract::<String>()?));
                    }
                }
                crate::core::series::Series::new_string(name, v)
            }
            DType::Categorical => {
                // Categorical: 只接受字符串, 自动去重编码
                let mut raw: Vec<Option<String>> = Vec::with_capacity(pylist.len());
                for item in pylist.iter() {
                    if item.is_none() {
                        raw.push(None);
                    } else if let Ok(s) = item.cast::<PyString>() {
                        raw.push(Some(s.extract::<String>()?));
                    } else {
                        raw.push(Some(item.str()?.extract::<String>()?));
                    }
                }
                // 构建 categories 映射
                let mut cat_map: std::collections::HashMap<String, i32> =
                    std::collections::HashMap::new();
                let mut categories: Vec<String> = Vec::new();
                let mut codes: Vec<Option<i32>> = Vec::with_capacity(raw.len());
                for val in &raw {
                    match val {
                        Some(s) => {
                            let next_idx = categories.len() as i32;
                            let code = *cat_map.entry(s.clone()).or_insert_with(|| {
                                categories.push(s.clone());
                                next_idx
                            });
                            codes.push(Some(code));
                        }
                        None => codes.push(None),
                    }
                }
                crate::core::series::Series {
                    name,
                    data: ColumnData::Categorical(CategoricalData {
                        categories,
                        codes,
                        ordered: false,
                    }),
                }
            }
        };

        Ok(PySeries { inner })
    }

    // ---------- 属性 ----------

    #[getter]
    fn name(&self) -> Option<&str> {
        self.inner.name()
    }
    #[setter]
    fn set_name(&mut self, value: Option<String>) {
        self.inner.set_name(value);
    }
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype_name()
    }
    #[getter]
    fn shape(&self) -> (usize,) {
        self.inner.shape()
    }
    #[getter]
    fn size(&self) -> usize {
        self.inner.len()
    }
    #[getter]
    fn empty(&self) -> bool {
        self.inner.is_empty()
    }
    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }

    /// 原始 list (None -> Python None)
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        self.inner.data.to_py_list(py)
    }

    /// 设置指定位置的值 (用于 Python 端 __setitem__)
    fn set_value(&mut self, idx: usize, value: &Bound<'_, PyAny>) -> PyResult<()> {
        // None 值: 各类型统一设为 None
        if value.is_none() {
            match &mut self.inner.data {
                ColumnData::Float(v) => {
                    if idx < v.len() {
                        v[idx] = None;
                    }
                }
                ColumnData::Int(v) => {
                    if idx < v.len() {
                        v[idx] = None;
                    }
                }
                ColumnData::Bool(v) => {
                    if idx < v.len() {
                        v[idx] = None;
                    }
                }
                ColumnData::String(v) => {
                    if idx < v.len() {
                        v[idx] = None;
                    }
                }
                ColumnData::Categorical(c) => {
                    if idx < c.codes.len() {
                        c.codes[idx] = None;
                    }
                }
            }
            return Ok(());
        }
        // 数值/字符串值: 按 data 类型提取并设置
        match &mut self.inner.data {
            ColumnData::Float(v) => {
                if idx >= v.len() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        "index out of range",
                    ));
                }
                let f: f64 = value.extract()?;
                v[idx] = Some(f);
            }
            ColumnData::Int(v) => {
                if idx >= v.len() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        "index out of range",
                    ));
                }
                // nan -> None; 否则 as i64
                if let Ok(f) = value.extract::<f64>() {
                    if f.is_nan() {
                        v[idx] = None;
                    } else {
                        v[idx] = Some(f as i64);
                    }
                }
            }
            ColumnData::Bool(v) => {
                if idx >= v.len() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        "index out of range",
                    ));
                }
                if let Ok(b) = value.extract::<bool>() {
                    v[idx] = Some(b);
                } else if let Ok(f) = value.extract::<f64>() {
                    v[idx] = Some(f != 0.0);
                }
            }
            ColumnData::String(v) => {
                if idx >= v.len() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        "index out of range",
                    ));
                }
                if let Ok(s) = value.extract::<String>() {
                    v[idx] = Some(s);
                }
            }
            ColumnData::Categorical(c) => {
                if idx >= c.codes.len() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(
                        "index out of range",
                    ));
                }
                if let Ok(f) = value.extract::<f64>() {
                    c.codes[idx] = if f.is_nan() { None } else { Some(f as i32) };
                }
            }
        }
        Ok(())
    }

    // ---------- 切片 / 过滤 ----------

    fn head(&self, py: Python<'_>, n: usize) -> Self {
        let inner = py.detach(|| self.inner.head(n));
        PySeries { inner }
    }
    fn tail(&self, py: Python<'_>, n: usize) -> Self {
        let inner = py.detach(|| self.inner.tail(n));
        PySeries { inner }
    }
    fn filter(&self, py: Python<'_>, mask: Vec<bool>) -> PyResult<Self> {
        if mask.len() != self.inner.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "mask length {} != series length {}",
                mask.len(),
                self.inner.len()
            )));
        }
        let inner = py.detach(|| self.inner.filter(&mask));
        Ok(PySeries { inner })
    }

    // ---------- 比较 (返回 Python list[bool]) ----------

    fn eq_scalar<'py>(
        &self,
        py: Python<'py>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        // 在 GIL 内提取 Rust 值，然后释放 GIL 计算掩码
        let mask: Vec<bool> = if let Ok(i) = value.cast::<PyInt>() {
            let v = i.extract::<i64>()?;
            py.detach(|| self.inner.eq_scalar_i64(v))
        } else if let Ok(f) = value.cast::<PyFloat>() {
            let v = f.extract::<f64>()?;
            py.detach(|| self.inner.eq_scalar_f64(v))
        } else if let Ok(b) = value.cast::<PyBool>() {
            let v = b.is_true();
            py.detach(|| self.inner.eq_scalar_bool(v))
        } else if let Ok(s) = value.cast::<PyString>() {
            let v = s.extract::<String>()?;
            py.detach(|| self.inner.eq_scalar_str(&v))
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "value type not supported",
            ));
        };
        PyList::new(py, mask.iter().copied())
    }

    fn gt_scalar<'py>(
        &self,
        py: Python<'py>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let mask: Vec<bool> = if let Ok(i) = value.cast::<PyInt>() {
            let v = i.extract::<i64>()?;
            py.detach(|| self.inner.gt_scalar_i64(v))
        } else if let Ok(f) = value.cast::<PyFloat>() {
            let v = f.extract::<f64>()?;
            py.detach(|| self.inner.gt_scalar_f64(v))
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "gt only supports int/float",
            ));
        };
        PyList::new(py, mask.iter().copied())
    }

    fn lt_scalar<'py>(
        &self,
        py: Python<'py>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let mask: Vec<bool> = if let Ok(i) = value.cast::<PyInt>() {
            let v = i.extract::<i64>()?;
            py.detach(|| self.inner.lt_scalar_i64(v))
        } else if let Ok(f) = value.cast::<PyFloat>() {
            let v = f.extract::<f64>()?;
            py.detach(|| self.inner.lt_scalar_f64(v))
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "lt only supports int/float",
            ));
        };
        PyList::new(py, mask.iter().copied())
    }

    fn ge_scalar<'py>(
        &self,
        py: Python<'py>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let mask: Vec<bool> = if let Ok(i) = value.cast::<PyInt>() {
            let v = i.extract::<i64>()?;
            py.detach(|| self.inner.ge_scalar_i64(v))
        } else if let Ok(f) = value.cast::<PyFloat>() {
            let v = f.extract::<f64>()?;
            py.detach(|| self.inner.ge_scalar_f64(v))
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "ge only supports int/float",
            ));
        };
        PyList::new(py, mask.iter().copied())
    }

    fn le_scalar<'py>(
        &self,
        py: Python<'py>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let mask: Vec<bool> = if let Ok(i) = value.cast::<PyInt>() {
            let v = i.extract::<i64>()?;
            py.detach(|| self.inner.le_scalar_i64(v))
        } else if let Ok(f) = value.cast::<PyFloat>() {
            let v = f.extract::<f64>()?;
            py.detach(|| self.inner.le_scalar_f64(v))
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "le only supports int/float",
            ));
        };
        PyList::new(py, mask.iter().copied())
    }

    // ---------- 聚合 ----------

    fn count(&self, py: Python<'_>) -> usize {
        py.detach(|| self.inner.count())
    }

    fn sum<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dtype = self.inner.dtype();
        let result = py.detach(|| match dtype {
            DType::Int64 => self
                .inner
                .sum_i64()
                .map(AggResult::Int)
                .unwrap_or(AggResult::None),
            DType::Float64 => self
                .inner
                .sum_f64()
                .map(AggResult::Float)
                .unwrap_or(AggResult::None),
            DType::Bool => AggResult::Usize(self.inner.sum_bool()),
            _ => AggResult::None,
        });
        match result {
            AggResult::Int(v) => Ok(v.into_pyobject(py)?.into_any()),
            AggResult::Float(v) => Ok(v.into_pyobject(py)?.into_any()),
            AggResult::Usize(v) => Ok(v.into_pyobject(py)?.into_any()),
            AggResult::Bool(v) => Ok(v.into_pyobject(py)?.as_any().clone()),
            AggResult::Str(v) => Ok(v.into_pyobject(py)?.into_any()),
            AggResult::None => Ok(py.None().into_bound(py)),
        }
    }

    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let result = py.detach(|| self.inner.mean().map(AggResult::Float));
        match result {
            Some(AggResult::Float(v)) => Ok(v.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py)),
            _ => Ok(py.None().into_bound(py)),
        }
    }

    fn min<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dtype = self.inner.dtype();
        let result = py.detach(|| match dtype {
            DType::Int64 => self
                .inner
                .min_i64()
                .map(AggResult::Int)
                .unwrap_or(AggResult::None),
            DType::Float64 => self
                .inner
                .min_f64()
                .map(AggResult::Float)
                .unwrap_or(AggResult::None),
            DType::Object => self
                .inner
                .min_str()
                .map(AggResult::Str)
                .unwrap_or(AggResult::None),
            _ => AggResult::None,
        });
        match result {
            AggResult::Int(v) => Ok(v.into_pyobject(py)?.into_any()),
            AggResult::Float(v) => Ok(v.into_pyobject(py)?.into_any()),
            AggResult::Usize(v) => Ok(v.into_pyobject(py)?.into_any()),
            AggResult::Bool(v) => Ok(v.into_pyobject(py)?.as_any().clone()),
            AggResult::Str(v) => Ok(v.into_pyobject(py)?.into_any()),
            AggResult::None => Ok(py.None().into_bound(py)),
        }
    }

    fn max<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dtype = self.inner.dtype();
        let result = py.detach(|| match dtype {
            DType::Int64 => self
                .inner
                .max_i64()
                .map(AggResult::Int)
                .unwrap_or(AggResult::None),
            DType::Float64 => self
                .inner
                .max_f64()
                .map(AggResult::Float)
                .unwrap_or(AggResult::None),
            DType::Object => self
                .inner
                .max_str()
                .map(AggResult::Str)
                .unwrap_or(AggResult::None),
            _ => AggResult::None,
        });
        match result {
            AggResult::Int(v) => Ok(v.into_pyobject(py)?.into_any()),
            AggResult::Float(v) => Ok(v.into_pyobject(py)?.into_any()),
            AggResult::Usize(v) => Ok(v.into_pyobject(py)?.into_any()),
            AggResult::Bool(v) => Ok(v.into_pyobject(py)?.as_any().clone()),
            AggResult::Str(v) => Ok(v.into_pyobject(py)?.into_any()),
            AggResult::None => Ok(py.None().into_bound(py)),
        }
    }

    fn std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let result = py.detach(|| self.inner.std().map(AggResult::Float));
        match result {
            Some(AggResult::Float(v)) => Ok(v.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py)),
            _ => Ok(py.None().into_bound(py)),
        }
    }

    fn var<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let result = py.detach(|| self.inner.var().map(AggResult::Float));
        match result {
            Some(AggResult::Float(v)) => Ok(v.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py)),
            _ => Ok(py.None().into_bound(py)),
        }
    }

    fn median<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let result = py.detach(|| self.inner.median().map(AggResult::Float));
        match result {
            Some(AggResult::Float(v)) => Ok(v.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py)),
            _ => Ok(py.None().into_bound(py)),
        }
    }

    fn any<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let result = py.detach(|| self.inner.any().map(AggResult::Bool));
        match result {
            Some(AggResult::Bool(v)) => Ok(v.into_pyobject(py)?.as_any().clone()),
            None => Ok(py.None().into_bound(py)),
            _ => Ok(py.None().into_bound(py)),
        }
    }

    fn all<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let result = py.detach(|| self.inner.all().map(AggResult::Bool));
        match result {
            Some(AggResult::Bool(v)) => Ok(v.into_pyobject(py)?.as_any().clone()),
            None => Ok(py.None().into_bound(py)),
            _ => Ok(py.None().into_bound(py)),
        }
    }

    // ---------- 缺失值 ----------

    fn isnull<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mask = py.detach(|| self.inner.isnull());
        PyList::new(py, mask.iter().copied())
    }

    fn notnull<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mask = py.detach(|| self.inner.notnull());
        PyList::new(py, mask.iter().copied())
    }

    fn dropna(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.dropna());
        PySeries { inner }
    }

    /// 填充缺失值 (根据 dtype 自动选择)
    fn fillna<'py>(&self, py: Python<'py>, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dtype = self.inner.dtype();
        let inner = match dtype {
            DType::Int64 => {
                let v: i64 = value.extract::<i64>()?;
                py.detach(|| self.inner.fillna_i64(v))
            }
            DType::Float64 => {
                let v: f64 = value.extract::<f64>()?;
                py.detach(|| self.inner.fillna_f64(v))
            }
            DType::Bool => {
                let v: bool = value.extract::<bool>()?;
                py.detach(|| self.inner.fillna_bool(v))
            }
            DType::Object => {
                let v: String = value.extract::<String>()?;
                py.detach(|| self.inner.fillna_string(&v))
            }
            DType::Categorical => {
                let v: String = value.extract::<String>()?;
                py.detach(|| self.inner.fillna_categorical(&v))
            }
        };
        Ok(PySeries { inner })
    }

    // ---------- 唯一值 ----------

    fn unique(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.unique());
        PySeries { inner }
    }

    fn nunique(&self, py: Python<'_>) -> usize {
        py.detach(|| self.inner.nunique())
    }

    // ---------- Categorical 访问器 ----------

    fn cat_categories<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let cats: Option<Vec<String>> = py.detach(|| self.inner.cat_categories().cloned());
        match cats {
            Some(c) => Ok(c.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn cat_codes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let codes: Option<Vec<Option<i32>>> = py.detach(|| self.inner.cat_codes().cloned());
        match codes {
            Some(codes) => {
                let list = PyList::empty(py);
                for c in codes {
                    match c {
                        Some(v) => list.append(v).unwrap(),
                        None => list.append(py.None()).unwrap(),
                    }
                }
                Ok(list.into_any())
            }
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn cat_ordered(&self) -> Option<bool> {
        self.inner.cat_ordered()
    }

    fn cat_add_categories(&self, py: Python<'_>, new_cats: Vec<String>) -> Option<PySeries> {
        py.detach(|| {
            self.inner
                .cat_add_categories(&new_cats)
                .map(|s| PySeries { inner: s })
        })
    }

    fn cat_remove_unused_categories(&self, py: Python<'_>) -> Option<PySeries> {
        py.detach(|| {
            self.inner
                .cat_remove_unused_categories()
                .map(|s| PySeries { inner: s })
        })
    }

    fn cat_rename_categories(&self, py: Python<'_>, new_names: Vec<String>) -> Option<PySeries> {
        py.detach(|| {
            self.inner
                .cat_rename_categories(&new_names)
                .map(|s| PySeries { inner: s })
        })
    }

    fn cat_as_ordered(&self) -> Option<PySeries> {
        self.inner.cat_as_ordered().map(|s| PySeries { inner: s })
    }

    fn cat_as_unordered(&self) -> Option<PySeries> {
        self.inner.cat_as_unordered().map(|s| PySeries { inner: s })
    }

    // ---------- 显示辅助 ----------

    /// 转换为字符串列表 (None -> "NaN")
    fn to_string_vec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let svec: Vec<String> = py.detach(|| self.inner.to_string_vec());
        PyList::new(py, svec.iter().map(|s| s.as_str()))
    }

    // ---------- 排序 ----------

    fn sort_values(&self, py: Python<'_>, ascending: bool) -> Self {
        let inner = py.detach(|| self.inner.sort_values(ascending));
        PySeries { inner }
    }

    fn sort_index(&self, py: Python<'_>, ascending: bool) -> Self {
        let inner = py.detach(|| self.inner.sort_index(ascending));
        PySeries { inner }
    }

    /// 返回 nlargest/nsmallest 对应的原始索引列表。
    /// - n: 数量
    /// - keep: "first" / "last" / "all"
    /// - largest: true=nlargest，false=nsmallest
    fn arg_top_n<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        keep: &str,
        largest: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let idx: Vec<usize> = py.detach(|| self.inner.arg_top_n(n, keep, largest));
        PyList::new(py, idx.iter().copied())
    }

    // ---------- 前向/后向填充 ----------

    fn ffill(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.ffill());
        PySeries { inner }
    }

    fn bfill(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.bfill());
        PySeries { inner }
    }

    // ---------- 字符串方法 ----------

    fn str_upper(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.str_upper());
        PySeries { inner }
    }

    fn str_lower(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.str_lower());
        PySeries { inner }
    }

    fn str_len(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.str_len());
        PySeries { inner }
    }

    fn str_strip(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.str_strip());
        PySeries { inner }
    }

    fn str_contains<'py>(&self, py: Python<'py>, pattern: &str) -> PyResult<Bound<'py, PyList>> {
        // pattern 是 &str，闭包需要捕获它；&str 是 Sync，可以用
        let mask: Vec<bool> = py.detach(|| self.inner.str_contains(pattern));
        PyList::new(py, mask.iter().copied())
    }

    fn str_replace(&self, py: Python<'_>, from: &str, to: &str) -> Self {
        // 将 &str 转为 String，避免闭包捕获引用生命期问题
        let from_owned = from.to_string();
        let to_owned = to.to_string();
        let inner = py.detach(|| self.inner.str_replace(&from_owned, &to_owned));
        PySeries { inner }
    }

    // ---------- 分位数 / 排名 / searchsorted ----------

    fn quantile<'py>(&self, py: Python<'py>, q: f64) -> PyResult<Bound<'py, PyAny>> {
        match py.detach(|| self.inner.quantile(q)) {
            Some(v) => Ok(v.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    /// numpy.searchsorted：返回 values 在已升序（或经 sorter 重排后升序）的
    /// self 中的插入位置。返回与 values 等长的 usize 列表。
    #[pyo3(signature = (values, side="left", sorter=None))]
    fn searchsorted<'py>(
        &self,
        py: Python<'py>,
        values: &Bound<'py, PyAny>,
        side: &str,
        sorter: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyList>> {
        // 解析 values：标量 → 单元素列表；列表/元组 → 元素逐一转 f64
        let vals_f64: Vec<f64> = if let Ok(list) = values.cast::<PyList>() {
            list.iter()
                .map(|x| {
                    x.extract::<f64>().or_else(|_| {
                        x.extract::<i64>()
                            .map(|i| i as f64)
                            .or_else(|_| x.extract::<bool>().map(|b| if b { 1.0 } else { 0.0 }))
                    })
                })
                .collect::<PyResult<Vec<f64>>>()?
        } else if let Ok(t) = values.cast::<pyo3::types::PyTuple>() {
            t.iter()
                .map(|x| {
                    x.extract::<f64>().or_else(|_| {
                        x.extract::<i64>()
                            .map(|i| i as f64)
                            .or_else(|_| x.extract::<bool>().map(|b| if b { 1.0 } else { 0.0 }))
                    })
                })
                .collect::<PyResult<Vec<f64>>>()?
        } else {
            // 标量
            let v: f64 = values
                .extract::<f64>()
                .or_else(|_| values.extract::<i64>().map(|i| i as f64))
                .or_else(|_| values.extract::<bool>().map(|b| if b { 1.0 } else { 0.0 }))
                .map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "searchsorted 'value' must be numeric scalar or iterable",
                    )
                })?;
            vec![v]
        };

        // 解析 sorter：Option<Vec<usize>>。传入 None/list/tuple。
        let sorter_vec: Option<Vec<usize>> = if let Some(s) = sorter {
            if s.is_none() {
                None
            } else if let Ok(list) = s.cast::<PyList>() {
                let mut v = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let i: i64 = item.extract().map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(
                            "searchsorted sorter must be list[int]",
                        )
                    })?;
                    v.push(i.max(0) as usize);
                }
                Some(v)
            } else if let Ok(t) = s.cast::<pyo3::types::PyTuple>() {
                let mut v = Vec::with_capacity(t.len());
                for item in t.iter() {
                    let i: i64 = item.extract().map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(
                            "searchsorted sorter must be tuple[int]",
                        )
                    })?;
                    v.push(i.max(0) as usize);
                }
                Some(v)
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "searchsorted sorter must be None or list[int]/tuple[int]",
                ));
            }
        } else {
            None
        };

        let side_valid = if side != "left" && side != "right" {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "searchsorted side must be 'left' or 'right', got '{side}'"
            )));
        } else {
            side
        };

        let result = py.detach(|| match &sorter_vec {
            Some(idx) => self.inner.searchsorted(&vals_f64, side_valid, Some(idx)),
            None => self.inner.searchsorted(&vals_f64, side_valid, None),
        });
        PyList::new(py, result)
    }

    #[pyo3(signature = (method, ascending, na_option=None))]
    fn rank<'py>(
        &self,
        py: Python<'py>,
        method: &str,
        ascending: bool,
        na_option: Option<&str>,
    ) -> PyResult<Bound<'py, PyList>> {
        let ranks = py.detach(|| self.inner.rank(method, ascending, na_option));
        // 将 Option<f64> 转为 Python list
        let list = PyList::empty(py);
        for r in ranks {
            match r {
                Some(v) => list.append(v)?,
                None => list.append(py.None())?,
            }
        }
        Ok(list)
    }

    /// 值计数: 返回 ([values], [counts])，Python 层再转 Series
    fn value_counts<'py>(
        &self,
        py: Python<'py>,
        sort: bool,
        ascending: bool,
    ) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
        let (vals, counts) = py.detach(|| self.inner.value_counts(sort, ascending));
        let v_list = PyList::new(py, vals.iter().map(|s| s.as_str()))?;
        let c_list = PyList::new(py, counts.iter().copied())?;
        Ok((v_list, c_list))
    }

    // ---------- 滚动窗口 ----------

    fn rolling_sum<'py>(
        &self,
        py: Python<'py>,
        window: usize,
        min_periods: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let result = py.detach(|| self.inner.rolling_sum(window, min_periods));
        let list = PyList::empty(py);
        for v in result {
            match v {
                Some(val) => list.append(val)?,
                None => list.append(py.None())?,
            }
        }
        Ok(list)
    }

    fn rolling_mean<'py>(
        &self,
        py: Python<'py>,
        window: usize,
        min_periods: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let result = py.detach(|| self.inner.rolling_mean(window, min_periods));
        let list = PyList::empty(py);
        for v in result {
            match v {
                Some(val) => list.append(val)?,
                None => list.append(py.None())?,
            }
        }
        Ok(list)
    }

    fn rolling_std<'py>(
        &self,
        py: Python<'py>,
        window: usize,
        min_periods: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let result = py.detach(|| self.inner.rolling_std(window, min_periods));
        let list = PyList::empty(py);
        for v in result {
            match v {
                Some(val) => list.append(val)?,
                None => list.append(py.None())?,
            }
        }
        Ok(list)
    }

    // ---------- 扩展窗口 ----------

    fn expanding_sum<'py>(
        &self,
        py: Python<'py>,
        min_periods: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let result = py.detach(|| self.inner.expanding_sum(min_periods));
        let list = PyList::empty(py);
        for v in result {
            match v {
                Some(val) => list.append(val)?,
                None => list.append(py.None())?,
            }
        }
        Ok(list)
    }

    fn expanding_mean<'py>(
        &self,
        py: Python<'py>,
        min_periods: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let result = py.detach(|| self.inner.expanding_mean(min_periods));
        let list = PyList::empty(py);
        for v in result {
            match v {
                Some(val) => list.append(val)?,
                None => list.append(py.None())?,
            }
        }
        Ok(list)
    }

    // ---------- 指数加权 ----------

    fn ewm_mean<'py>(
        &self,
        py: Python<'py>,
        alpha: f64,
        min_periods: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let result = py.detach(|| self.inner.ewm_mean(alpha, min_periods));
        let list = PyList::empty(py);
        for v in result {
            match v {
                Some(val) => list.append(val)?,
                None => list.append(py.None())?,
            }
        }
        Ok(list)
    }

    // ---------- 日期时间方法 ----------

    fn dt_year(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.dt_year());
        PySeries { inner }
    }

    fn dt_month(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.dt_month());
        PySeries { inner }
    }

    fn dt_day(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.dt_day());
        PySeries { inner }
    }

    fn dt_hour(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.dt_hour());
        PySeries { inner }
    }

    fn dt_minute(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.dt_minute());
        PySeries { inner }
    }

    fn dt_second(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.dt_second());
        PySeries { inner }
    }

    fn dt_dayofweek(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.dt_dayofweek());
        PySeries { inner }
    }

    // ---------- 插值 / 采样 / 重采样 ----------

    /// 线性插值填充 None
    fn interpolate(&self, py: Python<'_>, method: &str, limit: Option<usize>) -> Self {
        let method_owned = method.to_string();
        let inner = py.detach(|| self.inner.interpolate(&method_owned, limit));
        PySeries { inner }
    }

    /// 随机采样
    fn sample(
        &self,
        py: Python<'_>,
        n: Option<usize>,
        frac: Option<f64>,
        replace: bool,
        seed: Option<u64>,
    ) -> Self {
        let inner = py.detach(|| self.inner.sample(n, frac, replace, seed));
        PySeries { inner }
    }

    /// 时间序列重采样聚合
    /// 返回 (桶起始时间列表, 聚合值列表)
    fn resample<'py>(
        &self,
        py: Python<'py>,
        timestamps: Vec<f64>,
        freq_seconds: f64,
        agg: &str,
    ) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
        let agg_owned = agg.to_string();
        let (out_ts, out_vals) =
            py.detach(|| self.inner.resample(&timestamps, freq_seconds, &agg_owned));
        let ts_list = PyList::new(py, out_ts.iter().copied())?;
        let val_list = PyList::new(py, out_vals.iter().map(|v| v.into_pyobject(py).ok()))?;
        Ok((ts_list, val_list))
    }

    // ---------- SeriesGroupBy 聚合 ----------

    /// 按字符串列表分组聚合
    /// 返回 (group_keys, agg_values)
    fn groupby_agg_series<'py>(
        &self,
        py: Python<'py>,
        by: Vec<String>,
        agg: &str,
    ) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
        let agg_owned = agg.to_string();
        let (keys, vals) = py.detach(|| self.inner.groupby_agg_series(&by, &agg_owned));
        let keys_list = PyList::new(py, keys.iter().map(|s| s.as_str()))?;
        let vals_list = PyList::new(py, vals.iter().map(|v| v.into_pyobject(py).ok()))?;
        Ok((keys_list, vals_list))
    }

    // ---------- 批量聚合（一次遍历多聚合） ----------

    /// 一次遍历计算多个聚合值
    /// aggs: 聚合名列表
    fn batch_agg<'py>(&self, py: Python<'py>, aggs: Vec<String>) -> PyResult<Bound<'py, PyList>> {
        let result = py.detach(|| self.inner.batch_agg(&aggs));
        let list = PyList::new(py, result.iter().map(|v| v.into_pyobject(py).ok()))?;
        Ok(list)
    }

    // ---------- 简单表达式过滤（query 简化版） ----------

    /// 简单比较过滤：列 op 标量
    /// op: ">" / "<" / ">=" / "<=" / "==" / "!="
    fn compare_scalar<'py>(
        &self,
        py: Python<'py>,
        op: &str,
        value: f64,
    ) -> PyResult<Bound<'py, PyList>> {
        let op_owned = op.to_string();
        let mask = py.detach(|| self.inner.compare_scalar(&op_owned, value));
        PyList::new(py, mask.iter().copied())
    }
}
