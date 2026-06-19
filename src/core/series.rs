//! Series: 单列数据结构 + PyO3 绑定
//!
//! PyO3 0.29 API: PyAnyMethods trait 提供 downcast/is_instance_of 等

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBoolMethods, PyFloat, PyInt, PyList, PyString};
use rayon::prelude::*;
use pyo3::types::PyAnyMethods;
use pyo3::IntoPyObject;
use std::collections::HashMap;

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
    pub fn new_categorical(name: Option<String>, categories: Vec<String>, codes: Vec<Option<i32>>, ordered: bool) -> Self {
        Self { name, data: ColumnData::Categorical(super::dtype::CategoricalData { categories, codes, ordered }) }
    }

    // 别名: 方便 CSV 等模块调用
    pub fn from_options_i64(name: String, v: &[Option<i64>]) -> Self {
        Self { name: Some(name), data: ColumnData::Int(v.to_vec()) }
    }
    pub fn from_options_f64(name: String, v: &[Option<f64>]) -> Self {
        Self { name: Some(name), data: ColumnData::Float(v.to_vec()) }
    }
    pub fn from_options_bool(name: String, v: &[Option<bool>]) -> Self {
        Self { name: Some(name), data: ColumnData::Bool(v.to_vec()) }
    }
    pub fn from_options_string(name: String, v: &[Option<String>]) -> Self {
        Self { name: Some(name), data: ColumnData::String(v.to_vec()) }
    }

    /// 获取索引位置的字符串表示 (用于 CSV 写出)
    pub fn get_str_at(&self, i: usize) -> String {
        match &self.data {
            ColumnData::Int(v) => v.get(i).map(|x| x.map(|n| n.to_string()).unwrap_or_default()).unwrap_or_default(),
            ColumnData::Float(v) => v.get(i).map(|x| x.map(|n| n.to_string()).unwrap_or_default()).unwrap_or_default(),
            ColumnData::Bool(v) => v.get(i).map(|x| x.map(|b| b.to_string()).unwrap_or_default()).unwrap_or_default(),
            ColumnData::String(v) => v.get(i).cloned().flatten().unwrap_or_default(),
            ColumnData::Categorical(c) => c.codes.get(i)
                .and_then(|code| code.map(|idx| {
                    c.categories.get(idx as usize).cloned().unwrap_or_default()
                }))
                .unwrap_or_default(),
        }
    }

    pub fn new_null(name: Option<String>, dtype: DType, len: usize) -> Self {
        let data = match dtype {
            DType::Int64 => ColumnData::Int(vec![None; len]),
            DType::Float64 => ColumnData::Float(vec![None; len]),
            DType::Bool => ColumnData::Bool(vec![None; len]),
            DType::Object => ColumnData::String(vec![None; len]),
            DType::Categorical => ColumnData::Categorical(super::dtype::CategoricalData {
                categories: Vec::new(),
                codes: vec![None; len],
                ordered: false,
            }),
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
                .par_iter()
                .map(|s| s.as_ref().map(|x| x.len()).unwrap_or(0))
                .sum::<usize>()
                + v.len() * std::mem::size_of::<Option<String>>(),
            ColumnData::Categorical(c) => {
                c.codes.len() * std::mem::size_of::<Option<i32>>()
                    + c.categories.par_iter().map(|s| s.len()).sum::<usize>()
                    + c.categories.len() * std::mem::size_of::<String>()
            }
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
            ColumnData::Int(col) => col.par_iter().map(|x| matches!(x, Some(x) if *x == v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn eq_scalar_f64(&self, v: f64) -> Vec<bool> {
        match &self.data {
            ColumnData::Float(col) => col.par_iter().map(|x| matches!(x, Some(x) if *x == v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn eq_scalar_bool(&self, v: bool) -> Vec<bool> {
        match &self.data {
            ColumnData::Bool(col) => col.par_iter().map(|x| matches!(x, Some(x) if *x == v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn eq_scalar_str(&self, v: &str) -> Vec<bool> {
        match &self.data {
            ColumnData::String(col) => col.par_iter().map(|x| matches!(x, Some(x) if x == v)).collect(),
            _ => vec![false; self.len()],
        }
    }

    pub fn gt_scalar_i64(&self, v: i64) -> Vec<bool> {
        match &self.data {
            ColumnData::Int(col) => col.par_iter().map(|x| matches!(x, Some(x) if *x > v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn gt_scalar_f64(&self, v: f64) -> Vec<bool> {
        match &self.data {
            ColumnData::Float(col) => col.par_iter().map(|x| matches!(x, Some(x) if *x > v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn lt_scalar_i64(&self, v: i64) -> Vec<bool> {
        match &self.data {
            ColumnData::Int(col) => col.par_iter().map(|x| matches!(x, Some(x) if *x < v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn lt_scalar_f64(&self, v: f64) -> Vec<bool> {
        match &self.data {
            ColumnData::Float(col) => col.par_iter().map(|x| matches!(x, Some(x) if *x < v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn ge_scalar_i64(&self, v: i64) -> Vec<bool> {
        match &self.data {
            ColumnData::Int(col) => col.par_iter().map(|x| matches!(x, Some(x) if *x >= v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn ge_scalar_f64(&self, v: f64) -> Vec<bool> {
        match &self.data {
            ColumnData::Float(col) => col.par_iter().map(|x| matches!(x, Some(x) if *x >= v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn le_scalar_i64(&self, v: i64) -> Vec<bool> {
        match &self.data {
            ColumnData::Int(col) => col.par_iter().map(|x| matches!(x, Some(x) if *x <= v)).collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn le_scalar_f64(&self, v: f64) -> Vec<bool> {
        match &self.data {
            ColumnData::Float(col) => col.par_iter().map(|x| matches!(x, Some(x) if *x <= v)).collect(),
            _ => vec![false; self.len()],
        }
    }

    // ---------- 聚合 ----------

    pub fn count(&self) -> usize { self.data.count_non_null() }

    pub fn sum_i64(&self) -> Option<i64> {
        if let ColumnData::Int(v) = &self.data {
            Some(v.par_iter().filter_map(|x| *x).sum())
        } else { None }
    }
    pub fn sum_f64(&self) -> Option<f64> {
        if let ColumnData::Float(v) = &self.data {
            Some(v.par_iter().filter_map(|x| *x).sum())
        } else { None }
    }
    pub fn sum_bool(&self) -> usize {
        if let ColumnData::Bool(v) = &self.data {
            v.par_iter().filter(|x| matches!(x, Some(true))).count()
        } else { 0 }
    }

    pub fn mean(&self) -> Option<f64> {
        let cnt = self.count();
        if cnt == 0 { return None; }
        match &self.data {
            ColumnData::Int(v) => {
                let s: i64 = v.par_iter().filter_map(|x| *x).sum();
                Some(s as f64 / cnt as f64)
            }
            ColumnData::Float(v) => {
                let s: f64 = v.par_iter().filter_map(|x| *x).sum();
                Some(s / cnt as f64)
            }
            _ => None,
        }
    }

    pub fn min_i64(&self) -> Option<i64> {
        if let ColumnData::Int(v) = &self.data {
            v.par_iter().filter_map(|x| *x).min()
        } else { None }
    }
    pub fn min_f64(&self) -> Option<f64> {
        if let ColumnData::Float(v) = &self.data {
            v.par_iter().filter_map(|x| *x).min_by(|a, b| a.partial_cmp(b).unwrap())
        } else { None }
    }
    pub fn min_str(&self) -> Option<String> {
        if let ColumnData::String(v) = &self.data {
            v.par_iter().filter_map(|x| x.clone()).min()
        } else { None }
    }
    pub fn max_i64(&self) -> Option<i64> {
        if let ColumnData::Int(v) = &self.data {
            v.par_iter().filter_map(|x| *x).max()
        } else { None }
    }
    pub fn max_f64(&self) -> Option<f64> {
        if let ColumnData::Float(v) = &self.data {
            v.par_iter().filter_map(|x| *x).max_by(|a, b| a.partial_cmp(b).unwrap())
        } else { None }
    }
    pub fn max_str(&self) -> Option<String> {
        if let ColumnData::String(v) = &self.data {
            v.par_iter().filter_map(|x| x.clone()).max()
        } else { None }
    }

    pub fn std(&self) -> Option<f64> { self.var().map(|v| v.sqrt()) }
    pub fn var(&self) -> Option<f64> {
        let m = self.mean()?;
        let cnt = self.count();
        if cnt == 0 { return None; }
        let s = match &self.data {
            ColumnData::Int(v) => v
                .par_iter().filter_map(|x| *x)
                .map(|x| (x as f64 - m).powi(2))
                .sum::<f64>(),
            ColumnData::Float(v) => v
                .par_iter().filter_map(|x| *x)
                .map(|x| (x - m).powi(2))
                .sum::<f64>(),
            _ => return None,
        };
        Some(s / cnt as f64)
    }
    pub fn median(&self) -> Option<f64> {
        let mut vs: Vec<f64> = match &self.data {
            ColumnData::Int(v) => v.par_iter().filter_map(|x| *x).map(|x| x as f64).collect(),
            ColumnData::Float(v) => v.par_iter().filter_map(|x| *x).collect(),
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
            Some(v.par_iter().any(|x| matches!(x, Some(true))))
        } else { None }
    }
    pub fn all(&self) -> Option<bool> {
        if let ColumnData::Bool(v) = &self.data {
            Some(v.par_iter().all(|x| matches!(x, Some(true))))
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
    pub fn fillna_categorical(&self, v: &str) -> Series {
        Self { name: self.name.clone(), data: self.data.fillna_categorical(v) }
    }

    // ---------- 唯一值 ----------

    pub fn unique(&self) -> Series {
        Self { name: self.name.clone(), data: self.data.unique() }
    }

    pub fn nunique(&self) -> usize {
        match &self.data {
            ColumnData::Int(v) => {
                v.par_iter()
                    .filter_map(|x| *x)
                    .fold(
                        || std::collections::HashSet::new(),
                        |mut set, val| { set.insert(val); set }
                    )
                    .reduce(
                        || std::collections::HashSet::new(),
                        |mut a, b| { a.extend(b); a }
                    )
                    .len()
            }
            ColumnData::Float(v) => {
                // 使用 OrderedFloat 风格的 HashSet 来并行化浮点去重
                v.par_iter()
                    .filter_map(|x| *x)
                    .fold(
                        || std::collections::HashSet::new(),
                        |mut set, val| {
                            // 使用 u64 位表示来避免浮点 NaN 问题
                            set.insert(val.to_bits());
                            set
                        }
                    )
                    .reduce(
                        || std::collections::HashSet::new(),
                        |mut a, b| { a.extend(b); a }
                    )
                    .len()
            }
            ColumnData::Bool(v) => {
                v.par_iter().filter_map(|x| *x).collect::<std::collections::HashSet<_>>().len()
            }
            ColumnData::String(v) => {
                v.par_iter()
                    .filter_map(|x| x.clone())
                    .fold(
                        || std::collections::HashSet::new(),
                        |mut set, val| { set.insert(val); set }
                    )
                    .reduce(
                        || std::collections::HashSet::new(),
                        |mut a, b| { a.extend(b); a }
                    )
                    .len()
            }
            ColumnData::Categorical(c) => {
                c.codes.par_iter()
                    .filter_map(|x| *x)
                    .fold(
                        || std::collections::HashSet::new(),
                        |mut set, val| { set.insert(val); set }
                    )
                    .reduce(
                        || std::collections::HashSet::new(),
                        |mut a, b| { a.extend(b); a }
                    )
                    .len()
            }
        }
    }

    // ---------- Categorical 操作 ----------

    /// 获取 categories 列表
    pub fn cat_categories(&self) -> Option<&Vec<String>> {
        if let ColumnData::Categorical(c) = &self.data {
            Some(&c.categories)
        } else { None }
    }

    /// 获取 codes 列表
    pub fn cat_codes(&self) -> Option<&Vec<Option<i32>>> {
        if let ColumnData::Categorical(c) = &self.data {
            Some(&c.codes)
        } else { None }
    }

    /// 是否有序
    pub fn cat_ordered(&self) -> Option<bool> {
        if let ColumnData::Categorical(c) = &self.data {
            Some(c.ordered)
        } else { None }
    }

    /// 添加新的 categories
    pub fn cat_add_categories(&self, new_cats: &[String]) -> Option<Series> {
        if let ColumnData::Categorical(c) = &self.data {
            let mut categories = c.categories.clone();
            for cat in new_cats {
                if !categories.contains(cat) {
                    categories.push(cat.clone());
                }
            }
            Some(Series {
                name: self.name.clone(),
                data: ColumnData::Categorical(super::dtype::CategoricalData {
                    categories,
                    codes: c.codes.clone(),
                    ordered: c.ordered,
                }),
            })
        } else { None }
    }

    /// 移除未使用的 categories
    pub fn cat_remove_unused_categories(&self) -> Option<Series> {
        if let ColumnData::Categorical(c) = &self.data {
            let used_codes: std::collections::HashSet<i32> = c.codes.iter()
                .filter_map(|x| *x).collect();
            let mut new_categories: Vec<String> = Vec::new();
            let mut code_map: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
            for (i, cat) in c.categories.iter().enumerate() {
                let old_code = i as i32;
                if used_codes.contains(&old_code) {
                    let new_code = new_categories.len() as i32;
                    new_categories.push(cat.clone());
                    code_map.insert(old_code, new_code);
                }
            }
            let new_codes: Vec<Option<i32>> = c.codes.iter().map(|code| {
                code.and_then(|old| code_map.get(&old).copied())
            }).collect();
            Some(Series {
                name: self.name.clone(),
                data: ColumnData::Categorical(super::dtype::CategoricalData {
                    categories: new_categories,
                    codes: new_codes,
                    ordered: c.ordered,
                }),
            })
        } else { None }
    }

    /// 重命名 categories
    pub fn cat_rename_categories(&self, new_names: &[String]) -> Option<Series> {
        if let ColumnData::Categorical(c) = &self.data {
            if new_names.len() != c.categories.len() {
                return None;
            }
            Some(Series {
                name: self.name.clone(),
                data: ColumnData::Categorical(super::dtype::CategoricalData {
                    categories: new_names.to_vec(),
                    codes: c.codes.clone(),
                    ordered: c.ordered,
                }),
            })
        } else { None }
    }

    /// 设置 ordered 标志
    pub fn cat_as_ordered(&self) -> Option<Series> {
        if let ColumnData::Categorical(c) = &self.data {
            Some(Series {
                name: self.name.clone(),
                data: ColumnData::Categorical(super::dtype::CategoricalData {
                    categories: c.categories.clone(),
                    codes: c.codes.clone(),
                    ordered: true,
                }),
            })
        } else { None }
    }

    pub fn cat_as_unordered(&self) -> Option<Series> {
        if let ColumnData::Categorical(c) = &self.data {
            Some(Series {
                name: self.name.clone(),
                data: ColumnData::Categorical(super::dtype::CategoricalData {
                    categories: c.categories.clone(),
                    codes: c.codes.clone(),
                    ordered: false,
                }),
            })
        } else { None }
    }

    /// 转换为字符串列表 (None -> "NaN") - 给 DataFrame 显示用
    /// 使用 rayon 并行化字符串转换，大数据量下有显著提升
    pub fn to_string_vec(&self) -> Vec<String> {
        match &self.data {
            ColumnData::Int(v) => v
                .par_iter()
                .map(|x| match x { Some(n) => n.to_string(), None => "NaN".to_string() })
                .collect(),
            ColumnData::Float(v) => v
                .par_iter()
                .map(|x| match x { Some(n) => n.to_string(), None => "NaN".to_string() })
                .collect(),
            ColumnData::Bool(v) => v
                .par_iter()
                .map(|x| match x { Some(b) => b.to_string(), None => "NaN".to_string() })
                .collect(),
            ColumnData::String(v) => v
                .par_iter()
                .map(|x| match x { Some(s) => s.clone(), None => "NaN".to_string() })
                .collect(),
            ColumnData::Categorical(c) => c.codes
                .par_iter()
                .map(|code| match code {
                    Some(idx) => c.categories.get(*idx as usize).cloned().unwrap_or_else(|| "NaN".to_string()),
                    None => "NaN".to_string(),
                })
                .collect(),
        }
    }
}

// =====================================================================
// PyO3 绑定
// =====================================================================

/// Python 端 _Series，包装 Rust Series
#[pyclass(name = "_Series", module = "rspandas", from_py_object)]
#[derive(Debug, Clone)]
pub struct PySeries {
    pub inner: Series,
}

impl PySeries {
    fn new_with_dtype(pylist: &Bound<'_, PyList>, name: Option<String>, dtype: DType) -> PyResult<Self> {
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
            DType::Categorical => {
                let mut raw: Vec<Option<String>> = Vec::with_capacity(pylist.len());
                for item in pylist.iter() {
                    if item.is_none() { raw.push(None); }
                    else if let Ok(s) = item.cast::<PyString>() {
                        raw.push(Some(s.extract::<String>()?));
                    } else {
                        raw.push(Some(item.str()?.extract::<String>()?));
                    }
                }
                let mut cat_map: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
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
                Series {
                    name,
                    data: ColumnData::Categorical(super::dtype::CategoricalData {
                        categories,
                        codes,
                        ordered: false,
                    }),
                }
            }
        };
        Ok(PySeries { inner })
    }
}

#[pymethods]
impl PySeries {
    /// 构造: data 必须是 list，每个元素是 None/bool/int/float/str
    #[new]
    #[pyo3(signature = (data, name=None, dtype=None))]
    fn new(data: &Bound<'_, PyAny>, name: Option<String>, dtype: Option<&str>) -> PyResult<Self> {
        let pylist: &Bound<'_, PyList> = data.cast::<PyList>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Series data must be a list")
        })?;

        // 如果指定了 dtype，使用指定的类型
        if let Some(dt_str) = dtype {
            let dt = DType::from_str(dt_str)
                .unwrap_or_else(|| DType::Object);
            return Self::new_with_dtype(pylist, name, dt);
        }

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
            DType::Categorical => {
                // Categorical: 只接受字符串, 自动去重编码
                let mut raw: Vec<Option<String>> = Vec::with_capacity(pylist.len());
                for item in pylist.iter() {
                    if item.is_none() { raw.push(None); }
                    else if let Ok(s) = item.cast::<PyString>() {
                        raw.push(Some(s.extract::<String>()?));
                    } else {
                        raw.push(Some(item.str()?.extract::<String>()?));
                    }
                }
                // 构建 categories 映射
                let mut cat_map: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
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
                Series {
                    name,
                    data: ColumnData::Categorical(super::dtype::CategoricalData {
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
            DType::Categorical => Ok(py.None().into_bound(py)),
        }
    }

    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.mean() {
            Some(v) => Ok(v.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn min<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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
            DType::Categorical => Ok(py.None().into_bound(py)),
        }
    }

    fn max<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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
            DType::Categorical => Ok(py.None().into_bound(py)),
        }
    }

    fn std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.std() {
            Some(v) => Ok(v.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn var<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.var() {
            Some(v) => Ok(v.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn median<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.median() {
            Some(v) => Ok(v.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn any<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.any() {
            Some(v) => Ok(v.into_pyobject(py)?.as_any().clone()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn all<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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
            DType::Categorical => {
                let v: String = value.extract::<String>()?;
                self.inner.fillna_categorical(&v)
            }
        };
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

    // ---------- Categorical 访问器 ----------

    fn cat_categories<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.cat_categories() {
            Some(cats) => Ok(cats.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn cat_codes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.cat_codes() {
            Some(codes) => {
                let list = PyList::empty(py);
                for c in codes {
                    match c {
                        Some(v) => list.append(*v).unwrap(),
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

    fn cat_add_categories(&self, new_cats: Vec<String>) -> Option<PySeries> {
        self.inner.cat_add_categories(&new_cats).map(|s| PySeries { inner: s })
    }

    fn cat_remove_unused_categories(&self) -> Option<PySeries> {
        self.inner.cat_remove_unused_categories().map(|s| PySeries { inner: s })
    }

    fn cat_rename_categories(&self, new_names: Vec<String>) -> Option<PySeries> {
        self.inner.cat_rename_categories(&new_names).map(|s| PySeries { inner: s })
    }

    fn cat_as_ordered(&self) -> Option<PySeries> {
        self.inner.cat_as_ordered().map(|s| PySeries { inner: s })
    }

    fn cat_as_unordered(&self) -> Option<PySeries> {
        self.inner.cat_as_unordered().map(|s| PySeries { inner: s })
    }

    /// value_counts: 返回 (value, count) 两条 Series
    fn value_counts<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
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

// =====================================================================
// factorize 函数
// =====================================================================

/// 对输入值进行 factorize 编码 (类似 pandas.factorize)
/// 返回 (codes, categories)
#[pyfunction]
pub fn factorize<'py>(py: Python<'py>, values: &Bound<'py, PyList>) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    let mut cat_map: HashMap<String, i32> = HashMap::new();
    let mut categories: Vec<String> = Vec::new();
    let mut codes: Vec<i32> = Vec::with_capacity(values.len());

    for item in values.iter() {
        if item.is_none() {
            codes.push(-1);
        } else {
            let s: String = if let Ok(s) = item.cast::<PyString>() {
                s.extract::<String>()?
            } else {
                item.str()?.extract::<String>()?
            };
            let next_idx = categories.len() as i32;
            let code = *cat_map.entry(s.clone()).or_insert_with(|| {
                categories.push(s);
                next_idx
            });
            codes.push(code);
        }
    }

    let codes_list = PyList::new(py, codes.iter().map(|x| *x))?;
    let cats_list = PyList::new(py, categories.iter().map(|s| s.as_str()))?;
    Ok((codes_list, cats_list))
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
