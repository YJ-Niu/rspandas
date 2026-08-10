//! Series: 单列数据结构 + PyO3 绑定
//!
//! PyO3 0.29 API: PyAnyMethods trait 提供 downcast/is_instance_of 等

use pyo3::IntoPyObject;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pyo3::types::{PyBool, PyBoolMethods, PyFloat, PyInt, PyList, PyString};
use rayon::prelude::*;
use std::collections::HashMap;

use super::dtype::{ColumnData, DType};

/// Series: 带名字的单列
#[derive(Debug, Clone)]
pub struct Series {
    pub name: Option<String>,
    pub data: ColumnData,
}

/// 聚合结果统一类型
/// 用于在 `py.detach` 闭包中跨 GIL 边界返回不同类型的结果
enum AggResult {
    Int(i64),
    Float(f64),
    Usize(usize),
    Bool(bool),
    Str(String),
    None,
}

impl Series {
    // ---------- 构造器 ----------

    pub fn new_int(name: Option<String>, v: Vec<Option<i64>>) -> Self {
        Self {
            name,
            data: ColumnData::Int(v),
        }
    }
    pub fn new_float(name: Option<String>, v: Vec<Option<f64>>) -> Self {
        Self {
            name,
            data: ColumnData::Float(v),
        }
    }
    pub fn new_bool(name: Option<String>, v: Vec<Option<bool>>) -> Self {
        Self {
            name,
            data: ColumnData::Bool(v),
        }
    }
    pub fn new_string(name: Option<String>, v: Vec<Option<String>>) -> Self {
        Self {
            name,
            data: ColumnData::String(v),
        }
    }
    pub fn new_categorical(
        name: Option<String>,
        categories: Vec<String>,
        codes: Vec<Option<i32>>,
        ordered: bool,
    ) -> Self {
        Self {
            name,
            data: ColumnData::Categorical(super::dtype::CategoricalData {
                categories,
                codes,
                ordered,
            }),
        }
    }

    // 别名: 方便 CSV 等模块调用
    pub fn from_options_i64(name: String, v: &[Option<i64>]) -> Self {
        Self {
            name: Some(name),
            data: ColumnData::Int(v.to_vec()),
        }
    }
    pub fn from_options_f64(name: String, v: &[Option<f64>]) -> Self {
        Self {
            name: Some(name),
            data: ColumnData::Float(v.to_vec()),
        }
    }
    pub fn from_options_bool(name: String, v: &[Option<bool>]) -> Self {
        Self {
            name: Some(name),
            data: ColumnData::Bool(v.to_vec()),
        }
    }
    pub fn from_options_string(name: String, v: &[Option<String>]) -> Self {
        Self {
            name: Some(name),
            data: ColumnData::String(v.to_vec()),
        }
    }

    /// 获取索引位置的字符串表示 (用于 CSV 写出)
    pub fn get_str_at(&self, i: usize) -> String {
        match &self.data {
            ColumnData::Int(v) => v
                .get(i)
                .map(|x| x.map(|n| n.to_string()).unwrap_or_default())
                .unwrap_or_default(),
            ColumnData::Float(v) => v
                .get(i)
                .map(|x| x.map(|n| n.to_string()).unwrap_or_default())
                .unwrap_or_default(),
            ColumnData::Bool(v) => v
                .get(i)
                .map(|x| x.map(|b| b.to_string()).unwrap_or_default())
                .unwrap_or_default(),
            ColumnData::String(v) => v.get(i).cloned().flatten().unwrap_or_default(),
            ColumnData::Categorical(c) => c
                .codes
                .get(i)
                .and_then(|code| {
                    code.map(|idx| c.categories.get(idx as usize).cloned().unwrap_or_default())
                })
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

    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn shape(&self) -> (usize,) {
        (self.data.len(),)
    }
    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }
    pub fn dtype_name(&self) -> &'static str {
        self.data.dtype_name()
    }
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    pub fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }
    pub fn nbytes(&self) -> usize {
        match &self.data {
            ColumnData::Int(v) => v.len() * std::mem::size_of::<Option<i64>>(),
            ColumnData::Float(v) => v.len() * std::mem::size_of::<Option<f64>>(),
            ColumnData::Bool(v) => v.len() * std::mem::size_of::<Option<bool>>(),
            ColumnData::String(v) => {
                v.par_iter()
                    .map(|s| s.as_ref().map(|x| x.len()).unwrap_or(0))
                    .sum::<usize>()
                    + v.len() * std::mem::size_of::<Option<String>>()
            }
            ColumnData::Categorical(c) => {
                c.codes.len() * std::mem::size_of::<Option<i32>>()
                    + c.categories.par_iter().map(|s| s.len()).sum::<usize>()
                    + c.categories.len() * std::mem::size_of::<String>()
            }
        }
    }

    // ---------- 切片 ----------

    pub fn slice(&self, start: usize, end: usize) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.slice(start, end),
        }
    }
    pub fn head(&self, n: usize) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.head(n),
        }
    }
    pub fn tail(&self, n: usize) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.tail(n),
        }
    }
    pub fn filter(&self, mask: &[bool]) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.filter(mask),
        }
    }

    // ---------- 比较 (返回 mask) ----------

    pub fn eq_scalar_i64(&self, v: i64) -> Vec<bool> {
        match &self.data {
            ColumnData::Int(col) => col
                .par_iter()
                .map(|x| matches!(x, Some(x) if *x == v))
                .collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn eq_scalar_f64(&self, v: f64) -> Vec<bool> {
        match &self.data {
            ColumnData::Float(col) => col
                .par_iter()
                .map(|x| matches!(x, Some(x) if *x == v))
                .collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn eq_scalar_bool(&self, v: bool) -> Vec<bool> {
        match &self.data {
            ColumnData::Bool(col) => col
                .par_iter()
                .map(|x| matches!(x, Some(x) if *x == v))
                .collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn eq_scalar_str(&self, v: &str) -> Vec<bool> {
        match &self.data {
            ColumnData::String(col) => col
                .par_iter()
                .map(|x| matches!(x, Some(x) if x == v))
                .collect(),
            _ => vec![false; self.len()],
        }
    }

    pub fn gt_scalar_i64(&self, v: i64) -> Vec<bool> {
        match &self.data {
            ColumnData::Int(col) => col
                .par_iter()
                .map(|x| matches!(x, Some(x) if *x > v))
                .collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn gt_scalar_f64(&self, v: f64) -> Vec<bool> {
        match &self.data {
            ColumnData::Float(col) => col
                .par_iter()
                .map(|x| matches!(x, Some(x) if *x > v))
                .collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn lt_scalar_i64(&self, v: i64) -> Vec<bool> {
        match &self.data {
            ColumnData::Int(col) => col
                .par_iter()
                .map(|x| matches!(x, Some(x) if *x < v))
                .collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn lt_scalar_f64(&self, v: f64) -> Vec<bool> {
        match &self.data {
            ColumnData::Float(col) => col
                .par_iter()
                .map(|x| matches!(x, Some(x) if *x < v))
                .collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn ge_scalar_i64(&self, v: i64) -> Vec<bool> {
        match &self.data {
            ColumnData::Int(col) => col
                .par_iter()
                .map(|x| matches!(x, Some(x) if *x >= v))
                .collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn ge_scalar_f64(&self, v: f64) -> Vec<bool> {
        match &self.data {
            ColumnData::Float(col) => col
                .par_iter()
                .map(|x| matches!(x, Some(x) if *x >= v))
                .collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn le_scalar_i64(&self, v: i64) -> Vec<bool> {
        match &self.data {
            ColumnData::Int(col) => col
                .par_iter()
                .map(|x| matches!(x, Some(x) if *x <= v))
                .collect(),
            _ => vec![false; self.len()],
        }
    }
    pub fn le_scalar_f64(&self, v: f64) -> Vec<bool> {
        match &self.data {
            ColumnData::Float(col) => col
                .par_iter()
                .map(|x| matches!(x, Some(x) if *x <= v))
                .collect(),
            _ => vec![false; self.len()],
        }
    }

    // ---------- 聚合 ----------

    pub fn count(&self) -> usize {
        self.data.count_non_null()
    }

    pub fn sum_i64(&self) -> Option<i64> {
        if let ColumnData::Int(v) = &self.data {
            Some(v.par_iter().filter_map(|x| *x).sum())
        } else {
            None
        }
    }
    pub fn sum_f64(&self) -> Option<f64> {
        if let ColumnData::Float(v) = &self.data {
            Some(
                v.par_iter()
                    .filter_map(|x| *x)
                    .filter(|x| !x.is_nan())
                    .sum(),
            )
        } else {
            None
        }
    }
    pub fn sum_bool(&self) -> usize {
        if let ColumnData::Bool(v) = &self.data {
            v.par_iter().filter(|x| matches!(x, Some(true))).count()
        } else {
            0
        }
    }

    pub fn mean(&self) -> Option<f64> {
        // 过滤 None 和 NaN 后计算均值 (NaN 语义上等同缺失值)
        let (sum, cnt) = match &self.data {
            ColumnData::Int(v) => {
                let filtered: Vec<i64> = v.par_iter().filter_map(|x| *x).collect();
                let cnt = filtered.len();
                let s: i64 = filtered.into_par_iter().sum();
                (s as f64, cnt)
            }
            ColumnData::Float(v) => {
                let filtered: Vec<f64> = v
                    .par_iter()
                    .filter_map(|x| x.filter(|v| !v.is_nan()))
                    .collect();
                let cnt = filtered.len();
                let s: f64 = filtered.into_par_iter().sum();
                (s, cnt)
            }
            _ => return None,
        };
        if cnt == 0 {
            return None;
        }
        Some(sum / cnt as f64)
    }

    pub fn min_i64(&self) -> Option<i64> {
        if let ColumnData::Int(v) = &self.data {
            v.par_iter().filter_map(|x| *x).min()
        } else {
            None
        }
    }
    pub fn min_f64(&self) -> Option<f64> {
        if let ColumnData::Float(v) = &self.data {
            // 过滤 None 和 NaN (NaN 语义上等同缺失值)
            v.par_iter()
                .filter_map(|x| x.filter(|v| !v.is_nan()))
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        } else {
            None
        }
    }
    pub fn min_str(&self) -> Option<String> {
        if let ColumnData::String(v) = &self.data {
            v.par_iter().filter_map(|x| x.clone()).min()
        } else {
            None
        }
    }
    pub fn max_i64(&self) -> Option<i64> {
        if let ColumnData::Int(v) = &self.data {
            v.par_iter().filter_map(|x| *x).max()
        } else {
            None
        }
    }
    pub fn max_f64(&self) -> Option<f64> {
        if let ColumnData::Float(v) = &self.data {
            // 过滤 None 和 NaN (NaN 语义上等同缺失值)
            v.par_iter()
                .filter_map(|x| x.filter(|v| !v.is_nan()))
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        } else {
            None
        }
    }
    pub fn max_str(&self) -> Option<String> {
        if let ColumnData::String(v) = &self.data {
            v.par_iter().filter_map(|x| x.clone()).max()
        } else {
            None
        }
    }

    pub fn std(&self) -> Option<f64> {
        self.var().map(|v| v.sqrt())
    }
    pub fn var(&self) -> Option<f64> {
        let m = self.mean()?;
        // 使用过滤后的非 NaN 值计算方差 (与 mean 保持一致)
        let (sum_sq, cnt) = match &self.data {
            ColumnData::Int(v) => {
                let filtered: Vec<i64> = v.par_iter().filter_map(|x| *x).collect();
                let cnt = filtered.len();
                let s: f64 = filtered
                    .into_par_iter()
                    .map(|x| (x as f64 - m).powi(2))
                    .sum();
                (s, cnt)
            }
            ColumnData::Float(v) => {
                let filtered: Vec<f64> = v
                    .par_iter()
                    .filter_map(|x| x.filter(|v| !v.is_nan()))
                    .collect();
                let cnt = filtered.len();
                let s: f64 = filtered.into_par_iter().map(|x| (x - m).powi(2)).sum();
                (s, cnt)
            }
            _ => return None,
        };
        if cnt == 0 {
            return None;
        }
        Some(sum_sq / cnt as f64)
    }
    pub fn median(&self) -> Option<f64> {
        // 过滤 None 和 NaN 后计算中位数
        let mut vs: Vec<f64> = match &self.data {
            ColumnData::Int(v) => v.par_iter().filter_map(|x| *x).map(|x| x as f64).collect(),
            ColumnData::Float(v) => v
                .par_iter()
                .filter_map(|x| x.filter(|v| !v.is_nan()))
                .collect(),
            _ => return None,
        };
        if vs.is_empty() {
            return None;
        }
        vs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vs.len();
        if n % 2 == 1 {
            Some(vs[n / 2])
        } else {
            Some((vs[n / 2 - 1] + vs[n / 2]) / 2.0)
        }
    }

    pub fn any(&self) -> Option<bool> {
        if let ColumnData::Bool(v) = &self.data {
            Some(v.par_iter().any(|x| matches!(x, Some(true))))
        } else {
            None
        }
    }
    pub fn all(&self) -> Option<bool> {
        if let ColumnData::Bool(v) = &self.data {
            Some(v.par_iter().all(|x| matches!(x, Some(true))))
        } else {
            None
        }
    }

    // ---------- 缺失值 ----------

    pub fn isnull(&self) -> Vec<bool> {
        self.data.isnull()
    }
    pub fn notnull(&self) -> Vec<bool> {
        self.data.notnull()
    }

    pub fn dropna(&self) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.dropna(),
        }
    }

    pub fn fillna_i64(&self, v: i64) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.fillna_i64(v),
        }
    }
    pub fn fillna_f64(&self, v: f64) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.fillna_f64(v),
        }
    }
    pub fn fillna_bool(&self, v: bool) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.fillna_bool(v),
        }
    }
    pub fn fillna_string(&self, v: &str) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.fillna_string(v),
        }
    }
    pub fn fillna_categorical(&self, v: &str) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.fillna_categorical(v),
        }
    }

    // ---------- 唯一值 ----------

    pub fn unique(&self) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.unique(),
        }
    }

    pub fn nunique(&self) -> usize {
        match &self.data {
            ColumnData::Int(v) => v
                .par_iter()
                .filter_map(|x| *x)
                .fold(std::collections::HashSet::new, |mut set, val| {
                    set.insert(val);
                    set
                })
                .reduce(std::collections::HashSet::new, |mut a, b| {
                    a.extend(b);
                    a
                })
                .len(),
            ColumnData::Float(v) => {
                // 使用 OrderedFloat 风格的 HashSet 来并行化浮点去重
                v.par_iter()
                    .filter_map(|x| *x)
                    .fold(std::collections::HashSet::new, |mut set, val| {
                        // 使用 u64 位表示来避免浮点 NaN 问题
                        set.insert(val.to_bits());
                        set
                    })
                    .reduce(std::collections::HashSet::new, |mut a, b| {
                        a.extend(b);
                        a
                    })
                    .len()
            }
            ColumnData::Bool(v) => v
                .par_iter()
                .filter_map(|x| *x)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            ColumnData::String(v) => v
                .par_iter()
                .filter_map(|x| x.clone())
                .fold(std::collections::HashSet::new, |mut set, val| {
                    set.insert(val);
                    set
                })
                .reduce(std::collections::HashSet::new, |mut a, b| {
                    a.extend(b);
                    a
                })
                .len(),
            ColumnData::Categorical(c) => c
                .codes
                .par_iter()
                .filter_map(|x| *x)
                .fold(std::collections::HashSet::new, |mut set, val| {
                    set.insert(val);
                    set
                })
                .reduce(std::collections::HashSet::new, |mut a, b| {
                    a.extend(b);
                    a
                })
                .len(),
        }
    }

    // ---------- Categorical 操作 ----------

    /// 获取 categories 列表
    pub fn cat_categories(&self) -> Option<&Vec<String>> {
        if let ColumnData::Categorical(c) = &self.data {
            Some(&c.categories)
        } else {
            None
        }
    }

    /// 获取 codes 列表
    pub fn cat_codes(&self) -> Option<&Vec<Option<i32>>> {
        if let ColumnData::Categorical(c) = &self.data {
            Some(&c.codes)
        } else {
            None
        }
    }

    /// 是否有序
    pub fn cat_ordered(&self) -> Option<bool> {
        if let ColumnData::Categorical(c) = &self.data {
            Some(c.ordered)
        } else {
            None
        }
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
        } else {
            None
        }
    }

    /// 移除未使用的 categories
    pub fn cat_remove_unused_categories(&self) -> Option<Series> {
        if let ColumnData::Categorical(c) = &self.data {
            let used_codes: std::collections::HashSet<i32> =
                c.codes.iter().filter_map(|x| *x).collect();
            let mut new_categories: Vec<String> = Vec::new();
            let mut code_map: std::collections::HashMap<i32, i32> =
                std::collections::HashMap::new();
            for (i, cat) in c.categories.iter().enumerate() {
                let old_code = i as i32;
                if used_codes.contains(&old_code) {
                    let new_code = new_categories.len() as i32;
                    new_categories.push(cat.clone());
                    code_map.insert(old_code, new_code);
                }
            }
            let new_codes: Vec<Option<i32>> = c
                .codes
                .iter()
                .map(|code| code.and_then(|old| code_map.get(&old).copied()))
                .collect();
            Some(Series {
                name: self.name.clone(),
                data: ColumnData::Categorical(super::dtype::CategoricalData {
                    categories: new_categories,
                    codes: new_codes,
                    ordered: c.ordered,
                }),
            })
        } else {
            None
        }
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
        } else {
            None
        }
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
        } else {
            None
        }
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
        } else {
            None
        }
    }

    /// 转换为字符串列表 (None -> "NaN") - 给 DataFrame 显示用
    /// 使用 rayon 并行化字符串转换，大数据量下有显著提升
    pub fn to_string_vec(&self) -> Vec<String> {
        match &self.data {
            ColumnData::Int(v) => v
                .par_iter()
                .map(|x| match x {
                    Some(n) => n.to_string(),
                    None => "NaN".to_string(),
                })
                .collect(),
            ColumnData::Float(v) => v
                .par_iter()
                .map(|x| match x {
                    Some(n) => n.to_string(),
                    None => "NaN".to_string(),
                })
                .collect(),
            ColumnData::Bool(v) => v
                .par_iter()
                .map(|x| match x {
                    Some(true) => "True".to_string(),
                    Some(false) => "False".to_string(),
                    None => "NaN".to_string(),
                })
                .collect(),
            ColumnData::String(v) => v
                .par_iter()
                .map(|x| match x {
                    Some(s) => s.clone(),
                    None => "NaN".to_string(),
                })
                .collect(),
            ColumnData::Categorical(c) => c
                .codes
                .par_iter()
                .map(|code| match code {
                    Some(idx) => c
                        .categories
                        .get(*idx as usize)
                        .cloned()
                        .unwrap_or_else(|| "NaN".to_string()),
                    None => "NaN".to_string(),
                })
                .collect(),
        }
    }

    // ---------- 日期时间方法（对 Float 类型有效，值为 Unix 纪元秒） ----------

    /// 提取年份
    pub fn dt_year(&self) -> Series {
        let data = match &self.data {
            ColumnData::Float(v) => {
                let years: Vec<Option<i64>> = v
                    .par_iter()
                    .map(|x| {
                        x.map(|ts| {
                            // 简单日期计算：从 Unix 纪元秒提取年份
                            let days = (ts / 86400.0) as i64;
                            let mut year = 1970i64;
                            let mut remaining = days;
                            loop {
                                let is_leap =
                                    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
                                let days_in_year = if is_leap { 366 } else { 365 };
                                if remaining < days_in_year {
                                    break;
                                }
                                remaining -= days_in_year;
                                year += 1;
                            }
                            year
                        })
                    })
                    .collect();
                ColumnData::Int(years)
            }
            _ => ColumnData::Int(vec![None; self.len()]),
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 提取月份 (1-12)
    pub fn dt_month(&self) -> Series {
        let data = match &self.data {
            ColumnData::Float(v) => {
                let months: Vec<Option<i64>> = v
                    .par_iter()
                    .map(|x| {
                        x.map(|ts| {
                            let days = (ts / 86400.0) as i64;
                            let mut year = 1970i64;
                            let mut remaining = days;
                            loop {
                                let is_leap =
                                    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
                                let days_in_year = if is_leap { 366 } else { 365 };
                                if remaining < days_in_year {
                                    break;
                                }
                                remaining -= days_in_year;
                                year += 1;
                            }
                            // remaining 是当年第几天（0-based）
                            let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
                            let month_days = if is_leap {
                                [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                            } else {
                                [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                            };
                            let mut month = 1;
                            let mut day = remaining;
                            for &md in &month_days {
                                if day < md {
                                    break;
                                }
                                day -= md;
                                month += 1;
                            }
                            month as i64
                        })
                    })
                    .collect();
                ColumnData::Int(months)
            }
            _ => ColumnData::Int(vec![None; self.len()]),
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 提取日 (1-31)
    pub fn dt_day(&self) -> Series {
        let data = match &self.data {
            ColumnData::Float(v) => {
                let days_vec: Vec<Option<i64>> = v
                    .par_iter()
                    .map(|x| {
                        x.map(|ts| {
                            let days = (ts / 86400.0) as i64;
                            let mut year = 1970i64;
                            let mut remaining = days;
                            loop {
                                let is_leap =
                                    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
                                let days_in_year = if is_leap { 366 } else { 365 };
                                if remaining < days_in_year {
                                    break;
                                }
                                remaining -= days_in_year;
                                year += 1;
                            }
                            let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
                            let month_days = if is_leap {
                                [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                            } else {
                                [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                            };
                            let mut day = remaining;
                            for &md in &month_days {
                                if day < md {
                                    break;
                                }
                                day -= md;
                            }
                            day + 1 // 1-based
                        })
                    })
                    .collect();
                ColumnData::Int(days_vec)
            }
            _ => ColumnData::Int(vec![None; self.len()]),
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 提取小时 (0-23)
    pub fn dt_hour(&self) -> Series {
        let data = match &self.data {
            ColumnData::Float(v) => {
                let hours: Vec<Option<i64>> = v
                    .par_iter()
                    .map(|x| {
                        x.map(|ts| {
                            let day_remainder = ts.rem_euclid(86400.0);
                            (day_remainder / 3600.0) as i64
                        })
                    })
                    .collect();
                ColumnData::Int(hours)
            }
            _ => ColumnData::Int(vec![None; self.len()]),
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 提取分钟 (0-59)
    pub fn dt_minute(&self) -> Series {
        let data = match &self.data {
            ColumnData::Float(v) => {
                let minutes: Vec<Option<i64>> = v
                    .par_iter()
                    .map(|x| {
                        x.map(|ts| {
                            let hour_remainder = ts.rem_euclid(3600.0);
                            (hour_remainder / 60.0) as i64
                        })
                    })
                    .collect();
                ColumnData::Int(minutes)
            }
            _ => ColumnData::Int(vec![None; self.len()]),
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 提取秒 (0-59)
    pub fn dt_second(&self) -> Series {
        let data = match &self.data {
            ColumnData::Float(v) => {
                let seconds: Vec<Option<i64>> = v
                    .par_iter()
                    .map(|x| x.map(|ts| ts.rem_euclid(60.0) as i64))
                    .collect();
                ColumnData::Int(seconds)
            }
            _ => ColumnData::Int(vec![None; self.len()]),
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 提取星期几 (0=周一, 6=周日)
    pub fn dt_dayofweek(&self) -> Series {
        let data = match &self.data {
            ColumnData::Float(v) => {
                let dows: Vec<Option<i64>> = v
                    .par_iter()
                    .map(|x| {
                        x.map(|ts| {
                            let days = (ts / 86400.0) as i64;
                            // 1970-01-01 是周四 (3)
                            (days + 3) % 7
                        })
                    })
                    .collect();
                ColumnData::Int(dows)
            }
            _ => ColumnData::Int(vec![None; self.len()]),
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    // ---------- 分位数 / 排名 / 窗口计算 ----------

    /// 将 Series 转为 Vec<Option<f64>>（用于窗口计算）
    pub fn as_f64_vec(&self) -> Vec<Option<f64>> {
        match &self.data {
            ColumnData::Int(v) => v.par_iter().map(|x| x.map(|v| v as f64)).collect(),
            ColumnData::Float(v) => v.par_iter().map(|x| *x).collect(),
            ColumnData::Bool(v) => v
                .par_iter()
                .map(|x| x.map(|b| if b { 1.0 } else { 0.0 }))
                .collect(),
            _ => vec![None; self.len()],
        }
    }

    /// 计算分位数（线性插值法）
    /// q: 0.0-1.0 之间的分位数
    pub fn quantile(&self, q: f64) -> Option<f64> {
        // 收集非 None 且非 NaN 的 f64 值
        let mut vs: Vec<f64> = match &self.data {
            ColumnData::Int(v) => v.par_iter().filter_map(|x| *x).map(|x| x as f64).collect(),
            ColumnData::Float(v) => v
                .par_iter()
                .filter_map(|x| x.filter(|v| !v.is_nan()))
                .collect(),
            _ => return None,
        };
        if vs.is_empty() {
            return None;
        }
        vs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vs.len();
        if n == 1 {
            return Some(vs[0]);
        }
        let pos = q * (n - 1) as f64;
        let lower = pos.floor() as usize;
        let upper = pos.ceil() as usize;
        let frac = pos - lower as f64;
        Some(vs[lower] * (1.0 - frac) + vs[upper] * frac)
    }

    /// 计算排名
    /// method: "average"=平均排名, "min"=最小排名, "max"=最大排名, "first"=出现顺序
    /// ascending: true=升序, false=降序
    pub fn rank(&self, method: &str, ascending: bool) -> Vec<Option<f64>> {
        let n = self.len();
        let mut result: Vec<Option<f64>> = vec![None; n];

        // 收集 (value, index) 对，跳过 None
        let mut indexed: Vec<(f64, usize)> = match &self.data {
            ColumnData::Int(v) => v
                .iter()
                .enumerate()
                .filter_map(|(i, x)| x.map(|val| (val as f64, i)))
                .collect(),
            ColumnData::Float(v) => v
                .iter()
                .enumerate()
                .filter_map(|(i, x)| x.map(|val| (val, i)))
                .collect(),
            _ => return result,
        };

        // 排序
        indexed.sort_by(|a, b| {
            if ascending {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        // 分配排名
        let m = indexed.len();
        let mut i = 0;
        while i < m {
            let mut j = i + 1;
            while j < m && indexed[j].0 == indexed[i].0 {
                j += 1;
            }
            // [i, j) 是相同值的范围
            match method {
                "average" => {
                    let avg = (i + j + 1) as f64 / 2.0; // 1-based 平均
                    for k in i..j {
                        result[indexed[k].1] = Some(avg);
                    }
                }
                "min" => {
                    for k in i..j {
                        result[indexed[k].1] = Some((i + 1) as f64);
                    }
                }
                "max" => {
                    for k in i..j {
                        result[indexed[k].1] = Some(j as f64);
                    }
                }
                "first" => {
                    for k in i..j {
                        result[indexed[k].1] = Some((k + 1) as f64);
                    }
                }
                _ => {
                    // 默认 average
                    let avg = (i + j - 1) as f64 / 2.0 + 0.5;
                    for k in i..j {
                        result[indexed[k].1] = Some(avg);
                    }
                }
            }
            i = j;
        }
        result
    }

    /// 滑动窗口求和
    pub fn rolling_sum(&self, window: usize, min_periods: Option<usize>) -> Vec<Option<f64>> {
        let n = self.len();
        let min_per = min_periods.unwrap_or(window);
        let values = self.as_f64_vec();
        let mut result: Vec<Option<f64>> = vec![None; n];

        for (i, result_slot) in result.iter_mut().enumerate() {
            if i + 1 < min_per {
                continue;
            }
            let start = (i + 1).saturating_sub(window);
            let end = i + 1;
            let mut sum = 0.0;
            let mut count = 0;
            for v in values[start..end].iter().copied().flatten() {
                sum += v;
                count += 1;
            }
            if count >= min_per {
                *result_slot = Some(sum);
            }
        }
        result
    }

    /// 滑动窗口均值
    pub fn rolling_mean(&self, window: usize, min_periods: Option<usize>) -> Vec<Option<f64>> {
        let n = self.len();
        let min_per = min_periods.unwrap_or(window);
        let values = self.as_f64_vec();
        let mut result: Vec<Option<f64>> = vec![None; n];

        for (i, result_slot) in result.iter_mut().enumerate() {
            if i + 1 < min_per {
                continue;
            }
            let start = (i + 1).saturating_sub(window);
            let end = i + 1;
            let mut sum = 0.0;
            let mut count = 0;
            for v in values[start..end].iter().copied().flatten() {
                sum += v;
                count += 1;
            }
            if count >= min_per {
                *result_slot = Some(sum / count as f64);
            }
        }
        result
    }

    /// 滑动窗口标准差
    pub fn rolling_std(&self, window: usize, min_periods: Option<usize>) -> Vec<Option<f64>> {
        let n = self.len();
        let min_per = min_periods.unwrap_or(window);
        let values = self.as_f64_vec();
        let mut result: Vec<Option<f64>> = vec![None; n];

        for (i, result_slot) in result.iter_mut().enumerate() {
            if i + 1 < min_per {
                continue;
            }
            let start = (i + 1).saturating_sub(window);
            let end = i + 1;
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let mut count = 0;
            for v in values[start..end].iter().copied().flatten() {
                sum += v;
                sum_sq += v * v;
                count += 1;
            }
            if count >= min_per && count > 0 {
                let mean = sum / count as f64;
                let var = (sum_sq / count as f64) - mean * mean;
                *result_slot = Some(var.max(0.0).sqrt());
            }
        }
        result
    }

    /// 扩展窗口求和
    pub fn expanding_sum(&self, min_periods: Option<usize>) -> Vec<Option<f64>> {
        let n = self.len();
        let min_per = min_periods.unwrap_or(1);
        let values = self.as_f64_vec();
        let mut result: Vec<Option<f64>> = vec![None; n];
        let mut cumsum = 0.0;
        let mut count = 0;

        for i in 0..n {
            if let Some(v) = values[i] {
                cumsum += v;
                count += 1;
            }
            if count >= min_per {
                result[i] = Some(cumsum);
            }
        }
        result
    }

    /// 扩展窗口均值
    pub fn expanding_mean(&self, min_periods: Option<usize>) -> Vec<Option<f64>> {
        let n = self.len();
        let min_per = min_periods.unwrap_or(1);
        let values = self.as_f64_vec();
        let mut result: Vec<Option<f64>> = vec![None; n];
        let mut cumsum = 0.0;
        let mut count = 0;

        for i in 0..n {
            if let Some(v) = values[i] {
                cumsum += v;
                count += 1;
            }
            if count >= min_per && count > 0 {
                result[i] = Some(cumsum / count as f64);
            }
        }
        result
    }

    /// 指数加权移动平均
    /// alpha: 平滑因子 (0, 1]
    pub fn ewm_mean(&self, alpha: f64, min_periods: Option<usize>) -> Vec<Option<f64>> {
        let n = self.len();
        let min_per = min_periods.unwrap_or(0);
        let values = self.as_f64_vec();
        let mut result: Vec<Option<f64>> = vec![None; n];

        let mut prev_ema: Option<f64> = None;
        let mut count = 0;

        for i in 0..n {
            if let Some(v) = values[i] {
                count += 1;
                match prev_ema {
                    None => prev_ema = Some(v),
                    Some(prev) => prev_ema = Some(alpha * v + (1.0 - alpha) * prev),
                }
            }
            if count > min_per && prev_ema.is_some() {
                result[i] = prev_ema;
            }
        }
        result
    }

    // ---------- 排序 ----------

    /// 按值排序
    /// ascending=true: 升序，None 放最后
    /// ascending=false: 降序，None 放最前
    pub fn sort_values(&self, ascending: bool) -> Series {
        let data = match &self.data {
            ColumnData::Int(v) => {
                let mut indexed: Vec<(Option<i64>, usize)> =
                    v.iter().enumerate().map(|(i, x)| (*x, i)).collect();
                indexed.sort_by(|a, b| match (a.0, b.0) {
                    (Some(x), Some(y)) => {
                        if ascending {
                            x.cmp(&y)
                        } else {
                            y.cmp(&x)
                        }
                    }
                    (Some(_), None) => {
                        if ascending {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    }
                    (None, Some(_)) => {
                        if ascending {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Less
                        }
                    }
                    (None, None) => std::cmp::Ordering::Equal,
                });
                let sorted: Vec<Option<i64>> = indexed.into_iter().map(|(x, _)| x).collect();
                ColumnData::Int(sorted)
            }
            ColumnData::Float(v) => {
                let mut indexed: Vec<(Option<f64>, usize)> =
                    v.iter().enumerate().map(|(i, x)| (*x, i)).collect();
                indexed.sort_by(|a, b| match (a.0, b.0) {
                    (Some(x), Some(y)) => {
                        if ascending {
                            x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
                        } else {
                            y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)
                        }
                    }
                    (Some(_), None) => {
                        if ascending {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    }
                    (None, Some(_)) => {
                        if ascending {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Less
                        }
                    }
                    (None, None) => std::cmp::Ordering::Equal,
                });
                let sorted: Vec<Option<f64>> = indexed.into_iter().map(|(x, _)| x).collect();
                ColumnData::Float(sorted)
            }
            ColumnData::Bool(v) => {
                let mut indexed: Vec<(Option<bool>, usize)> =
                    v.iter().enumerate().map(|(i, x)| (*x, i)).collect();
                indexed.sort_by(|a, b| match (a.0, b.0) {
                    (Some(x), Some(y)) => {
                        if ascending {
                            x.cmp(&y)
                        } else {
                            y.cmp(&x)
                        }
                    }
                    (Some(_), None) => {
                        if ascending {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    }
                    (None, Some(_)) => {
                        if ascending {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Less
                        }
                    }
                    (None, None) => std::cmp::Ordering::Equal,
                });
                let sorted: Vec<Option<bool>> = indexed.into_iter().map(|(x, _)| x).collect();
                ColumnData::Bool(sorted)
            }
            ColumnData::String(v) => {
                let mut indexed: Vec<(Option<String>, usize)> =
                    v.iter().enumerate().map(|(i, x)| (x.clone(), i)).collect();
                indexed.sort_by(|a, b| match (&a.0, &b.0) {
                    (Some(x), Some(y)) => {
                        if ascending {
                            x.cmp(y)
                        } else {
                            y.cmp(x)
                        }
                    }
                    (Some(_), None) => {
                        if ascending {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    }
                    (None, Some(_)) => {
                        if ascending {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Less
                        }
                    }
                    (None, None) => std::cmp::Ordering::Equal,
                });
                let sorted: Vec<Option<String>> = indexed.into_iter().map(|(x, _)| x).collect();
                ColumnData::String(sorted)
            }
            ColumnData::Categorical(c) => {
                // 对 categorical 按其字符串值排序
                let mut indexed: Vec<(Option<String>, usize)> = c
                    .codes
                    .iter()
                    .enumerate()
                    .map(|(i, code)| {
                        let s = code.and_then(|idx| c.categories.get(idx as usize).cloned());
                        (s, i)
                    })
                    .collect();
                indexed.sort_by(|a, b| match (&a.0, &b.0) {
                    (Some(x), Some(y)) => {
                        if ascending {
                            x.cmp(y)
                        } else {
                            y.cmp(x)
                        }
                    }
                    (Some(_), None) => {
                        if ascending {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    }
                    (None, Some(_)) => {
                        if ascending {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Less
                        }
                    }
                    (None, None) => std::cmp::Ordering::Equal,
                });
                let sorted_codes: Vec<Option<i32>> = indexed
                    .into_iter()
                    .map(|(_, i)| c.codes.get(i).copied().flatten())
                    .collect();
                ColumnData::Categorical(super::dtype::CategoricalData {
                    categories: c.categories.clone(),
                    codes: sorted_codes,
                    ordered: c.ordered,
                })
            }
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 按索引排序
    /// ascending=true: 保持原始顺序
    /// ascending=false: 反转原始顺序
    pub fn sort_index(&self, ascending: bool) -> Series {
        if ascending {
            return self.clone();
        }
        let data = match &self.data {
            ColumnData::Int(v) => ColumnData::Int(v.iter().rev().cloned().collect()),
            ColumnData::Float(v) => ColumnData::Float(v.iter().rev().cloned().collect()),
            ColumnData::Bool(v) => ColumnData::Bool(v.iter().rev().cloned().collect()),
            ColumnData::String(v) => ColumnData::String(v.iter().rev().cloned().collect()),
            ColumnData::Categorical(c) => ColumnData::Categorical(super::dtype::CategoricalData {
                categories: c.categories.clone(),
                codes: c.codes.iter().rev().cloned().collect(),
                ordered: c.ordered,
            }),
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    // ---------- 前向/后向填充 ----------

    /// 前向填充：将 None 填充为前一个非 None 值
    /// 若开头连续 None，则保持 None
    pub fn ffill(&self) -> Series {
        let data = match &self.data {
            ColumnData::Int(v) => {
                let mut result: Vec<Option<i64>> = Vec::with_capacity(v.len());
                let mut last: Option<i64> = None;
                for x in v.iter() {
                    if let Some(val) = x {
                        last = Some(*val);
                        result.push(Some(*val));
                    } else if let Some(l) = last {
                        result.push(Some(l));
                    } else {
                        result.push(None);
                    }
                }
                ColumnData::Int(result)
            }
            ColumnData::Float(v) => {
                let mut result: Vec<Option<f64>> = Vec::with_capacity(v.len());
                let mut last: Option<f64> = None;
                for x in v.iter() {
                    if let Some(val) = x {
                        last = Some(*val);
                        result.push(Some(*val));
                    } else if let Some(l) = last {
                        result.push(Some(l));
                    } else {
                        result.push(None);
                    }
                }
                ColumnData::Float(result)
            }
            ColumnData::Bool(v) => {
                let mut result: Vec<Option<bool>> = Vec::with_capacity(v.len());
                let mut last: Option<bool> = None;
                for x in v.iter() {
                    if let Some(val) = x {
                        last = Some(*val);
                        result.push(Some(*val));
                    } else if let Some(l) = last {
                        result.push(Some(l));
                    } else {
                        result.push(None);
                    }
                }
                ColumnData::Bool(result)
            }
            ColumnData::String(v) => {
                let mut result: Vec<Option<String>> = Vec::with_capacity(v.len());
                let mut last: Option<String> = None;
                for x in v.iter() {
                    if let Some(val) = x {
                        last = Some(val.clone());
                        result.push(Some(val.clone()));
                    } else if let Some(l) = &last {
                        result.push(Some(l.clone()));
                    } else {
                        result.push(None);
                    }
                }
                ColumnData::String(result)
            }
            ColumnData::Categorical(c) => {
                let mut result: Vec<Option<i32>> = Vec::with_capacity(c.codes.len());
                let mut last: Option<i32> = None;
                for code in c.codes.iter() {
                    if let Some(val) = code {
                        last = Some(*val);
                        result.push(Some(*val));
                    } else if let Some(l) = last {
                        result.push(Some(l));
                    } else {
                        result.push(None);
                    }
                }
                ColumnData::Categorical(super::dtype::CategoricalData {
                    categories: c.categories.clone(),
                    codes: result,
                    ordered: c.ordered,
                })
            }
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 后向填充：将 None 填充为后一个非 None 值
    /// 若末尾连续 None，则保持 None
    pub fn bfill(&self) -> Series {
        let data = match &self.data {
            ColumnData::Int(v) => {
                let n = v.len();
                let mut result: Vec<Option<i64>> = vec![None; n];
                let mut last: Option<i64> = None;
                for i in (0..n).rev() {
                    if let Some(val) = v[i] {
                        last = Some(val);
                        result[i] = Some(val);
                    } else if let Some(l) = last {
                        result[i] = Some(l);
                    }
                }
                ColumnData::Int(result)
            }
            ColumnData::Float(v) => {
                let n = v.len();
                let mut result: Vec<Option<f64>> = vec![None; n];
                let mut last: Option<f64> = None;
                for i in (0..n).rev() {
                    if let Some(val) = v[i] {
                        last = Some(val);
                        result[i] = Some(val);
                    } else if let Some(l) = last {
                        result[i] = Some(l);
                    }
                }
                ColumnData::Float(result)
            }
            ColumnData::Bool(v) => {
                let n = v.len();
                let mut result: Vec<Option<bool>> = vec![None; n];
                let mut last: Option<bool> = None;
                for i in (0..n).rev() {
                    if let Some(val) = v[i] {
                        last = Some(val);
                        result[i] = Some(val);
                    } else if let Some(l) = last {
                        result[i] = Some(l);
                    }
                }
                ColumnData::Bool(result)
            }
            ColumnData::String(v) => {
                let n = v.len();
                let mut result: Vec<Option<String>> = vec![None; n];
                let mut last: Option<String> = None;
                for i in (0..n).rev() {
                    if let Some(val) = &v[i] {
                        last = Some(val.clone());
                        result[i] = Some(val.clone());
                    } else if let Some(l) = &last {
                        result[i] = Some(l.clone());
                    }
                }
                ColumnData::String(result)
            }
            ColumnData::Categorical(c) => {
                let n = c.codes.len();
                let mut result: Vec<Option<i32>> = vec![None; n];
                let mut last: Option<i32> = None;
                for i in (0..n).rev() {
                    if let Some(val) = c.codes[i] {
                        last = Some(val);
                        result[i] = Some(val);
                    } else if let Some(l) = last {
                        result[i] = Some(l);
                    }
                }
                ColumnData::Categorical(super::dtype::CategoricalData {
                    categories: c.categories.clone(),
                    codes: result,
                    ordered: c.ordered,
                })
            }
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    // ---------- 字符串方法（仅对 String 类型有效） ----------

    /// 转大写
    pub fn str_upper(&self) -> Series {
        if let ColumnData::String(v) = &self.data {
            let new_v: Vec<Option<String>> = v
                .par_iter()
                .map(|x| x.as_ref().map(|s| s.to_uppercase()))
                .collect();
            Series {
                name: self.name.clone(),
                data: ColumnData::String(new_v),
            }
        } else {
            self.clone()
        }
    }

    /// 转小写
    pub fn str_lower(&self) -> Series {
        if let ColumnData::String(v) = &self.data {
            let new_v: Vec<Option<String>> = v
                .par_iter()
                .map(|x| x.as_ref().map(|s| s.to_lowercase()))
                .collect();
            Series {
                name: self.name.clone(),
                data: ColumnData::String(new_v),
            }
        } else {
            self.clone()
        }
    }

    /// 字符串长度（返回 Int 类型 Series）
    pub fn str_len(&self) -> Series {
        if let ColumnData::String(v) = &self.data {
            let new_v: Vec<Option<i64>> = v
                .par_iter()
                .map(|x| x.as_ref().map(|s| s.chars().count() as i64))
                .collect();
            Series {
                name: self.name.clone(),
                data: ColumnData::Int(new_v),
            }
        } else {
            // 非 String 类型：返回全 None 的 Int 系列
            Series::new_null(self.name.clone(), DType::Int64, self.len())
        }
    }

    /// 去除首尾空白
    pub fn str_strip(&self) -> Series {
        if let ColumnData::String(v) = &self.data {
            let new_v: Vec<Option<String>> = v
                .par_iter()
                .map(|x| x.as_ref().map(|s| s.trim().to_string()))
                .collect();
            Series {
                name: self.name.clone(),
                data: ColumnData::String(new_v),
            }
        } else {
            self.clone()
        }
    }

    /// 是否包含子串（返回 Vec<bool>，None 视为 false）
    pub fn str_contains(&self, pattern: &str) -> Vec<bool> {
        if let ColumnData::String(v) = &self.data {
            v.par_iter()
                .map(|x| x.as_ref().map(|s| s.contains(pattern)).unwrap_or(false))
                .collect()
        } else {
            vec![false; self.len()]
        }
    }

    /// 子串替换
    pub fn str_replace(&self, from: &str, to: &str) -> Series {
        if let ColumnData::String(v) = &self.data {
            let new_v: Vec<Option<String>> = v
                .par_iter()
                .map(|x| x.as_ref().map(|s| s.replace(from, to)))
                .collect();
            Series {
                name: self.name.clone(),
                data: ColumnData::String(new_v),
            }
        } else {
            self.clone()
        }
    }

    // ---------- 插值 / 采样 / 重采样 ----------

    /// 线性插值填充 None
    /// method: "linear" / "nearest" / "zero"
    /// limit: 最多连续填充数量（None 表示无限制）
    pub fn interpolate(&self, method: &str, limit: Option<usize>) -> Series {
        let f64_vec = self.as_f64_vec();
        let n = f64_vec.len();
        if n == 0 {
            return self.clone();
        }
        let mut result: Vec<Option<f64>> = f64_vec.clone();
        let mut i = 0;
        while i < n {
            if result[i].is_some() {
                i += 1;
                continue;
            }
            // 找到连续 None 区间 [i, j)
            let mut j = i;
            while j < n && result[j].is_none() {
                j += 1;
            }
            let left_val = if i > 0 { result[i - 1] } else { None };
            let right_val = if j < n { result[j] } else { None };
            match (left_val, right_val) {
                (Some(lv), Some(rv)) => {
                    let gap = j - i;
                    for (k, slot) in result.iter_mut().enumerate().take(j).skip(i) {
                        if let Some(lim) = limit
                            && k - i >= lim
                        {
                            break;
                        }
                        let frac = (k - i + 1) as f64 / (gap + 1) as f64;
                        *slot = Some(lv + (rv - lv) * frac);
                    }
                }
                (Some(lv), None) => {
                    if method == "linear" {
                        for (k, slot) in result.iter_mut().enumerate().take(j).skip(i) {
                            if let Some(lim) = limit
                                && k - i >= lim
                            {
                                break;
                            }
                            *slot = Some(lv);
                        }
                    }
                }
                (None, Some(rv)) => {
                    if method == "linear" {
                        for (k, slot) in result.iter_mut().enumerate().take(j).skip(i) {
                            if let Some(lim) = limit
                                && j - k > lim
                            {
                                continue;
                            }
                            *slot = Some(rv);
                        }
                    }
                }
                (None, None) => {}
            }
            i = j;
        }
        Series {
            name: self.name.clone(),
            data: ColumnData::Float(result),
        }
    }

    /// 随机采样
    /// n: 采样数量（None 时使用 frac）
    /// frac: 采样比例
    /// replace: 是否有放回
    /// seed: 随机种子
    pub fn sample(
        &self,
        n: Option<usize>,
        frac: Option<f64>,
        replace: bool,
        seed: Option<u64>,
    ) -> Series {
        let len = self.len();
        if len == 0 {
            return self.clone();
        }
        let sample_n = if let Some(f) = frac {
            ((len as f64) * f).round() as usize
        } else {
            n.unwrap_or(1)
        };
        // 简单 LCG 伪随机数生成器（可重现）
        let mut state = seed.unwrap_or(0xC0FFEE);
        let next_rand = |state: &mut u64| {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*state >> 33) as usize
        };
        let indices: Vec<usize> = if replace {
            (0..sample_n).map(|_| next_rand(&mut state) % len).collect()
        } else {
            let take_n = sample_n.min(len);
            // Fisher-Yates 洗牌
            let mut idx: Vec<usize> = (0..len).collect();
            for k in 0..take_n {
                let r = k + next_rand(&mut state) % (len - k);
                idx.swap(k, r);
            }
            idx.truncate(take_n);
            idx
        };
        // 按采样索引构造新 Series
        let mut sampled: Vec<Option<f64>> = Vec::with_capacity(indices.len());
        let f64_vec = self.as_f64_vec();
        for idx in &indices {
            sampled.push(f64_vec[*idx]);
        }
        Series {
            name: self.name.clone(),
            data: ColumnData::Float(sampled),
        }
    }

    /// 时间序列重采样聚合
    /// timestamps: 每行对应的 epoch 秒
    /// freq_seconds: 桶宽度（秒）
    /// agg: "sum"/"mean"/"count"/"min"/"max"/"median"/"first"/"last"
    /// 返回 (桶起始时间列表, 聚合值列表)
    pub fn resample(
        &self,
        timestamps: &[f64],
        freq_seconds: f64,
        agg: &str,
    ) -> (Vec<f64>, Vec<Option<f64>>) {
        let n = self.len();
        if n == 0 || timestamps.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let values = self.as_f64_vec();
        // 按 floor(ts/freq) 分组
        let mut buckets: HashMap<i64, Vec<f64>> = HashMap::new();
        let mut bucket_order: Vec<i64> = Vec::new();
        for i in 0..n.min(timestamps.len()) {
            if values[i].is_none() {
                continue;
            }
            let bucket_id = (timestamps[i] / freq_seconds).floor() as i64;
            if !buckets.contains_key(&bucket_id) {
                bucket_order.push(bucket_id);
            }
            buckets
                .entry(bucket_id)
                .or_default()
                .push(values[i].unwrap());
        }
        // 按时间戳升序
        bucket_order.sort();
        let out_ts: Vec<f64> = bucket_order
            .iter()
            .map(|b| (*b as f64) * freq_seconds)
            .collect();
        let out_vals: Vec<Option<f64>> = bucket_order
            .iter()
            .map(|b| {
                let nums = buckets.get(b).unwrap();
                if nums.is_empty() {
                    return None;
                }
                match agg {
                    "sum" => Some(nums.iter().sum()),
                    "mean" => Some(nums.iter().sum::<f64>() / nums.len() as f64),
                    "count" => Some(nums.len() as f64),
                    "min" => nums.iter().cloned().fold(None::<f64>, |acc, x| {
                        Some(match acc {
                            Some(a) => a.min(x),
                            None => x,
                        })
                    }),
                    "max" => nums.iter().cloned().fold(None::<f64>, |acc, x| {
                        Some(match acc {
                            Some(a) => a.max(x),
                            None => x,
                        })
                    }),
                    "median" => {
                        let mut sorted = nums.clone();
                        sorted
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let m = sorted.len();
                        if m % 2 == 1 {
                            Some(sorted[m / 2])
                        } else {
                            Some((sorted[m / 2 - 1] + sorted[m / 2]) / 2.0)
                        }
                    }
                    "first" => Some(nums[0]),
                    "last" => Some(nums[nums.len() - 1]),
                    _ => None,
                }
            })
            .collect();
        (out_ts, out_vals)
    }

    // ---------- SeriesGroupBy 聚合 ----------

    /// 按 by 列（Vec<String>）分组聚合
    /// by: 每行对应的分组键（字符串表示）
    /// agg: "sum"/"mean"/"count"/"min"/"max"/"median"/"std"/"var"/"prod"/"first"/"last"
    /// 返回 (group_keys, agg_values)
    pub fn groupby_agg_series(&self, by: &[String], agg: &str) -> (Vec<String>, Vec<Option<f64>>) {
        let n = self.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let values = self.as_f64_vec();
        // 分组：键 -> 行索引列表
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        let mut group_order: Vec<String> = Vec::new();
        for (i, key) in by.iter().enumerate().take(n.min(by.len())) {
            if !groups.contains_key(key) {
                group_order.push(key.clone());
            }
            groups.entry(key.clone()).or_default().push(i);
        }
        // 排序键（保持稳定）
        group_order.sort();
        let out_vals: Vec<Option<f64>> = group_order
            .iter()
            .map(|k| {
                let indices = groups.get(k).unwrap();
                let nums: Vec<f64> = indices.iter().filter_map(|&i| values[i]).collect();
                if nums.is_empty() {
                    return None;
                }
                match agg {
                    "sum" => Some(nums.iter().sum()),
                    "mean" => Some(nums.iter().sum::<f64>() / nums.len() as f64),
                    "count" => Some(nums.len() as f64),
                    "min" => nums.iter().cloned().fold(None::<f64>, |acc, x| {
                        Some(match acc {
                            Some(a) => a.min(x),
                            None => x,
                        })
                    }),
                    "max" => nums.iter().cloned().fold(None::<f64>, |acc, x| {
                        Some(match acc {
                            Some(a) => a.max(x),
                            None => x,
                        })
                    }),
                    "median" => {
                        let mut sorted = nums.clone();
                        sorted
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let m = sorted.len();
                        if m % 2 == 1 {
                            Some(sorted[m / 2])
                        } else {
                            Some((sorted[m / 2 - 1] + sorted[m / 2]) / 2.0)
                        }
                    }
                    "std" => {
                        if nums.len() < 2 {
                            return None;
                        }
                        let m = nums.iter().sum::<f64>() / nums.len() as f64;
                        let v =
                            nums.iter().map(|x| (x - m).powi(2)).sum::<f64>() / nums.len() as f64;
                        Some(v.sqrt())
                    }
                    "var" => {
                        if nums.len() < 2 {
                            return None;
                        }
                        let m = nums.iter().sum::<f64>() / nums.len() as f64;
                        Some(nums.iter().map(|x| (x - m).powi(2)).sum::<f64>() / nums.len() as f64)
                    }
                    "prod" => Some(nums.iter().product()),
                    "first" => Some(nums[0]),
                    "last" => Some(nums[nums.len() - 1]),
                    _ => None,
                }
            })
            .collect();
        (group_order, out_vals)
    }

    // ---------- 批量聚合（一次遍历多聚合） ----------

    /// 一次遍历计算多个聚合值
    /// aggs: 聚合名列表，如 ["sum", "mean", "min", "max", "count", "std", "var"]
    pub fn batch_agg(&self, aggs: &[String]) -> Vec<Option<f64>> {
        let values: Vec<f64> = self.as_f64_vec().into_iter().flatten().collect();
        let cnt = values.len();
        if cnt == 0 {
            return aggs.iter().map(|_| None).collect();
        }
        let sum: f64 = values.iter().sum();
        let mean = sum / cnt as f64;
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        aggs.iter()
            .map(|a| match a.as_str() {
                "sum" => Some(sum),
                "mean" => Some(mean),
                "count" => Some(cnt as f64),
                "min" => Some(sorted[0]),
                "max" => Some(sorted[cnt - 1]),
                "median" => {
                    if cnt % 2 == 1 {
                        Some(sorted[cnt / 2])
                    } else {
                        Some((sorted[cnt / 2 - 1] + sorted[cnt / 2]) / 2.0)
                    }
                }
                "std" => {
                    if cnt < 2 {
                        None
                    } else {
                        let v = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / cnt as f64;
                        Some(v.sqrt())
                    }
                }
                "var" => {
                    if cnt < 2 {
                        None
                    } else {
                        Some(values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / cnt as f64)
                    }
                }
                "prod" => Some(values.iter().product()),
                _ => None,
            })
            .collect()
    }

    // ---------- 简单表达式过滤（query 简化版） ----------

    /// 简单比较过滤：列 op 标量
    /// op: ">" / "<" / ">=" / "<=" / "==" / "!="
    /// 返回符合条件行的 mask
    pub fn compare_scalar(&self, op: &str, value: f64) -> Vec<bool> {
        let f64_vec = self.as_f64_vec();
        f64_vec
            .iter()
            .map(|x| match (x, op) {
                (Some(v), ">") => *v > value,
                (Some(v), "<") => *v < value,
                (Some(v), ">=") => *v >= value,
                (Some(v), "<=") => *v <= value,
                (Some(v), "==") => (*v - value).abs() < f64::EPSILON,
                (Some(v), "!=") => (*v - value).abs() >= f64::EPSILON,
                _ => false,
            })
            .collect()
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
    fn new_with_dtype(
        pylist: &Bound<'_, PyList>,
        name: Option<String>,
        dtype: DType,
    ) -> PyResult<Self> {
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
                Series::new_bool(name, v)
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
                Series::new_int(name, v)
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
                Series::new_float(name, v)
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
                Series::new_string(name, v)
            }
            DType::Categorical => {
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
                Series::new_bool(name, v)
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
                Series::new_int(name, v)
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
                Series::new_float(name, v)
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
                Series::new_string(name, v)
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

    /// value_counts: 返回 (value, count) 两条 Series
    fn value_counts<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
        let (order, cnts): (Vec<String>, Vec<usize>) = py.detach(|| {
            let mut counts: HashMap<String, usize> = HashMap::new();
            let mut order: Vec<String> = Vec::new();
            for s in self.inner.to_string_vec() {
                // NaN 跳过
                if s == "NaN" {
                    continue;
                }
                if let std::collections::hash_map::Entry::Vacant(e) = counts.entry(s.clone()) {
                    order.push(s.clone());
                    e.insert(0);
                }
                *counts.get_mut(&s).unwrap() += 1;
            }
            let cnts: Vec<usize> = order.iter().map(|s| counts[s]).collect();
            (order, cnts)
        });
        let values: Vec<&str> = order.iter().map(|s| s.as_str()).collect();
        Ok((PyList::new(py, values)?, PyList::new(py, cnts)?))
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

    // ---------- 分位数 / 排名 ----------

    fn quantile<'py>(&self, py: Python<'py>, q: f64) -> PyResult<Bound<'py, PyAny>> {
        match py.detach(|| self.inner.quantile(q)) {
            Some(v) => Ok(v.into_pyobject(py)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    fn rank<'py>(
        &self,
        py: Python<'py>,
        method: &str,
        ascending: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let ranks = py.detach(|| self.inner.rank(method, ascending));
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

// =====================================================================
// factorize 函数
// =====================================================================

/// 对输入值进行 factorize 编码 (类似 pandas.factorize)
/// 返回 (codes, categories)
#[pyfunction]
pub fn factorize<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyList>,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
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

    let codes_list = PyList::new(py, codes.iter().copied())?;
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

    #[test]
    fn test_series_sort_values() {
        // 整型升序：None 放最后
        let s = Series::new_int(None, vec![Some(3), None, Some(1), Some(2)]);
        let sorted = s.sort_values(true);
        if let ColumnData::Int(v) = &sorted.data {
            assert_eq!(v[0], Some(1));
            assert_eq!(v[1], Some(2));
            assert_eq!(v[2], Some(3));
            assert_eq!(v[3], None);
        } else {
            panic!("dtype 错误");
        }

        // 整型降序：None 放最前
        let sorted_desc = s.sort_values(false);
        if let ColumnData::Int(v) = &sorted_desc.data {
            assert_eq!(v[0], None);
            assert_eq!(v[1], Some(3));
            assert_eq!(v[2], Some(2));
            assert_eq!(v[3], Some(1));
        } else {
            panic!("dtype 错误");
        }

        // 浮点型排序
        let sf = Series::new_float(None, vec![Some(3.0), Some(1.0), Some(2.0)]);
        let sorted_f = sf.sort_values(true);
        if let ColumnData::Float(v) = &sorted_f.data {
            assert_eq!(v[0], Some(1.0));
            assert_eq!(v[1], Some(2.0));
            assert_eq!(v[2], Some(3.0));
        } else {
            panic!("dtype 错误");
        }

        // 字符串排序
        let ss = Series::new_string(
            None,
            vec![Some("banana".to_string()), Some("apple".to_string())],
        );
        let sorted_s = ss.sort_values(true);
        if let ColumnData::String(v) = &sorted_s.data {
            assert_eq!(v[0], Some("apple".to_string()));
            assert_eq!(v[1], Some("banana".to_string()));
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_series_sort_index() {
        let s = Series::new_int(None, vec![Some(1), Some(2), Some(3)]);
        // 升序：保持原顺序
        let sorted_asc = s.sort_index(true);
        if let ColumnData::Int(v) = &sorted_asc.data {
            assert_eq!(v[0], Some(1));
            assert_eq!(v[1], Some(2));
            assert_eq!(v[2], Some(3));
        } else {
            panic!("dtype 错误");
        }

        // 降序：反转原顺序
        let sorted_desc = s.sort_index(false);
        if let ColumnData::Int(v) = &sorted_desc.data {
            assert_eq!(v[0], Some(3));
            assert_eq!(v[1], Some(2));
            assert_eq!(v[2], Some(1));
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_series_ffill() {
        // 前向填充：None 被前一个非 None 替代
        let s = Series::new_int(None, vec![None, Some(1), None, None, Some(2), None]);
        let filled = s.ffill();
        if let ColumnData::Int(v) = &filled.data {
            assert_eq!(v[0], None); // 开头 None 保持
            assert_eq!(v[1], Some(1));
            assert_eq!(v[2], Some(1));
            assert_eq!(v[3], Some(1));
            assert_eq!(v[4], Some(2));
            assert_eq!(v[5], Some(2));
        } else {
            panic!("dtype 错误");
        }

        // 浮点型 ffill
        let sf = Series::new_float(None, vec![Some(1.5), None, Some(2.5)]);
        let filled_f = sf.ffill();
        if let ColumnData::Float(v) = &filled_f.data {
            assert_eq!(v[0], Some(1.5));
            assert_eq!(v[1], Some(1.5));
            assert_eq!(v[2], Some(2.5));
        } else {
            panic!("dtype 错误");
        }

        // 字符串 ffill
        let ss = Series::new_string(
            None,
            vec![Some("a".to_string()), None, Some("b".to_string())],
        );
        let filled_s = ss.ffill();
        if let ColumnData::String(v) = &filled_s.data {
            assert_eq!(v[0], Some("a".to_string()));
            assert_eq!(v[1], Some("a".to_string()));
            assert_eq!(v[2], Some("b".to_string()));
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_series_bfill() {
        // 后向填充：None 被后一个非 None 替代
        let s = Series::new_int(None, vec![None, Some(1), None, None, Some(2), None]);
        let filled = s.bfill();
        if let ColumnData::Int(v) = &filled.data {
            assert_eq!(v[0], Some(1));
            assert_eq!(v[1], Some(1));
            assert_eq!(v[2], Some(2));
            assert_eq!(v[3], Some(2));
            assert_eq!(v[4], Some(2));
            assert_eq!(v[5], None); // 末尾 None 保持
        } else {
            panic!("dtype 错误");
        }

        // 浮点型 bfill
        let sf = Series::new_float(None, vec![Some(1.5), None, Some(2.5)]);
        let filled_f = sf.bfill();
        if let ColumnData::Float(v) = &filled_f.data {
            assert_eq!(v[0], Some(1.5));
            assert_eq!(v[1], Some(2.5));
            assert_eq!(v[2], Some(2.5));
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_series_str_upper() {
        let s = Series::new_string(
            None,
            vec![Some("abc".to_string()), Some("Hello".to_string()), None],
        );
        let upper = s.str_upper();
        if let ColumnData::String(v) = &upper.data {
            assert_eq!(v[0], Some("ABC".to_string()));
            assert_eq!(v[1], Some("HELLO".to_string()));
            assert_eq!(v[2], None);
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_series_str_lower() {
        let s = Series::new_string(
            None,
            vec![Some("ABC".to_string()), Some("Hello".to_string()), None],
        );
        let lower = s.str_lower();
        if let ColumnData::String(v) = &lower.data {
            assert_eq!(v[0], Some("abc".to_string()));
            assert_eq!(v[1], Some("hello".to_string()));
            assert_eq!(v[2], None);
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_series_str_len() {
        let s = Series::new_string(
            None,
            vec![
                Some("abc".to_string()),
                Some("hello".to_string()),
                Some("中".to_string()), // 单字符中文
                None,
            ],
        );
        let len_s = s.str_len();
        if let ColumnData::Int(v) = &len_s.data {
            assert_eq!(v[0], Some(3));
            assert_eq!(v[1], Some(5));
            assert_eq!(v[2], Some(1)); // 字符数而非字节数
            assert_eq!(v[3], None);
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_series_str_contains() {
        let s = Series::new_string(
            None,
            vec![
                Some("hello world".to_string()),
                Some("say hello to rust".to_string()),
                Some("foo".to_string()),
                None,
            ],
        );
        // 包含子串 "hello"
        let mask = s.str_contains("hello");
        assert_eq!(mask, vec![true, true, false, false]);

        // 大小写敏感：Hello 与 hello 不同
        let mask2 = s.str_contains("Hello");
        assert_eq!(mask2, vec![false, false, false, false]);

        // 子串 "ru"
        let mask3 = s.str_contains("ru");
        assert_eq!(mask3, vec![false, true, false, false]);
    }

    #[test]
    fn test_series_str_replace() {
        let s = Series::new_string(
            None,
            vec![
                Some("hello world".to_string()),
                Some("hello rust".to_string()),
                None,
            ],
        );
        let replaced = s.str_replace("hello", "hi");
        if let ColumnData::String(v) = &replaced.data {
            assert_eq!(v[0], Some("hi world".to_string()));
            assert_eq!(v[1], Some("hi rust".to_string()));
            assert_eq!(v[2], None);
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_series_bool_aggregation() {
        // any: 至少一个为 true
        let s1 = Series::new_bool(None, vec![Some(false), Some(true), Some(false)]);
        assert_eq!(s1.any(), Some(true));
        assert_eq!(s1.all(), Some(false));

        // all: 全为 true
        let s2 = Series::new_bool(None, vec![Some(true), Some(true), Some(true)]);
        assert_eq!(s2.any(), Some(true));
        assert_eq!(s2.all(), Some(true));

        // 全 false
        let s3 = Series::new_bool(None, vec![Some(false), Some(false)]);
        assert_eq!(s3.any(), Some(false));
        assert_eq!(s3.all(), Some(false));

        // 空 series
        let s4 = Series::new_bool(None, vec![]);
        assert_eq!(s4.any(), Some(false));
        assert_eq!(s4.all(), Some(true));

        // 非 bool 类型应返回 None
        let s5 = Series::new_int(None, vec![Some(1), Some(2)]);
        assert_eq!(s5.any(), None);
        assert_eq!(s5.all(), None);
    }

    #[test]
    fn test_series_categorical_basic() {
        // 构造一个 categorical Series
        let s = Series::new_categorical(
            Some("cat".to_string()),
            vec!["low".to_string(), "mid".to_string(), "high".to_string()],
            vec![Some(0), Some(2), Some(1), Some(0), None],
            false,
        );
        assert_eq!(s.dtype(), DType::Categorical);
        assert_eq!(s.dtype_name(), "category");
        assert_eq!(s.len(), 5);
        assert_eq!(s.count(), 4); // 5 个中 1 个 None

        // 验证 categories
        let cats = s.cat_categories().expect("应有 categories");
        assert_eq!(cats.len(), 3);
        assert_eq!(cats[0], "low");
        assert_eq!(cats[1], "mid");
        assert_eq!(cats[2], "high");

        // 验证 codes
        let codes = s.cat_codes().expect("应有 codes");
        assert_eq!(codes.len(), 5);
        assert_eq!(codes[0], Some(0));
        assert_eq!(codes[1], Some(2));
        assert_eq!(codes[2], Some(1));
        assert_eq!(codes[3], Some(0));
        assert_eq!(codes[4], None);

        // 验证 ordered 标志
        assert!(!s.cat_ordered().unwrap());

        // 添加新 categories
        let s_add = s
            .cat_add_categories(&["extra".to_string()])
            .expect("应能添加 categories");
        let cats2 = s_add.cat_categories().expect("应有 categories");
        assert_eq!(cats2.len(), 4);
        assert_eq!(cats2[3], "extra");

        // 重命名 categories
        let s_rename = s
            .cat_rename_categories(&["L".to_string(), "M".to_string(), "H".to_string()])
            .expect("应能重命名 categories");
        let cats3 = s_rename.cat_categories().expect("应有 categories");
        assert_eq!(cats3[0], "L");
        assert_eq!(cats3[1], "M");
        assert_eq!(cats3[2], "H");

        // 移除未使用 categories：原 categories 都被使用，所以保持不变
        let s_unused = s.cat_remove_unused_categories().expect("应能移除未使用");
        let cats4 = s_unused.cat_categories().expect("应有 categories");
        assert_eq!(cats4.len(), 3);
    }

    #[test]
    fn test_series_quantile() {
        let s = Series::new_float(
            None,
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)],
        );
        assert!((s.quantile(0.5).unwrap() - 3.0).abs() < 1e-10);
        assert!((s.quantile(0.0).unwrap() - 1.0).abs() < 1e-10);
        assert!((s.quantile(1.0).unwrap() - 5.0).abs() < 1e-10);
        assert!((s.quantile(0.25).unwrap() - 2.0).abs() < 1e-10);
        assert!((s.quantile(0.75).unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_series_rank() {
        let s = Series::new_float(None, vec![Some(3.0), Some(1.0), Some(2.0), Some(1.0)]);
        let ranks = s.rank("average", true);
        // 3.0 -> rank 4.0, 1.0 -> rank 1.5, 2.0 -> rank 3.0, 1.0 -> rank 1.5
        assert!((ranks[0].unwrap() - 4.0).abs() < 1e-10);
        assert!((ranks[1].unwrap() - 1.5).abs() < 1e-10);
        assert!((ranks[2].unwrap() - 3.0).abs() < 1e-10);
        assert!((ranks[3].unwrap() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_series_rolling() {
        let s = Series::new_float(
            None,
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)],
        );
        let sums = s.rolling_sum(3, None);
        assert_eq!(sums[0], None); // 窗口不足
        assert_eq!(sums[1], None);
        assert!((sums[2].unwrap() - 6.0).abs() < 1e-10); // 1+2+3
        assert!((sums[3].unwrap() - 9.0).abs() < 1e-10); // 2+3+4
        assert!((sums[4].unwrap() - 12.0).abs() < 1e-10); // 3+4+5

        let means = s.rolling_mean(3, None);
        assert!((means[2].unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_series_expanding() {
        let s = Series::new_float(None, vec![Some(1.0), Some(2.0), Some(3.0)]);
        let sums = s.expanding_sum(Some(1));
        assert!((sums[0].unwrap() - 1.0).abs() < 1e-10);
        assert!((sums[1].unwrap() - 3.0).abs() < 1e-10);
        assert!((sums[2].unwrap() - 6.0).abs() < 1e-10);

        let means = s.expanding_mean(Some(1));
        assert!((means[0].unwrap() - 1.0).abs() < 1e-10);
        assert!((means[1].unwrap() - 1.5).abs() < 1e-10);
        assert!((means[2].unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_series_ewm() {
        let s = Series::new_float(None, vec![Some(1.0), Some(2.0), Some(3.0)]);
        let ema = s.ewm_mean(0.5, Some(0));
        assert!((ema[0].unwrap() - 1.0).abs() < 1e-10);
        assert!((ema[1].unwrap() - 1.5).abs() < 1e-10);
        assert!((ema[2].unwrap() - 2.25).abs() < 1e-10);
    }

    #[test]
    fn test_series_dt_year() {
        // 2020-01-01 00:00:00 UTC = 1577836800
        let s = Series::new_float(None, vec![Some(1577836800.0), Some(1609459200.0)]); // 2020, 2021
        let years = s.dt_year();
        if let ColumnData::Int(v) = &years.data {
            assert_eq!(v[0], Some(2020));
            assert_eq!(v[1], Some(2021));
        } else {
            panic!("应为 Int 类型");
        }
    }

    #[test]
    fn test_series_dt_month_day() {
        // 2020-03-15 12:30:45 UTC
        let ts = 1584275445.0;
        let s = Series::new_float(None, vec![Some(ts)]);
        let month = s.dt_month();
        let day = s.dt_day();
        let hour = s.dt_hour();
        let minute = s.dt_minute();
        let second = s.dt_second();
        if let ColumnData::Int(v) = &month.data {
            assert_eq!(v[0], Some(3));
        } else {
            panic!("月份应为3");
        }
        if let ColumnData::Int(v) = &day.data {
            assert_eq!(v[0], Some(15));
        } else {
            panic!("日应为15");
        }
        if let ColumnData::Int(v) = &hour.data {
            assert_eq!(v[0], Some(12));
        } else {
            panic!("小时应为12");
        }
        if let ColumnData::Int(v) = &minute.data {
            assert_eq!(v[0], Some(30));
        } else {
            panic!("分钟应为30");
        }
        if let ColumnData::Int(v) = &second.data {
            assert_eq!(v[0], Some(45));
        } else {
            panic!("秒应为45");
        }
    }

    #[test]
    fn test_series_dt_dayofweek() {
        // 1970-01-01 = 周四 = 3
        let s = Series::new_float(None, vec![Some(0.0)]);
        let dow = s.dt_dayofweek();
        if let ColumnData::Int(v) = &dow.data {
            assert_eq!(v[0], Some(3));
        } else {
            panic!("dayofweek 应为3（周四）");
        }
    }
}
