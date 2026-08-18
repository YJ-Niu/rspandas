//! Series 基础方法：构造器、属性、切片、唯一值与通用辅助方法。
//!
//! 这里集中放置与单个 Series 自身形态相关、且被多模块复用的辅助方法
//! （如 :meth:`as_f64_vec` / :meth:`to_string_vec`）。

use rayon::prelude::*;

use crate::core::dtype::{CategoricalData, ColumnData, DType};
use crate::core::series::Series;

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
            data: ColumnData::Categorical(CategoricalData {
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
            DType::Categorical => ColumnData::Categorical(CategoricalData {
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

    // ---------- 通用辅助方法 ----------

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
}
