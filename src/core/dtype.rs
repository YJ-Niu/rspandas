//! 数据类型系统
//!
//! 定义 DType 枚举和 ColumnData 变体枚举。
//! ColumnData 是列存储的核心，使用 Vec<Option<T>> 表达缺失值。

use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;

/// 逻辑类型，对应 pandas 的 dtype 名称
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Int64,
    Float64,
    Bool,
    Object,
    Categorical,
}

impl DType {
    /// 返回与 pandas 等价的 dtype 字符串
    pub fn as_str(&self) -> &'static str {
        match self {
            DType::Int64 => "int64",
            DType::Float64 => "float64",
            DType::Bool => "bool",
            DType::Object => "object",
            DType::Categorical => "category",
        }
    }

    pub fn from_str(s: &str) -> Option<DType> {
        match s.to_lowercase().as_str() {
            "int64" | "int" | "i64" => Some(DType::Int64),
            "float64" | "float" | "f64" => Some(DType::Float64),
            "bool" | "boolean" => Some(DType::Bool),
            "object" | "str" | "string" => Some(DType::Object),
            "category" | "categorical" => Some(DType::Categorical),
            _ => None,
        }
    }
}

/// 分类数据：用整数 codes 引用 categories 列表
#[derive(Debug, Clone)]
pub struct CategoricalData {
    pub categories: Vec<String>,
    pub codes: Vec<Option<i32>>,
    pub ordered: bool,
}

/// 单列数据，支持 4 种基础类型 + 缺失值 (None) + Categorical
#[derive(Debug, Clone)]
pub enum ColumnData {
    Int(Vec<Option<i64>>),
    Float(Vec<Option<f64>>),
    Bool(Vec<Option<bool>>),
    String(Vec<Option<String>>),
    Categorical(CategoricalData),
}

impl ColumnData {
    /// 返回元素数量
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Int(v) => v.len(),
            ColumnData::Float(v) => v.len(),
            ColumnData::Bool(v) => v.len(),
            ColumnData::String(v) => v.len(),
            ColumnData::Categorical(c) => c.codes.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 逻辑 dtype
    pub fn dtype(&self) -> DType {
        match self {
            ColumnData::Int(_) => DType::Int64,
            ColumnData::Float(_) => DType::Float64,
            ColumnData::Bool(_) => DType::Bool,
            ColumnData::String(_) => DType::Object,
            ColumnData::Categorical(_) => DType::Categorical,
        }
    }

    pub fn dtype_name(&self) -> &'static str {
        self.dtype().as_str()
    }

    /// 切片 [start, end)
    pub fn slice(&self, start: usize, end: usize) -> ColumnData {
        let end = end.min(self.len());
        let start = start.min(end);
        match self {
            ColumnData::Int(v) => ColumnData::Int(v[start..end].to_vec()),
            ColumnData::Float(v) => ColumnData::Float(v[start..end].to_vec()),
            ColumnData::Bool(v) => ColumnData::Bool(v[start..end].to_vec()),
            ColumnData::String(v) => ColumnData::String(v[start..end].to_vec()),
            ColumnData::Categorical(c) => ColumnData::Categorical(CategoricalData {
                categories: c.categories.clone(),
                codes: c.codes[start..end].to_vec(),
                ordered: c.ordered,
            }),
        }
    }

    pub fn head(&self, n: usize) -> ColumnData {
        let n = n.min(self.len());
        self.slice(0, n)
    }

    pub fn tail(&self, n: usize) -> ColumnData {
        let len = self.len();
        let n = n.min(len);
        self.slice(len - n, len)
    }

    /// 非缺失值数量
    pub fn count_non_null(&self) -> usize {
        match self {
            ColumnData::Int(v) => v.par_iter().filter(|x| x.is_some()).count(),
            ColumnData::Float(v) => v.par_iter().filter(|x| x.is_some()).count(),
            ColumnData::Bool(v) => v.par_iter().filter(|x| x.is_some()).count(),
            ColumnData::String(v) => v.par_iter().filter(|x| x.is_some()).count(),
            ColumnData::Categorical(c) => c.codes.par_iter().filter(|x| x.is_some()).count(),
        }
    }

    /// 根据 mask 过滤行 (mask.len() == self.len())
    pub fn filter(&self, mask: &[bool]) -> ColumnData {
        match self {
            ColumnData::Int(v) => {
                let r: Vec<Option<i64>> = v
                    .par_iter()
                    .zip(mask.par_iter())
                    .filter_map(|(val, m)| if *m { Some(val.clone()) } else { None })
                    .collect();
                ColumnData::Int(r)
            }
            ColumnData::Float(v) => {
                let r: Vec<Option<f64>> = v
                    .par_iter()
                    .zip(mask.par_iter())
                    .filter_map(|(val, m)| if *m { Some(val.clone()) } else { None })
                    .collect();
                ColumnData::Float(r)
            }
            ColumnData::Bool(v) => {
                let r: Vec<Option<bool>> = v
                    .par_iter()
                    .zip(mask.par_iter())
                    .filter_map(|(val, m)| if *m { Some(val.clone()) } else { None })
                    .collect();
                ColumnData::Bool(r)
            }
            ColumnData::String(v) => {
                let r: Vec<Option<String>> = v
                    .par_iter()
                    .zip(mask.par_iter())
                    .filter_map(|(val, m)| if *m { Some(val.clone()) } else { None })
                    .collect();
                ColumnData::String(r)
            }
            ColumnData::Categorical(c) => {
                let r: Vec<Option<i32>> = c.codes
                    .par_iter()
                    .zip(mask.par_iter())
                    .filter_map(|(val, m)| if *m { Some(val.clone()) } else { None })
                    .collect();
                ColumnData::Categorical(CategoricalData {
                    categories: c.categories.clone(),
                    codes: r,
                    ordered: c.ordered,
                })
            }
        }
    }

    /// 转换为 Python list (None -> Python None)
    pub fn to_py_list<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        match self {
            ColumnData::Int(v) => {
                let list = PyList::empty(py);
                for opt in v {
                    match opt {
                        Some(x) => list.append(*x).unwrap(),
                        None => list.append(py.None()).unwrap(),
                    }
                }
                list
            }
            ColumnData::Float(v) => {
                let list = PyList::empty(py);
                for opt in v {
                    match opt {
                        Some(x) => list.append(*x).unwrap(),
                        None => list.append(py.None()).unwrap(),
                    }
                }
                list
            }
            ColumnData::Bool(v) => {
                let list = PyList::empty(py);
                for opt in v {
                    match opt {
                        Some(x) => list.append(*x).unwrap(),
                        None => list.append(py.None()).unwrap(),
                    }
                }
                list
            }
            ColumnData::String(v) => {
                let list = PyList::empty(py);
                for opt in v {
                    match opt {
                        Some(x) => list.append(x).unwrap(),
                        None => list.append(py.None()).unwrap(),
                    }
                }
                list
            }
            ColumnData::Categorical(c) => {
                let list = PyList::empty(py);
                for code in &c.codes {
                    match code {
                        Some(code_idx) => {
                            let cat_str = c.categories.get(*code_idx as usize)
                                .cloned()
                                .unwrap_or_else(|| "NaN".to_string());
                            list.append(cat_str).unwrap();
                        }
                        None => list.append(py.None()).unwrap(),
                    }
                }
                list
            }
        }
    }

    /// 返回 bool 列: True 表示该位置是 None
    pub fn isnull(&self) -> Vec<bool> {
        match self {
            ColumnData::Int(v) => v.par_iter().map(|x| x.is_none()).collect(),
            ColumnData::Float(v) => v.par_iter().map(|x| x.is_none()).collect(),
            ColumnData::Bool(v) => v.par_iter().map(|x| x.is_none()).collect(),
            ColumnData::String(v) => v.par_iter().map(|x| x.is_none()).collect(),
            ColumnData::Categorical(c) => c.codes.par_iter().map(|x| x.is_none()).collect(),
        }
    }

    /// 返回 bool 列: True 表示该位置不是 None
    pub fn notnull(&self) -> Vec<bool> {
        match self {
            ColumnData::Int(v) => v.par_iter().map(|x| x.is_some()).collect(),
            ColumnData::Float(v) => v.par_iter().map(|x| x.is_some()).collect(),
            ColumnData::Bool(v) => v.par_iter().map(|x| x.is_some()).collect(),
            ColumnData::String(v) => v.par_iter().map(|x| x.is_some()).collect(),
            ColumnData::Categorical(c) => c.codes.par_iter().map(|x| x.is_some()).collect(),
        }
    }

    /// 用给定的值填充 None (类型必须匹配)
    pub fn fillna_i64(&self, v: i64) -> ColumnData {
        if let ColumnData::Int(col) = self {
            ColumnData::Int(col.par_iter().map(|x| Some(x.unwrap_or(v))).collect())
        } else { self.clone() }
    }
    pub fn fillna_f64(&self, v: f64) -> ColumnData {
        if let ColumnData::Float(col) = self {
            ColumnData::Float(col.par_iter().map(|x| Some(x.unwrap_or(v))).collect())
        } else { self.clone() }
    }
    pub fn fillna_bool(&self, v: bool) -> ColumnData {
        if let ColumnData::Bool(col) = self {
            ColumnData::Bool(col.par_iter().map(|x| Some(x.unwrap_or(v))).collect())
        } else { self.clone() }
    }
    pub fn fillna_string(&self, v: &str) -> ColumnData {
        if let ColumnData::String(col) = self {
            let v_clone = v.to_string();
            ColumnData::String(col.par_iter().map(|x| Some(x.clone().unwrap_or_else(|| v_clone.clone()))).collect())
        } else { self.clone() }
    }
    pub fn fillna_categorical(&self, v: &str) -> ColumnData {
        if let ColumnData::Categorical(c) = self {
            let v_clone = v.to_string();
            // 找到或添加 fill value 的 code
            let fill_code = c.categories.iter().position(|cat| cat == &v_clone)
                .map(|i| i as i32)
                .unwrap_or_else(|| c.categories.len() as i32);
            let mut new_categories = c.categories.clone();
            let new_codes: Vec<Option<i32>> = c.codes.par_iter().map(|code| {
                match code {
                    Some(c) => Some(*c),
                    None => {
                        if fill_code as usize >= new_categories.len() {
                            Some(fill_code) // 需要外部同步添加
                        } else {
                            Some(fill_code)
                        }
                    }
                }
            }).collect();
            // 如果 fill_code 对应的 category 不在列表中，添加它
            if fill_code as usize >= new_categories.len() {
                new_categories.push(v_clone);
            }
            ColumnData::Categorical(CategoricalData {
                categories: new_categories,
                codes: new_codes,
                ordered: c.ordered,
            })
        } else { self.clone() }
    }

    /// 删除 None 所在行
    pub fn dropna(&self) -> ColumnData {
        match self {
            ColumnData::Int(v) => {
                ColumnData::Int(v.par_iter().filter_map(|x| *x).map(Some).collect())
            }
            ColumnData::Float(v) => {
                ColumnData::Float(v.par_iter().filter_map(|x| *x).map(Some).collect())
            }
            ColumnData::Bool(v) => {
                ColumnData::Bool(v.par_iter().filter_map(|x| *x).map(Some).collect())
            }
            ColumnData::String(v) => {
                ColumnData::String(v.par_iter().filter_map(|x| x.clone()).map(Some).collect())
            }
            ColumnData::Categorical(c) => {
                let codes: Vec<Option<i32>> = c.codes.par_iter()
                    .filter_map(|x| *x).map(Some).collect();
                ColumnData::Categorical(CategoricalData {
                    categories: c.categories.clone(),
                    codes,
                    ordered: c.ordered,
                })
            }
        }
    }

    /// 获取不同值 (保持首次出现顺序)
    pub fn unique(&self) -> ColumnData {
        match self {
            ColumnData::Int(v) => {
                let mut seen = std::collections::HashSet::new();
                let mut out: Vec<Option<i64>> = Vec::new();
                for x in v {
                    if let Some(val) = x {
                        if seen.insert(*val) {
                            out.push(Some(*val));
                        }
                    }
                }
                ColumnData::Int(out)
            }
            ColumnData::Float(v) => {
                let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
                let mut out: Vec<Option<f64>> = Vec::new();
                for x in v {
                    if let Some(val) = x {
                        if seen.insert(val.to_bits()) {
                            out.push(Some(*val));
                        }
                    }
                }
                ColumnData::Float(out)
            }
            ColumnData::Bool(v) => {
                let mut seen = std::collections::HashSet::new();
                let mut out: Vec<Option<bool>> = Vec::new();
                for x in v {
                    if let Some(val) = x {
                        if seen.insert(*val) {
                            out.push(Some(*val));
                        }
                    }
                }
                ColumnData::Bool(out)
            }
            ColumnData::String(v) => {
                let mut seen = std::collections::HashSet::new();
                let mut out: Vec<Option<String>> = Vec::new();
                for x in v {
                    if let Some(val) = x {
                        if seen.insert(val.clone()) {
                            out.push(Some(val.clone()));
                        }
                    }
                }
                ColumnData::String(out)
            }
            ColumnData::Categorical(c) => {
                let mut seen = std::collections::HashSet::new();
                let mut out_codes: Vec<Option<i32>> = Vec::new();
                let mut out_categories: Vec<String> = Vec::new();
                for code in &c.codes {
                    if let Some(code_idx) = code {
                        if seen.insert(*code_idx) {
                            let cat_str = c.categories.get(*code_idx as usize)
                                .cloned()
                                .unwrap_or_default();
                            let new_code = out_categories.len() as i32;
                            out_categories.push(cat_str);
                            out_codes.push(Some(new_code));
                        }
                    }
                }
                ColumnData::Categorical(CategoricalData {
                    categories: out_categories,
                    codes: out_codes,
                    ordered: c.ordered,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_as_str() {
        assert_eq!(DType::Int64.as_str(), "int64");
        assert_eq!(DType::Float64.as_str(), "float64");
        assert_eq!(DType::Bool.as_str(), "bool");
        assert_eq!(DType::Object.as_str(), "object");
    }

    #[test]
    fn test_dtype_from_str() {
        assert_eq!(DType::from_str("int64"), Some(DType::Int64));
        assert_eq!(DType::from_str("FLOAT64"), Some(DType::Float64));
        assert_eq!(DType::from_str("bool"), Some(DType::Bool));
        assert_eq!(DType::from_str("str"), Some(DType::Object));
        assert_eq!(DType::from_str("unknown"), None);
    }

    #[test]
    fn test_column_len() {
        let c = ColumnData::Int(vec![Some(1), Some(2), None]);
        assert_eq!(c.len(), 3);
        assert!(!c.is_empty());
    }

    #[test]
    fn test_column_count_non_null() {
        let c = ColumnData::Float(vec![Some(1.0), None, Some(2.0), None, Some(3.0)]);
        assert_eq!(c.count_non_null(), 3);
    }

    #[test]
    fn test_column_dtype() {
        assert_eq!(ColumnData::Int(vec![]).dtype(), DType::Int64);
        assert_eq!(ColumnData::Float(vec![]).dtype(), DType::Float64);
        assert_eq!(ColumnData::Bool(vec![]).dtype(), DType::Bool);
        assert_eq!(ColumnData::String(vec![]).dtype(), DType::Object);
    }

    #[test]
    fn test_column_slice() {
        let c = ColumnData::Int(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        let s = c.slice(1, 4);
        if let ColumnData::Int(v) = s {
            assert_eq!(v, vec![Some(2), Some(3), Some(4)]);
        } else {
            panic!("wrong type");
        }
    }

    #[test]
    fn test_column_head_tail() {
        let c = ColumnData::Int(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        if let ColumnData::Int(v) = c.head(2) {
            assert_eq!(v, vec![Some(1), Some(2)]);
        } else {
            panic!();
        }
        if let ColumnData::Int(v) = c.tail(2) {
            assert_eq!(v, vec![Some(4), Some(5)]);
        } else {
            panic!();
        }
    }

    #[test]
    fn test_column_filter() {
        let c = ColumnData::Int(vec![Some(1), Some(2), Some(3), Some(4)]);
        let mask = vec![true, false, true, false];
        if let ColumnData::Int(v) = c.filter(&mask) {
            assert_eq!(v, vec![Some(1), Some(3)]);
        } else {
            panic!();
        }
    }
}
