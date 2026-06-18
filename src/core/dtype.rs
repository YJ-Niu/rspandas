//! 数据类型系统
//!
//! 定义 DType 枚举和 ColumnData 变体枚举。
//! ColumnData 是列存储的核心，使用 Vec<Option<T>> 表达缺失值。

use pyo3::prelude::*;
use pyo3::types::PyList;

/// 逻辑类型，对应 pandas 的 dtype 名称
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Int64,
    Float64,
    Bool,
    Object,
}

impl DType {
    /// 返回与 pandas 等价的 dtype 字符串
    pub fn as_str(&self) -> &'static str {
        match self {
            DType::Int64 => "int64",
            DType::Float64 => "float64",
            DType::Bool => "bool",
            DType::Object => "object",
        }
    }

    pub fn from_str(s: &str) -> Option<DType> {
        match s.to_lowercase().as_str() {
            "int64" | "int" | "i64" => Some(DType::Int64),
            "float64" | "float" | "f64" => Some(DType::Float64),
            "bool" | "boolean" => Some(DType::Bool),
            "object" | "str" | "string" => Some(DType::Object),
            _ => None,
        }
    }
}

/// 单列数据，支持 4 种基础类型 + 缺失值 (None)
#[derive(Debug, Clone)]
pub enum ColumnData {
    Int(Vec<Option<i64>>),
    Float(Vec<Option<f64>>),
    Bool(Vec<Option<bool>>),
    String(Vec<Option<String>>),
}

impl ColumnData {
    /// 返回元素数量
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Int(v) => v.len(),
            ColumnData::Float(v) => v.len(),
            ColumnData::Bool(v) => v.len(),
            ColumnData::String(v) => v.len(),
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
            ColumnData::Int(v) => v.iter().filter(|x| x.is_some()).count(),
            ColumnData::Float(v) => v.iter().filter(|x| x.is_some()).count(),
            ColumnData::Bool(v) => v.iter().filter(|x| x.is_some()).count(),
            ColumnData::String(v) => v.iter().filter(|x| x.is_some()).count(),
        }
    }

    /// 根据 mask 过滤行 (mask.len() == self.len())
    pub fn filter(&self, mask: &[bool]) -> ColumnData {
        match self {
            ColumnData::Int(v) => {
                let r: Vec<Option<i64>> = v
                    .iter()
                    .zip(mask.iter())
                    .filter_map(|(val, m)| if *m { Some(val.clone()) } else { None })
                    .collect();
                ColumnData::Int(r)
            }
            ColumnData::Float(v) => {
                let r: Vec<Option<f64>> = v
                    .iter()
                    .zip(mask.iter())
                    .filter_map(|(val, m)| if *m { Some(val.clone()) } else { None })
                    .collect();
                ColumnData::Float(r)
            }
            ColumnData::Bool(v) => {
                let r: Vec<Option<bool>> = v
                    .iter()
                    .zip(mask.iter())
                    .filter_map(|(val, m)| if *m { Some(val.clone()) } else { None })
                    .collect();
                ColumnData::Bool(r)
            }
            ColumnData::String(v) => {
                let r: Vec<Option<String>> = v
                    .iter()
                    .zip(mask.iter())
                    .filter_map(|(val, m)| if *m { Some(val.clone()) } else { None })
                    .collect();
                ColumnData::String(r)
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
