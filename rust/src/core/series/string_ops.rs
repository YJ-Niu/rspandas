//! Series 字符串方法：str_upper / str_lower / str_len / str_strip / str_contains / str_replace。
//!
//! 字符串方法仅对 `ColumnData::String` 列有效；其他类型：
//! - 转换类 (upper/lower/strip/replace) 返回原 Series 克隆
//! - str_len 返回全 None 的 Int 系列
//! - str_contains 返回全 false 掩码

use rayon::prelude::*;

use crate::core::dtype::{ColumnData, DType};
use crate::core::series::Series;

impl Series {
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

    /// 返回每个元素对应的类型编码（缺失值为 None）。
    ///
    /// 供 Python 层 ``apply(type)`` 使用，避免在 Python 中逐元素调用 ``type()``
    /// 并做二次 ``str()`` 转换。object 列按存储的字符串内容分类：
    ///
    /// - 1 -> int
    /// - 2 -> float
    /// - 3 -> bool
    /// - 4 -> str
    ///
    /// 其它基础类型列（int64/float64/bool）返回固定编码；分类列视为 str。
    pub fn type_codes(&self) -> Vec<Option<u8>> {
        match &self.data {
            ColumnData::Int(v) => v.iter().map(|x| x.map(|_| 1u8)).collect(),
            ColumnData::Float(v) => v.iter().map(|x| x.map(|_| 2u8)).collect(),
            ColumnData::Bool(v) => v.iter().map(|x| x.map(|_| 3u8)).collect(),
            ColumnData::Categorical(c) => c.codes.iter().map(|x| x.map(|_| 4u8)).collect(),
            ColumnData::String(v) => v
                .par_iter()
                .map(|x| x.as_ref().map(|s| classify_type_code(s)))
                .collect(),
        }
    }
}

/// 将字符串分类为 Python 类型编码（与 ``infer_typed_col`` 的分类顺序一致）。
fn classify_type_code(s: &str) -> u8 {
    if s.parse::<i64>().is_ok() {
        1
    } else if s.parse::<f64>().is_ok() {
        2
    } else if matches!(s, "True" | "TRUE" | "true" | "False" | "FALSE" | "false") {
        3
    } else {
        4
    }
}
