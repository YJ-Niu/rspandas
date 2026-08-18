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
}
