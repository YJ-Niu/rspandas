//! Series 比较运算：返回布尔掩码 (`Vec<bool>`) 的标量比较方法。
//!
//! 这些方法仅对匹配 dtype 的列产生有效掩码，其他类型返回全 false，
//! 与 pandas 中类型不匹配时静默不命中的语义保持一致。

use rayon::prelude::*;

use crate::core::dtype::ColumnData;
use crate::core::series::Series;

impl Series {
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
}
