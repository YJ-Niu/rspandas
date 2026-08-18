//! Series 聚合统计：count / sum / mean / min / max / std / var / median / any / all。
//!
//! 聚合语义与 pandas 默认行为对齐：
//! - 缺失值 (None 与 NaN) 一律跳过 (skipna=True)
//! - 空列或全缺失列返回 None；any/all 在空列上分别返回 false/true

use rayon::prelude::*;

use crate::core::dtype::ColumnData;
use crate::core::series::Series;

impl Series {
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
        let r: bool = match &self.data {
            ColumnData::Bool(v) => v.par_iter().filter_map(|x| *x).any(|x| x),
            ColumnData::Int(v) => v.par_iter().filter_map(|x| *x).any(|x| x != 0),
            ColumnData::Float(v) => v.par_iter().filter_map(|x| *x).any(|x| x != 0.0),
            ColumnData::String(v) => v
                .par_iter()
                .filter_map(|x| x.clone())
                .any(|s| !s.is_empty()),
            ColumnData::Categorical(c) => c.codes.par_iter().any(|x| x.is_some()),
        };
        Some(r)
    }
    pub fn all(&self) -> Option<bool> {
        // 空列：all 返回 True（与 pandas 一致）
        // 有缺失值：skipna=True（默认）跳过 None；全为 None 则返回 True
        let r: bool = match &self.data {
            ColumnData::Bool(v) => v.par_iter().filter_map(|x| *x).all(|x| x),
            ColumnData::Int(v) => v.par_iter().filter_map(|x| *x).all(|x| x != 0),
            ColumnData::Float(v) => v.par_iter().filter_map(|x| *x).all(|x| x != 0.0),
            ColumnData::String(v) => v
                .par_iter()
                .filter_map(|x| x.clone())
                .all(|s| !s.is_empty()),
            ColumnData::Categorical(c) => {
                // 分类列：所有非 None 视为 True
                c.codes.par_iter().all(|x| x.is_some())
            }
        };
        Some(r)
    }
}
