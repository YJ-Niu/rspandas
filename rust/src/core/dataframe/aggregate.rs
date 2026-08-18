//! DataFrame 并行聚合方法：所有列并行的 sum/mean/std/min/max/count/quantile/any/all/nunique/isnull/notnull。
//!
//! 这些方法利用 rayon 并行遍历所有列，每个 Series 调用对应的聚合方法。
//! 返回值按列顺序排列，非数值列返回 None。

use rayon::prelude::*;

use crate::core::dataframe::DataFrame;
use crate::core::dtype::DType;
use crate::core::series::Series;

impl DataFrame {
    // ---------- 列并行批量方法（Python 层变薄：一次性 R 调用） ----------

    /// 所有列并行执行批量聚合（每个 Series 调用 batch_agg）
    /// aggs: 聚合名列表如 ["count","sum","mean","std","min","max"]
    /// 返回: 按列顺序，每列对应一个聚合结果 Vec<Option<f64>>（长度 = aggs.len()）
    pub fn par_batch_agg(&self, aggs: &[String]) -> Vec<Vec<Option<f64>>> {
        self.data.par_iter().map(|s| s.batch_agg(aggs)).collect()
    }

    /// 所有列并行计算 sum（跳过 None / NaN）
    /// 返回按列顺序的结果: f64 列返回值，非数值列返回 None
    pub fn par_sum_all(&self) -> Vec<Option<f64>> {
        self.data
            .par_iter()
            .map(|s| match s.dtype() {
                DType::Float64 => s.sum_f64(),
                DType::Int64 => s.sum_i64().map(|v| v as f64),
                DType::Bool => Some(s.sum_bool() as f64),
                _ => None,
            })
            .collect()
    }

    /// 所有列并行计算 mean（仅数值列）
    pub fn par_mean_all(&self) -> Vec<Option<f64>> {
        self.data.par_iter().map(|s| s.mean()).collect()
    }

    /// 所有列并行计算 std（ddof=1 默认）
    pub fn par_std_all(&self, ddof: usize) -> Vec<Option<f64>> {
        self.data
            .par_iter()
            .map(|s| {
                let vs: Vec<f64> = s.as_f64_vec().into_iter().flatten().collect();
                let n = vs.len();
                if n <= ddof {
                    return None;
                }
                let mean: f64 = vs.iter().sum::<f64>() / n as f64;
                let var: f64 =
                    vs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - ddof) as f64;
                Some(var.sqrt())
            })
            .collect()
    }

    /// 所有列并行计算 min（仅数值列）
    pub fn par_min_all(&self) -> Vec<Option<f64>> {
        self.data
            .par_iter()
            .map(|s| match s.dtype() {
                DType::Float64 => s.min_f64(),
                DType::Int64 => s.min_i64().map(|v| v as f64),
                _ => None,
            })
            .collect()
    }

    /// 所有列并行计算 max（仅数值列）
    pub fn par_max_all(&self) -> Vec<Option<f64>> {
        self.data
            .par_iter()
            .map(|s| match s.dtype() {
                DType::Float64 => s.max_f64(),
                DType::Int64 => s.max_i64().map(|v| v as f64),
                _ => None,
            })
            .collect()
    }

    /// 所有列并行 count（非空值数量）
    pub fn par_count_all(&self) -> Vec<usize> {
        self.data.par_iter().map(|s| s.count()).collect()
    }

    /// 所有列并行 quantile（仅数值列）
    pub fn par_quantile_all(&self, q: f64) -> Vec<Option<f64>> {
        self.data.par_iter().map(|s| s.quantile(q)).collect()
    }

    /// 所有列并行 any（跳过 None / NaN）
    pub fn par_any_all(&self) -> Vec<Option<bool>> {
        self.data.par_iter().map(|s| s.any()).collect()
    }

    /// 所有列并行 all（跳过 None / NaN）
    pub fn par_all_all(&self) -> Vec<Option<bool>> {
        self.data.par_iter().map(|s| s.all()).collect()
    }

    /// 所有列并行 nunique（统计唯一值数量，自动跳过 None）
    pub fn par_nunique_all(&self) -> Vec<usize> {
        self.data.par_iter().map(|s| s.nunique()).collect()
    }

    /// 所有列并行 isnull：返回全新的 bool DataFrame（列名不变）
    pub fn par_isnull_all(&self) -> DataFrame {
        let n_data: Vec<Series> = self
            .data
            .par_iter()
            .map(|s| {
                let mask: Vec<Option<bool>> = s.isnull().into_iter().map(Some).collect();
                Series::new_bool(s.name.clone(), mask)
            })
            .collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

    /// 所有列并行 notnull：返回全新的 bool DataFrame（列名不变）
    pub fn par_notnull_all(&self) -> DataFrame {
        let n_data: Vec<Series> = self
            .data
            .par_iter()
            .map(|s| {
                let mask: Vec<Option<bool>> = s.notnull().into_iter().map(Some).collect();
                Series::new_bool(s.name.clone(), mask)
            })
            .collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }
}
