//! DataFrame 缺失值方法：行级缺失检测、全列填充、前后向填充。
//!
//! - isnull_rows / notnull_rows：返回每行是否有缺失值
//! - dropna：删除包含缺失值的行
//! - fillna_all_f64 / fillna_all_i64 / fillna_all_string：全列填充
//! - par_ffill_all / par_bfill_all：全列前后向填充

use rayon::prelude::*;

use crate::core::dataframe::DataFrame;
use crate::core::dtype::ColumnData;
use crate::core::series::Series;

impl DataFrame {
    /// 返回每行是否有缺失值
    pub fn isnull_rows(&self) -> Vec<bool> {
        let nrows = self.nrows();
        (0..nrows)
            .into_par_iter()
            .map(|i| {
                self.data.iter().any(|s| match &s.data {
                    ColumnData::Int(v) => v[i].is_none(),
                    ColumnData::Float(v) => v[i].is_none(),
                    ColumnData::Bool(v) => v[i].is_none(),
                    ColumnData::String(v) => v[i].is_none(),
                    ColumnData::Categorical(c) => c.codes[i].is_none(),
                })
            })
            .collect()
    }

    /// 返回每行是否全部非缺失
    pub fn notnull_rows(&self) -> Vec<bool> {
        let nrows = self.nrows();
        (0..nrows)
            .into_par_iter()
            .map(|i| {
                self.data.iter().all(|s| match &s.data {
                    ColumnData::Int(v) => v[i].is_some(),
                    ColumnData::Float(v) => v[i].is_some(),
                    ColumnData::Bool(v) => v[i].is_some(),
                    ColumnData::String(v) => v[i].is_some(),
                    ColumnData::Categorical(c) => c.codes[i].is_some(),
                })
            })
            .collect()
    }

    /// 删除包含缺失值的行
    pub fn dropna(&self) -> DataFrame {
        self.dropna_rows()
    }

    /// 填充所有列的缺失值 (f64，仅对 Float64 列生效)
    pub fn fillna_all_f64(&self, v: f64) -> DataFrame {
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.fillna_f64(v)).collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

    /// 填充所有列的缺失值 (i64，仅对 Int64 列生效)
    pub fn fillna_all_i64(&self, v: i64) -> DataFrame {
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.fillna_i64(v)).collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

    /// 填充所有列的缺失值 (string，仅对 Object 列生效)
    pub fn fillna_all_string(&self, v: &str) -> DataFrame {
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.fillna_string(v)).collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

    /// 所有列并行前向填充 (ffill)
    pub fn par_ffill_all(&self) -> DataFrame {
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.ffill()).collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

    /// 所有列并行后向填充 (bfill)
    pub fn par_bfill_all(&self) -> DataFrame {
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.bfill()).collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }
}
