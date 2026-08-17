//! DataFrame: 列存储的多列数据结构 + PyO3 绑定

use pyo3::IntoPyObject;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use std::collections::HashMap;

use super::dtype::{ColumnData, DType};
use super::series::{PySeries, Series};

#[derive(Debug, Clone)]
pub struct DataFrame {
    pub columns: Vec<String>,
    pub data: Vec<Series>,
}

impl DataFrame {
    pub fn new_empty() -> Self {
        Self {
            columns: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn from_series(columns: Vec<String>, data: Vec<Series>) -> PyResult<Self> {
        if columns.len() != data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "columns len {} != data len {}",
                columns.len(),
                data.len()
            )));
        }
        // 校验列名去重
        let mut seen = std::collections::HashSet::new();
        for c in &columns {
            if !seen.insert(c.clone()) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "duplicate column name: {}",
                    c
                )));
            }
        }
        // 校验每列长度一致
        if let Some(first) = data.first() {
            let n = first.len();
            for (i, s) in data.iter().enumerate() {
                if s.len() != n {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "column '{}' length {} != row length {}",
                        columns[i],
                        s.len(),
                        n
                    )));
                }
            }
        }
        Ok(Self { columns, data })
    }

    pub fn nrows(&self) -> usize {
        self.data.first().map(|s| s.len()).unwrap_or(0)
    }
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
    pub fn column_names(&self) -> &[String] {
        &self.columns
    }

    pub fn dtypes(&self) -> Vec<(&str, &'static str)> {
        self.data
            .iter()
            .zip(self.columns.iter())
            .map(|(s, c)| (c.as_str(), s.dtype_name()))
            .collect()
    }

    pub fn get_column(&self, name: &str) -> Option<&Series> {
        self.columns
            .iter()
            .position(|c| c == name)
            .and_then(|i| self.data.get(i))
    }

    pub fn get_column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }

    pub fn get_column_at(&self, idx: usize) -> Option<&Series> {
        self.data.get(idx)
    }

    pub fn head(&self, n: usize) -> DataFrame {
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.head(n)).collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

    pub fn tail(&self, n: usize) -> DataFrame {
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.tail(n)).collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

    pub fn filter_rows(&self, mask: &[bool]) -> PyResult<DataFrame> {
        if mask.len() != self.nrows() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "mask length {} != nrows {}",
                mask.len(),
                self.nrows()
            )));
        }
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.filter(mask)).collect();
        Ok(DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        })
    }

    /// 删除任意一列为 None 的行 (axis=0)
    pub fn dropna_rows(&self) -> DataFrame {
        if self.nrows() == 0 {
            return self.clone();
        }
        let nrows = self.nrows();
        // 并行计算每列的非空 mask，然后合并 (任意列 None 则整行删除)
        let keep: Vec<bool> = (0..nrows)
            .into_par_iter()
            .map(|i| {
                self.data.iter().all(|s| match &s.data {
                    super::dtype::ColumnData::Int(v) => v[i].is_some(),
                    super::dtype::ColumnData::Float(v) => v[i].is_some(),
                    super::dtype::ColumnData::Bool(v) => v[i].is_some(),
                    super::dtype::ColumnData::String(v) => v[i].is_some(),
                    super::dtype::ColumnData::Categorical(c) => c.codes[i].is_some(),
                })
            })
            .collect();
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.filter(&keep)).collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

    /// 填充整个 DataFrame 中所有列的 None 值
    pub fn fillna_rows(
        &self,
        fill_dict: &std::collections::HashMap<String, FillValue>,
    ) -> PyResult<DataFrame> {
        let n_data: Vec<Series> = self
            .columns
            .par_iter()
            .zip(self.data.par_iter())
            .map(|(col, series)| {
                if let Some(v) = fill_dict.get(col) {
                    match (v, series.dtype()) {
                        (FillValue::Int(x), DType::Int64) => series.fillna_i64(*x),
                        (FillValue::Float(x), DType::Float64) => series.fillna_f64(*x),
                        (FillValue::Bool(x), DType::Bool) => series.fillna_bool(*x),
                        (FillValue::String(x), DType::Object) => series.fillna_string(x),
                        _ => series.clone(),
                    }
                } else {
                    series.clone()
                }
            })
            .collect();
        Ok(DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        })
    }

    /// 按指定列的值排序
    /// by 是列索引列表，按这些列的值排序行
    /// 使用第一列排序即可，多列排序取第一个
    pub fn sort_values(&self, by: &[usize], ascending: bool) -> DataFrame {
        // 空数据或空 by 直接返回克隆
        if by.is_empty() || self.nrows() == 0 || self.data.is_empty() {
            return self.clone();
        }
        let col_idx = by[0];
        let Some(sort_col) = self.data.get(col_idx) else {
            return self.clone();
        };
        let nrows = self.nrows();

        // 生成排序索引 (permutation): perm[i] 是排序后第 i 位对应的原行号
        let mut perm: Vec<usize> = (0..nrows).collect();

        // 根据列类型进行排序
        // ascending=true: 升序，None 放最后
        // ascending=false: 降序，None 放最前
        match &sort_col.data {
            ColumnData::Int(v) => {
                perm.sort_by(|&a, &b| {
                    cmp_opt_ord(
                        v.get(a).and_then(|x| x.as_ref()),
                        v.get(b).and_then(|x| x.as_ref()),
                        ascending,
                    )
                });
            }
            ColumnData::Float(v) => {
                perm.sort_by(|&a, &b| {
                    cmp_opt_f64(
                        v.get(a).and_then(|x| *x),
                        v.get(b).and_then(|x| *x),
                        ascending,
                    )
                });
            }
            ColumnData::Bool(v) => {
                perm.sort_by(|&a, &b| {
                    cmp_opt_ord(
                        v.get(a).and_then(|x| x.as_ref()),
                        v.get(b).and_then(|x| x.as_ref()),
                        ascending,
                    )
                });
            }
            ColumnData::String(v) => {
                perm.sort_by(|&a, &b| {
                    cmp_opt_str(
                        v.get(a).and_then(|x| x.as_deref()),
                        v.get(b).and_then(|x| x.as_deref()),
                        ascending,
                    )
                });
            }
            ColumnData::Categorical(c) => {
                // 对 categorical 按其字符串值排序
                perm.sort_by(|&a, &b| {
                    let sa = c.codes.get(a).and_then(|code| {
                        code.as_ref()
                            .and_then(|&idx| c.categories.get(idx as usize))
                            .map(|s| s.as_str())
                    });
                    let sb = c.codes.get(b).and_then(|code| {
                        code.as_ref()
                            .and_then(|&idx| c.categories.get(idx as usize))
                            .map(|s| s.as_str())
                    });
                    cmp_opt_str(sa, sb, ascending)
                });
            }
        }

        // 应用 permutation 到所有列 (并行)
        let n_data: Vec<Series> = self
            .data
            .par_iter()
            .map(|s| gather_series(s, &perm))
            .collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

    /// 按索引排序
    /// ascending=true: 保持原顺序
    /// ascending=false: 反转行顺序
    pub fn sort_index(&self, ascending: bool) -> DataFrame {
        if ascending || self.nrows() == 0 {
            return self.clone();
        }
        // 反转: permutation = [n-1, n-2, ..., 0]
        let nrows = self.nrows();
        let perm: Vec<usize> = (0..nrows).rev().collect();
        let n_data: Vec<Series> = self
            .data
            .par_iter()
            .map(|s| gather_series(s, &perm))
            .collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

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

    // ---------- 列并行批量方法（Python 层变薄：一次性 R 调用） ----------

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

    /// 基于键列的哈希连接
    /// left_on/right_on: 左右表的键列索引
    /// how: "inner"=内连接, "left"=左连接, "right"=右连接, "outer"=外连接
    pub fn merge(
        &self,
        right: &DataFrame,
        left_on: usize,
        right_on: usize,
        how: &str,
    ) -> DataFrame {
        let n_left = self.data.first().map(|s| s.len()).unwrap_or(0);
        let n_right = right.data.first().map(|s| s.len()).unwrap_or(0);

        // 构建 right 表的键到行索引的映射
        let mut right_map: HashMap<String, Vec<usize>> = HashMap::new();
        if let Some(right_key_col) = right.data.get(right_on) {
            for i in 0..n_right {
                let key = right_key_col.get_str_at(i);
                right_map.entry(key).or_default().push(i);
            }
        }

        // 左表键列
        let left_key_col = self.data.get(left_on);

        // 收集匹配的行对 (左行索引, 右行索引)
        // 左表无匹配时右索引为 None；右表无匹配时左索引为 usize::MAX
        let mut matched: Vec<(usize, Option<usize>)> = Vec::new();
        let mut right_matched: std::collections::HashSet<usize> = std::collections::HashSet::new();

        if let Some(lkc) = left_key_col {
            for i in 0..n_left {
                let key = lkc.get_str_at(i);
                if let Some(right_indices) = right_map.get(&key) {
                    for &j in right_indices {
                        matched.push((i, Some(j)));
                        right_matched.insert(j);
                    }
                } else if how == "left" || how == "outer" {
                    matched.push((i, None));
                }
            }
        }

        // 右连接/外连接：补充未匹配的右表行
        if how == "right" || how == "outer" {
            for j in 0..n_right {
                if !right_matched.contains(&j) {
                    // usize::MAX 表示左表无匹配
                    matched.push((usize::MAX, Some(j)));
                }
            }
        }

        // 构建结果列
        let mut result_columns: Vec<String> = Vec::new();
        let mut result_data: Vec<Series> = Vec::new();

        // 左表所有列（保留全部列，pandas 风格）
        for (col_idx, col_name) in self.columns.iter().enumerate() {
            result_columns.push(col_name.clone());
            let series = &self.data[col_idx];
            let mut values: Vec<Option<String>> = Vec::with_capacity(matched.len());
            for &(li, _) in &matched {
                if li != usize::MAX && li < n_left {
                    let val = series.get_str_at(li);
                    values.push(if val.is_empty() { None } else { Some(val) });
                } else {
                    values.push(None);
                }
            }
            // 统一转为 String 列（保持原列名）
            result_data.push(Series::from_options_string(col_name.clone(), &values));
        }

        // 右表列（跳过键列以避免重复）
        for (col_idx, col_name) in right.columns.iter().enumerate() {
            if col_idx == right_on {
                continue;
            }
            result_columns.push(col_name.clone());
            let series = &right.data[col_idx];
            let mut values: Vec<Option<String>> = Vec::with_capacity(matched.len());
            for &(_, rj) in &matched {
                if let Some(j) = rj {
                    if j < n_right {
                        let val = series.get_str_at(j);
                        values.push(if val.is_empty() { None } else { Some(val) });
                    } else {
                        values.push(None);
                    }
                } else {
                    values.push(None);
                }
            }
            result_data.push(Series::from_options_string(col_name.clone(), &values));
        }

        DataFrame {
            columns: result_columns,
            data: result_data,
        }
    }

    /// 按 by 列分组并对每列执行聚合函数
    /// by: 分组键列索引
    /// agg: 聚合函数名 ("sum"/"mean"/"count"/"min"/"max")
    /// 返回 (group_keys, aggregated_data) — 每组一行结果
    pub fn groupby_agg(&self, by: usize, agg: &str) -> (Vec<String>, DataFrame) {
        let n = self.data.first().map(|s| s.len()).unwrap_or(0);

        // 按键列分组，记录每组的行索引；group_order 保留首次出现顺序
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        let mut group_order: Vec<String> = Vec::new();

        if let Some(key_col) = self.data.get(by) {
            for i in 0..n {
                let key = key_col.get_str_at(i);
                if !groups.contains_key(&key) {
                    group_order.push(key.clone());
                }
                groups.entry(key).or_default().push(i);
            }
        }

        // 构建结果
        let mut result_columns = vec![self.columns[by].clone()];
        let mut result_data: Vec<Series> = Vec::new();

        // 键列
        let key_values: Vec<Option<String>> = group_order.iter().map(|s| Some(s.clone())).collect();
        result_data.push(Series::from_options_string(
            self.columns[by].clone(),
            &key_values,
        ));

        // 对每列（除键列外）执行聚合
        for (col_idx, col_name) in self.columns.iter().enumerate() {
            if col_idx == by {
                continue;
            }
            let series = &self.data[col_idx];
            let mut agg_values: Vec<Option<f64>> = Vec::with_capacity(group_order.len());

            for key in &group_order {
                let indices = &groups[key];
                match agg {
                    "count" => {
                        let count = indices
                            .iter()
                            .filter(|&&i| {
                                let v = series.get_str_at(i);
                                !v.is_empty() && v != "NaN"
                            })
                            .count();
                        agg_values.push(Some(count as f64));
                    }
                    "sum" => {
                        let sum: f64 = indices
                            .iter()
                            .filter_map(|&i| series.get_str_at(i).parse::<f64>().ok())
                            .sum();
                        agg_values.push(Some(sum));
                    }
                    "mean" => {
                        let values: Vec<f64> = indices
                            .iter()
                            .filter_map(|&i| series.get_str_at(i).parse::<f64>().ok())
                            .collect();
                        if values.is_empty() {
                            agg_values.push(None);
                        } else {
                            agg_values.push(Some(values.iter().sum::<f64>() / values.len() as f64));
                        }
                    }
                    "min" => {
                        let min_val = indices
                            .iter()
                            .filter_map(|&i| series.get_str_at(i).parse::<f64>().ok())
                            .fold(f64::INFINITY, f64::min);
                        if min_val.is_infinite() {
                            agg_values.push(None);
                        } else {
                            agg_values.push(Some(min_val));
                        }
                    }
                    "max" => {
                        let max_val = indices
                            .iter()
                            .filter_map(|&i| series.get_str_at(i).parse::<f64>().ok())
                            .fold(f64::NEG_INFINITY, f64::max);
                        if max_val.is_infinite() {
                            agg_values.push(None);
                        } else {
                            agg_values.push(Some(max_val));
                        }
                    }
                    _ => agg_values.push(None),
                }
            }

            // 数值聚合结果统一使用 Float 类型
            result_columns.push(col_name.clone());
            result_data.push(Series::new_float(Some(col_name.clone()), agg_values));
        }

        (
            group_order,
            DataFrame {
                columns: result_columns,
                data: result_data,
            },
        )
    }

    /// 透视表：按 index_col 分组，columns_col 的值作为新列名，values_col 聚合
    pub fn pivot(
        &self,
        index_col: usize,
        columns_col: usize,
        values_col: usize,
        agg_func: &str,
    ) -> DataFrame {
        let n = self.data.first().map(|s| s.len()).unwrap_or(0);

        // 收集所有 index 值和 column 值（保持出现顺序、去重）
        let mut index_values: Vec<String> = Vec::new();
        let mut index_seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut column_values: Vec<String> = Vec::new();
        let mut column_seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        let index_series = &self.data[index_col];
        let columns_series = &self.data[columns_col];
        let values_series = &self.data[values_col];

        for i in 0..n {
            let idx = index_series.get_str_at(i);
            if !index_seen.contains(&idx) {
                index_seen.insert(idx.clone());
                index_values.push(idx);
            }
            let col = columns_series.get_str_at(i);
            if !column_seen.contains(&col) {
                column_seen.insert(col.clone());
                column_values.push(col);
            }
        }

        // 构建 (index, column) -> Vec<f64> 映射
        let mut pivot_map: HashMap<(String, String), Vec<f64>> = HashMap::new();
        for i in 0..n {
            let idx = index_series.get_str_at(i);
            let col = columns_series.get_str_at(i);
            if let Ok(v) = values_series.get_str_at(i).parse::<f64>() {
                pivot_map.entry((idx, col)).or_default().push(v);
            }
        }

        // 构建结果
        let mut result_columns = vec![self.columns[index_col].clone()];
        for col in &column_values {
            result_columns.push(col.clone());
        }

        let mut result_data: Vec<Series> = Vec::new();
        // 索引列
        result_data.push(Series::from_options_string(
            self.columns[index_col].clone(),
            &index_values
                .iter()
                .map(|s| Some(s.clone()))
                .collect::<Vec<_>>(),
        ));

        // 每个 column 值一列
        for col in &column_values {
            let mut values: Vec<Option<f64>> = Vec::with_capacity(index_values.len());
            for idx in &index_values {
                if let Some(vals) = pivot_map.get(&(idx.clone(), col.clone())) {
                    match agg_func {
                        "mean" => {
                            if vals.is_empty() {
                                values.push(None);
                            } else {
                                values.push(Some(vals.iter().sum::<f64>() / vals.len() as f64));
                            }
                        }
                        "count" => values.push(Some(vals.len() as f64)),
                        "min" => {
                            values.push(Some(vals.iter().copied().fold(f64::INFINITY, f64::min)))
                        }
                        "max" => values
                            .push(Some(vals.iter().copied().fold(f64::NEG_INFINITY, f64::max))),
                        // sum 及其它未知函数默认按 sum 聚合
                        _ => values.push(Some(vals.iter().sum())),
                    }
                } else {
                    values.push(None);
                }
            }
            result_data.push(Series::new_float(Some(col.clone()), values));
        }

        DataFrame {
            columns: result_columns,
            data: result_data,
        }
    }

    /// 宽转长：将指定的值列转为 (variable, value) 两列
    pub fn melt(&self, id_cols: &[usize], value_cols: &[usize]) -> DataFrame {
        let n = self.data.first().map(|s| s.len()).unwrap_or(0);

        // 结果列 = id_cols + ["variable", "value"]
        let mut result_columns: Vec<String> = Vec::new();
        for &i in id_cols {
            result_columns.push(self.columns[i].clone());
        }
        result_columns.push("variable".to_string());
        result_columns.push("value".to_string());

        // 每行 x 每个值列 -> 展开为多行
        let n_value_cols = value_cols.len();
        let n_result_rows = n * n_value_cols;

        let mut result_data: Vec<Series> = Vec::new();

        // id 列
        for &id_idx in id_cols {
            let series = &self.data[id_idx];
            let mut values: Vec<Option<String>> = Vec::with_capacity(n_result_rows);
            for i in 0..n {
                for _ in 0..n_value_cols {
                    let v = series.get_str_at(i);
                    values.push(if v.is_empty() { None } else { Some(v) });
                }
            }
            result_data.push(Series::from_options_string(
                self.columns[id_idx].clone(),
                &values,
            ));
        }

        // variable 列
        let mut var_values: Vec<Option<String>> = Vec::with_capacity(n_result_rows);
        for _ in 0..n {
            for &vc in value_cols {
                var_values.push(Some(self.columns[vc].clone()));
            }
        }
        result_data.push(Series::from_options_string(
            "variable".to_string(),
            &var_values,
        ));

        // value 列
        let mut val_values: Vec<Option<String>> = Vec::with_capacity(n_result_rows);
        for i in 0..n {
            for &vc in value_cols {
                let v = self.data[vc].get_str_at(i);
                val_values.push(if v.is_empty() { None } else { Some(v) });
            }
        }
        result_data.push(Series::from_options_string(
            "value".to_string(),
            &val_values,
        ));

        DataFrame {
            columns: result_columns,
            data: result_data,
        }
    }

    // ---------- stack / unstack ----------

    /// 将列堆叠为行：返回 DataFrame 包含 index/variable/value 三列
    /// level 参数仅用于 API 兼容，当前简化版忽略（仅单层列）
    pub fn stack(&self, level: i64) -> DataFrame {
        let n = self.nrows();
        let n_cols = self.data.len();
        let total_rows = n * n_cols;
        let mut idx_values: Vec<Option<i64>> = Vec::with_capacity(total_rows);
        let mut var_values: Vec<Option<String>> = Vec::with_capacity(total_rows);
        let mut val_values: Vec<Option<String>> = Vec::with_capacity(total_rows);
        for i in 0..n {
            for (j, s) in self.data.iter().enumerate() {
                idx_values.push(Some(i as i64));
                var_values.push(Some(self.columns[j].clone()));
                let v = s.get_str_at(i);
                val_values.push(if v.is_empty() { None } else { Some(v) });
            }
        }
        let _ = level; // 单层简化版未使用
        let idx_series = Series::new_int(Some("index".to_string()), idx_values);
        let var_series = Series::new_string(Some("variable".to_string()), var_values);
        let val_series = Series::from_options_string("value".to_string(), &val_values);
        DataFrame {
            columns: vec![
                "index".to_string(),
                "variable".to_string(),
                "value".to_string(),
            ],
            data: vec![idx_series, var_series, val_series],
        }
    }

    /// unstack：将包含 variable/value 列的 DataFrame 透视为宽表
    pub fn unstack(&self, index_col: usize, var_col: usize, value_col: usize) -> DataFrame {
        let n = self.nrows();
        // 收集所有 variable 值（保序去重）
        let mut var_order: Vec<String> = Vec::new();
        let mut var_seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut index_seen: Vec<String> = Vec::new();
        let mut index_map: HashMap<String, usize> = HashMap::new();
        for i in 0..n {
            let var = self.data[var_col].get_str_at(i);
            if !var.is_empty() && var_seen.insert(var.clone()) {
                var_order.push(var);
            }
            let idx = self.data[index_col].get_str_at(i);
            if !index_map.contains_key(&idx) {
                index_map.insert(idx.clone(), index_seen.len());
                index_seen.push(idx);
            }
        }
        // 初始化结果列
        let mut result_data: Vec<Vec<Option<String>>> = (0..var_order.len())
            .map(|_| vec![None; index_seen.len()])
            .collect();
        // 填充
        for i in 0..n {
            let idx_str = self.data[index_col].get_str_at(i);
            let var_str = self.data[var_col].get_str_at(i);
            let val_str = self.data[value_col].get_str_at(i);
            if let Some(&row_idx) = index_map.get(&idx_str)
                && let Some(col_idx) = var_order.iter().position(|v| v == &var_str)
            {
                result_data[col_idx][row_idx] = if val_str.is_empty() {
                    None
                } else {
                    Some(val_str)
                };
            }
        }
        let mut result_columns = vec!["index".to_string()];
        result_columns.extend(var_order.clone());
        let mut result_series: Vec<Series> = Vec::new();
        // index 列
        result_series.push(Series::from_options_string(
            "index".to_string(),
            &index_seen
                .iter()
                .map(|s| Some(s.clone()))
                .collect::<Vec<_>>(),
        ));
        for (j, col) in var_order.iter().enumerate() {
            result_series.push(Series::from_options_string(col.clone(), &result_data[j]));
        }
        DataFrame {
            columns: result_columns,
            data: result_series,
        }
    }

    // ---------- 简单查询（query 简化版） ----------

    /// 按列比较标量过滤行
    /// col_idx: 列索引
    /// op: ">" / "<" / ">=" / "<=" / "==" / "!="
    /// value: 比较值
    pub fn query_filter(&self, col_idx: usize, op: &str, value: f64) -> DataFrame {
        if col_idx >= self.data.len() {
            return DataFrame::new_empty();
        }
        let mask = self.data[col_idx].compare_scalar(op, value);
        self.filter_rows(&mask)
            .unwrap_or_else(|_| DataFrame::new_empty())
    }
}

/// 通用 Option 比较器 (用于 Ord 类型)
/// ascending=true: 升序，None 放最后
/// ascending=false: 降序，None 放最前
fn cmp_opt_ord<T: Ord + ?Sized>(
    a: Option<&T>,
    b: Option<&T>,
    ascending: bool,
) -> std::cmp::Ordering {
    match (a, b) {
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
    }
}

/// f64 专用比较器 (f64 不实现 Ord，使用 partial_cmp)
fn cmp_opt_f64(a: Option<f64>, b: Option<f64>, ascending: bool) -> std::cmp::Ordering {
    match (a, b) {
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
    }
}

/// &str 比较器
fn cmp_opt_str(a: Option<&str>, b: Option<&str>, ascending: bool) -> std::cmp::Ordering {
    cmp_opt_ord(a, b, ascending)
}

/// 按索引列表收集 Series 中的元素，返回新的 Series
/// 使用 Vec::with_capacity 预分配内存
fn gather_series(s: &Series, indices: &[usize]) -> Series {
    let new_data = match &s.data {
        ColumnData::Int(v) => {
            let mut out: Vec<Option<i64>> = Vec::with_capacity(indices.len());
            for &i in indices {
                out.push(v.get(i).and_then(|x| *x));
            }
            ColumnData::Int(out)
        }
        ColumnData::Float(v) => {
            let mut out: Vec<Option<f64>> = Vec::with_capacity(indices.len());
            for &i in indices {
                out.push(v.get(i).and_then(|x| *x));
            }
            ColumnData::Float(out)
        }
        ColumnData::Bool(v) => {
            let mut out: Vec<Option<bool>> = Vec::with_capacity(indices.len());
            for &i in indices {
                out.push(v.get(i).and_then(|x| *x));
            }
            ColumnData::Bool(out)
        }
        ColumnData::String(v) => {
            let mut out: Vec<Option<String>> = Vec::with_capacity(indices.len());
            for &i in indices {
                out.push(v.get(i).and_then(|x| x.clone()));
            }
            ColumnData::String(out)
        }
        ColumnData::Categorical(c) => {
            let mut out: Vec<Option<i32>> = Vec::with_capacity(indices.len());
            for &i in indices {
                out.push(c.codes.get(i).and_then(|x| *x));
            }
            ColumnData::Categorical(super::dtype::CategoricalData {
                categories: c.categories.clone(),
                codes: out,
                ordered: c.ordered,
            })
        }
    };
    Series {
        name: s.name.clone(),
        data: new_data,
    }
}

/// DataFrame fillna 用的填充值类型
#[derive(Debug, Clone)]
pub enum FillValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

// =====================================================================
// PyO3 绑定
// =====================================================================

#[pyclass(name = "_DataFrame", module = "rspandas", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyDataFrame {
    pub inner: DataFrame,
}

#[pymethods]
impl PyDataFrame {
    /// 构造: 接受 columns (list[str]) 和 series (list[_Series])
    #[new]
    fn new(columns: Vec<String>, series: Vec<PySeries>) -> PyResult<Self> {
        let data: Vec<Series> = series.into_iter().map(|s| s.inner).collect();
        let inner = DataFrame::from_series(columns, data)?;
        Ok(PyDataFrame { inner })
    }

    // ---------- 属性 ----------

    #[getter]
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }
    #[getter]
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }
    #[getter]
    fn size(&self) -> usize {
        self.inner.nrows() * self.inner.ncols()
    }
    #[getter]
    fn empty(&self) -> bool {
        self.inner.nrows() == 0
    }

    #[getter]
    fn columns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.inner.columns.iter().map(|s| s.as_str()))
    }

    #[getter]
    fn dtypes<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let d = PyDict::new(py);
        for (name, dt) in self.inner.dtypes() {
            d.set_item(name, dt).unwrap();
        }
        d
    }

    // ---------- 子集 ----------

    /// 按列名取列 -> _Series
    fn get_column(&self, name: &str) -> PyResult<PySeries> {
        match self.inner.get_column(name) {
            Some(s) => Ok(PySeries { inner: s.clone() }),
            None => Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "column not found: {}",
                name
            ))),
        }
    }

    /// 按索引取列 -> _Series
    fn get_column_at(&self, idx: usize) -> PyResult<PySeries> {
        match self.inner.get_column_at(idx) {
            Some(s) => Ok(PySeries { inner: s.clone() }),
            None => Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "column index out of range: {}",
                idx
            ))),
        }
    }

    /// 列名 -> 索引
    fn column_index(&self, name: &str) -> Option<usize> {
        self.inner.get_column_index(name)
    }

    fn head(&self, py: Python<'_>, n: usize) -> Self {
        let inner = py.detach(|| self.inner.head(n));
        PyDataFrame { inner }
    }
    fn tail(&self, py: Python<'_>, n: usize) -> Self {
        let inner = py.detach(|| self.inner.tail(n));
        PyDataFrame { inner }
    }
    fn filter_rows(&self, py: Python<'_>, mask: Vec<bool>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.filter_rows(&mask))?;
        Ok(PyDataFrame { inner })
    }

    // ---------- 缺失值 ----------

    fn dropna(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.dropna());
        PyDataFrame { inner }
    }

    /// fillna: 接受 dict {col_name: value}，只填充指定列
    fn fillna(&self, py: Python<'_>, fill_dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut converted = std::collections::HashMap::new();
        for (key, val) in fill_dict.iter() {
            let col: String = key.extract()?;
            // 优先尝试 bool，再 int，再 float，最后 string
            if let Ok(b) = val.extract::<bool>() {
                converted.insert(col, FillValue::Bool(b));
            } else if let Ok(i) = val.extract::<i64>() {
                converted.insert(col, FillValue::Int(i));
            } else if let Ok(f) = val.extract::<f64>() {
                converted.insert(col, FillValue::Float(f));
            } else if let Ok(s) = val.extract::<String>() {
                converted.insert(col, FillValue::String(s));
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "unsupported fill value type for column '{}'",
                    col
                )));
            }
        }
        let inner = py.detach(|| self.inner.fillna_rows(&converted))?;
        Ok(PyDataFrame { inner })
    }

    // ---------- 排序 ----------

    /// 按指定列的值排序 (by 为列索引列表，取第一个列排序)
    fn sort_values(&self, py: Python<'_>, by: Vec<usize>, ascending: bool) -> Self {
        let inner = py.detach(|| self.inner.sort_values(&by, ascending));
        PyDataFrame { inner }
    }

    /// 按索引排序 (ascending=true 保持原顺序，false 反转)
    fn sort_index(&self, py: Python<'_>, ascending: bool) -> Self {
        let inner = py.detach(|| self.inner.sort_index(ascending));
        PyDataFrame { inner }
    }

    // ---------- 行级缺失值检测 ----------

    /// 返回每行是否有缺失值
    fn isnull_rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mask = py.detach(|| self.inner.isnull_rows());
        PyList::new(py, mask.iter().copied())
    }

    /// 返回每行是否全部非缺失
    fn notnull_rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mask = py.detach(|| self.inner.notnull_rows());
        PyList::new(py, mask.iter().copied())
    }

    // ---------- 全列填充 ----------

    /// 填充所有列的缺失值 (f64，仅对 Float64 列生效)
    fn fillna_all_f64(&self, py: Python<'_>, v: f64) -> Self {
        let inner = py.detach(|| self.inner.fillna_all_f64(v));
        PyDataFrame { inner }
    }

    /// 填充所有列的缺失值 (i64，仅对 Int64 列生效)
    fn fillna_all_i64(&self, py: Python<'_>, v: i64) -> Self {
        let inner = py.detach(|| self.inner.fillna_all_i64(v));
        PyDataFrame { inner }
    }

    /// 填充所有列的缺失值 (string，仅对 Object 列生效)
    fn fillna_all_string(&self, py: Python<'_>, v: &str) -> Self {
        let v_owned = v.to_string();
        let inner = py.detach(|| self.inner.fillna_all_string(&v_owned));
        PyDataFrame { inner }
    }

    // ---------- 列并行批量方法（Python for 循环 → Rust rayon 并行） ----------

    /// 所有列并行 ffill（一次调用替代 Python 逐列 for 循环）
    fn par_ffill_all(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.par_ffill_all());
        PyDataFrame { inner }
    }

    /// 所有列并行 bfill
    fn par_bfill_all(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.par_bfill_all());
        PyDataFrame { inner }
    }

    /// 所有列并行 sum → list[float|None]（按列顺序）
    fn par_sum_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let result = py.detach(|| self.inner.par_sum_all());
        let list = PyList::empty(py);
        for r in result {
            match r {
                Some(v) => list.append(v).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list
    }

    /// 所有列并行 mean
    fn par_mean_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let result = py.detach(|| self.inner.par_mean_all());
        let list = PyList::empty(py);
        for r in result {
            match r {
                Some(v) => list.append(v).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list
    }

    /// 所有列并行 std（ddof 自由度）
    fn par_std_all<'py>(&self, py: Python<'py>, ddof: usize) -> Bound<'py, PyList> {
        let result = py.detach(move || self.inner.par_std_all(ddof));
        let list = PyList::empty(py);
        for r in result {
            match r {
                Some(v) => list.append(v).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list
    }

    /// 所有列并行 min
    fn par_min_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let result = py.detach(|| self.inner.par_min_all());
        let list = PyList::empty(py);
        for r in result {
            match r {
                Some(v) => list.append(v).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list
    }

    /// 所有列并行 max
    fn par_max_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let result = py.detach(|| self.inner.par_max_all());
        let list = PyList::empty(py);
        for r in result {
            match r {
                Some(v) => list.append(v).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list
    }

    /// 所有列并行 count（非空值数）
    fn par_count_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let result = py.detach(|| self.inner.par_count_all());
        PyList::new(py, result).unwrap()
    }

    /// 所有列并行 quantile
    fn par_quantile_all<'py>(&self, py: Python<'py>, q: f64) -> Bound<'py, PyList> {
        let result = py.detach(move || self.inner.par_quantile_all(q));
        let list = PyList::empty(py);
        for r in result {
            match r {
                Some(v) => list.append(v).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list
    }

    /// 所有列并行批量聚合（返回每列的聚合结果二维列表，行=列，列=aggs顺序）
    fn par_batch_agg<'py>(&self, py: Python<'py>, aggs: Vec<String>) -> Bound<'py, PyList> {
        let result = py.detach(move || self.inner.par_batch_agg(&aggs));
        let outer = PyList::empty(py);
        for col_r in result {
            let inner_list = PyList::empty(py);
            for r in col_r {
                match r {
                    Some(v) => inner_list.append(v).unwrap(),
                    None => inner_list.append(py.None()).unwrap(),
                }
            }
            outer.append(inner_list).unwrap();
        }
        outer
    }

    /// 所有列并行 any → list[bool|None]（按列顺序）
    fn par_any_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let result = py.detach(|| self.inner.par_any_all());
        let list = PyList::empty(py);
        for r in result {
            match r {
                Some(v) => list.append(v).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list
    }

    /// 所有列并行 all → list[bool|None]
    fn par_all_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let result = py.detach(|| self.inner.par_all_all());
        let list = PyList::empty(py);
        for r in result {
            match r {
                Some(v) => list.append(v).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list
    }

    /// 所有列并行 nunique → list[int]（按列顺序）
    fn par_nunique_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let result = py.detach(|| self.inner.par_nunique_all());
        PyList::new(py, result).unwrap()
    }

    /// 所有列并行 isnull → 返回 bool DataFrame（每列变为 bool 类型）
    fn par_isnull_all(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.par_isnull_all());
        PyDataFrame { inner }
    }

    /// 所有列并行 notnull → 返回 bool DataFrame
    fn par_notnull_all(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.par_notnull_all());
        PyDataFrame { inner }
    }

    // ---------- 显示辅助 ----------

    /// 逐行构造 dict 列表 (用于 Python 端显示)
    fn to_rows<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let rows = PyList::empty(py);
        let nrows = self.inner.nrows();
        for i in 0..nrows {
            let row = PyDict::new(py);
            for (col_name, series) in self.inner.columns.iter().zip(self.inner.data.iter()) {
                let val: pyo3::Py<pyo3::PyAny> = match &series.data {
                    super::dtype::ColumnData::Int(v) => match v.get(i) {
                        Some(Some(n)) => (*n).into_pyobject(py).unwrap().into_any().unbind(),
                        _ => py.None(),
                    },
                    super::dtype::ColumnData::Float(v) => match v.get(i) {
                        Some(Some(n)) => (*n).into_pyobject(py).unwrap().into_any().unbind(),
                        _ => py.None(),
                    },
                    super::dtype::ColumnData::Bool(v) => match v.get(i) {
                        Some(Some(n)) => (*n).into_pyobject(py).unwrap().as_any().clone().unbind(),
                        _ => py.None(),
                    },
                    super::dtype::ColumnData::String(v) => match v.get(i) {
                        Some(Some(s)) => s.clone().into_pyobject(py).unwrap().into_any().unbind(),
                        _ => py.None(),
                    },
                    super::dtype::ColumnData::Categorical(c) => match c.codes.get(i) {
                        Some(Some(code_idx)) => {
                            let cat_str = c
                                .categories
                                .get(*code_idx as usize)
                                .cloned()
                                .unwrap_or_else(|| "NaN".to_string());
                            cat_str.into_pyobject(py).unwrap().into_any().unbind()
                        }
                        _ => py.None(),
                    },
                };
                row.set_item(col_name, val).unwrap();
            }
            rows.append(row).unwrap();
        }
        rows
    }

    /// 每列的 string 列表 (用于显示)
    fn columns_to_string<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let d = PyDict::new(py);
        // 释放 GIL 进行并行字符串转换
        let pairs: Vec<(String, Vec<String>)> = py.detach(|| {
            self.inner
                .columns
                .par_iter()
                .zip(self.inner.data.par_iter())
                .map(|(col_name, series)| (col_name.clone(), series.to_string_vec()))
                .collect()
        });
        for (col_name, svec) in pairs {
            let pylist: Bound<'_, PyList> =
                PyList::new(py, svec.iter().map(|s| s.as_str())).unwrap();
            d.set_item(col_name, pylist).unwrap();
        }
        d
    }

    /// 各列 dtype 的 dict
    fn dtypes_dict<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        self.dtypes(py)
    }

    // ---------- 合并 ----------

    /// 基于键列的哈希连接
    /// how: "inner"/"left"/"right"/"outer"
    fn merge(
        &self,
        py: Python<'_>,
        right: &PyDataFrame,
        left_on: usize,
        right_on: usize,
        how: &str,
    ) -> Self {
        let inner = py.detach(|| self.inner.merge(&right.inner, left_on, right_on, how));
        PyDataFrame { inner }
    }

    // ---------- 分组聚合 ----------

    /// 按 by 列分组并对每列执行聚合
    /// agg: "sum"/"mean"/"count"/"min"/"max"
    /// 返回 (group_keys, aggregated_df)
    fn groupby_agg(&self, py: Python<'_>, by: usize, agg: &str) -> (Vec<String>, PyDataFrame) {
        let (keys, df) = py.detach(|| self.inner.groupby_agg(by, agg));
        (keys, PyDataFrame { inner: df })
    }

    // ---------- 透视与逆透视 ----------

    /// 透视表：按 index_col 分组，columns_col 的值作为新列名，values_col 聚合
    /// agg_func: "sum"/"mean"/"count"/"min"/"max"
    fn pivot(
        &self,
        py: Python<'_>,
        index_col: usize,
        columns_col: usize,
        values_col: usize,
        agg_func: &str,
    ) -> Self {
        let inner = py.detach(|| {
            self.inner
                .pivot(index_col, columns_col, values_col, agg_func)
        });
        PyDataFrame { inner }
    }

    /// 宽转长：将指定的值列转为 (variable, value) 两列
    fn melt(&self, py: Python<'_>, id_cols: Vec<usize>, value_cols: Vec<usize>) -> Self {
        let inner = py.detach(|| self.inner.melt(&id_cols, &value_cols));
        PyDataFrame { inner }
    }

    // ---------- stack / unstack ----------

    /// 将列堆叠为行
    fn stack(&self, py: Python<'_>, level: i64) -> Self {
        let inner = py.detach(|| self.inner.stack(level));
        PyDataFrame { inner }
    }

    /// unstack：将 variable/value 列透视为宽表
    fn unstack(&self, py: Python<'_>, index_col: usize, var_col: usize, value_col: usize) -> Self {
        let inner = py.detach(|| self.inner.unstack(index_col, var_col, value_col));
        PyDataFrame { inner }
    }

    // ---------- 简单查询（query 简化版） ----------

    /// 按列比较标量过滤行
    fn query_filter(&self, py: Python<'_>, col_idx: usize, op: &str, value: f64) -> Self {
        let op_owned = op.to_string();
        let inner = py.detach(|| self.inner.query_filter(col_idx, &op_owned, value));
        PyDataFrame { inner }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataframe_basic() {
        let s1 = Series::new_int(Some("a".to_string()), vec![Some(1), Some(2), Some(3)]);
        let s2 = Series::new_string(
            Some("b".to_string()),
            vec![
                Some("x".to_string()),
                Some("y".to_string()),
                Some("z".to_string()),
            ],
        );
        let df =
            DataFrame::from_series(vec!["a".to_string(), "b".to_string()], vec![s1, s2]).unwrap();
        assert_eq!(df.shape(), (3, 2));
        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 2);
    }

    #[test]
    fn test_dataframe_head_tail() {
        let s1 = Series::new_int(None, vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        let df = DataFrame::from_series(vec!["a".to_string()], vec![s1]).unwrap();
        assert_eq!(df.head(2).nrows(), 2);
        assert_eq!(df.tail(2).nrows(), 2);
    }

    #[test]
    fn test_dataframe_filter() {
        let s1 = Series::new_int(None, vec![Some(1), Some(2), Some(3), Some(4)]);
        let df = DataFrame::from_series(vec!["a".to_string()], vec![s1]).unwrap();
        let filtered = df.filter_rows(&[true, false, true, false]).unwrap();
        assert_eq!(filtered.nrows(), 2);
    }

    #[test]
    fn test_dataframe_duplicate_col() {
        let s1 = Series::new_int(None, vec![Some(1)]);
        let s2 = Series::new_int(None, vec![Some(2)]);
        let r = DataFrame::from_series(vec!["a".to_string(), "a".to_string()], vec![s1, s2]);
        assert!(r.is_err());
    }

    #[test]
    fn test_dataframe_shape_mismatch() {
        let s1 = Series::new_int(None, vec![Some(1), Some(2)]);
        let s2 = Series::new_int(None, vec![Some(3)]);
        let r = DataFrame::from_series(vec!["a".to_string(), "b".to_string()], vec![s1, s2]);
        assert!(r.is_err());
    }

    #[test]
    fn test_dataframe_sort_values() {
        // 整型列排序：升序，None 放最后
        let s1 = Series::new_int(None, vec![Some(3), None, Some(1), Some(2)]);
        let s2 = Series::new_string(
            None,
            vec![
                Some("c".to_string()),
                Some("d".to_string()),
                Some("a".to_string()),
                Some("b".to_string()),
            ],
        );
        let df = DataFrame::from_series(vec!["num".to_string(), "str".to_string()], vec![s1, s2])
            .expect("DataFrame 构建失败");

        // 按第 0 列 (num) 升序排序
        let sorted_asc = df.sort_values(&[0], true);
        assert_eq!(sorted_asc.nrows(), 4);
        // 验证 num 列顺序: 1, 2, 3, None
        if let ColumnData::Int(v) = &sorted_asc.data[0].data {
            assert_eq!(v[0], Some(1));
            assert_eq!(v[1], Some(2));
            assert_eq!(v[2], Some(3));
            assert_eq!(v[3], None);
        } else {
            panic!("dtype 错误");
        }
        // 验证 str 列跟随重排: a, b, c, d
        if let ColumnData::String(v) = &sorted_asc.data[1].data {
            assert_eq!(v[0], Some("a".to_string()));
            assert_eq!(v[1], Some("b".to_string()));
            assert_eq!(v[2], Some("c".to_string()));
            assert_eq!(v[3], Some("d".to_string()));
        } else {
            panic!("dtype 错误");
        }

        // 按第 0 列降序排序: None, 3, 2, 1
        let sorted_desc = df.sort_values(&[0], false);
        if let ColumnData::Int(v) = &sorted_desc.data[0].data {
            assert_eq!(v[0], None);
            assert_eq!(v[1], Some(3));
            assert_eq!(v[2], Some(2));
            assert_eq!(v[3], Some(1));
        } else {
            panic!("dtype 错误");
        }

        // 按字符串列 (索引 1) 升序排序
        let s3 = Series::new_string(
            None,
            vec![
                Some("banana".to_string()),
                Some("apple".to_string()),
                Some("cherry".to_string()),
            ],
        );
        let s4 = Series::new_int(None, vec![Some(10), Some(20), Some(30)]);
        let df2 = DataFrame::from_series(vec!["s".to_string(), "n".to_string()], vec![s3, s4])
            .expect("DataFrame 构建失败");
        let sorted_str = df2.sort_values(&[0], true);
        if let ColumnData::String(v) = &sorted_str.data[0].data {
            assert_eq!(v[0], Some("apple".to_string()));
            assert_eq!(v[1], Some("banana".to_string()));
            assert_eq!(v[2], Some("cherry".to_string()));
        } else {
            panic!("dtype 错误");
        }
        // 验证 n 列跟随重排: 20, 10, 30
        if let ColumnData::Int(v) = &sorted_str.data[1].data {
            assert_eq!(v[0], Some(20));
            assert_eq!(v[1], Some(10));
            assert_eq!(v[2], Some(30));
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_dataframe_sort_index() {
        let s1 = Series::new_int(None, vec![Some(1), Some(2), Some(3)]);
        let s2 = Series::new_float(None, vec![Some(1.5), Some(2.5), Some(3.5)]);
        let df = DataFrame::from_series(vec!["a".to_string(), "b".to_string()], vec![s1, s2])
            .expect("DataFrame 构建失败");

        // ascending=true: 保持原顺序
        let sorted_asc = df.sort_index(true);
        assert_eq!(sorted_asc.nrows(), 3);
        if let ColumnData::Int(v) = &sorted_asc.data[0].data {
            assert_eq!(v[0], Some(1));
            assert_eq!(v[1], Some(2));
            assert_eq!(v[2], Some(3));
        } else {
            panic!("dtype 错误");
        }
        if let ColumnData::Float(v) = &sorted_asc.data[1].data {
            assert_eq!(v[0], Some(1.5));
            assert_eq!(v[1], Some(2.5));
            assert_eq!(v[2], Some(3.5));
        } else {
            panic!("dtype 错误");
        }

        // ascending=false: 反转行顺序
        let sorted_desc = df.sort_index(false);
        if let ColumnData::Int(v) = &sorted_desc.data[0].data {
            assert_eq!(v[0], Some(3));
            assert_eq!(v[1], Some(2));
            assert_eq!(v[2], Some(1));
        } else {
            panic!("dtype 错误");
        }
        if let ColumnData::Float(v) = &sorted_desc.data[1].data {
            assert_eq!(v[0], Some(3.5));
            assert_eq!(v[1], Some(2.5));
            assert_eq!(v[2], Some(1.5));
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_dataframe_dropna() {
        // 列 a: [1, None, 3, None]
        // 列 b: [1.0, 2.0, None, None]
        // 期望删除后只剩第 0 行 (两列都非空)
        let s1 = Series::new_int(None, vec![Some(1), None, Some(3), None]);
        let s2 = Series::new_float(None, vec![Some(1.0), Some(2.0), None, None]);
        let df = DataFrame::from_series(vec!["a".to_string(), "b".to_string()], vec![s1, s2])
            .expect("DataFrame 构建失败");
        assert_eq!(df.nrows(), 4);

        let dropped = df.dropna();
        assert_eq!(dropped.nrows(), 1);
        // 验证剩余的第 0 行数据
        if let ColumnData::Int(v) = &dropped.data[0].data {
            assert_eq!(v[0], Some(1));
        } else {
            panic!("dtype 错误");
        }
        if let ColumnData::Float(v) = &dropped.data[1].data {
            assert_eq!(v[0], Some(1.0));
        } else {
            panic!("dtype 错误");
        }

        // 全非空 DataFrame 删除后行数不变
        let s3 = Series::new_int(None, vec![Some(1), Some(2)]);
        let s4 = Series::new_string(None, vec![Some("x".to_string()), Some("y".to_string())]);
        let df2 = DataFrame::from_series(vec!["a".to_string(), "b".to_string()], vec![s3, s4])
            .expect("DataFrame 构建失败");
        assert_eq!(df2.dropna().nrows(), 2);

        // 空 DataFrame dropna 安全
        let empty = DataFrame::new_empty();
        assert_eq!(empty.dropna().nrows(), 0);
    }

    #[test]
    fn test_dataframe_isnull() {
        // 列 a: [1, None, 3]
        // 列 b: ["x", "y", None]
        // isnull_rows: [false, true, true] (任意列为 None)
        // notnull_rows: [true, false, false] (所有列非 None)
        let s1 = Series::new_int(None, vec![Some(1), None, Some(3)]);
        let s2 = Series::new_string(
            None,
            vec![Some("x".to_string()), Some("y".to_string()), None],
        );
        let df = DataFrame::from_series(vec!["a".to_string(), "b".to_string()], vec![s1, s2])
            .expect("DataFrame 构建失败");

        let isnull = df.isnull_rows();
        assert_eq!(isnull, vec![false, true, true]);

        let notnull = df.notnull_rows();
        assert_eq!(notnull, vec![true, false, false]);

        // 全非空: isnull 全 false, notnull 全 true
        let s3 = Series::new_int(None, vec![Some(1), Some(2)]);
        let s4 = Series::new_float(None, vec![Some(1.5), Some(2.5)]);
        let df2 = DataFrame::from_series(vec!["a".to_string(), "b".to_string()], vec![s3, s4])
            .expect("DataFrame 构建失败");
        assert_eq!(df2.isnull_rows(), vec![false, false]);
        assert_eq!(df2.notnull_rows(), vec![true, true]);

        // 空 DataFrame: isnull/notnull 返回空 Vec
        let empty = DataFrame::new_empty();
        assert!(empty.isnull_rows().is_empty());
        assert!(empty.notnull_rows().is_empty());
    }

    // 防止 DType 未使用警告
    #[test]
    fn test_dtype_compile() {
        let _ = DType::Int64;
    }

    #[test]
    fn test_dataframe_merge() {
        // 构建左表: id, name
        let left = DataFrame {
            columns: vec!["id".to_string(), "name".to_string()],
            data: vec![
                Series::from_options_i64("id".to_string(), &[Some(1), Some(2), Some(3)]),
                Series::from_options_string(
                    "name".to_string(),
                    &[
                        Some("a".to_string()),
                        Some("b".to_string()),
                        Some("c".to_string()),
                    ],
                ),
            ],
        };

        // 构建右表: id, value
        let right = DataFrame {
            columns: vec!["id".to_string(), "value".to_string()],
            data: vec![
                Series::from_options_i64("id".to_string(), &[Some(1), Some(2), Some(4)]),
                Series::from_options_string(
                    "value".to_string(),
                    &[
                        Some("x".to_string()),
                        Some("y".to_string()),
                        Some("z".to_string()),
                    ],
                ),
            ],
        };

        // 内连接
        let merged = left.merge(&right, 0, 0, "inner");
        // 列: id(左), name, value(右表 id 被跳过)
        assert_eq!(merged.columns.len(), 3);
        // 应匹配 id=1 和 id=2 两行
        assert_eq!(merged.data[0].len(), 2);
    }

    #[test]
    fn test_dataframe_groupby() {
        // 构建表: category, value
        let df = DataFrame {
            columns: vec!["category".to_string(), "value".to_string()],
            data: vec![
                Series::from_options_string(
                    "category".to_string(),
                    &[
                        Some("A".to_string()),
                        Some("B".to_string()),
                        Some("A".to_string()),
                    ],
                ),
                Series::from_options_f64(
                    "value".to_string(),
                    &[Some(10.0), Some(20.0), Some(30.0)],
                ),
            ],
        };

        let (keys, result) = df.groupby_agg(0, "sum");
        // A, B 两组
        assert_eq!(keys.len(), 2);
        // category, value
        assert_eq!(result.columns.len(), 2);
        // A 组 sum = 10.0 + 30.0 = 40.0, B 组 sum = 20.0
        if let ColumnData::Float(v) = &result.data[1].data {
            assert!((v[0].unwrap() - 40.0).abs() < 1e-10);
            assert!((v[1].unwrap() - 20.0).abs() < 1e-10);
        } else {
            panic!("应为 Float 类型");
        }
    }

    #[test]
    fn test_dataframe_pivot() {
        // 构建表: id, category, value
        let df = DataFrame {
            columns: vec![
                "id".to_string(),
                "category".to_string(),
                "value".to_string(),
            ],
            data: vec![
                Series::from_options_string(
                    "id".to_string(),
                    &[
                        Some("a".to_string()),
                        Some("a".to_string()),
                        Some("b".to_string()),
                    ],
                ),
                Series::from_options_string(
                    "category".to_string(),
                    &[
                        Some("X".to_string()),
                        Some("Y".to_string()),
                        Some("X".to_string()),
                    ],
                ),
                Series::from_options_f64("value".to_string(), &[Some(1.0), Some(2.0), Some(3.0)]),
            ],
        };

        let pivoted = df.pivot(0, 1, 2, "sum");
        // 列: id, X, Y
        assert_eq!(pivoted.columns.len(), 3);
        assert_eq!(pivoted.data[0].len(), 2); // a, b 两行
        if let ColumnData::Float(v) = &pivoted.data[1].data {
            // a 的 X 列 = 1.0, b 的 X 列 = 3.0
            assert!((v[0].unwrap() - 1.0).abs() < 1e-10);
            assert!((v[1].unwrap() - 3.0).abs() < 1e-10);
        } else {
            panic!("应为 Float 类型");
        }
    }

    #[test]
    fn test_dataframe_melt() {
        // 构建表: id, A, B
        let df = DataFrame {
            columns: vec!["id".to_string(), "A".to_string(), "B".to_string()],
            data: vec![
                Series::from_options_string(
                    "id".to_string(),
                    &[Some("x".to_string()), Some("y".to_string())],
                ),
                Series::from_options_f64("A".to_string(), &[Some(1.0), Some(2.0)]),
                Series::from_options_f64("B".to_string(), &[Some(3.0), Some(4.0)]),
            ],
        };

        let melted = df.melt(&[0], &[1, 2]);
        // 列: id, variable, value
        assert_eq!(melted.columns.len(), 3);
        // 行数 = 2 * 2 = 4
        assert_eq!(melted.data[0].len(), 4);
    }
}
