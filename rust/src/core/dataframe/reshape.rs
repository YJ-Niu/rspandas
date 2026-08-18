//! DataFrame 形状变换方法：pivot / melt / stack / unstack。
//!
//! - pivot：透视表，按 index_col 分组，columns_col 的值作为新列名，values_col 聚合
//! - melt：宽转长，将指定的值列转为 (variable, value) 两列
//! - stack：将列堆叠为行，返回 index/variable/value 三列
//! - unstack：将 variable/value 列透视为宽表

use std::collections::HashMap;

use crate::core::dataframe::DataFrame;
use crate::core::series::Series;

impl DataFrame {
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
}
