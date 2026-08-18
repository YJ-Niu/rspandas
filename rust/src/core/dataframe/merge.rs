//! DataFrame 合并与分组方法：merge / groupby_agg / query_filter。
//!
//! - merge：基于键列的哈希连接（inner/left/right/outer）
//! - groupby_agg：按 by 列分组并对每列执行聚合函数
//! - query_filter：按列比较标量过滤行（query 简化版）

use std::collections::HashMap;

use crate::core::dataframe::DataFrame;
use crate::core::series::Series;

impl DataFrame {
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
