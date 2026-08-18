//! Series 分组与批量聚合：groupby_agg_series / batch_agg / compare_scalar。
//!
//! - groupby_agg_series 按字符串键分组后单趟聚合（sum/mean/count/min/max/...）
//! - batch_agg 一次遍历计算多个聚合值
//! - compare_scalar 提供 query 简化版的列 op 标量过滤

use std::collections::HashMap;

use crate::core::series::Series;

impl Series {
    // ---------- SeriesGroupBy 聚合 ----------

    /// 按 by 列（Vec<String>）分组聚合
    /// by: 每行对应的分组键（字符串表示）
    /// agg: "sum"/"mean"/"count"/"min"/"max"/"median"/"std"/"var"/"prod"/"first"/"last"
    /// 返回 (group_keys, agg_values)
    pub fn groupby_agg_series(&self, by: &[String], agg: &str) -> (Vec<String>, Vec<Option<f64>>) {
        let n = self.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let values = self.as_f64_vec();
        // 分组：键 -> 行索引列表
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        let mut group_order: Vec<String> = Vec::new();
        for (i, key) in by.iter().enumerate().take(n.min(by.len())) {
            if !groups.contains_key(key) {
                group_order.push(key.clone());
            }
            groups.entry(key.clone()).or_default().push(i);
        }
        // 排序键（保持稳定）
        group_order.sort();
        let out_vals: Vec<Option<f64>> = group_order
            .iter()
            .map(|k| {
                let indices = groups.get(k).unwrap();
                let nums: Vec<f64> = indices.iter().filter_map(|&i| values[i]).collect();
                if nums.is_empty() {
                    return None;
                }
                match agg {
                    "sum" => Some(nums.iter().sum()),
                    "mean" => Some(nums.iter().sum::<f64>() / nums.len() as f64),
                    "count" => Some(nums.len() as f64),
                    "min" => nums.iter().cloned().fold(None::<f64>, |acc, x| {
                        Some(match acc {
                            Some(a) => a.min(x),
                            None => x,
                        })
                    }),
                    "max" => nums.iter().cloned().fold(None::<f64>, |acc, x| {
                        Some(match acc {
                            Some(a) => a.max(x),
                            None => x,
                        })
                    }),
                    "median" => {
                        let mut sorted = nums.clone();
                        sorted
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let m = sorted.len();
                        if m % 2 == 1 {
                            Some(sorted[m / 2])
                        } else {
                            Some((sorted[m / 2 - 1] + sorted[m / 2]) / 2.0)
                        }
                    }
                    "std" => {
                        if nums.len() < 2 {
                            return None;
                        }
                        let m = nums.iter().sum::<f64>() / nums.len() as f64;
                        let v =
                            nums.iter().map(|x| (x - m).powi(2)).sum::<f64>() / nums.len() as f64;
                        Some(v.sqrt())
                    }
                    "var" => {
                        if nums.len() < 2 {
                            return None;
                        }
                        let m = nums.iter().sum::<f64>() / nums.len() as f64;
                        Some(nums.iter().map(|x| (x - m).powi(2)).sum::<f64>() / nums.len() as f64)
                    }
                    "prod" => Some(nums.iter().product()),
                    "first" => Some(nums[0]),
                    "last" => Some(nums[nums.len() - 1]),
                    _ => None,
                }
            })
            .collect();
        (group_order, out_vals)
    }

    // ---------- 批量聚合（一次遍历多聚合） ----------

    /// 一次遍历计算多个聚合值
    /// aggs: 聚合名列表，如 ["sum", "mean", "min", "max", "count", "std", "var"]
    pub fn batch_agg(&self, aggs: &[String]) -> Vec<Option<f64>> {
        let values: Vec<f64> = self.as_f64_vec().into_iter().flatten().collect();
        let cnt = values.len();
        if cnt == 0 {
            return aggs.iter().map(|_| None).collect();
        }
        let sum: f64 = values.iter().sum();
        let mean = sum / cnt as f64;
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        aggs.iter()
            .map(|a| match a.as_str() {
                "sum" => Some(sum),
                "mean" => Some(mean),
                "count" => Some(cnt as f64),
                "min" => Some(sorted[0]),
                "max" => Some(sorted[cnt - 1]),
                "median" => {
                    if cnt % 2 == 1 {
                        Some(sorted[cnt / 2])
                    } else {
                        Some((sorted[cnt / 2 - 1] + sorted[cnt / 2]) / 2.0)
                    }
                }
                "std" => {
                    // 默认 ddof=1（与 pandas 一致）
                    if cnt < 2 {
                        None
                    } else {
                        let v = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                            / (cnt - 1) as f64;
                        Some(v.sqrt())
                    }
                }
                "var" => {
                    // 默认 ddof=1（与 pandas 一致）
                    if cnt < 2 {
                        None
                    } else {
                        Some(
                            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                                / (cnt - 1) as f64,
                        )
                    }
                }
                "prod" => Some(values.iter().product()),
                _ => None,
            })
            .collect()
    }

    // ---------- 简单表达式过滤（query 简化版） ----------

    /// 简单比较过滤：列 op 标量
    /// op: ">" / "<" / ">=" / "<=" / "==" / "!="
    /// 返回符合条件行的 mask
    pub fn compare_scalar(&self, op: &str, value: f64) -> Vec<bool> {
        let f64_vec = self.as_f64_vec();
        f64_vec
            .iter()
            .map(|x| match (x, op) {
                (Some(v), ">") => *v > value,
                (Some(v), "<") => *v < value,
                (Some(v), ">=") => *v >= value,
                (Some(v), "<=") => *v <= value,
                (Some(v), "==") => (*v - value).abs() < f64::EPSILON,
                (Some(v), "!=") => (*v - value).abs() >= f64::EPSILON,
                _ => false,
            })
            .collect()
    }
}
