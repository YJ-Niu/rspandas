//! Series 排序与查找：sort_values / sort_index / searchsorted / arg_top_n。
//!
//! - sort_values / sort_index 与 pandas 默认行为对齐（None 的位置随 ascending 切换）
//! - searchsorted 等价于 numpy.searchsorted，支持 side 与 sorter
//! - arg_top_n 对应 pandas.Series.nsmallest / nlargest，支持 keep=first/last/all

use crate::core::dtype::{CategoricalData, ColumnData};
use crate::core::series::Series;

impl Series {
    // ---------- 排序 ----------

    /// 按值排序
    /// ascending=true: 升序，None 放最后
    /// ascending=false: 降序，None 放最前
    pub fn sort_values(&self, ascending: bool) -> Series {
        let data = match &self.data {
            ColumnData::Int(v) => {
                let mut indexed: Vec<(Option<i64>, usize)> =
                    v.iter().enumerate().map(|(i, x)| (*x, i)).collect();
                indexed.sort_by(|a, b| match (a.0, b.0) {
                    (Some(x), Some(y)) => {
                        if ascending {
                            x.cmp(&y)
                        } else {
                            y.cmp(&x)
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
                });
                let sorted: Vec<Option<i64>> = indexed.into_iter().map(|(x, _)| x).collect();
                ColumnData::Int(sorted)
            }
            ColumnData::Float(v) => {
                let mut indexed: Vec<(Option<f64>, usize)> =
                    v.iter().enumerate().map(|(i, x)| (*x, i)).collect();
                indexed.sort_by(|a, b| match (a.0, b.0) {
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
                });
                let sorted: Vec<Option<f64>> = indexed.into_iter().map(|(x, _)| x).collect();
                ColumnData::Float(sorted)
            }
            ColumnData::Bool(v) => {
                let mut indexed: Vec<(Option<bool>, usize)> =
                    v.iter().enumerate().map(|(i, x)| (*x, i)).collect();
                indexed.sort_by(|a, b| match (a.0, b.0) {
                    (Some(x), Some(y)) => {
                        if ascending {
                            x.cmp(&y)
                        } else {
                            y.cmp(&x)
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
                });
                let sorted: Vec<Option<bool>> = indexed.into_iter().map(|(x, _)| x).collect();
                ColumnData::Bool(sorted)
            }
            ColumnData::String(v) => {
                let mut indexed: Vec<(Option<String>, usize)> =
                    v.iter().enumerate().map(|(i, x)| (x.clone(), i)).collect();
                indexed.sort_by(|a, b| match (&a.0, &b.0) {
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
                });
                let sorted: Vec<Option<String>> = indexed.into_iter().map(|(x, _)| x).collect();
                ColumnData::String(sorted)
            }
            ColumnData::Categorical(c) => {
                // 对 categorical 按其字符串值排序
                let mut indexed: Vec<(Option<String>, usize)> = c
                    .codes
                    .iter()
                    .enumerate()
                    .map(|(i, code)| {
                        let s = code.and_then(|idx| c.categories.get(idx as usize).cloned());
                        (s, i)
                    })
                    .collect();
                indexed.sort_by(|a, b| match (&a.0, &b.0) {
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
                });
                let sorted_codes: Vec<Option<i32>> = indexed
                    .into_iter()
                    .map(|(_, i)| c.codes.get(i).copied().flatten())
                    .collect();
                ColumnData::Categorical(CategoricalData {
                    categories: c.categories.clone(),
                    codes: sorted_codes,
                    ordered: c.ordered,
                })
            }
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 按索引排序
    /// ascending=true: 保持原始顺序
    /// ascending=false: 反转原始顺序
    pub fn sort_index(&self, ascending: bool) -> Series {
        if ascending {
            return self.clone();
        }
        let data = match &self.data {
            ColumnData::Int(v) => ColumnData::Int(v.iter().rev().cloned().collect()),
            ColumnData::Float(v) => ColumnData::Float(v.iter().rev().cloned().collect()),
            ColumnData::Bool(v) => ColumnData::Bool(v.iter().rev().cloned().collect()),
            ColumnData::String(v) => ColumnData::String(v.iter().rev().cloned().collect()),
            ColumnData::Categorical(c) => ColumnData::Categorical(CategoricalData {
                categories: c.categories.clone(),
                codes: c.codes.iter().rev().cloned().collect(),
                ordered: c.ordered,
            }),
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 二分查找插入位置（等价于 numpy.searchsorted）。
    ///
    /// - values: 需要查找的目标值列表（已转成 f64）
    /// - side: "left" | "right"
    /// - sorter: 可选长度等于 self.len() 的索引数组，使得 `a[sorter]` 为升序；
    ///   若为 None 则假定 self 本身已升序。
    ///
    /// 行为：NaNs（self 或 values 中的 NaN）与 numpy 保持一致：
    /// - self 中的 NaN 会被放在有序序列末尾（与 pandas.Series.searchsorted 行为一致，
    ///   NaN 比较视为大于任何有限值）；
    /// - values 中的 NaN 将插入到最末尾（>= len(arr)）。
    pub fn searchsorted(&self, values: &[f64], side: &str, sorter: Option<&[usize]>) -> Vec<usize> {
        let f64_vals = self.as_f64_vec();
        let n = f64_vals.len();

        // 构造排序后的 arr_sorted：按 sorter 或原顺序，过滤 None 但保留 NaN
        let mut arr_sorted: Vec<f64> = Vec::with_capacity(n);
        if let Some(idx) = sorter {
            for &i in idx.iter().take(n) {
                if let Some(v) = f64_vals[i] {
                    arr_sorted.push(v);
                }
            }
        } else {
            for v in f64_vals.iter().flatten() {
                arr_sorted.push(*v);
            }
        }

        let is_right = matches!(side, "right");
        values
            .iter()
            .map(|&x| {
                if x.is_nan() {
                    return arr_sorted.len();
                }
                if is_right {
                    // bisect_right: 第一个 > x 的位置
                    arr_sorted.partition_point(|&v| {
                        !v.is_nan() && (v <= x || v.partial_cmp(&x).is_none())
                    })
                } else {
                    // bisect_left: 第一个 >= x 的位置（NaN 视为 > 任何有限值）
                    arr_sorted
                        .partition_point(|&v| !v.is_nan() && (v < x || v.partial_cmp(&x).is_none()))
                }
            })
            .collect()
    }

    /// 返回最小/最大 n 个元素在原 Series 中的位置（索引）。
    ///
    /// - `n`: 要返回的元素数量
    /// - `keep`: 重复值保留方式（"first" / "last" / "all"）
    /// - `largest`: true=nlargest（降序），false=nsmallest（升序）
    ///
    /// 行为与 pandas.Series.nsmallest/nlargest 一致：
    /// - None 与 NaN 一律跳过（不返回）
    /// - 使用稳定排序，``keep="first"`` 时保留原顺序中先出现者
    /// - ``keep="all"`` 时返回所有等于第 n 个值的元素（可能多于 n）
    pub fn arg_top_n(&self, n: usize, keep: &str, largest: bool) -> Vec<usize> {
        let f64_vals = self.as_f64_vec();
        // 收集 (原索引, 值)，跳过 None 与 NaN
        let mut indexed: Vec<(usize, f64)> = f64_vals
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.filter(|x| !x.is_nan()).map(|x| (i, x)))
            .collect();
        if indexed.is_empty() || n == 0 {
            return Vec::new();
        }
        // 稳定排序：largest=降序，否则升序
        indexed.sort_by(|a, b| {
            let ord = a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal);
            if largest { ord.reverse() } else { ord }
        });

        let total = indexed.len();
        // 选取索引列表
        let selected: Vec<usize> = match keep {
            "last" => {
                // 保持按值升/降序，但对相同值内的元素反转顺序
                // （即相同值优先取最后出现的）
                // 实现：按 (值, Reverse(原索引)) 排序
                indexed.sort_by(|a, b| {
                    let ord = a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal);
                    if ord == std::cmp::Ordering::Equal {
                        // 相同值时按原索引倒序
                        b.0.cmp(&a.0)
                    } else if largest {
                        ord.reverse()
                    } else {
                        ord
                    }
                });
                indexed.into_iter().take(n).map(|(i, _)| i).collect()
            }
            "all" => {
                if total <= n {
                    indexed.into_iter().map(|(i, _)| i).collect()
                } else {
                    // 阈值：第 n 个值
                    let threshold = indexed[n - 1].1;
                    indexed
                        .into_iter()
                        .take_while(|(_, v)| {
                            if largest {
                                *v >= threshold
                            } else {
                                *v <= threshold
                            }
                        })
                        .map(|(i, _)| i)
                        .collect()
                }
            }
            // 默认 "first"
            _ => indexed.into_iter().take(n).map(|(i, _)| i).collect(),
        };
        selected
    }
}
