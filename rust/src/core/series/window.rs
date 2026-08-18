//! Series 窗口与排名计算：quantile / rank / value_counts / rolling_* / expanding_* / ewm_mean。
//!
//! 滚动窗口使用 O(n) 前缀和实现，优于朴素 O(nw) 滑动求和；
//! rank 支持 average/min/max/first/dense 五种方法与 keep/top/bottom 三种 NA 位置策略。

use rayon::prelude::*;

use crate::core::dtype::ColumnData;
use crate::core::series::Series;

impl Series {
    // ---------- 分位数 / 排名 / 窗口计算 ----------

    /// 计算分位数（线性插值法）
    /// q: 0.0-1.0 之间的分位数
    pub fn quantile(&self, q: f64) -> Option<f64> {
        // 收集非 None 且非 NaN 的 f64 值
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
        if n == 1 {
            return Some(vs[0]);
        }
        let pos = q * (n - 1) as f64;
        let lower = pos.floor() as usize;
        let upper = pos.ceil() as usize;
        let frac = pos - lower as f64;
        Some(vs[lower] * (1.0 - frac) + vs[upper] * frac)
    }

    /// 计算排名（扩展版，含 dense + na_option）
    /// method: "average"=平均排名, "min"=最小排名, "max"=最大排名,
    ///         "first"=出现顺序, "dense"=密集排名（不跳号）
    /// ascending: true=升序, false=降序
    /// na_option: "keep"=None保持None, "top"=None排最前, "bottom"=None排最后
    pub fn rank(&self, method: &str, ascending: bool, na_option: Option<&str>) -> Vec<Option<f64>> {
        let n = self.len();
        let mut result: Vec<Option<f64>> = vec![None; n];
        let na = na_option.unwrap_or("keep");

        // 收集 (value, original_index, is_none)
        #[derive(Clone, Copy)]
        enum EntryVal {
            Val(f64),
            NaN,
        }
        let mut indexed: Vec<(EntryVal, usize)> = match &self.data {
            ColumnData::Int(v) => v
                .iter()
                .enumerate()
                .map(|(i, x)| match x {
                    Some(val) => (EntryVal::Val(*val as f64), i),
                    None => (EntryVal::NaN, i),
                })
                .collect(),
            ColumnData::Float(v) => v
                .iter()
                .enumerate()
                .map(|(i, x)| match x {
                    Some(val) => {
                        if val.is_nan() {
                            (EntryVal::NaN, i)
                        } else {
                            (EntryVal::Val(*val), i)
                        }
                    }
                    None => (EntryVal::NaN, i),
                })
                .collect(),
            _ => return result,
        };

        // 排序：根据 na_option 决定 None 值的位置
        indexed.sort_by(|a, b| {
            use EntryVal::*;
            match (a.0, b.0, na, ascending) {
                // keep: NaN 之间视为相等，且 NaN 与 Val 不参与排名（留最后）
                (NaN, NaN, "keep", _) => std::cmp::Ordering::Equal,
                (NaN, Val(_), "keep", _) => std::cmp::Ordering::Greater,
                (Val(_), NaN, "keep", _) => std::cmp::Ordering::Less,
                // top: NaN 排在最前
                (NaN, NaN, "top", _) => std::cmp::Ordering::Equal,
                (NaN, Val(_), "top", _) => std::cmp::Ordering::Less,
                (Val(_), NaN, "top", _) => std::cmp::Ordering::Greater,
                // bottom: NaN 排在最后
                (NaN, NaN, "bottom", _) => std::cmp::Ordering::Equal,
                (NaN, Val(_), "bottom", _) => std::cmp::Ordering::Greater,
                (Val(_), NaN, "bottom", _) => std::cmp::Ordering::Less,
                // 有效值之间按正常比较
                (Val(x), Val(y), _, true) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
                (Val(x), Val(y), _, false) => {
                    y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)
                }
                _ => std::cmp::Ordering::Equal,
            }
        });

        // 分配排名：找到第一个有效排名起点（跳过 keep 模式的前导 NaN）
        let m = indexed.len();
        // "keep" 模式下：NaN 保持 result None；有效值的排名范围 1..=有效数
        // "top"/"bottom" 模式下：NaN 也参与排名
        let effective_start = if na == "keep" {
            indexed.partition_point(|(v, _)| matches!(v, EntryVal::NaN))
        } else {
            0
        };
        let effective_end = if na == "keep" {
            // keep 模式下：跳过尾部 NaN（从左起遇到 NaN 停止）
            indexed
                .iter()
                .position(|(v, _)| matches!(v, EntryVal::NaN))
                .unwrap_or(m)
        } else {
            m
        };

        let effective_slice = &indexed[effective_start..effective_end];
        let m_eff = effective_slice.len();
        let mut i = 0usize;
        let mut dense_rank_counter = 0usize;

        while i < m_eff {
            let mut j = i + 1;
            while j < m_eff {
                let a = effective_slice[i].0;
                let b = effective_slice[j].0;
                let same = match (a, b) {
                    (EntryVal::Val(x), EntryVal::Val(y)) => x == y,
                    (EntryVal::NaN, EntryVal::NaN) => true,
                    _ => false,
                };
                if same {
                    j += 1;
                } else {
                    break;
                }
            }
            // 全局偏移位置
            let global_i = effective_start + i;
            let global_j = effective_start + j;
            dense_rank_counter += 1;

            match method {
                "average" => {
                    let avg = (global_i + global_j + 1) as f64 / 2.0;
                    for k in global_i..global_j {
                        result[indexed[k].1] = Some(avg);
                    }
                }
                "min" => {
                    let rank_val = (global_i + 1) as f64;
                    for k in global_i..global_j {
                        result[indexed[k].1] = Some(rank_val);
                    }
                }
                "max" => {
                    let rank_val = global_j as f64;
                    for k in global_i..global_j {
                        result[indexed[k].1] = Some(rank_val);
                    }
                }
                "first" => {
                    for k in global_i..global_j {
                        result[indexed[k].1] = Some((k + 1) as f64);
                    }
                }
                "dense" => {
                    let rank_val = dense_rank_counter as f64;
                    for k in global_i..global_j {
                        result[indexed[k].1] = Some(rank_val);
                    }
                }
                _ => {
                    // 默认 average
                    let avg = (global_i + global_j + 1) as f64 / 2.0;
                    for k in global_i..global_j {
                        result[indexed[k].1] = Some(avg);
                    }
                }
            }
            i = j;
        }
        result
    }

    /// 值计数 (value_counts)
    /// 返回 (unique_values: Vec<String>, counts: Vec<usize>)
    /// normalize=true 时 counts 占比为 f64，这里我们用 (values, counts_as_f64 或 counts)，
    /// 为简单统一返回 (values: Vec<Py<PyAny>>, counts: Vec<u64>) 用 Vec<(String, u64)> 代替
    pub fn value_counts(&self, sort: bool, ascending: bool) -> (Vec<String>, Vec<u64>) {
        use std::collections::HashMap;
        let mut map: HashMap<String, u64> = HashMap::new();
        match &self.data {
            ColumnData::Int(v) => {
                for x in v.iter() {
                    match x {
                        Some(val) => *map.entry(val.to_string()).or_insert(0) += 1,
                        None => *map.entry("None".to_string()).or_insert(0) += 1,
                    }
                }
            }
            ColumnData::Float(v) => {
                for x in v.iter() {
                    match x {
                        Some(val) => {
                            if val.is_nan() {
                                *map.entry("NaN".to_string()).or_insert(0) += 1;
                            } else {
                                *map.entry(val.to_string()).or_insert(0) += 1;
                            }
                        }
                        None => *map.entry("None".to_string()).or_insert(0) += 1,
                    }
                }
            }
            ColumnData::Bool(v) => {
                for x in v.iter() {
                    match x {
                        Some(val) => *map.entry(val.to_string()).or_insert(0) += 1,
                        None => *map.entry("None".to_string()).or_insert(0) += 1,
                    }
                }
            }
            ColumnData::String(v) => {
                for x in v.iter() {
                    match x {
                        Some(val) => *map.entry(val.clone()).or_insert(0) += 1,
                        None => *map.entry("None".to_string()).or_insert(0) += 1,
                    }
                }
            }
            ColumnData::Categorical(cd) => {
                for code in cd.codes.iter() {
                    match code {
                        Some(c) if (*c as usize) < cd.categories.len() => {
                            *map.entry(cd.categories[*c as usize].clone()).or_insert(0) += 1;
                        }
                        _ => *map.entry("None".to_string()).or_insert(0) += 1,
                    }
                }
            }
        }
        // dropna=True：移除缺失值（None 表示为 "None"；float NaN 表示为 "NaN"）
        map.remove("None");
        map.remove("NaN");
        let mut pairs: Vec<(String, u64)> = map.into_iter().collect();
        if sort {
            if ascending {
                pairs.sort_by_key(|a| a.1);
            } else {
                pairs.sort_by_key(|b| std::cmp::Reverse(b.1));
            }
        }
        let mut values = Vec::with_capacity(pairs.len());
        let mut counts = Vec::with_capacity(pairs.len());
        for (v, c) in pairs {
            values.push(v);
            counts.push(c);
        }
        (values, counts)
    }

    /// 滑动窗口求和（使用 O(n) 前缀和 + 缺失值计数，优于朴素 O(nw) 滑动求和）
    pub fn rolling_sum(&self, window: usize, min_periods: Option<usize>) -> Vec<Option<f64>> {
        let n = self.len();
        let min_per = min_periods.unwrap_or(window);
        let values = self.as_f64_vec();

        // prefix_sum[i] = sum(values[0..i])（仅对 Some(v) 累加，None 视为 0）
        // prefix_cnt[i] = 0..i 中非空个数
        let mut prefix_sum: Vec<f64> = vec![0.0; n + 1];
        let mut prefix_cnt: Vec<usize> = vec![0; n + 1];
        for (i, item) in values.iter().enumerate() {
            prefix_sum[i + 1] = prefix_sum[i];
            prefix_cnt[i + 1] = prefix_cnt[i];
            if let Some(v) = item {
                prefix_sum[i + 1] += *v;
                prefix_cnt[i + 1] += 1;
            }
        }

        let result: Vec<Option<f64>> = (0..n)
            .map(|i| {
                if i + 1 < min_per {
                    return None;
                }
                let start = (i + 1).saturating_sub(window);
                let end = i + 1;
                let cnt = prefix_cnt[end] - prefix_cnt[start];
                if cnt >= min_per {
                    Some(prefix_sum[end] - prefix_sum[start])
                } else {
                    None
                }
            })
            .collect();
        result
    }

    /// 滑动窗口均值（O(n) 前缀和，替代 O(nw) 朴素版）
    pub fn rolling_mean(&self, window: usize, min_periods: Option<usize>) -> Vec<Option<f64>> {
        let n = self.len();
        let min_per = min_periods.unwrap_or(window);
        let values = self.as_f64_vec();

        let mut prefix_sum: Vec<f64> = vec![0.0; n + 1];
        let mut prefix_cnt: Vec<usize> = vec![0; n + 1];
        for (i, item) in values.iter().enumerate() {
            prefix_sum[i + 1] = prefix_sum[i];
            prefix_cnt[i + 1] = prefix_cnt[i];
            if let Some(v) = item {
                prefix_sum[i + 1] += *v;
                prefix_cnt[i + 1] += 1;
            }
        }

        let result: Vec<Option<f64>> = (0..n)
            .map(|i| {
                if i + 1 < min_per {
                    return None;
                }
                let start = (i + 1).saturating_sub(window);
                let end = i + 1;
                let cnt = prefix_cnt[end] - prefix_cnt[start];
                if cnt >= min_per && cnt > 0 {
                    Some((prefix_sum[end] - prefix_sum[start]) / cnt as f64)
                } else {
                    None
                }
            })
            .collect();
        result
    }

    /// 滑动窗口标准差（O(n) 前缀和 sum/sum_sq 版本）
    pub fn rolling_std(&self, window: usize, min_periods: Option<usize>) -> Vec<Option<f64>> {
        let n = self.len();
        let min_per = min_periods.unwrap_or(window);
        let values = self.as_f64_vec();

        let mut prefix_sum: Vec<f64> = vec![0.0; n + 1];
        let mut prefix_sum_sq: Vec<f64> = vec![0.0; n + 1];
        let mut prefix_cnt: Vec<usize> = vec![0; n + 1];
        for (i, item) in values.iter().enumerate() {
            prefix_sum[i + 1] = prefix_sum[i];
            prefix_sum_sq[i + 1] = prefix_sum_sq[i];
            prefix_cnt[i + 1] = prefix_cnt[i];
            if let Some(v) = item {
                prefix_sum[i + 1] += *v;
                prefix_sum_sq[i + 1] += v * v;
                prefix_cnt[i + 1] += 1;
            }
        }

        let result: Vec<Option<f64>> = (0..n)
            .map(|i| {
                if i + 1 < min_per {
                    return None;
                }
                let start = (i + 1).saturating_sub(window);
                let end = i + 1;
                let cnt = prefix_cnt[end] - prefix_cnt[start];
                if cnt >= min_per && cnt > 1 {
                    let s = prefix_sum[end] - prefix_sum[start];
                    let s2 = prefix_sum_sq[end] - prefix_sum_sq[start];
                    let mean = s / cnt as f64;
                    // 总体方差 = s2/n - mean^2；样本方差需 /(n-1)（与 pandas ddof=1 一致）
                    let var = (s2 - s * mean) / (cnt - 1) as f64;
                    Some(var.max(0.0).sqrt())
                } else {
                    None
                }
            })
            .collect();
        result
    }

    /// 扩展窗口求和
    pub fn expanding_sum(&self, min_periods: Option<usize>) -> Vec<Option<f64>> {
        let n = self.len();
        let min_per = min_periods.unwrap_or(1);
        let values = self.as_f64_vec();
        let mut result: Vec<Option<f64>> = vec![None; n];
        let mut cumsum = 0.0;
        let mut count = 0;

        for i in 0..n {
            if let Some(v) = values[i] {
                cumsum += v;
                count += 1;
            }
            if count >= min_per {
                result[i] = Some(cumsum);
            }
        }
        result
    }

    /// 扩展窗口均值
    pub fn expanding_mean(&self, min_periods: Option<usize>) -> Vec<Option<f64>> {
        let n = self.len();
        let min_per = min_periods.unwrap_or(1);
        let values = self.as_f64_vec();
        let mut result: Vec<Option<f64>> = vec![None; n];
        let mut cumsum = 0.0;
        let mut count = 0;

        for i in 0..n {
            if let Some(v) = values[i] {
                cumsum += v;
                count += 1;
            }
            if count >= min_per && count > 0 {
                result[i] = Some(cumsum / count as f64);
            }
        }
        result
    }

    /// 指数加权移动平均
    /// alpha: 平滑因子 (0, 1]
    pub fn ewm_mean(&self, alpha: f64, min_periods: Option<usize>) -> Vec<Option<f64>> {
        let n = self.len();
        let min_per = min_periods.unwrap_or(0);
        let values = self.as_f64_vec();
        let mut result: Vec<Option<f64>> = vec![None; n];

        let mut prev_ema: Option<f64> = None;
        let mut count = 0;

        for i in 0..n {
            if let Some(v) = values[i] {
                count += 1;
                match prev_ema {
                    None => prev_ema = Some(v),
                    Some(prev) => prev_ema = Some(alpha * v + (1.0 - alpha) * prev),
                }
            }
            if count > min_per && prev_ema.is_some() {
                result[i] = prev_ema;
            }
        }
        result
    }
}
