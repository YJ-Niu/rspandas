//! Series 缺失值处理：isnull / notnull / dropna / fillna / ffill / bfill / interpolate / sample / resample。
//!
//! - 缺失值统一以 None 表示（浮点 NaN 也视作缺失）
//! - ffill / bfill / interpolate 与 pandas 默认行为对齐
//! - sample 提供可重现的 LCG 伪随机采样
//! - resample 按时间桶聚合

use std::collections::HashMap;

use crate::core::dtype::{CategoricalData, ColumnData};
use crate::core::series::Series;

impl Series {
    // ---------- 缺失值 ----------

    pub fn isnull(&self) -> Vec<bool> {
        self.data.isnull()
    }
    pub fn notnull(&self) -> Vec<bool> {
        self.data.notnull()
    }

    pub fn dropna(&self) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.dropna(),
        }
    }

    pub fn fillna_i64(&self, v: i64) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.fillna_i64(v),
        }
    }
    pub fn fillna_f64(&self, v: f64) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.fillna_f64(v),
        }
    }
    pub fn fillna_bool(&self, v: bool) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.fillna_bool(v),
        }
    }
    pub fn fillna_string(&self, v: &str) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.fillna_string(v),
        }
    }
    pub fn fillna_categorical(&self, v: &str) -> Series {
        Self {
            name: self.name.clone(),
            data: self.data.fillna_categorical(v),
        }
    }

    // ---------- 前向/后向填充 ----------

    /// 前向填充：将 None 填充为前一个非 None 值
    /// 若开头连续 None，则保持 None
    pub fn ffill(&self) -> Series {
        let data = match &self.data {
            ColumnData::Int(v) => {
                let mut result: Vec<Option<i64>> = Vec::with_capacity(v.len());
                let mut last: Option<i64> = None;
                for x in v.iter() {
                    if let Some(val) = x {
                        last = Some(*val);
                        result.push(Some(*val));
                    } else if let Some(l) = last {
                        result.push(Some(l));
                    } else {
                        result.push(None);
                    }
                }
                ColumnData::Int(result)
            }
            ColumnData::Float(v) => {
                let mut result: Vec<Option<f64>> = Vec::with_capacity(v.len());
                let mut last: Option<f64> = None;
                for x in v.iter() {
                    if let Some(val) = x {
                        last = Some(*val);
                        result.push(Some(*val));
                    } else if let Some(l) = last {
                        result.push(Some(l));
                    } else {
                        result.push(None);
                    }
                }
                ColumnData::Float(result)
            }
            ColumnData::Bool(v) => {
                let mut result: Vec<Option<bool>> = Vec::with_capacity(v.len());
                let mut last: Option<bool> = None;
                for x in v.iter() {
                    if let Some(val) = x {
                        last = Some(*val);
                        result.push(Some(*val));
                    } else if let Some(l) = last {
                        result.push(Some(l));
                    } else {
                        result.push(None);
                    }
                }
                ColumnData::Bool(result)
            }
            ColumnData::String(v) => {
                let mut result: Vec<Option<String>> = Vec::with_capacity(v.len());
                let mut last: Option<String> = None;
                for x in v.iter() {
                    if let Some(val) = x {
                        last = Some(val.clone());
                        result.push(Some(val.clone()));
                    } else if let Some(l) = &last {
                        result.push(Some(l.clone()));
                    } else {
                        result.push(None);
                    }
                }
                ColumnData::String(result)
            }
            ColumnData::Categorical(c) => {
                let mut result: Vec<Option<i32>> = Vec::with_capacity(c.codes.len());
                let mut last: Option<i32> = None;
                for code in c.codes.iter() {
                    if let Some(val) = code {
                        last = Some(*val);
                        result.push(Some(*val));
                    } else if let Some(l) = last {
                        result.push(Some(l));
                    } else {
                        result.push(None);
                    }
                }
                ColumnData::Categorical(CategoricalData {
                    categories: c.categories.clone(),
                    codes: result,
                    ordered: c.ordered,
                })
            }
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 后向填充：将 None 填充为后一个非 None 值
    /// 若末尾连续 None，则保持 None
    pub fn bfill(&self) -> Series {
        let data = match &self.data {
            ColumnData::Int(v) => {
                let n = v.len();
                let mut result: Vec<Option<i64>> = vec![None; n];
                let mut last: Option<i64> = None;
                for i in (0..n).rev() {
                    if let Some(val) = v[i] {
                        last = Some(val);
                        result[i] = Some(val);
                    } else if let Some(l) = last {
                        result[i] = Some(l);
                    }
                }
                ColumnData::Int(result)
            }
            ColumnData::Float(v) => {
                let n = v.len();
                let mut result: Vec<Option<f64>> = vec![None; n];
                let mut last: Option<f64> = None;
                for i in (0..n).rev() {
                    if let Some(val) = v[i] {
                        last = Some(val);
                        result[i] = Some(val);
                    } else if let Some(l) = last {
                        result[i] = Some(l);
                    }
                }
                ColumnData::Float(result)
            }
            ColumnData::Bool(v) => {
                let n = v.len();
                let mut result: Vec<Option<bool>> = vec![None; n];
                let mut last: Option<bool> = None;
                for i in (0..n).rev() {
                    if let Some(val) = v[i] {
                        last = Some(val);
                        result[i] = Some(val);
                    } else if let Some(l) = last {
                        result[i] = Some(l);
                    }
                }
                ColumnData::Bool(result)
            }
            ColumnData::String(v) => {
                let n = v.len();
                let mut result: Vec<Option<String>> = vec![None; n];
                let mut last: Option<String> = None;
                for i in (0..n).rev() {
                    if let Some(val) = &v[i] {
                        last = Some(val.clone());
                        result[i] = Some(val.clone());
                    } else if let Some(l) = &last {
                        result[i] = Some(l.clone());
                    }
                }
                ColumnData::String(result)
            }
            ColumnData::Categorical(c) => {
                let n = c.codes.len();
                let mut result: Vec<Option<i32>> = vec![None; n];
                let mut last: Option<i32> = None;
                for i in (0..n).rev() {
                    if let Some(val) = c.codes[i] {
                        last = Some(val);
                        result[i] = Some(val);
                    } else if let Some(l) = last {
                        result[i] = Some(l);
                    }
                }
                ColumnData::Categorical(CategoricalData {
                    categories: c.categories.clone(),
                    codes: result,
                    ordered: c.ordered,
                })
            }
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    // ---------- 插值 / 采样 / 重采样 ----------

    /// 线性插值填充 None
    /// method: "linear" / "nearest" / "zero"
    /// limit: 最多连续填充数量（None 表示无限制）
    pub fn interpolate(&self, method: &str, limit: Option<usize>) -> Series {
        let f64_vec = self.as_f64_vec();
        let n = f64_vec.len();
        if n == 0 {
            return self.clone();
        }
        let mut result: Vec<Option<f64>> = f64_vec.clone();
        let mut i = 0;
        while i < n {
            if result[i].is_some() {
                i += 1;
                continue;
            }
            // 找到连续 None 区间 [i, j)
            let mut j = i;
            while j < n && result[j].is_none() {
                j += 1;
            }
            let left_val = if i > 0 { result[i - 1] } else { None };
            let right_val = if j < n { result[j] } else { None };
            match (left_val, right_val) {
                (Some(lv), Some(rv)) => {
                    let gap = j - i;
                    for (k, slot) in result.iter_mut().enumerate().take(j).skip(i) {
                        if let Some(lim) = limit
                            && k - i >= lim
                        {
                            break;
                        }
                        let frac = (k - i + 1) as f64 / (gap + 1) as f64;
                        *slot = Some(lv + (rv - lv) * frac);
                    }
                }
                (Some(lv), None) => {
                    if method == "linear" {
                        for (k, slot) in result.iter_mut().enumerate().take(j).skip(i) {
                            if let Some(lim) = limit
                                && k - i >= lim
                            {
                                break;
                            }
                            *slot = Some(lv);
                        }
                    }
                }
                (None, Some(rv)) => {
                    if method == "linear" {
                        for (k, slot) in result.iter_mut().enumerate().take(j).skip(i) {
                            if let Some(lim) = limit
                                && j - k > lim
                            {
                                continue;
                            }
                            *slot = Some(rv);
                        }
                    }
                }
                (None, None) => {}
            }
            i = j;
        }
        Series {
            name: self.name.clone(),
            data: ColumnData::Float(result),
        }
    }

    /// 随机采样
    /// n: 采样数量（None 时使用 frac）
    /// frac: 采样比例
    /// replace: 是否有放回
    /// seed: 随机种子
    pub fn sample(
        &self,
        n: Option<usize>,
        frac: Option<f64>,
        replace: bool,
        seed: Option<u64>,
    ) -> Series {
        let len = self.len();
        if len == 0 {
            return self.clone();
        }
        let sample_n = if let Some(f) = frac {
            ((len as f64) * f).round() as usize
        } else {
            n.unwrap_or(1)
        };
        // 简单 LCG 伪随机数生成器（可重现）
        let mut state = seed.unwrap_or(0xC0FFEE);
        let next_rand = |state: &mut u64| {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*state >> 33) as usize
        };
        let indices: Vec<usize> = if replace {
            (0..sample_n).map(|_| next_rand(&mut state) % len).collect()
        } else {
            let take_n = sample_n.min(len);
            // Fisher-Yates 洗牌
            let mut idx: Vec<usize> = (0..len).collect();
            for k in 0..take_n {
                let r = k + next_rand(&mut state) % (len - k);
                idx.swap(k, r);
            }
            idx.truncate(take_n);
            idx
        };
        // 按采样索引构造新 Series
        let mut sampled: Vec<Option<f64>> = Vec::with_capacity(indices.len());
        let f64_vec = self.as_f64_vec();
        for idx in &indices {
            sampled.push(f64_vec[*idx]);
        }
        Series {
            name: self.name.clone(),
            data: ColumnData::Float(sampled),
        }
    }

    /// 时间序列重采样聚合
    /// timestamps: 每行对应的 epoch 秒
    /// freq_seconds: 桶宽度（秒）
    /// agg: "sum"/"mean"/"count"/"min"/"max"/"median"/"first"/"last"
    /// 返回 (桶起始时间列表, 聚合值列表)
    pub fn resample(
        &self,
        timestamps: &[f64],
        freq_seconds: f64,
        agg: &str,
    ) -> (Vec<f64>, Vec<Option<f64>>) {
        let n = self.len();
        if n == 0 || timestamps.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let values = self.as_f64_vec();
        // 按 floor(ts/freq) 分组
        let mut buckets: HashMap<i64, Vec<f64>> = HashMap::new();
        let mut bucket_order: Vec<i64> = Vec::new();
        for i in 0..n.min(timestamps.len()) {
            if values[i].is_none() {
                continue;
            }
            let bucket_id = (timestamps[i] / freq_seconds).floor() as i64;
            if !buckets.contains_key(&bucket_id) {
                bucket_order.push(bucket_id);
            }
            buckets
                .entry(bucket_id)
                .or_default()
                .push(values[i].unwrap());
        }
        // 按时间戳升序
        bucket_order.sort();
        let out_ts: Vec<f64> = bucket_order
            .iter()
            .map(|b| (*b as f64) * freq_seconds)
            .collect();
        let out_vals: Vec<Option<f64>> = bucket_order
            .iter()
            .map(|b| {
                let nums = buckets.get(b).unwrap();
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
                    "first" => Some(nums[0]),
                    "last" => Some(nums[nums.len() - 1]),
                    _ => None,
                }
            })
            .collect();
        (out_ts, out_vals)
    }
}
