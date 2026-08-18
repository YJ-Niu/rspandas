//! Series 访问器：Categorical 操作与日期时间 (dt_*) 方法。
//!
//! - Categorical 操作只对 `ColumnData::Categorical` 列有效，其他类型返回 None
//! - 日期时间方法对 `ColumnData::Float` 列有效（值为 Unix 纪元秒）

use rayon::prelude::*;

use crate::core::dtype::{CategoricalData, ColumnData};
use crate::core::series::Series;

impl Series {
    // ---------- Categorical 操作 ----------

    /// 获取 categories 列表
    pub fn cat_categories(&self) -> Option<&Vec<String>> {
        if let ColumnData::Categorical(c) = &self.data {
            Some(&c.categories)
        } else {
            None
        }
    }

    /// 获取 codes 列表
    pub fn cat_codes(&self) -> Option<&Vec<Option<i32>>> {
        if let ColumnData::Categorical(c) = &self.data {
            Some(&c.codes)
        } else {
            None
        }
    }

    /// 是否有序
    pub fn cat_ordered(&self) -> Option<bool> {
        if let ColumnData::Categorical(c) = &self.data {
            Some(c.ordered)
        } else {
            None
        }
    }

    /// 添加新的 categories
    pub fn cat_add_categories(&self, new_cats: &[String]) -> Option<Series> {
        if let ColumnData::Categorical(c) = &self.data {
            let mut categories = c.categories.clone();
            for cat in new_cats {
                if !categories.contains(cat) {
                    categories.push(cat.clone());
                }
            }
            Some(Series {
                name: self.name.clone(),
                data: ColumnData::Categorical(CategoricalData {
                    categories,
                    codes: c.codes.clone(),
                    ordered: c.ordered,
                }),
            })
        } else {
            None
        }
    }

    /// 移除未使用的 categories
    pub fn cat_remove_unused_categories(&self) -> Option<Series> {
        if let ColumnData::Categorical(c) = &self.data {
            let used_codes: std::collections::HashSet<i32> =
                c.codes.iter().filter_map(|x| *x).collect();
            let mut new_categories: Vec<String> = Vec::new();
            let mut code_map: std::collections::HashMap<i32, i32> =
                std::collections::HashMap::new();
            for (i, cat) in c.categories.iter().enumerate() {
                let old_code = i as i32;
                if used_codes.contains(&old_code) {
                    let new_code = new_categories.len() as i32;
                    new_categories.push(cat.clone());
                    code_map.insert(old_code, new_code);
                }
            }
            let new_codes: Vec<Option<i32>> = c
                .codes
                .iter()
                .map(|code| code.and_then(|old| code_map.get(&old).copied()))
                .collect();
            Some(Series {
                name: self.name.clone(),
                data: ColumnData::Categorical(CategoricalData {
                    categories: new_categories,
                    codes: new_codes,
                    ordered: c.ordered,
                }),
            })
        } else {
            None
        }
    }

    /// 重命名 categories
    pub fn cat_rename_categories(&self, new_names: &[String]) -> Option<Series> {
        if let ColumnData::Categorical(c) = &self.data {
            if new_names.len() != c.categories.len() {
                return None;
            }
            Some(Series {
                name: self.name.clone(),
                data: ColumnData::Categorical(CategoricalData {
                    categories: new_names.to_vec(),
                    codes: c.codes.clone(),
                    ordered: c.ordered,
                }),
            })
        } else {
            None
        }
    }

    /// 设置 ordered 标志
    pub fn cat_as_ordered(&self) -> Option<Series> {
        if let ColumnData::Categorical(c) = &self.data {
            Some(Series {
                name: self.name.clone(),
                data: ColumnData::Categorical(CategoricalData {
                    categories: c.categories.clone(),
                    codes: c.codes.clone(),
                    ordered: true,
                }),
            })
        } else {
            None
        }
    }

    pub fn cat_as_unordered(&self) -> Option<Series> {
        if let ColumnData::Categorical(c) = &self.data {
            Some(Series {
                name: self.name.clone(),
                data: ColumnData::Categorical(CategoricalData {
                    categories: c.categories.clone(),
                    codes: c.codes.clone(),
                    ordered: false,
                }),
            })
        } else {
            None
        }
    }

    // ---------- 日期时间方法（对 Float 类型有效，值为 Unix 纪元秒） ----------

    /// 提取年份
    pub fn dt_year(&self) -> Series {
        let data = match &self.data {
            ColumnData::Float(v) => {
                let years: Vec<Option<i64>> = v
                    .par_iter()
                    .map(|x| {
                        x.map(|ts| {
                            // 简单日期计算：从 Unix 纪元秒提取年份
                            let days = (ts / 86400.0) as i64;
                            let mut year = 1970i64;
                            let mut remaining = days;
                            loop {
                                let is_leap =
                                    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
                                let days_in_year = if is_leap { 366 } else { 365 };
                                if remaining < days_in_year {
                                    break;
                                }
                                remaining -= days_in_year;
                                year += 1;
                            }
                            year
                        })
                    })
                    .collect();
                ColumnData::Int(years)
            }
            _ => ColumnData::Int(vec![None; self.len()]),
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 提取月份 (1-12)
    pub fn dt_month(&self) -> Series {
        let data = match &self.data {
            ColumnData::Float(v) => {
                let months: Vec<Option<i64>> = v
                    .par_iter()
                    .map(|x| {
                        x.map(|ts| {
                            let days = (ts / 86400.0) as i64;
                            let mut year = 1970i64;
                            let mut remaining = days;
                            loop {
                                let is_leap =
                                    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
                                let days_in_year = if is_leap { 366 } else { 365 };
                                if remaining < days_in_year {
                                    break;
                                }
                                remaining -= days_in_year;
                                year += 1;
                            }
                            // remaining 是当年第几天（0-based）
                            let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
                            let month_days = if is_leap {
                                [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                            } else {
                                [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                            };
                            let mut month = 1;
                            let mut day = remaining;
                            for &md in &month_days {
                                if day < md {
                                    break;
                                }
                                day -= md;
                                month += 1;
                            }
                            month as i64
                        })
                    })
                    .collect();
                ColumnData::Int(months)
            }
            _ => ColumnData::Int(vec![None; self.len()]),
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 提取日 (1-31)
    pub fn dt_day(&self) -> Series {
        let data = match &self.data {
            ColumnData::Float(v) => {
                let days_vec: Vec<Option<i64>> = v
                    .par_iter()
                    .map(|x| {
                        x.map(|ts| {
                            let days = (ts / 86400.0) as i64;
                            let mut year = 1970i64;
                            let mut remaining = days;
                            loop {
                                let is_leap =
                                    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
                                let days_in_year = if is_leap { 366 } else { 365 };
                                if remaining < days_in_year {
                                    break;
                                }
                                remaining -= days_in_year;
                                year += 1;
                            }
                            let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
                            let month_days = if is_leap {
                                [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                            } else {
                                [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                            };
                            let mut day = remaining;
                            for &md in &month_days {
                                if day < md {
                                    break;
                                }
                                day -= md;
                            }
                            day + 1 // 1-based
                        })
                    })
                    .collect();
                ColumnData::Int(days_vec)
            }
            _ => ColumnData::Int(vec![None; self.len()]),
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 提取小时 (0-23)
    pub fn dt_hour(&self) -> Series {
        let data = match &self.data {
            ColumnData::Float(v) => {
                let hours: Vec<Option<i64>> = v
                    .par_iter()
                    .map(|x| {
                        x.map(|ts| {
                            let day_remainder = ts.rem_euclid(86400.0);
                            (day_remainder / 3600.0) as i64
                        })
                    })
                    .collect();
                ColumnData::Int(hours)
            }
            _ => ColumnData::Int(vec![None; self.len()]),
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 提取分钟 (0-59)
    pub fn dt_minute(&self) -> Series {
        let data = match &self.data {
            ColumnData::Float(v) => {
                let minutes: Vec<Option<i64>> = v
                    .par_iter()
                    .map(|x| {
                        x.map(|ts| {
                            let hour_remainder = ts.rem_euclid(3600.0);
                            (hour_remainder / 60.0) as i64
                        })
                    })
                    .collect();
                ColumnData::Int(minutes)
            }
            _ => ColumnData::Int(vec![None; self.len()]),
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 提取秒 (0-59)
    pub fn dt_second(&self) -> Series {
        let data = match &self.data {
            ColumnData::Float(v) => {
                let seconds: Vec<Option<i64>> = v
                    .par_iter()
                    .map(|x| x.map(|ts| ts.rem_euclid(60.0) as i64))
                    .collect();
                ColumnData::Int(seconds)
            }
            _ => ColumnData::Int(vec![None; self.len()]),
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }

    /// 提取星期几 (0=周一, 6=周日)
    pub fn dt_dayofweek(&self) -> Series {
        let data = match &self.data {
            ColumnData::Float(v) => {
                let dows: Vec<Option<i64>> = v
                    .par_iter()
                    .map(|x| {
                        x.map(|ts| {
                            let days = (ts / 86400.0) as i64;
                            // 1970-01-01 是周四 (3)
                            (days + 3) % 7
                        })
                    })
                    .collect();
                ColumnData::Int(dows)
            }
            _ => ColumnData::Int(vec![None; self.len()]),
        };
        Series {
            name: self.name.clone(),
            data,
        }
    }
}
