//! Series 逐元素运算：Series vs Series 的算术与比较加速路径。
//!
//! 仅加速 Int64/Float64 dtype 且长度相同的场景，其他情况返回 None
//! 由 Python 层回退到通用实现。结果 dtype 选择规则：
//! - 两边都是 Int，运算结果无 None 且都是整数值 → Int64
//! - 只要有 Float 或运算产生 None → Float64（因为 Int 列无法装 None）
//! - 比较运算统一返回 Bool 类型（None 保留为 None）

use rayon::prelude::*;

use crate::core::dtype::ColumnData;
use crate::core::series::Series;

impl Series {
    // ---------- 逐元素算术运算 (Series vs Series) ----------

    /// 逐元素算术运算：self op other
    ///
    /// - `op`: "add", "sub", "mul", "truediv", "floordiv", "mod", "pow"
    /// - `reverse`: 为 true 时交换操作数（用于反向运算符 rsub 等）
    /// - `fill_value`: 至多一侧缺失时用该值替换缺失方；两侧都缺失仍为 None
    ///
    /// 仅支持 Int64/Float64 且长度相同，其他情况返回 None（Python 层回退）。
    pub fn elementwise_arith_series(
        &self,
        other: &Series,
        op: &str,
        reverse: bool,
        fill_value: Option<f64>,
    ) -> Option<Series> {
        // 提前校验 op 是否支持
        if !matches!(
            op,
            "add" | "sub" | "mul" | "truediv" | "floordiv" | "mod" | "pow"
        ) {
            return None;
        }
        // 长度必须相同
        if self.len() != other.len() {
            return None;
        }
        let n = self.len();

        // 提取两边数据，仅接受 Int/Float
        let self_f64: Option<Vec<Option<f64>>> = match &self.data {
            ColumnData::Int(v) => Some(v.par_iter().map(|x| x.map(|i| i as f64)).collect()),
            ColumnData::Float(v) => Some(v.clone()),
            _ => None,
        };
        let other_f64: Option<Vec<Option<f64>>> = match &other.data {
            ColumnData::Int(v) => Some(v.par_iter().map(|x| x.map(|i| i as f64)).collect()),
            ColumnData::Float(v) => Some(v.clone()),
            _ => None,
        };
        let (sf, of) = match (self_f64, other_f64) {
            (Some(a), Some(b)) => (a, b),
            _ => return None,
        };

        // 判断是否本来两边都是 Int（用于结果是否可能保留 Int）
        let both_int =
            matches!(&self.data, ColumnData::Int(_)) && matches!(&other.data, ColumnData::Int(_));

        // 统一使用 f64 计算
        let float_result: Vec<Option<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let a = sf[i];
                let b = of[i];
                let a_missing = a.is_none();
                let b_missing = b.is_none();

                // 两边都缺失 → 结果 None
                if a_missing && b_missing {
                    return None;
                }
                // 至多一边缺失：用 fill_value 替换
                let a_val = match a {
                    Some(v) => v,
                    None => match fill_value {
                        Some(fv) => fv,
                        None => return None,
                    },
                };
                let b_val = match b {
                    Some(v) => v,
                    None => match fill_value {
                        Some(fv) => fv,
                        None => return None,
                    },
                };
                // reverse: 交换 x, y
                let (x, y) = if reverse {
                    (b_val, a_val)
                } else {
                    (a_val, b_val)
                };
                match op {
                    "add" => Some(x + y),
                    "sub" => Some(x - y),
                    "mul" => Some(x * y),
                    "truediv" => {
                        if y == 0.0 {
                            None
                        } else {
                            Some(x / y)
                        }
                    }
                    "floordiv" => {
                        if y == 0.0 {
                            None
                        } else {
                            Some((x / y).floor())
                        }
                    }
                    "mod" => {
                        if y == 0.0 {
                            None
                        } else {
                            Some(x % y)
                        }
                    }
                    "pow" => Some(x.powf(y)),
                    _ => None, // 未知 op（前面已校验，理论不会到这里）
                }
            })
            .collect();

        // 判断结果能否保留为 Int64
        let can_be_int = both_int && float_result.par_iter().all(|opt| {
            matches!(opt, Some(v) if v.fract() == 0.0 && *v != f64::INFINITY && *v != f64::NEG_INFINITY && !v.is_nan())
        });

        let data = if can_be_int {
            let int_vec: Vec<Option<i64>> = float_result
                .par_iter()
                .map(|opt| opt.map(|v| v as i64))
                .collect();
            ColumnData::Int(int_vec)
        } else {
            ColumnData::Float(float_result)
        };

        Some(Series {
            name: self.name.clone(),
            data,
        })
    }

    // ---------- 逐元素比较运算 (Series vs Series) ----------

    /// 逐元素比较运算：self op_name other
    ///
    /// - `op_name`: "eq", "ne", "lt", "gt", "le", "ge"
    /// - `fill_value`: 至多一侧缺失时用该值替换缺失方；两侧都缺失时按
    ///   NaN 规则（ne → True，其他 → False）
    ///
    /// 仅支持 Int64/Float64 且长度相同，其他情况返回 None（Python 层回退）。
    /// 返回 Bool 类型 Series，None 位置保留为 None。
    pub fn elementwise_compare_series(
        &self,
        other: &Series,
        op_name: &str,
        fill_value: Option<f64>,
    ) -> Option<Series> {
        // 提前校验 op_name 是否支持
        if !matches!(op_name, "eq" | "ne" | "lt" | "gt" | "le" | "ge") {
            return None;
        }
        // 长度必须相同
        if self.len() != other.len() {
            return None;
        }
        let n = self.len();

        // 提取两边数据，仅接受 Int/Float
        let self_f64: Option<Vec<Option<f64>>> = match &self.data {
            ColumnData::Int(v) => Some(v.par_iter().map(|x| x.map(|i| i as f64)).collect()),
            ColumnData::Float(v) => Some(v.clone()),
            _ => None,
        };
        let other_f64: Option<Vec<Option<f64>>> = match &other.data {
            ColumnData::Int(v) => Some(v.par_iter().map(|x| x.map(|i| i as f64)).collect()),
            ColumnData::Float(v) => Some(v.clone()),
            _ => None,
        };
        let (sf, of) = match (self_f64, other_f64) {
            (Some(a), Some(b)) => (a, b),
            _ => return None,
        };

        let bool_result: Vec<Option<bool>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let a = sf[i];
                let b = of[i];
                let a_missing = a.is_none();
                let b_missing = b.is_none();

                // 两边都缺失 → 按任务要求 None 保留为 None
                if a_missing && b_missing {
                    return None;
                }
                // 至多一边缺失：用 fill_value 替换
                let a_val = match a {
                    Some(v) => v,
                    None => match fill_value {
                        Some(fv) => fv,
                        None => return None, // fill_value 为 None 时结果仍为 None
                    },
                };
                let b_val = match b {
                    Some(v) => v,
                    None => match fill_value {
                        Some(fv) => fv,
                        None => return None,
                    },
                };
                Some(match op_name {
                    "eq" => a_val == b_val,
                    "ne" => a_val != b_val,
                    "lt" => a_val < b_val,
                    "gt" => a_val > b_val,
                    "le" => a_val <= b_val,
                    "ge" => a_val >= b_val,
                    _ => unreachable!(), // 前面已校验
                })
            })
            .collect();

        Some(Series {
            name: self.name.clone(),
            data: ColumnData::Bool(bool_result),
        })
    }
}
