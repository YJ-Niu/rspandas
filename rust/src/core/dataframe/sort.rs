//! DataFrame 排序方法：sort_values / sort_index。
//!
//! - sort_values 按指定列的值排序行，支持升/降序，None 的位置随 ascending 切换
//! - sort_index 按索引排序（ascending=true 保持原顺序，false 反转）

use rayon::prelude::*;

use crate::core::dataframe::{DataFrame, cmp_opt_f64, cmp_opt_ord, cmp_opt_str, gather_series};
use crate::core::dtype::ColumnData;
use crate::core::series::Series;

impl DataFrame {
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
}
