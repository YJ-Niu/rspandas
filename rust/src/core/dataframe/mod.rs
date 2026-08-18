//! DataFrame: 列存储的多列数据结构 + PyO3 绑定。
//!
//! 该模块按功能拆分为多个子模块，每个子模块为 :struct:`DataFrame` 提供一组方法：
//! - :mod:`.sort`：排序（sort_values / sort_index）
//! - :mod:`.missing`：缺失值检测、全列填充、前后向填充
//! - :mod:`.aggregate`：所有列并行的聚合（sum/mean/std/min/max/count/quantile/any/all/nunique/isnull/notnull）
//! - :mod:`.merge`：哈希连接、分组聚合、简单查询过滤
//! - :mod:`.reshape`：透视/逆透视、melt、stack/unstack
//! - :mod:`.pymethods`：PyO3 绑定（`#[pymethods] impl PyDataFrame`）
//!
//! 本模块（:mod:`.dataframe`）保留：结构体定义、构造器、基本属性与切片方法、
//! 辅助比较/收集函数、以及单元测试。

pub mod aggregate;
pub mod merge;
pub mod missing;
pub mod pymethods;
pub mod reshape;
pub mod sort;

use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

use super::dtype::{ColumnData, DType};
use super::series::Series;

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
    pub fn fillna_rows(&self, fill_dict: &HashMap<String, FillValue>) -> PyResult<DataFrame> {
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
}

/// 通用 Option 比较器 (用于 Ord 类型)
/// ascending=true: 升序，None 放最后
/// ascending=false: 降序，None 放最前
pub(crate) fn cmp_opt_ord<T: Ord + ?Sized>(
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
pub(crate) fn cmp_opt_f64(a: Option<f64>, b: Option<f64>, ascending: bool) -> std::cmp::Ordering {
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
pub(crate) fn cmp_opt_str(a: Option<&str>, b: Option<&str>, ascending: bool) -> std::cmp::Ordering {
    cmp_opt_ord(a, b, ascending)
}

/// 按索引列表收集 Series 中的元素，返回新的 Series
/// 使用 Vec::with_capacity 预分配内存
pub(crate) fn gather_series(s: &Series, indices: &[usize]) -> Series {
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
