//! Series: 单列数据结构 + PyO3 绑定。
//!
//! 该模块按功能拆分为多个子模块，每个子模块为 :struct:`Series` 提供一组方法：
//! - :mod:`.basic`：构造器 / 属性 / 切片 / 唯一值 / 通用辅助方法
//! - :mod:`.compare`：比较运算（返回布尔掩码）
//! - :mod:`.stats`：聚合统计（count/sum/mean/min/max/std/var/median/any/all）
//! - :mod:`.missing`：缺失值、ffill/bfill、插值/采样/重采样
//! - :mod:`.ops`：Series vs Series 逐元素算术/比较运算（Rust 加速路径）
//! - :mod:`.accessors`：Categorical 操作与日期时间访问器
//! - :mod:`.window`：分位数 / 排名 / 值计数 / 滚动 / 扩展 / 指数加权窗口
//! - :mod:`.sort`：排序、searchsorted、arg_top_n
//! - :mod:`.string_ops`：字符串方法
//! - :mod:`.groupby`：分组聚合、批量聚合、表达式过滤
//! - :mod:`.pymethods`：PyO3 绑定（`#[pymethods] impl PySeries`）
//!
//! PyO3 0.29 API: PyAnyMethods trait 提供 downcast/is_instance_of 等。

pub mod accessors;
pub mod basic;
pub mod compare;
pub mod groupby;
pub mod missing;
pub mod ops;
pub mod pymethods;
pub mod sort;
pub mod stats;
pub mod string_ops;
pub mod type_counts; // 单次遍历 apply(type)+计数
pub mod window;

use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pyo3::types::{PyBool, PyBoolMethods, PyFloat, PyInt, PyList, PyString};
use std::collections::HashMap;

use crate::core::dtype::{CategoricalData, ColumnData, DType};

/// Series: 带名字的单列
#[derive(Debug, Clone)]
pub struct Series {
    pub name: Option<String>,
    pub data: ColumnData,
}

/// 聚合结果统一类型
/// 用于在 `py.detach` 闭包中跨 GIL 边界返回不同类型的结果。
/// 仅在本模块及其子模块（:mod:`.pymethods`）内使用，故设为 `pub(crate)`。
pub(crate) enum AggResult {
    Int(i64),
    Float(f64),
    Usize(usize),
    Bool(bool),
    Str(String),
    None,
}

// =====================================================================
// PyO3 绑定
// =====================================================================

/// Python 端 _Series，包装 Rust Series
#[pyclass(name = "_Series", module = "rspandas", from_py_object)]
#[derive(Debug, Clone)]
pub struct PySeries {
    pub inner: Series,
}

impl PySeries {
    fn new_with_dtype(
        pylist: &Bound<'_, PyList>,
        name: Option<String>,
        dtype: DType,
    ) -> PyResult<Self> {
        let inner = match dtype {
            DType::Bool => {
                let mut v: Vec<Option<bool>> = Vec::with_capacity(pylist.len());
                for item in pylist.iter() {
                    if item.is_none() {
                        v.push(None);
                    } else if let Ok(b) = item.cast::<PyBool>() {
                        v.push(Some(b.is_true()));
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "type mismatch (bool)",
                        ));
                    }
                }
                Series::new_bool(name, v)
            }
            DType::Int64 => {
                let mut v: Vec<Option<i64>> = Vec::with_capacity(pylist.len());
                for item in pylist.iter() {
                    if item.is_none() {
                        v.push(None);
                    } else if let Ok(i) = item.cast::<PyInt>() {
                        v.push(Some(i.extract::<i64>()?));
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "type mismatch (int)",
                        ));
                    }
                }
                Series::new_int(name, v)
            }
            DType::Float64 => {
                let mut v: Vec<Option<f64>> = Vec::with_capacity(pylist.len());
                for item in pylist.iter() {
                    if item.is_none() {
                        v.push(None);
                    } else if let Ok(f) = item.cast::<PyFloat>() {
                        v.push(Some(f.extract::<f64>()?));
                    } else if let Ok(i) = item.cast::<PyInt>() {
                        v.push(Some(i.extract::<i64>()? as f64));
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "type mismatch (float)",
                        ));
                    }
                }
                Series::new_float(name, v)
            }
            DType::Object => {
                let mut v: Vec<Option<String>> = Vec::with_capacity(pylist.len());
                for item in pylist.iter() {
                    if item.is_none() {
                        v.push(None);
                    } else if let Ok(s) = item.cast::<PyString>() {
                        v.push(Some(s.extract::<String>()?));
                    } else if let Ok(b) = item.cast::<PyBool>() {
                        v.push(Some(b.extract::<bool>()?.to_string()));
                    } else if let Ok(i) = item.cast::<PyInt>() {
                        v.push(Some(i.extract::<i64>()?.to_string()));
                    } else if let Ok(f) = item.cast::<PyFloat>() {
                        let fv = f.extract::<f64>()?;
                        if fv.is_nan() {
                            // NaN 在 object dtype 中存储为 None, 便于缺失值检测
                            v.push(None);
                        } else {
                            v.push(Some(fv.to_string()));
                        }
                    } else {
                        // 其他类型 (如 list/dict) 使用 str() 转为字符串
                        let s = item.str()?;
                        v.push(Some(s.extract::<String>()?));
                    }
                }
                Series::new_string(name, v)
            }
            DType::Categorical => {
                let mut raw: Vec<Option<String>> = Vec::with_capacity(pylist.len());
                for item in pylist.iter() {
                    if item.is_none() {
                        raw.push(None);
                    } else if let Ok(s) = item.cast::<PyString>() {
                        raw.push(Some(s.extract::<String>()?));
                    } else {
                        raw.push(Some(item.str()?.extract::<String>()?));
                    }
                }
                let mut cat_map: std::collections::HashMap<String, i32> =
                    std::collections::HashMap::new();
                let mut categories: Vec<String> = Vec::new();
                let mut codes: Vec<Option<i32>> = Vec::with_capacity(raw.len());
                for val in &raw {
                    match val {
                        Some(s) => {
                            let next_idx = categories.len() as i32;
                            let code = *cat_map.entry(s.clone()).or_insert_with(|| {
                                categories.push(s.clone());
                                next_idx
                            });
                            codes.push(Some(code));
                        }
                        None => codes.push(None),
                    }
                }
                Series {
                    name,
                    data: ColumnData::Categorical(CategoricalData {
                        categories,
                        codes,
                        ordered: false,
                    }),
                }
            }
        };
        Ok(PySeries { inner })
    }
}

// =====================================================================
// factorize 函数
// =====================================================================

/// 对输入值进行 factorize 编码 (类似 pandas.factorize)
/// 返回 (codes, categories)
#[pyfunction]
pub fn factorize<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyList>,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    let mut cat_map: HashMap<String, i32> = HashMap::new();
    let mut categories: Vec<String> = Vec::new();
    let mut codes: Vec<i32> = Vec::with_capacity(values.len());

    for item in values.iter() {
        if item.is_none() {
            codes.push(-1);
        } else {
            let s: String = if let Ok(s) = item.cast::<PyString>() {
                s.extract::<String>()?
            } else {
                item.str()?.extract::<String>()?
            };
            let next_idx = categories.len() as i32;
            let code = *cat_map.entry(s.clone()).or_insert_with(|| {
                categories.push(s);
                next_idx
            });
            codes.push(code);
        }
    }

    let codes_list = PyList::new(py, codes.iter().copied())?;
    let cats_list = PyList::new(py, categories.iter().map(|s| s.as_str()))?;
    Ok((codes_list, cats_list))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_series_basic() {
        let s = Series::new_int(Some("a".to_string()), vec![Some(1), Some(2), Some(3)]);
        assert_eq!(s.len(), 3);
        assert_eq!(s.dtype(), DType::Int64);
        assert_eq!(s.dtype_name(), "int64");
        assert_eq!(s.name(), Some("a"));
    }

    #[test]
    fn test_series_sum_mean() {
        let s = Series::new_float(None, vec![Some(1.0), Some(2.0), Some(3.0)]);
        assert_eq!(s.sum_f64(), Some(6.0));
        assert_eq!(s.mean(), Some(2.0));
        assert_eq!(s.min_f64(), Some(1.0));
        assert_eq!(s.max_f64(), Some(3.0));
    }

    #[test]
    fn test_series_with_null() {
        let s = Series::new_int(None, vec![Some(1), None, Some(3)]);
        assert_eq!(s.count(), 2);
        assert_eq!(s.sum_i64(), Some(4));
        assert_eq!(s.mean(), Some(2.0));
    }

    #[test]
    fn test_series_filter() {
        let s = Series::new_int(None, vec![Some(1), Some(2), Some(3)]);
        let filtered = s.filter(&[true, false, true]);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_series_sort_values() {
        // 整型升序：None 放最后
        let s = Series::new_int(None, vec![Some(3), None, Some(1), Some(2)]);
        let sorted = s.sort_values(true);
        if let ColumnData::Int(v) = &sorted.data {
            assert_eq!(v[0], Some(1));
            assert_eq!(v[1], Some(2));
            assert_eq!(v[2], Some(3));
            assert_eq!(v[3], None);
        } else {
            panic!("dtype 错误");
        }

        // 整型降序：None 放最前
        let sorted_desc = s.sort_values(false);
        if let ColumnData::Int(v) = &sorted_desc.data {
            assert_eq!(v[0], None);
            assert_eq!(v[1], Some(3));
            assert_eq!(v[2], Some(2));
            assert_eq!(v[3], Some(1));
        } else {
            panic!("dtype 错误");
        }

        // 浮点型排序
        let sf = Series::new_float(None, vec![Some(3.0), Some(1.0), Some(2.0)]);
        let sorted_f = sf.sort_values(true);
        if let ColumnData::Float(v) = &sorted_f.data {
            assert_eq!(v[0], Some(1.0));
            assert_eq!(v[1], Some(2.0));
            assert_eq!(v[2], Some(3.0));
        } else {
            panic!("dtype 错误");
        }

        // 字符串排序
        let ss = Series::new_string(
            None,
            vec![Some("banana".to_string()), Some("apple".to_string())],
        );
        let sorted_s = ss.sort_values(true);
        if let ColumnData::String(v) = &sorted_s.data {
            assert_eq!(v[0], Some("apple".to_string()));
            assert_eq!(v[1], Some("banana".to_string()));
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_series_sort_index() {
        let s = Series::new_int(None, vec![Some(1), Some(2), Some(3)]);
        // 升序：保持原顺序
        let sorted_asc = s.sort_index(true);
        if let ColumnData::Int(v) = &sorted_asc.data {
            assert_eq!(v[0], Some(1));
            assert_eq!(v[1], Some(2));
            assert_eq!(v[2], Some(3));
        } else {
            panic!("dtype 错误");
        }

        // 降序：反转原顺序
        let sorted_desc = s.sort_index(false);
        if let ColumnData::Int(v) = &sorted_desc.data {
            assert_eq!(v[0], Some(3));
            assert_eq!(v[1], Some(2));
            assert_eq!(v[2], Some(1));
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_series_ffill() {
        // 前向填充：None 被前一个非 None 替代
        let s = Series::new_int(None, vec![None, Some(1), None, None, Some(2), None]);
        let filled = s.ffill();
        if let ColumnData::Int(v) = &filled.data {
            assert_eq!(v[0], None); // 开头 None 保持
            assert_eq!(v[1], Some(1));
            assert_eq!(v[2], Some(1));
            assert_eq!(v[3], Some(1));
            assert_eq!(v[4], Some(2));
            assert_eq!(v[5], Some(2));
        } else {
            panic!("dtype 错误");
        }

        // 浮点型 ffill
        let sf = Series::new_float(None, vec![Some(1.5), None, Some(2.5)]);
        let filled_f = sf.ffill();
        if let ColumnData::Float(v) = &filled_f.data {
            assert_eq!(v[0], Some(1.5));
            assert_eq!(v[1], Some(1.5));
            assert_eq!(v[2], Some(2.5));
        } else {
            panic!("dtype 错误");
        }

        // 字符串 ffill
        let ss = Series::new_string(
            None,
            vec![Some("a".to_string()), None, Some("b".to_string())],
        );
        let filled_s = ss.ffill();
        if let ColumnData::String(v) = &filled_s.data {
            assert_eq!(v[0], Some("a".to_string()));
            assert_eq!(v[1], Some("a".to_string()));
            assert_eq!(v[2], Some("b".to_string()));
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_series_bfill() {
        // 后向填充：None 被后一个非 None 替代
        let s = Series::new_int(None, vec![None, Some(1), None, None, Some(2), None]);
        let filled = s.bfill();
        if let ColumnData::Int(v) = &filled.data {
            assert_eq!(v[0], Some(1));
            assert_eq!(v[1], Some(1));
            assert_eq!(v[2], Some(2));
            assert_eq!(v[3], Some(2));
            assert_eq!(v[4], Some(2));
            assert_eq!(v[5], None); // 末尾 None 保持
        } else {
            panic!("dtype 错误");
        }

        // 浮点型 bfill
        let sf = Series::new_float(None, vec![Some(1.5), None, Some(2.5)]);
        let filled_f = sf.bfill();
        if let ColumnData::Float(v) = &filled_f.data {
            assert_eq!(v[0], Some(1.5));
            assert_eq!(v[1], Some(2.5));
            assert_eq!(v[2], Some(2.5));
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_series_str_upper() {
        let s = Series::new_string(
            None,
            vec![Some("abc".to_string()), Some("Hello".to_string()), None],
        );
        let upper = s.str_upper();
        if let ColumnData::String(v) = &upper.data {
            assert_eq!(v[0], Some("ABC".to_string()));
            assert_eq!(v[1], Some("HELLO".to_string()));
            assert_eq!(v[2], None);
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_series_str_lower() {
        let s = Series::new_string(
            None,
            vec![Some("ABC".to_string()), Some("Hello".to_string()), None],
        );
        let lower = s.str_lower();
        if let ColumnData::String(v) = &lower.data {
            assert_eq!(v[0], Some("abc".to_string()));
            assert_eq!(v[1], Some("hello".to_string()));
            assert_eq!(v[2], None);
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_series_str_len() {
        let s = Series::new_string(
            None,
            vec![
                Some("abc".to_string()),
                Some("hello".to_string()),
                Some("中".to_string()), // 单字符中文
                None,
            ],
        );
        let len_s = s.str_len();
        if let ColumnData::Int(v) = &len_s.data {
            assert_eq!(v[0], Some(3));
            assert_eq!(v[1], Some(5));
            assert_eq!(v[2], Some(1)); // 字符数而非字节数
            assert_eq!(v[3], None);
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_series_str_contains() {
        let s = Series::new_string(
            None,
            vec![
                Some("hello world".to_string()),
                Some("say hello to rust".to_string()),
                Some("foo".to_string()),
                None,
            ],
        );
        // 包含子串 "hello"
        let mask = s.str_contains("hello");
        assert_eq!(mask, vec![true, true, false, false]);

        // 大小写敏感：Hello 与 hello 不同
        let mask2 = s.str_contains("Hello");
        assert_eq!(mask2, vec![false, false, false, false]);

        // 子串 "ru"
        let mask3 = s.str_contains("ru");
        assert_eq!(mask3, vec![false, true, false, false]);
    }

    #[test]
    fn test_series_str_replace() {
        let s = Series::new_string(
            None,
            vec![
                Some("hello world".to_string()),
                Some("hello rust".to_string()),
                None,
            ],
        );
        let replaced = s.str_replace("hello", "hi");
        if let ColumnData::String(v) = &replaced.data {
            assert_eq!(v[0], Some("hi world".to_string()));
            assert_eq!(v[1], Some("hi rust".to_string()));
            assert_eq!(v[2], None);
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_series_bool_aggregation() {
        // any: 至少一个为 true
        let s1 = Series::new_bool(None, vec![Some(false), Some(true), Some(false)]);
        assert_eq!(s1.any(), Some(true));
        assert_eq!(s1.all(), Some(false));

        // all: 全为 true
        let s2 = Series::new_bool(None, vec![Some(true), Some(true), Some(true)]);
        assert_eq!(s2.any(), Some(true));
        assert_eq!(s2.all(), Some(true));

        // 全 false
        let s3 = Series::new_bool(None, vec![Some(false), Some(false)]);
        assert_eq!(s3.any(), Some(false));
        assert_eq!(s3.all(), Some(false));

        // 空 series
        let s4 = Series::new_bool(None, vec![]);
        assert_eq!(s4.any(), Some(false));
        assert_eq!(s4.all(), Some(true));

        // 非 bool 类型应返回 None
        let s5 = Series::new_int(None, vec![Some(1), Some(2)]);
        assert_eq!(s5.any(), None);
        assert_eq!(s5.all(), None);
    }

    #[test]
    fn test_series_categorical_basic() {
        // 构造一个 categorical Series
        let s = Series::new_categorical(
            Some("cat".to_string()),
            vec!["low".to_string(), "mid".to_string(), "high".to_string()],
            vec![Some(0), Some(2), Some(1), Some(0), None],
            false,
        );
        assert_eq!(s.dtype(), DType::Categorical);
        assert_eq!(s.dtype_name(), "category");
        assert_eq!(s.len(), 5);
        assert_eq!(s.count(), 4); // 5 个中 1 个 None

        // 验证 categories
        let cats = s.cat_categories().expect("应有 categories");
        assert_eq!(cats.len(), 3);
        assert_eq!(cats[0], "low");
        assert_eq!(cats[1], "mid");
        assert_eq!(cats[2], "high");

        // 验证 codes
        let codes = s.cat_codes().expect("应有 codes");
        assert_eq!(codes.len(), 5);
        assert_eq!(codes[0], Some(0));
        assert_eq!(codes[1], Some(2));
        assert_eq!(codes[2], Some(1));
        assert_eq!(codes[3], Some(0));
        assert_eq!(codes[4], None);

        // 验证 ordered 标志
        assert!(!s.cat_ordered().unwrap());

        // 添加新 categories
        let s_add = s
            .cat_add_categories(&["extra".to_string()])
            .expect("应能添加 categories");
        let cats2 = s_add.cat_categories().expect("应有 categories");
        assert_eq!(cats2.len(), 4);
        assert_eq!(cats2[3], "extra");

        // 重命名 categories
        let s_rename = s
            .cat_rename_categories(&["L".to_string(), "M".to_string(), "H".to_string()])
            .expect("应能重命名 categories");
        let cats3 = s_rename.cat_categories().expect("应有 categories");
        assert_eq!(cats3[0], "L");
        assert_eq!(cats3[1], "M");
        assert_eq!(cats3[2], "H");

        // 移除未使用 categories：原 categories 都被使用，所以保持不变
        let s_unused = s.cat_remove_unused_categories().expect("应能移除未使用");
        let cats4 = s_unused.cat_categories().expect("应有 categories");
        assert_eq!(cats4.len(), 3);
    }

    #[test]
    fn test_series_quantile() {
        let s = Series::new_float(
            None,
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)],
        );
        assert!((s.quantile(0.5).unwrap() - 3.0).abs() < 1e-10);
        assert!((s.quantile(0.0).unwrap() - 1.0).abs() < 1e-10);
        assert!((s.quantile(1.0).unwrap() - 5.0).abs() < 1e-10);
        assert!((s.quantile(0.25).unwrap() - 2.0).abs() < 1e-10);
        assert!((s.quantile(0.75).unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_series_rank() {
        let s = Series::new_float(None, vec![Some(3.0), Some(1.0), Some(2.0), Some(1.0)]);
        let ranks = s.rank("average", true, None);
        // 3.0 -> rank 4.0, 1.0 -> rank 1.5, 2.0 -> rank 3.0, 1.0 -> rank 1.5
        assert!((ranks[0].unwrap() - 4.0).abs() < 1e-10);
        assert!((ranks[1].unwrap() - 1.5).abs() < 1e-10);
        assert!((ranks[2].unwrap() - 3.0).abs() < 1e-10);
        assert!((ranks[3].unwrap() - 1.5).abs() < 1e-10);
        // dense rank
        let r_dense = s.rank("dense", true, None);
        assert_eq!(r_dense[0], Some(3.0)); // 3 -> 3rd unique value
        assert_eq!(r_dense[1], Some(1.0)); // 1 -> 1st
        assert_eq!(r_dense[2], Some(2.0)); // 2 -> 2nd
        assert_eq!(r_dense[3], Some(1.0)); // 1 -> 1st
    }

    #[test]
    fn test_series_rolling() {
        let s = Series::new_float(
            None,
            vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)],
        );
        let sums = s.rolling_sum(3, None);
        assert_eq!(sums[0], None); // 窗口不足
        assert_eq!(sums[1], None);
        assert!((sums[2].unwrap() - 6.0).abs() < 1e-10); // 1+2+3
        assert!((sums[3].unwrap() - 9.0).abs() < 1e-10); // 2+3+4
        assert!((sums[4].unwrap() - 12.0).abs() < 1e-10); // 3+4+5

        let means = s.rolling_mean(3, None);
        assert!((means[2].unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_series_expanding() {
        let s = Series::new_float(None, vec![Some(1.0), Some(2.0), Some(3.0)]);
        let sums = s.expanding_sum(Some(1));
        assert!((sums[0].unwrap() - 1.0).abs() < 1e-10);
        assert!((sums[1].unwrap() - 3.0).abs() < 1e-10);
        assert!((sums[2].unwrap() - 6.0).abs() < 1e-10);

        let means = s.expanding_mean(Some(1));
        assert!((means[0].unwrap() - 1.0).abs() < 1e-10);
        assert!((means[1].unwrap() - 1.5).abs() < 1e-10);
        assert!((means[2].unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_series_ewm() {
        let s = Series::new_float(None, vec![Some(1.0), Some(2.0), Some(3.0)]);
        let ema = s.ewm_mean(0.5, Some(0));
        assert!((ema[0].unwrap() - 1.0).abs() < 1e-10);
        assert!((ema[1].unwrap() - 1.5).abs() < 1e-10);
        assert!((ema[2].unwrap() - 2.25).abs() < 1e-10);
    }

    #[test]
    fn test_series_dt_year() {
        // 2020-01-01 00:00:00 UTC = 1577836800
        let s = Series::new_float(None, vec![Some(1577836800.0), Some(1609459200.0)]); // 2020, 2021
        let years = s.dt_year();
        if let ColumnData::Int(v) = &years.data {
            assert_eq!(v[0], Some(2020));
            assert_eq!(v[1], Some(2021));
        } else {
            panic!("应为 Int 类型");
        }
    }

    #[test]
    fn test_series_dt_month_day() {
        // 2020-03-15 12:30:45 UTC
        let ts = 1584275445.0;
        let s = Series::new_float(None, vec![Some(ts)]);
        let month = s.dt_month();
        let day = s.dt_day();
        let hour = s.dt_hour();
        let minute = s.dt_minute();
        let second = s.dt_second();
        if let ColumnData::Int(v) = &month.data {
            assert_eq!(v[0], Some(3));
        } else {
            panic!("月份应为3");
        }
        if let ColumnData::Int(v) = &day.data {
            assert_eq!(v[0], Some(15));
        } else {
            panic!("日应为15");
        }
        if let ColumnData::Int(v) = &hour.data {
            assert_eq!(v[0], Some(12));
        } else {
            panic!("小时应为12");
        }
        if let ColumnData::Int(v) = &minute.data {
            assert_eq!(v[0], Some(30));
        } else {
            panic!("分钟应为30");
        }
        if let ColumnData::Int(v) = &second.data {
            assert_eq!(v[0], Some(45));
        } else {
            panic!("秒应为45");
        }
    }

    #[test]
    fn test_series_dt_dayofweek() {
        // 1970-01-01 = 周四 = 3
        let s = Series::new_float(None, vec![Some(0.0)]);
        let dow = s.dt_dayofweek();
        if let ColumnData::Int(v) = &dow.data {
            assert_eq!(v[0], Some(3));
        } else {
            panic!("dayofweek 应为3（周四）");
        }
    }
}
