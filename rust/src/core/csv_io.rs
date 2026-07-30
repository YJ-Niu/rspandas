//! CSV 读写。
//!
//! - 读取: 自动推断每列类型 (int -> float -> bool -> string)，None 表示空字符串
//! - 写入: 顺序写出列

use crate::core::dtype::DType;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Write};

use crate::core::series::{PySeries, Series};

type CsvParseResult = (Vec<String>, Vec<Vec<Option<String>>>);

/// 从 CSV 字符串构造 DataFrame
fn parse_csv_string(content: &str, has_header: bool) -> PyResult<CsvParseResult> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(has_header)
        .flexible(true)
        .from_reader(content.as_bytes());

    let mut headers: Vec<String> = Vec::new();
    if has_header {
        for h in rdr
            .headers()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("csv header: {e}")))?
            .iter()
        {
            headers.push(h.to_string());
        }
    }

    let mut cols: Vec<Vec<Option<String>>> = Vec::new();
    let mut ncols_hint: Option<usize> = None;
    for (row_idx, result) in rdr.records().enumerate() {
        let record = result.map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("csv row {row_idx}: {e}"))
        })?;
        if !has_header && headers.is_empty() {
            for i in 0..record.len() {
                headers.push(format!("col{i}"));
            }
        }
        // 初始化列
        if ncols_hint.is_none() {
            ncols_hint = Some(record.len());
            cols = vec![Vec::new(); ncols_hint.unwrap()];
        }
        for (i, val) in record.iter().enumerate() {
            if i >= cols.len() {
                cols.push(Vec::new());
                if !has_header {
                    headers.push(format!("col{i}"));
                }
            }
            if val.is_empty() {
                cols[i].push(None);
            } else {
                cols[i].push(Some(val.to_string()));
            }
        }
        cols.iter_mut().skip(record.len()).for_each(|col| {
            col.push(None);
        });
    }

    // 只有表头没有数据: 用表头长度初始化空列
    if cols.is_empty() && !headers.is_empty() {
        cols = vec![Vec::new(); headers.len()];
    }

    Ok((headers, cols))
}

/// 推断字符串列的类型 (并行化)
fn infer_column(values: &[Option<String>]) -> (crate::core::dtype::DType, Vec<Option<String>>) {
    let (all_int, all_float, all_bool, any_non_null) = values
        .par_iter()
        .map(|v| match v {
            Some(s) => {
                let int_ok = s.parse::<i64>().is_ok();
                let float_ok = s.parse::<f64>().is_ok();
                let sl = s.to_lowercase();
                let bool_ok = sl == "true" || sl == "false" || sl == "0" || sl == "1";
                (int_ok, float_ok, bool_ok, true)
            }
            None => (true, true, true, false),
        })
        .reduce(
            || (true, true, true, false),
            |(a_int, a_float, a_bool, a_any), (b_int, b_float, b_bool, b_any)| {
                (
                    a_int && b_int,
                    a_float && b_float,
                    a_bool && b_bool,
                    a_any || b_any,
                )
            },
        );
    let dtype = if !any_non_null {
        DType::Object
    } else if all_bool {
        DType::Bool
    } else if all_int {
        DType::Int64
    } else if all_float {
        DType::Float64
    } else {
        DType::Object
    };
    (dtype, values.to_vec())
}

/// 将 string 列转换为目标 dtype 的 string 表示 (并行化)
fn cast_strings(
    values: &[Option<String>],
    target: crate::core::dtype::DType,
) -> Vec<Option<String>> {
    values
        .par_iter()
        .map(|opt| match opt {
            None => None,
            Some(s) => match target {
                DType::Int64 => {
                    if let Ok(i) = s.parse::<i64>() {
                        Some(i.to_string())
                    } else {
                        Some(s.clone())
                    }
                }
                DType::Float64 => {
                    if let Ok(f) = s.parse::<f64>() {
                        Some(f.to_string())
                    } else {
                        Some(s.clone())
                    }
                }
                DType::Bool => {
                    let sl = s.to_lowercase();
                    if sl == "true" || sl == "1" {
                        Some("true".to_string())
                    } else if sl == "false" || sl == "0" {
                        Some("false".to_string())
                    } else {
                        Some(s.clone())
                    }
                }
                DType::Object => Some(s.clone()),
                DType::Categorical => Some(s.clone()),
            },
        })
        .collect()
}

/// 从 CSV 字符串构造 DataFrame（释放 GIL 进行解析）
#[pyfunction]
pub fn read_csv_string<'py>(
    py: Python<'py>,
    content: &str,
    has_header: bool,
) -> PyResult<(Vec<String>, Vec<PySeries>)> {
    // 释放 GIL 进行 CSV 解析和类型推断
    let (headers, cols) = py.detach(|| parse_csv_string(content, has_header))?;
    let series_list: Vec<PySeries> = headers
        .par_iter()
        .zip(cols.par_iter())
        .map(|(h, col)| {
            // 空列: 强制为 object dtype (0 长度)
            if col.is_empty() {
                return PySeries {
                    inner: Series::from_options_string(h.clone(), &[]),
                };
            }
            let (dtype, _strings) = infer_column(col);
            let casted = cast_strings(col, dtype);
            let series = match dtype {
                crate::core::dtype::DType::Int64 => {
                    let ints: Vec<Option<i64>> = casted
                        .par_iter()
                        .map(|v| v.as_ref().and_then(|s| s.parse::<i64>().ok()))
                        .collect();
                    Series::from_options_i64(h.clone(), &ints)
                }
                crate::core::dtype::DType::Float64 => {
                    let floats: Vec<Option<f64>> = casted
                        .par_iter()
                        .map(|v| v.as_ref().and_then(|s| s.parse::<f64>().ok()))
                        .collect();
                    Series::from_options_f64(h.clone(), &floats)
                }
                crate::core::dtype::DType::Bool => {
                    let bools: Vec<Option<bool>> = casted
                        .par_iter()
                        .map(|v| {
                            v.as_ref().map(|s| {
                                let sl = s.to_lowercase();
                                sl == "true" || sl == "1"
                            })
                        })
                        .collect();
                    Series::from_options_bool(h.clone(), &bools)
                }
                crate::core::dtype::DType::Object => Series::from_options_string(h.clone(), col),
                crate::core::dtype::DType::Categorical => {
                    Series::from_options_string(h.clone(), col)
                }
            };
            PySeries { inner: series }
        })
        .collect();
    Ok((headers, series_list))
}

/// CSV 字段转义: 如果包含 , " 或换行则用引号包裹，引号转义为 ""
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// 写入 CSV 为字符串（释放 GIL 进行并行序列化）
#[pyfunction]
pub fn write_csv_string<'py>(
    py: Python<'py>,
    columns: Vec<String>,
    series_list: Vec<PySeries>,
    include_header: bool,
) -> PyResult<String> {
    // 释放 GIL 进行并行 CSV 序列化
    py.detach(|| {
        let mut buf = String::new();
        let sep = ',';

        if include_header {
            let escaped: Vec<String> = columns.iter().map(|c| csv_escape(c)).collect();
            buf.push_str(&escaped.join(&sep.to_string()));
            buf.push('\n');
        }

        if series_list.is_empty() {
            return Ok(buf);
        }

        let nrows = series_list[0].inner.len();
        // 并行生成每行的 CSV 字符串
        let row_strings: Vec<String> = (0..nrows)
            .into_par_iter()
            .map(|i| {
                let mut record: Vec<String> = Vec::with_capacity(series_list.len());
                for s in &series_list {
                    let v = s.inner.get_str_at(i);
                    record.push(csv_escape(&v));
                }
                record.join(&sep.to_string())
            })
            .collect();
        for row_str in &row_strings {
            buf.push_str(row_str);
            buf.push('\n');
        }
        Ok(buf)
    })
}

/// 从文件路径读取 CSV（释放 GIL 进行文件读取和解析）
#[pyfunction]
pub fn read_csv_path<'py>(
    py: Python<'py>,
    path: &str,
    has_header: bool,
) -> PyResult<(Vec<String>, Vec<PySeries>)> {
    let content = py.detach(|| -> PyResult<String> {
        let mut content = String::new();
        let mut file = File::open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("open {path}: {e}")))?;
        file.read_to_string(&mut content)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("read {path}: {e}")))?;
        Ok(content)
    })?;
    read_csv_string(py, &content, has_header)
}

/// 写入 CSV 到文件路径（释放 GIL 进行文件写入）
#[pyfunction]
pub fn write_csv_path<'py>(
    py: Python<'py>,
    path: &str,
    columns: Vec<String>,
    series_list: Vec<PySeries>,
    include_header: bool,
) -> PyResult<()> {
    let content = write_csv_string(py, columns, series_list, include_header)?;
    py.detach(|| -> PyResult<()> {
        let mut file = File::create(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("create {path}: {e}")))?;
        file.write_all(content.as_bytes())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("write {path}: {e}")))?;
        Ok(())
    })
}

/// 分块读取 CSV：返回每个块的 (headers, PySeries 列表) 列表
/// chunk_size: 每块的行数
#[pyfunction]
pub fn read_csv_chunks<'py>(
    py: Python<'py>,
    content: &str,
    has_header: bool,
    chunk_size: usize,
) -> PyResult<Vec<(Vec<String>, Vec<PySeries>)>> {
    if chunk_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "chunk_size must be > 0",
        ));
    }
    // 释放 GIL 解析 CSV
    let (headers, cols) = py.detach(|| parse_csv_string(content, has_header))?;
    if cols.is_empty() {
        return Ok(Vec::new());
    }
    let n_rows = cols[0].len();
    let mut chunks: Vec<(Vec<String>, Vec<PySeries>)> = Vec::new();
    let mut start = 0;
    while start < n_rows {
        let end = (start + chunk_size).min(n_rows);
        // 切片每列
        let chunk_cols: Vec<Vec<Option<String>>> = cols
            .par_iter()
            .map(|col| col[start..end].to_vec())
            .collect();
        // 类型推断并构建 Series
        let series_list: Vec<PySeries> = headers
            .par_iter()
            .zip(chunk_cols.par_iter())
            .map(|(h, col)| {
                if col.is_empty() {
                    return PySeries {
                        inner: Series::from_options_string(h.clone(), &[]),
                    };
                }
                let (dtype, _strings) = infer_column(col);
                let casted = cast_strings(col, dtype);
                let series = match dtype {
                    DType::Int64 => {
                        let ints: Vec<Option<i64>> = casted
                            .par_iter()
                            .map(|v| v.as_ref().and_then(|s| s.parse::<i64>().ok()))
                            .collect();
                        Series::from_options_i64(h.clone(), &ints)
                    }
                    DType::Float64 => {
                        let floats: Vec<Option<f64>> = casted
                            .par_iter()
                            .map(|v| v.as_ref().and_then(|s| s.parse::<f64>().ok()))
                            .collect();
                        Series::from_options_f64(h.clone(), &floats)
                    }
                    DType::Bool => {
                        let bools: Vec<Option<bool>> = casted
                            .par_iter()
                            .map(|v| {
                                v.as_ref().map(|s| {
                                    let sl = s.to_lowercase();
                                    sl == "true" || sl == "1"
                                })
                            })
                            .collect();
                        Series::from_options_bool(h.clone(), &bools)
                    }
                    DType::Object => Series::from_options_string(h.clone(), col),
                    DType::Categorical => Series::from_options_string(h.clone(), col),
                };
                PySeries { inner: series }
            })
            .collect();
        chunks.push((headers.clone(), series_list));
        start = end;
    }
    Ok(chunks)
}
