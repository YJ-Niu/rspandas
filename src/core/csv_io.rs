//! CSV 读写。
//!
//! - 读取: 自动推断每列类型 (int -> float -> bool -> string)，None 表示空字符串
//! - 写入: 顺序写出列

use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use std::fs::File;
use std::io::{Read, Write};

use crate::core::series::{PySeries, Series};

/// 从 CSV 字符串构造 DataFrame
fn parse_csv_string(content: &str, has_header: bool) -> PyResult<(Vec<String>, Vec<Vec<Option<String>>>)> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(has_header)
        .flexible(true)
        .from_reader(content.as_bytes());

    let mut headers: Vec<String> = Vec::new();
    if has_header {
        for h in rdr.headers().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("csv header: {e}"))
        })?.iter() {
            headers.push(h.to_string());
        }
    }

    let mut cols: Vec<Vec<Option<String>>> = Vec::new();
    let mut ncols_hint: Option<usize> = None;
    let mut row_idx = 0;
    for result in rdr.records() {
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
        for j in record.len()..cols.len() {
            cols[j].push(None);
        }
        row_idx += 1;
    }

    // 只有表头没有数据: 用表头长度初始化空列
    if cols.is_empty() && !headers.is_empty() {
        cols = vec![Vec::new(); headers.len()];
    }

    Ok((headers, cols))
}

/// 推断字符串列的类型
fn infer_column(values: &[Option<String>]) -> (crate::core::dtype::DType, Vec<Option<String>>) {
    use crate::core::dtype::DType;
    // 不含 None 且全部能解析为 i64 -> Int64
    let mut all_int = true;
    let mut all_float = true;
    let mut all_bool = true;
    let mut any_non_null = false;
    for v in values {
        match v {
            Some(s) => {
                any_non_null = true;
                if s.parse::<i64>().is_err() {
                    all_int = false;
                }
                if s.parse::<f64>().is_err() {
                    all_float = false;
                }
                let sl = s.to_lowercase();
                if !(sl == "true" || sl == "false" || sl == "0" || sl == "1") {
                    all_bool = false;
                }
            }
            None => {}
        }
    }
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

/// 将 string 列转换为目标 dtype 的 string 表示
fn cast_strings(values: &[Option<String>], target: crate::core::dtype::DType) -> Vec<Option<String>> {
    use crate::core::dtype::DType;
    values.iter().map(|opt| {
        match opt {
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
            }
        }
    }).collect()
}

#[pyfunction]
pub fn read_csv_string<'py>(content: &str, has_header: bool) -> PyResult<(Vec<String>, Vec<PySeries>)> {
    let (headers, cols) = parse_csv_string(content, has_header)?;
    let mut series_list = Vec::new();
    for (h, col) in headers.iter().zip(cols.iter()) {
        // 空列: 强制为 object dtype (0 长度)
        if col.is_empty() {
            series_list.push(PySeries { inner: Series::from_options_string(h.clone(), &[]) });
            continue;
        }
        let (dtype, _strings) = infer_column(col);
        let casted = cast_strings(col, dtype);
        let series = match dtype {
            crate::core::dtype::DType::Int64 => {
                let ints: Vec<Option<i64>> = casted.iter().map(|v| {
                    v.as_ref().and_then(|s| s.parse::<i64>().ok())
                }).collect();
                Series::from_options_i64(h.clone(), &ints)
            }
            crate::core::dtype::DType::Float64 => {
                let floats: Vec<Option<f64>> = casted.iter().map(|v| {
                    v.as_ref().and_then(|s| s.parse::<f64>().ok())
                }).collect();
                Series::from_options_f64(h.clone(), &floats)
            }
            crate::core::dtype::DType::Bool => {
                let bools: Vec<Option<bool>> = casted.iter().map(|v| {
                    v.as_ref().map(|s| {
                        let sl = s.to_lowercase();
                        sl == "true" || sl == "1"
                    })
                }).collect();
                Series::from_options_bool(h.clone(), &bools)
            }
            crate::core::dtype::DType::Object => {
                Series::from_options_string(h.clone(), col)
            }
        };
        series_list.push(PySeries { inner: series });
    }
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

#[pyfunction]
pub fn write_csv_string<'py>(
    columns: Vec<String>,
    series_list: Vec<PySeries>,
    include_header: bool,
) -> PyResult<String> {
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
    for i in 0..nrows {
        let mut record: Vec<String> = Vec::with_capacity(series_list.len());
        for s in &series_list {
            let v = s.inner.get_str_at(i);
            record.push(csv_escape(&v));
        }
        buf.push_str(&record.join(&sep.to_string()));
        buf.push('\n');
    }
    Ok(buf)
}

#[pyfunction]
pub fn read_csv_path<'py>(path: &str, has_header: bool) -> PyResult<(Vec<String>, Vec<PySeries>)> {
    let mut content = String::new();
    let mut file = File::open(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("open {path}: {e}"))
    })?;
    file.read_to_string(&mut content).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("read {path}: {e}"))
    })?;
    read_csv_string(&content, has_header)
}

#[pyfunction]
pub fn write_csv_path(
    path: &str,
    columns: Vec<String>,
    series_list: Vec<PySeries>,
    include_header: bool,
) -> PyResult<()> {
    let content = write_csv_string(columns, series_list, include_header)?;
    let mut file = File::create(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("create {path}: {e}"))
    })?;
    file.write_all(content.as_bytes()).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("write {path}: {e}"))
    })?;
    Ok(())
}
